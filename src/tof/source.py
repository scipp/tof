# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import plopp as pp
import scipp as sc
from scipp.scipy.interpolate import interp1d

from . import facilities
from .utils import Plot, wavelength_to_speed


# def _convert_if_not_none(
#     x: Union[None, sc.Variable, sc.DataArray], unit: str, coord: Optional[str] = None
# ) -> Union[None, sc.Variable, sc.DataArray]:
#     if x is not None:
#         if coord is None:
#             return x.to(dtype=float, unit=unit)
#         out = x.copy()
#         out.coords[coord] = out.coords[coord].to(dtype=float, unit=unit)
#         return out


def _convert_coord(da: sc.DataArray, unit: str, coord: str) -> sc.DataArray:
    out = da.copy(deep=False)
    out.coords[coord] = out.coords[coord].to(dtype=float, unit=unit)
    return out


def _make_pulses(
    neutrons: int = 1_000_000,
    frequency: Optional[sc.Variable] = None,
    pulses: int = 1,
    p_time: Optional[sc.DataArray] = None,
    p_wav: Optional[sc.DataArray] = None,
    tmin: Optional[sc.Variable] = None,
    tmax: Optional[sc.Variable] = None,
    wmin: Optional[sc.Variable] = None,
    wmax: Optional[sc.Variable] = None,
    sampling: Optional[int] = 1000,
):
    """
    Create a pulse from time a wavelength probability distributions.
    The distributions should be supplied as DataArrays where the coordinates
    are the values of the distribution, and the values are the probability.
    Note that the time and wavelength distributions are independent. A neutron with
    a randomly selected birth time from ``p_time`` can adopt any wavelength in
    ``p_wav`` (in other words, the two distributions are simply broadcast into a
    square 2D parameter space).

    Parameters
    ----------
    neutrons:
        Number of neutrons per pulse.
    frequency:
        Pulse frequency.
    pulses:
        Number of pulses.
    p_time:
        Time probability distribution for a single pulse.
    p_wav:
        Wavelength probability distribution for a single pulse.
    tmin:
        Start time of the first pulse.
    tmax:
        End time of the first pulse.
    wmin:
        Minimum neutron wavelength.
    wmax:
        Maximum neutron wavelength.
    sampling:
        Number of points used to sample the probability distributions.
    """
    t_dim = 'time'
    w_dim = 'wavelength'
    t_u = 's'
    w_u = 'angstrom'

    # tmin = _convert_if_not_none(tmin, unit=t_u)
    # tmax = _convert_if_not_none(tmax, unit=t_u)
    # wmin = _convert_if_not_none(wmin, unit=w_u)
    # wmax = _convert_if_not_none(wmax, unit=w_u)
    # p_time = _convert_if_not_none(p_time, unit=t_u, coord=t_dim)
    # p_wav = _convert_if_not_none(p_wav, unit=w_u, coord=w_dim)

    p_time = _convert_coord(p_time, unit=t_u, coord=t_dim)
    p_wav = _convert_coord(p_wav, unit=w_u, coord=w_dim)
    sampling = int(sampling)

    # if p_time is None:
    #     if (tmin is None) and (tmax is None):
    #         raise ValueError("Either `p_time` or `tmin` and `tmax` must be specified.")
    #     dt = (tmax - tmin).value / (sampling - 1)
    #     p_time = sc.DataArray(
    #         data=sc.array(dims=[t_dim], values=[1.0, 1.0]),
    #         coords={
    #             t_dim: sc.array(
    #                 dims=[t_dim], values=[tmin.value + dt, tmax.value + dt], unit=t_u
    #             )
    #         },
    #     )
    # if p_wav is None:
    #     if (wmin is None) and (wmax is None):
    #         raise ValueError("Either `p_wav` or `wmin` and `wmax` must be specified.")
    #     dw = (wmax - wmin).value / (sampling - 1)
    #     p_wav = sc.DataArray(
    #         data=sc.array(dims=[w_dim], values=[1.0, 1.0]),
    #         coords={
    #             w_dim: sc.array(
    #                 dims=[w_dim], values=[wmin.value + dw, wmax.value + dw], unit=w_u
    #             )
    #         },
    #     )

    # if tmin is None:
    #     tmin = p_time.coords[t_dim].min()
    # if tmax is None:
    #     tmax = p_time.coords[t_dim].max()
    # if wmin is None:
    #     wmin = p_wav.coords[w_dim].min()
    # if wmax is None:
    #     wmax = p_wav.coords[w_dim].max()

    tmin = p_time.coords[t_dim].min()
    tmax = p_time.coords[t_dim].max()
    wmin = p_wav.coords[w_dim].min()
    wmax = p_wav.coords[w_dim].max()

    time_interpolator = interp1d(p_time, dim=t_dim, fill_value='extrapolate')
    wav_interpolator = interp1d(p_wav, dim=w_dim, fill_value='extrapolate')
    x_time = sc.linspace(
        dim=t_dim,
        start=tmin.value,
        stop=tmax.value,
        num=sampling,
        unit=tmin.unit,
    )
    x_wav = sc.linspace(
        dim=w_dim,
        start=wmin.value,
        stop=wmax.value,
        num=sampling,
        unit=wmin.unit,
    )
    p_time = time_interpolator(x_time)
    p_time /= p_time.data.sum()
    p_wav = wav_interpolator(x_wav)
    p_wav /= p_wav.data.sum()

    # In the following, random.choice only allows to select from the values listed
    # in the coordinate of the probability distribution arrays. This leads to data
    # grouped into spikes and empty in between because the sampling resolution used
    # in the linear interpolation above is usually kept low for performance.
    # To make the distribution more uniform, we add some random noise to the chosen
    # values, which effectively fills in the gaps between the spikes.
    # Scipy has some methods to sample from a continuous distribution, but they are
    # prohibitively slow.
    # See https://docs.scipy.org/doc/scipy/tutorial/stats/sampling.html for more
    # information.
    dt = 0.5 * (tmax - tmin).value / (p_time.sizes[t_dim] - 1)
    dw = 0.5 * (wmax - wmin).value / (p_wav.sizes[w_dim] - 1)

    # Because of the added noise, some values end up being outside the specified range
    # for the birth times and wavelengths. Using naive clipping leads to pile-up on the
    # edges of the range. To avoid this, we remove the outliers and resample until we
    # have the desired number of neutrons.
    n = 0
    times = []
    wavs = []
    ntot = pulses * neutrons
    while n < ntot:
        size = ntot - n
        t = np.random.choice(
            p_time.coords[t_dim].values, size=size, p=p_time.values
        ) + np.random.normal(scale=dt, size=size)
        w = np.random.choice(
            p_wav.coords[w_dim].values, size=size, p=p_wav.values
        ) + np.random.normal(scale=dw, size=size)
        mask = (
            (t >= tmin.value)
            & (t <= tmax.value)
            & (w >= wmin.value)
            & (w <= wmax.value)
        )
        times.append(t[mask])
        wavs.append(w[mask])
        n += mask.sum()

    dim = 'event'
    birth_time = sc.array(
        dims=[dim],
        values=np.concatenate(times),
        unit='s',
    ).fold(
        dim=dim, sizes={'pulse': pulses, dim: neutrons}
    ) + (sc.arange('pulse', pulses) / frequency)

    wavelength = sc.array(
        dims=[dim],
        values=np.concatenate(wavs),
        unit='angstrom',
    ).fold(dim=dim, sizes={'pulse': pulses, dim: neutrons})
    speed = wavelength_to_speed(wavelength)
    return {
        'time': birth_time,
        'wavelength': wavelength,
        'speed': speed,
        # 'tmin': tmin,
        # 'tmax': tmax,
        # 'wmin': wmin,
        # 'wmax': wmax,
    }


class Source:
    """
    A class that represents a source of neutrons.
    It is defined by the number of neutrons, a wavelength range, and a time range.
    The default way of creating a pulse is to supply the name of a facility
    (e.g. ``'ess'``) and the number of neutrons. This will create a pulse with the
    default time and wavelength ranges for that facility.
    The time and wavelength ranges can be further constrained by setting the ``tmin``,
    ``tmax``, ``wmin``, and ``wmax`` parameters.

    Parameters
    ----------
    facility:
        Name of a pre-defined pulse shape from a neutron facility.
    neutrons:
        Number of neutrons per pulse.
    pulses:
        Number of pulses.
    tmin:
        Start time of the first pulse.
    tmax:
        End time of the first pulse.
    wmin:
        Minimum neutron wavelength.
    wmax:
        Maximum neutron wavelength.
    sampling:
        Number of points used to interpolate the probability distributions.
    """

    def __init__(
        self,
        facility: str,
        neutrons: int = 1_000_000,
        pulses: int = 1,
        # tmin: Optional[sc.Variable] = None,
        # tmax: Optional[sc.Variable] = None,
        # wmin: Optional[sc.Variable] = None,
        # wmax: Optional[sc.Variable] = None,
        sampling: int = 1000,
    ):
        self.facility = facility
        self.neutrons = int(neutrons)
        self.pulses = int(pulses)
        self.data = None
        self.frequency = None

        if facility is not None:
            facility_params = getattr(facilities, self.facility)
            self.frequency = facility_params.frequency
            pulse_params = _make_pulses(
                # tmin=tmin,
                # tmax=tmax,
                # wmin=wmin,
                # wmax=wmax,
                neutrons=self.neutrons,
                p_time=facility_params.time,
                p_wav=facility_params.wavelength,
                sampling=sampling,
                frequency=self.frequency,
                pulses=pulses,
            )
            self.data = sc.DataArray(
                data=sc.ones(sizes=pulse_params['time'].sizes, unit='counts'),
                coords={
                    'time': pulse_params['time'],
                    'wavelength': pulse_params['wavelength'],
                    'speed': pulse_params['speed'],
                },
            )
            # self.tmin = pulse_params['tmin']
            # self.tmax = pulse_params['tmax']
            # self.wmin = pulse_params['wmin']
            # self.wmax = pulse_params['wmax']
            # self.frequency = facility_params.frequency

    @classmethod
    def from_neutrons(
        cls,
        birth_times: sc.Variable,
        wavelengths: sc.Variable,
        frequency: sc.Variable,
        pulses: int = 1,
    ):
        """
        Create source pulses from a list of neutrons.
        Both ``birth times`` and ``wavelengths`` should be one-dimensional and have the
        same length. They represent the neutrons inside one pulse. If ``pulses`` is
        greater than one, the neutrons will be repeated ``pulses`` times.

        Parameters
        ----------
        birth_times:
            Birth times of neutrons in the pulse.
        wavelengths:
            Wavelengths of neutrons in the pulse.
        frequency:
            Frequency of the pulse.
        pulses:
            Number of pulses.
        """
        source = cls(facility=None, neutrons=len(birth_times))
        # pulse.birth_times = birth_times.to(unit='s')
        # pulse.wavelengths = wavelengths.to(unit='angstrom')

        # # pulse.tmin = pulse.birth_times.min()
        # # pulse.tmax = pulse.birth_times.max()
        # # pulse.wmin = pulse.wavelengths.min()
        # # pulse.wmax = pulse.wavelengths.max()
        # pulse.speeds = wavelength_to_speed(pulse.wavelengths)

        birth_times = birth_times.to(unit='s', copy=False) + (
            sc.arange('pulse', pulses) / frequency
        )
        wavelengths = sc.broadcast(
            wavelengths.to(unit='angstrom', copy=False), sizes=birth_times.sizes
        )

        source.data = sc.DataArray(
            data=sc.ones(sizes=birth_times.sizes, unit='counts'),
            coords={
                'time': birth_times,
                'wavelength': wavelengths,
                'speed': wavelength_to_speed(wavelengths).to(unit='m/s', copy=False),
            },
        )

        return source

    @classmethod
    def from_distribution(
        cls,
        neutrons: int = 1_000_000,
        p_time: Optional[sc.DataArray] = None,
        p_wav: Optional[sc.DataArray] = None,
        # tmin: Optional[sc.Variable] = None,
        # tmax: Optional[sc.Variable] = None,
        # wmin: Optional[sc.Variable] = None,
        # wmax: Optional[sc.Variable] = None,
        sampling: Optional[int] = 1000,
    ):
        """
        Create source pulses from time a wavelength probability distributions.
        The distributions should be supplied as DataArrays where the coordinates
        are the values of the distribution, and the values are the probability.
        Note that the time and wavelength distributions are independent. A neutron with
        a randomly selected birth time from ``p_time`` can adopt any wavelength in
        ``p_wav`` (in other words, the two distributions are simply broadcast into a
        square 2D parameter space).

        Parameters
        ----------
        neutrons:
            Number of neutrons in the pulse.
        p_time:
            Time probability distribution.
        p_wav:
            Wavelength probability distribution.
        tmin:
            Start time of the pulse.
        tmax:
            End time of the pulse.
        wmin:
            Minimum wavelength of the pulse.
        wmax:
            Maximum wavelength of the pulse.
        sampling:
            Number of points used to interpolate the probability distributions.
        """

        source = cls(facility=None, neutrons=neutrons)
        pulse_params = _make_pulses(
            # tmin=tmin,
            # tmax=tmax,
            # wmin=wmin,
            # wmax=wmax,
            neutrons=neutrons,
            p_time=p_time,
            p_wav=p_wav,
            sampling=sampling,
        )
        source.data = sc.DataArray(
            data=sc.ones(sizes=pulse_params['time'].sizes, unit='counts'),
            coords={
                'time': pulse_params['time'],
                'wavelength': pulse_params['wavelength'],
                'speed': pulse_params['speed'],
            },
        )
        # pulse.birth_times = params['birth_times']
        # pulse.wavelengths = params['wavelengths']
        # pulse.speeds = params['speeds']
        # pulse.tmin = params['tmin']
        # pulse.tmax = params['tmax']
        # pulse.wmin = params['wmin']
        # pulse.wmax = params['wmax']
        return source

    # @property
    # def duration(self) -> float:
    #     """Duration of the pulse."""
    #     return self.tmax - self.tmin

    # @property
    # def duration(self) -> float:
    #     """Duration of the pulse."""
    #     return self.tmax - self.tmin

    def plot(self, bins: int = 300) -> tuple:
        """
        Plot the pulses of the source.

        Parameters
        ----------
        bins:
            Number of bins to use for histogramming the neutrons.
        """
        fig, ax = plt.subplots(1, 2)
        dim = (set(self.data.dims) - {'pulse'}).pop()
        collapsed = sc.collapse(self.data, keep=dim)
        pp.plot(
            {k: da.hist(time=bins) for k, da in collapsed.items()},
            ax=ax[0],
        )
        pp.plot(
            {k: da.hist(wavelength=bins) for k, da in collapsed.items()},
            ax=ax[1],
        )
        size = fig.get_size_inches()
        fig.set_size_inches(size[0] * 2, size[1])
        fig.tight_layout()
        return Plot(fig=fig, ax=ax)

    def as_readonly(self):
        return SourceParameters(
            data=self.data,
            facility=self.facility,
            neutrons=self.neutrons,
            frequency=self.frequency,
            pulses=self.pulses,
            # tmin=self.tmin,
            # tmax=self.tmax,
            # wmin=self.wmin,
            # wmax=self.wmax,
        )

    def __repr__(self) -> str:
        return (
            f"Source:\n"  #  tmin={self.tmin:c}, tmax={self.tmax:c}\n"
            f"  pulses={self.pulses}, neutrons per pulse={self.neutrons}\n"
            # f"  wavelength range=[{self.wmin:c}, {self.wmax:c}]\n"
            f"  frequency={self.frequency:c}\n  facility='{self.facility}'"
        )


@dataclass(frozen=True)
class SourceParameters:
    """
    Read-only container for the parameters of a source.
    """

    data: sc.DataArray
    facility: Optional[str]
    neutrons: int
    frequency: sc.Variable
    pulses: int
    # tmin: sc.Variable
    # tmax: sc.Variable
    # wmin: sc.Variable
    # wmax: sc.Variable
