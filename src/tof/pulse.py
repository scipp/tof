# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc
from scipp.scipy.interpolate import interp1d

from . import facilities
from .utils import Plot, wavelength_to_speed


def _make_pulse(
    neutrons: int = 1_000_000,
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
    p_time:
        Time probability distribution.
    p_wav:
        Wavelength probability distribution.
    neutrons:
        Number of neutrons in the pulse.
    sampling:
        Number of points used to sample the probability distributions. If not set,
        the size of the distributions will be used.
    """
    if p_time is None:
        if (tmin is None) and (tmax is None):
            raise ValueError("Either `p_time` or `tmin` and `tmax` must be specified.")
        # TODO: set p_time to a uniform distribution between tmin and tmax
    if p_wav is None:
        if (wmin is None) and (wmax is None):
            raise ValueError("Either `p_wav` or `wmin` and `wmax` must be specified.")
        # TODO: set p_wav to a uniform distribution between wmin and wmax

    t_dim = 'time'
    w_dim = 'wavelength'
    t_u = 's'
    w_u = 'angstrom'

    p_time = p_time.to(dtype=float)
    p_wav = p_wav.to(dtype=float)
    p_time.coords[t_dim] = p_time.coords[t_dim].to(unit=t_u)
    p_wav.coords[w_dim] = p_wav.coords[w_dim].to(unit=w_u)

    if tmin is None:
        tmin = p_time.coords[t_dim].min()
    if tmax is None:
        tmax = p_time.coords[t_dim].max()
    if wmin is None:
        wmin = p_wav.coords[w_dim].min()
    if wmax is None:
        wmax = p_wav.coords[w_dim].max()
    tmin = tmin.to(unit=t_u)
    tmax = tmax.to(unit=t_u)
    wmin = wmin.to(unit=w_u)
    wmax = wmax.to(unit=w_u)

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

    birth_times = sc.array(
        dims=['event'],
        values=np.random.choice(
            p_time.coords[t_dim].values, size=neutrons, p=p_time.values
        )
        + np.random.uniform(-dt, dt, size=neutrons),
        unit='s',
    )
    wavelengths = sc.array(
        dims=['event'],
        values=np.random.choice(
            p_wav.coords[w_dim].values, size=neutrons, p=p_wav.values
        )
        + np.random.uniform(-dw, dw, size=neutrons),
        unit='angstrom',
    )
    speeds = wavelength_to_speed(wavelengths)
    return {
        'birth_times': birth_times,
        'wavelengths': wavelengths,
        'speeds': speeds,
        'tmin': tmin,
        'tmax': tmax,
        'wmin': wmin,
        'wmax': wmax,
    }


class Pulse:
    """
    A class that represents a pulse of neutrons.
    It is defined by the number of neutrons, a wavelength range, and a time range.
    A probability distribution for the wavelengths and times can be provided (the
    distribution is flat by default).
    In addition, some pre-defined pulses from neutron facilities can be used by setting
    the ``facility`` parameter.
    Finally, the wavelengths and times of the neutrons can be provided directly.

    Parameters
    ----------
    tmin:
        Start time of the pulse.
    tmax:
        End time of the pulse.
    wmin:
        Minimum wavelength of the pulse.
    wmax:
        Maximum wavelength of the pulse.
    neutrons:
        Number of neutrons in the pulse.
    kind:
        Name of a pre-defined pulse from a neutron facility.
    p_wav:
        Probability distribution for the wavelengths.
    p_time:
        Probability distribution for the times.
    sampling_resolution:
        Number of points used to sample the probability distributions.
    birth_times:
        Birth times of neutrons in the pulse.
    wavelengths:
        Wavelengths of neutrons in the pulse.
    """

    def __init__(
        self,
        facility: str,
        tmin: Optional[sc.Variable] = None,
        tmax: Optional[sc.Variable] = None,
        wmin: Optional[sc.Variable] = None,
        wmax: Optional[sc.Variable] = None,
        neutrons: int = 1_000_000,
        sampling: int = 1000,
    ):
        self.facility = facility
        self.neutrons = neutrons

        if facility is not None:
            dists = getattr(facilities, facility)
            params = _make_pulse(
                tmin=tmin,
                tmax=tmax,
                wmin=wmin,
                wmax=wmax,
                neutrons=neutrons,
                p_time=dists.time,
                p_wav=dists.wavelength,
                sampling=sampling,
            )
            self.birth_times = params['birth_times']
            self.wavelengths = params['wavelengths']
            self.speeds = params['speeds']
            self.tmin = params['tmin']
            self.tmax = params['tmax']
            self.wmin = params['wmin']
            self.wmax = params['wmax']

    def __len__(self) -> int:
        """
        Return the number of neutrons in the pulse.
        """
        return self.neutrons

    @classmethod
    def from_neutrons(cls, birth_times: sc.Variable, wavelengths: sc.Variable):
        """
        Create a pulse from a list of neutrons.
        Both ``birth times`` and ``wavelengths`` should be one-dimensional, and have the
        same length and dimension label.

        Parameters
        ----------
        birth_times:
            Birth times of neutrons in the pulse.
        wavelengths:
            Wavelengths of neutrons in the pulse.
        """
        pulse = cls(facility=None, neutrons=len(birth_times))
        pulse.birth_times = birth_times.to(unit='s')
        pulse.wavelengths = wavelengths.to(unit='angstrom')
        pulse.tmin = pulse.birth_times.min()
        pulse.tmax = pulse.birth_times.max()
        pulse.wmin = pulse.wavelengths.min()
        pulse.wmax = pulse.wavelengths.max()
        pulse.speeds = wavelength_to_speed(pulse.wavelengths)
        return pulse

    @classmethod
    def from_distribution(
        cls,
        neutrons: int = 1_000_000,
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
        p_time:
            Time probability distribution.
        p_wav:
            Wavelength probability distribution.
        neutrons:
            Number of neutrons in the pulse.
        sampling:
            Number of points used to sample the probability distributions. If not set,
            the size of the distributions will be used.
        """

        pulse = cls(facility=None, neutrons=neutrons)
        params = _make_pulse(
            tmin=tmin,
            tmax=tmax,
            wmin=wmin,
            wmax=wmax,
            neutrons=neutrons,
            p_time=p_time,
            p_wav=p_wav,
            sampling=sampling,
        )
        pulse.birth_times = params['birth_times']
        pulse.wavelengths = params['wavelengths']
        pulse.speeds = params['speeds']
        pulse.tmin = params['tmin']
        pulse.tmax = params['tmax']
        pulse.wmin = params['wmin']
        pulse.wmax = params['wmax']
        return pulse

    @property
    def duration(self) -> float:
        """Duration of the pulse."""
        return self.tmax - self.tmin

    def plot(self, bins: int = 300) -> tuple:
        """
        Plot the pulse.

        Parameters
        ----------
        bins:
            Number of bins to use for histogramming the neutrons.
        """
        fig, ax = plt.subplots(1, 2)
        self.birth_times.hist(time=bins).plot(ax=ax[0])
        self.wavelengths.hist(wavelength=bins).plot(ax=ax[1])
        size = fig.get_size_inches()
        fig.set_size_inches(size[0] * 2, size[1])
        fig.tight_layout()
        return Plot(fig=fig, ax=ax)

    def as_readonly(self):
        return PulseParameters(
            birth_times=self.birth_times,
            wavelengths=self.wavelengths,
            speeds=self.speeds,
            facility=self.facility,
            neutrons=self.neutrons,
            tmin=self.tmin,
            tmax=self.tmax,
            wmin=self.wmin,
            wmax=self.wmax,
        )

    def __repr__(self) -> str:
        return (
            f"Pulse(tmin={self.tmin:c}, tmax={self.tmax:c}, "
            f"wmin={self.wmin:c}, wmax={self.wmax:c}, "
            f"neutrons={self.neutrons}, facility='{self.facility}')"
        )


@dataclass(frozen=True)
class PulseParameters:
    birth_times: sc.DataArray
    wavelengths: sc.DataArray
    speeds: sc.DataArray
    facility: Optional[str]
    neutrons: int
    tmin: sc.Variable
    tmax: sc.Variable
    wmin: sc.Variable
    wmax: sc.Variable
