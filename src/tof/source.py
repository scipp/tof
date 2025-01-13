# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import plopp as pp
import scipp as sc
from scipp.scipy.interpolate import interp1d

from .facilities import library as facilities
from .utils import Plot, wavelength_to_speed

TIME_UNIT = "us"
WAV_UNIT = "angstrom"


def _default_frequency(frequency: Union[None, sc.Variable], pulses: int) -> sc.Variable:
    if frequency is None:
        if pulses > 1:
            raise ValueError(
                "If pulses is greater than one, a frequency must be supplied."
            )
        frequency = 1.0 * sc.Unit("Hz")
    return frequency


def _convert_coord(da: sc.DataArray, unit: str, coord: str) -> sc.DataArray:
    out = da.copy(deep=False)
    out.coords[coord] = out.coords[coord].to(dtype=float, unit=unit)
    return out


def _make_pulses(
    neutrons: int,
    frequency: sc.Variable,
    pulses: int,
    p_time: sc.DataArray,
    p_wav: sc.DataArray,
    sampling: int,
    seed: Optional[int],
    wmin: Optional[sc.Variable] = None,
    wmax: Optional[sc.Variable] = None,
):
    """
    Create pulses from time a wavelength probability distributions.
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
    sampling:
        Number of points used to sample the probability distributions.
    seed:
        Seed for the random number generator.
    wmin:
        Minimum neutron wavelength.
    wmax:
        Maximum neutron wavelength.
    """
    t_dim = "time"
    w_dim = "wavelength"

    p_time = _convert_coord(p_time, unit=TIME_UNIT, coord=t_dim)
    p_wav = _convert_coord(p_wav, unit=WAV_UNIT, coord=w_dim)
    sampling = int(sampling)

    tmin = p_time.coords[t_dim].min()
    tmax = p_time.coords[t_dim].max()
    if wmin is None:
        wmin = p_wav.coords[w_dim].min()
    if wmax is None:
        wmax = p_wav.coords[w_dim].max()

    time_interpolator = interp1d(p_time, dim=t_dim, fill_value="extrapolate")
    wav_interpolator = interp1d(p_wav, dim=w_dim, fill_value="extrapolate")
    x_time = sc.linspace(
        dim=t_dim,
        start=tmin.value,
        stop=tmax.value,
        num=sampling,
        unit=TIME_UNIT,
    )
    x_wav = sc.linspace(
        dim=w_dim,
        start=wmin.value,
        stop=wmax.value,
        num=sampling,
        unit=WAV_UNIT,
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
    rng = np.random.default_rng(seed)
    while n < ntot:
        size = ntot - n
        t = rng.choice(
            p_time.coords[t_dim].values, size=size, p=p_time.values
        ) + rng.normal(scale=dt, size=size)
        w = rng.choice(
            p_wav.coords[w_dim].values, size=size, p=p_wav.values
        ) + rng.normal(scale=dw, size=size)
        mask = (
            (t >= tmin.value)
            & (t <= tmax.value)
            & (w >= wmin.value)
            & (w <= wmax.value)
        )
        times.append(t[mask])
        wavs.append(w[mask])
        n += mask.sum()

    dim = "event"
    birth_time = sc.array(
        dims=[dim],
        values=np.concatenate(times),
        unit=TIME_UNIT,
    ).fold(dim=dim, sizes={"pulse": pulses, dim: neutrons}) + (
        sc.arange("pulse", pulses) / frequency
    ).to(
        unit=TIME_UNIT, copy=False
    )

    wavelength = sc.array(
        dims=[dim],
        values=np.concatenate(wavs),
        unit=WAV_UNIT,
    ).fold(dim=dim, sizes={"pulse": pulses, dim: neutrons})
    speed = wavelength_to_speed(wavelength)
    return {
        "time": birth_time,
        "wavelength": wavelength,
        "speed": speed,
    }


class Source:
    """
    A class that represents a source of neutrons.
    It is defined by the number of neutrons, a wavelength range, and a time range.
    The default way of creating a pulse is to supply the name of a facility
    (e.g. ``'ess'``) and the number of neutrons. This will create a pulse with the
    default time and wavelength ranges for that facility.

    Parameters
    ----------
    facility:
        Name of a pre-defined pulse shape from a neutron facility.
    neutrons:
        Number of neutrons per pulse.
    pulses:
        Number of pulses.
    sampling:
        Number of points used to interpolate the probability distributions.
    wmin:
        Minimum neutron wavelength.
    wmax:
        Maximum neutron wavelength.
    seed:
        Seed for the random number generator.
    """

    def __init__(
        self,
        facility: str,
        neutrons: int = 1_000_000,
        pulses: int = 1,
        sampling: int = 1000,
        wmin: Optional[sc.Variable] = None,
        wmax: Optional[sc.Variable] = None,
        seed: Optional[int] = None,
    ):
        self.facility = facility
        self.neutrons = int(neutrons)
        self.pulses = int(pulses)
        self.data = None

        if facility is not None:
            facility_params = facilities[self.facility]
            self.frequency = facility_params.frequency
            pulse_params = _make_pulses(
                neutrons=self.neutrons,
                p_time=facility_params.time,
                p_wav=facility_params.wavelength,
                sampling=sampling,
                frequency=self.frequency,
                pulses=pulses,
                wmin=wmin,
                wmax=wmax,
                seed=seed,
            )
            self.data = sc.DataArray(
                data=sc.ones(sizes=pulse_params["time"].sizes, unit="counts"),
                coords={
                    "time": pulse_params["time"],
                    "wavelength": pulse_params["wavelength"],
                    "speed": pulse_params["speed"],
                    "id": sc.arange("event", pulse_params["time"].size, unit=None).fold(
                        "event", sizes=pulse_params["time"].sizes
                    ),
                },
            )

    @classmethod
    def from_neutrons(
        cls,
        birth_times: sc.Variable,
        wavelengths: sc.Variable,
        frequency: Optional[sc.Variable] = None,
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
        source = cls(facility=None, neutrons=len(birth_times), pulses=pulses)
        source.frequency = _default_frequency(frequency, pulses)

        birth_times = (sc.arange("pulse", pulses) / source.frequency).to(
            unit=TIME_UNIT, copy=False
        ) + birth_times.to(unit=TIME_UNIT, copy=False)
        wavelengths = sc.broadcast(
            wavelengths.to(unit=WAV_UNIT, copy=False), sizes=birth_times.sizes
        )

        source.data = sc.DataArray(
            data=sc.ones(sizes=birth_times.sizes, unit="counts"),
            coords={
                "time": birth_times,
                "wavelength": wavelengths,
                "speed": wavelength_to_speed(wavelengths).to(unit="m/s", copy=False),
                "id": sc.arange("event", birth_times.size, unit=None).fold(
                    "event", sizes=birth_times.sizes
                ),
            },
        )

        return source

    @classmethod
    def from_distribution(
        cls,
        p_time: sc.DataArray,
        p_wav: sc.DataArray,
        neutrons: int = 1_000_000,
        pulses: int = 1,
        frequency: Optional[sc.Variable] = None,
        sampling: Optional[int] = 1000,
        seed: Optional[int] = None,
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
        p_time:
            Time probability distribution.
        p_wav:
            Wavelength probability distribution.
        neutrons:
            Number of neutrons in the pulse.
        pulses:
            Number of pulses.
        frequency:
            Frequency of the pulse.
        sampling:
            Number of points used to interpolate the probability distributions.
        seed:
            Seed for the random number generator.
        """

        source = cls(facility=None, neutrons=neutrons, pulses=pulses)
        source.frequency = _default_frequency(frequency, pulses)
        pulse_params = _make_pulses(
            neutrons=neutrons,
            p_time=p_time,
            p_wav=p_wav,
            frequency=source.frequency,
            pulses=pulses,
            sampling=sampling,
            seed=seed,
        )
        source.data = sc.DataArray(
            data=sc.ones(sizes=pulse_params["time"].sizes, unit="counts"),
            coords={
                "time": pulse_params["time"],
                "wavelength": pulse_params["wavelength"],
                "speed": pulse_params["speed"],
                "id": sc.arange("event", pulse_params["time"].size, unit=None).fold(
                    "event", sizes=pulse_params["time"].sizes
                ),
            },
        )
        return source

    def __len__(self) -> int:
        return self.data.sizes["pulse"]

    def plot(self, bins: int = 300) -> tuple:
        """
        Plot the pulses of the source.

        Parameters
        ----------
        bins:
            Number of bins to use for histogramming the neutrons.
        """
        fig, ax = plt.subplots(1, 2)
        dim = (set(self.data.dims) - {"pulse"}).pop()
        collapsed = sc.collapse(self.data, keep=dim)
        pp.plot(
            {k: da.hist(time=bins) for k, da in collapsed.items()},
            ax=ax[0],
        )
        pp.plot(
            {k: da.hist(wavelength=bins) for k, da in collapsed.items()},
            ax=ax[1],
        )
        fig.set_size_inches(10, 4)
        fig.tight_layout()
        return Plot(fig=fig, ax=ax)

    def as_readonly(self):
        return SourceParameters(
            data=self.data,
            facility=self.facility,
            neutrons=self.neutrons,
            frequency=self.frequency,
            pulses=self.pulses,
        )

    def __repr__(self) -> str:
        return (
            f"Source:\n"
            f"  pulses={self.pulses}, neutrons per pulse={self.neutrons}\n"
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
