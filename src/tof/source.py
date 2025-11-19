# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import warnings
from dataclasses import dataclass

import numpy as np
import plopp as pp
import scipp as sc

from .utils import wavelength_to_speed

TIME_UNIT = "us"
WAV_UNIT = "angstrom"


def _default_frequency(frequency: sc.Variable | None, pulses: int) -> sc.Variable:
    if frequency is None:
        if pulses > 1:
            raise ValueError(
                "If pulses is greater than one, a frequency must be supplied."
            )
        frequency = 1.0 * sc.Unit("Hz")
    return frequency


def _make_pulses(
    neutrons: int,
    frequency: sc.Variable,
    pulses: int,
    seed: int | None,
    p: sc.DataArray | None = None,
    p_time: sc.DataArray | None = None,
    p_wav: sc.DataArray | None = None,
    wmin: sc.Variable | None = None,
    wmax: sc.Variable | None = None,
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
    seed:
        Seed for the random number generator.
    p:
        2D probability distribution for a single pulse.
    p_time:
        Time probability distribution for a single pulse.
    p_wav:
        Wavelength probability distribution for a single pulse.
    wmin:
        Minimum neutron wavelength.
    wmax:
        Maximum neutron wavelength.
    """
    t_dim = "birth_time"
    w_dim = "wavelength"

    if p is None:
        if None in (p_time, p_wav):
            raise ValueError(
                "Either p (2D) or both p_time (1D) and p_wav (1D) must be supplied."
            )
        p_wav_sum = p_wav.data.sum()
        p_time_sum = p_time.data.sum()
        if p_wav_sum.value <= 0:
            raise ValueError(
                "Wavelength distribution must have at least one positive "
                f"probability value. Sum of probabilities is {p_wav_sum.value}"
            )
        if p_time_sum.value <= 0:
            raise ValueError(
                "Time distribution must have at least one positive "
                f"probability value. Sum of probabilities is {p_time_sum.value}"
            )
        p = (p_wav / p_wav_sum) * (p_time / p_time_sum)
    else:
        p = p.copy(deep=False)

    if p.sizes[t_dim] < 2 or p.sizes[w_dim] < 2:
        raise ValueError(
            f"Distribution must have at least 2 points in each dimension. "
            f"Got {p.sizes[t_dim]} in {t_dim}, {p.sizes[w_dim]} in {w_dim}"
        )

    p.coords[t_dim] = p.coords[t_dim].to(dtype=float, unit=TIME_UNIT)
    p.coords[w_dim] = p.coords[w_dim].to(dtype=float, unit=WAV_UNIT)

    tmin = p.coords[t_dim].min()
    tmax = p.coords[t_dim].max()
    if wmin is None:
        wmin = p.coords[w_dim].min()
    if wmax is None:
        wmax = p.coords[w_dim].max()

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
    dt = 0.5 * (tmax - tmin).value / (p.sizes[t_dim] - 1)
    dw = 0.5 * (wmax - wmin).value / (p.sizes[w_dim] - 1)

    # Because of the added noise, some values end up being outside the specified range
    # for the birth times and wavelengths. Using naive clipping leads to pile-up on the
    # edges of the range. To avoid this, we remove the outliers and resample until we
    # have the desired number of neutrons.
    n = 0
    times = []
    wavs = []
    ntot = pulses * neutrons
    rng = np.random.default_rng(seed)
    p_flat = p.flatten(to='x')

    p_sum = p_flat.data.sum()
    if p_sum.value <= 0:
        raise ValueError(
            "Distribution must have at least one positive probability value. "
            f"Sum of probabilities is {p_sum.value}"
        )
    p_flat /= p_sum
    while n < ntot:
        size = ntot - n
        inds = rng.choice(len(p_flat), size=size, p=p_flat.values)
        t = p_flat.coords[t_dim].values[inds] + rng.normal(scale=dt, size=size)
        w = p_flat.coords[w_dim].values[inds] + rng.normal(scale=dw, size=size)
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
    ).to(unit=TIME_UNIT, copy=False)

    wavelengths = np.concatenate(wavs)
    if np.any(wavelengths <= 0):
        warnings.warn(
            "Some neutron wavelengths are negative.", RuntimeWarning, stacklevel=2
        )
    wavelength = sc.array(dims=[dim], values=wavelengths, unit=WAV_UNIT).fold(
        dim=dim, sizes={"pulse": pulses, dim: neutrons}
    )
    speed = wavelength_to_speed(wavelength)
    return {
        "birth_time": birth_time,
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
    wmin:
        Minimum neutron wavelength.
    wmax:
        Maximum neutron wavelength.
    seed:
        Seed for the random number generator.
    """

    def __init__(
        self,
        facility: str | None,
        neutrons: int = 1_000_000,
        pulses: int = 1,
        wmin: sc.Variable | None = None,
        wmax: sc.Variable | None = None,
        seed: int | None = None,
    ):
        self._facility = facility.lower() if facility is not None else None
        self._neutrons = int(neutrons)
        self._pulses = int(pulses)
        self._data = None
        self.seed = seed

        if self._facility is not None:
            from .facilities import get_source_path

            file_path = get_source_path(self._facility)
            facility_pulse = sc.io.load_hdf5(file_path)

            self._frequency = facility_pulse.coords["frequency"]
            self._distance = facility_pulse.coords["distance"]
            pulse_params = _make_pulses(
                neutrons=self._neutrons,
                p=facility_pulse,
                frequency=self._frequency,
                pulses=self._pulses,
                wmin=wmin,
                wmax=wmax,
                seed=seed,
            )
            self._data = sc.DataArray(
                data=sc.ones(sizes=pulse_params["birth_time"].sizes, unit="counts"),
                coords={
                    "birth_time": pulse_params["birth_time"],
                    "wavelength": pulse_params["wavelength"],
                    "speed": pulse_params["speed"],
                    "id": sc.arange(
                        "event", pulse_params["birth_time"].size, unit=None
                    ).fold("event", sizes=pulse_params["birth_time"].sizes),
                },
            )

    @property
    def facility(self) -> str | None:
        """
        The name of the facility used to create the source.
        """
        return self._facility

    @property
    def neutrons(self) -> int:
        """
        The number of neutrons per pulse.
        """
        return self._neutrons

    @property
    def frequency(self) -> sc.Variable:
        """
        The frequency of the pulse.
        """
        return self._frequency

    @property
    def distance(self) -> sc.Variable:
        """
        The position of the source along the beamline.
        """
        return self._distance

    @property
    def pulses(self) -> int:
        """
        The number of pulses.
        """
        return self._pulses

    @property
    def data(self) -> sc.DataArray:
        """
        The data array containing the neutrons in the pulse.
        """
        return self._data

    @classmethod
    def from_neutrons(
        cls,
        birth_times: sc.Variable,
        wavelengths: sc.Variable,
        frequency: sc.Variable | None = None,
        pulses: int = 1,
        distance: sc.Variable | None = None,
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
        distance:
            Position of the source along the beamline.
        """
        source = cls(facility=None, neutrons=len(birth_times), pulses=pulses)
        source._frequency = _default_frequency(frequency, pulses)
        source._distance = (
            distance if distance is not None else sc.scalar(0.0, unit="m")
        )

        if np.any(wavelengths.values <= 0):
            warnings.warn(
                "Some neutron wavelengths are negative.", RuntimeWarning, stacklevel=2
            )

        birth_times = (sc.arange("pulse", pulses) / source._frequency).to(
            unit=TIME_UNIT, copy=False
        ) + birth_times.to(unit=TIME_UNIT, copy=False)
        wavelengths = sc.broadcast(
            wavelengths.to(unit=WAV_UNIT, copy=False), sizes=birth_times.sizes
        )

        source._data = sc.DataArray(
            data=sc.ones(sizes=birth_times.sizes, unit="counts"),
            coords={
                "birth_time": birth_times,
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
        p: sc.DataArray | None = None,
        p_time: sc.DataArray | None = None,
        p_wav: sc.DataArray | None = None,
        neutrons: int = 1_000_000,
        pulses: int = 1,
        frequency: sc.Variable | None = None,
        seed: int | None = None,
        distance: sc.Variable | None = None,
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
        p:
            2D probability distribution for a single pulse.

            .. versionadded:: 25.12.0
        p_time:
            Time probability distribution (1D) for a single pulse.
        p_wav:
            Wavelength probability distribution (1D) for a single pulse.
        neutrons:
            Number of neutrons in the pulse.
        pulses:
            Number of pulses.
        frequency:
            Frequency of the pulse.
        seed:
            Seed for the random number generator.
        distance:
            Position of the source along the beamline.
        """

        source = cls(facility=None, neutrons=neutrons, pulses=pulses)
        source._distance = (
            distance if distance is not None else sc.scalar(0.0, unit="m")
        )
        source._frequency = _default_frequency(frequency, pulses)
        pulse_params = _make_pulses(
            neutrons=neutrons,
            p=p,
            p_time=p_time,
            p_wav=p_wav,
            frequency=source._frequency,
            pulses=pulses,
            seed=seed,
        )
        source._data = sc.DataArray(
            data=sc.ones(sizes=pulse_params["birth_time"].sizes, unit="counts"),
            coords={
                "birth_time": pulse_params["birth_time"],
                "wavelength": pulse_params["wavelength"],
                "speed": pulse_params["speed"],
                "id": sc.arange(
                    "event", pulse_params["birth_time"].size, unit=None
                ).fold("event", sizes=pulse_params["birth_time"].sizes),
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
        dim = (set(self.data.dims) - {"pulse"}).pop()
        collapsed = sc.collapse(self.data, keep=dim)
        f1 = pp.plot(
            {k: da.hist(birth_time=bins) for k, da in collapsed.items()},
        )
        f2 = pp.plot(
            {k: da.hist(wavelength=bins) for k, da in collapsed.items()},
        )
        return f1 + f2

    def as_readonly(self):
        return SourceParameters(
            data=self.data,
            facility=self.facility,
            neutrons=self.neutrons,
            frequency=self.frequency,
            pulses=self.pulses,
            distance=self.distance,
        )

    def __repr__(self) -> str:
        return (
            f"Source:\n"
            f"  pulses={self.pulses}, neutrons per pulse={self.neutrons}\n"
            f"  frequency={self.frequency:c}\n  facility='{self.facility}'\n"
            f"  distance={self.distance:c}"
        )

    def as_json(self) -> dict:
        """
        Return the source as a JSON-serializable dictionary.
        """
        return {
            'facility': self.facility,
            'neutrons': int(self.neutrons),
            'pulses': self.pulses,
            'seed': self.seed,
            'type': 'source',
        }


@dataclass(frozen=True)
class SourceParameters:
    """
    Read-only container for the parameters of a source.
    """

    data: sc.DataArray
    facility: str | None
    neutrons: int
    frequency: sc.Variable
    pulses: int
    distance: sc.Variable
