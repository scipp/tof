# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import plopp as pp
import scipp as sc
from matplotlib.path import Path

from .chopper import Chopper
from .component import ComponentReading
from .optimization import FrameSequence, Subframe, polygon_grid_overlap_mask
from .utils import wavelength_to_speed

TIME_UNIT = "us"
WAV_UNIT = "angstrom"
T_DIM = "birth_time"
W_DIM = "wavelength"


@dataclass(frozen=True)
class SourceReading(ComponentReading):
    """
    Read-only container for the parameters of a source.
    """

    data: sc.DataArray
    facility: str | None
    neutrons: int
    frequency: sc.Variable
    pulses: int
    distance: sc.Variable

    @property
    def kind(self) -> str:
        return "source"

    def plot_on_time_distance_diagram(self, ax, pulse) -> None:
        birth_time = self.data.coords["birth_time"]["pulse", pulse]
        tmin = birth_time.min().value
        dist = self.distance.value
        ax.plot([tmin, birth_time.max().value], [dist] * 2, color="gray", lw=3)
        ax.text(tmin, dist, "Pulse", ha="left", va="top", color="gray")


@dataclass
class SourceDistribution:
    probability: sc.DataArray
    acceptance_polygons: list[Subframe]


def _default_frequency(frequency: sc.Variable | None, pulses: int) -> sc.Variable:
    if frequency is None:
        if pulses > 1:
            raise ValueError(
                "If pulses is greater than one, a frequency must be supplied."
            )
        frequency = 1.0 * sc.Unit("Hz")
    return frequency


def _bin_edges_to_midpoints(
    da: sc.DataArray, dims: list[str] | tuple[str]
) -> sc.DataArray:
    return da.assign_coords(
        {
            dim: sc.midpoints(da.coords[dim], dim)
            for dim in dims
            if da.coords.is_edges(dim)
        }
    )


def _midpoints_to_edges(x, dim):
    if x.sizes[dim] < 2:
        half = sc.scalar(0.5, unit=x.unit)
        return sc.concat([x[dim, 0:1] - half, x[dim, 0:1] + half], dim)
    else:
        center = sc.midpoints(x, dim=dim)
        # Note: use range of 0:1 to keep dimension dim in the slice to avoid
        # switching round dimension order in concatenate step.
        left = center[dim, 0:1] - (x[dim, 1] - x[dim, 0])
        right = center[dim, -1] + (x[dim, -1] - x[dim, -2])
        return sc.concat([left, center, right], dim)


def _compute_grid_spacing(x: sc.Variable, dim: str) -> sc.Variable:
    out = sc.empty_like(x)
    dx = x[dim, 1:] - x[dim, :-1]
    out[dim, 1:-1] = 0.5 * (dx[dim, :-1] + dx[dim, 1:])
    out[dim, 0] = dx[dim, 0]
    out[dim, -1] = dx[dim, -1]
    return out


def _load_facility_pulse_profile(facility):
    from .facilities import get_source_path

    file_path = get_source_path(facility)
    facility_pulse = sc.io.load_hdf5(file_path)
    facility_pulse.coords.update(
        {
            f"{T_DIM}_edges": _midpoints_to_edges(
                facility_pulse.coords[T_DIM], T_DIM
            ).to(unit=TIME_UNIT, copy=False),
            f"{W_DIM}_edges": _midpoints_to_edges(
                facility_pulse.coords[W_DIM], W_DIM
            ).to(unit=WAV_UNIT, copy=False),
        }
    )
    return facility_pulse


def _make_pulses(
    neutrons: int,
    frequency: sc.Variable,
    pulses: int,
    seed: int | None,
    p: sc.DataArray | None = None,
):
    """
    Create pulses from a 2D time and wavelength probability distribution.
    The distribution ``p`` should be supplied as a DataArray where the coordinates
    are the values of the distribution, and the values are the probability.

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
    """

    acceptance_polygons = p.acceptance_polygons
    p = p.probability.copy(deep=False)

    if p.sizes[T_DIM] < 2 or p.sizes[W_DIM] < 2:
        raise ValueError(
            f"Distribution must have at least 2 points in each dimension. "
            f"Got {p.sizes[T_DIM]} in {T_DIM}, {p.sizes[W_DIM]} in {W_DIM}"
        )

    p.coords[T_DIM] = p.coords[T_DIM].to(dtype=float, unit=TIME_UNIT)
    p.coords[W_DIM] = p.coords[W_DIM].to(dtype=float, unit=WAV_UNIT)

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

    widths = {
        dim: _compute_grid_spacing(p.coords[dim], dim)
        .broadcast(sizes=p.sizes)
        .flatten(to='x')
        for dim in (T_DIM, W_DIM)
    }

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
        # The first iteration, we sample all neutrons. Because some get discarded, we
        # will have to iterate at least twice. Instead of just sampling the missing
        # number of neutrons, subsequent iterations over-sample (here we choose 50% of
        # the total number requested) and then trim the excess.
        # This leads to less iterations overall, as we can have many iterations at the
        # end when we are missing just one or two neutrons and we keep sampling them
        # outside of the accepted regions.
        size = max(ntot - n, ntot // 2)
        inds = rng.choice(len(p_flat), size=size, p=p_flat.values)
        t = p_flat.coords[T_DIM].values[inds] + (
            rng.normal(scale=0.5, size=size) * widths[T_DIM].values[inds]
        )
        w = p_flat.coords[W_DIM].values[inds] + (
            rng.normal(scale=0.5, size=size) * widths[W_DIM].values[inds]
        )

        # We filter events that are outside the accepted regions. Accepted regions can
        # be defined by either a cropping of the probability distribution (using tmin,
        # tmax, wmin, wmax) or generated by a chopper acceptance diagram for sources
        # that are optimized for a specific chopper cascade.
        points = np.array([t, w]).T
        sel = np.zeros(shape=t.shape, dtype=bool)
        for poly in acceptance_polygons:
            verts = np.column_stack(
                [
                    poly.time.to(unit=TIME_UNIT).values,
                    poly.wavelength.to(unit=WAV_UNIT).values,
                ]
            )
            # Use Matplotlib's efficient Path.contains_points
            path = Path(verts)
            sel |= path.contains_points(points)

        # We could end up with more neutrons than needed, so only select the number
        # we need.
        inds = np.arange(size)[sel][: ntot - n]
        times.append(t[inds])
        wavs.append(w[inds])
        n += len(inds)

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


def _optimize_source(
    p,
    wmin: sc.Variable | None = None,
    wmax: sc.Variable | None = None,
    tmin: sc.Variable | None = None,
    tmax: sc.Variable | None = None,
    choppers: list[Chopper] | None = None,
) -> SourceDistribution:
    """
    Optimize the source distribution by applying acceptance criteria.
    If choppers are provided, they will be used to define the acceptance diagram
    computed analytically.
    Otherwise, a single acceptance polygon will be created based on the provided
    min/max values for time and wavelength.

    Probabilities outside the acceptance regions will be set to zero.
    """
    time_edges = p.coords[f"{T_DIM}_edges"]
    wave_edges = p.coords[f"{W_DIM}_edges"]

    if choppers is not None:
        frames = FrameSequence.from_source_pulse(
            time_min=time_edges.min(),
            time_max=time_edges.max(),
            wavelength_min=wave_edges.min(),
            wavelength_max=wave_edges.max(),
        )
        frames = frames.chop(choppers)
        # Propagate frames back to source
        frames = FrameSequence(
            [frame.propagate_to(p.coords['distance']) for frame in frames]
        )
        polygons = frames[-1].subframes
    else:
        if tmin is None:
            tmin = time_edges[0]
        if tmax is None:
            tmax = time_edges[-1]
        if wmin is None:
            wmin = wave_edges[0]
        if wmax is None:
            wmax = wave_edges[-1]
        polygons = [
            Subframe(
                time=sc.concat([tmin, tmax, tmax, tmin, tmin], dim='vertex'),
                wavelength=sc.concat([wmin, wmin, wmax, wmax, wmin], dim='vertex'),
            )
        ]

    X, Y = np.meshgrid(time_edges.values, wave_edges.values)
    mask = np.zeros(shape=p.shape, dtype=bool)
    for poly in polygons:
        mask |= polygon_grid_overlap_mask(
            np.column_stack(
                [
                    poly.time.to(unit=TIME_UNIT).values,
                    poly.wavelength.to(unit=WAV_UNIT).values,
                ]
            ),
            X,
            Y,
        )

    prob = p.copy(deep=True)
    prob.values = np.where(mask, prob.values, 0.0)
    return SourceDistribution(probability=prob, acceptance_polygons=polygons)


def _sample_source_data_from_distribution(
    neutrons: int,
    frequency: sc.Variable,
    p: sc.Variable | None,
    p_time: sc.Variable | None,
    p_wav: sc.Variable | None,
    distance: sc.Variable | None,
    wmin: sc.Variable | None,
    wmax: sc.Variable | None,
    tmin: sc.Variable | None,
    tmax: sc.Variable | None,
    seed: int | None,
    optimize_for: list[Chopper] | None,
    pulses: int,
) -> sc.DataArray:
    """
    Sample neutrons from the source distribution.
    """
    if p is not None:
        if (p_time is not None) or (p_wav is not None):
            raise ValueError(
                "If p (2D) is supplied, both p_time (1D) and p_wav (1D) must be None."
            )
        p = _bin_edges_to_midpoints(p, dims=["birth_time", "wavelength"])
    else:
        if None in (p_time, p_wav):
            raise ValueError(
                "Either p (2D) or both p_time (1D) and p_wav (1D) must be supplied."
            )
        p_time = _bin_edges_to_midpoints(p_time, dims=["birth_time"])
        p_wav = _bin_edges_to_midpoints(p_wav, dims=["wavelength"])

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

    p.coords.update(
        {
            f"{T_DIM}_edges": _midpoints_to_edges(p.coords[T_DIM], T_DIM).to(
                unit=TIME_UNIT, copy=False
            ),
            f"{W_DIM}_edges": _midpoints_to_edges(p.coords[W_DIM], W_DIM).to(
                unit=WAV_UNIT, copy=False
            ),
        }
    )
    if 'distance' not in p.coords:
        p.coords['distance'] = distance

    distribution = _optimize_source(
        p=p,
        wmin=wmin,
        wmax=wmax,
        tmin=tmin,
        tmax=tmax,
        choppers=optimize_for,
    )

    pulse_params = _make_pulses(
        neutrons=neutrons,
        p=distribution,
        frequency=frequency,
        pulses=pulses,
        seed=seed,
    )
    return sc.DataArray(
        data=sc.ones(sizes=pulse_params["birth_time"].sizes, unit="counts"),
        coords={
            "birth_time": pulse_params["birth_time"],
            "wavelength": pulse_params["wavelength"],
            "speed": pulse_params["speed"],
            "id": sc.arange("event", pulse_params["birth_time"].size, unit=None).fold(
                "event", sizes=pulse_params["birth_time"].sizes
            ),
        },
    )


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
        Name of a pre-defined pulse shape from a neutron facility. Currently, the
        following facilities are supported:

        - 'ess': the standard ESS source profile, applicable for all ESS instruments
        - 'ess-odin': a source specific to the ESS Odin instrument, sampled at the location where cold and thermal neutrons are combined (~2.35m away from the surface of the moderator)
    neutrons:
        Number of neutrons per pulse.
    pulses:
        Number of pulses.
    wmin:
        Minimum neutron wavelength.
    wmax:
        Maximum neutron wavelength.
    tmin:
        Minimum neutron birth time.
    tmax:
        Maximum neutron birth time.
    seed:
        Seed for the random number generator.
    optimize_for:
        List of choppers to optimize the source for. A chopper acceptance diagram will
        be overlaid on the source distribution, and samples will be taken only from
        the regions where the chopper cascade is accepting neutrons.

        .. versionadded:: 26.4.0
    """  # noqa: E501

    def __init__(
        self,
        facility: str | None,
        neutrons: int = 1_000_000,
        pulses: int = 1,
        wmin: sc.Variable | None = None,
        wmax: sc.Variable | None = None,
        tmin: sc.Variable | None = None,
        tmax: sc.Variable | None = None,
        seed: int | None = None,
        optimize_for: list[Chopper] | None = None,
    ):
        self._facility = facility.lower() if facility is not None else None
        self._neutrons = int(neutrons)
        self._pulses = int(pulses)
        self._data = None
        self.seed = seed

        if self._facility is None:
            return

        facility_pulse = _load_facility_pulse_profile(self._facility)

        self._distance = facility_pulse.coords['distance']
        self._frequency = facility_pulse.coords['frequency']
        self._data = _sample_source_data_from_distribution(
            p=facility_pulse,
            neutrons=neutrons,
            pulses=pulses,
            frequency=self._frequency,
            seed=seed,
            distance=self._distance,
            wmin=wmin,
            wmax=wmax,
            tmin=tmin,
            tmax=tmax,
            optimize_for=optimize_for,
            p_time=None,
            p_wav=None,
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
        return self._data.assign_coords(
            {
                "distance": self._distance,
                "eto": self._data.coords["birth_time"]
                % (1.0 / self._frequency).to(unit=TIME_UNIT, copy=False),
                "toa": self._data.coords["birth_time"],
            }
        )

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
        wmin: sc.Variable | None = None,
        wmax: sc.Variable | None = None,
        tmin: sc.Variable | None = None,
        tmax: sc.Variable | None = None,
        optimize_for: list[Chopper] | None = None,
    ):
        """
        Create source pulses from time and wavelength probability distributions.
        The distributions should be supplied as DataArrays where the coordinates
        are the values of the distribution, and the values are the probability.
        This can either be a single 2D distribution, or separate 1D distributions
        for time and wavelength. Note that in this case, the time and wavelength
        distributions are considered independent. A neutron with a randomly selected
        birth time from ``p_time`` can adopt any wavelength in ``p_wav`` (in other
        words, the two distributions are simply broadcast into a square 2D parameter
        space).

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
        wmin:
            Minimum wavelength.
        wmax:
            Maximum wavelength.
        tmin:
            Minimum birth time.
        tmax:
            Maximum birth time.
        optimize_for:
            List of choppers to optimize the source for. A chopper acceptance diagram
            will be overlaid on the source distribution, and samples will be taken only
            from the regions where the chopper cascade is accepting neutrons.

            .. versionadded:: 26.4.0
        """

        source = cls(facility=None, neutrons=neutrons, pulses=pulses)

        if distance is None:
            distance = sc.scalar(0.0, unit="m")
        frequency = _default_frequency(frequency, pulses)

        source._distance = distance
        source._frequency = frequency
        source._data = _sample_source_data_from_distribution(
            p=p,
            p_time=p_time,
            p_wav=p_wav,
            neutrons=neutrons,
            pulses=pulses,
            frequency=frequency,
            seed=seed,
            distance=distance,
            wmin=wmin,
            wmax=wmax,
            tmin=tmin,
            tmax=tmax,
            optimize_for=optimize_for,
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
        return SourceReading(
            data=self.data.assign_masks(
                blocked_by_others=sc.zeros_like(self.data.data, dtype=bool, unit=None)
            ),
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

    @classmethod
    def from_json(cls, params: dict) -> Source:
        """
        Create a source from a JSON-serializable dictionary.
        Currently, only sources from facilities are supported when loading from JSON.

        The dictionary should have the following format:

        .. code-block:: json

            {
                "type": "source",
                "facility": "ess",
                "neutrons": 1000000,
                "pulses": 1,
                "seed": 42
            }
        """
        if params.get("facility") is None:
            raise ValueError(
                "Currently, only sources from facilities are supported when "
                "loading from JSON."
            )
        return cls(**{k: v for k, v in params.items() if k != "type"})

    def as_json(self) -> dict:
        """
        Return the source as a JSON-serializable dictionary.

        .. versionadded:: 25.11.0
        """
        return {
            'facility': self.facility,
            'neutrons': int(self.neutrons),
            'pulses': self.pulses,
            'seed': self.seed,
            'type': 'source',
        }
