# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import plopp as pp
import scipp as sc

from .component import ComponentReading
from .utils import wavelength_to_speed

TIME_UNIT = "us"
WAV_UNIT = "angstrom"
T_DIM = "birth_time"
W_DIM = "wavelength"


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


def _initialize_probability_distribution(
    frequency: sc.Variable,
    pulses: int,
    p: sc.DataArray | None = None,
    p_time: sc.DataArray | None = None,
    p_wav: sc.DataArray | None = None,
    tmin: sc.Variable | None = None,
    tmax: sc.Variable | None = None,
    wmin: sc.Variable | None = None,
    wmax: sc.Variable | None = None,
):
    """
    Initialize the probability distribution as a function of time and wavelength.
    The inputs can either be two separate 1D distributions for time and wavelength,
    or a single 2D distribution for both.
    The distributions should be supplied as DataArrays where the coordinates
    are the values of the distribution, and the values are the probability.

    In the case of two separate distributions, the time and wavelength distributions
    are considered independent (the two distributions are simply broadcast into a
    square 2D parameter space).

    Parameters
    ----------
    frequency:
        Pulse frequency.
    pulses:
        Number of pulses.
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

    if p.sizes[T_DIM] < 2 or p.sizes[W_DIM] < 2:
        raise ValueError(
            f"Distribution must have at least 2 points in each dimension. "
            f"Got {p.sizes[T_DIM]} in {T_DIM}, {p.sizes[W_DIM]} in {W_DIM}"
        )

    p.coords[T_DIM] = p.coords[T_DIM].to(dtype=float, unit=TIME_UNIT)
    p.coords[W_DIM] = p.coords[W_DIM].to(dtype=float, unit=WAV_UNIT)
    # Filter parameter space defined by limits
    ind_tmin, ind_tmax = 0, p.sizes[T_DIM]
    ind_wmin, ind_wmax = 0, p.sizes[W_DIM]
    trange = sc.arange(T_DIM, p.sizes[T_DIM])
    wrange = sc.arange(W_DIM, p.sizes[W_DIM])
    if tmin is not None:
        ind_tmin = max(trange[p.coords[T_DIM] >= tmin][0].value - 1, 0)
    if tmax is not None:
        ind_tmax = min(trange[p.coords[T_DIM] <= tmax][-1].value + 2, p.sizes[T_DIM])
    if wmin is not None:
        ind_wmin = max(wrange[p.coords[W_DIM] >= wmin][0].value - 1, 0)
    if wmax is not None:
        ind_wmax = min(wrange[p.coords[W_DIM] <= wmax][-1].value + 2, p.sizes[W_DIM])
    prob = p[T_DIM, ind_tmin:ind_tmax][W_DIM, ind_wmin:ind_wmax]
    # prob = sc.concat([prob] * pulses, dim='pulse')
    # prob.coords['birth_time'] = (sc.arange("pulse", pulses) / p.coords['frequency']).to(
    #     unit=TIME_UNIT, copy=False
    # ) + prob.coords['birth_time']

    p_sum = prob.data.sum()
    if p_sum.value <= 0:
        raise ValueError(
            "Distribution must have at least one positive probability value. "
            f"Sum of probabilities is {p_sum.value}"
        )

    return sc.concat([prob / p_sum] * pulses, dim='pulse')


def _make_data(
    birth_time: sc.Variable,
    wavelength: sc.Variable,
    period: sc.Variable,
    # frequency: sc.Variable,
    # distance: sc.Variable,
) -> sc.DataGroup:
    return sc.DataArray(
        data=sc.ones(sizes=birth_time.sizes, unit="counts"),
        coords={
            "birth_time": birth_time,
            "wavelength": wavelength,
            "speed": wavelength_to_speed(wavelength),
            "id": sc.arange("event", birth_time.size, unit=None).fold(
                "event", sizes=birth_time.sizes
            ),
            # "distance": distance,
            "eto": birth_time % period,
            "toa": birth_time,
            "birth_wavelength": wavelength,
        },
        masks={
            "blocked_by_others": sc.zeros(sizes=birth_time.sizes, unit=None, dtype=bool)
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
    seed:
        Seed for the random number generator.
    """  # noqa: E501

    def __init__(
        self,
        facility: str | None,
        neutrons: int = 1_000_000,
        pulses: int = 1,
        # TODO: Add frequency arg
        tmin: sc.Variable | None = None,
        tmax: sc.Variable | None = None,
        wmin: sc.Variable | None = None,
        wmax: sc.Variable | None = None,
        seed: int | None = None,
    ):
        self._facility = facility.lower() if facility is not None else None
        self._neutrons = int(neutrons)
        self._pulses = int(pulses)
        self._tmin = tmin
        self._tmax = tmax
        self._wmin = wmin
        self._wmax = wmax
        # self._data = None
        self._distribution = None

        self.seed = seed
        if self.seed is None:
            self.seed = np.random.SeedSequence().entropy

        if self._facility is not None:
            from .facilities import get_source_path

            file_path = get_source_path(self._facility)
            facility_pulse = sc.io.load_hdf5(file_path)

            self._frequency = facility_pulse.coords["frequency"]
            self._distance = facility_pulse.coords["distance"]
            self._distribution = _initialize_probability_distribution(
                p=facility_pulse,
                frequency=self._frequency,
                pulses=self._pulses,
                tmin=self._tmin,
                tmax=self._tmax,
                wmin=self._wmin,
                wmax=self._wmax,
            )
            self._post_init()

    def _post_init(self):
        print(self._distribution)
        self._pflat = self._distribution.flatten(
            dims=['wavelength', 'birth_time'], to='x'
        )

        t = self._distribution.coords[T_DIM]  # ['pulse', 0]
        w = self._distribution.coords[W_DIM]
        self._widths = {T_DIM: sc.empty_like(t), W_DIM: sc.empty_like(w)}

        dt = t[T_DIM, 1:] - t[T_DIM, :-1]
        self._widths[T_DIM][T_DIM, 1:-1] = 0.5 * (dt[T_DIM, :-1] + dt[T_DIM, 1:])
        self._widths[T_DIM][T_DIM, 0] = dt[T_DIM, 0]
        self._widths[T_DIM][T_DIM, -1] = dt[T_DIM, -1]

        dw = w[W_DIM, 1:] - w[W_DIM, :-1]
        self._widths[W_DIM][W_DIM, 1:-1] = 0.5 * (dw[W_DIM, :-1] + dw[W_DIM, 1:])
        self._widths[W_DIM][W_DIM, 0] = dw[W_DIM, 0]
        self._widths[W_DIM][W_DIM, -1] = dw[W_DIM, -1]

        self._widths[T_DIM] = (
            self._widths[T_DIM]
            .broadcast(sizes=self._distribution.sizes)
            .flatten(to='x')
        )
        self._widths[W_DIM] = (
            self._widths[W_DIM]
            .broadcast(sizes=self._distribution.sizes)
            .flatten(to='x')
        )

        self._distribution.coords[f"{T_DIM}_edges"] = _midpoints_to_edges(
            self._distribution.coords[T_DIM], dim=T_DIM
        )
        self._distribution.coords[f"{W_DIM}_edges"] = _midpoints_to_edges(
            self._distribution.coords[W_DIM], dim=W_DIM
        )

        self._tmin = (
            self._tmin
            if self._tmin is not None
            else self._distribution.coords[f"{T_DIM}_edges"].min()
        )
        self._tmax = (
            self._tmax
            if self._tmax is not None
            else self._distribution.coords[f"{T_DIM}_edges"].max()
        )
        self._wmin = (
            self._wmin
            if self._wmin is not None
            else self._distribution.coords[f"{W_DIM}_edges"].min()
        )
        self._wmax = (
            self._wmax
            if self._wmax is not None
            else self._distribution.coords[f"{W_DIM}_edges"].max()
        )

    def sample(self, neutrons: int | None = None) -> sc.DataArray:
        """
        Sample neutrons from the source.

        For the sampling, random.choice only allows to select from the values listed
        in the coordinate of the probability distribution arrays. This leads to data
        grouped into spikes and empty in between because the sampling resolution used
        in the linear interpolation above is usually kept low for performance.
        To make the distribution more uniform, we add some random (Gaussian) noise to
        the chosen values, which effectively fills in the gaps between the spikes.
        Scipy has some methods to sample from a continuous distribution, but they are
        prohibitively slow.
        See https://docs.scipy.org/doc/scipy/tutorial/stats/sampling.html for more
        information.

        Parameters
        ----------
        neutrons : int
            Number of neutrons to sample.
        """
        if self._distribution is None:
            return self._custom_neutrons

        if neutrons is None:
            neutrons = self._neutrons
        neutrons = int(neutrons)
        # n = 0
        # times = [[] for _ in range(self._pulses)]
        # wavs = [[] for _ in range(self._pulses)]
        # ntot = self._pulses * neutrons
        rng = np.random.default_rng(self.seed)
        period = self.period.to(unit=TIME_UNIT)

        pulses = []

        for ip in range(self._pulses):
            times = []
            wavs = []
            # Because of the added noise, some values end up being outside the specified
            # range for the birth times and wavelengths. Using naive clipping leads to
            # pile-up on the edges of the range. To avoid this, we remove the outliers and
            # resample until we have the desired number of neutrons.
            count = 0
            p = self._pflat['pulse', ip]
            n = 0
            while n < neutrons:
                count += 1
                size = neutrons - n
                inds = rng.choice(len(p), size=size, p=p.values, replace=True)
                t = p.coords[T_DIM].values[inds] + (
                    rng.normal(scale=0.5, size=size) * self._widths[T_DIM].values[inds]
                )
                w = p.coords[W_DIM].values[inds] + (
                    rng.normal(scale=0.5, size=size) * self._widths[W_DIM].values[inds]
                )
                sel = (
                    (t >= self._tmin.value)
                    & (t <= self._tmax.value)
                    & (w >= self._wmin.value)
                    & (w <= self._wmax.value)
                )
                # wsel = (w >= self._wmin.value) & (w <= self._wmax.value)
                # m = 0
                # for i in range(self._pulses):
                #     sel = wsel & (
                #         (t >= (self._tmin + (period * i)).value)
                #         & (t <= (self._tmax + (period * i)).value)
                #     )
                times.append(t[sel])
                wavs.append(w[sel])
                # m += sel.sum()
                n += sel.sum()

            times = np.concatenate(times)
            times += period.value * ip
            pulses.append(
                _make_data(
                    birth_time=sc.array(dims=["event"], values=times, unit=TIME_UNIT),
                    wavelength=sc.array(
                        dims=["event"], values=np.concatenate(wavs), unit=WAV_UNIT
                    ),
                    period=period,
                    # frequency=self._frequency,
                    # distance=self._distance,
                ).fold(dim="event", sizes={"pulse": 1, "event": -1})
            )

        print(count, "iterations during sampling")

        # return times

        # # dim = "event"
        # events = sc.DataGroup(
        #     {
        #         f"pulse-{i}": _make_data(
        #             birth_time=sc.array(
        #                 dims=["event"], values=np.concatenate(t), unit=TIME_UNIT
        #             ),
        #             wavelength=sc.array(
        #                 dims=["event"], values=np.concatenate(w), unit=WAV_UNIT
        #             ),
        #             frequency=self._frequency,
        #         )
        #         for i, (t, w) in enumerate(zip(times, wavs, strict=True))
        #     }
        # )
        # birth_time = sc.array(
        #     dims=["pulse", "event"],
        #     values=[np.concatenate(pieces_t) for pieces_t in times],
        #     unit=TIME_UNIT,
        # )  # .fold(dim=dim, sizes={"pulse": self._pulses, dim: -1})

        # wavelengths = np.concatenate(wavs)
        # if np.any(wavelengths <= 0):
        #     warnings.warn(
        #         "Some neutron wavelengths are negative.", RuntimeWarning, stacklevel=2
        #     )
        # wavelength = sc.DataGroup(
        #     {
        #         f"pulse-{i}": sc.array(
        #             dims=["event"], values=np.concatenate(v), unit=WAV_UNIT
        #         )
        #         for i, v in enumerate(wavs)
        #     }
        # )

        #  "distance": self._distance,
        # "eto": self._data.coords["birth_time"]
        # % (1.0 / self._frequency).to(unit=TIME_UNIT, copy=False),
        # "toa": self._data.coords["birth_time"],
        # "birth_wavelength": self._data.coords["wavelength"],

        # print("PULSES", pulses)
        # print("===============")
        # print(sc.concat(pulses, dim="pulse"))

        return sc.concat(pulses, dim="pulse").assign_coords(
            distance=self._distance, frequency=self._frequency
        )

        # events = sc.DataGroup(
        #     {
        #         f"pulse-{i}": _make_data(
        #             birth_time=sc.array(
        #                 dims=["event"], values=np.concatenate(t), unit=TIME_UNIT
        #             ),
        #             wavelength=sc.array(
        #                 dims=["event"], values=np.concatenate(w), unit=WAV_UNIT
        #             ),
        #             frequency=self._frequency,
        #             distance=self._distance,
        #         )
        #         for i, (t, w) in enumerate(zip(times, wavs, strict=True))
        #     }
        # )
        # return events

    def copy(self):
        """
        Make a copy of the source.
        """
        out = self.__class__(
            facility=None,
            neutrons=self._neutrons,
            pulses=self._pulses,
        )
        out._facility = self._facility
        out._frequency = self._frequency.copy(deep=True)
        out._distance = self._distance.copy(deep=True)
        out._distribution = self._distribution.copy(deep=True)
        out._post_init()
        return out

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
    def period(self) -> sc.Variable:
        """
        The period of the pulse.
        """
        return sc.reciprocal(self.frequency)

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
    def distribution(self) -> sc.DataArray:
        """
        The probability distribution of the source. Neutrons will be sampled from this
        distribution when calling the :func:`sample` method.
        """
        return self._distribution

    # @property
    # def data(self) -> sc.DataArray:
    #     """
    #     The data array containing the neutrons in the pulse.
    #     """
    #     return self._data.assign_coords(
    #         {
    #             "distance": self._distance,
    #             "eto": self._data.coords["birth_time"]
    #             % (1.0 / self._frequency).to(unit=TIME_UNIT, copy=False),
    #             "toa": self._data.coords["birth_time"],
    #             "birth_wavelength": self._data.coords["wavelength"],
    #         }
    #     )

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

        source._custom_neutrons = _make_data(
            birth_time=birth_times,
            wavelength=wavelengths,
            distance=source._distance,
            frequency=source._frequency,
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
        tmin: sc.Variable | None = None,
        tmax: sc.Variable | None = None,
        wmin: sc.Variable | None = None,
        wmax: sc.Variable | None = None,
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

        source = cls(
            facility=None,
            neutrons=neutrons,
            pulses=pulses,
            tmin=tmin,
            tmax=tmax,
            wmin=wmin,
            wmax=wmax,
        )
        source._distance = (
            distance if distance is not None else sc.scalar(0.0, unit="m")
        )
        source._frequency = _default_frequency(frequency, pulses)

        if p is not None:
            p = _bin_edges_to_midpoints(p, dims=["birth_time", "wavelength"])
        if p_time is not None:
            p_time = _bin_edges_to_midpoints(p_time, dims=["birth_time"])
        if p_wav is not None:
            p_wav = _bin_edges_to_midpoints(p_wav, dims=["wavelength"])

        source._probability = _initialize_probability_distribution(
            p=p,
            p_time=p_time,
            p_wav=p_wav,
            frequency=source._frequency,
            pulses=source._pulses,
            tmin=source._tmin,
            tmax=source._tmax,
            wmin=source._wmin,
            wmax=source._wmax,
        )
        source._post_init()

        # pulse_params = _make_pulses(
        #     neutrons=neutrons,
        #     p=p,
        #     p_time=p_time,
        #     p_wav=p_wav,
        #     frequency=source._frequency,
        #     pulses=pulses,
        #     seed=seed,
        # )
        # source._data = sc.DataArray(
        #     data=sc.ones(sizes=pulse_params["birth_time"].sizes, unit="counts"),
        #     coords={
        #         "birth_time": pulse_params["birth_time"],
        #         "wavelength": pulse_params["wavelength"],
        #         "speed": pulse_params["speed"],
        #         "id": sc.arange(
        #             "event", pulse_params["birth_time"].size, unit=None
        #         ).fold("event", sizes=pulse_params["birth_time"].sizes),
        #     },
        # )
        # source._probability = pulse_params["probability"]
        return source

    def __len__(self) -> int:
        return self._pulses

    def plot(self) -> tuple:
        """
        Plot the pulses of the source.
        """
        style = {"ls": "-", "marker": None}
        f1 = self._distribution['pulse', 0].sum('wavelength').plot(**style)
        f2 = self._distribution['pulse', 0].sum('birth_time').plot(**style)
        return self._distribution['pulse', 0].plot() + f1 + f2

    def as_readonly(self, data: sc.DataArray):
        return SourceReading(
            data=data,
            # .assign_masks(
            #     blocked_by_others=sc.zeros_like(data.data, dtype=bool, unit=None)
            # ),
            # data=None,
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
