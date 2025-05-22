# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

from itertools import chain
from types import MappingProxyType

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc
from matplotlib.collections import LineCollection

from .chopper import Chopper, ChopperReading
from .detector import Detector, DetectorReading
from .source import Source, SourceParameters
from .utils import Plot, one_mask


def _add_rays(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: np.ndarray | str,
    cbar: bool = True,
    cmap: str = "gist_rainbow_r",
    vmin: float | None = None,
    vmax: float | None = None,
    cax: plt.Axes | None = None,
    zorder: int = 1,
):
    coll = LineCollection(np.stack((x, y), axis=2), zorder=zorder)
    if isinstance(color, str):
        coll.set_color(color)
    else:
        coll.set_cmap(plt.colormaps[cmap])
        coll.set_array(color)
        coll.set_norm(plt.Normalize(vmin, vmax))
        if cbar:
            cb = plt.colorbar(coll, ax=ax, cax=cax)
            cb.ax.yaxis.set_label_coords(-0.9, 0.5)
            cb.set_label("Wavelength [Å]")
    ax.add_collection(coll)


class Result:
    """
    Result of a simulation.

    Parameters
    ----------
    source:
        The source of neutrons.
    choppers:
        The choppers in the model.
    detectors:
        The detectors in the model.
    """

    def __init__(
        self,
        source: Source,
        choppers: dict[str, Chopper],
        detectors: dict[str, Detector],
    ):
        self._source = source.as_readonly()
        self._choppers = {}
        for name, chopper in choppers.items():
            self._choppers[name] = ChopperReading(
                distance=chopper["distance"],
                name=chopper["name"],
                frequency=chopper["frequency"],
                open=chopper["open"],
                close=chopper["close"],
                phase=chopper["phase"],
                open_times=chopper["open_times"],
                close_times=chopper["close_times"],
                data=chopper["data"],
            )

        self._detectors = {}
        for name, det in detectors.items():
            self._detectors[name] = DetectorReading(
                distance=det["distance"], name=det["name"], data=det["data"]
            )

        self._choppers = MappingProxyType(self._choppers)
        self._detectors = MappingProxyType(self._detectors)

    @property
    def choppers(self) -> MappingProxyType[str, ChopperReading]:
        """The choppers in the model."""
        return self._choppers

    @property
    def detectors(self) -> MappingProxyType[str, DetectorReading]:
        """The detectors in the model."""
        return self._detectors

    @property
    def source(self) -> SourceParameters:
        """The source of neutrons."""
        return self._source

    def __iter__(self):
        return chain(self._choppers, self._detectors)

    def __getitem__(self, name: str) -> ChopperReading | DetectorReading:
        if name not in self:
            raise KeyError(f"No component with name {name} was found.")
        return self._choppers[name] if name in self._choppers else self._detectors[name]

    def plot(
        self,
        visible_rays: int = 1000,
        blocked_rays: int = 0,
        figsize: tuple[float, float] | None = None,
        ax: plt.Axes | None = None,
        cax: plt.Axes | None = None,
        cbar: bool = True,
        cmap: str = "gist_rainbow_r",
        seed: int | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> Plot:
        """
        Plot the time-distance diagram for the instrument, including the rays of
        neutrons that make it to the furthest detector.
        As plotting many lines can be slow, the number of rays to plot can be
        limited by setting ``visible_rays``.
        In addition, it is possible to also plot the rays that are blocked by
        choppers along the flight path by setting ``blocked_rays > 0``.

        Parameters
        ----------
        visible_rays:
            Maximum number of rays to plot.
        blocked_rays:
            Number of blocked rays to plot.
        figsize:
            Figure size.
        ax:
            Axes to plot on.
        cax:
            Axes to use for the colorbar.
        cbar:
            Show a colorbar for the wavelength if ``True``.
        cmap:
            Colormap to use for the wavelength colorbar.
        seed:
            Random seed for reproducibility.
        vmin:
            Minimum value for the colorbar.
        vmax:
            Maximum value for the colorbar.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        components = sorted(
            chain(self.choppers.values(), self.detectors.values()),
            key=lambda c: c.distance,
        )
        furthest_component = components[-1]
        repeats = [1] + [2] * len(components)

        wavelengths = sc.DataArray(
            data=furthest_component.data.coords["wavelength"],
            masks=furthest_component.data.masks,
        )
        wmin, wmax = wavelengths.min(), wavelengths.max()

        rng = np.random.default_rng(seed)

        for i in range(self._source.data.sizes["pulse"]):
            source_data = self.source.data["pulse", i]
            component_data = furthest_component.data["pulse", i]
            ids = np.arange(self.source.neutrons)
            # Plot visible rays
            blocked = one_mask(component_data.masks).values
            nblocked = int(blocked.sum())
            if nblocked < self.source.neutrons:
                inds = rng.choice(
                    ids[~blocked],
                    size=min(self.source.neutrons - nblocked, visible_rays),
                    replace=False,
                )

                xstart = source_data.coords["birth_time"].values[inds]
                xend = component_data.coords["toa"].values[inds]
                ystart = np.zeros_like(xstart)
                yend = np.full_like(ystart, furthest_component.distance.value)

                _add_rays(
                    ax=ax,
                    x=np.stack((xstart, xend), axis=1),
                    y=np.stack((ystart, yend), axis=1),
                    color=source_data.coords["wavelength"].values[inds],
                    cbar=cbar and (i == 0),
                    cmap=cmap,
                    vmin=wmin.value if vmin is None else vmin,
                    vmax=wmax.value if vmax is None else vmax,
                    cax=cax,
                )

            # Plot blocked rays
            inds = rng.choice(
                ids[blocked], size=min(blocked_rays, nblocked), replace=False
            )
            x = np.repeat(
                np.stack(
                    [source_data.coords["birth_time"].values[inds]]
                    + [
                        c.data.coords["toa"]["pulse", i].values[inds]
                        for c in components
                    ],
                    axis=1,
                ),
                repeats,
                axis=1,
            )
            y = np.repeat(
                np.stack(
                    [np.zeros_like(x[:, 0])]
                    + [np.full_like(x[:, 0], c.distance.value) for c in components],
                    axis=1,
                ),
                repeats,
                axis=1,
            )
            for j, c in enumerate(components):
                comp_data = c.data["pulse", i]
                m_others = comp_data.masks["blocked_by_others"].values[inds]
                x[:, 2 * j + 1][m_others] = np.nan
                y[:, 2 * j + 1][m_others] = np.nan
                if "blocked_by_me" in comp_data.masks:
                    m_me = comp_data.masks["blocked_by_me"].values[inds]
                    x[:, 2 * j + 2][m_me] = np.nan
                    y[:, 2 * j + 2][m_me] = np.nan
            _add_rays(ax=ax, x=x, y=y, color="lightgray", zorder=-1)

            # Plot pulse
            time_coord = source_data.coords["birth_time"].values
            tmin = time_coord.min()
            ax.plot([tmin, time_coord.max()], [0, 0], color="gray", lw=3)
            ax.text(tmin, 0, "Pulse", ha="left", va="top", color="gray")

        if furthest_component.toa.data.sum().value > 0:
            toa_max = furthest_component.toa.max().value
        else:
            toa_max = furthest_component.toa.data.coords["toa"].max().value
        dx = 0.05 * toa_max
        # Plot choppers
        for ch in self._choppers.values():
            x0 = ch.open_times.values
            x1 = ch.close_times.values
            x = np.empty(3 * x0.size, dtype=x0.dtype)
            x[0::3] = x0
            x[1::3] = 0.5 * (x0 + x1)
            x[2::3] = x1
            x = np.concatenate(
                ([[0]] if x[0] > 0 else [x[0:1]])
                + [x]
                + ([[toa_max + dx]] if x[-1] < toa_max else [])
            )
            y = np.full_like(x, ch.distance.value)
            y[2::3] = None
            inds = np.argsort(x)
            ax.plot(x[inds], y[inds], color="k")
            ax.text(
                toa_max, ch.distance.value, ch.name, ha="right", va="bottom", color="k"
            )

        # Plot detectors
        for det in self._detectors.values():
            ax.plot([0, toa_max], [det.distance.value] * 2, color="gray", lw=3)
            ax.text(
                0, det.distance.value, det.name, ha="left", va="bottom", color="gray"
            )

        ax.set(xlabel="Time [μs]", ylabel="Distance [m]")
        ax.set_xlim(0 - dx, toa_max + dx)
        if figsize is None:
            inches = fig.get_size_inches()
            fig.set_size_inches((min(inches[0] * self.source.pulses, 12.0), inches[1]))
        fig.tight_layout()
        return Plot(fig=fig, ax=ax)

    def __repr__(self) -> str:
        out = (
            f"Result:\n  Source: {self.source.pulses} pulses, "
            f"{self.source.neutrons} neutrons per pulse.\n  Choppers:\n"
        )
        for name, ch in self._choppers.items():
            out += f"    {name}: {ch._repr_stats()}\n"
        out += "  Detectors:\n"
        for name, det in self._detectors.items():
            out += f"    {name}: {det._repr_stats()}\n"
        return out

    def __str__(self) -> str:
        return self.__repr__()

    def to_nxevent_data(self, key: str | None = None) -> sc.DataArray:
        """
        Convert a detector reading to event data that resembles event data found in a
        NeXus file.

        Parameters
        ----------
        key:
            Name of the detector. If ``None``, all detectors are included.
        """
        start = sc.datetime("2024-01-01T12:00:00.000000")
        period = sc.reciprocal(self.source.frequency)

        keys = list(self._detectors.keys()) if key is None else [key]

        event_data = []
        for name in keys:
            raw_data = self._detectors[name].data.flatten(to="event")
            events = (
                raw_data[~raw_data.masks["blocked_by_others"]]
                .copy()
                .drop_masks("blocked_by_others")
            )
            events.coords["distance"] = sc.broadcast(
                events.coords["distance"], sizes=events.sizes
            ).copy()
            event_data.append(events)

        event_data = sc.concat(event_data, dim=event_data[0].dim)
        dt = period.to(unit=event_data.coords["toa"].unit)
        event_time_zero = (dt * (event_data.coords["toa"] // dt)).to(dtype=int) + start
        event_data.coords["event_time_zero"] = event_time_zero
        event_data.coords["event_time_offset"] = event_data.coords.pop(
            "toa"
        ) % period.to(unit=dt.unit)
        out = (
            event_data.drop_coords(["tof", "speed", "birth_time", "wavelength"])
            .group("distance")
            .rename_dims(distance="detector_number")
        )
        out.coords["Ltotal"] = out.coords.pop("distance")
        return out
