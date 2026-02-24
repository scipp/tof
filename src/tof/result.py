# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

from itertools import chain
from types import MappingProxyType

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc
from matplotlib.collections import LineCollection

from .chopper import ChopperReading
from .component import ComponentReading
from .detector import DetectorReading
from .source import SourceReading
from .utils import Plot, one_mask


def _get_rays(
    components: list[ComponentReading], pulse: int, inds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x = []
    y = []
    c = []
    data = components[0].data["pulse", pulse]
    # TODO optimize: should not index multiple times
    xstart = data.coords["toa"].values[inds]
    ystart = np.full_like(xstart, components[0].distance.value)
    color = data.coords["wavelength"].values[inds]
    for comp in components[1:]:
        xend = comp.data["pulse", pulse].coords["toa"].values[inds]
        yend = np.full_like(xend, comp.distance.value)
        x.append([xstart, xend])
        y.append([ystart, yend])
        c.append(color)
        xstart, ystart = xend, yend
        color = comp.data["pulse", pulse].coords["wavelength"].values[inds]

    return np.array(x), np.array(y), np.array(c)


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
    x, y = (np.array(a).transpose((0, 2, 1)).reshape((-1, 2)) for a in (x, y))
    coll = LineCollection(np.stack((x, y), axis=2), zorder=zorder)
    if isinstance(color, str):
        coll.set_color(color)
    else:
        coll.set_cmap(plt.colormaps[cmap])
        coll.set_array(color.ravel())
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
    results:
        The state of neutrons at each component in the model.
    """

    def __init__(self, source: SourceReading, readings: dict[str, dict]):
        self._source = source
        self._components = MappingProxyType(readings)

    @property
    def choppers(self) -> MappingProxyType[str, ChopperReading]:
        """The choppers in the model."""
        return MappingProxyType(
            {
                key: comp
                for key, comp in self._components.items()
                if comp.kind == "chopper"
            }
        )

    @property
    def detectors(self) -> MappingProxyType[str, DetectorReading]:
        """The detectors in the model."""
        return MappingProxyType(
            {
                key: comp
                for key, comp in self._components.items()
                if comp.kind == "detector"
            }
        )

    @property
    def source(self) -> SourceReading:
        """The source of neutrons."""
        return self._source

    def __iter__(self):
        return iter(self._components)

    def __getitem__(self, name: str) -> ComponentReading:
        return self._components[name]

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
        title: str | None = None,
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
            chain((self.source,), self._components.values()), key=lambda c: c.distance
        )
        furthest_component = components[-1]

        wavelengths = sc.DataArray(
            data=furthest_component.data.coords["wavelength"],
            masks=furthest_component.data.masks,
        )
        wmin, wmax = wavelengths.min(), wavelengths.max()

        rng = np.random.default_rng(seed)

        for i in range(self._source.data.sizes["pulse"]):
            component_data = furthest_component.data["pulse", i]
            ids = component_data.coords["id"].values

            # Plot visible rays
            blocked = one_mask(component_data.masks).values
            nblocked = int(blocked.sum())
            if nblocked < self.source.neutrons:
                inds = rng.choice(
                    ids[~blocked],
                    size=min(self.source.neutrons - nblocked, visible_rays),
                    replace=False,
                )
                x, y, c = _get_rays(components, pulse=i, inds=inds)
                _add_rays(
                    ax=ax,
                    x=x,
                    y=y,
                    color=c,
                    cbar=cbar and (i == 0),
                    cmap=cmap,
                    vmin=wmin.value if vmin is None else vmin,
                    vmax=wmax.value if vmax is None else vmax,
                    cax=cax,
                )

            # # Plot blocked rays
            # inds = rng.choice(
            #     ids[blocked], size=min(blocked_rays, nblocked), replace=False
            # )
            # x, y = _get_rays(components, pulse=i, inds=inds)
            # blocked_by_others = np.stack(
            #     [
            #         comp.data["pulse", i].masks["blocked_by_others"].values[inds]
            #         for comp in components
            #     ],
            #     axis=1,
            # )
            # x[blocked_by_others] = np.nan
            # y[blocked_by_others] = np.nan
            # _add_rays(ax=ax, x=x, y=y, color="lightgray", zorder=-1)

            # Plot pulse
            self.source.plot_on_time_distance_diagram(ax, pulse=i)
        if furthest_component.toa.data.sum().value > 0:
            toa_max = furthest_component.toa.max().value
        else:
            toa_max = furthest_component.toa.data.coords["toa"].max().value

        # Plot components
        for comp in self._components.values():
            comp.plot_on_time_distance_diagram(ax=ax, tmax=toa_max)

        dx = 0.05 * toa_max
        ax.set(xlabel="Time [μs]", ylabel="Distance [m]")
        ax.set_xlim(0 - dx, toa_max + dx)
        if figsize is None:
            inches = fig.get_size_inches()
            fig.set_size_inches((min(inches[0] * self.source.pulses, 12.0), inches[1]))
        fig.tight_layout()
        if title is not None:
            ax.set_title(title)
        return Plot(fig=fig, ax=ax)

    def __repr__(self) -> str:
        out = (
            f"Result:\n  Source: {self.source.pulses} pulses, "
            f"{self.source.neutrons} neutrons per pulse.\n"
        )
        groups = {}
        for comp in self._components.values():
            if comp.kind not in groups:
                groups[comp.kind] = []
            groups[comp.kind].append(comp)

        for group, comps in groups.items():
            out += f"  {group.capitalize()}s:\n"
            for comp in sorted(comps, key=lambda c: c.distance):
                out += f"    {comp.name}: {comp._repr_stats()}\n"

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

    @property
    def data(self) -> sc.DataGroup:
        """
        Get the data for the source, choppers, and detectors, as a DataGroup.
        The components are sorted by distance.
        """
        out = {"source": self.source.data}
        components = sorted(
            chain(self.choppers.values(), self.detectors.values()),
            key=lambda c: c.distance.value,
        )
        for comp in components:
            out[comp.name] = comp.data
        return sc.DataGroup(out)
