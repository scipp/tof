# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from functools import reduce
from itertools import chain
from types import MappingProxyType
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc
from matplotlib.collections import LineCollection

from .chopper import Chopper, ChopperReading
from .component import ComponentData, Data
from .detector import Detector, DetectorReading
from .source import Source, SourceParameters
from .utils import Plot


def _make_component_data(component, dim, is_chopper=False):
    visible = {}
    blocked = {} if is_chopper else None
    keep_dim = (set(component.dims) - {'pulse'}).pop()
    for name, da in sc.collapse(component, keep=keep_dim).items():
        one_mask = ~reduce(lambda a, b: a | b, da.masks.values())
        vsel = da[one_mask]
        visible[name] = sc.DataArray(data=vsel.data, coords={dim: vsel.coords[dim]})
        if is_chopper:
            bsel = da[da.masks['blocked_by_me']]
            blocked[name] = sc.DataArray(data=bsel.data, coords={dim: bsel.coords[dim]})
    return ComponentData(
        visible=Data(data=sc.DataGroup(visible), dim=dim),
        blocked=Data(data=sc.DataGroup(blocked), dim=dim) if is_chopper else None,
    )


def _add_rays(
    ax: plt.Axes,
    tofs: sc.Variable,
    birth_times: sc.Variable,
    distances: sc.Variable,
    cbar: bool = True,
    wavelengths: Optional[sc.Variable] = None,
    wmin: Optional[sc.Variable] = None,
    wmax: Optional[sc.Variable] = None,
):
    x0 = birth_times.values.reshape(-1, 1)
    x1 = tofs.values.reshape(-1, 1)
    y0 = np.zeros(x0.size).reshape(-1, 1)
    y1 = distances.values.reshape(-1, 1)
    segments = np.concatenate(
        (
            np.concatenate((x0, y0), axis=1).reshape(-1, 1, 2),
            np.concatenate((x1, y1), axis=1).reshape(-1, 1, 2),
        ),
        axis=1,
    )
    coll = LineCollection(segments)
    if wavelengths is not None:
        coll.set_cmap(plt.cm.gist_rainbow_r)
        coll.set_array(wavelengths.values)
        coll.set_norm(plt.Normalize(wmin.value, wmax.value))
        if cbar:
            cb = plt.colorbar(coll, ax=ax)
            cb.ax.yaxis.set_label_coords(-0.9, 0.5)
            cb.set_label('Wavelength (Å)')
    else:
        coll.set_color('lightgray')
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
        choppers: Dict[str, Chopper],
        detectors: Dict[str, Detector],
    ):
        self._source = source.as_readonly()
        self._masks = {}
        self._arrival_times = {}
        self._choppers = {}
        fields = {
            'tofs': 'tof',
            'wavelengths': 'wavelength',
            'birth_times': 'time',
            'speeds': 'speed',
        }
        for name, chopper in choppers.items():
            self._masks[name] = chopper['visible_mask']
            self._arrival_times[name] = chopper['data'].coords['tof']
            self._choppers[name] = ChopperReading(
                distance=chopper['distance'],
                name=chopper['name'],
                frequency=chopper['frequency'],
                open=chopper['open'],
                close=chopper['close'],
                phase=chopper['phase'],
                open_times=chopper['open_times'],
                close_times=chopper['close_times'],
                data=chopper['data'],
                **{
                    key: _make_component_data(chopper['data'], dim=dim, is_chopper=True)
                    for key, dim in fields.items()
                },
            )

        self._detectors = {}
        for name, det in detectors.items():
            self._masks[name] = det['visible_mask']
            self._arrival_times[name] = det['data'].coords['tof']
            self._detectors[name] = DetectorReading(
                distance=det['distance'],
                name=det['name'],
                data=det['data'],
                **{
                    key: _make_component_data(det['data'], dim=dim)
                    for key, dim in fields.items()
                },
            )

        self._choppers = MappingProxyType(self._choppers)
        self._detectors = MappingProxyType(self._detectors)
        self._masks = MappingProxyType(self._masks)
        self._arrival_times = MappingProxyType(self._arrival_times)

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

    def __getitem__(self, name: str) -> Union[ChopperReading, DetectorReading]:
        if name not in self:
            raise KeyError(f"No component with name {name} was found.")
        return self._choppers[name] if name in self._choppers else self._detectors[name]

    def _plot_visible_rays(
        self,
        max_rays: int,
        pulse_index: int,
        furthest_detector: DetectorReading,
        ax: plt.Axes,
        cbar: bool,
        wmin: sc.Variable,
        wmax: sc.Variable,
    ):
        if max_rays <= 0:
            return
        da = furthest_detector.data['pulse', pulse_index]
        visible = da[~da.masks['blocked_by_others']]
        tofs = visible.coords['tof']
        if (max_rays is not None) and (len(tofs) > max_rays):
            inds = np.random.choice(len(tofs), size=max_rays, replace=False)
        else:
            inds = slice(None)
        birth_times = visible.coords['time'][inds]
        wavelengths = visible.coords['wavelength'][inds]
        distances = furthest_detector.distance.broadcast(sizes=birth_times.sizes)
        _add_rays(
            ax=ax,
            tofs=tofs[inds],
            birth_times=birth_times,
            distances=distances,
            cbar=cbar,
            wavelengths=wavelengths,
            wmin=wmin,
            wmax=wmax,
        )

    def _plot_blocked_rays(
        self,
        blocked_rays: int,
        pulse_index: int,
        furthest_detector: DetectorReading,
        ax: plt.Axes,
    ):
        if blocked_rays <= 0:
            return
        slc = ('pulse', pulse_index)
        inv_mask = ~self._masks[furthest_detector.name][slc]
        nrays = int(inv_mask.sum())
        if nrays > blocked_rays:
            inds = np.random.choice(nrays, size=blocked_rays, replace=False)
        else:
            inds = slice(None)
        birth_times = self._source.data.coords['time'][slc][inv_mask][inds]

        components = sorted(
            chain(self._choppers.values(), [furthest_detector]),
            key=lambda c: c.distance.value,
        )
        dim = 'component'
        tofs = sc.concat(
            [
                self._arrival_times[comp.name][slc][inv_mask][inds]
                for comp in components
            ],
            dim=dim,
        )
        distances = sc.concat(
            [comp.distance.broadcast(sizes=birth_times.sizes) for comp in components],
            dim=dim,
        )
        masks = sc.concat(
            [sc.ones(sizes=birth_times.sizes, dtype=bool)]
            + [self._masks[comp.name][slc][inv_mask][inds] for comp in components],
            dim=dim,
        )

        diff = sc.abs(masks[dim, 1:].to(dtype=int) - masks[dim, :-1].to(dtype=int))
        diff.unit = ''
        _add_rays(
            ax=ax,
            tofs=(tofs * diff).max(dim=dim),
            birth_times=birth_times,
            distances=(distances * diff).max(dim=dim),
        )

    def _plot_pulse(self, pulse_index: int, ax: plt.Axes):
        time_coord = self.source.data.coords['time']['pulse', pulse_index]
        tmin = time_coord.min().value
        ax.plot(
            [tmin, time_coord.max().value],
            [0, 0],
            color="gray",
            lw=3,
        )
        ax.text(tmin, 0, "Pulse", ha="left", va="top", color="gray")

    def plot(
        self,
        max_rays: int = 1000,
        blocked_rays: int = 0,
        figsize: Optional[Tuple[float, float]] = None,
        ax: Optional[plt.Axes] = None,
        cbar: bool = True,
    ) -> Plot:
        """
        Plot the time-distance diagram for the instrument, including the rays of
        neutrons that make it to the furthest detector.
        As plotting many lines can be slow, the number of rays to plot can be
        limited by setting ``max_rays``.
        In addition, it is possible to also plot the rays that are blocked by
        choppers along the flight path by setting ``blocked_rays > 0``.

        Parameters
        ----------
        max_rays:
            Maximum number of rays to plot.
        blocked_rays:
            Number of blocked rays to plot.
        figsize:
            Figure size.
        ax:
            Axes to plot on.
        cbar:
            Show a colorbar for the wavelength if ``True``.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        furthest_detector = max(self._detectors.values(), key=lambda d: d.distance)
        wavelengths = sc.DataArray(
            data=furthest_detector.data.coords['wavelength'],
            masks=furthest_detector.data.masks,
        )
        for i in range(self._source.data.sizes['pulse']):
            self._plot_blocked_rays(
                blocked_rays=blocked_rays,
                pulse_index=i,
                furthest_detector=furthest_detector,
                ax=ax,
            )
            self._plot_visible_rays(
                max_rays=max_rays,
                pulse_index=i,
                furthest_detector=furthest_detector,
                ax=ax,
                cbar=cbar and (i == 0),
                wmin=wavelengths.min(),
                wmax=wavelengths.max(),
            )
            self._plot_pulse(pulse_index=i, ax=ax)

        det_data = furthest_detector.tofs.visible.data
        if len(det_data) == 1:
            tof_max = det_data['pulse:0'].coords['tof'].max().value
        else:
            tof_max = reduce(
                lambda a, b: max(a.max(), b.max()),
                [da.coords['tof'] for da in det_data.values()],
            ).value
        dx = 0.05 * tof_max
        # Plot choppers
        for ch in self._choppers.values():
            x0 = ch.open_times.values
            x1 = ch.close_times.values
            x = np.empty(3 * x0.size, dtype=x0.dtype)
            x[0::3] = x0
            x[1::3] = 0.5 * (x0 + x1)
            x[2::3] = x1
            x = np.concatenate([[0], x] + ([[tof_max + dx]] if x[-1] < tof_max else []))
            y = np.full_like(x, ch.distance.value)
            y[2::3] = None
            ax.plot(x, y, color="k")
            ax.text(
                tof_max, ch.distance.value, ch.name, ha="right", va="bottom", color="k"
            )

        # Plot detectors
        for det in self._detectors.values():
            ax.plot([0, tof_max], [det.distance.value] * 2, color="gray", lw=3)
            ax.text(
                0, det.distance.value, det.name, ha="left", va="bottom", color="gray"
            )

        ax.set_xlabel("Time-of-flight (us)")
        ax.set_ylabel("Distance (m)")
        ax.set_xlim(0 - dx, tof_max + dx)
        if figsize is None:
            inches = fig.get_size_inches()
            fig.set_size_inches(
                (
                    min(inches[0] * furthest_detector.data.sizes['pulse'], 12.0),
                    inches[1],
                )
            )
        fig.tight_layout()
        return Plot(fig=fig, ax=ax)

    def __repr__(self) -> str:
        source_sizes = self._source.data.sizes
        other_dim = (set(source_sizes) - {'pulse'}).pop()
        out = (
            f"Result:\n  Source: {source_sizes['pulse']} pulses, "
            f"{source_sizes[other_dim]} neutrons per pulse.\n  Choppers:\n"
        )
        for name, ch in self._choppers.items():
            out += f"    {name}: {ch.tofs._repr_string_body()}\n"
        out += "  Detectors:\n"
        for name, det in self._detectors.items():
            out += f"    {name}: {det.tofs._repr_string_body()}\n"
        return out

    def __str__(self) -> str:
        return self.__repr__()
