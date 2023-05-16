# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from itertools import chain
from types import MappingProxyType
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc
from matplotlib.collections import LineCollection

from .chopper import Chopper, ChopperReading
from .component import ComponentData, Data
from .detector import Detector, DetectorReading
from .source import Source, SourceParameters
from .utils import Plot, merge_masks


# def _make_data(array, dim):
#     return Data(
#         data=sc.DataArray(
#             data=sc.ones(sizes=array.sizes, unit='counts'), coords={dim: array}
#         ),
#         dim=dim,
#     )


def _make_component_data(component, dim, is_chopper=False):
    visible = {}
    blocked = {} if is_chopper else None
    for name, da in sc.collapse(component, keep='event').items():
        msk = ~merge_masks(da.masks)
        vsel = da[msk]
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
    x0 = birth_times.to(unit='us', copy=False).values.reshape(-1, 1)
    x1 = tofs.to(unit='us', copy=False).values.reshape(-1, 1)
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
            cb = plt.colorbar(coll)
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
        # fields = {
        #     'tofs': ('arrival_times', 'tof'),
        #     'wavelengths': ('wavelengths', 'wavelength'),
        #     'birth_times': ('birth_times', 'time'),
        #     'speeds': ('speeds', 'speed'),
        # }
        # fields = ['tof', 'wavelength', 'time', 'speed']
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
                # data=chopper['data'],
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
                # data=det['data']
                **{
                    key: _make_component_data(det, dim=dim)
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

    def plot(
        self,
        max_rays: int = 1000,
        blocked_rays: int = 0,
        figsize=None,
        ax=None,
        cbar=True,
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

        if blocked_rays > 0:
            inv_mask = ~self._masks[furthest_detector.name]
            nrays = int(inv_mask.sum())
            if nrays > blocked_rays:
                inds = np.random.choice(nrays, size=blocked_rays, replace=False)
            else:
                inds = slice(None)
            birth_times = self.pulse.birth_times[inv_mask][inds]

            components = sorted(
                chain(self._choppers.values(), [furthest_detector]),
                key=lambda c: c.distance.value,
            )
            dim = 'component'
            tofs = sc.concat(
                [self._arrival_times[comp.name][inv_mask][inds] for comp in components],
                dim=dim,
            )
            distances = sc.concat(
                [
                    comp.distance.broadcast(sizes=birth_times.sizes)
                    for comp in components
                ],
                dim=dim,
            )
            masks = sc.concat(
                [sc.ones(sizes=birth_times.sizes, dtype=bool)]
                + [self._masks[comp.name][inv_mask][inds] for comp in components],
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

        # Normal rays
        if max_rays > 0:
            for i, da in enumerate(
                sc.collapse(furthest_detector.data, keep='event').values()
            ):
                # tofs = furthest_detector.tofs.visible.data.coords['tof']
                # da = furthest_detector.data['pulse', i]
                visible = da[~da.masks['blocked_by_others']]
                tofs = visible.coords['tof']
                if (max_rays is not None) and (len(tofs) > max_rays):
                    inds = np.random.choice(len(tofs), size=max_rays, replace=False)
                else:
                    inds = slice(None)
                birth_times = visible.coords['time'][inds]
                wavelengths = visible.coords['wavelength'][inds]
                distances = furthest_detector.distance.broadcast(
                    sizes=birth_times.sizes
                )
                _add_rays(
                    ax=ax,
                    tofs=tofs[inds],
                    birth_times=birth_times,
                    distances=distances,
                    cbar=cbar and (i == 0),
                    wavelengths=wavelengths,
                    wmin=self._source.wmin,
                    wmax=self._source.wmax,
                )

                # Plot pulse
                tmin = visible.coords['time'].min().to(unit='us').value
                ax.plot(
                    [tmin, visible.coords['time'].max().to(unit='us').value],
                    [0, 0],
                    color="gray",
                    lw=3,
                )
                ax.text(tmin, 0, "Pulse", ha="left", va="top", color="gray")

        tof_max = tofs.max().value
        dx = 0.05 * tof_max
        # Plot choppers
        for ch in self._choppers.values():
            x0 = ch.open_times.to(unit='us').values
            x1 = ch.close_times.to(unit='us').values
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

        # # Plot pulse
        # tmin = self.pulse.tmin.to(unit='us').value
        # ax.plot(
        #     [tmin, self.pulse.tmax.to(unit='us').value],
        #     [0, 0],
        #     color="gray",
        #     lw=3,
        # )
        # ax.text(tmin, 0, "Pulse", ha="left", va="top", color="gray")

        ax.set_xlabel("Time-of-flight (us)")
        ax.set_ylabel("Distance (m)")
        ax.set_xlim(0 - dx, tof_max + dx)
        if figsize is None:
            inches = fig.get_size_inches()
            fig.set_size_inches(
                (
                    max(inches[0] * furthest_detector.data.sizes['pulse'], 12.0),
                    inches[1],
                )
            )
        fig.tight_layout()
        return Plot(fig=fig, ax=ax)

    def __repr__(self) -> str:
        source_sizes = self._source.data.sizes
        out = (
            f"Result:\n  Source: {source_sizes['pulse']} pulses, "
            f"{source_sizes['event']} neutrons per pulse.\n  Choppers:\n"
        )
        for name, ch in self._choppers.items():
            tofs = ch.tofs
            # vis = [str(len(tofs.visible[0])), str(len(tofs.visible[-1]))]
            # blk = [str(len(tofs.blocked[0])), str(len(tofs.blocked[-1]))]
            # if source_sizes['pulse'] > 2:
            #     vis.insert(1, '...')
            #     blk.insert(1, '...')
            out += f"    {name}: {ch.tofs._repr_string_body()}\n"
        out += "  Detectors:\n"
        for name, det in self._detectors.items():
            tofs = det.tofs
            vis = [str(len(tofs.visible[0])), str(len(tofs.visible[-1]))]
            if source_sizes['pulse'] > 2:
                vis.insert(1, '...')
            out += f"    {name}: visible=[{', '.join(vis)}]\n"
        return out

    def __str__(self) -> str:
        return self.__repr__()
