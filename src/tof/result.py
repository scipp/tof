# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from itertools import chain
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc
from matplotlib.collections import LineCollection


from .chopper import Chopper
from .detector import Detector
from .pulse import Pulse
from .utils import Plot


# class Data:
#     """
#     A data object contains the data (visible or blocked) for a component data
#     (time-of-flight or wavelengths).

#     Parameters
#     ----------
#     data:
#         The data to hold.
#     dim:
#         The dimension label of the data.
#     """

#     def __init__(self, data: sc.DataArray, dim: str):
#         self._data = data
#         self._dim = dim

#     @property
#     def data(self) -> sc.DataArray:
#         """
#         The underlying data.
#         """
#         return self._data

#     @property
#     def shape(self) -> Tuple[int]:
#         """
#         The shape of the data.
#         """
#         return self._data.shape

#     @property
#     def sizes(self) -> Dict[str, int]:
#         """
#         The sizes of the data.
#         """
#         return self._data.sizes

#     def __len__(self) -> int:
#         """
#         The length of the data.
#         """
#         return len(self._data)

#     def plot(self, bins: Union[int, sc.Variable] = 300, **kwargs):
#         """
#         Plot the neutrons that reach the component as a histogram.

#         Parameters
#         ----------
#         bins:
#             The bins to use for histogramming the neutrons.
#         """
#         return self._data.hist({self._dim: bins}).plot(**kwargs)

#     def __repr__(self) -> str:
#         return f"Data(data={self._data})"


# class ComponentData:
#     """
#     A component data object contains the data (time-of-flight or wavelengths) for a
#     component.

#     Parameters
#     ----------
#     data:
#         The data to hold.
#     mask:
#         A mask to apply to the data which will select the neutrons that pass through
#         the component.
#     dim:
#         The dimension label of the data.
#     blocking:
#         A mask to apply to the data which will select the neutrons that are blocked by
#         the component (this only defined in the case of a :class:`Chopper` component
#         since a :class:`Detector` does not block any neutrons).
#     """

#     def __init__(
#         self,
#         data: sc.Variable,
#         mask: sc.Variable,
#         dim: str,
#         blocking: Optional[sc.Variable] = None,
#     ):
#         self._data = data
#         self._mask = mask
#         self._blocking = blocking
#         self._dim = dim

#     @property
#     def visible(self) -> Data:
#         """
#         The data for the neutrons that pass through the component.
#         """
#         a = self._data[self._mask]
#         return Data(
#             data=sc.DataArray(
#                 data=sc.ones(sizes=a.sizes, unit='counts'), coords={self._dim: a}
#             ),
#             dim=self._dim,
#         )

#     @property
#     def blocked(self) -> Data:
#         """
#         The data for the neutrons that are blocked by the component.
#         """
#         if self._blocking is None:
#             return
#         a = self._data[self._blocking]
#         return Data(
#             data=sc.DataArray(
#                 data=sc.ones(sizes=a.sizes, unit='counts'), coords={self._dim: a}
#             ),
#             dim=self._dim,
#         )

#     @property
#     def data(self) -> sc.DataGroup:
#         """
#         The neutrons that reach the component, split up into those that are blocked by
#         the component and those that are not.
#         """
#         out = {'visible': self.visible.data}
#         if self._blocking is not None:
#             out['blocked'] = self.blocked.data
#         return sc.DataGroup(out)

#     def __repr__(self) -> str:
#         return (
#             f"ComponentData(visible={sc.sum(self._mask).value}, "
#             f"blocked={sc.sum(~self._mask).value}, dim={self._dim})"
#         )

#     def plot(self, bins: Union[int, sc.Variable] = 300, **kwargs):
#         """
#         Plot the data for the neutrons that reach the component, split up into those
#         that are blocked by the component and those that are not.

#         Parameters
#         ----------
#         bins:
#             The bins to use for histogramming the neutrons.
#         """
#         if self._blocking is None:
#             return self.visible.plot(bins=bins, **kwargs)
#         visible = self.visible.data
#         blocked = self.blocked.data
#         if isinstance(bins, int):
#             bins = sc.linspace(
#                 dim=self._dim,
#                 start=min(
#                     visible.coords[self._dim].min(), blocked.coords[self._dim].min()
#                 ).value,
#                 stop=max(
#                     visible.coords[self._dim].max(), blocked.coords[self._dim].max()
#                 ).value,
#                 num=bins,
#                 unit=visible.coords[self._dim].unit,
#             )
#         return pp.plot(
#             {
#                 'visible': visible.hist({self._dim: bins}),
#                 'blocked': blocked.hist({self._dim: bins}),
#             },
#             **{**{'color': {'blocked': 'gray'}}, **kwargs},
#         )


# class Component:
#     """
#     A component is placed in the beam path. After the model is run, the component
#     will have a record of the arrival times and wavelengths of the neutrons that
#     passed through it.
#     """

#     def __init__(self):
#         # for key, value in kwargs.items():
#         #     setattr(self, key, value)
#         self._arrival_times = None
#         self._wavelengths = None
#         self._mask = None
#         self._own_mask = None

#     @property
#     def tofs(self) -> ComponentData:
#         """
#         The arrival times of the neutrons that passed through the component.
#         """
#         return ComponentData(
#             data=self._arrival_times.to(unit='us'),
#             mask=self._mask,
#             blocking=self._own_mask,
#             dim='tof',
#         )

#     @property
#     def wavelengths(self) -> ComponentData:
#         """
#         The wavelengths of the neutrons that passed through the component.
#         """
#         return ComponentData(
#             data=self._wavelengths,
#             mask=self._mask,
#             blocking=self._own_mask,
#             dim='wavelength',
#         )

#     def plot(self, bins: int = 300) -> tuple:
#         """
#         Plot both the tof and wavelength data side by side.

#         Parameters
#         ----------
#         bins:
#             Number of bins to use for histogramming the neutrons.
#         """
#         fig, ax = plt.subplots(1, 2)
#         self.tofs.plot(bins=bins, ax=ax[0])
#         self.wavelengths.plot(bins=bins, ax=ax[1])
#         size = fig.get_size_inches()
#         fig.set_size_inches(size[0] * 2, size[1])
#         fig.tight_layout()
#         return Plot(fig=fig, ax=ax)

#     def __repr__(self) -> str:
#         return f"Component(tofs={self.tofs}, wavelengths={self.wavelengths})"


def _add_rays(
    ax: plt.Axes,
    tofs: sc.Variable,
    birth_times: sc.Variable,
    distances: sc.Variable,
    cbar: bool = True,
    wavelengths: Optional[sc.Variable] = None,
    wav_min: Optional[sc.Variable] = None,
    wav_max: Optional[sc.Variable] = None,
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
        coll.set_norm(plt.Normalize(wav_min.value, wav_max.value))
        if cbar:
            cb = plt.colorbar(coll)
            cb.ax.yaxis.set_label_coords(-0.9, 0.5)
            cb.set_label('Wavelength (Å)')
    else:
        coll.set_color('lightgray')
    ax.add_collection(coll)


class Result:
    """Result of a simulation."""

    def __init__(
        self, pulse: Pulse, choppers: Dict[str, Chopper], detectors: Dict[str, Detector]
    ):
        self._pulse = pulse
        self._choppers = choppers
        self._detectors = detectors

    @property
    def choppers(self) -> Dict[str, Chopper]:
        """The choppers in the model."""
        return self._choppers

    @property
    def detectors(self) -> Dict[str, Detector]:
        """The detectors in the model."""
        return self._detectors

    @property
    def pulse(self) -> Pulse:
        """The pulse of neutrons."""
        return self._pulse

    def __getattr__(self, name):
        if name in self._choppers:
            return self._choppers[name]
        elif name in self._detectors:
            return self._detectors[name]
        else:
            raise AttributeError(f"Result has no attribute {name}.")

    def plot(
        self,
        max_rays: int = 1000,
        blocked_rays: int = 0,
        figsize=None,
        ax=None,
        cbar=True,
    ) -> tuple:
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
            inv_mask = ~furthest_detector._mask
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
                [comp._arrival_times[inv_mask][inds] for comp in components], dim=dim
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
                + [comp._mask[inv_mask][inds] for comp in components],
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
            tofs = furthest_detector.tofs.visible.data.coords['tof']
            if (max_rays is not None) and (len(tofs) > max_rays):
                inds = np.random.choice(len(tofs), size=max_rays, replace=False)
            else:
                inds = slice(None)
            birth_times = self.pulse.birth_times[furthest_detector._mask][inds]
            wavelengths = self.pulse.wavelengths[furthest_detector._mask][inds]
            distances = furthest_detector.distance.broadcast(sizes=birth_times.sizes)
            _add_rays(
                ax=ax,
                tofs=tofs[inds],
                birth_times=birth_times,
                distances=distances,
                cbar=cbar,
                wavelengths=wavelengths,
                wav_min=self._pulse.wav_min,
                wav_max=self._pulse.wav_max,
            )

        tof_max = tofs.max().value
        # Plot choppers
        for ch in self._choppers.values():
            x0 = ch.open_times.to(unit='us').values
            x1 = ch.close_times.to(unit='us').values
            x = np.empty(3 * x0.size, dtype=x0.dtype)
            x[0::3] = x0
            x[1::3] = 0.5 * (x0 + x1)
            x[2::3] = x1
            x = np.concatenate([[0], x, [tof_max]])
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

        # Plot pulse
        t_start = self.pulse.t_start.to(unit='us').value
        ax.plot(
            [t_start, self.pulse.t_end.to(unit='us').value],
            [0, 0],
            color="gray",
            lw=3,
        )
        ax.text(t_start, 0, "Pulse", ha="left", va="top", color="gray")

        ax.set_xlabel("Time-of-flight (us)")
        ax.set_ylabel("Distance (m)")
        fig.tight_layout()
        return Plot(fig=fig, ax=ax)

    def __repr__(self) -> str:
        out = f"Pulse: {self._pulse.neutrons} neutrons.\nChoppers:\n"
        for name, ch in self._choppers.items():
            tofs = ch.tofs
            out += (
                f"  {name}: visible={len(tofs.visible.data)}, "
                f"blocked={len(tofs.blocked.data)}\n"
            )
        out += "Detectors:\n"
        for name, det in self._detectors.items():
            out += f"  {name}: visible={len(det.tofs.visible.data)}\n"
        return out
