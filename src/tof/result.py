# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from itertools import chain
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc
from matplotlib.collections import LineCollection

from .chopper import Chopper
from .detector import Detector
from .pulse import Pulse
from .utils import Plot


def _readonly(self, *args, **kwargs):
    raise RuntimeError("Cannot modify ReadOnlyDict")


class ReadonlyDict(dict):
    __setitem__ = _readonly
    __delitem__ = _readonly
    pop = _readonly
    popitem = _readonly
    clear = _readonly
    update = _readonly
    setdefault = _readonly


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
                wmin=self._pulse.wmin,
                wmax=self._pulse.wmax,
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
        tmin = self.pulse.tmin.to(unit='us').value
        ax.plot(
            [tmin, self.pulse.tmax.to(unit='us').value],
            [0, 0],
            color="gray",
            lw=3,
        )
        ax.text(tmin, 0, "Pulse", ha="left", va="top", color="gray")

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
