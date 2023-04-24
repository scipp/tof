# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from itertools import chain
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc
from matplotlib.collections import LineCollection

from .chopper import Chopper
from .detector import Detector
from .pulse import Pulse


class Model:
    def __init__(
        self,
        choppers: Union[Chopper, List[Chopper]],
        detectors: Union[Detector, List[Detector]],
        pulse: Pulse,
    ):
        self.choppers = choppers
        if not isinstance(self.choppers, (list, tuple)):
            self.choppers = [self.choppers]
        self.detectors = detectors
        if not isinstance(self.detectors, (list, tuple)):
            self.detectors = [self.detectors]
        self.pulse = pulse

    def run(self, npulses: int = 1):
        # TODO: ray-trace multiple pulses
        components = sorted(
            chain(self.choppers, self.detectors),
            key=lambda c: c.distance.value,
        )

        initial_mask = sc.ones(
            sizes=self.pulse.birth_times.sizes, unit=None, dtype=bool
        )
        for comp in components:
            comp._wavelengths = self.pulse.wavelengths
            t = self.pulse.birth_times + comp.distance / self.pulse.speeds
            comp._arrival_times = t
            if isinstance(comp, Detector):
                comp._mask = initial_mask
                continue
            m = sc.zeros(sizes=t.sizes, unit=None, dtype=bool)
            to = comp.open_times
            tc = comp.close_times
            for i in range(len(to)):
                m |= (t > to[i]) & (t < tc[i])
            combined = initial_mask & m
            comp._mask = combined
            initial_mask = combined

    def _add_rays(self, ax, tofs, birth_times, distances, wavelengths=None):
        x0 = birth_times.to(unit='us').values.reshape(-1, 1)
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
            cmap = plt.cm.gist_rainbow_r.copy()
            cmap.set_bad('lightgray')
            colors = (
                (wavelengths - self.pulse.lmin) / (self.pulse.lmax - self.pulse.lmin)
            ).values
            coll.set_cmap(cmap)
            coll.set_array(colors)
        ax.add_collection(coll)

    def plot(self, max_rays: int = 1000, blocked_rays: int = 0) -> tuple:
        fig, ax = plt.subplots()
        furthest_detector = max(self.detectors, key=lambda d: d.distance)

        # if show_blocked_rays:
        #     tofs = furthest_detector._arrival_times.to(unit='us').values
        #     selection_mask = slice(None)
        # else:

        # Blocked rays
        if blocked_rays > 0:
            inv_mask = ~furthest_detector._mask
            nrays = inv_mask.sum()
            # tofs = furthest_detector._arrival_times[inv_mask].to(unit='us')
            if nrays > blocked_rays:
                inds = np.random.choice(nrays, size=blocked_rays, replace=False)
            else:
                inds = slice(None)
            birth_times = self.pulse.birth_times[inv_mask][inds]

            # Compute arrival times for all rays at all choppers and the furthest
            # detector
            components = sorted(
                chain(self.choppers, [furthest_detector]),
                key=lambda c: c.distance.value,
            )
            tofs = sc.concat(
                [comp._arrival_times for comp in components], dim='component'
            )
            distances = sc.concat(
                [comp.distance for comp in components], dim='component'
            )
            masks = sc.concat([comp._mask for comp in components], dim='component')

            # # chopper_masks = [c._mask[inv_mask][inds] for c in self.choppers]
            # chopper_masks = sc.concat(
            #     [c._mask[inv_mask][inds] for c in self.choppers], dim='chopper'
            # )
            self._add_rays(
                ax=ax,
                tofs=tofs[inds],
                birth_times=birth_times,
                distances=distances,
                wavelengths=wavelengths,
            )

        # Normal rays
        tofs = furthest_detector.tofs.coords['tof']
        if (max_rays is not None) and (len(tofs) > max_rays):
            inds = np.random.choice(len(tofs), size=max_rays, replace=False)
        else:
            inds = slice(None)
        birth_times = self.pulse.birth_times[furthest_detector._mask][inds]
        wavelengths = self.pulse.wavelengths[furthest_detector._mask][inds]
        distances = furthest_detector.distance.broadcast(sizes=birth_times.sizes)
        self._add_rays(
            ax=ax,
            tofs=tofs[inds],
            birth_times=birth_times,
            distances=distances,
            wavelengths=wavelengths,
        )

        # # Blocked rays
        # tofs = furthest_detector.tofs.coords['tof'].values
        # if (max_rays is not None) and (len(tofs) > max_rays):
        #     inds = np.random.choice(len(tofs), size=max_rays, replace=False)
        # else:
        #     inds = slice(None)
        # birth_times = self.pulse.birth_times[selection_mask][inds]
        # wavelengths = self.pulse.birth_times[selection_mask][inds]
        # self._add_rays(ax=ax, tofs=tofs, birth_times=birth_times, wavelengths=wavelengths)

        # selection_mask = furthest_detector._mask

        # tof_max = tofs.max()
        # if (max_rays is not None) and (len(tofs) > max_rays):
        #     inds = np.random.choice(len(tofs), size=max_rays, replace=False)
        # else:
        #     inds = slice(None)

        # # Plot rays
        # x0 = (
        #     self.pulse.birth_times[selection_mask][inds]
        #     .to(unit='us')
        #     .values.reshape(-1, 1)
        # )
        # x1 = tofs[inds].reshape(-1, 1)
        # nrays = x0.size
        # y0 = np.zeros(nrays).reshape(-1, 1)
        # y1 = np.full(nrays, furthest_detector.distance.value).reshape(-1, 1)
        # segments = np.concatenate(
        #     (
        #         np.concatenate((x0, y0), axis=1).reshape(-1, 1, 2),
        #         np.concatenate((x1, y1), axis=1).reshape(-1, 1, 2),
        #     ),
        #     axis=1,
        # )
        # cmap = plt.cm.gist_rainbow_r.copy()
        # cmap.set_bad('lightgray')
        # coll = LineCollection(segments, cmap=cmap)
        # colors = (
        #     (self.pulse.wavelengths - self.pulse.lmin)
        #     / (self.pulse.lmax - self.pulse.lmin)
        # ).values
        # colors[~furthest_detector._mask.values] = np.nan
        # print(colors.max(), colors.dtype)
        # coll.set_array(
        #     colors[
        #         selection_mask
        #         if isinstance(selection_mask, slice)
        #         else selection_mask.values
        #     ][inds]
        # )
        # ax.add_collection(coll)

        tof_max = tofs.max().value

        # Plot choppers
        for ch in self.choppers:
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
        for det in self.detectors:
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
        return fig, ax

    def __repr__(self) -> str:
        return (
            f"Model(choppers={self.choppers},\n      "
            f"detectors={self.detectors},\n      "
            f"pulse={self.pulse},\n      "
            f"neutrons={len(self.pulse.birth_times)})"
        )
