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
    """
    A class that represents a neutron instrument.
    It is defined by a list of choppers, a list of detectors, and a pulse.

    Parameters
    ----------
    choppers:
        A list of choppers.
    detectors:
        A list of detectors.
    pulse:
        A pulse of neutrons.
    """

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
        """
        Run the simulation.

        Parameters
        ----------
        npulses:
            Number of pulses to simulate.
        """
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
            comp._own_mask = ~m & initial_mask
            initial_mask = combined

    def _add_rays(self, ax, tofs, birth_times, distances, wavelengths=None):
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
            coll.set_norm(plt.Normalize(self.pulse.lmin.value, self.pulse.lmax.value))
            cbar = plt.colorbar(coll)
            cbar.ax.yaxis.set_label_coords(-0.9, 0.5)
            cbar.set_label('Wavelength (Å)')
        else:
            coll.set_color('lightgray')
        ax.add_collection(coll)

    def plot(self, max_rays: int = 1000, blocked_rays: int = 0, figsize=None) -> tuple:
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
        """
        fig, ax = plt.subplots(figsize=figsize)
        furthest_detector = max(self.detectors, key=lambda d: d.distance)

        if blocked_rays > 0:
            inv_mask = ~furthest_detector._mask
            nrays = int(inv_mask.sum())
            if nrays > blocked_rays:
                inds = np.random.choice(nrays, size=blocked_rays, replace=False)
            else:
                inds = slice(None)
            birth_times = self.pulse.birth_times[inv_mask][inds]

            components = sorted(
                chain(self.choppers, [furthest_detector]),
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
            self._add_rays(
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
            self._add_rays(
                ax=ax,
                tofs=tofs[inds],
                birth_times=birth_times,
                distances=distances,
                wavelengths=wavelengths,
            )

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
