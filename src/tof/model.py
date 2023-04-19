from itertools import chain

import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

from .detector import Detector
from .pulse import Pulse
from .tools import Plot
from .units import s_to_us


class Model:
    def __init__(self, choppers, detectors, pulse):
        self.choppers = choppers
        if not isinstance(self.choppers, dict):
            self.choppers = {self.choppers.name: self.choppers}
        self.detectors = detectors
        if not isinstance(self.detectors, dict):
            self.detectors = {self.detectors.name: self.detectors}
        self.pulse = pulse

    def ray_trace(self, npulses=1):
        # TODO: ray-trace multiple pulses
        components = sorted(
            chain(self.choppers.values(), self.detectors.values()),
            key=lambda c: c.distance,
        )

        initial_mask = np.full_like(self.pulse.birth_times, True, dtype=bool)
        for comp in components:
            t = self.pulse.birth_times + comp.distance / self.pulse.speeds
            comp._arrival_times = t
            if isinstance(comp, Detector):
                comp._mask = initial_mask
                continue
            m = np.full_like(t, False, dtype=bool)
            to = comp.open_times
            tc = comp.close_times
            for i in range(len(to)):
                m |= (t > to[i]) & (t < tc[i])
            combined = initial_mask & m
            comp._mask = combined
            initial_mask = combined

        # self.detector._arrival_times = (
        #     self.pulse.birth_times + self.detector.distance / self.pulse.speeds
        # )
        # self.detector._mask = initial_mask

    def plot(self, nrays=1000):
        fig, ax = plt.subplots()

        furthest_detector = max(self.detectors.values(), key=lambda d: d.distance)

        tofs = furthest_detector.tofs
        tof_max = tofs.max()
        inds = np.random.choice(len(tofs), size=nrays, replace=False)
        # Plot rays
        x0 = s_to_us(self.pulse.birth_times[furthest_detector._mask][inds]).reshape(
            -1, 1
        )
        x1 = tofs[inds].reshape(-1, 1)
        y0 = np.zeros(nrays).reshape(-1, 1)
        y1 = np.full(nrays, furthest_detector.distance).reshape(-1, 1)
        segments = np.concatenate(
            (
                np.concatenate((x0, y0), axis=1).reshape(-1, 1, 2),
                np.concatenate((x1, y1), axis=1).reshape(-1, 1, 2),
            ),
            axis=1,
        )
        coll = LineCollection(segments, cmap=plt.cm.gist_rainbow_r)
        coll.set_array(
            (self.pulse.wavelengths[furthest_detector._mask][inds] - self.pulse.lmin)
            / (self.pulse.lmax - self.pulse.lmin)
        )
        ax.add_collection(coll)
        # Plot choppers
        for name, ch in self.choppers.items():
            x0 = s_to_us(ch.open_times)
            x1 = s_to_us(ch.close_times)
            x = np.empty(3 * x0.size, dtype=x0.dtype)
            x[0::3] = x0
            x[1::3] = 0.5 * (x0 + x1)
            x[2::3] = x1
            x = np.concatenate([[0], x, [tof_max]])
            y = np.full_like(x, ch.distance)
            y[2::3] = None
            ax.plot(x, y, color="k")
            ax.text(tof_max, ch.distance, name, ha="right", va="bottom", color="k")

        # Plot detectors
        for name, det in self.detectors.items():
            ax.plot([0, tof_max], [det.distance] * 2, color="gray", lw=3)
            ax.text(0, det.distance, name, ha="left", va="bottom", color="gray")

        ax.set_xlabel("Time-of-flight (us)")
        ax.set_ylabel("Distance (m)")
        return Plot(fig=fig, ax=ax)

    def __repr__(self):
        return (
            f"Model(choppers={self.choppers},\n      "
            f"detectors={self.detectors},\n      "
            f"pulse={self.pulse},\n      "
            f"neutrons={len(self.pulse.birth_times)})"
        )
