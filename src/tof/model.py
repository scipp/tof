import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

from .units import s_to_us
from .pulse import Pulse
from .tools import Plot


class Model:
    def __init__(self, choppers, detector, neutrons=1_000_000, pulse=None):
        self.choppers = choppers
        self.detector = detector
        self.pulse = pulse

        if self.pulse is None:
            self.pulse = Pulse(kind="ess")

        self.pulse.make_neutrons(neutrons)

    def ray_trace(self):
        sorted_choppers = [
            k
            for k, v in sorted(
                [(name, ch.distance) for name, ch in self.choppers.items()],
                key=lambda item: item[1],
            )
        ]

        initial_mask = np.full_like(self.pulse.birth_times, True, dtype=bool)
        for name in sorted_choppers:
            ch = self.choppers[name]
            t = self.pulse.birth_times + ch.distance / self.pulse.speeds
            ch._arrival_times = t
            m = np.full_like(t, False, dtype=bool)
            to = ch.open_times
            tc = ch.close_times
            for i in range(len(to)):
                m |= (t > to[i]) & (t < tc[i])
            combined = initial_mask & m
            ch._mask = combined
            initial_mask = combined

        self.detector._arrival_times = (
            self.pulse.birth_times + self.detector.distance / self.pulse.speeds
        )
        self.detector._mask = initial_mask

    def plot(self, nrays=1000):
        fig, ax = plt.subplots()
        tofs = self.detector.tofs
        inds = np.random.choice(len(tofs), size=nrays, replace=False)
        # Plot rays
        x0 = s_to_us(self.pulse.birth_times[self.detector._mask][inds]).reshape(-1, 1)
        x1 = tofs[inds].reshape(-1, 1)
        y0 = np.zeros(nrays).reshape(-1, 1)
        y1 = np.full(nrays, self.detector.distance).reshape(-1, 1)
        segments = np.concatenate(
            (
                np.concatenate((x0, y0), axis=1).reshape(-1, 1, 2),
                np.concatenate((x1, y1), axis=1).reshape(-1, 1, 2),
            ),
            axis=1,
        )
        coll = LineCollection(segments, cmap=plt.cm.gist_rainbow_r)
        coll.set_array(
            (
                self.pulse.wavelengths[self.detector._mask][inds]
                - self.pulse.wavelength_min
            )
            / (self.pulse.wavelength_max - self.pulse.wavelength_min)
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
            x = np.concatenate([[0], x, [tofs.max()]])
            y = np.full_like(x, ch.distance)
            y[2::3] = None
            ax.plot(x, y, color="k")
        return Plot(fig=fig, ax=ax)

    def __repr__(self):
        return (
            f"Model(choppers={self.choppers},\n      detector={self.detector},\n      "
            f"pulse={self.pulse},\n      neutrons={len(self.pulse.birth_times)})"
        )
