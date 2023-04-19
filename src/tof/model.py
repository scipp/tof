import numpy as np
from matplotlib import colormaps
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
        cmap = colormaps["gist_rainbow_r"]
        fig, ax = plt.subplots()
        tofs = self.detector.tofs
        inds = np.random.choice(len(tofs), size=nrays, replace=False)
        for i in inds:
            ax.plot(
                [s_to_us(self.pulse.birth_times[self.detector._mask][i]), tofs[i]],
                [0, self.detector.distance],
                color=cmap(
                    (
                        self.pulse.wavelengths[self.detector._mask][i]
                        - self.pulse.wavelength_min
                    )
                    / (self.pulse.wavelength_max - self.pulse.wavelength_min)
                ),
            )
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
