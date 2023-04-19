import numpy as np
import matplotlib.pyplot as plt


from .tools import Plot
from .units import s_to_us


class Detector:
    def __init__(self, distance: float = 0.0):
        self.distance = distance
        self._arrival_times = None
        self._mask = None

    @property
    def tofs(self):
        return s_to_us(self._arrival_times[self._mask])

    def hist(self, bins=100):
        return np.histogram(self.tofs, bins=bins)

    def plot(self, bins=100):
        h, edges = self.hist(bins=bins)
        fig, ax = plt.subplots()
        x = np.concatenate([edges, edges[-1:]])
        y = np.concatenate([[0], h, [0]])
        ax.step(x, y)
        return Plot(fig=fig, ax=ax)

    def __repr__(self):
        return f"Detector(distance={self.distance})"
