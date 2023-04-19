import numpy as np
import matplotlib.pyplot as plt


from .tools import Plot
from .units import s_to_us


class Detector:
    def __init__(self, distance: float = 0.0, name: str = "detector"):
        self.distance = distance
        self.name = name
        self._arrival_times = None
        self._mask = None

    @property
    def tofs(self):
        return s_to_us(self._arrival_times[self._mask])

    def hist(self, bins=300):
        return np.histogram(self.tofs, bins=bins)

    def plot(self, bins=300):
        h, edges = self.hist(bins=bins)
        fig, ax = plt.subplots()
        x = np.concatenate([edges, edges[-1:]])
        y = np.concatenate([[0], h, [0]])
        ax.step(x, y)
        ax.fill_between(x, 0, y, step="pre", alpha=0.5)
        ax.set_xlabel('Time-of-flight (us)')
        ax.set_ylabel('Counts')
        ax.set_title(f"Detector: {self.name}")
        return Plot(fig=fig, ax=ax)

    def __repr__(self):
        return f"Detector(name={self.name}, distance={self.distance})"
