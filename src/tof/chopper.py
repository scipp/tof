# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from . import units
from .tools import Plot


class Chopper:
    def __init__(
        self,
        frequency: float,
        open: Union[List, np.ndarray],
        close: Union[List, np.ndarray],
        distance: float,
        phase: float = 0,
        unit: str = "deg",
        name: str = "",
    ):
        self.frequency = frequency
        self.open = open
        self.close = close
        self.distance = distance
        self.phase = phase
        if unit == "deg":
            self.open = units.deg_to_rad(self.open)
            self.close = units.deg_to_rad(self.close)
            self.phase = units.deg_to_rad(self.phase)
        self.name = name

        self._arrival_times = None
        self._mask = None

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.frequency

    @property
    def open_times(self) -> np.ndarray:
        return (self.open + self.phase) / self.omega

    @property
    def close_times(self) -> np.ndarray:
        return (self.close + self.phase) / self.omega

    @property
    def tofs(self) -> np.ndarray:
        return units.s_to_us(self._arrival_times[self._mask])

    def hist(self, bins: Union[int, np.ndarray] = 300) -> Tuple[np.ndarray, np.ndarray]:
        return np.histogram(self.tofs, bins=bins)

    def plot(self, bins: Union[int, np.ndarray] = 300) -> Plot:
        h, edges = self.hist(bins=bins)
        fig, ax = plt.subplots()
        x = np.concatenate([edges, edges[-1:]])
        y = np.concatenate([[0], h, [0]])
        ax.step(x, y)
        ax.fill_between(x, 0, y, step="pre", alpha=0.5)
        ax.set_xlabel('Time-of-flight (us)')
        ax.set_ylabel('Counts')
        ax.set_title(f"Chopper: {self.name}")
        return Plot(fig=fig, ax=ax)

    def __repr__(self) -> str:
        return (
            f"Chopper(name={self.name}, distance={self.distance}, "
            f"frequency={self.frequency}, phase={self.phase}, "
            f"cutouts={len(self.open)})"
        )
