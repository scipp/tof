from dataclasses import dataclass
from matplotlib.pyplot import Axes, Figure


@dataclass
class Plot:
    ax: Axes
    fig: Figure
