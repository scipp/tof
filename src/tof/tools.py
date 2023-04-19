# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass

from matplotlib.pyplot import Axes, Figure


@dataclass
class Plot:
    ax: Axes
    fig: Figure
