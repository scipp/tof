# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc

from .component import Component


class Detector(Component):
    def __init__(self, distance: sc.Variable = 0.0, name: str = "detector"):
        self.distance = distance
        self.name = name
        self._arrival_times = None
        self._wavelengths = None
        self._mask = None

    def __repr__(self) -> str:
        return f"Detector(name={self.name}, distance={self.distance:c})"
