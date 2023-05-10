# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass

import scipp as sc

from .component import Component, ComponentData


class Detector:
    """
    A detector component does not block any neutrons, it sees all neutrons passing
    through it.

    Parameters
    ----------
    distance:
        The distance from the source to the detector.
    name:
        The name of the detector.
    """

    def __init__(self, distance: sc.Variable, name: str = "detector"):
        self.distance = distance.to(dtype=float, copy=False)
        self.name = name

    def __repr__(self) -> str:
        return f"Detector(name={self.name}, distance={self.distance:c})"

    def as_dict(self):
        return {'distance': self.distance, 'name': self.name}


@dataclass(frozen=True)
class DetectorReading(Component):
    distance: sc.Variable
    name: str
    tofs: ComponentData
    wavelengths: ComponentData
    birth_times: ComponentData
    speeds: ComponentData

    def __repr__(self) -> str:
        out = f"Detector: '{self.name}'\n"
        out += f"  distance: {self.distance:c}\n"
        for key, dim in {
            'tofs': 'tof',
            'wavelengths': 'wavelength',
            'birth_times': 'time',
            'speeds': 'speed',
        }.items():
            coord = getattr(self, key).visible.data.coords[dim]
            out += f"  {key}: [{coord.min():c} - {coord.max():c}]\n"
        return out

    def __str__(self) -> str:
        return self.__repr__()
