# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc

from .component import Component


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

    # def to_dict(self):
    #     return {'distance': self.distance, 'name': self.name}

    def as_readonly(self):
        return ReadonlyDetector(self)


class ReadonlyDetector(Component):
    """ """

    def __init__(self, detector: Detector):
        self._distance = detector.distance
        self._name = detector.name
        super().__init__()

    @property
    def distance(self) -> sc.Variable:
        return self._distance

    @property
    def name(self) -> str:
        return self._name
