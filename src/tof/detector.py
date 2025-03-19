# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass

import scipp as sc

from .reading import ComponentReading


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

    def __init__(self, distance: sc.Variable, name: str):
        self.distance = distance.to(dtype=float, copy=False)
        self.name = name

    def __repr__(self) -> str:
        return f"Detector(name={self.name}, distance={self.distance:c})"

    def as_dict(self):
        return {'distance': self.distance, 'name': self.name}


@dataclass(frozen=True)
class DetectorReading(ComponentReading):
    """
    Read-only container for the neutrons that reach the detector.
    """

    distance: sc.Variable
    name: str
    data: sc.DataArray

    def _repr_stats(self) -> str:
        return f"visible={int(self.data.sum().value)}"

    def __repr__(self) -> str:
        return f"""DetectorReading: '{self.name}'
  distance: {self.distance:c}
  neutrons: {self._repr_stats()}
"""

    def __str__(self) -> str:
        return self.__repr__()
