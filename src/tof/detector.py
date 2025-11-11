# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from dataclasses import dataclass, replace

import scipp as sc

from .reading import ComponentReading
from .utils import var_to_dict


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

    def as_dict(self) -> dict:
        """
        Return the detector as a dictionary.
        """
        return {'distance': self.distance, 'name': self.name}

    def as_json(self) -> dict:
        """
        Return the detector as a JSON-serializable dictionary.
        """
        return {
            'type': 'detector',
            'distance': var_to_dict(self.distance),
            'name': self.name,
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Detector):
            return NotImplemented
        return self.name == other.name and sc.identical(self.distance, other.distance)


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

    def __getitem__(
        self, val: int | slice | tuple[str, int | slice]
    ) -> DetectorReading:
        if isinstance(val, int):
            val = ('pulse', val)
        return replace(self, data=self.data[val])
