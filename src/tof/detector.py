# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from dataclasses import dataclass, replace

import scipp as sc

from .component import Component
from .reading import ComponentReading
from .utils import var_to_dict


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


class Detector(Component):
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
        self.kind = "detector"

    def __repr__(self) -> str:
        return f"Detector(name={self.name}, distance={self.distance:c})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Detector):
            return NotImplemented
        return self.name == other.name and sc.identical(self.distance, other.distance)

    def as_dict(self) -> dict:
        """
        Return the detector as a dictionary.
        """
        return {'distance': self.distance, 'name': self.name}

    @classmethod
    def from_json(cls, name: str, params: dict) -> Detector:
        """
        Create a detector from a JSON-serializable dictionary.
        """
        return cls(
            distance=sc.scalar(
                params["distance"]["value"], unit=params["distance"]["unit"]
            ),
            name=name,
        )

    def as_json(self) -> dict:
        """
        Return the detector as a JSON-serializable dictionary.

        .. versionadded:: 25.11.0
        """
        return {
            'type': 'detector',
            'distance': var_to_dict(self.distance),
            'name': self.name,
        }

    def make_reading(self, neutrons: sc.DataGroup) -> DetectorReading:
        return DetectorReading(distance=self.distance, name=self.name, data=neutrons)

    def apply(self, neutrons: sc.DataGroup, time_limit: sc.Variable) -> sc.DataGroup:
        """
        Apply the detector to the given neutrons.
        A detector does not modify the neutrons, it simply records them without
        blocking any.

        Parameters
        ----------
        neutrons:
            The neutrons to which the detector will be applied.
        time_limit:
            The time limit for the neutrons to be considered as reaching the detector.
        """
        # neutrons.pop("blocked_by_me", None)
        return neutrons, self.make_reading(neutrons)
