# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

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
    # toa: ReadingField
    # wavelength: ReadingField
    # birth_time: ReadingField
    # speed: ReadingField

    def _repr_stats(self) -> str:
        return f"visible={int(self.data.sum().value)}"

    def __repr__(self) -> str:
        out = f"DetectorReading: '{self.name}'\n"
        out += f"  distance: {self.distance:c}\n"
        out += "  fields: toa, wavelength, birth_time, speed\n  "
        # out += f"  visible={int(self.data.sum().value)}\n"
        return out + self._repr_stats() + "\n"

    def __str__(self) -> str:
        return self.__repr__()

    # @property
    # @deprecated("Use 'toa' instead.")
    # def tof(self) -> ReadingField:
    #     return self.toa

    # @property
    # def toa(self) -> ReadingField:
    #     return ReadingField(
    #         data=sc.DataArray(
    #             data=self.data.data,
    #             coords={"toa": self.data.coords["toa"]},
    #             masks=self.data.masks,
    #         ),
    #         dim="toa",
    #     )

    # @property
    # def wavelength(self) -> ReadingField:
    #     return ReadingField(
    #         data=sc.DataArray(
    #             data=self.data.data,
    #             coords={"wavelength": self.data.coords["wavelength"]},
    #             masks=self.data.masks,
    #         ),
    #         dim="wavelength",
    #     )

    # @property
    # def birth_time(self) -> ReadingField:
    #     return ReadingField(
    #         data=sc.DataArray(
    #             data=self.data.data,
    #             coords={"birth_time": self.data.coords["birth_time"]},
    #             masks=self.data.masks,
    #         ),
    #         dim="birth_time",
    #     )

    # @property
    # def speed(self) -> ReadingField:
    #     return ReadingField(
    #         data=sc.DataArray(
    #             data=self.data.data,
    #             coords={"speed": self.data.coords["speed"]},
    #             masks=self.data.masks,
    #         ),
    #         dim="speed",
    #     )
