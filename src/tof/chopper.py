# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc

from .component import Component


class Chopper(Component):
    def __init__(
        self,
        frequency: sc.Variable,
        open: sc.Variable,
        close: sc.Variable,
        distance: sc.Variable,
        phase: sc.Variable,
        name: str = "",
    ):
        self.frequency = frequency
        self.open = open
        self.close = close
        self.distance = distance
        self.phase = phase
        self.name = name
        super().__init__()

    @property
    def omega(self) -> sc.Variable:
        return sc.constants.pi * (2.0 * sc.units.rad) * self.frequency

    @property
    def open_times(self) -> sc.Variable:
        return (
            self.open.to(unit='rad', copy=False) + self.phase.to(unit='rad', copy=False)
        ) / self.omega

    @property
    def close_times(self) -> sc.Variable:
        return (
            self.close.to(unit='rad', copy=False)
            + self.phase.to(unit='rad', copy=False)
        ) / self.omega

    def __repr__(self) -> str:
        return (
            f"Chopper(name={self.name}, distance={self.distance:c}, "
            f"frequency={self.frequency:c}, phase={self.phase:c}, "
            f"cutouts={len(self.open)})"
        )
