# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import uuid
from dataclasses import dataclass
from typing import Tuple

import scipp as sc

from .component import Component, ComponentData
from .utils import two_pi


class Chopper:
    """
    A chopper is a rotating device with cutouts that blocks the beam at certain times.

    Parameters
    ----------
    frequency:
        The frequency of the chopper.
    open:
        The opening angles of the chopper cutouts.
    close:
        The closing angles of the chopper cutouts.
    distance:
        The distance from the source to the chopper.
    phase:
        The phase of the chopper.
    name:
        The name of the chopper.
    """

    def __init__(
        self,
        frequency: sc.Variable,
        open: sc.Variable,
        close: sc.Variable,
        distance: sc.Variable,
        phase: sc.Variable,
        name: str = "",
    ):
        self.frequency = frequency.to(dtype=float, copy=False)
        self.open = (open if open.dims else open.flatten(to='cutout')).to(
            dtype=float, copy=False
        )
        self.close = (close if close.dims else close.flatten(to='cutout')).to(
            dtype=float, copy=False
        )
        self.distance = distance.to(dtype=float, copy=False)
        self.phase = phase.to(dtype=float, copy=False)
        self.name = name
        super().__init__()

    @property
    def omega(self) -> sc.Variable:
        """
        The angular velocity of the chopper.
        """
        return two_pi * self.frequency

    def open_close_times(
        self, time_limit: sc.Variable
    ) -> Tuple[sc.Variable, sc.Variable]:
        """
        The times at which the chopper opens and closes.

        Parameters
        ----------
        time_limit:
            Determines how many rotations the chopper needs to perform to reach the time
            limit.
        """
        time_limit = time_limit.to(unit='s')
        nrot = max(int(sc.ceil(time_limit * self.frequency).value), 1)
        phases = sc.arange(uuid.uuid4().hex, nrot) * two_pi + self.phase.to(unit='rad')
        # Note that the order is important here: we need (phases + open/close) to get
        # the correct dimension order when we flatten below.
        open_times = (phases + self.open.to(unit='rad', copy=False)) / self.omega
        close_times = (phases + self.close.to(unit='rad', copy=False)) / self.omega
        return (
            open_times.flatten(to=self.open.dim),
            close_times.flatten(to=self.close.dim),
        )

    def __repr__(self) -> str:
        return (
            f"Chopper(name={self.name}, distance={self.distance:c}, "
            f"frequency={self.frequency:c}, phase={self.phase:c}, "
            f"cutouts={len(self.open)})"
        )

    def as_dict(self):
        return {
            'frequency': self.frequency,
            'open': self.open,
            'close': self.close,
            'distance': self.distance,
            'phase': self.phase,
            'name': self.name,
        }


@dataclass(frozen=True)
class ChopperReading(Component):
    """
    Read-only container for the neutrons that reach the chopper.
    """

    distance: sc.Variable
    name: str
    frequency: sc.Variable
    open: sc.Variable
    close: sc.Variable
    phase: sc.Variable
    open_times: sc.Variable
    close_times: sc.Variable
    tofs: ComponentData
    wavelengths: ComponentData
    birth_times: ComponentData
    speeds: ComponentData

    def __repr__(self) -> str:
        out = f"Chopper: '{self.name}'\n"
        out += f"  distance: {self.distance:c}\n"
        out += f"  frequency: {self.frequency:c}\n"
        out += f"  phase: {self.phase:c}\n"
        out += f"  cutouts: {len(self.open)}\n"
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
