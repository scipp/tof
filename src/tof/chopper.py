# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import uuid
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import scipp as sc

from .reading import ComponentReading, ReadingField
from .utils import two_pi


class Chopper:
    """
    A chopper is a rotating device with cutouts that blocks the beam at certain times.

    Parameters
    ----------
    frequency:
        The frequency of the chopper. Must be positive.
    open:
        The opening angles of the chopper cutouts. Note that the order of the values
        listed here defines the order in which the windows will pass in front of the
        beam. If your chopper is counter-rotating, you have to list the last window
        first.
    close:
        The closing angles of the chopper cutouts. Note that the order of the values
        listed here defines the order in which the windows will pass in front of the
        beam. If your chopper is counter-rotating, you have to list the last window
        first.
    distance:
        The distance from the source to the chopper.
    phase:
        The phase of the chopper. In addition to the normal phase of the chopper, this
        can be used to account for the fact that the chopper may be below, above, or
        to the side of the beam.
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
        direction: Literal['clockwise', 'anticlockwise'] = 'clockwise',
        name: str = "",
    ):
        if frequency < (0.0 * frequency.unit):
            raise ValueError(f"Chopper frequency must be positive, got {frequency:c}")
        self.frequency = frequency.to(dtype=float, copy=False)
        self.open = (open if open.dims else open.flatten(to='cutout')).to(
            dtype=float, copy=False
        )
        self.close = (close if close.dims else close.flatten(to='cutout')).to(
            dtype=float, copy=False
        )
        self.direction = direction
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
        self, time_limit: sc.Variable, unit: Optional[str] = None
    ) -> Tuple[sc.Variable, sc.Variable]:
        """
        The times at which the chopper opens and closes.

        Parameters
        ----------
        time_limit:
            Determines how many rotations the chopper needs to perform to reach the time
            limit.
        unit:
            The unit of the returned times. If not specified, the unit of `time_limit`
            is used.
        """
        if unit is None:
            unit = time_limit.unit
        nrot = max(int(sc.ceil((time_limit * self.frequency).to(unit='')).value), 1)
        # Start at -1 to catch early openings in case the phase or opening angles are
        # large
        phases = sc.arange(uuid.uuid4().hex, -1, nrot) * two_pi + self.phase.to(
            unit='rad'
        ) * (int(self.direction == 'clockwise') * 2 - 1)
        # Note that the order is important here: we need (phases + open/close) to get
        # the correct dimension order when we flatten below.
        open_times = (phases + self.open.to(unit='rad', copy=False)).flatten(
            to=self.open.dim
        )
        close_times = (phases + self.close.to(unit='rad', copy=False)).flatten(
            to=self.close.dim
        )
        # If the chopper is rotating anti-clockwise, we mirror the openings because the
        # first cutout will be the last to open.
        if self.direction != 'clockwise':
            open_times, close_times = (
                sc.array(
                    dims=close_times.dims,
                    values=(two_pi - close_times).values[::-1],
                    unit=close_times.unit,
                ),
                sc.array(
                    dims=open_times.dims,
                    values=(two_pi - open_times).values[::-1],
                    unit=open_times.unit,
                ),
            )
        open_times /= self.omega
        close_times /= self.omega
        return (
            open_times.to(unit=unit, copy=False),
            close_times.to(unit=unit, copy=False),
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
class ChopperReading(ComponentReading):
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
    data: sc.DataArray
    tofs: ReadingField
    wavelengths: ReadingField
    birth_times: ReadingField
    speeds: ReadingField

    def __repr__(self) -> str:
        out = f"ChopperReading: '{self.name}'\n"
        out += f"  distance: {self.distance:c}\n"
        out += f"  frequency: {self.frequency:c}\n"
        out += f"  phase: {self.phase:c}\n"
        out += f"  cutouts: {len(self.open)}\n"
        out += "\n".join(
            f"  {key}: {getattr(self, key)}"
            for key in ('tofs', 'wavelengths', 'birth_times', 'speeds')
        )
        return out

    def __str__(self) -> str:
        return self.__repr__()
