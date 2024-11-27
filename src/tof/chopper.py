# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import scipp as sc

from .deprecation import deprecated
from .reading import ComponentReading, ReadingField
from .utils import two_pi


class Direction(Enum):
    CLOCKWISE = 1
    ANTICLOCKWISE = 2


Clockwise = Direction.CLOCKWISE
AntiClockwise = Direction.ANTICLOCKWISE


class Chopper:
    """
    A chopper is a rotating device with cutouts that blocks the beam at certain times.

    Parameters
    ----------
    frequency:
        The frequency of the chopper. Must be positive.
    distance:
        The distance from the source to the chopper.
    name:
        The name of the chopper.
    phase:
        The phase of the chopper. Because the phase offset implemented as a time delay
        on real beamline choppers, it is applied in the opposite direction
        to the chopper rotation direction. For example, if the chopper rotates
        clockwise, a phase of 10 degrees will shift all window angles by 10 degrees
        in the anticlockwise direction, which will result in the windows opening later.
    open:
        The opening angles of the chopper cutouts.
    close:
        The closing angles of the chopper cutouts.
    centers:
        The centers of the chopper cutouts.
    widths:
        The widths of the chopper cutouts.

    Notes
    -----
    Either `open` and `close` or `centers` and `widths` must be provided, but not both.
    """

    def __init__(
        self,
        *,
        frequency: sc.Variable,
        distance: sc.Variable,
        name: str,
        phase: Optional[sc.Variable] = None,
        open: Optional[sc.Variable] = None,
        close: Optional[sc.Variable] = None,
        centers: Optional[sc.Variable] = None,
        widths: Optional[sc.Variable] = None,
        direction: Direction = Clockwise,
    ):
        if frequency <= (0.0 * frequency.unit):
            raise ValueError(f"Chopper frequency must be positive, got {frequency:c}.")
        self.frequency = frequency.to(dtype=float, copy=False)
        if direction not in (Clockwise, AntiClockwise):
            raise ValueError(
                "Chopper direction must be Clockwise or AntiClockwise"
                f", got {direction}."
            )
        if phase is None:
            phase = sc.scalar(0.0, unit='deg')
        self.direction = direction
        # Check that either open/close or centers/widths are provided, but not both
        if tuple(x for x in (open, close, centers, widths) if x is not None) not in (
            (open, close),
            (centers, widths),
        ):
            raise ValueError(
                "Either open/close or centers/widths must be provided, got"
                f" open={open}, close={close}, centers={centers}, widths={widths}."
            )
        if open is None:
            half_width = widths * 0.5
            open = centers - half_width
            close = centers + half_width

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
        self, time_limit: Optional[sc.Variable] = None, unit: Optional[str] = None
    ) -> Tuple[sc.Variable, sc.Variable]:
        """
        The times at which the chopper opens and closes.

        Parameters
        ----------
        time_limit:
            Determines how many rotations the chopper needs to perform to reach the time
            limit. If not specified, the chopper will perform a single rotation.
        unit:
            The unit of the returned times. If not specified, the unit of `time_limit`
            is used.
        """
        if time_limit is None:
            time_limit = sc.scalar(0.0, unit='us')
        if unit is None:
            unit = time_limit.unit
        nrot = max(int(sc.ceil((time_limit * self.frequency).to(unit='')).value), 1)
        # Start at -1 to catch early openings in case the phase or opening angles are
        # large
        phases = sc.arange(uuid.uuid4().hex, -1, nrot) * two_pi + self.phase.to(
            unit='rad'
        )

        open_times = self.open.to(unit='rad', copy=False)
        close_times = self.close.to(unit='rad', copy=False)
        # If the chopper is rotating anti-clockwise, we mirror the openings because the
        # first cutout will be the last to open.
        if self.direction == AntiClockwise:
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
        # Note that the order is important here: we need (phases + open/close) to get
        # the correct dimension order when we flatten.
        open_times = (phases + open_times).flatten(to=self.open.dim)
        close_times = (phases + close_times).flatten(to=self.close.dim)
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
            f"direction={self.direction.name}, cutouts={len(self.open)})"
        )

    def as_dict(self):
        return {
            'frequency': self.frequency,
            'open': self.open,
            'close': self.close,
            'distance': self.distance,
            'phase': self.phase,
            'name': self.name,
            'direction': self.direction,
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
    toas: ReadingField
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
            for key in ('toas', 'wavelengths', 'birth_times', 'speeds')
        )
        return out

    def __str__(self) -> str:
        return self.__repr__()

    @property
    @deprecated("Use 'toas' instead.")
    def tofs(self) -> ReadingField:
        return self.toas
