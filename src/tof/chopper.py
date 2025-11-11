# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

import scipp as sc

from .reading import ComponentReading
from .utils import two_pi, var_to_dict

if TYPE_CHECKING:
    try:
        from scippneutron.chopper import DiskChopper
    except ImportError:
        DiskChopper = object


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
        phase: sc.Variable | None = None,
        open: sc.Variable | None = None,
        close: sc.Variable | None = None,
        centers: sc.Variable | None = None,
        widths: sc.Variable | None = None,
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
        self, time_limit: sc.Variable | None = None, unit: str | None = None
    ) -> tuple[sc.Variable, sc.Variable]:
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
        # We make a unique dim name that is different from self.open.dim and
        # self.close.dim to make use of automatic broadcasting below.
        # We also start at -1 to catch early openings in case the phase or opening
        # angles are large
        phases = sc.arange(
            f"{self.open.dim}-{self.close.dim}", -1, nrot
        ) * two_pi + self.phase.to(unit='rad')

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

    def as_dict(self) -> dict:
        """
        Return the chopper as a dictionary.
        """
        return {
            'frequency': self.frequency,
            'open': self.open,
            'close': self.close,
            'distance': self.distance,
            'phase': self.phase,
            'name': self.name,
            'direction': self.direction,
        }

    def as_json(self) -> dict:
        """
        Return the chopper as a JSON-serializable dictionary.
        """
        out = {
            key: var_to_dict(value)
            for key, value in self.as_dict().items()
            if isinstance(value, sc.Variable)
        }
        out.update(
            {
                'type': 'chopper',
                'direction': self.direction.name.lower(),
                'name': self.name,
            }
        )
        return out

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chopper):
            return NotImplemented
        if self.name != other.name:
            return False
        if self.direction != other.direction:
            return False
        return all(
            sc.identical(getattr(self, field), getattr(other, field))
            for field in ('frequency', 'distance', 'phase', 'open', 'close')
        )

    @classmethod
    def from_diskchopper(cls, disk_chopper: DiskChopper) -> Chopper:
        """
        Create a Chopper from a scippneutron DiskChopper.

        Example
        -------
        >>> import scipp as sc
        >>> import tof
        >>> from scippneutron.chopper import DiskChopper
        >>> disk_chopper = DiskChopper(
        ...     frequency=sc.scalar(28.0, unit='Hz'),
        ...     beam_position=sc.scalar(30.0, unit='deg'),
        ...     slit_begin=sc.array(
        ...         dims=['slit'], values=[0.0, 90.0, 150.0], unit='deg'
        ...     ),
        ...     slit_end=sc.array(
        ...         dims=['slit'], values=[60.0, 120.0, 210.0], unit='deg'
        ...     ),
        ...     phase=sc.scalar(15.0, unit='deg'),
        ...     axle_position=sc.vector(value=[0.0, 0.0, 13.0], unit='m'),
        ... )
        >>> chopper = tof.Chopper.from_diskchopper(disk_chopper)
        """

        distance = disk_chopper.axle_position.fields.z
        freq = abs(disk_chopper.frequency)
        return cls(
            frequency=freq,
            direction=AntiClockwise
            if (disk_chopper.frequency.value > 0.0)
            else Clockwise,
            open=disk_chopper.slit_begin - disk_chopper.beam_position,
            close=disk_chopper.slit_end - disk_chopper.beam_position,
            phase=disk_chopper.phase
            if disk_chopper.frequency.value > 0.0
            else -disk_chopper.phase,
            distance=distance,
            name=(
                f"Chopper@{distance.value:.1f}{distance.unit}_"
                f"{int(freq.value)}{freq.unit}"
            ),
        )

    @classmethod
    def from_nexus(cls, nexus_chopper) -> Chopper:
        """
        Create a Chopper from a NeXus chopper group.

        Example
        -------
        >>> # Assuming a chopper NeXus structure
        >>> import scipp as sc
        >>> import tof
        >>> nexus_chopper = {
        ...     'position': sc.vector([0.0, 0.0, 2.0], unit='m'),
        ...     'rotation_speed': sc.scalar(12.0, unit='Hz'),
        ...     'beam_position': sc.scalar(45.0, unit='deg'),
        ...     'phase': sc.scalar(20.0, unit='deg'),
        ...     'slit_edges': sc.array(
        ...         dims=['slit'],
        ...         values=[0.0, 60.0, 124.0, 126.0],
        ...         unit='deg',
        ...     ),
        ...     'slit_height': sc.scalar(0.4, unit='m'),
        ...     'radius': sc.scalar(0.5, unit='m'),
        ... }
        >>> chopper = tof.Chopper.from_nexus(nexus_chopper)
        """
        distance = nexus_chopper['position'].fields.z
        freq = abs(nexus_chopper['rotation_speed'])
        return cls(
            frequency=freq,
            direction=AntiClockwise
            if (nexus_chopper['rotation_speed'].value > 0.0)
            else Clockwise,
            open=nexus_chopper['slit_edges'][::2] - nexus_chopper['beam_position'],
            close=nexus_chopper['slit_edges'][1::2] - nexus_chopper['beam_position'],
            phase=nexus_chopper['phase']
            if nexus_chopper['rotation_speed'].value > 0.0
            else -nexus_chopper['phase'],
            distance=distance,
            name=(
                f"Chopper@{distance.value:.1f}{distance.unit}_"
                f"{int(freq.value)}{freq.unit}",
            ),
        )

    def to_diskchopper(self) -> DiskChopper:
        """
        Export the chopper as a scippneutron DiskChopper.
        """
        from scippneutron.chopper import DiskChopper

        frequency = (
            self.frequency if self.direction == AntiClockwise else -self.frequency
        )
        phase = self.phase if self.direction == AntiClockwise else -self.phase
        return DiskChopper(
            frequency=frequency,
            beam_position=sc.scalar(0.0, unit='deg'),
            slit_begin=self.open,
            slit_end=self.close,
            phase=phase,
            axle_position=sc.vector(
                value=[0.0, 0.0, self.distance.value], unit=self.distance.unit
            ),
        )


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

    def _repr_stats(self) -> str:
        return (
            f"visible={int(self.data.sum().value)}, "
            f"blocked={int(self.data.masks['blocked_by_me'].sum().value)}"
        )

    def __repr__(self) -> str:
        return f"""ChopperReading: '{self.name}'
  distance: {self.distance:c}
  frequency: {self.frequency:c}
  phase: {self.phase:c}
  cutouts: {len(self.open)}
  neutrons: {self._repr_stats()}
"""

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, val: int | slice | tuple[str, int | slice]) -> ChopperReading:
        if isinstance(val, int):
            val = ('pulse', val)
        return replace(self, data=self.data[val])
