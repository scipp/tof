# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Tuple

import scipp as sc

from .component import Component
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
        open_times = []
        close_times = []
        nrot = 0
        phase = self.phase.to(unit='rad')
        time_limit = time_limit.to(unit='s')
        while True:
            nrot += 1
            open_times.append(
                (self.open.to(unit='rad', copy=False) + phase) / self.omega
            )
            close_times.append(
                (self.close.to(unit='rad', copy=False) + phase) / self.omega
            )
            if close_times[-1].max() > time_limit:
                break
            phase += two_pi
        return (
            sc.concat(open_times, self.open.dim),
            sc.concat(close_times, self.close.dim),
        )

    def __repr__(self) -> str:
        return (
            f"Chopper(name={self.name}, distance={self.distance:c}, "
            f"frequency={self.frequency:c}, phase={self.phase:c}, "
            f"cutouts={len(self.open)})"
        )

    def as_readonly(self):
        return ReadonlyChopper(self)


class ReadonlyChopper(Component):
    def __init__(self, chopper: Chopper):
        self._frequency = chopper.frequency
        self._open = chopper.open
        self._close = chopper.close
        self._distance = chopper.distance
        self._phase = chopper.phase
        self._name = chopper.name
        self._open_times = None
        self._close_times = None
        super().__init__()

    @property
    def frequency(self) -> sc.Variable:
        return self._frequency

    @property
    def open(self) -> sc.Variable:
        return self._open

    @property
    def close(self) -> sc.Variable:
        return self._close

    @property
    def distance(self) -> sc.Variable:
        return self._distance

    @property
    def phase(self) -> sc.Variable:
        return self._phase

    @property
    def name(self) -> str:
        return self._name

    @property
    def open_times(self) -> sc.Variable:
        return self._open_times

    @property
    def close_times(self) -> sc.Variable:
        return self._close_times

    def __repr__(self) -> str:
        return (
            f"Chopper(name={self.name}, distance={self.distance:c}, "
            f"frequency={self.frequency:c}, phase={self.phase:c}, "
            f"cutouts={len(self.open)})"
        )
