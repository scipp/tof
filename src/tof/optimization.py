# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Compute result of applying a chopper cascade to a neutron pulse at a time-of-flight
neutron source.

This is copied over from scippneutron/tof/chopper_cascade.py, with the unused parts
stripped out (see https://github.com/scipp/scippneutron).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipp as sc

from .chopper import Chopper
from .utils import wavelength_to_inverse_speed


def propagate_times(
    time: sc.Variable, wavelength: sc.Variable, distance: sc.Variable
) -> sc.Variable:
    """
    Propagate a neutron frame by a distance.

    Parameters
    ----------
    time:
        Time of the neutron frame.
    wavelength:
        Wavelength of the neutron frame.
    distance:
        Distance to propagate. Can be a range of distances.

    Returns
    -------
    :
        Propagated time.
    """
    inverse_velocity = wavelength_to_inverse_speed(wavelength)
    return time + (distance * inverse_velocity).to(unit=time.unit, copy=False)


class Subframe:
    """
    Neutron "subframe" at a time-of-flight neutron source, described as the corners of a
    polygon (initially a rectangle) in time and wavelength.
    """

    def __init__(self, time: sc.Variable, wavelength: sc.Variable):
        if {dim: time.sizes.get(dim) for dim in wavelength.sizes} != wavelength.sizes:
            raise sc.DimensionError(
                f'Inconsistent dims or shape: {time.sizes} vs {wavelength.sizes}'
            )
        self.time = time.to(unit='s', copy=False)
        self.wavelength = wavelength.to(unit='angstrom', copy=False)

    def propagate_by(self, distance: sc.Variable) -> Subframe:
        """
        Propagate subframe by a distance.

        Parameters
        ----------
        distance:
            Distance to propagate. Note that this is a difference, not an absolute
            value, in contrast to the distance in :py:meth:`Frame.propagate_to`.

        Returns
        -------
        :
            Propagated subframe.
        """
        return Subframe(
            time=propagate_times(self.time, self.wavelength, distance),
            wavelength=self.wavelength,
        )


@dataclass
class Frame:
    """
    A frame of neutrons, created from a single neutron pulse, potentially chopped into
    subframes by choppers.
    """

    distance: sc.Variable
    subframes: list[Subframe]

    def propagate_to(self, distance: sc.Variable) -> Frame:
        """
        Compute new frame by propagating to a distance.

        Parameters
        ----------
        distance:
            New distance.

        Returns
        -------
        :
            Propagated frame.
        """
        delta = distance.to(unit=self.distance.unit, copy=False) - self.distance
        subframes = [subframe.propagate_by(delta) for subframe in self.subframes]
        return Frame(distance=distance, subframes=subframes)

    def chop(self, chopper: Chopper) -> Frame:
        """
        Compute a new frame by applying a chopper.

        A frame is a polygon in time and wavelength. Its initial shape is distorted
        by propagation to the chopper. The chopper then cuts off the parts of the frame
        that is outside of the chopper opening. Here we apply and algorithm that
        computes a new polygon that is the intersection of the frame and the chopper
        opening.

        In practice a chopper may have multiple openings, so a frame may be chopped into
        a number of subframes.

        Parameters
        ----------
        chopper:
            Chopper to apply.

        Returns
        -------
        :
            Chopped frame.
        """
        distance = chopper.distance.to(unit=self.distance.unit, copy=False)
        if distance < self.distance:
            raise ValueError(
                f'Chopper distance {distance} is smaller than frame distance '
                f'{self.distance}'
            )
        frame = self.propagate_to(distance)

        # A chopper can have multiple openings, call _chop for each of them. The result
        # is the union of the resulting subframes.
        chopped = Frame(distance=frame.distance, subframes=[])
        open_times, close_times = (t.to(unit='s') for t in chopper.open_close_times())
        for subframe in frame.subframes:
            for open, close in zip(open_times, close_times, strict=True):
                if (tmp := _chop(subframe, open, close_to_open=True)) is not None:
                    if (tmp := _chop(tmp, close, close_to_open=False)) is not None:
                        chopped.subframes.append(tmp)
        return chopped


@dataclass
class FrameSequence:
    """
    A sequence of frames, created from a single neutron pulse, potentially chopped into
    subframes by choppers.

    It is recommended to use the :py:meth:`from_source_pulse` constructor to create a
    frame sequence from a source pulse. Then, a chopper cascade can be applied using
    :py:meth:`chop`.
    """

    frames: list[Frame]

    @staticmethod
    def from_source_pulse(
        time_min: sc.Variable,
        time_max: sc.Variable,
        wavelength_min: sc.Variable,
        wavelength_max: sc.Variable,
    ):
        """
        Initialize a frame sequence from min/max time and wavelength of a pulse.

        The distance is set to 0 m.
        """
        time = sc.concat([time_min, time_max, time_max, time_min], dim='vertex').to(
            unit='s'
        )
        wavelength = sc.concat(
            [wavelength_min, wavelength_min, wavelength_max, wavelength_max],
            dim='vertex',
        ).to(unit='angstrom')
        frames = [
            Frame(
                distance=sc.scalar(0, unit='m'),
                subframes=[Subframe(time=time, wavelength=wavelength)],
            )
        ]
        return FrameSequence(frames)

    def __len__(self) -> int:
        """Number of frames."""
        return len(self.frames)

    def __getitem__(self, item: int | sc.Variable) -> Frame:
        """Get a frame by index or distance."""
        if isinstance(item, int):
            return self.frames[item]
        distance = item.to(unit='m')
        frame_before_detector = None
        for frame in self:
            if frame.distance > distance:
                break
            frame_before_detector = frame

        return frame_before_detector.propagate_to(distance)

    def propagate_to(self, distance: sc.Variable) -> FrameSequence:
        """
        Propagate the frame sequence to a distance, adding a new frame.

        Use this, e.g., to propagate to the sample position after applying choppers.

        Parameters
        ----------
        distance:
            Distance to propagate.

        Returns
        -------
        :
            New frame sequence.
        """
        return FrameSequence([*self.frames, self.frames[-1].propagate_to(distance)])

    def chop(self, choppers: list[Chopper]) -> FrameSequence:
        """
        Chop the frame sequence by a list of choppers.

        The choppers will be sorted by their distance, and applied in order.

        Parameters
        ----------
        choppers:
            List of choppers.

        Returns
        -------
        :
            New frame sequence.
        """
        choppers = sorted(choppers, key=lambda x: x.distance)
        frames = list(self.frames)
        for chopper in choppers:
            frames.append(frames[-1].chop(chopper))
        return FrameSequence(frames)


def _chop(frame: Subframe, time: sc.Variable, close_to_open: bool) -> Subframe | None:
    inside = frame.time >= time if close_to_open else frame.time <= time
    output = []
    for i in range(len(frame.time)):
        # Note how j wraps around to 0
        j = (i + 1) % len(frame.time)
        inside_i = inside[i]
        inside_j = inside[j]
        if inside_i:
            output.append((frame.time[i], frame.wavelength[i]))
        if inside_i != inside_j:
            # Intersection
            t = (time - frame.time[i]) / (frame.time[j] - frame.time[i])
            v = (1 - t) * frame.wavelength[i] + t * frame.wavelength[j]
            output.append((time, v))
    if not output:
        return None
    time = sc.concat([t for t, _ in output], dim=frame.time.dim)
    wavelength = sc.concat([v for _, v in output], dim=frame.wavelength.dim)
    return Subframe(time=time, wavelength=wavelength)
