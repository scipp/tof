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
            # for open, close in zip(chopper.time_open, chopper.time_close, strict=True):
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


def polygon_grid_overlap_mask(
    polygon: np.ndarray, X: np.ndarray, Y: np.ndarray, tol: float | None = None
) -> np.ndarray:
    """
    Compute a mask indicating which grid cells overlap with a polygon.
    This is used to select relevant regions of the source distribution where the
    selection is made from the polygons generated by the FrameSequence.

    Parameters
    ----------
    polygon : (N,2) array
        Coordinates of the polygon vertices.
    X, Y    : (H,W) array
        Coordinates of the grid corners.
    tol     : float, optional
        Tolerance for intersection checks (default = small fraction of grid spacing).

    Notes:
        Generated by ChatGPT-5.2.

        Original prompt:
        "Using numpy, can you make a function that would take in a polygon as a set
        of vertices, and a 2d grid of xy coordinates, and return a mask that would
        indicate which squares in the grid overlap with the polygon (even in the
        slightest of ways). The math should be exact, I can't afford to approximate
        the polygon with a cloud of points.
        I don't need to know the area of overlap between squares and the polygon,
        just whether there is overlap or not (True/False).

        Refinement:
        Can you:
        1. Integrate bounding-box pruning per edge directly into the function
        2. Add some tolerance when checking for equalities (maybe something like a
        fraction of the grid spacing) because it can sometimes go wrong when a
        polygon edge is perfectly aligned with a grid edge

    Returns:
        mask (H-1, W-1)
    """

    H, W = X.shape

    # --- Cell bounds ---
    cells_xmin = X[:-1, :-1]
    cells_xmax = X[1:, 1:]
    cells_ymin = Y[:-1, :-1]
    cells_ymax = Y[1:, 1:]

    # --- Auto tolerance ---
    if tol is None:
        dx = np.min(np.diff(X, axis=1))
        dy = np.min(np.diff(Y, axis=0))
        tol = 1e-9 + 1e-6 * min(dx, dy)

    # Expand cell bounds slightly (robustness)
    cells_xmin -= tol
    cells_xmax += tol
    cells_ymin -= tol
    cells_ymax += tol

    # --- Point in polygon ---
    def points_in_poly(px, py, poly):
        x = poly[:, 0]
        y = poly[:, 1]
        x2 = np.roll(x, -1)
        y2 = np.roll(y, -1)

        inside = np.zeros(px.shape, dtype=bool)

        for i in range(len(poly)):
            cond = ((y[i] > py) != (y2[i] > py)) & (
                px < (x2[i] - x[i]) * (py - y[i]) / (y2[i] - y[i] + tol) + x[i]
            )
            inside ^= cond

        return inside

    # --- 1. Corner inside polygon ---
    corners_x = np.stack([X[:-1, :-1], X[1:, :-1], X[:-1, 1:], X[1:, 1:]], axis=0)
    corners_y = np.stack([Y[:-1, :-1], Y[1:, :-1], Y[:-1, 1:], Y[1:, 1:]], axis=0)

    corner_inside = np.any(points_in_poly(corners_x, corners_y, polygon), axis=0)

    # --- 2. Polygon vertex inside cell ---
    vx = polygon[:, 0][:, None, None]
    vy = polygon[:, 1][:, None, None]

    vertex_inside = (
        (vx >= cells_xmin)
        & (vx <= cells_xmax)
        & (vy >= cells_ymin)
        & (vy <= cells_ymax)
    ).any(axis=0)

    # --- Segment intersection ---
    def segments_intersect(p1, p2, q1, q2):
        def orient(a, b, c):
            return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (
                b[..., 1] - a[..., 1]
            ) * (c[..., 0] - a[..., 0])

        o1 = orient(p1, p2, q1)
        o2 = orient(p1, p2, q2)
        o3 = orient(q1, q2, p1)
        o4 = orient(q1, q2, p2)

        return (o1 * o2 <= tol) & (o3 * o4 <= tol)

    edges_p1 = polygon
    edges_p2 = np.roll(polygon, -1, axis=0)

    intersect_mask = np.zeros((H - 1, W - 1), dtype=bool)

    # Cell edges
    cell_edges = [
        (
            np.stack([cells_xmin, cells_ymin], axis=-1),
            np.stack([cells_xmax, cells_ymin], axis=-1),
        ),
        (
            np.stack([cells_xmin, cells_ymax], axis=-1),
            np.stack([cells_xmax, cells_ymax], axis=-1),
        ),
        (
            np.stack([cells_xmin, cells_ymin], axis=-1),
            np.stack([cells_xmin, cells_ymax], axis=-1),
        ),
        (
            np.stack([cells_xmax, cells_ymin], axis=-1),
            np.stack([cells_xmax, cells_ymax], axis=-1),
        ),
    ]

    # --- 3. Edge intersection with pruning ---
    for p1, p2 in zip(edges_p1, edges_p2, strict=True):
        # Edge bounding box (+ tolerance)
        xmin = min(p1[0], p2[0]) - tol
        xmax = max(p1[0], p2[0]) + tol
        ymin = min(p1[1], p2[1]) - tol
        ymax = max(p1[1], p2[1]) + tol

        # Candidate cells only
        candidate = (
            (cells_xmax >= xmin)
            & (cells_xmin <= xmax)
            & (cells_ymax >= ymin)
            & (cells_ymin <= ymax)
        )

        if not np.any(candidate):
            continue

        p1e = p1[None, None, :]
        p2e = p2[None, None, :]

        for q1, q2 in cell_edges:
            hit = segments_intersect(p1e, p2e, q1, q2)
            intersect_mask[candidate] |= hit[candidate]

    return corner_inside | vertex_inside | intersect_mask
