# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Pre-configured chopper and detector parameters for Dream.
"""

from ...chopper import Chopper
from ...detector import Detector
from ..common import make_beamline

dream_high_flux = {
    "choppers": {
        "PSC1": {
            "frequency": 14.0,
            "phase": 286.0 - 180.0,
            "distance": 6.145,
            "centers": [0, 72, 86.4, 115.2, 172.8, 273.6, 288.0, 302.4],
            "widths": [2.46, 3.02, 3.27, 3.27, 5.02, 3.93, 3.93, 2.46],
            "direction": "anti-clockwise",
        },
        "PSC2": {
            "frequency": 14.0,
            "phase": 236.0,
            "distance": 6.155,
            "centers": [0, 28.8, 57.6, 144.0, 158.4, 216.0, 259.2, 316.8],
            "widths": [2.46, 3.60, 3.60, 3.23, 3.27, 3.77, 3.94, 2.62],
            "direction": "clockwise",
        },
        "OC": {
            "frequency": 14.0,
            "phase": 297.0 - 180.0 - 90.0,
            "distance": 6.174,
            "centers": [0.0],
            "widths": [27.6],
            "direction": "anti-clockwise",
        },
        "BC": {
            "frequency": 112.0,
            "phase": 240.0 - 180.0,
            "distance": 9.78,
            "centers": [0.0, 180.0],
            "widths": [73.75, 73.75],
            "direction": "anti-clockwise",
        },
        "T0": {
            "frequency": 28.0,
            "phase": 280.0 - 180.0,
            "distance": 13.05,
            "centers": [0.0],
            "widths": [314.9],
            "direction": "anti-clockwise",
        },
    },
    "detectors": {
        "mantle": {"distance": 77.675},
        "end-cap": {"distance": 77.675},
        "high-resolution": {"distance": 79.05},
        "sans": {"distance": 79.05},
    },
}


def dream(
    high_flux=False, high_resolution=False
) -> dict[str, list[Chopper] | list[Detector]]:
    if high_flux and high_resolution:
        raise ValueError("Select either high flux or high resolution, not both.")
    if high_flux:
        return make_beamline(dream_high_flux)
    if high_resolution:
        # TODO: Add high-resolution configuration
        raise NotImplementedError("High-resolution configuration not yet implemented.")
    raise ValueError("Either high_flux or high_resolution must be True.")
