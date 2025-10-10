# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Pre-configured chopper and detector parameters for Dream.
"""

from ...chopper import Chopper
from ...detector import Detector
from ...model import make_beamline

dream_high_flux = {
    "PSC1": {
        "type": "chopper",
        "frequency": {"value": 14.0, "unit": "Hz"},
        "phase": {"value": 286.0 - 180.0, "unit": "deg"},
        "distance": {"value": 6.145, "unit": "m"},
        "centers": {
            "value": [0, 72, 86.4, 115.2, 172.8, 273.6, 288.0, 302.4],
            "unit": "deg",
        },
        "widths": {
            "value": [2.46, 3.02, 3.27, 3.27, 5.02, 3.93, 3.93, 2.46],
            "unit": "deg",
        },
        "direction": "anti-clockwise",
    },
    "PSC2": {
        "type": "chopper",
        "frequency": {"value": 14.0, "unit": "Hz"},
        "phase": {"value": 236.0, "unit": "deg"},
        "distance": {"value": 6.155, "unit": "m"},
        "centers": {
            "value": [0, 28.8, 57.6, 144.0, 158.4, 216.0, 259.2, 316.8],
            "unit": "deg",
        },
        "widths": {
            "value": [2.46, 3.60, 3.60, 3.23, 3.27, 3.77, 3.94, 2.62],
            "unit": "deg",
        },
        "direction": "clockwise",
    },
    "OC": {
        "type": "chopper",
        "frequency": {"value": 14.0, "unit": "Hz"},
        "phase": {"value": 297.0 - 180.0 - 90.0, "unit": "deg"},
        "distance": {"value": 6.174, "unit": "m"},
        "centers": {"value": [0.0], "unit": "deg"},
        "widths": {"value": [27.6], "unit": "deg"},
        "direction": "anti-clockwise",
    },
    "BC": {
        "type": "chopper",
        "frequency": {"value": 112.0, "unit": "Hz"},
        "phase": {"value": 240.0 - 180.0, "unit": "deg"},
        "distance": {"value": 9.78, "unit": "m"},
        "centers": {"value": [0.0, 180.0], "unit": "deg"},
        "widths": {"value": [73.75, 73.75], "unit": "deg"},
        "direction": "anti-clockwise",
    },
    "T0": {
        "type": "chopper",
        "frequency": {"value": 28.0, "unit": "Hz"},
        "phase": {"value": 280.0 - 180.0, "unit": "deg"},
        "distance": {"value": 13.05, "unit": "m"},
        "centers": {"value": [0.0], "unit": "deg"},
        "widths": {"value": [314.9], "unit": "deg"},
        "direction": "anti-clockwise",
    },
    "mantle": {"type": "detector", "distance": {"value": 77.675, "unit": "m"}},
    "end-cap": {"type": "detector", "distance": {"value": 77.675, "unit": "m"}},
    "high-resolution": {"type": "detector", "distance": {"value": 79.05, "unit": "m"}},
    "sans": {"type": "detector", "distance": {"value": 79.05, "unit": "m"}},
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
