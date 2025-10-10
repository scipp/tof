# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Pre-configured chopper and detector parameters for Odin.
"""

import copy

from ...chopper import Chopper
from ...detector import Detector
from ...model import make_beamline

odin_non_pulse_skipping = {
    "WFMC_1": {
        "type": "chopper",
        "frequency": {"value": 56.0, "unit": "Hz"},
        "phase": {"value": 93.244, "unit": "deg"},
        "distance": {"value": 6.85, "unit": "m"},
        "open": {
            "value": [-1.9419, 49.5756, 98.9315, 146.2165, 191.5176, 234.9179],
            "unit": "deg",
        },
        "close": {
            "value": [1.9419, 55.7157, 107.2332, 156.5891, 203.8741, 249.1752],
            "unit": "deg",
        },
        "direction": "clockwise",
    },
    "WFMC_2": {
        "type": "chopper",
        "frequency": {"value": 56.0, "unit": "Hz"},
        "phase": {"value": 97.128, "unit": "deg"},
        "distance": {"value": 7.15, "unit": "m"},
        "open": {
            "value": [-1.9419, 51.8318, 103.3493, 152.7052, 199.9903, 245.2914],
            "unit": "deg",
        },
        "close": {
            "value": [1.9419, 57.9719, 111.6510, 163.0778, 212.3468, 259.5486],
            "unit": "deg",
        },
        "direction": "clockwise",
    },
    "FOC_1": {
        "type": "chopper",
        "frequency": {"value": 42.0, "unit": "Hz"},
        "phase": {"value": 81.303297, "unit": "deg"},
        "distance": {"value": 8.4, "unit": "m"},
        "open": {
            "value": [-5.1362, 42.5536, 88.2425, 132.0144, 173.9497, 216.7867],
            "unit": "deg",
        },
        "close": {
            "value": [5.1362, 54.2095, 101.2237, 146.2653, 189.417, 230.7582],
            "unit": "deg",
        },
        "direction": "clockwise",
    },
    "BP_1": {
        "type": "chopper",
        "frequency": {"value": 14.0, "unit": "Hz"},
        "phase": {"value": 31.080 + 14.0, "unit": "deg"},
        "distance": {"value": 8.45, "unit": "m"},
        "open": {"value": [-23.6029], "unit": "deg"},
        "close": {"value": [23.6029], "unit": "deg"},
        "direction": "clockwise",
    },
    "FOC_2": {
        "type": "chopper",
        "frequency": {"value": 42.0, "unit": "Hz"},
        "phase": {"value": 107.013442, "unit": "deg"},
        "distance": {"value": 12.2, "unit": "m"},
        "open": {
            "value": [-16.3227, 53.7401, 120.8633, 185.1701, 246.7787, 307.0165],
            "unit": "deg",
        },
        "close": {
            "value": [16.3227, 86.8303, 154.3794, 218.7551, 280.7508, 340.3188],
            "unit": "deg",
        },
        "direction": "clockwise",
    },
    "BP_2": {
        "type": "chopper",
        "frequency": {"value": 14.0, "unit": "Hz"},
        "phase": {"value": 44.224 + 14.0, "unit": "deg"},
        "distance": {"value": 12.25, "unit": "m"},
        "open": {"value": [-34.4663], "unit": "deg"},
        "close": {"value": [34.4663], "unit": "deg"},
        "direction": "clockwise",
    },
    "T0_alpha": {
        "type": "chopper",
        "frequency": {"value": 14.0, "unit": "Hz"},
        "phase": {"value": 179.672, "unit": "deg"},
        "distance": {"value": 13.5, "unit": "m"},
        "open": {"value": [-167.8986], "unit": "deg"},
        "close": {"value": [167.8986], "unit": "deg"},
        "direction": "clockwise",
    },
    "T0_beta": {
        "type": "chopper",
        "frequency": {"value": 14.0, "unit": "Hz"},
        "phase": {"value": 179.672, "unit": "deg"},
        "distance": {"value": 13.7, "unit": "m"},
        "open": {"value": [-167.8986], "unit": "deg"},
        "close": {"value": [167.8986], "unit": "deg"},
        "direction": "clockwise",
    },
    "FOC_3": {
        "type": "chopper",
        "frequency": {"value": 28.0, "unit": "Hz"},
        "phase": {"value": 92.993, "unit": "deg"},
        "distance": {"value": 17.0, "unit": "m"},
        "open": {
            "value": [-20.302, 45.247, 108.0457, 168.2095, 225.8489, 282.2199],
            "unit": "deg",
        },
        "close": {
            "value": [20.302, 85.357, 147.6824, 207.3927, 264.5977, 319.4024],
            "unit": "deg",
        },
        "direction": "clockwise",
    },
    "FOC_4": {
        "type": "chopper",
        "frequency": {"value": 14.0, "unit": "Hz"},
        "phase": {"value": 61.584, "unit": "deg"},
        "distance": {"value": 23.69, "unit": "m"},
        "open": {
            "value": [-16.7157, 29.1882, 73.1661, 115.2988, 155.6636, 195.5254],
            "unit": "deg",
        },
        "close": {
            "value": [16.7157, 61.8217, 105.0352, 146.4355, 186.0987, 224.0978],
            "unit": "deg",
        },
        "direction": "clockwise",
    },
    "FOC_5": {
        "type": "chopper",
        "frequency": {"value": 14.0, "unit": "Hz"},
        "phase": {"value": 82.581, "unit": "deg"},
        "distance": {"value": 33.0, "unit": "m"},
        "open": {
            "value": [-25.8514, 38.3239, 99.8064, 160.1254, 217.4321, 272.5426],
            "unit": "deg",
        },
        "close": {
            "value": [25.8514, 88.4621, 147.4729, 204.0245, 257.7603, 313.7139],
            "unit": "deg",
        },
        "direction": "clockwise",
    },
    "detector": {"type": "detector", "distance": {"value": 60.5, "unit": "m"}},
}


odin_pulse_skipping = copy.deepcopy(odin_non_pulse_skipping)
odin_pulse_skipping["BP_1"]["frequency"] = {"value": 7.0, "unit": "Hz"}
odin_pulse_skipping["BP_1"]["phase"] = {"value": 31.080, "unit": "deg"}
odin_pulse_skipping["BP_2"]["frequency"] = {"value": 7.0, "unit": "Hz"}
odin_pulse_skipping["BP_2"]["phase"] = {"value": 44.224, "unit": "deg"}


def odin(
    pulse_skipping: bool = False,
) -> dict[str, list[Chopper] | list[Detector]]:
    """
    Return a pre-configured Odin beamline containing choppers and a detector.

    Parameters
    ----------
    pulse_skipping:
        If True, configure the beamline for pulse skipping mode (band-pass choppers will
        rotate at 7 Hz instead of 14 Hz).
    """
    return make_beamline(
        odin_pulse_skipping if pulse_skipping else odin_non_pulse_skipping
    )
