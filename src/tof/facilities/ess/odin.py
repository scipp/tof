# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Pre-configured chopper and detector parameters for Odin.
"""

import copy

from ...chopper import Chopper
from ...detector import Detector
from ..common import make_beamline

odin_non_pulse_skipping = {
    "choppers": {
        "WFMC_1": {
            "frequency": 56.0,
            "phase": 93.244,
            "distance": 6.85,
            "open": [-1.9419, 49.5756, 98.9315, 146.2165, 191.5176, 234.9179],
            "close": [1.9419, 55.7157, 107.2332, 156.5891, 203.8741, 249.1752],
            "direction": "clockwise",
        },
        "WFMC_2": {
            "frequency": 56.0,
            "phase": 97.128,
            "distance": 7.15,
            "open": [-1.9419, 51.8318, 103.3493, 152.7052, 199.9903, 245.2914],
            "close": [1.9419, 57.9719, 111.6510, 163.0778, 212.3468, 259.5486],
            "direction": "clockwise",
        },
        "FOC_1": {
            "frequency": 42.0,
            "phase": 81.303297,
            "distance": 8.4,
            "open": [-5.1362, 42.5536, 88.2425, 132.0144, 173.9497, 216.7867],
            "close": [5.1362, 54.2095, 101.2237, 146.2653, 189.417, 230.7582],
            "direction": "clockwise",
        },
        "BP_1": {
            "frequency": 14.0,
            "phase": 31.080 + 14.0,
            "distance": 8.45,
            "open": [-23.6029],
            "close": [23.6029],
            "direction": "clockwise",
        },
        "FOC_2": {
            "frequency": 42.0,
            "phase": 107.013442,
            "distance": 12.2,
            "open": [-16.3227, 53.7401, 120.8633, 185.1701, 246.7787, 307.0165],
            "close": [16.3227, 86.8303, 154.3794, 218.7551, 280.7508, 340.3188],
            "direction": "clockwise",
        },
        "BP_2": {
            "frequency": 14.0,
            "phase": 44.224 + 14.0,
            "distance": 12.25,
            "open": [-34.4663],
            "close": [34.4663],
            "direction": "clockwise",
        },
        "T0_alpha": {
            "frequency": 14.0,
            "phase": 179.672,
            "distance": 13.5,
            "open": [-167.8986],
            "close": [167.8986],
            "direction": "clockwise",
        },
        "T0_beta": {
            "frequency": 14.0,
            "phase": 179.672,
            "distance": 13.7,
            "open": [-167.8986],
            "close": [167.8986],
            "direction": "clockwise",
        },
        "FOC_3": {
            "frequency": 28.0,
            "phase": 92.993,
            "distance": 17.0,
            "open": [-20.302, 45.247, 108.0457, 168.2095, 225.8489, 282.2199],
            "close": [20.302, 85.357, 147.6824, 207.3927, 264.5977, 319.4024],
            "direction": "clockwise",
        },
        "FOC_4": {
            "frequency": 14.0,
            "phase": 61.584,
            "distance": 23.69,
            "open": [-16.7157, 29.1882, 73.1661, 115.2988, 155.6636, 195.5254],
            "close": [16.7157, 61.8217, 105.0352, 146.4355, 186.0987, 224.0978],
            "direction": "clockwise",
        },
        "FOC_5": {
            "frequency": 14.0,
            "phase": 82.581,
            "distance": 33.0,
            "open": [-25.8514, 38.3239, 99.8064, 160.1254, 217.4321, 272.5426],
            "close": [25.8514, 88.4621, 147.4729, 204.0245, 257.7603, 313.7139],
            "direction": "clockwise",
        },
    },
    "detectors": {"detector": {"distance": 60.5}},
}


odin_pulse_skipping = copy.deepcopy(odin_non_pulse_skipping)
odin_pulse_skipping["choppers"]["BP_1"]["frequency"] = 7.0
odin_pulse_skipping["choppers"]["BP_1"]["phase"] = 31.080
odin_pulse_skipping["choppers"]["BP_2"]["frequency"] = 7.0
odin_pulse_skipping["choppers"]["BP_2"]["phase"] = 44.224


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
