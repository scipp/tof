# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Pre-configured chopper and detector parameters for Magic.
"""

from ...chopper import Chopper
from ...detector import Detector
from ...model import make_beamline


def magic(
    psc_opening_angle: float = 105, wavelength_band_min: float = 0.5
) -> dict[str, list[Chopper] | list[Detector]]:
    detla_t23 = 0.5 * 2.5 * 0.001  # seconds

    psc1_pos = 6.229
    psc2_pos = 6.244
    sc_pos = 6.735
    bm_bunker_pos_1 = 7.823
    bm_bunker_pos_2 = 7.823
    bc_pos = 79.9
    bm_cave_pos = 157.903
    detector_pos = 160.403

    psc_slit_a = 8.6
    psc_slit_b = 105
    psc_nu = 154
    sc_slit = 20.6
    sc_nu = 14
    bc_slit = 180
    bc_nu = 14

    if psc_opening_angle <= psc_slit_a:
        i_slit = 0
    elif psc_opening_angle <= psc_slit_b:
        i_slit = 1
    else:
        psc_opening_angle = psc_slit_b
        i_slit = 1
        raise UserWarning(
            f"PSC opening angle exceeds maximum value. Set to maximum ({psc_slit_b} deg.)."
        )

    sc_phase = (
        wavelength_band_min * sc_pos / 3956 * sc_nu * 360
        + 0.5 * sc_slit
        + detla_t23 * sc_nu * 360
    )
    bc_phase = (
        wavelength_band_min * bc_pos / 3956 * bc_nu * 360
        + 0.5 * bc_slit
        + detla_t23 * bc_nu * 360
    )

    if i_slit == 0:
        psc_values = 0
        psc1_phase = (
            wavelength_band_min * psc1_pos / 3956 * psc_nu * 360
            + 0.5 * psc_slit_a
            + detla_t23 * psc_nu * 360
        )
        psc2_phase = (
            wavelength_band_min * psc2_pos / 3956 * psc_nu * 360
            + 0.5 * psc_slit_a
            + detla_t23 * psc_nu * 360
        )
    else:
        psc_values = -90
        psc1_phase = (
            wavelength_band_min * psc1_pos / 3956 * psc_nu * 360
            + 0.5 * psc_slit_b
            + detla_t23 * psc_nu * 360
        )
        psc2_phase = (
            wavelength_band_min * psc2_pos / 3956 * psc_nu * 360
            + 0.5 * psc_slit_b
            + detla_t23 * psc_nu * 360
        )

    return make_beamline(
        {
            "PSC1": {
                "type": "chopper",
                "frequency": {"value": psc_nu, "unit": "Hz"},
                "phase": {"value": psc1_phase, "unit": "deg"},
                "distance": {"value": psc1_pos, "unit": "m"},
                "open": {
                    "value": [
                        -0.5 * psc_slit_a + psc_values,
                        90 - 0.5 * psc_slit_b + psc_values,
                    ],
                    "unit": "deg",
                },
                "close": {
                    "value": [
                        0.5 * psc_slit_a + psc_values,
                        90 + 0.5 * psc_slit_b + psc_values,
                    ],
                    "unit": "deg",
                },
                "direction": "clockwise",
            },
            "PSC2": {
                "type": "chopper",
                "frequency": {"value": psc_nu, "unit": "Hz"},
                "phase": {"value": psc2_phase, "unit": "deg"},
                "distance": {"value": psc2_pos, "unit": "m"},
                "open": {
                    "value": [
                        -0.5 * psc_slit_a - psc_values,
                        270 - 0.5 * psc_slit_b - psc_values,
                    ],
                    "unit": "deg",
                },
                "close": {
                    "value": [
                        0.5 * psc_slit_a - psc_values,
                        270 + 0.5 * psc_slit_b - psc_values,
                    ],
                    "unit": "deg",
                },
                "direction": "clockwise",
            },
            "SC": {
                "type": "chopper",
                "frequency": {"value": sc_nu, "unit": "Hz"},
                "phase": {"value": sc_phase, "unit": "deg"},
                "distance": {"value": sc_pos, "unit": "m"},
                "open": {"value": [-0.5 * sc_slit], "unit": "deg"},
                "close": {"value": [0.5 * sc_slit], "unit": "deg"},
                "direction": "clockwise",
            },
            "BC": {
                "type": "chopper",
                "frequency": {"value": bc_nu, "unit": "Hz"},
                "phase": {"value": bc_phase, "unit": "deg"},
                "distance": {"value": bc_pos, "unit": "m"},
                "open": {"value": [-0.5 * bc_slit], "unit": "deg"},
                "close": {"value": [0.5 * bc_slit], "unit": "deg"},
                "direction": "clockwise",
            },
            "Bunker BM-1": {
                "type": "detector",
                "distance": {"value": bm_bunker_pos_1, "unit": "m"},
            },
            "Bunker BM-2": {
                "type": "detector",
                "distance": {"value": bm_bunker_pos_2, "unit": "m"},
            },
            "Cave BM": {
                "type": "detector",
                "distance": {"value": bm_cave_pos, "unit": "m"},
            },
            "Detector": {
                "type": "detector",
                "distance": {"value": detector_pos, "unit": "m"},
            },
        }
    )
