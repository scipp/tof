# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)


import pytest

import tof


@pytest.mark.parametrize("mode", ['high_flux', 'high_resolution'])
def test_run_dream_model(mode):
    source = tof.Source(facility='ess', neutrons=100_000, pulses=2)

    match mode:
        case 'high_flux':
            params = {"high_flux": True}
        case 'high_resolution':
            params = {"high_resolution": True}

    dream_params = tof.facilities.ess.dream(**params)
    model = tof.Model(source=source, **dream_params)
    results = model.run()
    assert len(results.choppers) > 0
    assert len(results.detectors) > 0


@pytest.mark.parametrize("pulse_skipping", [True, False])
@pytest.mark.parametrize("facility", ['ess', 'ess-odin'])
def test_run_odin_model(pulse_skipping, facility):
    source = tof.Source(facility=facility, neutrons=100_000, pulses=2)
    odin_params = tof.facilities.ess.odin(pulse_skipping=pulse_skipping)
    model = tof.Model(source=source, **odin_params)
    results = model.run()
    assert len(results.choppers) > 0
    assert len(results.detectors) > 0
