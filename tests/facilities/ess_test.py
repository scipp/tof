# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)


import pytest

import tof


def test_run_dream_model_high_flux():
    source = tof.Source(facility='ess', neutrons=100_000, pulses=2)
    dream_params = tof.facilities.ess.dream(high_flux=True)
    model = tof.Model(source=source, **dream_params)
    results = model.run()
    assert len(results.choppers) > 0
    assert len(results.detectors) > 0


def test_run_dream_model_high_resolution():
    source = tof.Source(facility='ess', neutrons=100_000, pulses=2)
    dream_params = tof.facilities.ess.dream(high_resolution=True)
    model = tof.Model(source=source, **dream_params)
    results = model.run()
    assert len(results.choppers) > 0
    assert len(results.detectors) > 0


@pytest.mark.parametrize("pulse_skipping", [True, False])
def test_run_odin_model(pulse_skipping):
    source = tof.Source(facility='ess', neutrons=100_000, pulses=2)
    odin_params = tof.facilities.ess.odin(pulse_skipping=pulse_skipping)
    model = tof.Model(source=source, **odin_params)
    results = model.run()
    assert len(results.choppers) > 0
    assert len(results.detectors) > 0


@pytest.mark.parametrize("pulse_skipping", [True, False])
def test_run_odin_model_with_odin_source(pulse_skipping):
    source = tof.Source(facility='ess-odin', neutrons=100_000, pulses=2)
    odin_params = tof.facilities.ess.odin(pulse_skipping=pulse_skipping)
    model = tof.Model(source=source, **odin_params)
    results = model.run()
    assert len(results.choppers) > 0
    assert len(results.detectors) > 0
