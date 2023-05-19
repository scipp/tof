# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import FrozenInstanceError

import pytest
import scipp as sc

import tof

from .common import make_chopper, make_source

Hz = sc.Unit('Hz')
deg = sc.Unit('deg')
meter = sc.Unit('m')
ms = sc.Unit('ms')
topen = 10.0 * ms
tclose = 20.0 * ms


@pytest.fixture
def chopper():
    return make_chopper(
        topen=[topen],
        tclose=[tclose],
        f=10.0 * Hz,
        phase=0.0 * deg,
        distance=10 * meter,
        name='chopper',
    )


@pytest.fixture
def detector():
    return tof.Detector(distance=20 * meter, name='detector')


@pytest.fixture
def source(chopper):
    return make_source(
        arrival_times=sc.concat(
            [0.9 * topen, 0.5 * (topen + tclose), 1.1 * tclose], dim='event'
        ),
        distance=chopper.distance,
    )


@pytest.fixture
def multi_pulse_source(chopper):
    return make_source(
        arrival_times=sc.concat(
            [0.9 * topen, 0.5 * (topen + tclose), 1.1 * tclose], dim='event'
        ),
        distance=chopper.distance,
        pulses=3,
        frequency=14.0 * Hz,
    )


@pytest.fixture
def model(chopper, detector, source):
    return tof.Model(source=source, choppers=[chopper], detectors=[detector])


def test_source_results_are_read_only(source, model):
    res = model.run()

    # Check that basic properties are accessible
    assert res.source.neutrons == 3
    assert sc.identical(
        res.source.data['pulse', 0].coords['wavelength'],
        source.data['pulse', 0].coords['wavelength'],
    )
    # Check that values cannot be overwritten
    with pytest.raises(FrozenInstanceError, match='cannot assign to field'):
        res.source.neutrons = 5
    with pytest.raises(FrozenInstanceError, match='cannot assign to field'):
        res.source.data = 'corrupted'


def test_chopper_results_are_read_only(chopper, model):
    res = model.run()
    ch = res.choppers['chopper']

    # Check that basic properties are accessible
    assert len(ch.tofs.visible[0]) == 1
    assert len(ch.tofs.blocked[0]) == 2
    assert sc.identical(ch.distance, chopper.distance)
    assert sc.identical(ch.phase, chopper.phase)
    # Check that values cannot be overwritten
    with pytest.raises(FrozenInstanceError, match='cannot assign to field'):
        ch.phase = 63.0 * deg
    with pytest.raises(FrozenInstanceError, match='cannot assign to field'):
        ch.frequency = 21.0 * Hz
    with pytest.raises(FrozenInstanceError, match='cannot assign to field'):
        ch.tofs = [1, 2, 3, 4, 5]
    # Check that choppers cannot be added or removed
    with pytest.raises(TypeError, match='object does not support item assignment'):
        res.choppers['chopper2'] = chopper
    with pytest.raises(TypeError, match='object does not support item deletion'):
        del res.choppers['chopper']


def test_detector_results_are_read_only(detector, model):
    res = model.run()
    det = res.detectors['detector']

    # Check that basic properties are accessible
    assert len(det.tofs.visible[0]) == 1
    assert sc.identical(det.distance, detector.distance)
    # Check that values cannot be overwritten
    with pytest.raises(FrozenInstanceError, match='cannot assign to field'):
        det.distance = 55.0 * meter
    with pytest.raises(FrozenInstanceError, match='cannot assign to field'):
        det.wavelengths = sc.arange('wavelength', 20.0, unit='angstrom')
    # Check that detectors cannot be added or removed
    with pytest.raises(TypeError, match='object does not support item assignment'):
        res.detectors['detector2'] = detector
    with pytest.raises(TypeError, match='object does not support item deletion'):
        del res.detectors['detector']


def test_component_results_data_access(chopper, detector, multi_pulse_source):
    model = tof.Model(
        source=multi_pulse_source, choppers=[chopper], detectors=[detector]
    )
    res = model.run()
    ch = res.choppers['chopper']
    det = res.detectors['detector']

    for field in ('tofs', 'wavelengths', 'birth_times', 'speeds'):
        assert 'visible' in getattr(ch, field).data
        assert 'blocked' in getattr(ch, field).data
        assert 'visible' in getattr(det, field).data
        assert 'blocked' not in getattr(det, field).data

    assert list(ch.tofs.visible.data.keys()) == [f'pulse:{i}' for i in range(3)]
    assert list(det.wavelengths.visible.data.keys()) == [f'pulse:{i}' for i in range(3)]


def test_component_results_data_slicing(chopper, detector, multi_pulse_source):
    model = tof.Model(
        source=multi_pulse_source, choppers=[chopper], detectors=[detector]
    )
    res = model.run()
    ch = res.choppers['chopper']

    tofs = ch.tofs[0]
    assert 'visible' in tofs.data
    assert 'blocked' in tofs.data
    vis = ch.tofs.visible[1]
    assert sc.identical(vis.data, ch.tofs.visible.data['pulse:1'])
