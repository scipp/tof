# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import FrozenInstanceError

import pytest
import scipp as sc

import tof

from .common import make_chopper, make_pulse

Hz = sc.Unit('Hz')
deg = sc.Unit('deg')
meter = sc.Unit('m')
ms = sc.Unit('ms')


def test_pulse_results_are_read_only():
    topen = 10.0 * ms
    tclose = 20.0 * ms
    chopper = make_chopper(
        topen=[topen],
        tclose=[tclose],
        f=10.0 * Hz,
        phase=0.0 * deg,
        distance=10 * meter,
        name='chopper',
    )
    detector = tof.Detector(distance=20 * meter, name='detector')

    pulse = make_pulse(
        arrival_times=sc.concat(
            [0.9 * topen, 0.5 * (topen + tclose), 1.1 * tclose], dim='event'
        ),
        distance=chopper.distance,
    )

    model = tof.Model(pulse=pulse, choppers=[chopper], detectors=[detector])
    res = model.run()

    # Check that basic properties are accessible
    assert res.pulse.neutrons == 3
    assert sc.identical(res.pulse.wavelengths, pulse.wavelengths)
    # Check that values cannot be overwritten
    with pytest.raises(FrozenInstanceError, match='cannot assign to field'):
        res.pulse.neutrons = 5
    with pytest.raises(FrozenInstanceError, match='cannot assign to field'):
        res.pulse.wavelengths = 'corrupted'
    with pytest.raises(FrozenInstanceError, match='cannot assign to field'):
        res.pulse.birth_times = sc.ones_like(res.pulse.birth_times)


def test_chopper_results_are_read_only():
    topen = 10.0 * ms
    tclose = 20.0 * ms
    chopper = make_chopper(
        topen=[topen],
        tclose=[tclose],
        f=10.0 * Hz,
        phase=0.0 * deg,
        distance=10 * meter,
        name='chopper',
    )
    detector = tof.Detector(distance=20 * meter, name='detector')

    pulse = make_pulse(
        arrival_times=sc.concat(
            [0.9 * topen, 0.5 * (topen + tclose), 1.1 * tclose], dim='event'
        ),
        distance=chopper.distance,
    )

    model = tof.Model(pulse=pulse, choppers=[chopper], detectors=[detector])
    res = model.run()
    ch = res.choppers['chopper']

    # Check that basic properties are accessible
    assert len(ch.tofs.visible) == 1
    assert len(ch.tofs.blocked) == 2
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


def test_detector_results_are_read_only():
    topen = 10.0 * ms
    tclose = 20.0 * ms
    chopper = make_chopper(
        topen=[topen],
        tclose=[tclose],
        f=10.0 * Hz,
        phase=0.0 * deg,
        distance=10 * meter,
        name='chopper',
    )
    detector = tof.Detector(distance=20 * meter, name='detector')

    pulse = make_pulse(
        arrival_times=sc.concat(
            [0.9 * topen, 0.5 * (topen + tclose), 1.1 * tclose], dim='event'
        ),
        distance=chopper.distance,
    )

    model = tof.Model(pulse=pulse, choppers=[chopper], detectors=[detector])
    res = model.run()

    det = res.detectors['detector']

    # Check that basic properties are accessible
    assert len(det.tofs.visible) == 1
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
