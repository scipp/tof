# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc

import tof

from .common import (
    dummy_chopper,
    dummy_detector,
    dummy_source,
    make_chopper,
    make_source,
)

Hz = sc.Unit('Hz')
deg = sc.Unit('deg')
meter = sc.Unit('m')
ms = sc.Unit('ms')


def test_one_chopper_one_opening():
    # Make a chopper open from 10-20 ms. Assume zero phase.
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

    # Make a pulse with 3 neutrons with one neutron going through the chopper opening
    # and the other two neutrons on either side of the opening.
    source = make_source(
        arrival_times=sc.concat(
            [0.9 * topen, 0.5 * (topen + tclose), 1.1 * tclose], dim='event'
        ),
        distance=chopper.distance,
    )

    model = tof.Model(source=source, choppers=[chopper], detectors=[detector])
    res = model.run()

    key = 'pulse:0'
    visible = res.choppers['chopper'].tofs.visible.data[key]
    blocked = res.choppers['chopper'].tofs.blocked.data[key]

    assert len(visible) == 1
    assert len(blocked) == 2
    assert sc.isclose(
        visible.coords['tof'][0],
        (0.5 * (topen + tclose)).to(unit='us'),
    )
    assert sc.isclose(blocked.coords['tof'][0], (0.9 * topen).to(unit='us'))
    assert sc.isclose(blocked.coords['tof'][1], (1.1 * tclose).to(unit='us'))

    det = res.detectors['detector'].tofs.visible.data[key]
    assert len(det) == 1
    assert sc.isclose(
        det.coords['tof'][0],
        (
            source.data.coords['wavelength']['pulse', 0][1]
            * detector.distance
            * tof.utils.m_over_h
        ).to(unit='us'),
    )


def test_two_choppers_one_opening():
    # Make a first chopper open from 5-16 ms. Assume zero phase.
    topen = 5.0 * ms
    tclose = 16.0 * ms
    chopper1 = make_chopper(
        topen=[topen],
        tclose=[tclose],
        f=10.0 * Hz,
        phase=0.0 * deg,
        distance=10 * meter,
        name='chopper1',
    )

    # Make a second chopper open from 15-20 ms.
    chopper2 = make_chopper(
        topen=[15.0 * ms],
        tclose=[20.0 * ms],
        f=15.0 * Hz,
        phase=0.0 * deg,
        distance=15 * meter,
        name='chopper2',
    )

    detector = tof.Detector(distance=20 * meter, name='detector')

    # Make a pulse with 3 neutrons with 2 neutrons going through the first chopper
    # opening and only one making it through the second chopper.
    source = make_source(
        arrival_times=sc.concat(
            [1.5 * topen, 0.5 * (topen + tclose), 1.1 * tclose], dim='event'
        ),
        distance=chopper1.distance,
    )

    model = tof.Model(
        source=source, choppers=[chopper1, chopper2], detectors=[detector]
    )
    res = model.run()

    key = 'pulse:0'
    ch1_tofs = res.choppers['chopper1'].tofs.data
    ch2_tofs = res.choppers['chopper2'].tofs.data
    wavs = source.data.coords['wavelength']['pulse', 0]
    det = res.detectors['detector'].tofs.visible.data[key]

    assert len(ch1_tofs['visible'][key]) == 2
    assert len(ch1_tofs['blocked'][key]) == 1
    assert sc.isclose(
        ch1_tofs['visible'][key].coords['tof'][0], (1.5 * topen).to(unit='us')
    )
    assert sc.isclose(
        ch1_tofs['visible'][key].coords['tof'][1],
        (0.5 * (topen + tclose)).to(unit='us'),
    )
    assert sc.isclose(
        ch1_tofs['blocked'][key].coords['tof'][0], (1.1 * tclose).to(unit='us')
    )
    assert len(ch2_tofs['visible'][key]) == 1
    # Blocks only one neutron, the other is blocked by chopper1
    assert len(ch2_tofs['blocked'][key]) == 1
    assert sc.isclose(
        ch2_tofs['visible'][key].coords['tof'][0],
        (wavs[1] * chopper2.distance * tof.utils.m_over_h).to(unit='us'),
    )
    assert sc.isclose(
        ch2_tofs['blocked'][key].coords['tof'][0],
        (wavs[0] * chopper2.distance * tof.utils.m_over_h).to(unit='us'),
    )
    assert len(det) == 1
    assert sc.isclose(
        det.coords['tof'][0],
        (wavs[1] * detector.distance * tof.utils.m_over_h).to(unit='us'),
    )


def test_two_choppers_one_and_two_openings():
    topen = 5.0 * ms
    tclose = 16.0 * ms
    chopper1 = make_chopper(
        topen=[topen],
        tclose=[tclose],
        f=10.0 * Hz,
        phase=0.0 * deg,
        distance=10 * meter,
        name='chopper1',
    )

    chopper2 = make_chopper(
        topen=[9.0 * ms, 15.0 * ms],
        tclose=[12.0 * ms, 20.0 * ms],
        f=15.0 * Hz,
        phase=0.0 * deg,
        distance=15 * meter,
        name='chopper2',
    )

    detector = tof.Detector(distance=20 * meter, name='detector')

    # Make a pulse with 7 neutrons:
    # - 2 neutrons blocked by chopper1
    # - 3 neutrons blocked by chopper2
    # - 2 neutrons detected (one through each of chopper2's openings)
    source = make_source(
        arrival_times=sc.concat(
            [
                0.9 * topen,
                1.01 * topen,
                1.3 * topen,
                1.8 * topen,
                0.5 * (topen + tclose),
                0.85 * tclose,
                1.1 * tclose,
            ],
            dim='event',
        ),
        distance=chopper1.distance,
    )

    model = tof.Model(
        source=source, choppers=[chopper1, chopper2], detectors=[detector]
    )
    res = model.run()

    key = 'pulse:0'
    assert len(res.choppers['chopper1'].tofs.visible.data[key]) == 5
    assert len(res.choppers['chopper1'].tofs.blocked.data[key]) == 2
    assert len(res.choppers['chopper2'].tofs.visible.data[key]) == 2
    assert len(res.choppers['chopper2'].tofs.blocked.data[key]) == 3
    assert len(res.detectors['detector'].tofs.visible.data[key]) == 2


def test_neutron_conservation():
    N = 100_000
    source = tof.Source(facility='ess', neutrons=N)

    chopper1 = make_chopper(
        topen=[5.0 * ms],
        tclose=[16.0 * ms],
        f=10.0 * Hz,
        phase=0.0 * deg,
        distance=10 * meter,
        name='chopper1',
    )
    chopper2 = make_chopper(
        topen=[9.0 * ms, 15.0 * ms],
        tclose=[15.0 * ms, 20.0 * ms],
        f=15.0 * Hz,
        phase=0.0 * deg,
        distance=15 * meter,
        name='chopper2',
    )

    detector = tof.Detector(distance=20 * meter, name='detector')
    model = tof.Model(
        source=source, choppers=[chopper1, chopper2], detectors=[detector]
    )
    res = model.run()

    key = 'pulse:0'
    ch1 = res.choppers['chopper1'].tofs.data
    ch2 = res.choppers['chopper2'].tofs.data

    assert (ch1['visible'][key].sum() + ch1['blocked'][key].sum()).value == N
    assert sc.identical(
        ch2['visible'][key].sum() + ch2['blocked'][key].sum(),
        ch1['visible'][key].sum(),
    )
    assert sc.identical(
        res.detectors['detector'].tofs.data['visible'][key].sum(),
        ch2['visible'][key].sum(),
    )


def test_add_chopper_and_detector():
    chopper = dummy_chopper()
    detector = dummy_detector()
    model = tof.Model(source=dummy_source())
    model.add(chopper)
    assert 'dummy_chopper' in model.choppers
    model.add(detector)
    assert 'dummy_detector' in model.detectors


def test_add_components_with_same_name_raises():
    chopper = dummy_chopper()
    detector = dummy_detector()
    model = tof.Model(source=dummy_source())
    model.add(chopper)
    with pytest.raises(
        KeyError, match='Component with name dummy_chopper already exists'
    ):
        model.add(chopper)
    model.add(detector)
    with pytest.raises(
        KeyError, match='Component with name dummy_detector already exists'
    ):
        model.add(detector)
    detector2 = tof.Detector(distance=22 * meter, name='dummy_chopper')
    with pytest.raises(
        KeyError, match='Component with name dummy_chopper already exists'
    ):
        model.add(detector2)


def test_iter():
    chopper = dummy_chopper()
    detector = dummy_detector()
    model = tof.Model(source=dummy_source())
    model.add(chopper)
    assert 'dummy_chopper' in model.choppers
    model.add(detector)
    assert 'dummy_detector' in model.detectors


def test_remove():
    chopper = dummy_chopper()
    detector = dummy_detector()
    model = tof.Model(source=dummy_source(), choppers=[chopper], detectors=[detector])
    del model['dummy_chopper']
    assert 'dummy_chopper' not in model
    assert 'dummy_detector' in model
    del model['dummy_detector']
    assert 'dummy_detector' not in model


def test_getitem():
    chopper = dummy_chopper()
    detector = dummy_detector()
    model = tof.Model(source=dummy_source(), choppers=[chopper], detectors=[detector])
    assert model['dummy_chopper'] is chopper
    assert model['dummy_detector'] is detector
    with pytest.raises(KeyError, match='No component with name foo'):
        model['foo']


def test_input_can_be_single_component():
    chopper = dummy_chopper()
    detector = dummy_detector()
    model = tof.Model(source=dummy_source(), choppers=chopper, detectors=detector)
    assert 'dummy_chopper' in model.choppers
    assert 'dummy_detector' in model.detectors


def test_bad_input_type_raises():
    chopper = dummy_chopper()
    detector = dummy_detector()
    with pytest.raises(TypeError, match='Invalid input type'):
        _ = tof.Model(source=dummy_source(), choppers='bad chopper')
    with pytest.raises(TypeError, match='Invalid input type'):
        _ = tof.Model(source=dummy_source(), choppers=[chopper], detectors='abc')
    with pytest.raises(TypeError, match='Invalid input type'):
        _ = tof.Model(source=dummy_source(), choppers=[chopper, 'bad chopper'])
    with pytest.raises(TypeError, match='Invalid input type'):
        _ = tof.Model(source=dummy_source(), detectors=(1234, detector))
    with pytest.raises(TypeError, match='Invalid input type'):
        _ = tof.Model(source=dummy_source(), choppers=[detector])
    with pytest.raises(TypeError, match='Invalid input type'):
        _ = tof.Model(source=dummy_source(), detectors=[chopper])
