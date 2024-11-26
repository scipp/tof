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
    visible = res.choppers['chopper'].toas.visible.data[key]
    blocked = res.choppers['chopper'].toas.blocked.data[key]

    assert len(visible) == 1
    assert len(blocked) == 2
    assert sc.isclose(
        visible.coords['toa'][0],
        (0.5 * (topen + tclose)).to(unit='us'),
    )
    assert sc.isclose(blocked.coords['toa'][0], (0.9 * topen).to(unit='us'))
    assert sc.isclose(blocked.coords['toa'][1], (1.1 * tclose).to(unit='us'))

    det = res.detectors['detector'].toas.visible.data[key]
    assert len(det) == 1
    assert sc.isclose(
        det.coords['toa'][0],
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
    ch1_toas = res.choppers['chopper1'].toas.data
    ch2_toas = res.choppers['chopper2'].toas.data
    wavs = source.data.coords['wavelength']['pulse', 0]
    det = res.detectors['detector'].toas.visible.data[key]

    assert len(ch1_toas['visible'][key]) == 2
    assert len(ch1_toas['blocked'][key]) == 1
    assert sc.isclose(
        ch1_toas['visible'][key].coords['toa'][0], (1.5 * topen).to(unit='us')
    )
    assert sc.isclose(
        ch1_toas['visible'][key].coords['toa'][1],
        (0.5 * (topen + tclose)).to(unit='us'),
    )
    assert sc.isclose(
        ch1_toas['blocked'][key].coords['toa'][0], (1.1 * tclose).to(unit='us')
    )
    assert len(ch2_toas['visible'][key]) == 1
    # Blocks only one neutron, the other is blocked by chopper1
    assert len(ch2_toas['blocked'][key]) == 1
    assert sc.isclose(
        ch2_toas['visible'][key].coords['toa'][0],
        (wavs[1] * chopper2.distance * tof.utils.m_over_h).to(unit='us'),
    )
    assert sc.isclose(
        ch2_toas['blocked'][key].coords['toa'][0],
        (wavs[0] * chopper2.distance * tof.utils.m_over_h).to(unit='us'),
    )
    assert len(det) == 1
    assert sc.isclose(
        det.coords['toa'][0],
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
    assert len(res.choppers['chopper1'].toas.visible.data[key]) == 5
    assert len(res.choppers['chopper1'].toas.blocked.data[key]) == 2
    assert len(res.choppers['chopper2'].toas.visible.data[key]) == 2
    assert len(res.choppers['chopper2'].toas.blocked.data[key]) == 3
    assert len(res.detectors['detector'].toas.visible.data[key]) == 2


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
    ch1 = res.choppers['chopper1'].toas.data
    ch2 = res.choppers['chopper2'].toas.data

    assert (ch1['visible'][key].sum() + ch1['blocked'][key].sum()).value == N
    assert sc.identical(
        ch2['visible'][key].sum() + ch2['blocked'][key].sum(),
        ch1['visible'][key].sum(),
    )
    assert sc.identical(
        res.detectors['detector'].toas.data['visible'][key].sum(),
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


def test_model_repr_does_not_raise():
    N = 10_000
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
    assert repr(model) is not None


def test_component_distance():
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
    monitor = tof.Detector(distance=17 * meter, name='monitor')
    detector = tof.Detector(distance=20 * meter, name='detector')

    # Make a pulse with 3 neutrons with one neutron going through the chopper opening
    # and the other two neutrons on either side of the opening.
    source = make_source(
        arrival_times=sc.concat(
            [0.9 * topen, 0.5 * (topen + tclose), 1.1 * tclose], dim='event'
        ),
        distance=chopper.distance,
    )

    model = tof.Model(source=source, choppers=[chopper], detectors=[monitor, detector])
    res = model.run()

    assert sc.identical(res['monitor'].data.coords['distance'], monitor.distance)
    assert sc.identical(res['detector'].data.coords['distance'], detector.distance)
    assert sc.identical(res['chopper'].data.coords['distance'], chopper.distance)


def test_to_nxevent_data():
    source = tof.Source(facility='ess', neutrons=100_000)
    choppers = [
        tof.Chopper(
            frequency=14.0 * Hz,
            open=sc.array(
                dims=['cutout'],
                values=[0.0],
                unit='deg',
            ),
            close=sc.array(
                dims=['cutout'],
                values=[10.0],
                unit='deg',
            ),
            phase=90.0 * deg,
            distance=8.0 * meter,
            name="chopper",
        )
    ]
    detectors = [
        tof.Detector(distance=26.0 * meter, name='monitor'),
        tof.Detector(distance=32.0 * meter, name='detector'),
    ]
    model = tof.Model(source=source, choppers=choppers, detectors=detectors)
    res = model.run()

    # There should be 1 pulse for monitor data, and 2 pulses for detector data as it
    # wraps around the pulse period.
    for key, npulses in zip(('monitor', 'detector'), (1, 2)):
        nxevent_data = res.to_nxevent_data(key)
        assert sc.identical(res['monitor'].data.sum().data, nxevent_data.sum().data)
        assert nxevent_data.sizes['pulse'] == npulses
        assert nxevent_data.bins.concat().value.coords[
            'event_time_offset'
        ].min() >= sc.scalar(0.0, unit='us')
        assert nxevent_data.bins.concat().value.coords[
            'event_time_offset'
        ].max() <= sc.reciprocal(source.frequency).to(unit='us')
