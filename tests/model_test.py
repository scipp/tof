# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
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

    toa = res.choppers['chopper'].toa.data
    assert toa.sum().value == 1
    assert toa.masks['blocked_by_me'].sum().value == 2
    assert np.array_equal(
        toa.masks['blocked_by_me'].squeeze().values, [True, False, True]
    )
    assert sc.allclose(
        toa.coords['toa']['pulse', 0],
        (
            source.data.coords['wavelength']['pulse', 0]
            * chopper.distance
            * tof.utils.m_over_h
        ).to(unit='us'),
    )

    toa = res.detectors['detector'].toa.data
    assert toa.sum().value == 1
    assert toa.masks['blocked_by_others'].sum().value == 2
    assert sc.allclose(
        toa.coords['toa']['pulse', 0],
        (
            source.data.coords['wavelength']['pulse', 0]
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

    ch1_toas = res.choppers['chopper1'].toa.data
    ch2_toas = res.choppers['chopper2'].toa.data
    det = res.detectors['detector'].toa.data

    # Blocks the third neutron
    assert ch1_toas.sum().value == 2
    assert ch1_toas.masks['blocked_by_me'].sum().value == 1
    assert ch1_toas.masks['blocked_by_others'].sum().value == 0
    assert np.array_equal(
        ch1_toas.masks['blocked_by_me'].squeeze().values, [False, False, True]
    )
    assert sc.allclose(
        ch1_toas.coords['toa']['pulse', 0],
        (
            source.data.coords['wavelength']['pulse', 0]
            * chopper1.distance
            * tof.utils.m_over_h
        ).to(unit='us'),
    )

    # Blocks the first neutron
    assert ch2_toas.sum().value == 1
    assert ch2_toas.masks['blocked_by_me'].sum().value == 1
    assert ch2_toas.masks['blocked_by_others'].sum().value == 1
    assert np.array_equal(
        ch2_toas.masks['blocked_by_me'].squeeze().values, [True, False, False]
    )
    assert sc.allclose(
        ch2_toas.coords['toa']['pulse', 0],
        (
            source.data.coords['wavelength']['pulse', 0]
            * chopper2.distance
            * tof.utils.m_over_h
        ).to(unit='us'),
    )

    assert det.sum().value == 1
    assert det.masks['blocked_by_others'].sum().value == 2
    assert sc.allclose(
        det.coords['toa']['pulse', 0],
        (
            source.data.coords['wavelength']['pulse', 0]
            * detector.distance
            * tof.utils.m_over_h
        ).to(unit='us'),
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

    assert res.choppers['chopper1'].toa.data.sum().value == 5
    assert res.choppers['chopper1'].toa.data.masks['blocked_by_me'].sum().value == 2
    assert res.choppers['chopper2'].toa.data.sum().value == 2
    assert res.choppers['chopper2'].toa.data.masks['blocked_by_me'].sum().value == 3
    assert res.detectors['detector'].toa.data.sum().value == 2
    assert (
        res.detectors['detector'].toa.data.masks['blocked_by_others'].sum().value == 5
    )


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

    ch1 = res.choppers['chopper1'].toa.data
    ch2 = res.choppers['chopper2'].toa.data
    det = res.detectors['detector'].toa.data

    assert ch1.sizes['event'] == N
    assert ch2.sizes['event'] == N

    assert sc.identical(ch1.masks['blocked_by_me'], ch2.masks['blocked_by_others'])
    assert sc.identical(
        det.masks['blocked_by_others'],
        ch2.masks['blocked_by_me'] | ch1.masks['blocked_by_me'],
    )
    assert det.sum().value + det.masks['blocked_by_others'].sum().value == N


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
        grouped = nxevent_data.group('event_time_zero')
        assert grouped.sizes['event_time_zero'] == npulses
        assert nxevent_data.bins.concat().value.coords[
            'event_time_offset'
        ].min() >= sc.scalar(0.0, unit='us')
        assert nxevent_data.bins.concat().value.coords[
            'event_time_offset'
        ].max() <= sc.reciprocal(source.frequency).to(unit='us')

    # Test when we include all detectors at once
    nxevent_data = res.to_nxevent_data()
    assert nxevent_data.sizes == {'detector_number': 2}
