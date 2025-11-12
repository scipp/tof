# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import FrozenInstanceError

import pytest
import scipp as sc

import tof

Hz = sc.Unit('Hz')
deg = sc.Unit('deg')
meter = sc.Unit('m')
ms = sc.Unit('ms')
topen = 10.0 * ms
tclose = 20.0 * ms


@pytest.fixture
def chopper(make_chopper):
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
def source(chopper, make_source):
    return make_source(
        arrival_times=sc.concat(
            [0.9 * topen, 0.5 * (topen + tclose), 1.1 * tclose], dim='event'
        ),
        distance=chopper.distance,
    )


@pytest.fixture
def multi_pulse_source(chopper, make_source):
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


def make_ess_model(pulses=1, frequency=None):
    source = tof.Source(facility='ess', neutrons=100_000, pulses=pulses)
    detector = tof.Detector(distance=30.0 * meter, name='detector')
    if frequency is None:
        frequency = 14.0 * Hz
    chopper = tof.Chopper(
        frequency=frequency,
        open=sc.array(
            dims=['cutout'],
            values=[30.0, 50.0],
            unit='deg',
        ),
        close=sc.array(
            dims=['cutout'],
            values=[40.0, 80.0],
            unit='deg',
        ),
        phase=0.0 * deg,
        distance=8 * meter,
        name='chopper',
    )
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
    assert ch.toa.data.sum().value == 1
    assert ch.toa.data.masks['blocked_by_me'].sum().value == 2
    assert sc.identical(ch.distance, chopper.distance)
    assert sc.identical(ch.phase, chopper.phase)
    # Check that values cannot be overwritten
    with pytest.raises(FrozenInstanceError, match='cannot assign to field'):
        ch.phase = 63.0 * deg
    with pytest.raises(FrozenInstanceError, match='cannot assign to field'):
        ch.frequency = 21.0 * Hz
    with pytest.raises(FrozenInstanceError, match='cannot assign to field'):
        ch.toa = [1, 2, 3, 4, 5]
    with pytest.raises(FrozenInstanceError, match='cannot assign to field'):
        ch.toa.data = [1, 2, 3, 4, 5]
    # Check that choppers cannot be added or removed
    with pytest.raises(TypeError, match='object does not support item assignment'):
        res.choppers['chopper2'] = chopper
    with pytest.raises(TypeError, match='object does not support item deletion'):
        del res.choppers['chopper']


def test_detector_results_are_read_only(detector, model):
    res = model.run()
    det = res.detectors['detector']

    # Check that basic properties are accessible
    assert det.toa.data.sum().value == 1
    assert sc.identical(det.distance, detector.distance)
    # Check that values cannot be overwritten
    with pytest.raises(FrozenInstanceError, match='cannot assign to field'):
        det.distance = 55.0 * meter
    with pytest.raises(FrozenInstanceError, match='cannot assign to field'):
        det.wavelength = sc.arange('wavelength', 20.0, unit='angstrom')
    with pytest.raises(FrozenInstanceError, match='cannot assign to field'):
        det.wavelength.data = sc.arange('wavelength', 20.0, unit='angstrom')
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

    for field in ('toa', 'wavelength', 'birth_time', 'speed'):
        assert field in getattr(ch, field).data.coords
        assert field in getattr(det, field).data.coords
        assert 'blocked_by_me' in getattr(ch, field).data.masks
        assert 'blocked_by_others' in getattr(ch, field).data.masks
        assert 'blocked_by_others' in getattr(det, field).data.masks
        assert getattr(ch, field).data.sizes['pulse'] == 3
        assert getattr(det, field).data.sizes['pulse'] == 3


def test_component_data_slicing(chopper, detector, multi_pulse_source):
    model = tof.Model(
        source=multi_pulse_source, choppers=[chopper], detectors=[detector]
    )
    res = model.run()
    ch = res.choppers['chopper']

    toas = ch.toa['pulse', 0].data
    assert 'pulse' not in toas.dims
    assert 'event' in toas.dims


def test_component_results_data_slice_range(chopper, detector, multi_pulse_source):
    model = tof.Model(
        source=multi_pulse_source, choppers=[chopper], detectors=[detector]
    )
    res = model.run()
    ch = res.choppers['chopper']

    toas = ch.toa['pulse', 0:2].data
    assert toas.sizes['pulse'] == 2
    assert 'event' in toas.dims


def test_component_results_data_slice_negative_index(
    chopper, detector, multi_pulse_source
):
    model = tof.Model(
        source=multi_pulse_source, choppers=[chopper], detectors=[detector]
    )
    res = model.run()
    ch = res.choppers['chopper']

    toas = ch.toa['pulse', 0:-1].data
    assert toas.sizes['pulse'] == 2
    assert 'event' in toas.dims


def test_component_results_data_slice_step(chopper, detector, multi_pulse_source):
    model = tof.Model(
        source=multi_pulse_source, choppers=[chopper], detectors=[detector]
    )
    res = model.run()
    ch = res.choppers['chopper']

    toas = ch.toa['pulse', ::2].data
    assert toas.sizes['pulse'] == 2
    assert 'event' in toas.dims


def test_component_results_slice_all_results(chopper, detector, multi_pulse_source):
    model = tof.Model(
        source=multi_pulse_source, choppers=[chopper], detectors=[detector]
    )
    res = model.run()
    ch = res.choppers['chopper']

    reading = ch['pulse', 0]
    toas = reading.toa.data
    assert 'pulse' not in toas.dims
    assert 'event' in toas.dims
    wavelengths = reading.wavelength.data
    assert 'pulse' not in wavelengths.dims
    assert 'event' in wavelengths.dims


def test_component_results_slice_no_pulse_dim(chopper, detector, multi_pulse_source):
    model = tof.Model(
        source=multi_pulse_source, choppers=[chopper], detectors=[detector]
    )
    res = model.run()
    ch = res.choppers['chopper']

    reading = ch[0]  # No pulse dim
    data = reading.data
    assert 'pulse' not in data.dims
    assert 'event' in data.dims

    toas = ch.toa[1].data
    assert 'pulse' not in toas.dims
    assert 'event' in toas.dims


def test_component_results_eto(chopper, detector, multi_pulse_source):
    model = tof.Model(
        source=multi_pulse_source, choppers=[chopper], detectors=[detector]
    )
    res = model.run()
    data = res['detector'].data
    assert 'eto' in data.coords
    assert data.coords['eto'].max() < (1 / model.source.frequency).to(
        unit=data.coords['eto'].unit
    )


def test_result_plot_does_not_raise():
    model = make_ess_model()
    res = model.run()
    res.plot()
    res.plot(visible_rays=5000)
    res.plot(visible_rays=50, blocked_rays=3000)


def test_result_multiple_pulses_plot_does_not_raise():
    model = make_ess_model(pulses=3)
    res = model.run()
    res.plot()
    res.plot(visible_rays=2000)
    res.plot(visible_rays=50, blocked_rays=3000)


def test_result_multiple_pulses_plot_with_no_events_in_last_frame_does_not_raise():
    model = make_ess_model(pulses=3, frequency=10.0 * Hz)
    res = model.run()
    res.plot()
    res.plot(visible_rays=2000)
    res.plot(visible_rays=50, blocked_rays=3000)


def test_result_all_neutrons_blocked_does_not_raise():
    model = make_ess_model(frequency=1.0 * Hz)
    res = model.run()
    res.plot()
    res.plot(blocked_rays=500)


def test_result_plot_cmap_does_not_raise():
    model = make_ess_model()
    res = model.run()
    res.plot(cmap='viridis')


def test_result_plot_no_detectors_does_not_raise():
    model = make_ess_model()
    model._detectors = {}
    res = model.run()
    res.plot()
    res.plot(visible_rays=5000)
    res.plot(visible_rays=50, blocked_rays=3000)


def test_result_repr_does_not_raise():
    model = make_ess_model()
    res = model.run()
    assert repr(res) is not None


def test_component_repr_does_not_raise():
    model = make_ess_model()
    res = model.run()
    assert repr(res.choppers['chopper']) is not None
    assert repr(res.detectors['detector']) is not None


def test_componentdata_repr_does_not_raise():
    model = make_ess_model()
    res = model.run()
    assert repr(res.choppers['chopper'].toa) is not None
    assert repr(res.detectors['detector'].wavelength) is not None


def test_plot_reading_does_not_raise(model):
    res = model.run()
    res.choppers['chopper'].toa.plot()
    res.choppers['chopper'].wavelength.plot()
    res.detectors['detector'].toa.plot()
    res.detectors['detector'].wavelength.plot()


def test_plot_reading_pulse_skipping_does_not_raise():
    model = make_ess_model(pulses=3)
    skip = tof.Chopper(
        frequency=7 * Hz,
        open=sc.array(dims=['cutout'], values=[0.0], unit='deg'),
        close=sc.array(dims=['cutout'], values=[180.0], unit='deg'),
        phase=0.0 * deg,
        distance=10 * meter,
        name='skip',
    )
    model.choppers['skip'] = skip
    res = model.run()
    res.choppers['chopper'].toa.plot()
    res.choppers['chopper'].wavelength.plot()
    res.detectors['detector'].toa.plot()
    res.detectors['detector'].wavelength.plot()


def test_plot_reading_nothing_to_plot_raises():
    model = make_ess_model(pulses=1)
    skip = tof.Chopper(
        frequency=1 * Hz,
        open=sc.array(dims=['cutout'], values=[0.0], unit='deg'),
        close=sc.array(dims=['cutout'], values=[1.0], unit='deg'),
        phase=0.0 * deg,
        distance=10 * meter,
        name='skip',
    )
    model.choppers['skip'] = skip
    res = model.run()
    with pytest.raises(RuntimeError, match="Nothing to plot."):
        res.detectors['detector'].toa.plot()
