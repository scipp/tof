# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import json
import os
import tempfile

import numpy as np
import pytest
import scipp as sc

import tof

Hz = sc.Unit('Hz')
deg = sc.Unit('deg')
meter = sc.Unit('m')
ms = sc.Unit('ms')


def test_inelastic_sample_flat_distribution():
    sample = tof.InelasticSample(
        distance=28.0 * meter,
        name="sample",
        delta_e=sc.DataArray(
            data=sc.ones(sizes={'e': 100}),
            coords={'e': sc.linspace('e', -0.2, 0.2, 100, unit='meV')},
        ),
    )

    choppers = [
        tof.Chopper(
            frequency=70.0 * Hz,
            open=sc.array(dims=['cutout'], values=[0.0], unit='deg'),
            close=sc.array(dims=['cutout'], values=[1.0], unit='deg'),
            phase=0.0 * deg,
            distance=20.0 * meter,
            name="fastchopper",
        ),
    ]

    detectors = [
        tof.Detector(distance=26.0 * meter, name='monitor'),
        tof.Detector(distance=32.0 * meter, name='detector'),
    ]

    source = tof.Source(facility='ess', neutrons=500_000, seed=77)

    model = tof.Model(source=source, components=choppers + detectors + [sample])
    model_no_sample = tof.Model(source=source, components=choppers + detectors)

    res = model.run()
    res_no_sample = model_no_sample.run()

    assert sc.identical(
        res_no_sample['monitor'].data.coords['wavelength'],
        res_no_sample['detector'].data.coords['wavelength'],
    )
    assert not sc.identical(
        res['monitor'].data.coords['wavelength'],
        res['detector'].data.coords['wavelength'],
    )
    assert not sc.allclose(
        res['monitor'].data.coords['wavelength'],
        res['detector'].data.coords['wavelength'],
    )


def test_inelastic_sample_doube_peaked_distribution():
    delta_e = sc.DataArray(
        data=sc.zeros(sizes={'e': 100}),
        coords={'e': sc.linspace('e', -0.2, 0.2, 100, unit='meV')},
    )
    delta_e.values[[0, -1]] = 1.0
    sample = tof.InelasticSample(distance=28.0 * meter, name="sample", delta_e=delta_e)

    choppers = [
        tof.Chopper(
            frequency=70.0 * Hz,
            open=sc.array(dims=['cutout'], values=[0.0], unit='deg'),
            close=sc.array(dims=['cutout'], values=[1.0], unit='deg'),
            phase=0.0 * deg,
            distance=20.0 * meter,
            name="fastchopper",
        ),
    ]

    detectors = [
        tof.Detector(distance=26.0 * meter, name='monitor'),
        tof.Detector(distance=32.0 * meter, name='detector'),
    ]

    source = tof.Source(facility='ess', neutrons=500_000, seed=78)

    model = tof.Model(source=source, components=choppers + detectors + [sample])
    model_no_sample = tof.Model(source=source, components=choppers + detectors)

    res = model.run()
    res_no_sample = model_no_sample.run()

    assert sc.identical(
        res_no_sample['monitor'].data.coords['wavelength'],
        res_no_sample['detector'].data.coords['wavelength'],
    )
    assert not sc.identical(
        res['monitor'].data.coords['wavelength'],
        res['detector'].data.coords['wavelength'],
    )
    assert not sc.allclose(
        res['monitor'].data.coords['wavelength'],
        res['detector'].data.coords['wavelength'],
    )


def test_inelastic_sample_normal_distribution():
    x = sc.linspace('e', -0.2, 0.2, 100, unit='meV')
    sig = sc.scalar(0.03, unit='meV')
    y = 1.0 / (np.sqrt(2.0 * np.pi) * sig) * sc.exp(-((x / sig) ** 2) / 2)
    y.unit = ""

    sample = tof.InelasticSample(
        distance=28.0 * meter,
        name="sample",
        delta_e=sc.DataArray(data=y, coords={'e': x}),
    )

    choppers = [
        tof.Chopper(
            frequency=70.0 * Hz,
            open=sc.array(dims=['cutout'], values=[0.0], unit='deg'),
            close=sc.array(dims=['cutout'], values=[1.0], unit='deg'),
            phase=0.0 * deg,
            distance=20.0 * meter,
            name="fastchopper",
        ),
    ]

    detectors = [
        tof.Detector(distance=26.0 * meter, name='monitor'),
        tof.Detector(distance=32.0 * meter, name='detector'),
    ]

    source = tof.Source(facility='ess', neutrons=500_000, seed=78)

    model = tof.Model(source=source, components=choppers + detectors + [sample])
    model_no_sample = tof.Model(source=source, components=choppers + detectors)

    res = model.run()
    res_no_sample = model_no_sample.run()

    assert sc.identical(
        res_no_sample['monitor'].data.coords['wavelength'],
        res_no_sample['detector'].data.coords['wavelength'],
    )
    assert not sc.identical(
        res['monitor'].data.coords['wavelength'],
        res['detector'].data.coords['wavelength'],
    )
    assert not sc.allclose(
        res['monitor'].data.coords['wavelength'],
        res['detector'].data.coords['wavelength'],
    )


def test_inelastic_sample_that_has_zero_delta_e():
    sample = tof.InelasticSample(
        distance=28.0 * meter,
        name="sample",
        delta_e=sc.DataArray(
            data=sc.array(dims=['e'], values=[1.0], unit=''),
            coords={'e': sc.array(dims=['e'], values=[0.0], unit='meV')},
        ),
    )

    choppers = [
        tof.Chopper(
            frequency=70.0 * Hz,
            open=sc.array(dims=['cutout'], values=[0.0], unit='deg'),
            close=sc.array(dims=['cutout'], values=[1.0], unit='deg'),
            phase=0.0 * deg,
            distance=20.0 * meter,
            name="fastchopper",
        ),
    ]

    detectors = [
        tof.Detector(distance=26.0 * meter, name='monitor'),
        tof.Detector(distance=32.0 * meter, name='detector'),
    ]

    source = tof.Source(facility='ess', neutrons=500_000, seed=78)

    model = tof.Model(source=source, components=choppers + detectors + [sample])
    model_no_sample = tof.Model(source=source, components=choppers + detectors)

    res = model.run()
    res_no_sample = model_no_sample.run()

    assert sc.identical(
        res_no_sample['monitor'].data.coords['wavelength'],
        res_no_sample['detector'].data.coords['wavelength'],
    )
    assert sc.allclose(
        res['monitor'].data.coords['wavelength'],
        res['detector'].data.coords['wavelength'],
    )


def test_inelastic_sample_as_json():
    p = np.array([0.4, 0.45, 0.7, 1.0, 0.7, 0.45, 0.4])
    e = np.array([-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5])

    delta_e = sc.DataArray(
        data=sc.array(dims=['e'], values=p),
        coords={'e': sc.array(dims=['e'], values=e, unit='meV')},
    )
    sample = tof.InelasticSample(
        distance=28.0 * meter, name="sample1", delta_e=delta_e, seed=66
    )

    json_dict = sample.as_json()
    assert json_dict['type'] == 'inelastic_sample'
    assert json_dict['name'] == 'sample1'
    assert json_dict['distance']['value'] == 28.0
    assert json_dict['distance']['unit'] == 'm'
    assert np.array_equal(json_dict['probabilities']['value'], p / p.sum())
    assert json_dict['probabilities']['unit'] == 'dimensionless'
    assert np.array_equal(json_dict['energies']['value'], e)
    assert json_dict['energies']['unit'] == 'meV'
    assert json_dict['seed'] == 66


def test_inelastic_sample_from_json():
    p = np.array([0.4, 0.7, 1.0, 0.7, 0.4])
    e = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    json_dict = {
        'type': 'inelastic_sample',
        'distance': {'value': 28.0, 'unit': 'm'},
        'name': 'sample1',
        'energies': {'value': e, 'unit': 'meV'},
        'probabilities': {'value': p, 'unit': ''},
        'seed': 78,
    }
    sample = tof.InelasticSample.from_json(name=json_dict['name'], params=json_dict)

    assert sample.distance.value == 28.0
    assert sample.distance.unit == 'm'
    assert sample.name == 'sample1'
    assert np.array_equal(sample.energies.values, e)
    assert sample.energies.unit == 'meV'
    assert np.array_equal(sample.probabilities.values, p / p.sum())
    assert sample.probabilities.unit == 'dimensionless'
    assert sample.seed == 78
