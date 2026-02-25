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
    # delta_e = sc.DataArray(
    #     data=sc.zeros(sizes={'e': 10}),
    #     coords={'e': sc.linspace('e', -0.2, 0.2, 10, unit='meV')},
    # )
    # delta_e.values[[0, -1]] = 1.0
    # sample = tof.InelasticSample(distance=28.0 * meter, name="sample", delta_e=delta_e)

    # json_dict = sample.as_json()
    # assert json_dict['type'] == 'inelastic_sample'
    # assert json_dict['name'] == 'sample'
    # assert json_dict['distance']['value'] == 28.0
    # assert json_dict['distance']['unit'] == 'm'
    # assert json_dict['delta_e']['values'][0] == 1.0
    # assert json_dict['delta_e']['values'][-1] == 1.0
    # assert json_dict['delta_e']['unit'] == 'meV'

    # sample_from_json = tof.InelasticSample.from_json(
    #     name=json_dict['name'], params=json_dict
    # )
    # assert sc.identical(sample.distance, sample_from_json.distance)
    # assert sc.identical(sample.delta_e, sample_from_json.delta_e)
    return
