# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import scipp as sc

import tof
from tof.utils import wavelength_to_energy

Hz = sc.Unit('Hz')
deg = sc.Unit('deg')
meter = sc.Unit('m')


def test_inelastic_sample_flat_distribution():
    rng = np.random.default_rng(seed=83)

    def uniform_deltae(e_i):
        # Uniform sampling between -0.2 and 0.2 meV
        de = sc.array(
            dims=e_i.dims, values=rng.uniform(-0.2, 0.2, size=e_i.shape), unit='meV'
        )
        return e_i.to(unit='meV', copy=False) - de

    sample = tof.InelasticSample(
        distance=28.0 * meter,
        name="sample",
        delta_e=uniform_deltae,
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


def test_inelastic_sample_double_peaked_distribution():
    rng = np.random.default_rng(seed=84)

    def double_peak(e_i):
        # Either -0.2 or 0.2 meV
        de = sc.array(
            dims=e_i.dims, values=rng.choice([-0.2, 0.2], size=e_i.shape), unit='meV'
        )
        return e_i.to(unit='meV', copy=False) - de

    sample = tof.InelasticSample(
        distance=28.0 * meter, name="sample", delta_e=double_peak
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


def test_inelastic_sample_normal_distribution():
    rng = np.random.default_rng(seed=85)

    def normal_deltae(e_i):
        de = sc.array(
            dims=e_i.dims, values=rng.normal(scale=0.05, size=e_i.shape), unit='meV'
        )
        return e_i.to(unit='meV', copy=False) - de

    sample = tof.InelasticSample(
        distance=28.0 * meter,
        name="sample",
        delta_e=normal_deltae,
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
    def zero_deltae(e_i):
        return e_i

    sample = tof.InelasticSample(
        distance=28.0 * meter,
        name="sample",
        delta_e=zero_deltae,
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


def test_inelastic_sample_final_energy_is_positive():
    rng = np.random.default_rng(seed=86)

    def uniform_deltae(e_i):
        # Uniform sampling between -0.6 and 0.6 meV
        de = sc.array(
            dims=e_i.dims, values=rng.uniform(-0.6, 0.6, size=e_i.shape), unit='meV'
        )
        return e_i.to(unit='meV', copy=False) - de

    sample = tof.InelasticSample(
        distance=28.0 * meter,
        name="sample",
        delta_e=uniform_deltae,
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
    res = model.run()

    final_energy = wavelength_to_energy(res['detector'].data.coords['wavelength'])

    assert sc.all(final_energy > sc.scalar(0.0, unit='meV'))
    assert sc.isclose(final_energy.min(), sc.scalar(1.0e-30, unit='meV'))


def test_inelastic_sample_as_json():
    sample = tof.InelasticSample(
        distance=28.0 * meter, name="sample1", delta_e=lambda x: x
    )

    json_dict = sample.as_json()
    assert json_dict['type'] == 'inelastic_sample'
    assert json_dict['name'] == 'sample1'
    assert json_dict['distance']['value'] == 28.0
    assert json_dict['distance']['unit'] == 'm'


# # TODO: Not implemented yet: how to save a callable to json
# def test_inelastic_sample_from_json():
#     pass
