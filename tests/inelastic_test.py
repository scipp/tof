# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc

import tof

Hz = sc.Unit('Hz')
deg = sc.Unit('deg')
meter = sc.Unit('m')


rng = np.random.default_rng(seed=83)


def uniform_deltae(e_i):
    # Uniform sampling between -0.2 and 0.2 meV
    de = sc.array(
        dims=e_i.dims, values=rng.uniform(-0.2, 0.2, size=e_i.shape), unit='meV'
    )
    return e_i.to(unit='meV', copy=False) - de


def double_peak(e_i):
    # Either -0.2 or 0.2 meV
    de = sc.array(
        dims=e_i.dims, values=rng.choice([-0.2, 0.2], size=e_i.shape), unit='meV'
    )
    return e_i.to(unit='meV', copy=False) - de


def normal_deltae(e_i):
    de = sc.array(
        dims=e_i.dims, values=rng.normal(scale=0.05, size=e_i.shape), unit='meV'
    )
    return e_i.to(unit='meV', copy=False) - de


@pytest.mark.parametrize("deltae_func", [uniform_deltae, double_peak, normal_deltae])
def test_inelastic_samples(deltae_func):

    sample = tof.InelasticSample(distance=28.0 * meter, name="sample", func=deltae_func)

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
    assert not sc.allclose(
        res['monitor'].data.coords['wavelength'],
        res['detector'].data.coords['wavelength'],
    )


def test_inelastic_sample_that_has_zero_energy_transfer():
    def zero_deltae(e_i):
        return e_i

    sample = tof.InelasticSample(distance=28.0 * meter, name="sample", func=zero_deltae)

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


def test_inelastic_sample_negative_final_energies_are_dropped():
    rng = np.random.default_rng(seed=86)

    def uniform_deltae(e_i):
        # Uniform sampling between -0.6 and 0.6 meV
        de = sc.array(
            dims=e_i.dims, values=rng.uniform(-0.6, 0.6, size=e_i.shape), unit='meV'
        )
        return e_i.to(unit='meV', copy=False) - de

    sample = tof.InelasticSample(
        distance=28.0 * meter, name="sample", func=uniform_deltae
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

    # Verify that some wavelengths are NaN
    assert sc.all(~sc.isnan(res['monitor'].data.coords['wavelength']))
    assert not sc.all(~sc.isnan(res['detector'].data.coords['wavelength']))


def test_two_inelastic_samples():

    sample1 = tof.InelasticSample(
        distance=28.0 * meter, name="sample1", func=uniform_deltae
    )
    sample2 = tof.InelasticSample(
        distance=32.0 * meter, name="sample2", func=double_peak
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
        tof.Detector(distance=26.0 * meter, name='monitor1'),
        tof.Detector(distance=30.0 * meter, name='monitor2'),
        tof.Detector(distance=34.0 * meter, name='detector'),
    ]

    source = tof.Source(facility='ess', neutrons=500_000, seed=77)

    model = tof.Model(
        source=source, components=choppers + detectors + [sample1, sample2]
    )

    res = model.run()

    assert not sc.allclose(
        res['monitor1'].data.coords['wavelength'],
        res['monitor2'].data.coords['wavelength'],
    )
    assert not sc.allclose(
        res['monitor2'].data.coords['wavelength'],
        res['detector'].data.coords['wavelength'],
    )


def test_inelastic_sample_between_two_choppers():

    sample1 = tof.InelasticSample(
        distance=15.0 * meter, name="sample1", func=uniform_deltae
    )
    sample2 = tof.InelasticSample(
        distance=32.0 * meter, name="sample2", func=double_peak
    )

    choppers = [
        tof.Chopper(
            frequency=70.0 * Hz,
            open=sc.array(dims=['cutout'], values=[0.0], unit='deg'),
            close=sc.array(dims=['cutout'], values=[150.0], unit='deg'),
            phase=60.0 * deg,
            distance=10.0 * meter,
            name="chop1",
        ),
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
        tof.Detector(distance=13.0 * meter, name='m1'),
        tof.Detector(distance=17.0 * meter, name='m2'),
        tof.Detector(distance=30.0 * meter, name='m3'),
        tof.Detector(distance=34.0 * meter, name='m4'),
    ]

    source = tof.Source(facility='ess', neutrons=500_000, seed=77)

    model = tof.Model(
        source=source, components=choppers + detectors + [sample1, sample2]
    )

    res = model.run()

    assert not sc.allclose(
        res['m1'].data.coords['wavelength'],
        res['m2'].data.coords['wavelength'],
    )

    assert not sc.allclose(
        res['m3'].data.coords['wavelength'],
        res['m4'].data.coords['wavelength'],
    )

    # Some neutrons were blocked by the second chopper
    assert res['m3'].data.sum().value < res['m2'].data.sum().value


def test_inelastic_sample_eq():
    def func1(e_i):
        return e_i

    def func2(e_i):
        return 1.0 * e_i

    sample1 = tof.InelasticSample(distance=28.0 * meter, name="sample1", func=func1)
    assert sample1 == sample1
    assert sample1 == tof.InelasticSample(
        distance=28.0 * meter, name="sample1", func=func1
    )
    assert sample1 != tof.InelasticSample(
        distance=28.2 * meter, name="sample1", func=func1
    )
    assert sample1 != tof.InelasticSample(
        distance=28.0 * meter, name="sample12", func=func1
    )
    assert sample1 != tof.InelasticSample(
        distance=28.0 * meter, name="sample1", func=func2
    )
