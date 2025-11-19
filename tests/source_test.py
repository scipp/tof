# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)


import numpy as np
import pytest
import scipp as sc

import tof


def test_ess_pulse():
    source = tof.Source(facility='ess', neutrons=100_000)
    # Check that the time distribution is low on edges and high in the middle
    times = source.data['pulse', 0].hist(birth_time=300)
    mean = times.mean()
    assert (times[0] < 0.5 * mean).value
    assert (times[-1] < 0.5 * mean).value
    assert (times[150] > 1.5 * mean).value
    # Check that there are more neutrons at low wavelengths
    wavs = source.data['pulse', 0].hist(wavelength=300)
    assert (wavs[:150].sum() > 1.5 * wavs[150:].sum()).value


@pytest.mark.parametrize('facility', ['ESS', 'Ess'])
def test_ess_pulse_uppercase(facility):
    source = tof.Source(facility=facility, neutrons=100_000)
    assert source.neutrons == 100_000
    assert sc.identical(source.frequency, sc.scalar(14.0, unit="Hz"))
    assert source.pulses == 1
    assert source.facility == 'ess'


def test_creation_from_neutrons():
    birth_times = sc.array(dims=['event'], values=[1000.0, 1500.0, 2000.0], unit='us')
    wavelengths = sc.array(dims=['event'], values=[1.0, 5.0, 10.0], unit='angstrom')
    source = tof.Source.from_neutrons(
        birth_times=birth_times,
        wavelengths=wavelengths,
    )
    assert source.neutrons == 3
    assert sc.identical(source.data['pulse', 0].coords['birth_time'], birth_times)
    assert sc.identical(source.data['pulse', 0].coords['wavelength'], wavelengths)


def test_creation_from_distribution_flat():
    birth_time = sc.linspace('birth_time', 1.0, 3.0, 100, unit='ms')
    p_time = sc.DataArray(
        data=sc.ones(sizes=birth_time.sizes),
        coords={'birth_time': birth_time},
    )
    wavelength = sc.linspace('wavelength', 1.0, 10.0, 100, unit='angstrom')
    p_wav = sc.DataArray(
        data=sc.ones(sizes=wavelength.sizes),
        coords={'wavelength': wavelength},
    )

    N = 123456
    source = tof.Source.from_distribution(
        neutrons=N,
        p_time=p_time,
        p_wav=p_wav,
    )

    assert source.neutrons == N
    assert source.data.sizes['event'] == N
    h = source.data['pulse', 0].hist(birth_time=10)
    assert sc.allclose(
        h.data,
        sc.full(value=N / 10.0, sizes={'birth_time': 10}, unit='counts'),
        rtol=sc.scalar(0.05),
    )


@pytest.mark.parametrize('distribution_2d', [False, True])
def test_creation_from_distribution(distribution_2d):
    v = np.ones(90) * 0.1
    v[30:60] = 1.0

    p_time = sc.DataArray(
        data=sc.array(dims=['birth_time'], values=v),
        coords={
            'birth_time': sc.linspace('birth_time', 0.0, 8000.0, len(v), unit='us')
        },
    )
    p_wav = sc.DataArray(
        data=sc.linspace('wavelength', 1.0, 4.0, 100, unit='angstrom'),
        coords={
            'wavelength': sc.linspace('wavelength', 1.0, 4.0, 100, unit='angstrom')
        },
    )
    if distribution_2d:
        source = tof.Source.from_distribution(neutrons=100_000, p=p_wav * p_time)
    else:
        source = tof.Source.from_distribution(
            neutrons=100_000, p_time=p_time, p_wav=p_wav
        )

    assert source.neutrons == 100_000
    da = source.data['pulse', 0]
    assert da.hist(
        birth_time=sc.array(dims=['birth_time'], values=[2100.0, 2900.0], unit='us')
    ).data.sum() > sc.scalar(0.0, unit='counts')
    assert da.hist(
        birth_time=sc.array(dims=['birth_time'], values=[5100.0, 5900.0], unit='us')
    ).data.sum() > sc.scalar(0.0, unit='counts')

    left = da.hist(
        birth_time=sc.array(dims=['birth_time'], values=[0.0, 2000.0], unit='us')
    ).data.sum()
    mid = da.hist(
        birth_time=sc.array(dims=['birth_time'], values=[3000.0, 5000.0], unit='us')
    ).data.sum()
    right = da.hist(
        birth_time=sc.array(dims=['birth_time'], values=[6000.0, 8000.0], unit='us')
    ).data.sum()
    rtol = sc.scalar(0.05)
    assert sc.isclose(mid / left, sc.scalar(10.0), rtol=rtol)
    assert sc.isclose(mid / right, sc.scalar(10.0), rtol=rtol)

    # Make sure distribution is monotonically increasing
    h = da.hist(wavelength=10)
    diff = h.data[1:] - h.data[:-1]
    assert sc.all(diff > sc.scalar(0.0, unit='counts'))


def test_non_integer_neutrons():
    source = tof.Source(facility='ess', neutrons=1e5)
    assert source.neutrons == 100_000


def test_multiple_pulses_ess():
    source = tof.Source(facility='ess', neutrons=100_000, pulses=3)
    assert source.data.sizes['pulse'] == 3
    assert source.data.sizes['event'] == 100_000
    assert sc.identical(source.frequency, sc.scalar(14.0, unit='Hz'))
    bins = (sc.arange('birth_time', 4) / source.frequency).to(unit='us')
    h = source.data.flatten(to='event').hist(birth_time=bins)
    assert sc.allclose(
        h.data, sc.full(value=1e5, sizes={'birth_time': 3}, unit='counts')
    )


def test_multiple_pulses_from_distribution():
    v = np.ones(9) * 0.1
    v[3:6] = 1.0

    p_time = sc.DataArray(
        data=sc.array(dims=['birth_time'], values=v),
        coords={
            'birth_time': sc.linspace('birth_time', 0.0, 8000.0, len(v), unit='us')
        },
    )
    p_wav = sc.DataArray(
        data=sc.array(dims=['wavelength'], values=[1.0, 2.0, 3.0, 4.0]),
        coords={
            'wavelength': sc.array(
                dims=['wavelength'], values=[1.0, 2.0, 3.0, 4.0], unit='angstrom'
            )
        },
    )

    source = tof.Source.from_distribution(
        neutrons=123_987,
        p_time=p_time,
        p_wav=p_wav,
        pulses=2,
        frequency=sc.scalar(100.0, unit='Hz'),
    )
    assert source.data.sizes['pulse'] == 2
    assert source.data.sizes['event'] == 123_987
    bins = (sc.arange('birth_time', 3) / source.frequency).to(unit='us')
    h = source.data.flatten(to='event').hist(birth_time=bins)
    assert sc.allclose(
        h.data, sc.full(value=123987.0, sizes={'birth_time': 2}, unit='counts')
    )


def test_multiple_pulses_from_neutrons():
    birth_times = sc.array(
        dims=['event'], values=[1111.0, 1567.0, 856.0, 2735.0], unit='us'
    )
    wavelengths = sc.array(
        dims=['event'], values=[1.0, 5.0, 8.0, 10.0], unit='angstrom'
    )
    source = tof.Source.from_neutrons(
        birth_times=birth_times,
        wavelengths=wavelengths,
        pulses=3,
        frequency=sc.scalar(50.0, unit='Hz'),
    )
    assert source.data.sizes['pulse'] == 3
    assert source.data.sizes['event'] == 4
    offsets = (sc.arange('pulse', 3) / source.frequency).to(unit='us')
    assert sc.allclose(source.data.coords['birth_time'], birth_times + offsets)
    assert sc.allclose(
        source.data.coords['wavelength'],
        sc.broadcast(wavelengths, sizes={'pulse': 3, 'event': 4}),
    )


def test_source_length():
    N = 17
    source = tof.Source(facility='ess', neutrons=3124, pulses=N)
    assert len(source) == N


def test_multiple_pulses_from_distribution_no_frequency_raises():
    v = np.ones(9) * 0.1
    v[3:6] = 1.0
    p_time = sc.DataArray(
        data=sc.array(dims=['birth_time'], values=v),
        coords={'birth_time': sc.linspace('birth_time', 0.0, 8.0, len(v), unit='ms')},
    )
    p_wav = sc.DataArray(
        data=sc.array(dims=['wavelength'], values=[1.0, 2.0, 3.0, 4.0]),
        coords={
            'wavelength': sc.array(
                dims=['wavelength'], values=[1.0, 2.0, 3.0, 4.0], unit='angstrom'
            )
        },
    )
    with pytest.raises(
        ValueError, match='If pulses is greater than one, a frequency must be supplied.'
    ):
        tof.Source.from_distribution(
            neutrons=22696,
            p_time=p_time,
            p_wav=p_wav,
            pulses=3,
        )


def test_multiple_pulses_from_neutrons_no_frequency_raises():
    birth_times = sc.array(
        dims=['event'], values=[1111.0, 1567.0, 856.0, 2735.0], unit='us'
    )
    wavelengths = sc.array(
        dims=['event'], values=[1.0, 5.0, 8.0, 10.0], unit='angstrom'
    )
    with pytest.raises(
        ValueError, match='If pulses is greater than one, a frequency must be supplied.'
    ):
        tof.Source.from_neutrons(
            birth_times=birth_times,
            wavelengths=wavelengths,
            pulses=3,
        )


def test_source_repr_does_not_raise():
    assert repr(tof.Source(facility='ess', neutrons=100_000)) is not None
    assert repr(tof.Source(facility='ess', neutrons=100_000, pulses=3)) is not None


def test_seed_ess_pulse():
    a = tof.Source(facility='ess', neutrons=100_000, seed=1234)
    b = tof.Source(facility='ess', neutrons=100_000, seed=1234)
    assert sc.identical(a.data, b.data)
    c = tof.Source(facility='ess', neutrons=100_000, seed=0)
    assert not sc.identical(a.data, c.data)


def test_seed_from_distribution():
    v = np.ones(9) * 0.1
    v[3:6] = 1.0

    p_time = sc.DataArray(
        data=sc.array(dims=['birth_time'], values=v),
        coords={
            'birth_time': sc.linspace('birth_time', 0.0, 8000.0, len(v), unit='us')
        },
    )
    p_wav = sc.DataArray(
        data=sc.array(dims=['wavelength'], values=[1.0, 2.0, 3.0, 4.0]),
        coords={
            'wavelength': sc.array(
                dims=['wavelength'], values=[1.0, 2.0, 3.0, 4.0], unit='angstrom'
            )
        },
    )

    a = tof.Source.from_distribution(
        neutrons=100_000, p_time=p_time, p_wav=p_wav, seed=12
    )
    b = tof.Source.from_distribution(
        neutrons=100_000, p_time=p_time, p_wav=p_wav, seed=12
    )
    assert sc.identical(a.data, b.data)
    c = tof.Source.from_distribution(
        neutrons=100_000, p_time=p_time, p_wav=p_wav, seed=1
    )
    assert not sc.identical(a.data, c.data)


def test_source_from_distribution_with_one_element_raises():
    p_time = sc.DataArray(
        data=sc.array(dims=['birth_time'], values=[1.0]),
        coords={'birth_time': sc.array(dims=['birth_time'], values=[1.0], unit='ms')},
    )
    p_wav = sc.DataArray(
        data=sc.array(dims=['wavelength'], values=[2.0]),
        coords={
            'wavelength': sc.array(dims=['wavelength'], values=[2.0], unit='angstrom')
        },
    )

    with pytest.raises(
        ValueError,
        match='Distribution must have at least 2 points in each dimension',
    ):
        tof.Source.from_distribution(neutrons=10, p_time=p_time, p_wav=p_wav)


def test_source_from_distrbution_all_zero_probability_raises():
    # All zero p_time
    p_time = sc.DataArray(
        data=sc.array(dims=['birth_time'], values=[0.0, 0.0, 0.0]),
        coords={'birth_time': sc.linspace('birth_time', 0.0, 2.0, 3, unit='ms')},
    )
    p_wav = sc.DataArray(
        data=sc.array(dims=['wavelength'], values=[0.0, 1.0, 0.0]),
        coords={'wavelength': sc.linspace('wavelength', 1.0, 3.0, 3, unit='angstrom')},
    )

    with pytest.raises(
        ValueError,
        match='Time distribution must have at least one positive probability value.',
    ):
        tof.Source.from_distribution(neutrons=10, p_time=p_time, p_wav=p_wav)

    # All zero p_wav
    p_time = sc.DataArray(
        data=sc.array(dims=['birth_time'], values=[0.0, 1.0, 0.0]),
        coords={'birth_time': sc.linspace('birth_time', 0.0, 2.0, 3, unit='ms')},
    )
    p_wav = sc.DataArray(
        data=sc.array(dims=['wavelength'], values=[0.0, 0.0, 0.0]),
        coords={'wavelength': sc.linspace('wavelength', 1.0, 3.0, 3, unit='angstrom')},
    )

    with pytest.raises(
        ValueError,
        match=(
            'Wavelength distribution must have at least one positive probability value'
        ),
    ):
        tof.Source.from_distribution(neutrons=10, p_time=p_time, p_wav=p_wav)

    # All zero p_time and p_wav
    p_time = sc.DataArray(
        data=sc.array(dims=['birth_time'], values=[0.0, 0.0, 0.0]),
        coords={'birth_time': sc.linspace('birth_time', 0.0, 2.0, 3, unit='ms')},
    )
    p_wav = sc.DataArray(
        data=sc.array(dims=['wavelength'], values=[0.0, 0.0, 0.0]),
        coords={'wavelength': sc.linspace('wavelength', 1.0, 3.0, 3, unit='angstrom')},
    )

    with pytest.raises(
        ValueError,
        match='Distribution must have at least one positive probability value.',
    ):
        tof.Source.from_distribution(neutrons=10, p=p_wav * p_time)
