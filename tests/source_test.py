# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)


import numpy as np
import pytest
import scipp as sc

import tof


def test_ess_pulse():
    source = tof.Source(facility='ess', neutrons=100_000)
    # Check that the time distribution is low on edges and high in the middle
    times = source.data['pulse', 0].hist(time=300)
    mean = times.mean()
    assert (times[0] < 0.5 * mean).value
    assert (times[-1] < 0.5 * mean).value
    assert (times[150] > 1.5 * mean).value
    # Check that there are more neutrons at low wavelengths
    wavs = source.data['pulse', 0].hist(wavelength=300)
    assert (wavs[:150].sum() > 1.5 * wavs[150:].sum()).value


def test_creation_from_neutrons():
    birth_times = sc.array(dims=['event'], values=[1000.0, 1500.0, 2000.0], unit='us')
    wavelengths = sc.array(dims=['event'], values=[1.0, 5.0, 10.0], unit='angstrom')
    source = tof.Source.from_neutrons(
        birth_times=birth_times,
        wavelengths=wavelengths,
    )
    assert source.neutrons == 3
    assert sc.identical(source.data['pulse', 0].coords['time'], birth_times)
    assert sc.identical(source.data['pulse', 0].coords['wavelength'], wavelengths)


def test_creation_from_distribution_flat():
    time = sc.array(dims=['time'], values=[1.0, 3.0], unit='ms')
    p_time = sc.DataArray(
        data=sc.ones(sizes=time.sizes),
        coords={'time': time},
    )
    wavelength = sc.array(dims=['wavelength'], values=[1.0, 10.0], unit='angstrom')
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
    h = source.data['pulse', 0].hist(time=10)
    assert sc.allclose(
        h.data,
        sc.full(value=N / 10.0, sizes={'time': 10}, unit='counts'),
        rtol=sc.scalar(0.05),
    )


def test_creation_from_distribution():
    v = np.ones(9) * 0.1
    v[3:6] = 1.0

    p_time = sc.DataArray(
        data=sc.array(dims=['time'], values=v),
        coords={'time': sc.linspace('time', 0.0, 8000.0, len(v), unit='us')},
    )
    p_wav = sc.DataArray(
        data=sc.array(dims=['wavelength'], values=[1.0, 2.0, 3.0, 4.0]),
        coords={
            'wavelength': sc.array(
                dims=['wavelength'], values=[1.0, 2.0, 3.0, 4.0], unit='angstrom'
            )
        },
    )

    source = tof.Source.from_distribution(neutrons=100_000, p_time=p_time, p_wav=p_wav)
    assert source.neutrons == 100_000
    da = source.data['pulse', 0]
    assert da.hist(
        time=sc.array(dims=['time'], values=[2100.0, 2900.0], unit='us')
    ).data.sum() > sc.scalar(0.0, unit='counts')
    assert da.hist(
        time=sc.array(dims=['time'], values=[5100.0, 5900.0], unit='us')
    ).data.sum() > sc.scalar(0.0, unit='counts')

    left = da.hist(
        time=sc.array(dims=['time'], values=[0.0, 2000.0], unit='us')
    ).data.sum()
    mid = da.hist(
        time=sc.array(dims=['time'], values=[3000.0, 5000.0], unit='us')
    ).data.sum()
    right = da.hist(
        time=sc.array(dims=['time'], values=[6000.0, 8000.0], unit='us')
    ).data.sum()
    rtol = sc.scalar(0.05)
    assert sc.isclose(mid / left, sc.scalar(10.0), rtol=rtol)
    assert sc.isclose(mid / right, sc.scalar(10.0), rtol=rtol)

    # Make sure distribution is monotonically increasing
    locs = np.linspace(1.0, 4.0, 20)
    step = 0.5 * (locs[1] - locs[0])
    for i in range(len(locs) - 2):
        a = da.hist(
            wavelength=sc.array(
                dims=['wavelength'],
                values=[locs[i] - step, locs[i] + step],
                unit='angstrom',
            )
        ).data.sum()
        b = da.hist(
            wavelength=sc.array(
                dims=['wavelength'],
                values=[locs[i + 1] - step, locs[i + 1] + step],
                unit='angstrom',
            )
        ).data.sum()
        assert b > a


def test_non_integer_sampling():
    N = 1_000_000
    source_float = tof.Source(facility='ess', neutrons=N, sampling=1e4)
    source_int = tof.Source(facility='ess', neutrons=N, sampling=10_000)
    assert source_float.neutrons == source_int.neutrons == N
    tedges = sc.linspace('time', 0.0, 5.0e3, 301, unit='us')
    wedges = sc.linspace('wavelength', 0.0, 20.0, 301, unit='angstrom')

    da_f = source_float.data['pulse', 0]
    da_i = source_int.data['pulse', 0]

    a = da_f.hist(time=tedges).data
    b = da_i.hist(time=tedges).data
    c = da_f.hist(wavelength=wedges).data
    d = da_i.hist(wavelength=wedges).data

    assert sc.allclose(a, b, atol=0.1 * a.max())
    assert sc.allclose(c, d, atol=0.1 * c.max())


def test_non_integer_neutrons():
    source = tof.Source(facility='ess', neutrons=1e5)
    assert source.neutrons == 100_000


def test_multiple_pulses_ess():
    source = tof.Source(facility='ess', neutrons=100_000, pulses=3)
    assert source.data.sizes['pulse'] == 3
    assert source.data.sizes['event'] == 100_000
    assert sc.identical(source.frequency, sc.scalar(14.0, unit='Hz'))
    bins = (sc.arange('time', 4) / source.frequency).to(unit='us')
    h = source.data.flatten(to='event').hist(time=bins)
    assert sc.allclose(h.data, sc.full(value=1e5, sizes={'time': 3}, unit='counts'))


def test_multiple_pulses_from_distribution():
    v = np.ones(9) * 0.1
    v[3:6] = 1.0

    p_time = sc.DataArray(
        data=sc.array(dims=['time'], values=v),
        coords={'time': sc.linspace('time', 0.0, 8000.0, len(v), unit='us')},
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
    bins = (sc.arange('time', 3) / source.frequency).to(unit='us')
    h = source.data.flatten(to='event').hist(time=bins)
    assert sc.allclose(
        h.data, sc.full(value=123987.0, sizes={'time': 2}, unit='counts')
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
    assert sc.allclose(source.data.coords['time'], birth_times + offsets)
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
        data=sc.array(dims=['time'], values=v),
        coords={'time': sc.linspace('time', 0.0, 8.0, len(v), unit='ms')},
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
        data=sc.array(dims=['time'], values=v),
        coords={'time': sc.linspace('time', 0.0, 8000.0, len(v), unit='us')},
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
