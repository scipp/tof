# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import scipp as sc

import tof


def test_creation_default():
    N = 1234
    tmin = sc.scalar(0.5e-3, unit='s')
    tmax = sc.scalar(2.7e-3, unit='s')
    wmin = sc.scalar(1.0, unit='angstrom')
    wmax = sc.scalar(10.0, unit='angstrom')
    pulse = tof.Pulse(neutrons=N, tmin=tmin, tmax=tmax, wmin=wmin, wmax=wmax)
    assert pulse.neutrons == N
    assert len(pulse.birth_times) == N
    assert len(pulse.wavelengths) == N
    assert len(pulse.speeds) == N
    assert pulse.birth_times.min() >= tmin
    assert pulse.birth_times.max() <= tmax
    assert pulse.wavelengths.min() >= wmin
    assert pulse.wavelengths.max() <= wmax


def test_creation_from_neutrons():
    birth_times = sc.array(dims=['event'], values=[1000.0, 1500.0, 2000.0], unit='us')
    wavelengths = sc.array(dims=['event'], values=[1.0, 5.0, 10.0], unit='angstrom')
    pulse = tof.Pulse.from_neutrons(
        birth_times=birth_times,
        wavelengths=wavelengths,
    )
    assert pulse.neutrons == 3
    assert sc.identical(pulse.birth_times, birth_times.to(unit='s'))
    assert sc.identical(pulse.wavelengths, wavelengths)
    assert pulse.tmin == sc.scalar(1.0e-3, unit='s')
    assert pulse.tmax == sc.scalar(2.0e-3, unit='s')
    assert pulse.wmin == sc.scalar(1.0, unit='angstrom')
    assert pulse.wmax == sc.scalar(10.0, unit='angstrom')


def test_creation_from_distribution():
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

    pulse = tof.Pulse.from_distribution(neutrons=100_000, p_time=p_time, p_wav=p_wav)
    assert pulse.neutrons == 100_000
    assert sc.identical(
        pulse.birth_times.hist(
            time=sc.array(dims=['time'], values=[4.1, 4.9], unit='ms').to(unit='s')
        ).data.sum(),
        sc.scalar(0.0, unit='counts'),
    )
    left = pulse.birth_times.hist(
        time=sc.array(dims=['time'], values=[-0.1, 2.9], unit='ms').to(unit='s')
    ).data.sum()
    mid = pulse.birth_times.hist(
        time=sc.array(dims=['time'], values=[2.9, 5.1], unit='ms').to(unit='s')
    ).data.sum()
    right = pulse.birth_times.hist(
        time=sc.array(dims=['time'], values=[5.1, 8.1], unit='ms').to(unit='s')
    ).data.sum()
    rtol = sc.scalar(0.03)
    assert sc.isclose(mid / left, sc.scalar(10.0), rtol=rtol)
    assert sc.isclose(mid / right, sc.scalar(10.0), rtol=rtol)

    assert sc.identical(
        pulse.wavelengths.hist(
            wavelength=sc.array(dims=['wavelength'], values=[1.1, 1.9], unit='angstrom')
        ).data.sum(),
        sc.scalar(0.0, unit='counts'),
    )
    a = pulse.wavelengths.hist(
        wavelength=sc.array(dims=['wavelength'], values=[0.9, 1.1], unit='angstrom')
    ).data.sum()
    b = pulse.wavelengths.hist(
        wavelength=sc.array(dims=['wavelength'], values=[1.9, 2.1], unit='angstrom')
    ).data.sum()
    c = pulse.wavelengths.hist(
        wavelength=sc.array(dims=['wavelength'], values=[2.9, 3.1], unit='angstrom')
    ).data.sum()
    d = pulse.wavelengths.hist(
        wavelength=sc.array(dims=['wavelength'], values=[3.9, 4.1], unit='angstrom')
    ).data.sum()
    assert sc.isclose(b / a, sc.scalar(2.0), rtol=rtol)
    assert sc.isclose(c / a, sc.scalar(3.0), rtol=rtol)
    assert sc.isclose(d / a, sc.scalar(4.0), rtol=rtol)


def test_creation_from_distribution_sampling():
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

    pulse = tof.Pulse.from_distribution(
        neutrons=100_000, p_time=p_time, p_wav=p_wav, sampling=100_000
    )
    assert pulse.neutrons == 100_000
    assert pulse.birth_times.hist(
        time=sc.array(dims=['time'], values=[2.1, 2.9], unit='ms').to(unit='s')
    ).data.sum() > sc.scalar(0.0, unit='counts')
    assert pulse.birth_times.hist(
        time=sc.array(dims=['time'], values=[5.1, 5.9], unit='ms').to(unit='s')
    ).data.sum() > sc.scalar(0.0, unit='counts')

    left = pulse.birth_times.hist(
        time=sc.array(dims=['time'], values=[0.0, 2.0], unit='ms').to(unit='s')
    ).data.sum()
    mid = pulse.birth_times.hist(
        time=sc.array(dims=['time'], values=[3.0, 5.0], unit='ms').to(unit='s')
    ).data.sum()
    right = pulse.birth_times.hist(
        time=sc.array(dims=['time'], values=[6.0, 8.0], unit='ms').to(unit='s')
    ).data.sum()
    rtol = sc.scalar(0.05)
    assert sc.isclose(mid / left, sc.scalar(10.0), rtol=rtol)
    assert sc.isclose(mid / right, sc.scalar(10.0), rtol=rtol)

    # Make sure distribution is monotonically increasing
    locs = np.linspace(1.0, 4.0, 20)
    step = 0.5 * (locs[1] - locs[0])
    for i in range(len(locs) - 2):
        a = pulse.wavelengths.hist(
            wavelength=sc.array(
                dims=['wavelength'],
                values=[locs[i] - step, locs[i] + step],
                unit='angstrom',
            )
        ).data.sum()
        b = pulse.wavelengths.hist(
            wavelength=sc.array(
                dims=['wavelength'],
                values=[locs[i + 1] - step, locs[i + 1] + step],
                unit='angstrom',
            )
        ).data.sum()
        assert b > a


def test_duration():
    N = 1234
    tmin = sc.scalar(0.5e-3, unit='s')
    tmax = sc.scalar(2.7e-3, unit='s')
    wmin = sc.scalar(1.0, unit='angstrom')
    wmax = sc.scalar(10.0, unit='angstrom')
    pulse = tof.Pulse(neutrons=N, tmin=tmin, tmax=tmax, wmin=wmin, wmax=wmax)
    assert pulse.duration == tmax - tmin


def test_ess_pulse():
    pulse = tof.Pulse.from_facility(kind='ess', neutrons=100_000)
    # Check that the time distribution is low on edges and high in the middle
    times = pulse.birth_times.hist(time=300)
    mean = times.mean()
    assert (times[0] < 0.5 * mean).value
    assert (times[-1] < 0.5 * mean).value
    assert (times[150] > 1.5 * mean).value
    # Check that there are more neutrons at low wavelengths
    wavs = pulse.wavelengths.hist(wavelength=300)
    assert (wavs[:150].sum() > 1.5 * wavs[150:].sum()).value
