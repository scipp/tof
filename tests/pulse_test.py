# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc

import tof


def test_creation_from_distribution():
    N = 1234
    tmin = sc.scalar(0.5e-3, unit='s')
    tmax = sc.scalar(2.7e-3, unit='s')
    lmin = sc.scalar(1.0, unit='angstrom')
    lmax = sc.scalar(10.0, unit='angstrom')
    pulse = tof.Pulse(neutrons=N, tmin=tmin, tmax=tmax, lmin=lmin, lmax=lmax)
    assert pulse.neutrons == N
    assert len(pulse.birth_times) == N
    assert len(pulse.wavelengths) == N
    assert len(pulse.speeds) == N
    assert pulse.birth_times.min() >= tmin
    assert pulse.birth_times.max() <= tmax
    assert pulse.wavelengths.min() >= lmin
    assert pulse.wavelengths.max() <= lmax


def test_creation_from_supplied_values():
    birth_times = sc.array(dims=['event'], values=[1000.0, 1500.0, 2000.0], unit='us')
    wavelengths = sc.array(dims=['event'], values=[1.0, 5.0, 10.0], unit='angstrom')
    pulse = tof.Pulse(
        birth_times=birth_times,
        wavelengths=wavelengths,
    )
    assert pulse.neutrons == 3
    assert sc.identical(pulse.birth_times, birth_times.to(unit='s'))
    assert sc.identical(pulse.wavelengths, wavelengths)
    assert pulse.tmin == sc.scalar(1.0e-3, unit='s')
    assert pulse.tmax == sc.scalar(2.0e-3, unit='s')
    assert pulse.lmin == sc.scalar(1.0, unit='angstrom')
    assert pulse.lmax == sc.scalar(10.0, unit='angstrom')


def test_duration():
    N = 1234
    tmin = sc.scalar(0.5e-3, unit='s')
    tmax = sc.scalar(2.7e-3, unit='s')
    lmin = sc.scalar(1.0, unit='angstrom')
    lmax = sc.scalar(10.0, unit='angstrom')
    pulse = tof.Pulse(neutrons=N, tmin=tmin, tmax=tmax, lmin=lmin, lmax=lmax)
    assert pulse.duration == tmax - tmin


def test_generate_new_neutrons():
    N = 1234
    tmin = sc.scalar(0.5e-3, unit='s')
    tmax = sc.scalar(2.7e-3, unit='s')
    lmin = sc.scalar(1.0, unit='angstrom')
    lmax = sc.scalar(10.0, unit='angstrom')
    pulse = tof.Pulse(neutrons=N, tmin=tmin, tmax=tmax, lmin=lmin, lmax=lmax)
    assert pulse.neutrons == N
    assert len(pulse.birth_times) == N
    pulse.generate(9999)
    assert pulse.neutrons == 9999
    assert len(pulse.birth_times) == 9999


def test_ess_pulse():
    pulse = tof.Pulse(kind='ess', neutrons=100_000)
    # Check that the time distribution is low on edges and high in the middle
    times = pulse.birth_times.hist(time=300)
    mean = times.mean()
    assert (times[0] < 0.5 * mean).value
    assert (times[-1] < 0.5 * mean).value
    assert (times[150] > 1.5 * mean).value
    # Check that there are more neutrons at low wavelengths
    wavs = pulse.wavelengths.hist(wavelength=300)
    assert (wavs[:150].sum() > 1.5 * wavs[150:].sum()).value
