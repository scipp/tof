# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc

import tof


def test_creation_default():
    N = 1234
    tsta = sc.scalar(0.5e-3, unit='s')
    tend = sc.scalar(2.7e-3, unit='s')
    wmin = sc.scalar(1.0, unit='angstrom')
    wmax = sc.scalar(10.0, unit='angstrom')
    pulse = tof.Pulse(neutrons=N, tmin=tsta, tmax=tend, wav_min=wmin, wav_max=wmax)
    assert pulse.neutrons == N
    assert len(pulse.birth_times) == N
    assert len(pulse.wavelengths) == N
    assert len(pulse.speeds) == N
    assert pulse.birth_times.min() >= tsta
    assert pulse.birth_times.max() <= tend
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
    assert pulse.wav_min == sc.scalar(1.0, unit='angstrom')
    assert pulse.wav_max == sc.scalar(10.0, unit='angstrom')


def test_duration():
    N = 1234
    tsta = sc.scalar(0.5e-3, unit='s')
    tend = sc.scalar(2.7e-3, unit='s')
    wmin = sc.scalar(1.0, unit='angstrom')
    wmax = sc.scalar(10.0, unit='angstrom')
    pulse = tof.Pulse(neutrons=N, tmin=tsta, tmax=tend, wav_min=wmin, wav_max=wmax)
    assert pulse.duration == tend - tsta


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
