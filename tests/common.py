# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc

import tof


Hz = sc.Unit('Hz')
deg = sc.Unit('deg')
meter = sc.Unit('m')
ms = sc.Unit('ms')
two_pi = sc.constants.pi * (2.0 * sc.units.rad)


def make_chopper(topen, tclose, f, phase, distance, name):
    aopen = two_pi * sc.concat(topen, dim='cutout').to(unit='s') * f
    aclose = two_pi * sc.concat(tclose, dim='cutout').to(unit='s') * f
    return tof.Chopper(
        frequency=f,
        open=aopen,
        close=aclose,
        phase=phase,
        distance=distance,
        name=name,
    )


def make_pulse(arrival_times, distance):
    # Arrival times are distance * alpha * wavelength
    return tof.Pulse.from_neutrons(
        birth_times=sc.array(
            dims=['event'],
            values=[0.0] * len(arrival_times),
            unit='s',
        ),
        wavelengths=arrival_times.to(unit='s') / (distance * tof.utils.alpha),
    )


def dummy_chopper():
    return tof.Chopper(
        frequency=1.0 * Hz,
        open=sc.array(dims=['cutout'], values=[0.0], unit='deg'),
        close=sc.array(dims=['cutout'], values=[1.0], unit='deg'),
        phase=0.0 * deg,
        distance=1.0 * meter,
        name='dummy_chopper',
    )


def dummy_detector():
    return tof.Detector(
        distance=1.0 * meter,
        name='dummy_detector',
    )


def dummy_pulse():
    return tof.Pulse.from_neutrons(
        birth_times=sc.array(
            dims=['event'],
            values=[0.0],
            unit='s',
        ),
        wavelengths=sc.array(
            dims=['event'],
            values=[1.0],
            unit='angstrom',
        ),
    )
