# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc

import tof

Hz = sc.Unit('Hz')
deg = sc.Unit('deg')
rad = sc.Unit('rad')
meter = sc.Unit('m')
two_pi = 2.0 * rad * sc.constants.pi


def test_angular_speed():
    f = 10.0 * Hz
    chopper = tof.Chopper(
        frequency=f,
        open=0.0 * deg,
        close=10.0 * deg,
        phase=0.0 * deg,
        distance=1.0 * meter,
    )
    assert chopper.omega == two_pi * f


def test_open_close_times():
    f = 10.0 * Hz
    chopper = tof.Chopper(
        frequency=f,
        open=sc.array(dims=['cutout'], values=[10.0], unit='deg'),
        close=sc.array(dims=['cutout'], values=[20.0], unit='deg'),
        phase=0.0 * deg,
        distance=1.0 * meter,
    )
    assert sc.identical(
        chopper.open_times[0],
        (10.0 * deg).to(unit='rad') / (two_pi * f),
    )
    assert sc.identical(
        chopper.close_times[0],
        (20.0 * deg).to(unit='rad') / (two_pi * f),
    )


def test_open_close_angles_scalars_converted_to_arrays():
    f = 10.0 * Hz
    chopper = tof.Chopper(
        frequency=f,
        open=10.0 * deg,
        close=20.0 * deg,
        phase=0.0 * deg,
        distance=1.0 * meter,
    )
    assert sc.identical(
        chopper.open_times[0],
        (10.0 * deg).to(unit='rad') / (two_pi * f),
    )
    assert sc.identical(
        chopper.close_times[0],
        (20.0 * deg).to(unit='rad') / (two_pi * f),
    )


def test_phase():
    f = 10.0 * Hz
    chopper1 = tof.Chopper(
        frequency=f,
        open=sc.array(dims=['cutout'], values=[10.0], unit='deg'),
        close=sc.array(dims=['cutout'], values=[20.0], unit='deg'),
        phase=0.0 * deg,
        distance=10.0 * meter,
    )
    open_times = chopper1.open_times
    close_times = chopper1.close_times
    chopper2 = tof.Chopper(
        frequency=f,
        open=sc.array(dims=['cutout'], values=[10.0], unit='deg'),
        close=sc.array(dims=['cutout'], values=[20.0], unit='deg'),
        phase=30.0 * deg,
        distance=10.0 * meter,
    )
    assert sc.identical(
        chopper2.open_times, open_times + (30.0 * deg).to(unit='rad') / chopper2.omega
    )
    assert sc.identical(
        chopper2.close_times, close_times + (30.0 * deg).to(unit='rad') / chopper2.omega
    )
