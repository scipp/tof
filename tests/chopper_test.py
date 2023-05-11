# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc

import tof

Hz = sc.Unit('Hz')
deg = sc.Unit('deg')
rad = sc.Unit('rad')
meter = sc.Unit('m')
sec = sc.Unit('s')
two_pi = 2.0 * rad * sc.constants.pi


def test_angular_speed():
    f = 10.0 * Hz
    chopper = tof.Chopper(
        frequency=f,
        open=0.0 * deg,
        close=10.0 * deg,
        phase=0.0 * deg,
        distance=5.0 * meter,
    )
    assert chopper.omega == two_pi * f


def test_open_close_times_one_rotation():
    f = 10.0 * Hz
    chopper = tof.Chopper(
        frequency=f,
        open=sc.array(dims=['cutout'], values=[10.0], unit='deg'),
        close=sc.array(dims=['cutout'], values=[20.0], unit='deg'),
        phase=0.0 * deg,
        distance=5.0 * meter,
    )

    topen, tclose = chopper.open_close_times(0 * sec)
    assert sc.identical(topen[0], (10.0 * deg).to(unit='rad') / (two_pi * f))
    assert sc.identical(tclose[0], (20.0 * deg).to(unit='rad') / (two_pi * f))


def test_open_close_times_three_rotations():
    f = 10.0 * Hz
    chopper = tof.Chopper(
        frequency=f,
        open=sc.array(dims=['cutout'], values=[10.0], unit='deg'),
        close=sc.array(dims=['cutout'], values=[20.0], unit='deg'),
        phase=0.0 * deg,
        distance=5.0 * meter,
    )

    topen, tclose = chopper.open_close_times(0.21 * sec)
    open0 = (10.0 * deg).to(unit='rad')
    close0 = (20.0 * deg).to(unit='rad')
    assert sc.identical(
        topen,
        sc.concat([open0, open0 + two_pi, open0 + 2 * two_pi], dim='cutout')
        / (two_pi * f),
    )
    assert sc.identical(
        tclose,
        sc.concat([close0, close0 + two_pi, close0 + 2 * two_pi], dim='cutout')
        / (two_pi * f),
    )


def test_open_close_angles_scalars_converted_to_arrays():
    f = 10.0 * Hz
    chopper = tof.Chopper(
        frequency=f,
        open=10.0 * deg,
        close=20.0 * deg,
        phase=0.0 * deg,
        distance=5.0 * meter,
    )
    topen, tclose = chopper.open_close_times(0.0 * sec)
    assert sc.identical(topen[0], (10.0 * deg).to(unit='rad') / (two_pi * f))
    assert sc.identical(tclose[0], (20.0 * deg).to(unit='rad') / (two_pi * f))


def test_phase():
    f = 10.0 * Hz
    chopper1 = tof.Chopper(
        frequency=f,
        open=sc.array(dims=['cutout'], values=[10.0], unit='deg'),
        close=sc.array(dims=['cutout'], values=[20.0], unit='deg'),
        phase=0.0 * deg,
        distance=10.0 * meter,
    )
    topen1, tclose1 = chopper1.open_close_times(0.0 * sec)
    chopper2 = tof.Chopper(
        frequency=f,
        open=sc.array(dims=['cutout'], values=[10.0], unit='deg'),
        close=sc.array(dims=['cutout'], values=[20.0], unit='deg'),
        phase=30.0 * deg,
        distance=10.0 * meter,
    )
    topen2, tclose2 = chopper2.open_close_times(0.0 * sec)
    assert sc.identical(topen2, topen1 + (30.0 * deg).to(unit='rad') / chopper2.omega)
    assert sc.identical(tclose2, tclose1 + (30.0 * deg).to(unit='rad') / chopper2.omega)


def test_phase_int():
    f = 10.0 * Hz
    op = sc.array(dims=['cutout'], values=[10.0], unit='deg')
    cl = sc.array(dims=['cutout'], values=[20.0], unit='deg')
    d = 10.0 * meter
    chopper1 = tof.Chopper(
        frequency=f,
        open=op,
        close=cl,
        phase=30.0 * deg,
        distance=d,
    )
    chopper2 = tof.Chopper(
        frequency=f,
        open=op,
        close=cl,
        phase=30 * deg,
        distance=d,
    )
    topen1, tclose1 = chopper1.open_close_times(0.0 * sec)
    topen2, tclose2 = chopper2.open_close_times(0.0 * sec)
    assert sc.identical(topen1, topen2)
    assert sc.identical(tclose1, tclose2)
