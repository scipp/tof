# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import pytest
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
        name='chopper',
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
        name='chopper',
    )

    topen, tclose = chopper.open_close_times(0 * sec)
    # Note that choppers perform one rotation before t=0 to make sure
    # no openings are missed at early times.
    assert sc.identical(topen[0], ((10.0 * deg).to(unit='rad') - two_pi) / (two_pi * f))
    assert sc.identical(
        tclose[0], ((20.0 * deg).to(unit='rad') - two_pi) / (two_pi * f)
    )
    assert sc.identical(topen[1], (10.0 * deg).to(unit='rad') / (two_pi * f))
    assert sc.identical(tclose[1], (20.0 * deg).to(unit='rad') / (two_pi * f))


def test_open_close_times_three_rotations():
    f = 10.0 * Hz
    chopper = tof.Chopper(
        frequency=f,
        open=sc.array(dims=['cutout'], values=[10.0], unit='deg'),
        close=sc.array(dims=['cutout'], values=[20.0], unit='deg'),
        phase=0.0 * deg,
        distance=5.0 * meter,
        name='chopper',
    )

    topen, tclose = chopper.open_close_times(0.21 * sec)
    open0 = (10.0 * deg).to(unit='rad')
    close0 = (20.0 * deg).to(unit='rad')
    assert sc.identical(
        topen,
        sc.concat(
            [open0 - two_pi, open0, open0 + two_pi, open0 + 2 * two_pi], dim='cutout'
        )
        / (two_pi * f),
    )
    assert sc.identical(
        tclose,
        sc.concat(
            [close0 - two_pi, close0, close0 + two_pi, close0 + 2 * two_pi],
            dim='cutout',
        )
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
        name='chopper',
    )
    topen, tclose = chopper.open_close_times(0.0 * sec)
    assert sc.identical(topen[1], (10.0 * deg).to(unit='rad') / (two_pi * f))
    assert sc.identical(tclose[1], (20.0 * deg).to(unit='rad') / (two_pi * f))


def test_phase():
    f = 10.0 * Hz
    chopper1 = tof.Chopper(
        frequency=f,
        open=sc.array(dims=['cutout'], values=[10.0], unit='deg'),
        close=sc.array(dims=['cutout'], values=[20.0], unit='deg'),
        phase=0.0 * deg,
        distance=10.0 * meter,
        name='chopper1',
    )
    topen1, tclose1 = chopper1.open_close_times(0.0 * sec)
    chopper2 = tof.Chopper(
        frequency=f,
        open=sc.array(dims=['cutout'], values=[10.0], unit='deg'),
        close=sc.array(dims=['cutout'], values=[20.0], unit='deg'),
        phase=30.0 * deg,
        distance=10.0 * meter,
        name='chopper2',
    )
    topen2, tclose2 = chopper2.open_close_times(0.0 * sec)
    assert sc.allclose(topen2, topen1 + (30.0 * deg).to(unit='rad') / chopper2.omega)
    assert sc.allclose(tclose2, tclose1 + (30.0 * deg).to(unit='rad') / chopper2.omega)


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
        name='chopper1',
    )
    chopper2 = tof.Chopper(
        frequency=f,
        open=op,
        close=cl,
        phase=30 * deg,
        distance=d,
        name='chopper2',
    )
    topen1, tclose1 = chopper1.open_close_times(0.0 * sec)
    topen2, tclose2 = chopper2.open_close_times(0.0 * sec)
    assert sc.allclose(topen1, topen2)
    assert sc.allclose(tclose1, tclose2)


def test_frequency_must_be_positive():
    with pytest.raises(ValueError, match="Chopper frequency must be positive"):
        tof.Chopper(
            frequency=-1.0 * Hz,
            open=0.0 * deg,
            close=10.0 * deg,
            phase=0.0 * deg,
            distance=5.0 * meter,
            name='chopper',
        )


def test_open_close_times_counter_rotation():
    f = 10.0 * Hz
    d = 10.0 * meter
    ph = 0.0 * deg
    chopper1 = tof.Chopper(
        frequency=f,
        open=sc.array(dims=['cutout'], values=[10.0, 90.0], unit='deg'),
        close=sc.array(dims=['cutout'], values=[20.0, 130.0], unit='deg'),
        phase=ph,
        distance=d,
        name='chopper1',
    )
    chopper2 = tof.Chopper(
        frequency=f,
        open=sc.array(
            dims=['cutout'], values=[360.0 - 130.0, 360.0 - 20.0], unit='deg'
        ),
        close=sc.array(
            dims=['cutout'], values=[360.0 - 90.0, 360.0 - 10.0], unit='deg'
        ),
        phase=ph,
        distance=d,
        direction=tof.AntiClockwise,
        name='chopper2',
    )

    topen1, tclose1 = chopper1.open_close_times(0.0 * sec)
    topen2, tclose2 = chopper2.open_close_times(0.0 * sec)
    assert sc.allclose(topen1, topen2)
    assert sc.allclose(tclose1, tclose2)


def test_open_close_times_counter_rotation_with_phase():
    f = 10.0 * Hz
    chopper1 = tof.Chopper(
        frequency=f,
        open=sc.array(dims=['cutout'], values=[80.0], unit='deg'),
        close=sc.array(dims=['cutout'], values=[90.0], unit='deg'),
        phase=0.0 * deg,
        distance=10.0 * meter,
        direction=tof.AntiClockwise,
        name='chopper1',
    )
    topen1, tclose1 = chopper1.open_close_times(0.0 * sec)
    chopper2 = tof.Chopper(
        frequency=f,
        open=sc.array(dims=['cutout'], values=[80.0], unit='deg'),
        close=sc.array(dims=['cutout'], values=[90.0], unit='deg'),
        phase=30.0 * deg,
        distance=10.0 * meter,
        direction=tof.AntiClockwise,
        name='chopper2',
    )
    topen2, tclose2 = chopper2.open_close_times(0.0 * sec)
    assert sc.allclose(
        topen2, topen1 + (30.0 * deg).to(unit='rad') / abs(chopper2.omega)
    )
    assert sc.allclose(
        tclose2, tclose1 + (30.0 * deg).to(unit='rad') / abs(chopper2.omega)
    )


def test_open_close_anticlockwise_multiple_rotations():
    chopper = tof.Chopper(
        frequency=10.0 * Hz,
        open=sc.array(dims=["cutout"], values=[10.0, 90.0], unit="deg"),
        close=sc.array(dims=["cutout"], values=[20.0, 130.0], unit="deg"),
        distance=10.0 * meter,
        phase=0 * deg,
        direction=tof.AntiClockwise,
        name='chopper',
    )

    two_rotations_open, two_rotations_close = chopper.open_close_times(0.0 * sec)
    three_rotations_open, three_rotations_close = chopper.open_close_times(0.2 * sec)
    four_rotations_open, four_rotations_close = chopper.open_close_times(0.3 * sec)

    assert len(two_rotations_open) == 4
    assert len(three_rotations_open) == 6
    assert len(four_rotations_open) == 8
    assert len(two_rotations_close) == 4
    assert len(three_rotations_close) == 6
    assert len(four_rotations_close) == 8

    assert sc.allclose(two_rotations_open, three_rotations_open[:-2])
    assert sc.allclose(two_rotations_close, three_rotations_close[:-2])
    assert sc.allclose(three_rotations_open, four_rotations_open[:-2])
    assert sc.allclose(three_rotations_close, four_rotations_close[:-2])
    assert sc.allclose(two_rotations_open, four_rotations_open[:-4])
    assert sc.allclose(two_rotations_close, four_rotations_close[:-4])


def test_bad_direction_raises():
    f = 10.0 * Hz
    op = sc.array(dims=['cutout'], values=[10.0], unit='deg')
    cl = sc.array(dims=['cutout'], values=[20.0], unit='deg')
    d = 10.0 * meter
    ph = 0.0 * deg
    tof.Chopper(
        frequency=f,
        open=op,
        close=cl,
        phase=ph,
        distance=d,
        direction=tof.Clockwise,
        name='chopper',
    )
    tof.Chopper(
        frequency=f,
        open=op,
        close=cl,
        phase=ph,
        distance=d,
        direction=tof.AntiClockwise,
        name='chopper',
    )
    with pytest.raises(
        ValueError, match="Chopper direction must be Clockwise or AntiClockwise"
    ):
        tof.Chopper(
            frequency=f,
            open=op,
            close=cl,
            phase=ph,
            distance=d,
            direction='clockwise',
            name='chopper',
        )
    with pytest.raises(
        ValueError, match="Chopper direction must be Clockwise or AntiClockwise"
    ):
        tof.Chopper(
            frequency=f,
            open=op,
            close=cl,
            phase=ph,
            distance=d,
            direction='anti-clockwise',
            name='chopper',
        )
    with pytest.raises(
        ValueError, match="Chopper direction must be Clockwise or AntiClockwise"
    ):
        tof.Chopper(
            frequency=f,
            open=op,
            close=cl,
            phase=ph,
            distance=d,
            direction=1,
            name='chopper',
        )


def test_chopper_create_from_centers_and_widths():
    f = 10.0 * Hz
    centers = sc.array(dims=['cutout'], values=[15.0, 46.0], unit='deg')
    widths = sc.array(dims=['cutout'], values=[10.0, 16.0], unit='deg')
    chopper = tof.Chopper(
        frequency=f,
        centers=centers,
        widths=widths,
        phase=0.0 * deg,
        distance=5.0 * meter,
        name='chopper',
    )
    assert sc.allclose(chopper.open, centers - widths / 2)
    assert sc.allclose(chopper.close, centers + widths / 2)

    expected = tof.Chopper(
        frequency=f,
        open=sc.array(dims=['cutout'], values=[10.0, 38.0], unit='deg'),
        close=sc.array(dims=['cutout'], values=[20.0, 54.0], unit='deg'),
        phase=0.0 * deg,
        distance=5.0 * meter,
        name='chopper',
    )
    assert sc.allclose(chopper.open, expected.open)
    assert sc.allclose(chopper.close, expected.close)


def test_chopper_create_raises_when_both_edges_and_centers_are_supplied():
    with pytest.raises(
        ValueError, match="Either open/close or centers/widths must be provided"
    ):
        tof.Chopper(
            frequency=10.0 * Hz,
            open=10.0 * deg,
            close=20.0 * deg,
            centers=15.0 * deg,
            widths=10.0 * deg,
            phase=0.0 * deg,
            distance=5.0 * meter,
            name='chopper',
        )
