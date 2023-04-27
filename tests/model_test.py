# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc

import tof

Hz = sc.Unit('Hz')
deg = sc.Unit('deg')
meter = sc.Unit('m')
ms = sc.Unit('ms')


def make_chopper(topen, tclose, f, phase, distance, name):
    aopen = (
        sc.constants.pi
        * (2.0 * sc.units.rad)
        * sc.concat(topen, dim='cutout').to(unit='s')
        * f
    )
    aclose = (
        sc.constants.pi
        * (2.0 * sc.units.rad)
        * sc.concat(tclose, dim='cutout').to(unit='s')
        * f
    )
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
    return tof.Pulse(
        birth_times=sc.array(
            dims=['event'],
            values=[0.0] * len(arrival_times),
            unit='s',
        ),
        wavelengths=arrival_times.to(unit='s') / (distance * tof.utils.alpha),
    )


def test_one_chopper_one_opening():
    # Make a chopper open from 10-20 ms. Assume zero phase.
    topen = 10.0 * ms
    tclose = 20.0 * ms
    chopper = make_chopper(
        topen=[topen],
        tclose=[tclose],
        f=10.0 * Hz,
        phase=0.0 * deg,
        distance=10 * meter,
        name="chopper",
    )
    detector = tof.Detector(distance=20 * meter, name="detector")

    # Make a pulse with 3 neutrons with one neutron going through the chopper opening
    # and the other two neutrons on either side of the opening.
    pulse = make_pulse(
        arrival_times=sc.concat(
            [0.9 * topen, 0.5 * (topen + tclose), 1.1 * tclose], dim='event'
        ),
        distance=chopper.distance,
    )

    model = tof.Model(pulse=pulse, choppers=[chopper], detectors=[detector])
    model.run()

    assert len(chopper.tofs.visible) == 1
    assert len(chopper.tofs.blocked) == 2
    assert sc.isclose(
        chopper.tofs.visible.data.coords['tof'][0],
        (0.5 * (topen + tclose)).to(unit='us'),
    )
    assert sc.isclose(
        chopper.tofs.blocked.data.coords['tof'][0], (0.9 * topen).to(unit='us')
    )
    assert sc.isclose(
        chopper.tofs.blocked.data.coords['tof'][1], (1.1 * tclose).to(unit='us')
    )
    assert len(detector.tofs.visible) == 1
    assert sc.isclose(
        detector.tofs.visible.data.coords['tof'][0],
        (pulse.wavelengths[1] * detector.distance * tof.utils.alpha).to(unit='us'),
    )


def test_two_choppers_one_opening():
    # Make a first chopper open from 5-16 ms. Assume zero phase.
    topen = 5.0 * ms
    tclose = 16.0 * ms
    chopper1 = make_chopper(
        topen=[topen],
        tclose=[tclose],
        f=10.0 * Hz,
        phase=0.0 * deg,
        distance=10 * meter,
        name="chopper1",
    )

    # Make a second chopper open from 15-20 ms.
    chopper2 = make_chopper(
        topen=[15.0 * ms],
        tclose=[20.0 * ms],
        f=15.0 * Hz,
        phase=0.0 * deg,
        distance=15 * meter,
        name="chopper2",
    )

    detector = tof.Detector(distance=20 * meter, name="detector")

    # Make a pulse with 3 neutrons with 2 neutrons going through the first chopper
    # opening and only one making it through the second chopper.
    pulse = make_pulse(
        arrival_times=sc.concat(
            [1.5 * topen, 0.5 * (topen + tclose), 1.1 * tclose], dim='event'
        ),
        distance=chopper1.distance,
    )

    model = tof.Model(pulse=pulse, choppers=[chopper1, chopper2], detectors=[detector])
    model.run()

    assert len(chopper1.tofs.visible) == 2
    assert len(chopper1.tofs.blocked) == 1
    assert sc.isclose(
        chopper1.tofs.visible.data.coords['tof'][0], (1.5 * topen).to(unit='us')
    )
    assert sc.isclose(
        chopper1.tofs.visible.data.coords['tof'][1],
        (0.5 * (topen + tclose)).to(unit='us'),
    )
    assert sc.isclose(
        chopper1.tofs.blocked.data.coords['tof'][0], (1.1 * tclose).to(unit='us')
    )
    assert len(chopper2.tofs.visible) == 1
    # Blocks only one neutron, the other is blocked by chopper1
    assert len(chopper2.tofs.blocked) == 1
    assert sc.isclose(
        chopper2.tofs.visible.data.coords['tof'][0],
        (pulse.wavelengths[1] * chopper2.distance * tof.utils.alpha).to(unit='us'),
    )
    assert sc.isclose(
        chopper2.tofs.blocked.data.coords['tof'][0],
        (pulse.wavelengths[0] * chopper2.distance * tof.utils.alpha).to(unit='us'),
    )
    assert len(detector.tofs.visible) == 1
    assert sc.isclose(
        detector.tofs.visible.data.coords['tof'][0],
        (pulse.wavelengths[1] * detector.distance * tof.utils.alpha).to(unit='us'),
    )


def test_two_choppers_one_and_two_openings():
    topen = 5.0 * ms
    tclose = 16.0 * ms
    chopper1 = make_chopper(
        topen=[topen],
        tclose=[tclose],
        f=10.0 * Hz,
        phase=0.0 * deg,
        distance=10 * meter,
        name="chopper1",
    )

    chopper2 = make_chopper(
        topen=[9.0 * ms, 15.0 * ms],
        tclose=[12.0 * ms, 20.0 * ms],
        f=15.0 * Hz,
        phase=0.0 * deg,
        distance=15 * meter,
        name="chopper2",
    )

    detector = tof.Detector(distance=20 * meter, name="detector")

    # Make a pulse with 7 neutrons:
    # - 2 neutrons blocked by chopper1
    # - 3 neutrons blocked by chopper2
    # - 2 neutrons detected (one through each of chopper2's openings)
    pulse = make_pulse(
        arrival_times=sc.concat(
            [
                0.9 * topen,
                1.01 * topen,
                1.3 * topen,
                1.8 * topen,
                0.5 * (topen + tclose),
                0.85 * tclose,
                1.1 * tclose,
            ],
            dim='event',
        ),
        distance=chopper1.distance,
    )

    model = tof.Model(pulse=pulse, choppers=[chopper1, chopper2], detectors=[detector])
    model.run()

    assert len(chopper1.tofs.visible) == 5
    assert len(chopper1.tofs.blocked) == 2
    assert len(chopper2.tofs.visible) == 2
    assert len(chopper2.tofs.blocked) == 3
    assert len(detector.tofs.visible) == 2


def test_neutron_conservation():
    N = 100_000
    pulse = tof.Pulse(neutrons=N, kind='ess')

    chopper1 = make_chopper(
        topen=[5.0 * ms],
        tclose=[16.0 * ms],
        f=10.0 * Hz,
        phase=0.0 * deg,
        distance=10 * meter,
        name="chopper1",
    )
    chopper2 = make_chopper(
        topen=[9.0 * ms, 15.0 * ms],
        tclose=[15.0 * ms, 20.0 * ms],
        f=15.0 * Hz,
        phase=0.0 * deg,
        distance=15 * meter,
        name="chopper2",
    )

    detector = tof.Detector(distance=20 * meter, name="detector")
    model = tof.Model(pulse=pulse, choppers=[chopper1, chopper2], detectors=[detector])
    model.run()

    assert (
        chopper1.tofs.visible.data.sum() + chopper1.tofs.blocked.data.sum()
    ).value == N
    assert sc.identical(
        chopper2.tofs.visible.data.sum() + chopper2.tofs.blocked.data.sum(),
        chopper1.tofs.visible.data.sum(),
    )
    assert sc.identical(
        detector.tofs.visible.data.sum(), chopper2.tofs.visible.data.sum()
    )