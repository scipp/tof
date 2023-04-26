# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import scipp as sc

import tof

Hz = sc.Unit('Hz')
deg = sc.Unit('deg')
meter = sc.Unit('m')
ms = sc.Unit('ms')


def test_one_chopper_one_opening():
    # Make a chopper with settings to that it is open from 10-20 ms. Assume zero phase.
    topen = 10.0 * ms
    tclose = 20.0 * ms
    f = 10.0 * Hz
    aopen = sc.constants.pi * (2.0 * sc.units.rad) * topen.to(unit='s') * f
    aclose = sc.constants.pi * (2.0 * sc.units.rad) * tclose.to(unit='s') * f
    chopper = tof.Chopper(
        frequency=10.0 * Hz,
        open=aopen.flatten(to='cutout'),
        close=aclose.flatten(to='cutout'),
        phase=0.0 * deg,
        distance=10 * meter,
        name="chopper",
    )
    detector = tof.Detector(distance=20 * meter, name="detector")

    # Make a pulse with 3 neutrons with one neutron going through the chopper opening
    # and the other two neutrons on either side of the opening.
    # Arrival times are distance * alpha * wavelength
    alpha = tof.utils.alpha
    times = sc.concat([0.9 * topen, 0.5 * (topen + tclose), 1.1 * tclose], dim='event')
    pulse = tof.Pulse(
        birth_times=sc.array(
            dims=['event'],
            values=[0.0, 0.0, 0.0],
            unit='s',
        ),
        wavelengths=times.to(unit='s') / (chopper.distance * alpha),
    )

    model = tof.Model(pulse=pulse, choppers=choppers, detectors=detectors)
    model.run()
