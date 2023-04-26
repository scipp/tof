# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc

import tof

Hz = sc.Unit('Hz')
deg = sc.Unit('deg')
meter = sc.Unit('m')


def test_one_chopper_one_opening():
    pulse = tof.Pulse(kind='ess', neutrons=100_000)
    choppers = [
        tof.Chopper(
            frequency=70.0 * Hz,
            open=sc.array(
                dims=['cutout'],
                values=[0],
                unit='deg',
            ),
            close=sc.array(
                dims=['cutout'],
                values=[30],
                unit='deg',
            ),
            phase=10 * deg,
            distance=10 * meter,
            name="chopper",
        ),
    ]
    detectors = [tof.Detector(distance=10 * meter, name="detector")]
    model = tof.Model(pulse=pulse, choppers=choppers, detectors=detectors)
    model.run()
