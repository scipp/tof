# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc

import tof

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
