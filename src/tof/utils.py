# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc
import scipp.constants as const

alpha = const.m_n / const.h


def speed_to_wavelength(x: sc.Variable, unit: str = 'angstrom') -> sc.Variable:
    return (1.0 / (alpha * x)).to(unit=unit)


def wavelength_to_speed(x: sc.Variable, unit: str = 'm/s') -> sc.Variable:
    return (1.0 / (alpha * x)).to(unit=unit)


def speed_to_energy(x: sc.Variable, unit='meV') -> sc.Variable:
    return (const.m_n * x * x).to(unit=unit)


def energy_to_speed(x: sc.Variable, unit='m/s') -> sc.Variable:
    return sc.sqrt(x / const.m_n).to(unit=unit)
