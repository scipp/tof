# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc
import scipp.constants as const

alpha = const.m_n / const.h


def speed_to_wavelength(x: sc.Variable, unit: str = 'angstrom') -> sc.Variable:
    """
    Convert neutron speeds to wavelengths.

    Parameters
    ----------
    x:
        Input speeds.
    unit:
        The unit of the output wavelengths.
    """
    return (1.0 / (alpha * x)).to(unit=unit)


def wavelength_to_speed(x: sc.Variable, unit: str = 'm/s') -> sc.Variable:
    """
    Convert neutron wavelengths to speeds.

    Parameters
    ----------
    x:
        Input wavelengths.
    unit:
        The unit of the output speeds.
    """
    return (1.0 / (alpha * x)).to(unit=unit)


def speed_to_energy(x: sc.Variable, unit='meV') -> sc.Variable:
    """
    Convert neutron speeds to energies.

    Parameters
    ----------
    x:
        Input speeds.
    unit:
        The unit of the output energies.
    """
    return (const.m_n * x * x).to(unit=unit)


def energy_to_speed(x: sc.Variable, unit='m/s') -> sc.Variable:
    """
    Convert neutron energies to speeds.

    Parameters
    ----------
    x:
        Input energies.
    unit:
        The unit of the output speeds.
    """
    return sc.sqrt(x / const.m_n).to(unit=unit)
