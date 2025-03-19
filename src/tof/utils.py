# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass
from functools import reduce
from types import MappingProxyType

import matplotlib.pyplot as plt
import scipp as sc
import scipp.constants as const

m_over_h = const.m_n / const.h
two_pi = sc.constants.pi * (2.0 * sc.units.rad)


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
    return (1.0 / (m_over_h * x)).to(unit=unit)


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
    return (1.0 / (m_over_h * x)).to(unit=unit)


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


def one_mask(
    masks: MappingProxyType[str, sc.Variable], unit: str | None = None
) -> sc.Variable:
    """
    Combine multiple masks into a single mask.

    Parameters
    ----------
    masks:
        The masks to combine.
    unit:
        The unit of the output mask.
    """
    out = reduce(lambda a, b: a | b, masks.values())
    out.unit = unit
    return out


@dataclass(frozen=True)
class FacilityPulse:
    birth_time: sc.DataArray
    wavelength: sc.DataArray
    frequency: sc.Variable


@dataclass
class Plot:
    ax: plt.Axes
    fig: plt.Figure
