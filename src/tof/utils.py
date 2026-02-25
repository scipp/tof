# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass
from functools import reduce
from types import MappingProxyType

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc
import scipp.constants as const

m_over_h = const.m_n / const.h
two_pi = sc.constants.pi * (2.0 * sc.units.rad)


def speed_to_wavelength(x: sc.Variable, unit: str = "angstrom") -> sc.Variable:
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


def wavelength_to_speed(x: sc.Variable, unit: str = "m/s") -> sc.Variable:
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


def speed_to_energy(x: sc.Variable, unit="meV") -> sc.Variable:
    """
    Convert neutron speeds to energies.

    Parameters
    ----------
    x:
        Input speeds.
    unit:
        The unit of the output energies.
    """
    return ((0.5 * const.m_n) * x * x).to(unit=unit)


def energy_to_speed(x: sc.Variable, unit="m/s") -> sc.Variable:
    """
    Convert neutron energies to speeds.

    Parameters
    ----------
    x:
        Input energies.
    unit:
        The unit of the output speeds.
    """
    return sc.sqrt(x / (0.5 * const.m_n)).to(unit=unit)


def wavelength_to_energy(x: sc.Variable, unit="meV") -> sc.Variable:
    """
    Convert neutron wavelengths to energies.

    Parameters
    ----------
    x:
        Input wavelengths.
    unit:
        The unit of the output energies.
    """
    return speed_to_energy(wavelength_to_speed(x)).to(unit=unit)


def energy_to_wavelength(x: sc.Variable, unit="angstrom") -> sc.Variable:
    """
    Convert neutron energies to wavelengths.

    Parameters
    ----------
    x:
        Input energies.
    unit:
        The unit of the output wavelengths.
    """
    return speed_to_wavelength(energy_to_speed(x)).to(unit=unit)


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


def var_to_dict(var: sc.Variable) -> dict:
    """
    Convert a scipp Variable to a dictionary with 'value' and 'unit' keys.

    Parameters
    ----------
    var:
        The variable to convert.
    """
    return {
        "value": var.values.tolist() if var.ndim > 0 else float(var.value),
        "unit": str(var.unit),
    }


def var_from_dict(data: dict, dim: str | None = None) -> sc.Variable:
    """
    Convert a dictionary with 'value' and 'unit' keys to a scipp Variable.

    Parameters
    ----------
    data:
        The dictionary to convert.
    dim:
        The dimension of the output variable (non-scalar data only).
    """
    values = np.asarray(data["value"])
    unit = data['unit']
    if values.shape:
        if dim is None:
            raise ValueError("Missing dimension to construct variable from json.")
        return sc.array(dims=[dim], values=values, unit=unit)
    return sc.scalar(values, unit=unit)


def extract_component_group(
    components: dict | MappingProxyType, kind: str
) -> MappingProxyType:
    """
    Extract a group of components of a given kind from a dictionary of components.

    Parameters
    ----------
    components:
        The components to extract from.
    kind:
        The kind of components to extract.
    """
    return MappingProxyType(
        {name: comp for name, comp in components.items() if kind in comp.kind}
    )


@dataclass
class Plot:
    ax: plt.Axes
    fig: plt.Figure


@dataclass
class PulseProfile:
    """
    Parameters of a neutron pulse (typically from a neutron facility).

    Parameters
    ----------
    frequency:
        Frequency of the pulse in Hz.
    birth_time:
        Probability distribution of neutrons in time within a single pulse.
    wavelength:
        Probability distribution of neutrons in wavelength within a single pulse.
    """

    frequency: sc.Variable
    birth_time: sc.DataArray
    wavelength: sc.DataArray
