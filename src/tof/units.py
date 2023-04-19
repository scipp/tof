# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Union

import numpy as np

mass = 1.674927471e-27  # Neutron mass in kg
alpha = 2.5278e-4  # Neutron mass over Planck constant
mev = 1.602176634e-22  # meV to Joule


def deg_to_rad(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.radians(x)


def rad_to_deg(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.degrees(x)


def us_to_s(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return x * 1.0e-6


def s_to_us(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return x * 1.0e6


def speed_to_wavelength(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 1.0 / (alpha * x)


def wavelength_to_speed(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 1.0 / (alpha * x)


def speed_to_mev(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return mass * x * x / mev


def mev_to_speed(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.sqrt(mev * x / mass)
