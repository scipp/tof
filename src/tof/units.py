# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc
import scipp.constants as const


class Units:
    def __init__(self):
        self.alpha = const.m_n / const.h

        self.Hz = sc.Unit('Hz')
        self.s = sc.Unit('s')
        self.us = sc.Unit('us')
        self.angstrom = sc.Unit('angstrom')
        self.m = sc.Unit('m')
        self.deg = sc.Unit('deg')
        self.rad = sc.Unit('rad')

    def speed_to_wavelength(
        self, x: sc.Variable, unit: str = 'angstrom'
    ) -> sc.Variable:
        return (1.0 / (self.alpha * x)).to(unit=unit)

    def wavelength_to_speed(self, x: sc.Variable, unit: str = 'm/s') -> sc.Variable:
        return (1.0 / (self.alpha * x)).to(unit=unit)

    def speed_to_energy(self, x: sc.Variable, unit='meV') -> sc.Variable:
        return (const.m_n * x * x).to(unit=unit)

    def energy_to_speed(self, x: sc.Variable, unit='m/s') -> sc.Variable:
        return sc.sqrt(x / const.m_n).to(unit=unit)


units = Units()
