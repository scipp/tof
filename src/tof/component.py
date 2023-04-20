# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc


class Component:
    @property
    def tofs(self) -> sc.Variable:
        t = self._arrival_times[self._mask].to(unit='us')
        return sc.DataArray(
            data=sc.ones(sizes=t.sizes, unit='counts'), coords={'tof': t}
        )

    @property
    def wavelengths(self) -> sc.Variable:
        w = self._wavelengths[self._mask]
        return sc.DataArray(
            data=sc.ones(sizes=w.sizes, unit='counts'),
            coords={'wavelength': w},
        )

    def plot(self, bins: int = 300):
        return self.tofs.hist(tof=bins).plot()
