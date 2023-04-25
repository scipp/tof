# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Union

import plopp as pp
import scipp as sc


class Component:
    def __init__(self):
        self._arrival_times = None
        self._wavelengths = None
        self._mask = None

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

    @property
    def blocked_tofs(self) -> sc.Variable:
        t = self._arrival_times[~self._mask].to(unit='us')
        return sc.DataArray(
            data=sc.ones(sizes=t.sizes, unit='counts'), coords={'tof': t}
        )

    @property
    def blocked_wavelengths(self) -> sc.Variable:
        w = self._wavelengths[~self._mask]
        return sc.DataArray(
            data=sc.ones(sizes=w.sizes, unit='counts'),
            coords={'wavelength': w},
        )

    def plot(self, bins: Union[int, sc.Variable] = 300, show_blocked: bool = False):
        tofs = self.tofs
        btofs = self.blocked_tofs
        if show_blocked:
            if isinstance(bins, int):
                bins = sc.linspace(
                    dim='tof',
                    start=min(
                        tofs.coords['tof'].min(), btofs.coords['tof'].min()
                    ).value,
                    stop=max(tofs.coords['tof'].max(), btofs.coords['tof'].max()).value,
                    num=bins,
                    unit=tofs.coords['tof'].unit,
                )
            return pp.plot(
                {
                    'visible': self.tofs.hist(tof=bins),
                    'blocked': self.blocked_tofs.hist(tof=bins),
                },
                color={'blocked': 'gray'},
            )
        else:
            return self.tofs.hist(tof=bins).plot()
