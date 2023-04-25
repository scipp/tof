# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Union

import plopp as pp
import scipp as sc


class Data:
    def __init__(self, data: sc.DataArray, dim: str):
        self._data = data
        self._dim = dim

    @property
    def data(self) -> sc.DataArray:
        return self._data

    def plot(self, bins: Union[int, sc.Variable] = 300):
        return self._data.hist({self._dim: bins}).plot()

    def __repr__(self) -> str:
        return f"Data(data={self._data})"


class ComponentData:
    def __init__(self, data: sc.Variable, mask: sc.Variable, dim: str):
        self._data = data
        self._mask = mask
        self._dim = dim

    @property
    def visible(self) -> Data:
        a = self._data[self._mask]
        return Data(
            data=sc.DataArray(
                data=sc.ones(sizes=a.sizes, unit='counts'), coords={self._dim: a}
            ),
            dim=self._dim,
        )

    @property
    def blocked(self) -> Data:
        a = self._data[~self._mask]
        return Data(
            data=sc.DataArray(
                data=sc.ones(sizes=a.sizes, unit='counts'), coords={self._dim: a}
            ),
            dim=self._dim,
        )

    @property
    def data(self) -> sc.DataGroup:
        return sc.DataGroup(
            {'visible': self.visible.data, 'blocked': self.blocked.data}
        )

    def __repr__(self) -> str:
        return (
            f"ComponentData(visible={sc.sum(self._mask).value}, "
            f"blocked={sc.sum(~self._mask).value}, dim={self._dim})"
        )

    def plot(self, bins: Union[int, sc.Variable] = 300):
        visible = self.visible.data
        blocked = self.blocked.data
        if isinstance(bins, int):
            bins = sc.linspace(
                dim=self._dim,
                start=min(
                    visible.coords[self._dim].min(), blocked.coords[self._dim].min()
                ).value,
                stop=max(
                    visible.coords[self._dim].max(), blocked.coords[self._dim].max()
                ).value,
                num=bins,
                unit=visible.coords[self._dim].unit,
            )
        return pp.plot(
            {
                'visible': visible.hist({self._dim: bins}),
                'blocked': blocked.hist({self._dim: bins}),
            },
            color={'blocked': 'gray'},
        )


class Component:
    def __init__(self):
        self._arrival_times = None
        self._wavelengths = None
        self._mask = None

    @property
    def tofs(self) -> sc.Variable:
        return ComponentData(
            data=self._arrival_times.to(unit='us'), mask=self._mask, dim='tof'
        )

    @property
    def wavelengths(self) -> sc.Variable:
        return ComponentData(data=self._wavelengths, mask=self._mask, dim='wavelength')
