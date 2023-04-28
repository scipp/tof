# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import plopp as pp
import scipp as sc

from .utils import Plot


class Data:
    """
    A data object contains the data (visible or blocked) for a component data
    (time-of-flight or wavelengths).

    Parameters
    ----------
    data:
        The data to hold.
    dim:
        The dimension label of the data.
    """

    def __init__(self, data: sc.DataArray, dim: str):
        self._data = data
        self._dim = dim

    @property
    def data(self) -> sc.DataArray:
        """
        The underlying data.
        """
        return self._data

    @property
    def shape(self) -> Tuple[int]:
        """
        The shape of the data.
        """
        return self._data.shape

    @property
    def sizes(self) -> Dict[str, int]:
        """
        The sizes of the data.
        """
        return self._data.sizes

    def __len__(self) -> int:
        """
        The length of the data.
        """
        return len(self._data)

    def plot(self, bins: Union[int, sc.Variable] = 300, **kwargs):
        """
        Plot the neutrons that reach the component as a histogram.

        Parameters
        ----------
        bins:
            The bins to use for histogramming the neutrons.
        """
        return self._data.hist({self._dim: bins}).plot(**kwargs)

    def __repr__(self) -> str:
        return f"Data(data={self._data})"


class ComponentData:
    """
    A component data object contains the data (time-of-flight or wavelengths) for a
    component.

    Parameters
    ----------
    data:
        The data to hold.
    mask:
        A mask to apply to the data which will select the neutrons that pass through
        the component.
    dim:
        The dimension label of the data.
    blocking:
        A mask to apply to the data which will select the neutrons that are blocked by
        the component (this only defined in the case of a :class:`Chopper` component
        since a :class:`Detector` does not block any neutrons).
    """

    def __init__(
        self,
        data: sc.Variable,
        mask: sc.Variable,
        dim: str,
        blocking: Optional[sc.Variable] = None,
    ):
        self._data = data
        self._mask = mask
        self._blocking = blocking
        self._dim = dim

    @property
    def visible(self) -> Data:
        """
        The data for the neutrons that pass through the component.
        """
        a = self._data[self._mask]
        return Data(
            data=sc.DataArray(
                data=sc.ones(sizes=a.sizes, unit='counts'), coords={self._dim: a}
            ),
            dim=self._dim,
        )

    @property
    def blocked(self) -> Data:
        """
        The data for the neutrons that are blocked by the component.
        """
        if self._blocking is None:
            return
        a = self._data[self._blocking]
        return Data(
            data=sc.DataArray(
                data=sc.ones(sizes=a.sizes, unit='counts'), coords={self._dim: a}
            ),
            dim=self._dim,
        )

    @property
    def data(self) -> sc.DataGroup:
        """
        The neutrons that reach the component, split up into those that are blocked by
        the component and those that are not.
        """
        out = {'visible': self.visible.data}
        if self._blocking is not None:
            out['blocked'] = self.blocked.data
        return sc.DataGroup(out)

    def __repr__(self) -> str:
        return (
            f"ComponentData(visible={sc.sum(self._mask).value}, "
            f"blocked={sc.sum(~self._mask).value}, dim={self._dim})"
        )

    def plot(self, bins: Union[int, sc.Variable] = 300, **kwargs):
        """
        Plot the data for the neutrons that reach the component, split up into those
        that are blocked by the component and those that are not.

        Parameters
        ----------
        bins:
            The bins to use for histogramming the neutrons.
        """
        if self._blocking is None:
            return self.visible.plot(bins=bins, **kwargs)
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
            **{**{'color': {'blocked': 'gray'}}, **kwargs},
        )


class Component:
    """
    A component is placed in the beam path. After the model is run, the component
    will have a record of the arrival times and wavelengths of the neutrons that
    passed through it.
    """

    def __init__(self):
        self._arrival_times = None
        self._wavelengths = None
        self._mask = None
        self._own_mask = None

    @property
    def tofs(self) -> ComponentData:
        """
        The arrival times of the neutrons that passed through the component.
        """
        return ComponentData(
            data=self._arrival_times.to(unit='us'),
            mask=self._mask,
            blocking=self._own_mask,
            dim='tof',
        )

    @property
    def wavelengths(self) -> ComponentData:
        """
        The wavelengths of the neutrons that passed through the component.
        """
        return ComponentData(
            data=self._wavelengths,
            mask=self._mask,
            blocking=self._own_mask,
            dim='wavelength',
        )

    def plot(self, bins: int = 300) -> tuple:
        """
        Plot both the tof and wavelength data side by side.

        Parameters
        ----------
        bins:
            Number of bins to use for histogramming the neutrons.
        """
        fig, ax = plt.subplots(1, 2)
        self.tofs.plot(bins=bins, ax=ax[0])
        self.wavelengths.plot(bins=bins, ax=ax[1])
        size = fig.get_size_inches()
        fig.set_size_inches(size[0] * 2, size[1])
        fig.tight_layout()
        return Plot(fig=fig, ax=ax)
