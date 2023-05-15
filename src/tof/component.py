# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import plopp as pp
import scipp as sc

from .utils import Plot


@dataclass(frozen=True)
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

    data: sc.DataArray
    dim: str

    @property
    def shape(self) -> Tuple[int]:
        """
        The shape of the data.
        """
        return self.data.shape

    @property
    def sizes(self) -> Dict[str, int]:
        """
        The sizes of the data.
        """
        return self.data.sizes

    def __len__(self) -> int:
        """
        The length of the data.
        """
        return len(self.data)

    def plot(self, bins: Union[int, sc.Variable] = 300, **kwargs):
        """
        Plot the neutrons that reach the component as a histogram.

        Parameters
        ----------
        bins:
            The bins to use for histogramming the neutrons.
        """
        return self.data.hist({self.dim: bins}).plot(**kwargs)

    def __repr__(self) -> str:
        coord = self.data.coords[self.dim]
        return (
            f"Data(dim='{self.dim}', events={len(self)}, "
            f"min={coord.min():c}, max={coord.max():c})"
        )


@dataclass(frozen=True)
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

    visible: Data
    blocked: Optional[Data]

    @property
    def data(self) -> sc.DataGroup:
        """
        The neutrons that reach the component, split up into those that are blocked by
        the component and those that are not.
        """
        out = {'visible': self.visible.data}
        if self.blocked is not None:
            out['blocked'] = self.blocked.data
        return sc.DataGroup(out)

    def __repr__(self) -> str:
        return (
            f"ComponentData(dim='{self.visible.dim}', visible={len(self.visible)}, "
            f"blocked={len(self.blocked) if self.blocked is not None else None})"
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
        if self.blocked is None:
            return self.visible.plot(bins=bins, **kwargs)
        visible = self.visible.data
        blocked = self.blocked.data
        dim = self.visible.dim
        if isinstance(bins, int):
            bins = sc.linspace(
                dim=dim,
                start=min(visible.coords[dim].min(), blocked.coords[dim].min()).value,
                stop=max(visible.coords[dim].max(), blocked.coords[dim].max()).value,
                num=bins,
                unit=visible.coords[dim].unit,
            )
        return pp.plot(
            {
                'visible': visible.hist({dim: bins}),
                'blocked': blocked.hist({dim: bins}),
            },
            **{**{'color': {'blocked': 'gray'}}, **kwargs},
        )


class Component:
    """
    A component is placed in the beam path. After the model is run, the component
    will have a record of the arrival times and wavelengths of the neutrons that
    passed through it.
    """

    # @property
    # def data(self) -> sc.DataGroup:
    #     """ """
    #     return sc.DataGroup(
    #         {
    #             'tofs': self.tofs.data,
    #             'wavelengths': self.wavelengths.data,
    #             'birth_times': self.birth_times.data,
    #             'speeds': self.speeds.data,
    #         }
    #     )

    def plot(self, bins: int = 300) -> Plot:
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
