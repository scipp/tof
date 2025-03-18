# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass

import plopp as pp
import scipp as sc

from .utils import Plot, one_mask


@dataclass(frozen=True)
class ReadingField:
    data: sc.DataArray
    dim: str

    def plot(self, bins: int = 300, **kwargs):
        by_pulse = sc.collapse(self.data, keep="event")
        to_plot = {}
        color = {}
        for i, (key, da) in enumerate(by_pulse.items()):
            sel = da[~da.masks["blocked_by_others"]]
            to_plot[key] = sel.hist({self.dim: bins})
            if "blocked_by_me" in self.data.masks:
                name = f"blocked-{key}"
                to_plot[name] = (
                    da[da.masks["blocked_by_me"]]
                    .drop_masks(list(da.masks.keys()))
                    .hist({self.dim: to_plot[key].coords[self.dim]})
                )
                color[name] = "gray"
        return pp.plot(to_plot, **{**{"color": color}, **kwargs})

    def min(self):
        mask = ~one_mask(self.data.masks)
        mask.unit = ""
        return (self.data.coords[self.dim] * mask).min()

    def max(self):
        mask = ~one_mask(self.data.masks)
        mask.unit = ""
        return (self.data.coords[self.dim] * mask).max()

    def __repr__(self) -> str:
        mask = ~one_mask(self.data.masks)
        mask.unit = ""
        coord = self.data.coords[self.dim] * mask
        return (
            f"{self.dim}: min={coord.min():c}, max={coord.max():c}, "
            f"events={int(self.data.sum().value)}"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, val):
        return self.__class__(data=self.data[val], dim=self.dim)


def _make_reading_field(da: sc.DataArray, dim: str) -> ReadingField:
    return ReadingField(
        data=sc.DataArray(
            data=da.data,
            coords={dim: da.coords[dim]},
            masks=da.masks,
        ),
        dim=dim,
    )


class ComponentReading:
    """
    Data reading for a component placed in the beam path. The reading will have a
    record of the arrival times and wavelengths of the neutrons that passed through it.
    """

    @property
    def toa(self) -> ReadingField:
        return _make_reading_field(self.data, dim="toa")

    @property
    def wavelength(self) -> ReadingField:
        return _make_reading_field(self.data, dim="wavelength")

    @property
    def birth_time(self) -> ReadingField:
        return _make_reading_field(self.data, dim="birth_time")

    @property
    def speed(self) -> ReadingField:
        return _make_reading_field(self.data, dim="speed")

    def plot(self, bins: int = 300) -> Plot:
        """
        Plot both the toa and wavelength data side by side.

        Parameters
        ----------
        bins:
            Number of bins to use for histogramming the neutrons.
        """
        return self.toa.plot(bins=bins) + self.wavelength.plot(bins=bins)
