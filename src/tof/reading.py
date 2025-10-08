# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

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
        for key, da in by_pulse.items():
            sel = da[~da.masks["blocked_by_others"]]
            if sel.size == 0:
                continue
            to_plot[key] = sel.hist({self.dim: bins})
            if "blocked_by_me" in self.data.masks:
                name = f"blocked-{key}"
                to_plot[name] = (
                    da[da.masks["blocked_by_me"]]
                    .drop_masks(list(da.masks.keys()))
                    .hist({self.dim: to_plot[key].coords[self.dim]})
                )
                color[name] = "gray"
        if not to_plot:
            raise RuntimeError("Nothing to plot.")
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

    def __getitem__(self, val: int | slice | tuple[str, int | slice]) -> ReadingField:
        if isinstance(val, int):
            val = ('pulse', val)
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
        """
        Time of arrival of the neutrons at the component.
        """
        return _make_reading_field(self.data, dim="toa")

    @property
    def eto(self) -> ReadingField:
        """
        Event time offset of the neutrons at the component (= toa modulo pulse period).
        """
        return _make_reading_field(self.data, dim="eto")

    @property
    def wavelength(self) -> ReadingField:
        """
        Wavelength of the neutrons at the component.
        """
        return _make_reading_field(self.data, dim="wavelength")

    @property
    def birth_time(self) -> ReadingField:
        """
        Birth time of the neutrons at the source.
        """
        return _make_reading_field(self.data, dim="birth_time")

    @property
    def speed(self) -> ReadingField:
        """
        Speed of the neutrons at the component.
        """
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
