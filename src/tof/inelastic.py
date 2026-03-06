# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import plopp as pp
import scipp as sc

from .component import Component, ComponentReading
from .utils import (
    energy_to_wavelength,
    var_to_dict,
    wavelength_to_energy,
    wavelength_to_speed,
)


@dataclass(frozen=True)
class InelasticSampleReading(ComponentReading):
    """
    Read-only container for the neutrons that reach the inelastic sample.
    """

    distance: sc.Variable
    name: str
    data: sc.DataArray

    @property
    def kind(self) -> str:
        return "inelastic_sample"

    def _repr_stats(self) -> str:
        return f"visible={int(self.data.sum().value)}"

    def __repr__(self) -> str:
        return f"""InelasticSampleReading: '{self.name}'
  distance: {self.distance:c}
  neutrons: {self._repr_stats()}
"""

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(
        self, val: int | slice | tuple[str, int | slice]
    ) -> InelasticSampleReading:
        if isinstance(val, int):
            val = ('pulse', val)
        return replace(self, data=self.data[val])

    def plot_on_time_distance_diagram(self, ax, tmax) -> None:
        ax.plot([0, tmax], [self.distance.value] * 2, color="tab:brown", lw=4)
        ax.text(
            0, self.distance.value, self.name, ha="left", va="bottom", color="tab:brown"
        )


class InelasticSample(Component):
    """
    An inelastic sample component changes the energy of the neutrons that pass through
    it, but does not block any.

    Parameters
    ----------
    distance:
        The distance from the source to the inelastic sample.
    name:
        The name of the inelastic sample.
    delta_e:
        The change in energy applied to the neutrons as they pass through the
        inelastic sample. This should be a function or callable that takes the
        incident energy and returns the final energy.
    """

    def __init__(
        self,
        distance: sc.Variable,
        name: str,
        delta_e: Callable[[sc.Variable], sc.Variable],
    ):
        self.distance = distance.to(dtype=float, copy=False)
        self.name = name
        self.delta_e = delta_e
        self.kind = "inelastic_sample"

    def __repr__(self) -> str:
        return f"InelasticSample(name={self.name}, distance={self.distance:c})"

    def plot(self, **kwargs) -> pp.FigureLike:
        return pp.xyplot(self.energies, self.probabilities, **kwargs)

    def as_dict(self) -> dict:
        """
        Return the inelastic sample as a dictionary.
        """
        return {'distance': self.distance, 'name': self.name, 'delta_e': self.delta_e}

    @classmethod
    def from_json(cls, name: str, params: dict) -> InelasticSample:
        """
        Create an inelastic sample from a JSON-serializable dictionary.
        """
        raise NotImplementedError

    def as_json(self) -> dict:
        """
        Return the inelastic sample as a JSON-serializable dictionary.
        .. versionadded:: 26.03.0
        """
        return {
            'type': 'inelastic_sample',
            'distance': var_to_dict(self.distance),
            'name': self.name,
        }

    def as_readonly(self, neutrons: sc.DataArray) -> InelasticSampleReading:
        return InelasticSampleReading(
            distance=self.distance, name=self.name, data=neutrons
        )

    def apply(
        self, neutrons: sc.DataArray, time_limit: sc.Variable
    ) -> tuple[sc.DataArray, InelasticSampleReading]:
        """
        Apply the change in energy to the given neutrons.
        The convention is that

        .. math::

            \\Delta E = E_i - E_f

        where :math:`E_i` is the initial energy and :math:`E_f` is the final energy.

        Neutrons that would end up with a negative final energy are removed from the
        output by setting their wavelength to NaN. This is done to avoid issues with
        neutrons that would have a negative final energy being plotted with a very large
        wavelength.

        Parameters
        ----------
        neutrons:
            The neutrons to which the inelastic sample will be applied.
        time_limit:
            The time limit for the neutrons to be considered as reaching the inelastic
            sample.
        """
        incident_wavelength = neutrons.coords["wavelength"]
        incident_energy = wavelength_to_energy(incident_wavelength)
        final_energy = self.delta_e(incident_energy)
        final_energy = sc.where(
            final_energy < sc.scalar(0.0, unit=final_energy.unit),
            sc.scalar(float("nan"), unit=final_energy.unit),
            final_energy,
        )
        w_final = energy_to_wavelength(final_energy, unit=incident_wavelength.unit)

        neutrons = neutrons.assign_coords(
            wavelength=w_final, speed=wavelength_to_speed(w_final)
        )
        return neutrons, self.as_readonly(neutrons)
