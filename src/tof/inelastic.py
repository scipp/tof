# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import plopp as pp
import scipp as sc

from .component import Component, ComponentReading
from .utils import var_to_dict, wavelength_to_speed


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
        ax.plot([0, tmax], [self.distance.value] * 2, color="gray", lw=3)
        ax.text(0, self.distance.value, self.name, ha="left", va="bottom", color="gray")


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
    """

    def __init__(
        self,
        distance: sc.Variable,
        name: str,
        delta_e: sc.DataArray,
        seed: int | None = None,
    ):
        self.distance = distance.to(dtype=float, copy=False)
        self.name = name
        if delta_e.ndim != 1:
            raise ValueError("delta_e must be a 1D array.")
        self.probabilities = delta_e.values
        self.probabilities = self.probabilities / self.probabilities.sum()
        dim = delta_e.dim
        self.energies = delta_e.coords[dim]
        # TODO: check for bin edges
        self._noise_scale = (
            0.5 * (self.energies.max() - self.energies.min()).value / (len(delta_e) - 1)
        )
        self.kind = "inelastic_sample"
        self._rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        return f"InelasticSample(name={self.name}, distance={self.distance:c})"

    def plot(self, **kwargs) -> pp.FigureLike:
        return pp.xyplot(self.energies, self.probabilities, **kwargs)

    # def __eq__(self, other: object) -> bool:
    #     if not isinstance(other, Detector):
    #         return NotImplemented
    #     return self.name == other.name and sc.identical(self.distance, other.distance)

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
        return cls(
            distance=sc.scalar(
                params["distance"]["value"], unit=params["distance"]["unit"]
            ),
            name=name,
            delta_e=sc.scalar(
                params["delta_e"]["value"], unit=params["delta_e"]["unit"]
            ),
            seed=params.get("seed"),
        )

    def as_json(self) -> dict:
        """
        Return the inelastic sample as a JSON-serializable dictionary.
        .. versionadded:: 26.03.0
        """
        return {
            'type': 'inelastic_sample',
            'distance': var_to_dict(self.distance),
            'name': self.name,
            'delta_e': var_to_dict(self.delta_e),
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

        Parameters
        ----------
        neutrons:
            The neutrons to which the inelastic sample will be applied.
        time_limit:
            The time limit for the neutrons to be considered as reaching the inelastic
            sample.
        """
        # neutrons.pop("blocked_by_me", None)
        # return neutrons, self.as_readonly(neutrons)
        w_initial = sc.reciprocal(neutrons.coords["wavelength"] ** 2)

        n = neutrons.shape
        inds = self._rng.choice(len(self.energies), size=n, p=self.probabilities)
        de = sc.array(
            dims=w_initial.dims,
            values=self.energies.values[inds]
            + self._rng.normal(scale=self._noise_scale, size=n),
            unit=self.energies.unit,
        )
        # Convert energy change to wavelength change
        # w_initial = sc.reciprocal(neutrons.coords["wavelength"] ** 2)
        w_final = sc.reciprocal(
            sc.sqrt(
                ((2 * sc.constants.m_n / (sc.constants.h**2)) * de).to(
                    unit=w_initial.unit
                )
                + w_initial
            )
        )
        neutrons = neutrons.assign_coords(
            wavelength=w_final, speed=wavelength_to_speed(w_final)
        )
        return neutrons, self.as_readonly(neutrons)
