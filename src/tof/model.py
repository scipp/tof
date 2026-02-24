# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import warnings
from itertools import chain

import scipp as sc

from .chopper import Chopper
from .component import Component
from .detector import Detector
from .result import Result
from .source import Source

ComponentType = Chopper | Detector


def make_beamline(instrument: dict) -> dict[str, list[Chopper] | list[Detector]]:
    """
    Create choppers and detectors from a dictionary.
    The dictionary is typically loaded from a pre-configured instrument library, or
    from a JSON file.

    Parameters
    ----------
    instrument:
        A dictionary defining the instrument components.
        Each key is the name of a component, and the value is a dictionary with the
        component parameters. Each component dictionary must have a "type" key, which
        must be either "chopper" or "detector". Other keys depend on the component
        type, see the documentation of the :class:`Chopper` and :class:`Detector`
        classes for details.
    """
    components = []
    mapping = {"chopper": Chopper, "detector": Detector}
    for name, comp in instrument.items():
        if comp["type"] == "source":
            continue
        if comp["type"] not in mapping:
            raise ValueError(
                f"Unknown component type: {comp['type']} for component {name}. "
            )
        components.append(mapping[comp["type"]].from_json(name=name, params=comp))
    return components


class Model:
    """
    A class that represents a neutron instrument.
    It is defined by a source and a list of components (choppers, detectors, etc.).

    Parameters
    ----------
    source:
        A source of neutrons.
    components:
        A list of components.
    choppers:
        A list of choppers. This is kept for backwards-compatibility; new code
        should use the `components` parameter instead.
    detectors:
        A list of detectors. This is kept for backwards-compatibility; new code
        should use the `components` parameter instead.
    """

    def __init__(
        self,
        source: Source | None = None,
        components: list[Component] | tuple[Component, ...] | None = None,
        choppers: list[Chopper] | tuple[Chopper, ...] | None = None,
        detectors: list[Detector] | tuple[Detector, ...] | None = None,
    ):
        self.source = source
        self.components = {}
        for comp in chain((choppers or ()), (detectors or ()), (components or ())):
            self.add(comp)

    @classmethod
    def from_json(cls, filename: str) -> Model:
        """
        Create a model from a JSON file.

        Currently, only sources from facilities are supported when loading from JSON.

        .. versionadded:: 25.10.1

        Parameters
        ----------
        filename:
            The path to the JSON file.
        """
        import json

        with open(filename) as f:
            instrument = json.load(f)
        beamline = make_beamline(instrument)
        source = None
        for item in instrument.values():
            if item.get("type") == "source":
                if "facility" not in item:
                    raise ValueError(
                        "Currently, only sources from facilities are supported when "
                        "loading from JSON."
                    )
                source_args = item.copy()
                del source_args["type"]
                source = Source(**source_args)
                break
        return cls(source=source, **beamline)

    def as_json(self) -> dict:
        """
        Return the model as a JSON-serializable dictionary.
        If the source is not from a facility, it is not included in the output.

        .. versionadded:: 25.11.0
        """
        instrument_dict = {}
        if self.source is not None:
            if self.source.facility is None:
                warnings.warn(
                    "The source is not from a facility, so it will not be included in "
                    "the JSON output.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                instrument_dict['source'] = self.source.as_json()
        for comp in self.components.values():
            instrument_dict[comp.name] = comp.as_json()
        return instrument_dict

    def to_json(self, filename: str) -> None:
        """
        Save the model to a JSON file.
        If the source is not from a facility, it is not included in the output.

        .. versionadded:: 25.11.0

        Parameters
        ----------
        filename:
            The path to the JSON file.
        """
        import json

        with open(filename, 'w') as f:
            json.dump(self.as_json(), f, indent=2)

    def add(self, component: Component) -> None:
        """
        Add a component to the instrument.
        Component names must be unique across choppers and detectors.
        The name "source" is reserved for the source, and can thus not be used for other
        components.

        Parameters
        ----------
        component:
            A chopper or detector.
        """
        # Note that the name "source" is reserved for the source.
        if component.name in (*self.components, "source"):
            raise KeyError(
                f"Component with name {component.name} already exists. "
                "If you wish to replace/update an existing component, use "
                "``model.components['name'] = new_component``."
            )
        self.components[component.name] = component

    def remove(self, name: str):
        """
        Remove a component.

        Parameters
        ----------
        name:
            The name of the component to remove.
        """
        del self.components[name]

    def run(self) -> Result:
        """
        Run the simulation.
        """
        if self.source is None:
            raise ValueError(
                "No source has been defined for this model. Please add a source using "
                "`model.source = Source(...)` before running the simulation."
            )
        components = sorted(self.components.values(), key=lambda c: c.distance.value)

        if len(components) == 0:
            raise ValueError("Cannot run model: no components have been defined.")

        if components[0].distance < self.source.distance:
            raise ValueError(
                "At least one component is located before the source "
                "itself. Please check the distances of the components."
            )

        neutrons = self.source.data.copy(deep=False)
        neutrons.masks["blocked_by_others"] = sc.zeros(
            sizes=neutrons.sizes, unit=None, dtype=bool
        )
        neutrons.coords.update(
            distance=self.source.distance, toa=neutrons.coords['birth_time']
        )

        time_unit = neutrons.coords['birth_time'].unit

        readings = {}
        time_limit = (
            neutrons.coords['birth_time']
            + (
                (components[-1].distance - self.source.distance)
                / neutrons.coords['speed']
            ).to(unit=time_unit)
        ).max()
        for comp in components:
            neutrons = neutrons.copy(deep=False)
            toa = neutrons.coords['toa'] + (
                (comp.distance - neutrons.coords['distance']) / neutrons.coords['speed']
            ).to(unit=time_unit, copy=False)
            neutrons.coords['toa'] = toa
            neutrons.coords['eto'] = toa % (1 / self.source.frequency).to(
                unit=time_unit, copy=False
            )
            neutrons.coords['distance'] = comp.distance

            if "blocked_by_me" in neutrons.masks:
                # Because we use shallow copies, we do not want to do an in-place |=
                # operation here
                neutrons.masks['blocked_by_others'] = neutrons.masks[
                    'blocked_by_others'
                ] | neutrons.masks.pop('blocked_by_me')

            neutrons, reading = comp.apply(neutrons=neutrons, time_limit=time_limit)
            readings[comp.name] = reading

        return Result(source=self.source.as_readonly(), readings=readings)

    def __repr__(self) -> str:
        out = f"Model:\n  Source: {self.source}\n"
        groups = {}
        for comp in self.components.values():
            if comp.kind not in groups:
                groups[comp.kind] = []
            groups[comp.kind].append(comp)

        for group, comps in groups.items():
            out += f"  {group.capitalize()}s:\n"
            for comp in sorted(comps, key=lambda c: c.distance):
                out += f"    {comp.name}: {comp}\n"
        return out

    def __str__(self) -> str:
        return self.__repr__()
