# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import warnings
from itertools import chain
from types import MappingProxyType

import scipp as sc

from .chopper import Chopper
from .component import Component
from .detector import Detector
from .result import Result
from .source import Source
from .utils import extract_component_group

ComponentType = Chopper | Detector


def make_beamline(instrument: dict) -> dict[str, Source | list[Component]]:
    """
    Create components from a dictionary.
    The dictionary is typically loaded from a pre-configured instrument library, or
    from a JSON file.

    Parameters
    ----------
    instrument:
        A dictionary defining the instrument components.
        Each key is the name of a component, and the value is a dictionary with the
        component parameters. Each component dictionary must have a "type" key.
        Other keys depend on the component type, see the documentation of the
        component classes for details.
    """
    beamline = {"components": []}
    # TODO: until we figure out how to serialize the InelasticSample, we do not support
    # creating it from a JSON blob.
    mapping = {"chopper": Chopper, "detector": Detector}
    for name, comp in instrument.items():
        if comp["type"] == "source":
            if "source" in beamline:
                raise ValueError(
                    "Only one source is allowed, but multiple were found in the "
                    "instrument parameters."
                )
            beamline["source"] = Source.from_json(params=comp)
            continue
        if comp["type"] not in mapping:
            raise ValueError(
                f"Unknown component type: {comp['type']} for component {name}. "
            )
        beamline["components"].append(
            mapping[comp["type"]].from_json(name=name, params=comp)
        )
    return beamline


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
        self._components = {}
        for comp in chain((choppers or ()), (detectors or ()), (components or ())):
            self.add(comp)

    @property
    def components(self) -> dict[str, Component]:
        """
        A dictionary of the components in the instrument.
        """
        return self._components

    @property
    def choppers(self) -> MappingProxyType[str, Chopper]:
        """
        A dictionary of the choppers in the instrument.
        """
        return extract_component_group(self._components, "chopper")

    @property
    def detectors(self) -> MappingProxyType[str, Detector]:
        """
        A dictionary of the detectors in the instrument.
        """
        return extract_component_group(self._components, "detector")

    @property
    def samples(self) -> MappingProxyType[str, Component]:
        """
        A dictionary of the samples in the instrument.
        """
        return extract_component_group(self._components, "inelastic_sample")

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
        return cls(**make_beamline(instrument))

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
        for comp in self._components.values():
            try:
                instrument_dict[comp.name] = comp.as_json()
            except NotImplementedError:
                warnings.warn(
                    f"Component '{comp.name}' does not support JSON serialization and "
                    "will be skipped.",
                    UserWarning,
                    stacklevel=2,
                )
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
        Component names must be unique for a given Model.
        The name "source" is reserved for the source, and can thus not be used for other
        components.

        Parameters
        ----------
        component:
            A component to add to the model (chopper, detector, sample, etc.).
        """
        if not isinstance(component, Component):
            raise TypeError(
                "Component must be an instance of Component or derived class, "
                f"but got {type(component)}."
            )
        # Note that the name "source" is reserved for the source.
        if component.name in (*self._components, "source"):
            raise KeyError(
                f"Component with name {component.name} already exists. "
                "If you wish to replace/update an existing component, use "
                "``model.components['name'] = new_component``."
            )
        self._components[component.name] = component

    def remove(self, name: str):
        """
        Remove a component.

        Parameters
        ----------
        name:
            The name of the component to remove.
        """
        del self._components[name]

    def run(self) -> Result:
        """
        Run the simulation.
        """
        if self.source is None:
            raise ValueError(
                "No source has been defined for this model. Please add a source using "
                "`model.source = Source(...)` before running the simulation."
            )
        components = sorted(self._components.values(), key=lambda c: c.distance.value)

        if len(components) == 0:
            raise ValueError("Cannot run model: no components have been defined.")

        if components[0].distance < self.source.distance:
            raise ValueError(
                "At least one component is located before the source "
                "itself. Please check the distances of the components."
            )

        neutrons = self.source.sample()
        source_reading = self.source.as_readonly(neutrons)
        # neutrons = neutrons.assign_masks(
        #     blocked_by_others=sc.zeros(sizes=neutrons.sizes, unit=None, dtype=bool)
        # )

        # .assign_coords(
        #     distance=self.source.distance, toa=self.source.data.coords['birth_time']
        # )

        time_unit = neutrons['pulse-0'].coords['birth_time'].unit

        readings = {}
        for comp in components:
            neutrons = neutrons.copy(deep=False)

            for key, pulse_data in neutrons.items():
                toa = pulse_data.coords['toa'] + (
                    (comp.distance - pulse_data.coords['distance'])
                    / pulse_data.coords['speed']
                ).to(unit=time_unit, copy=False)
                eto = toa % (1 / self.source.frequency).to(unit=time_unit, copy=False)

                new = pulse_data.assign_coords(toa=toa, eto=eto, distance=comp.distance)

                if "blocked_by_me" in new.masks:
                    # Because we use shallow copies, we do not want to do an in-place |=
                    # operation here
                    new.masks['blocked_by_others'] = new.masks[
                        'blocked_by_others'
                    ] | new.masks.pop('blocked_by_me')

                new, reading = comp.apply(neutrons=new)
                neutrons[key] = new
                readings[comp.name] = reading

        return Result(source=source_reading, readings=readings)

    def __repr__(self) -> str:
        out = f"Model:\n  Source: {self.source}\n"
        groups = {}
        for comp in self._components.values():
            if comp.kind not in groups:
                groups[comp.kind] = []
            groups[comp.kind].append(comp)

        for group, comps in groups.items():
            out += f"  {group.capitalize().replace('_', ' ')}s:\n"
            for comp in sorted(comps, key=lambda c: c.distance):
                out += f"    {comp.name}: {comp}\n"
        return out

    def __str__(self) -> str:
        return self.__repr__()
