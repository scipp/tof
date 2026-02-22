# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import warnings
from collections import defaultdict
from itertools import chain

import scipp as sc

# from tof.component import Component
from .chopper import AntiClockwise, Chopper, Clockwise
from .component import Component
from .detector import Detector
from .result import Result
from .source import Source

ComponentType = Chopper | Detector


# def _array_or_none(results: dict, key: str) -> sc.Variable | None:
#     return (
#         sc.array(
#             dims=["cutout"], values=results[key]["value"], unit=results[key]["unit"]
#         )
#         if key in results
#         else None
#     )


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
    # detectors = []
    mapping = {"chopper": Chopper, "detector": Detector}
    for name, comp in instrument.items():
        if comp["type"] not in mapping:
            raise ValueError(
                f"Unknown component type: {comp['type']} for component {name}. "
            )
        components.append(mapping[comp["type"]].from_json(name=name, params=comp))
        # component_class = comp_mapping[comp["type"]]
        # if comp["type"] == "chopper":
        #     direction = comp["direction"].lower()
        #     if direction == "clockwise":
        #         _dir = Clockwise
        #     elif any(x in direction for x in ("anti", "counter")):
        #         _dir = AntiClockwise
        #     else:
        #         raise ValueError(
        #             f"Chopper direction must be 'clockwise' or 'anti-clockwise', got "
        #             f"'{comp['direction']}' for component {name}."
        #         )
        #     choppers.append(
        #         Chopper(
        #             frequency=comp["frequency"]["value"]
        #             * sc.Unit(comp["frequency"]["unit"]),
        #             direction=_dir,
        #             open=_array_or_none(comp, "open"),
        #             close=_array_or_none(comp, "close"),
        #             centers=_array_or_none(comp, "centers"),
        #             widths=_array_or_none(comp, "widths"),
        #             phase=comp["phase"]["value"] * sc.Unit(comp["phase"]["unit"]),
        #             distance=comp["distance"]["value"]
        #             * sc.Unit(comp["distance"]["unit"]),
        #             name=name,
        #         )
        #     )
        # elif comp["type"] == "detector":
        #     detectors.append(
        #         Detector(
        #             distance=comp["distance"]["value"]
        #             * sc.Unit(comp["distance"]["unit"]),
        #             name=name,
        #         )
        #     )
        # elif comp["type"] == "source":
        #     continue
        # else:
        #     raise ValueError(
        #         f"Unknown component type: {comp['type']} for component {name}. "
        #         "Supported types are 'chopper', 'detector', and 'source'."
        #     )
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
        # self.choppers = {}
        # self.detectors = {}
        self.source = source
        self.components = {}
        # for components, kind in ((choppers, Chopper), (detectors, Detector)):
        for comp in chain((choppers or ()), (detectors or ()), (components or ())):
            self.add(comp)

    # @property
    # def choppers(self) -> dict[str, Chopper]:
    #     """
    #     Return a dictionary of all choppers in the model.
    #     This is meant for retro-compatibility with older versions of the code, and will
    #     be removed in a future version.
    #     """
    #     return {
    #         name: comp
    #         for name, comp in self.components.items()
    #         if comp.kind == "chopper"
    #     }

    # @property
    # def detectors(self) -> dict[str, Detector]:
    #     """
    #     Return a dictionary of all detectors in the model.
    #     This is meant for retro-compatibility with older versions of the code, and will
    #     be removed in a future version.
    #     """
    #     return {
    #         name: comp
    #         for name, comp in self.components.items()
    #         if comp.kind == "detector"
    #     }

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
        # if not isinstance(component, (Chopper | Detector)):
        #     raise TypeError(
        #         f"Cannot add component of type {type(component)} to the model. "
        #         "Only Chopper and Detector instances are allowed."
        #     )
        # Note that the name "source" is reserved for the source.
        if component.name in (*self.components, "source"):
            raise KeyError(
                f"Component with name {component.name} already exists. "
                "If you wish to replace/update an existing component, use "
                "``model.components['name'] = new_component``."
            )
        # results = self.choppers if isinstance(component, Chopper) else self.detectors
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
        # if name in self.choppers:
        #     del self.choppers[name]
        # elif name in self.detectors:
        #     del self.detectors[name]
        # else:
        #     raise KeyError(f"No component with name {name} was found.")

    # def __iter__(self):
    #     return chain(self.choppers, self.detectors)

    # def __getitem__(self, name) -> Chopper | Detector:
    #     if name not in self:
    #         raise KeyError(f"No component with name {name} was found.")
    #     return self.choppers[name] if name in self.choppers else self.detectors[name]

    # def __delitem__(self, name) -> None:
    #     self.remove(name)

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

        # birth_time = self.source.data.coords['birth_time']
        # speed = self.source.data.coords['speed']
        # neutrons = sc.DataGroup(self.source.data.coords)
        neutrons = self.source.data.copy(deep=False)
        neutrons.masks["blocked_by_others"] = sc.zeros(
            sizes=neutrons.sizes, unit=None, dtype=bool
        )
        neutrons.coords.update(
            distance=self.source.distance, toa=neutrons.coords['birth_time']
        )

        time_unit = neutrons.coords['birth_time'].unit

        readings = {}
        # result_choppers = {}
        # result_detectors = {}
        time_limit = (
            neutrons.coords['birth_time']
            + (
                (components[-1].distance - self.source.distance)
                / neutrons.coords['speed']
            ).to(unit=time_unit)
        ).max()
        for comp in components:
            # results = result_detectors if isinstance(c, Detector) else result_choppers
            # results[comp.name] = comp.as_dict()
            neutrons = neutrons.copy(deep=False)
            # tof = ((c.distance - self.source.distance) / neutrons.coords['speed']).to(
            #     unit=time_unit, copy=False
            # )
            # t = birth_time + tof
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

            # # data_at_comp.coords['tof'] = tof
            # if isinstance(c, Detector):
            #     data_at_comp.masks['blocked_by_others'] = ~initial_mask
            #     continue
            # m = sc.zeros(sizes=t.sizes, unit=None, dtype=bool)
            # to, tc = c.open_close_times(time_limit=time_limit)
            # results[c.name].update({'open_times': to, 'close_times': tc})
            # for i in range(len(to)):
            #     m |= (t > to[i]) & (t < tc[i])
            # combined = initial_mask & m
            # data_at_comp.masks['blocked_by_others'] = ~initial_mask
            # data_at_comp.masks['blocked_by_me'] = ~m & initial_mask
            # initial_mask = combined
            # neutrons = data_at_comp

        return Result(source=self.source, readings=readings)

    def __repr__(self) -> str:
        # out = f"Model:\n  Source: {self.source}\n  Choppers:\n"
        # for name, ch in self.choppers.items():
        #     out += f"    {name}: {ch}\n"
        # out += "  Detectors:\n"
        # for name, det in self.detectors.items():
        #     out += f"    {name}: {det}\n"
        # return out
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
