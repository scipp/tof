# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import warnings
from itertools import chain

import scipp as sc

from .chopper import AntiClockwise, Chopper, Clockwise
from .detector import Detector
from .result import Result
from .source import Source

ComponentType = Chopper | Detector


def _input_to_dict(
    obj: None | list[ComponentType] | tuple[ComponentType, ...] | ComponentType,
    kind: type,
):
    if isinstance(obj, list | tuple):
        out = {}
        for item in obj:
            new = _input_to_dict(item, kind=kind)
            for key in new.keys():
                if key in out:
                    raise ValueError(f"More than one component named '{key}' found.")
            out.update(new)
        return out
    elif isinstance(obj, kind):
        return {obj.name: obj}
    elif obj is None:
        return {}
    else:
        raise TypeError(
            "Invalid input type. Must be a Chopper or a Detector, "
            "or a list/tuple of Choppers or Detectors."
        )


def _array_or_none(container: dict, key: str) -> sc.Variable | None:
    return (
        sc.array(
            dims=["cutout"], values=container[key]["value"], unit=container[key]["unit"]
        )
        if key in container
        else None
    )


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
    choppers = []
    detectors = []
    for name, comp in instrument.items():
        if comp["type"] == "chopper":
            direction = comp["direction"].lower()
            if direction == "clockwise":
                _dir = Clockwise
            elif any(x in direction for x in ("anti", "counter")):
                _dir = AntiClockwise
            else:
                raise ValueError(
                    f"Chopper direction must be 'clockwise' or 'anti-clockwise', got "
                    f"'{comp['direction']}' for component {name}."
                )
            choppers.append(
                Chopper(
                    frequency=comp["frequency"]["value"]
                    * sc.Unit(comp["frequency"]["unit"]),
                    direction=_dir,
                    open=_array_or_none(comp, "open"),
                    close=_array_or_none(comp, "close"),
                    centers=_array_or_none(comp, "centers"),
                    widths=_array_or_none(comp, "widths"),
                    phase=comp["phase"]["value"] * sc.Unit(comp["phase"]["unit"]),
                    distance=comp["distance"]["value"]
                    * sc.Unit(comp["distance"]["unit"]),
                    name=name,
                )
            )
        elif comp["type"] == "detector":
            detectors.append(
                Detector(
                    distance=comp["distance"]["value"]
                    * sc.Unit(comp["distance"]["unit"]),
                    name=name,
                )
            )
        elif comp["type"] == "source":
            continue
        else:
            raise ValueError(
                f"Unknown component type: {comp['type']} for component {name}. "
                "Supported types are 'chopper', 'detector', and 'source'."
            )
    return {"choppers": choppers, "detectors": detectors}


class Model:
    """
    A class that represents a neutron instrument.
    It is defined by a list of choppers, a list of detectors, and a source.

    Parameters
    ----------
    choppers:
        A list of choppers.
    detectors:
        A list of detectors.
    source:
        A source of neutrons.
    """

    def __init__(
        self,
        source: Source | None = None,
        choppers: Chopper | list[Chopper] | tuple[Chopper, ...] | None = None,
        detectors: Detector | list[Detector] | tuple[Detector, ...] | None = None,
    ):
        self.choppers = _input_to_dict(choppers, kind=Chopper)
        self.detectors = _input_to_dict(detectors, kind=Detector)
        self.source = source

    @classmethod
    def from_json(cls, filename: str) -> Model:
        """
        Create a model from a JSON file.

        Currently, only sources from facilities are supported when loading from JSON.

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
        for ch in self.choppers.values():
            instrument_dict[ch.name] = ch.as_json()
        for det in self.detectors.values():
            instrument_dict[det.name] = det.as_json()
        return instrument_dict

    def to_json(self, filename: str):
        """
        Save the model to a JSON file.
        If the source is not from a facility, it is not included in the output.

        Parameters
        ----------
        filename:
            The path to the JSON file.
        """
        import json

        with open(filename, 'w') as f:
            json.dump(self.as_json(), f, indent=2)

    def add(self, component):
        """
        Add a component to the instrument.
        Component names must be unique across choppers and detectors.

        Parameters
        ----------
        component:
            A chopper or detector.
        """
        if component.name in chain(self.choppers, self.detectors):
            raise KeyError(
                f"Component with name {component.name} already exists. "
                "If you wish to replace/update an existing component, use "
                "``model.choppers['name'] = new_chopper`` or "
                "``model.detectors['name'] = new_detector``."
            )
        if isinstance(component, Chopper):
            self.choppers[component.name] = component
        elif isinstance(component, Detector):
            self.detectors[component.name] = component
        else:
            raise TypeError(
                f"Cannot add component of type {type(component)} to the model."
            )

    def remove(self, name: str):
        """
        Remove a component.

        Parameters
        ----------
        name:
            The name of the component to remove.
        """
        if name in self.choppers:
            del self.choppers[name]
        elif name in self.detectors:
            del self.detectors[name]
        else:
            raise KeyError(f"No component with name {name} was found.")

    def __iter__(self):
        return chain(self.choppers, self.detectors)

    def __getitem__(self, name) -> Chopper | Detector:
        if name not in self:
            raise KeyError(f"No component with name {name} was found.")
        return self.choppers[name] if name in self.choppers else self.detectors[name]

    def __delitem__(self, name):
        self.remove(name)

    def run(self):
        """
        Run the simulation.
        """
        if self.source is None:
            raise ValueError(
                "No source has been defined for this model. Please add a source using "
                "`model.source = Source(...)` before running the simulation."
            )
        components = sorted(
            chain(self.choppers.values(), self.detectors.values()),
            key=lambda c: c.distance.value,
        )

        birth_time = self.source.data.coords['birth_time']
        speed = self.source.data.coords['speed']
        initial_mask = sc.ones(sizes=birth_time.sizes, unit=None, dtype=bool)

        result_choppers = {}
        result_detectors = {}
        time_limit = (
            birth_time + (components[-1].distance / speed).to(unit=birth_time.unit)
        ).max()
        for c in components:
            container = result_detectors if isinstance(c, Detector) else result_choppers
            container[c.name] = c.as_dict()
            container[c.name]['data'] = self.source.data.copy(deep=False)
            t = birth_time + (c.distance / speed).to(unit=birth_time.unit, copy=False)
            container[c.name]['data'].coords['toa'] = t
            container[c.name]['data'].coords['eto'] = t % (
                1 / self.source.frequency
            ).to(unit=t.unit, copy=False)
            container[c.name]['data'].coords['distance'] = c.distance
            # TODO: remove 'tof' coordinate once deprecation period is over
            container[c.name]['data'].coords['tof'] = t
            if isinstance(c, Detector):
                container[c.name]['data'].masks['blocked_by_others'] = ~initial_mask
                continue
            m = sc.zeros(sizes=t.sizes, unit=None, dtype=bool)
            to, tc = c.open_close_times(time_limit=time_limit)
            container[c.name].update({'open_times': to, 'close_times': tc})
            for i in range(len(to)):
                m |= (t > to[i]) & (t < tc[i])
            combined = initial_mask & m
            container[c.name]['data'].masks['blocked_by_others'] = ~initial_mask
            container[c.name]['data'].masks['blocked_by_me'] = ~m & initial_mask
            initial_mask = combined

        return Result(
            source=self.source, choppers=result_choppers, detectors=result_detectors
        )

    def __repr__(self) -> str:
        out = f"Model:\n  Source: {self.source}\n  Choppers:\n"
        for name, ch in self.choppers.items():
            out += f"    {name}: {ch}\n"
        out += "  Detectors:\n"
        for name, det in self.detectors.items():
            out += f"    {name}: {det}\n"
        return out

    def __str__(self) -> str:
        return self.__repr__()
