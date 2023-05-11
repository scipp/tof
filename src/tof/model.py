# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from itertools import chain
from typing import Dict, List, Optional, Tuple, Union

import scipp as sc

from .chopper import Chopper
from .detector import Detector
from .pulse import Pulse
from .result import Result

ComponentType = Union[Chopper, Detector]


def _input_to_dict(
    obj: Union[
        None,
        Dict[str, ComponentType],
        List[ComponentType],
        Tuple[ComponentType, ...],
        ComponentType,
    ],
    kind: type,
):
    if isinstance(obj, (list, tuple)):
        out = {}
        for item in obj:
            out.update(_input_to_dict(item, kind=kind))
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


class Model:
    """
    A class that represents a neutron instrument.
    It is defined by a list of choppers, a list of detectors, and a pulse.

    Parameters
    ----------
    choppers:
        A list of choppers.
    detectors:
        A list of detectors.
    pulse:
        A pulse of neutrons.
    """

    def __init__(
        self,
        pulse: Pulse,
        choppers: Optional[Union[Chopper, List[Chopper], Tuple[Chopper, ...]]] = None,
        detectors: Optional[
            Union[Detector, List[Detector], Tuple[Detector, ...]]
        ] = None,
    ):
        self.choppers = _input_to_dict(choppers, kind=Chopper)
        self.detectors = _input_to_dict(detectors, kind=Detector)
        self.pulse = pulse

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

    def __getitem__(self, name) -> Union[Chopper, Detector]:
        if name not in self:
            raise KeyError(f"No component with name {name} was found.")
        return self.choppers[name] if name in self.choppers else self.detectors[name]

    def __delitem__(self, name):
        self.remove(name)

    def run(self, npulses: int = 1):
        """
        Run the simulation.

        Parameters
        ----------
        npulses:
            Number of pulses to simulate.
        """
        # TODO: ray-trace multiple pulses
        components = sorted(
            chain(self.choppers.values(), self.detectors.values()),
            key=lambda c: c.distance.value,
        )

        initial_mask = sc.ones(
            sizes=self.pulse.birth_times.sizes, unit=None, dtype=bool
        )

        result_choppers = {}
        result_detectors = {}
        time_limit = (
            self.pulse.birth_times + components[-1].distance / self.pulse.speeds
        ).max()
        for c in components:
            container = result_detectors if isinstance(c, Detector) else result_choppers
            container[c.name] = c.as_dict()
            container[c.name].update(
                {
                    'birth_times': self.pulse.birth_times,
                    'speeds': self.pulse.speeds,
                    'wavelengths': self.pulse.wavelengths,
                }
            )
            t = self.pulse.birth_times + c.distance / self.pulse.speeds
            container[c.name]['arrival_times'] = t.to(unit='us')
            if isinstance(c, Detector):
                container[c.name]['visible_mask'] = initial_mask
                continue
            m = sc.zeros(sizes=t.sizes, unit=None, dtype=bool)
            to, tc = c.open_close_times(time_limit=time_limit)
            container[c.name].update({'open_times': to, 'close_times': tc})
            for i in range(len(to)):
                m |= (t > to[i]) & (t < tc[i])
            combined = initial_mask & m
            container[c.name].update(
                {'visible_mask': combined, 'blocked_mask': ~m & initial_mask}
            )
            initial_mask = combined

        return Result(
            pulse=self.pulse, choppers=result_choppers, detectors=result_detectors
        )

    def __repr__(self) -> str:
        out = f"Model:\n  Pulse: {self.pulse}\n  Choppers:\n"
        for name, ch in self.choppers.items():
            out += f"    {name}: {ch}\n"
        out += "  Detectors:\n"
        for name, det in self.detectors.items():
            out += f"    {name}: {det}\n"
        return out

    def __str__(self) -> str:
        return self.__repr__()
