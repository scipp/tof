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
    obj: Optional[
        Union[
            Dict[str, ComponentType],
            List[ComponentType],
            Tuple[ComponentType, ...],
            ComponentType,
        ]
    ]
):
    if obj is None:
        return {}
    elif isinstance(obj, dict):
        return obj
    elif isinstance(obj, (list, tuple)):
        return {item.name: item for item in obj}
    else:
        return {obj.name: obj}


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
        choppers: Optional[
            Union[Chopper, List[Chopper], Tuple[Chopper, ...], Dict[str, Chopper]]
        ] = None,
        detectors: Optional[
            Union[Detector, List[Detector], Tuple[Detector, ...], Dict[str, Detector]]
        ] = None,
    ):
        self.choppers = _input_to_dict(choppers)
        self.detectors = _input_to_dict(detectors)
        self.pulse = pulse

    def add(self, component):
        """
        Add a component to the instrument.

        Parameters
        ----------
        component:
            A chopper or detector.
        """
        if isinstance(component, Chopper):
            if component.name in self.choppers:
                raise ValueError(
                    f"Chopper with name {component.name} already exists. "
                    "If you wish to replace/update an existing chopper, use "
                    "``model.choppers['name'] = new_chopper``."
                )
            self.choppers[component.name] = component
        elif isinstance(component, Detector):
            if component.name in self.detectors:
                raise ValueError(
                    f"Detector with name {component.name} already exists. "
                    "If you wish to replace/update an existing detector, use "
                    "``model.detectors['name'] = new_detector``."
                )
            self.detectors[component.name] = component
        else:
            raise ValueError(
                f"Cannot add component of type {type(component)} to the model."
            )

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
        for c in components:
            comp = c.as_readonly()
            comp._wavelengths = self.pulse.wavelengths
            t = self.pulse.birth_times + comp.distance / self.pulse.speeds
            comp._arrival_times = t
            if isinstance(c, Detector):
                comp._mask = initial_mask
                result_detectors[comp.name] = comp
                continue
            m = sc.zeros(sizes=t.sizes, unit=None, dtype=bool)
            to = c.open_times
            tc = c.close_times
            for i in range(len(to)):
                m |= (t > to[i]) & (t < tc[i])
            combined = initial_mask & m
            comp._mask = combined
            comp._own_mask = ~m & initial_mask
            initial_mask = combined
            result_choppers[comp.name] = comp

        return Result(
            pulse=self.pulse.as_readonly(),
            choppers=result_choppers,
            detectors=result_detectors,
        )

    def __repr__(self) -> str:
        return (
            f"Model(choppers={self.choppers},\n      "
            f"detectors={self.detectors},\n      "
            f"pulse={self.pulse},\n      "
            f"neutrons={len(self.pulse.birth_times)})"
        )
