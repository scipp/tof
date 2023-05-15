# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import plopp as pp
import scipp as sc

from .pulse import Pulse
from .utils import Plot


class Source:
    """ """

    def __init__(
        self,
        facility: str,
        frequency: sc.Variable,
        tmin: Optional[sc.Variable] = None,
        tmax: Optional[sc.Variable] = None,
        wmin: Optional[sc.Variable] = None,
        wmax: Optional[sc.Variable] = None,
        neutrons: int = 1_000_000,
    ):
        """ """
        self.facility = facility
        self.frequency = frequency
        self.tmin = tmin
        self.tmax = tmax
        self.wmin = wmin
        self.wmax = wmax
        self.neutrons = neutrons

    def make_pulse(self):
        return Pulse(
            facility=self.facility,
            tmin=self.tmin,
            tmax=self.tmax,
            wmin=self.wmin,
            wmax=self.wmax,
            neutrons=self.neutrons,
        )
