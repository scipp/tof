# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass

import scipp as sc


@dataclass
class FacilityPulse:
    time: sc.DataArray
    wavelength: sc.DataArray
