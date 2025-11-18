# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from .dream import dream
from .odin import odin

sources = {
    "ess": "ess/ess.h5",
    "ess-odin": "ess/ess-odin.h5",
}

__all__ = ["dream", "odin", "sources"]
