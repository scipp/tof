# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from . import ess

source_library = {
    "ess": "ess/ess.h5",
    "ess-odin": "ess/ess-odin.h5",
}

__all__ = ["ess", "source_library"]
