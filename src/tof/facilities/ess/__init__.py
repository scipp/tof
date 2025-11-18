# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from .dream import dream
from .odin import odin

sources = {
    "ess": {"path": "ess/ess.h5", "hash": "md5:f382e4a5c2171b4ef2285fb6625544e4"},
    "ess-odin": {
        "path": "ess/ess-odin.h5",
        "hash": "md5:4d64102d2a743bd7672967ad0cb94872",
    },
}


__all__ = ["dream", "odin", "sources"]
