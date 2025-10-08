# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, F401, I

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"


import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=['facilities'],
    submod_attrs={
        'chopper': ['AntiClockwise', 'Chopper', 'ChopperReading', 'Clockwise'],
        'dashboard': ['Dashboard'],
        'detector': ['Detector', 'DetectorReading'],
        'model': ['Model'],
        'reading': ['ComponentReading', 'ReadingField'],
        'result': ['Result'],
        'source': ['Source', 'SourceParameters'],
    },
)

del importlib
del lazy
