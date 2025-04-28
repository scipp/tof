# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, F401, I

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

from .chopper import AntiClockwise, Chopper, ChopperReading, Clockwise
from .dashboard import Dashboard
from .detector import Detector, DetectorReading
from .facilities import library as facilities
from .model import Model
from .reading import ComponentReading, ReadingField
from .result import Result
from .source import Source, SourceParameters

__all__ = [
    'AntiClockwise',
    'Chopper',
    'ChopperReading',
    'Clockwise',
    'ComponentReading',
    'Dashboard',
    'Detector',
    'DetectorReading',
    'Model',
    'ReadingField',
    'Result',
    'Source',
    'SourceParameters',
    'facilities',
]
