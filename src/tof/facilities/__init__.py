# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from .ess_pulse import pulse as esspulse

library = {'ess': esspulse}

__all__ = ['library']
