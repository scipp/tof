# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""
This module exists only to provide retro-compatibility for `essreduce` and
potentially other downstream packages that used to import pulse parameters from the old
`tof.facilities.ess_pulse` module.

This should be phased out in future releases.
"""

from ..ess.pulse_profile import pulse

frequency = pulse.frequency
birth_time = pulse.birth_time
wavelength = pulse.wavelength

__all__ = ["birth_time", "frequency", "wavelength"]

del pulse
