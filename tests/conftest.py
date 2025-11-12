# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc

import tof


@pytest.fixture
def make_chopper():
    def _make_chopper(topen, tclose, f, phase, distance, name):
        two_pi = 2.0 * np.pi * sc.units.rad
        aopen = two_pi * sc.concat(topen, dim='cutout').to(unit='s') * f
        aclose = two_pi * sc.concat(tclose, dim='cutout').to(unit='s') * f
        return tof.Chopper(
            frequency=f,
            open=aopen,
            close=aclose,
            phase=phase,
            distance=distance,
            name=name,
        )

    return _make_chopper


@pytest.fixture
def make_source():
    def _make_source(arrival_times, distance, pulses=1, frequency=None):
        # Arrival times are distance * m_over_h * wavelength
        return tof.Source.from_neutrons(
            birth_times=sc.array(
                dims=['event'], values=[0.0] * len(arrival_times), unit='s'
            ),
            wavelengths=arrival_times.to(unit='s') / (distance * tof.utils.m_over_h),
            pulses=pulses,
            frequency=frequency,
        )

    return _make_source


@pytest.fixture
def dummy_chopper():
    return tof.Chopper(
        frequency=sc.scalar(1.0, unit="Hz"),
        open=sc.array(dims=['cutout'], values=[0.0], unit='deg'),
        close=sc.array(dims=['cutout'], values=[1.0], unit='deg'),
        phase=sc.scalar(0.0, unit="deg"),
        distance=sc.scalar(1.0, unit="m"),
        name='dummy_chopper',
    )


@pytest.fixture
def dummy_detector():
    return tof.Detector(
        distance=sc.scalar(1.0, unit="m"),
        name='dummy_detector',
    )


@pytest.fixture
def dummy_source():
    return tof.Source.from_neutrons(
        birth_times=sc.array(dims=['event'], values=[0.0], unit='s'),
        wavelengths=sc.array(dims=['event'], values=[1.0], unit='angstrom'),
    )
