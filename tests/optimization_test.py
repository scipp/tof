# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)


import scipp as sc

import tof


def test_optimized_source_has_no_blocked_neutrons():
    beamline = tof.facilities.ess.odin(pulse_skipping=True)
    N = 100_000
    s1 = tof.Source(facility='ess', neutrons=N)
    m1 = tof.Model(source=s1, **beamline)
    r1 = m1.run()
    sum1 = r1['detector'].data.sum().value
    assert sum1 > 0
    assert sum1 < N

    choppers = {
        comp.name: comp
        for comp in beamline['components']
        if isinstance(comp, tof.Chopper)
    }
    s2 = tof.Source(facility='ess', neutrons=N, optimize_for=choppers)
    m2 = tof.Model(source=s2, **beamline)
    r2 = m2.run()
    sum2 = r2['detector'].data.sum().value
    assert sum2 == N


def test_optimize_for_an_early_chopper_still_has_some_blocked_neutrons():
    beamline = tof.facilities.ess.odin(pulse_skipping=True)
    N = 100_000
    s1 = tof.Source(facility='ess', neutrons=N)
    m1 = tof.Model(source=s1, **beamline)
    r1 = m1.run()
    sum1 = r1['detector'].data.sum().value
    assert sum1 > 0
    assert sum1 < N

    choppers = {
        comp.name: comp
        for comp in beamline['components']
        if comp.name in ("WFMC_1", "WFMC_2")
    }
    s2 = tof.Source(facility='ess', neutrons=N, optimize_for=choppers)
    m2 = tof.Model(source=s2, **beamline)
    r2 = m2.run()
    sum2 = r2['detector'].data.sum().value
    assert sum2 > 0
    assert sum2 < N
    assert sum2 > sum1


def test_optimize_combined_with_wmin_wmax():
    beamline = tof.facilities.ess.odin(pulse_skipping=True)
    N = 100_000
    choppers = {
        comp.name: comp
        for comp in beamline['components']
        if isinstance(comp, tof.Chopper)
    }

    wmin = sc.scalar(2.0, unit='angstrom')
    wmax = sc.scalar(8.0, unit='angstrom')

    s2 = tof.Source(
        facility='ess',
        neutrons=N,
        optimize_for=choppers,
        wmin=wmin,
        wmax=wmax,
    )
    m2 = tof.Model(source=s2, **beamline)
    r2 = m2.run()

    assert sc.all(s2.data.coords['wavelength'] >= wmin)
    assert sc.all(s2.data.coords['wavelength'] <= wmax)
    assert sc.all(r2['detector'].data.coords['wavelength'] >= wmin)
    assert sc.all(r2['detector'].data.coords['wavelength'] <= wmax)
