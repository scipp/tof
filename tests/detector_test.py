# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import scipp as sc

import tof


def test_as_json():
    detector = tof.Detector(distance=sc.scalar(54.0, unit='m'), name='DetectorX')

    json_str = detector.as_json()

    assert json_str['type'] == 'detector'
    assert json_str['distance']['value'] == detector.distance.value
    assert json_str['distance']['unit'] == str(detector.distance.unit)
    assert json_str['name'] == detector.name


def test_equal():
    detector1 = tof.Detector(distance=sc.scalar(54.0, unit='m'), name='DetectorX')
    detector2 = tof.Detector(distance=sc.scalar(54.0, unit='m'), name='DetectorX')
    detector3 = tof.Detector(distance=sc.scalar(60.0, unit='m'), name='DetectorY')
    detector4 = tof.Detector(distance=sc.scalar(54.0, unit='m'), name='DetectorY')
    detector5 = tof.Detector(
        distance=sc.scalar(54.0, unit='m').to(unit='cm'), name='DetectorX'
    )

    assert detector1 == detector2
    assert detector1 != detector3
    assert detector4 != detector3
    assert detector1 != detector4
    assert detector1 != detector5
