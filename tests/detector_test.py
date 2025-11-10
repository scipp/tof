# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc

import tof


def test_as_json():
    detector = tof.Detector(distance=sc.scalar(54.0, unit='m'), name='DetectorX')

    json_str = detector.as_json()

    assert json_str['type'] == 'detector'
    assert json_str['distance']['value'] == detector.distance.value
    assert json_str['distance']['unit'] == str(detector.distance.unit)
    assert json_str['name'] == detector.name
