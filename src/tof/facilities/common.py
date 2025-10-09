# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import scipp as sc

from ..chopper import AntiClockwise, Chopper, Clockwise
from ..detector import Detector


def _array_or_none(container, key):
    return (
        sc.array(dims=["cutout"], values=container[key], unit="deg")
        if key in container
        else None
    )


def make_beamline(instrument) -> dict[str, list[Chopper] | list[Detector]]:
    choppers = [
        Chopper(
            frequency=ch["frequency"] * sc.Unit("Hz"),
            direction={"clockwise": Clockwise, "anti-clockwise": AntiClockwise}[
                ch["direction"]
            ],
            open=_array_or_none(ch, "open"),
            close=_array_or_none(ch, "close"),
            centers=_array_or_none(ch, "centers"),
            widths=_array_or_none(ch, "widths"),
            phase=ch["phase"] * sc.Unit("deg"),
            distance=ch["distance"] * sc.Unit("m"),
            name=key,
        )
        for key, ch in instrument["choppers"].items()
    ]
    detectors = [
        Detector(distance=det["distance"] * sc.Unit("m"), name=key)
        for key, det in instrument["detectors"].items()
    ]
    return {"choppers": choppers, "detectors": detectors}
