# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import scipp as sc

from ...chopper import AntiClockwise, Chopper, Clockwise
from ...detector import Detector


def make_beamline(instrument) -> dict[str, list[Chopper] | list[Detector]]:
    choppers = [
        Chopper(
            frequency=ch["frequency"] * sc.Unit("Hz"),
            direction={"clockwise": Clockwise, "anti-clockwise": AntiClockwise}[
                ch["direction"]
            ],
            open=sc.array(dims=["cutout"], values=ch["open"], unit="deg")
            if "open" in ch
            else None,
            close=sc.array(dims=["cutout"], values=ch["close"], unit="deg")
            if "close" in ch
            else None,
            centers=sc.array(dims=["cutout"], values=ch["centers"], unit="deg")
            if "centers" in ch
            else None,
            widths=sc.array(dims=["cutout"], values=ch["widths"], unit="deg")
            if "widths" in ch
            else None,
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
