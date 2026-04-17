# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""
Sources for the instruments at the ESS facility.

They were generated using the tools/generate_ess_sources.ipynb notebook.

The following parameters were used:

Source:
Lmin=0.1, Lmax=20, focus_xw=0.1, focus_yh=0.1, dist=2

Monitor:
xwidth=1.0, yheight=1.0, restore_neutron=1, l bins=1024, limits=[0.1, 20], t bins=1024,
limits=[0 0.006], set_AT=0.05, RELATIVE=source

Instrument settings:
ncount=1.0e10

Gaussian smoothing:
sigma=2
"""

from .dream import dream
from .magic import magic
from .odin import odin

sources = {
    "ess": {
        "path": "ess/ess.h5",
        "hash": "md5:4d929bf73ba808f8a63ebaf137e4b2fc",
        "description": "The standard ESS source profile, applicable for all ESS "
        "instruments.",
    },
    "ess-beer": {
        "path": "ess/ess-beer.h5",
        "hash": "md5:9e4350a13c6f322ebe788ce874787a3b",
        "description": "A source for the ESS Beer instrument, sampled at the entrance "
        "of the beam port, 5cm from the surface of the moderator.",
    },
    "ess-bifrost": {
        "path": "ess/ess-bifrost.h5",
        "hash": "md5:83952831dc576124b8a4de6593c09c64",
        "description": "A source for the ESS Bifrost instrument, sampled at the "
        "entrance of the beam port, 5cm from the surface of the moderator.",
    },
    "ess-cspec": {
        "path": "ess/ess-cspec.h5",
        "hash": "md5:09e7ed297aefcddc4eb5a63146629b22",
        "description": "A source for the ESS C-SPEC instrument, sampled at the "
        "entrance of the beam port, 5cm from the surface of the moderator.",
    },
    "ess-dream": {
        "path": "ess/ess-dream.h5",
        "hash": "md5:ddd18500e4bd627fb6fca7d010bec019",
        "description": "A source for the ESS Dream instrument, sampled at the "
        "entrance of the beam port, 5cm from the surface of the moderator.",
    },
    "ess-estia": {
        "path": "ess/ess-estia.h5",
        "hash": "md5:2c6a0af4559be482bb30c67131d25be4",
        "description": "A source for the ESS Estia instrument, sampled at the "
        "entrance of the beam port, 5cm from the surface of the moderator.",
    },
    "ess-freia": {
        "path": "ess/ess-freia.h5",
        "hash": "md5:58fd7a0c720f4ce17a1d5713b02d9f96",
        "description": "A source for the ESS Freia instrument, sampled at the "
        "entrance of the beam port, 5cm from the surface of the moderator.",
    },
    "ess-heimdal": {
        "path": "ess/ess-heimdal.h5",
        "hash": "md5:40925e3267acaa8b6efcd05b67881107",
        "description": "A source for the ESS Heimdal instrument, sampled at the "
        "entrance of the beam port, 5cm from the surface of the moderator.",
    },
    "ess-loki": {
        "path": "ess/ess-loki.h5",
        "hash": "md5:6727cd93ea8775cd2ac6d8ce3c3446e8",
        "description": "A source for the ESS Loki instrument, sampled at the "
        "entrance of the beam port, 5cm from the surface of the moderator.",
    },
    "ess-magic": {
        "path": "ess/ess-magic.h5",
        "hash": "md5:de2989fd6123f3d4cda6a1c2c13a0e49",
        "description": "A source for the ESS Magic instrument, sampled at the "
        "entrance of the beam port, 5cm from the surface of the moderator.",
    },
    "ess-miracles": {
        "path": "ess/ess-miracles.h5",
        "hash": "md5:5d33490ae632a1ee33d617f0daf050a7",
        "description": "A source for the ESS Miracles instrument, sampled at the "
        "entrance of the beam port, 5cm from the surface of the moderator.",
    },
    "ess-nmx": {
        "path": "ess/ess-nmx.h5",
        "hash": "md5:dca26e529f01b8e8f55b5611802d36a5",
        "description": "A source for the ESS NMX instrument, sampled at the "
        "entrance of the beam port, 5cm from the surface of the moderator.",
    },
    "ess-odin": {
        "path": "ess/ess-odin.h5",
        "hash": "md5:4d64102d2a743bd7672967ad0cb94872",
        "description": "A source for the ESS Odin instrument, sampled at the "
        "location where cold and thermal neutrons are combined (~2.35m away from the "
        "surface of the moderator)",
    },
    "ess-skadi": {
        "path": "ess/ess-skadi.h5",
        "hash": "md5:eab4c20f95ed2a63479407a54a01cca9",
        "description": "A source for the ESS Skadi instrument, sampled at the "
        "entrance of the beam port, 5cm from the surface of the moderator.",
    },
    "ess-tbl": {
        "path": "ess/ess-tbl.h5",
        "hash": "md5:8af1a26d386f690e89ef23f35daeef13",
        "description": "A source for the ESS TBL instrument, sampled at the "
        "entrance of the beam port, 5cm from the surface of the moderator.",
    },
    "ess-trex": {
        "path": "ess/ess-trex.h5",
        "hash": "md5:3f6f77ce4b301867acb334b7378b7fe1",
        "description": "A source for the ESS Trex instrument, sampled at the "
        "entrance of the beam port, 5cm from the surface of the moderator.",
    },
    "ess-vespa": {
        "path": "ess/ess-vespa.h5",
        "hash": "md5:eac9c4c4b861090426f0ed3c2e2b479d",
        "description": "A source for the ESS Vespa instrument, sampled at the "
        "entrance of the beam port, 5cm from the surface of the moderator.",
    },
}


__all__ = ["dream", "magic", "odin", "sources"]
