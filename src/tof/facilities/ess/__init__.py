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
    "ess": {"path": "ess/ess.h5", "hash": "md5:689630334ae9a7f462f76943740a8ff2"},
    "ess-odin": {
        "path": "ess/ess-odin.h5",
        "hash": "md5:4d64102d2a743bd7672967ad0cb94872",
    },
    "ess-nmx": {
        "path": "ess/ess-nmx.h5",
        "hash": "md5:c8a3d0c6ef17c704c94afc38b031fa60",
    },
    "ess-beer": {
        "path": "ess/ess-beer.h5",
        "hash": "md5:7d98d4446f53f4c47c3d632ae2f886b7",
    },
    "ess-cspec": {
        "path": "ess/ess-cspec.h5",
        "hash": "md5:8f8268c3b8ae6219c595b9d099ee632d",
    },
    "ess-bifrost": {
        "path": "ess/ess-bifrost.h5",
        "hash": "md5:c05b50d2a3a919e10fb0abd44144c83d",
    },
    "ess-miracles": {
        "path": "ess/ess-miracles.h5",
        "hash": "md5:3760666ce2685ff88bb79cec8d19ef8c",
    },
    "ess-magic": {
        "path": "ess/ess-magic.h5",
        "hash": "md5:774e99c8130075139b8476a87fb43182",
    },
    "ess-trex": {
        "path": "ess/ess-trex.h5",
        "hash": "md5:1eb645ec2e71324c2d03a30c4f3525fb",
    },
    "ess-heimdal": {
        "path": "ess/ess-heimdal.h5",
        "hash": "md5:e95da29df2f478eb7357583b8c388123",
    },
    "ess-dream": {
        "path": "ess/ess-dream.h5",
        "hash": "md5:0957767f7439f70aaf1abf934d30da07",
    },
    "ess-loki": {
        "path": "ess/ess-loki.h5",
        "hash": "md5:319936b29e3e8ac70283408950e6195c",
    },
    "ess-freia": {
        "path": "ess/ess-freia.h5",
        "hash": "md5:cad1a612214d0155b060094b1078f254",
    },
    "ess-estia": {
        "path": "ess/ess-estia.h5",
        "hash": "md5:a0bdf9e17611a77054686c5041aaae3f",
    },
    "ess-skadi": {
        "path": "ess/ess-skadi.h5",
        "hash": "md5:521dc300296db137fa3dd5877882f68c",
    },
    "ess-vespa": {
        "path": "ess/ess-vespa.h5",
        "hash": "md5:298f599f7dc861d1319748c7ea3c54fc",
    },
    "ess-tbl": {
        "path": "ess/ess-tbl.h5",
        "hash": "md5:c9db7f205131c4caf491d5951253bea7",
    },
}


__all__ = ["dream", "magic", "odin", "sources"]
