# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import hashlib
from pathlib import Path

import pooch

from tof.facilities import _BASE_URLS, _source_library


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_source_library_files_identical_on_public_and_github(tmp_path: Path) -> None:
    registry = {f["path"]: f["hash"] for f in _source_library.values()}

    public = pooch.create(
        path=tmp_path / "cache_public", base_url=_BASE_URLS[0], registry=registry
    )

    gh = pooch.create(
        path=tmp_path / "cache_github", base_url=_BASE_URLS[1], registry=registry
    )

    for name, entry in _source_library.items():
        rel = entry["path"]

        p_gh = Path(gh.fetch(rel))
        p_public = Path(public.fetch(rel))

        # registry already verifies md5, but we compare content across hosts:
        assert sha256(p_gh) == sha256(p_public), f"Mismatch for {name} ({rel})"
