# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from pathlib import Path

import pooch
import pytest

import tof
from tof.facilities import _BASE_URLS, _source_library


def test_source_library_files_identical_on_public_and_github(tmp_path: Path) -> None:
    registry = {f["path"]: f["hash"] for f in _source_library.values()}

    public = pooch.create(
        path=tmp_path / "cache_public", base_url=_BASE_URLS[0], registry=registry
    )

    gh = pooch.create(
        path=tmp_path / "cache_github", base_url=_BASE_URLS[1], registry=registry
    )

    for entry in _source_library.values():
        rel = entry["path"]

        # Verify that hashes are the same in both registries.
        assert public.registry[rel] == gh.registry[rel]

        p_gh = Path(gh.fetch(rel))
        p_public = Path(public.fetch(rel))

        assert p_gh.name == p_public.name


@pytest.mark.parametrize("entry", _source_library.keys())
def test_can_create_source_for_library_entry(entry) -> None:
    source = tof.Source(facility=entry, neutrons=1000)
    assert source.neutrons == 1000
    assert "birth_time" in source.data.coords
    assert "wavelength" in source.data.coords
    assert source.frequency is not None
    assert source.distance is not None
