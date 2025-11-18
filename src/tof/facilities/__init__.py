# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pooch

from . import ess

_source_library = {}
_source_library.update(ess.sources)


_source_registry = pooch.create(
    path=pooch.os_cache("tof"),
    base_url="https://github.com/scipp/tof-sources/raw/refs/heads/main/1/",
    retry_if_failed=2,
    registry={f["path"]: f["hash"] for f in _source_library.values()},
)


def get_source_path(name: str) -> str:
    return _source_registry.fetch(_source_library[name]["path"])


__all__ = ["ess", "get_source_path"]
