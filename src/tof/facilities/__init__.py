# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pooch

from . import ess

source_library = {}
source_library.update(ess.sources)

_BASE_URLS = [
    "https://public.esss.dk/groups/scipp/tof/2/",  # primary: DMSC server
    "https://github.com/scipp/tof-sources/raw/refs/heads/main/2/",  # fallback: GitHub
]

# One registry per mirror URL
_registries = [
    pooch.create(
        path=pooch.os_cache("tof"),
        base_url=base_url,
        retry_if_failed=2,
        registry={f["path"]: f["hash"] for f in source_library.values()},
    )
    for base_url in _BASE_URLS
]


def get_source_path(name: str) -> str:
    path = source_library[name]["path"]
    last_exc = None
    for registry in _registries:
        try:
            return registry.fetch(path)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(
        f"Failed to download '{path}' from all mirrors: {_BASE_URLS}"
    ) from last_exc


__all__ = ["ess", "get_source_path", "source_library"]
