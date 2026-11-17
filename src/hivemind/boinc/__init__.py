# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Modified BOINC Orchestration, Signed Parquet Shards, Spot-Check Validation & Local GPU Astrophysics Campaign."""

from importlib import import_module
from typing import Any

# Resolved on first access: the astrophysics campaign pulls in Pillow and the whole
# coordinator/worker stack, which every `hivemind.boinc.<submodule>` import would
# otherwise pay for (and fail on when Pillow is absent).
_EXPORTS = {
    "create_signed_astrophysics_parquet_shard": "local_astrophysics_boinc",
    "render_astrophysics_24h_dashboard_png": "astrophysics_24h_campaign",
    "render_cosmology_deep_field_discovery_png": "astrophysics_24h_campaign",
    "run_24h_astrophysics_gpu_campaign": "astrophysics_24h_campaign",
    "run_local_boinc_astrophysics_campaign": "local_astrophysics_boinc",
}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
