#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Re-export wrapper for ``hivemind.router.hivemind_producer_agent`` (single canonical source, Audit C4)."""

from __future__ import annotations

from pathlib import Path
import sys

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from hivemind.router.hivemind_producer_agent import (  # noqa: E402,F401
    API_KEY_ENV,
    CATALOG_PATH,
    DEFAULT_COLD_START_S,
    DEFAULT_EXECUTION_CLASS,
    DEFAULT_INTERVAL_S,
    DEFAULT_LOCAL_ENDPOINT,
    ENDPOINT_TOKEN_ENV,
    HEARTBEAT_PATH,
    HTTP_TIMEOUT_S,
    MAX_CONTROL_VRAM_MIB,
    POC_NOTICE,
    REMOTE_CONFIG_PATH,
    RESIDENCY_TIMEOUT_S,
    Runner,
    _suggest,
    apply_control,
    apply_residency,
    build_heartbeat_payload,
    build_parser,
    detect_vram_gb,
    fetch_catalog,
    fetch_remote_config,
    filter_models_by_vram_budget,
    main,
    parse_control,
    parse_model_spec,
    preflight,
    probe_model_sizes_mib,
    probe_residency,
    read_control_state,
    recommend,
    send_heartbeat,
    write_control_state,
)

if __name__ == "__main__":
    sys.exit(main())
