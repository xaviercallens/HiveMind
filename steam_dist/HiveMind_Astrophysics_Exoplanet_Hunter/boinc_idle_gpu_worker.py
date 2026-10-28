#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Standalone Steam Free-to-Play Idle-GPU BOINC Citizen-Science Worker Daemon."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from hivemind_boinc_idle_worker import (  # noqa: E402
    DEFAULT_COORDINATOR_URL,
    get_user_idle_seconds,
    main,
    query_nvidia_gpu_status,
    run_one_idle_gpu_boinc_cycle,
    set_boinc_client_gpu_mode,
)

__all__ = [
    "DEFAULT_COORDINATOR_URL",
    "get_user_idle_seconds",
    "main",
    "query_nvidia_gpu_status",
    "run_one_idle_gpu_boinc_cycle",
    "set_boinc_client_gpu_mode",
]

if __name__ == "__main__":
    raise SystemExit(main(["--once", *sys.argv[1:]]))
