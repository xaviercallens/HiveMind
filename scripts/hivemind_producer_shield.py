#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Re-export wrapper for ``hivemind.router.hivemind_producer_shield`` (single canonical source, Audit C4)."""

from __future__ import annotations

from pathlib import Path
import sys

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from hivemind.router.hivemind_producer_shield import (  # noqa: E402,F401
    DEFAULT_MAX_BODY_BYTES,
    DEFAULT_MAX_CONCURRENT,
    ENDPOINT_TOKEN_ENV,
    MAX_TOKENS_CAP,
    MIN_TOKEN_CHARS,
    POC_NOTICE,
    UPSTREAM_TIMEOUT_S,
    _error,
    build_parser,
    create_shield_app,
    main,
)

if __name__ == "__main__":
    sys.exit(main())
