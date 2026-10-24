#!/usr/bin/env python3
"""Runner script for Project HiveMind Season 9 & Season 10 GPU & Swarm Contributor Campaigns."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hivemind.cli.season9_10_gpu_swarm_campaigns import main

if __name__ == "__main__":
    raise SystemExit(main())
