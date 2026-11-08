# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Files that arrive over the network (server-delivered adapters, volunteer-uploaded results) must be loaded with
``torch.load(..., weights_only=True)``: the default unpickler can run code from a crafted file (readiness review, 2026-11-02)."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NETWORK_FACING = [
    *sorted((ROOT / "src/hivemind/boinc/real_client").glob("*.py")),
    ROOT / "src/hivemind/swarm/hf_community_hub.py",
    ROOT / "scripts/boinc_core_node_e2e.py",
]


def _unsafe_loads(path: Path) -> list[int]:
    bad = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load":
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "torch":
                flag = next((k for k in node.keywords if k.arg == "weights_only"), None)
                if flag is None or not (isinstance(flag.value, ast.Constant) and flag.value.value is True):
                    bad.append(node.lineno)
    return bad


def test_network_facing_code_loads_tensors_with_weights_only() -> None:
    offenders = {str(p.relative_to(ROOT)): lines for p in NETWORK_FACING if (lines := _unsafe_loads(p))}
    assert not offenders, f"torch.load without weights_only=True on network-facing code: {offenders}"
    assert sum(1 for p in NETWORK_FACING if p.exists()) >= 7  # the guard really looked at the real_client modules
