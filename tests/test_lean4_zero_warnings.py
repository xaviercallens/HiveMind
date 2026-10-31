# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests verifying Lean 4 specification hygiene (zero sorry axioms, zero unused hypothesis warnings)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LEAN_SPEC = REPO_ROOT / "lean4" / "HiveMindSpec.lean"


def test_lean4_spec_has_zero_sorry_and_unused_hypotheses_prefixed() -> None:
    """Verify lean4/HiveMindSpec.lean contains zero sorry axioms and _h_K is underscored."""
    text = LEAN_SPEC.read_text(encoding="utf-8")
    no_block_comments = re.sub(r"/-.*?-/", "", text, flags=re.DOTALL)
    code_lines = [
        line.split("--", 1)[0]
        for line in no_block_comments.splitlines()
        if not line.lstrip().startswith("--")
    ]
    joined = "\n".join(code_lines)
    assert re.search(r"\bsorry\b", joined) is None
    assert "(_h_K : 1 <= K)" in text


def test_lean4_compiler_zero_warnings_when_lake_installed() -> None:
    """When lake is installed, verify `lake env lean HiveMindSpec.lean` exits 0 with zero warnings."""
    lake_bin = shutil.which("lake") or str(Path.home() / ".elan" / "bin" / "lake")
    if not Path(lake_bin).exists():
        return
    proc = subprocess.run(
        [lake_bin, "env", "lean", "HiveMindSpec.lean"],
        cwd=str(REPO_ROOT / "lean4"),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc.returncode == 0, f"lean failed: {proc.stdout}\n{proc.stderr}"
    combined = (proc.stdout + "\n" + proc.stderr).strip()
    assert "warning:" not in combined
