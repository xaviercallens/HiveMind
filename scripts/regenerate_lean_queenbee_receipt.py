#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regenerate `artifacts/lean_queenbee_check.json` with a section-scoped digest for `HiveMindSpec.ALIX_QueenBee` (IP-40)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "lean4" / "HiveMindSpec.lean"
RECEIPT_PATH = REPO_ROOT / "artifacts" / "lean_queenbee_check.json"


def compute_queenbee_section_sha256(source: str) -> str:
    """Compute a deterministic SHA-256 over the `ALIX_QueenBee` namespace and its `#print axioms` lines (IP-40)."""
    start_marker = "namespace ALIX_QueenBee"
    end_marker = "end ALIX_QueenBee"
    if start_marker not in source or end_marker not in source:
        raise ValueError("Missing ALIX_QueenBee namespace in Lean source")
    start_idx = source.index(start_marker)
    end_idx = source.index(end_marker, start_idx) + len(end_marker)
    section = source[start_idx:end_idx].strip()
    print_lines = sorted(
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith("#print axioms HiveMindSpec.ALIX_QueenBee.")
    )
    canonical = section + "\n" + "\n".join(print_lines) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def regenerate_lean_queenbee_receipt(
    spec_path: Path = SPEC_PATH,
    receipt_path: Path = RECEIPT_PATH,
    *,
    write_output: bool = True,
) -> dict[str, Any]:
    """Update `artifacts/lean_queenbee_check.json` from `lean4/HiveMindSpec.lean`."""
    source = spec_path.read_text(encoding="utf-8")
    existing: dict[str, Any] = {}
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))

    section_sha = compute_queenbee_section_sha256(source)
    theorem_count = len(re.findall(r"^theorem\s+", source, flags=re.M))
    full_sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()

    updated = dict(existing)
    updated["queenbee_namespace"] = "HiveMindSpec.ALIX_QueenBee"
    updated["queenbee_section_sha256"] = section_sha
    updated["spec_file"] = "lean4/HiveMindSpec.lean"
    updated["spec_sha256"] = full_sha
    updated["theorems_in_file"] = theorem_count

    if write_output:
        receipt_path.write_text(json.dumps(updated, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return updated


def main() -> int:
    updated = regenerate_lean_queenbee_receipt()
    print(
        json.dumps(
            {
                "queenbee_section_sha256": updated["queenbee_section_sha256"],
                "theorems_in_file": updated["theorems_in_file"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
