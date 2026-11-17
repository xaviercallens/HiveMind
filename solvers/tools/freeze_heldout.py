# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Write or verify solvers/problems/heldout/FROZEN.json, the sha256 manifest of the frozen held-out set (stdlib only).

    python3 solvers/tools/freeze_heldout.py            # write FROZEN.json (done once, BEFORE any solver run)
    python3 solvers/tools/freeze_heldout.py --verify   # exit 1 if any file differs, is missing or is extra

The held-out problems must never be edited after the freeze: a change is a different problem set.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HELDOUT = Path(__file__).resolve().parent.parent / "problems" / "heldout"
MANIFEST = "FROZEN.json"
SCHEMA = "hivemind-solver-heldout-frozen/1"


def digests(directory: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.glob("*.json")) if p.name != MANIFEST}


def write(directory: Path = HELDOUT) -> dict:
    doc = {
        "schema": SCHEMA,
        "note": "Frozen before the solver was ever run on these problems. Do not edit any listed file; "
                "a change makes this a different problem set.",
        "files": digests(directory),
    }
    (directory / MANIFEST).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


def verify(directory: Path = HELDOUT) -> list:
    doc = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        return [f"manifest schema must be {SCHEMA!r}"]
    want, have = doc.get("files", {}), digests(directory)
    errs = [f"{n}: sha256 differs from the frozen value" for n in sorted(want) if n in have and have[n] != want[n]]
    errs += [f"{n}: listed in {MANIFEST} but missing" for n in sorted(set(want) - set(have))]
    errs += [f"{n}: present but not listed in {MANIFEST}" for n in sorted(set(have) - set(want))]
    return errs


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--verify" in argv:
        errs = verify()
        for e in errs:
            print(e)
        print(f"{len(errs)} problem(s)")
        return 1 if errs else 0
    doc = write()
    print(f"wrote {MANIFEST} with {len(doc['files'])} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
