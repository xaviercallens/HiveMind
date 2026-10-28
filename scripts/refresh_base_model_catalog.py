#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Refresh ``configs/base_models_s21.json`` from the Hugging Face Hub (metadata, config.json and safetensors headers only).

Every number in the catalog comes from this fetch: the Hub API (``/api/models/<id>?blobs=true``) for the revision sha, license,
parameter count and file sizes; ``config.json`` at that revision for the geometry; and the safetensors headers (HTTP range
requests, no weights) for the per-component parameter split. Ids the Hub does not serve are listed under ``not_found``.
The catalog is validated (header totals must equal the Hub's own count) before it is written.

  python3 scripts/refresh_base_model_catalog.py                        # writes configs/base_models_s21.json
  python3 scripts/refresh_base_model_catalog.py --out /tmp/c.json --plan
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hivemind.core.base_model_catalog import (
    build_catalog,
    recommend_upgrade_plan,
    render_plan,
    urllib_get,
    validate_catalog,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "configs" / "base_models_s21.json"))
    ap.add_argument("--plan", action="store_true", help="also print the per-tier upgrade plan derived from the new catalog")
    a = ap.parse_args(argv)
    catalog = build_catalog(urllib_get)
    problems = validate_catalog(catalog)
    if problems:
        print("catalog rejected:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    Path(a.out).write_text(json.dumps(catalog, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {a.out}: {len(catalog['models'])} models, not found: {[m['model_id'] for m in catalog['not_found']]}")
    if a.plan:
        print(render_plan(recommend_upgrade_plan(catalog)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
