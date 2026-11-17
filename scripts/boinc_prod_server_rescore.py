#!/usr/bin/env python3
"""Rescore criterion C3 of a real-server e2e run from its stored full client log, and write scrubbed evidence.

The first version of boinc_prod_server_e2e.py looked for the emulator's output-file names (hmp_good_001_0_<n>);
the real server names them hmp_good_001_0_r<random>_<n>. The original report is kept (`criteria_original`).

    python3 scripts/boinc_prod_server_rescore.py --report RUN_report.json --client-log CLIENT/client_stdout.log --out-dir artifacts/boinc_prod_server
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--client-log", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    rep = json.loads(a.report.read_text())
    log = a.client_log.read_text(errors="replace")
    c3 = all(re.search(rf"Finished upload of hmp_good_001_0_r\d+_{i}\b", log) for i in (0, 1))
    rep["rescore"] = {
        "note": "C3 was scored false by the original matcher (emulator file naming); the real server uses hmp_good_001_0_r<random>_<n>. "
        "Rescored on the stored full client log; no new run.",
        "C3": c3,
    }
    rep["criteria_original"] = dict(rep["criteria"])
    rep["criteria"]["C3"] = c3
    rep["PASS_original"] = rep["PASS"]
    rep["PASS"] = all(rep["criteria"].values())
    home = str(Path.home())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "e2e_report.json").write_text(json.dumps(rep, indent=2, sort_keys=True, default=str).replace(home, "~"))
    (a.out_dir / "client_stdout.log").write_text(log[-90000:].replace(home, "~"))
    print(json.dumps({"criteria": rep["criteria"], "PASS": rep["PASS"]}, indent=2))
    return 0 if rep["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
