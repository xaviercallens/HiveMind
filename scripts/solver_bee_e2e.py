#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Solver Bee end-to-end check: REAL CLI binary (sandboxed) + REAL verifier on every library problem (stdlib only).

    solver_bee_e2e.py [PROBLEMS_DIR] [BINARY] [--out FILE] [--timeout S]

Defaults: PROBLEMS_DIR = solvers/problems, BINARY = the release build of hivemind-solver on the build disk,
FILE = artifacts/experiments/SOLVER_E2E_report.json.

For each hivemind-solver-problem/1 file: the spec is run through ``sandbox.run_binary``, then ``verifier.verify`` is called with the library problem and a
``rerun`` callable that runs the tightened spec through the SAME sandbox. The table and the JSON report show the verdict exactly as measured; nothing here
adjusts a tolerance, factor or threshold. A FAILED verdict is a result, not a harness error. The physical-model validity of a spec is never checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from hivemind.solverbee import sandbox, verifier  # noqa: E402

DEFAULT_BINARY = "/mnt/disks/disk-socrateai-local-1/solver-build/target-a/release/hivemind-solver"
DEFAULT_OUT = REPO / "artifacts" / "experiments" / "SOLVER_E2E_report.json"
SCHEMA = "hivemind-solver-e2e/1"


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def check_of(cert: dict, name: str) -> dict:
    for c in cert.get("checks") or []:
        if c.get("name") == name:
            return c
    return {"name": name, "status": "absent", "value": None, "bound": None}


def run_one(path: Path, binary: str, limits: sandbox.Limits, bin_sha: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        problem = json.load(fh)
    spec_bytes = json.dumps(problem["spec"], separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    t0 = time.monotonic()
    notes = []

    def solve(raw: bytes) -> dict:
        out = sandbox.run_binary([binary, "run"], raw, limits=limits, expected_sha256=bin_sha)
        if not out.ok or out.limit or out.rc != 0:
            raise RuntimeError("sandbox/solver failure: ok=%s limit=%s rc=%s err=%s" % (out.ok, out.limit, out.rc, (out.error or out.stderr_tail)[:200]))
        return json.loads(out.stdout.decode("utf-8"))

    try:
        result = solve(spec_bytes)
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        result = None
        notes.append(f"{type(exc).__name__}: {exc}")
    reruns = {"n": 0}

    def rerun(raw: bytes) -> dict:
        reruns["n"] += 1
        return solve(raw)

    cert = verifier.verify(spec_bytes, result, problem=problem, rerun=rerun, binary_sha256=bin_sha)
    wall_ms = int((time.monotonic() - t0) * 1000)
    ref, inv, tight = check_of(cert, "reference"), check_of(cert, "invariants"), check_of(cert, "tightening")
    return {
        "problem": problem["id"],
        "verdict": cert["verdict"],
        "reference": {"status": ref["status"], "worst_ratio": ref["value"], "bound": ref["bound"]},
        "invariants": {"status": inv["status"], "worst_ratio": inv["value"], "bound": inv["bound"]},
        "tightening": {"status": tight["status"], "worst_ratio": tight["value"], "bound": tight["bound"]},
        "policy_factor": (problem.get("tolerance_policy") or {}).get("max_rel_err_factor"),
        "rtol": problem["spec"].get("rtol"),
        "atol": problem["spec"].get("atol"),
        "wall_ms": wall_ms,
        "reasons": cert["reasons"],
        "notes": notes,
        "certificate": cert,
    }


def fmt(x) -> str:
    return "-" if x is None else ("%.4g" % x)


def table(rows: list) -> str:
    head = ("problem", "verdict", "ref worst/bound", "invariants", "tightening", "wall ms")
    lines = [head]
    for r in rows:
        ref = r["reference"]
        lines.append((r["problem"], r["verdict"], f"{ref['status']} {fmt(ref['worst_ratio'])}/{fmt(ref['bound'])}",
                      f"{r['invariants']['status']} {fmt(r['invariants']['worst_ratio'])}", f"{r['tightening']['status']} {fmt(r['tightening']['worst_ratio'])}",
                      str(r["wall_ms"])))
    w = [max(len(row[i]) for row in lines) for i in range(len(head))]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(row)) for row in lines)


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problems", nargs="?", default=str(REPO / "solvers" / "problems"))
    ap.add_argument("binary", nargs="?", default=DEFAULT_BINARY)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--timeout", type=float, default=120.0, help="wall-clock seconds per solver run")
    args = ap.parse_args(argv)

    if not Path(args.binary).is_file():
        print("solver_bee_e2e: binary not found", file=sys.stderr)
        return 3
    if not sandbox.namespace_available():
        print("solver_bee_e2e: user namespaces unavailable; refusing to run unsandboxed (%s)" % sandbox.availability_reason(), file=sys.stderr)
        return 3
    files = sorted(Path(args.problems).glob("*.json"))
    if not files:
        print("solver_bee_e2e: no problem files", file=sys.stderr)
        return 3
    bin_sha = sandbox.file_sha256(os.path.realpath(args.binary))
    limits = sandbox.Limits(timeout_s=args.timeout)
    load_before = os.getloadavg()
    rows = [run_one(p, args.binary, limits, bin_sha) for p in files]
    load_after = os.getloadavg()
    print(table(rows))
    counts: dict = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("\nverdicts:", json.dumps(counts, sort_keys=True))

    body = {
        "schema": SCHEMA,
        "tolerance_policy_note": "Verdicts are measured under the problem library's tolerance_policy as shipped (policy v1); no threshold was adjusted.",
        "caveat": verifier.CAVEAT,
        "binary_sha256": bin_sha,
        "verifier_version": verifier.VERIFIER_VERSION,
        "sandbox": "namespace",
        "machine": {"cpu_model": cpu_model(), "logical_cpus": os.cpu_count(), "loadavg_before": list(load_before), "loadavg_after": list(load_after)},
        "verdict_counts": counts,
        "rows": rows,
    }
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    body["receipt_sha256"] = hashlib.sha256(canon).hexdigest()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print("report:", out, "receipt_sha256:", body["receipt_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
