#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""MVP scorecard: the measure of the MVP loop (docs/MVP_TARGET.md). Deterministic, no LLM, no network.

Each criterion is PASS only if its evidence says so:
  * artifact criteria read ``artifacts/mvp/<ID>_*.json`` (written by the script that ran the real check) and require ``"passed": true``;
    B3 reads the train-to-serve gate artifacts (``"servable": true``);
  * D1 additionally checks the Cloud Run deploy flags against AGENTS.md; F1 checks that a guard test exists and that no page module
    carries the known fabricated strings.
Otherwise the row is FAIL (evidence says it failed) or NOT-RUN (no evidence yet), with the reason. Writes ``docs/MVP_SCORECARD.md``.

    python scripts/mvp_scorecard.py [--write] [--json]
Exit code 0 only if every row passes.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CRITERIA: dict[str, str] = {
    "A1": "Request-level MoE inference over >= 2 consumer GPUs, with failover",
    "A2": "One MoE split across 2 hosts, token-identical to single host",
    "B1": "Custom dataset -> signed shards + held-out split",
    "B2": "OpenDiLoCo-style QLoRA rounds on >= 2 physical hosts",
    "B3": "Trained adapter passes the train-to-serve gate",
    "C1": "AMCP events on a 3-voter Kafka cluster survive a voter loss",
    "C2": "Registered adapter is routed by domain; failed-gate adapters refused",
    "D1": "GCP control plane follows the low-cost rules; cost estimate recorded",
    "E1": "One-command onboarding; GPU yielded to the owner within seconds",
    "E2": "Poisoned delta and lazy worker rejected live; usage cross-checked",
    "F1": "Public numbers all come from measured_results.json (guarded by a test)",
}


@dataclass
class Row:
    cid: str
    status: str  # PASS | FAIL | NOT-RUN
    reason: str


def _artifact_rows(cid: str) -> Row:
    files = sorted(glob.glob(str(ROOT / "artifacts/mvp" / f"{cid}_*.json")))
    if not files:
        return Row(cid, "NOT-RUN", f"no artifacts/mvp/{cid}_*.json yet")
    data = json.loads(Path(files[-1]).read_text(encoding="utf-8"))
    if data.get("passed") is True:
        return Row(cid, "PASS", f"{Path(files[-1]).name}: passed")
    return Row(cid, "FAIL", f"{Path(files[-1]).name}: {data.get('reason') or 'passed is not true'}")


def check_b3() -> Row:
    files = sorted(glob.glob(str(ROOT / "artifacts/experiments/GRANITE_ADAPTER_*gate*.json")))
    primary = [f for f in files if "posthoc" not in f]
    if not primary:
        return Row("B3", "NOT-RUN", "no gate artifact")
    latest = json.loads(Path(sorted(primary)[-1]).read_text(encoding="utf-8"))
    ok = latest.get("servable") is True
    pooled = latest.get("pooled", {})
    return Row("B3", "PASS" if ok else "FAIL", f"{Path(sorted(primary)[-1]).name}: servable={latest.get('servable')} gain={pooled.get('mean_nll_gain_nats')} ci={pooled.get('bootstrap95')}")


def check_d1() -> Row:
    art = _artifact_rows("D1")
    deploy = ROOT / "deploy/gcp_openrouter/deploy.sh"
    if not deploy.exists():
        return Row("D1", "FAIL", "deploy/gcp_openrouter/deploy.sh missing")
    text = re.sub(r"(--[a-z-]+)=", r"\1 ", deploy.read_text(encoding="utf-8"))  # accept both `--flag=value` and `--flag value`
    text = re.sub(r"\\\s*\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    # hivemind-openrouter is the single-writer Litestream service (AGENTS.md section 2): min=max=1, no CPU throttling, concurrency 80, 1 vCPU / 1 GiB
    need = ["--min-instances 1", "--max-instances 1", "--no-cpu-throttling", "--cpu-boost", "--concurrency 80"]
    missing = [f for f in need if f not in text]
    if not re.search(r'--cpu "?(1|\$CPU)"?', text) and "CPU=1" not in text:
        missing.append("--cpu 1")
    if missing:
        return Row("D1", "FAIL", f"{art.reason or art.status}; also deploy.sh differs from the AGENTS.md flags {missing} (the live service is scale-to-zero, owner decision open)")
    if art.status != "PASS":
        return Row("D1", art.status, "flags match AGENTS.md; " + art.reason)
    return Row("D1", "PASS", "flags match AGENTS.md; " + art.reason)


def check_f1() -> Row:
    bad = ["1015.8", "71.87", "GPT-4o:", "100% vs 64%", "89% DAILY COVERAGE"]
    hits = []
    for p in (ROOT / "src/hivemind/router").glob("*page*.py"):
        t = p.read_text(encoding="utf-8", errors="ignore")
        hits += [f"{p.name}:{b}" for b in bad if b in t]
    if hits:
        return Row("F1", "FAIL", f"fabricated strings still in page modules: {hits[:4]}")
    guard = ROOT / "tests/test_static_numbers_have_artifacts.py"
    if not guard.exists():
        return Row("F1", "NOT-RUN", "benchmarks page is clean, but no CI guard test (tests/test_static_numbers_have_artifacts.py) scans the other page modules yet")
    base_file = ROOT / "tests/data/page_literals_baseline.json"
    remaining = sum(json.loads(base_file.read_text(encoding="utf-8")).values()) if base_file.exists() else None
    if remaining is None:
        return Row("F1", "NOT-RUN", "guard test present but no baseline file")
    if remaining > 0:
        decisions = ROOT / "docs/MVP_OWNER_DECISIONS.md"
        if decisions.exists() and "WAIVED, not passed" in decisions.read_text(encoding="utf-8"):
            return Row("F1", "WAIVED", f"owner decision (docs/MVP_OWNER_DECISIONS.md): {remaining} typed-in page figures kept, still unverified; the ratchet stops new ones. A waiver is not a pass")
        return Row("F1", "FAIL", f"{remaining} typed-in figures remain in page modules (the ratchet test only stops new ones); each must be removed or backed by an artifact")
    return Row("F1", "PASS", "page modules clean, guard test present, no typed-in figures left")


def run() -> list[Row]:
    rows = []
    for cid in CRITERIA:
        if cid == "B3":
            rows.append(check_b3())
        elif cid == "D1":
            rows.append(check_d1())
        elif cid == "F1":
            rows.append(check_f1())
        else:
            rows.append(_artifact_rows(cid))
    return rows


def render(rows: list[Row]) -> str:
    n = sum(r.status == "PASS" for r in rows)
    waived = sum(r.status == "WAIVED" for r in rows)
    out = [
        "# MVP scorecard", "",
        "> **Proof of concept (PoC):** not a production service. Generated by `python scripts/mvp_scorecard.py` from the artifacts in the repository; do not edit by hand.", "",
        f"**{n}/{len(rows)} criteria pass**" + (f" ({waived} more WAIVED by the owner: a waiver is not a pass)" if waived else "") + ". Definitions: `docs/MVP_TARGET.md`.", "",
        "| ID | Criterion | Status | Evidence |", "| :-- | :-- | :-- | :-- |",
    ]
    for r in rows:
        out.append(f"| {r.cid} | {CRITERIA[r.cid]} | **{r.status}** | {r.reason} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write docs/MVP_SCORECARD.md")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rows = run()
    if args.write:
        (ROOT / "docs/MVP_SCORECARD.md").write_text(render(rows), encoding="utf-8")
    print(json.dumps([r.__dict__ for r in rows], indent=2) if args.json else render(rows))
    return 0 if all(r.status == "PASS" for r in rows) else 1  # WAIVED does not make the exit code 0


if __name__ == "__main__":
    sys.exit(main())
