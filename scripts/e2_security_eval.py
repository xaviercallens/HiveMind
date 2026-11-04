#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""MVP criterion E2 evaluation: are poisoned and lazy contributions rejected, honest ones accepted, and producer-reported usage bounded?

Four independent pieces of evidence, each labelled with what it is:
  1. ``multikrum``  - the real filter (core/diloco_outer.py) on synthetic LoRA-shaped deltas: honest workers accepted, a 12x-scaled poisoned delta and a sign-flipped one rejected. OFFLINE.
  2. ``ledger``     - the real BOINC server's verdicts (read-only): an honest workunit verified, a lazy worker rejected by the replay spot-check. LIVE.
  3. ``metering``   - the gateway's anti-inflation cap (router/gateway.py meter_usage): the largest billable inflation over an honest 4-bytes-per-token report. MEASURED, and loose.
  4. ``wired``      - whether Multi-Krum is actually called by the BOINC training path (moe_app / moe_bridge). STATIC check.
``passed`` is true only if a poisoned delta was rejected LIVE (i.e. Multi-Krum is wired into the path volunteers use), the lazy worker was rejected LIVE, and billing inflation is bounded
within 1.5x of an honest report. Today it is not, and the artifact says why.

    python scripts/e2_security_eval.py [--ledger <dir>] [--out artifacts/mvp/E2_security.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_LEDGER = "/mnt/disks/disk-socrateai-local-1/boinc/stack/project/hmproj/hivemind_ledger"


def multikrum_eval(n_honest: int = 6, dim: int = 512, seed: int = 11) -> dict:
    import torch

    from hivemind.core.diloco_outer import filter_byzantine_deltas_multikrum

    g = torch.Generator().manual_seed(seed)
    direction = torch.randn(dim, generator=g)
    deltas = {}
    for i in range(n_honest):  # honest workers: the same descent direction plus small noise
        noise = 0.1 * torch.randn(dim, generator=g)
        deltas[f"honest-{i}"] = {"lora_A": (direction + noise)[: dim // 2].clone(), "lora_B": (direction + noise)[dim // 2 :].clone()}
    poisoned = {"lora_A": 12 * direction[: dim // 2], "lora_B": 12 * direction[dim // 2 :]}  # scaled 12x
    flipped = {"lora_A": -direction[: dim // 2], "lora_B": -direction[dim // 2 :]}  # sign-flipped
    deltas["poisoned-scaled-12x"], deltas["poisoned-sign-flip"] = poisoned, flipped
    report = filter_byzantine_deltas_multikrum(deltas, max_byzantine_f=2)
    honest_ids = {k for k in deltas if k.startswith("honest")}
    return {
        "kind": "offline, synthetic deltas, real filter", "accepted": sorted(report.accepted_workers), "rejected": sorted(report.rejected_workers),
        "quarantined_pre_krum": report.quarantined_pre_krum,
        "poisoned_rejected": all(p in report.rejected_workers or p in report.quarantined_pre_krum for p in ("poisoned-scaled-12x", "poisoned-sign-flip")),
        "honest_accepted_fraction": round(len(honest_ids & set(report.accepted_workers)) / len(honest_ids), 3),
    }


def ledger_evidence(ledger: str) -> dict:
    out = {"kind": "live BOINC server verdicts (read-only)", "ledger": ledger, "honest_verified": None, "lazy_rejected": None, "details": []}
    files = sorted(glob.glob(ledger + "/verdict_*.json"))
    if not files:
        out["error"] = "ledger not found"
        return out
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        wu = str(d.get("wu_id", ""))
        if wu.startswith("hmq_"):
            continue
        out["details"].append({"wu": wu, "valid": d.get("valid"), "reason": str(d.get("reason"))[:100]})
        if "good" in wu:
            out["honest_verified"] = bool(d.get("valid"))
        if "lazy" in wu:
            out["lazy_rejected"] = d.get("valid") is False
    chain = [json.loads(Path(f).read_text(encoding="utf-8")) for f in glob.glob(ledger + "/verdict_hmq_*.json")]
    out["chain_verdicts"] = {"valid": sum(1 for d in chain if d.get("valid")), "invalid": sum(1 for d in chain if not d.get("valid"))}
    return out


def metering_eval() -> dict:
    from hivemind.router.gateway import meter_usage

    # an honest producer reports ~1 token per 4 bytes. A dishonest one reports a huge number.
    completion_bytes = 4000
    honest = meter_usage(0, completion_bytes // 4, 0, completion_bytes, 1024)[1]
    inflated = meter_usage(0, 10**9, 0, completion_bytes, 10**6)[1]
    return {"kind": "measured on gateway.meter_usage", "completion_bytes": completion_bytes, "honest_billed_tokens": honest, "max_billable_tokens_for_these_bytes": inflated,
            "max_inflation_factor": round(inflated / max(honest, 1), 2),
            "note": "capped by the completion BYTES (a token is at least one byte), so a producer can still overstate by up to the bytes-per-token ratio (about 4x for English); there is no tokenizer-based cross-check yet"}


def wired_eval() -> dict:
    hits = {}
    for rel in ("src/hivemind/boinc/real_client/moe_app.py", "src/hivemind/boinc/real_client/moe_bridge.py", "src/hivemind/boinc/real_client/resident_validator.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        hits[rel] = any(w in text for w in ("filter_byzantine_deltas_multikrum", "DiLoCoOuterOptimizer", "multikrum"))
    return {"kind": "static check", "multikrum_called_from": hits, "multikrum_wired_into_boinc_training": any(hits.values())}


def run(ledger: str = DEFAULT_LEDGER) -> dict:
    mk, led, met, wired = multikrum_eval(), ledger_evidence(ledger), metering_eval(), wired_eval()
    live_poison = bool(wired["multikrum_wired_into_boinc_training"] and mk["poisoned_rejected"])
    checks = {
        "multikrum_rejects_poisoned_offline": bool(mk["poisoned_rejected"]),
        "multikrum_accepts_honest_offline": mk["honest_accepted_fraction"] >= 0.8,
        "honest_workunit_verified_live": led.get("honest_verified") is True,
        "lazy_worker_rejected_live": led.get("lazy_rejected") is True,
        "poisoned_delta_rejected_live": live_poison,
        "billing_inflation_within_1_5x": met["max_inflation_factor"] <= 1.5,  # the bytes cap allows about 4x over an honest report until a tokenizer cross-check exists
    }
    failed = [k for k, v in checks.items() if not v]
    return {"criterion": "E2", "multikrum": mk, "ledger": led, "metering": met, "wired": wired, "checks": checks, "passed": not failed,
            "reason": "" if not failed else "not met: " + "; ".join(failed)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=DEFAULT_LEDGER)
    ap.add_argument("--out", default=str(ROOT / "artifacts/mvp/E2_security.json"))
    args = ap.parse_args(argv)
    result = run(args.ledger)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": result["checks"], "reason": result["reason"]}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
