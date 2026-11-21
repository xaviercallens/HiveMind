#!/usr/bin/env python3
"""Write two receipts for the paper's live-status subsection: the chained QLoRA held-out status (from the BOINC ledger) and the
NIGHT-INTENSE-1 feasibility probes (from the queue's arm files). Read-only on the inputs; numbers are computed, never typed.

    python3 scripts/chain_status_receipt.py --ledger <assimilated.jsonl> --probe-dir <night_probe/20261028...> --out artifacts/experiments
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

DOMAINS = ["biomed", "edu_web", "french", "general", "math", "news"]
RANGES = ((1, 98), (99, 170), (171, 10**6))


def wu_number(row: dict) -> int:
    m = re.search(r"hmq_(\d+)", row.get("adapter", ""))
    return int(m.group(1)) if m else -1


def chain_receipt(ledger: Path) -> dict:
    raw = ledger.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if wu_number(r) >= 0]
    rows.sort(key=wu_number)
    out = {
        "schema": "hivemind-chain-status/1",
        "source": "BOINC QLoRA ledger assimilated.jsonl (not in the repository)",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "rows_with_wu": len(rows),
        "last_wu": wu_number(rows[-1]),
        "ranges": {},
    }
    for lo, hi in RANGES:
        sel = [r for r in rows if lo <= wu_number(r) <= hi]
        changes = []
        for r in sel:
            b, a = r.get("eval_by_domain_before") or {}, r.get("eval_by_domain_after") or {}
            changes += [a[d] - b[d] for d in DOMAINS if d in a and d in b]
        key = f"hmq_{lo:05d}_{hi if hi < 10**6 else 'end'}"
        out["ranges"][key] = {
            "rounds": len(sel),
            "domain_changes": len(changes),
            "median_change_nats": round(statistics.median(changes), 4) if changes else None,
            "share_worse": round(sum(1 for c in changes if c > 0) / len(changes), 3) if changes else None,
        }
    last = rows[-1]
    out["last_round"] = {"wu": wu_number(last), "domain": last.get("domain"),
                         "after": {k: round(v, 4) for k, v in (last.get("eval_by_domain_after") or {}).items()}}
    diag = [r for r in rows if wu_number(r) == 187]
    if diag:
        b, a = diag[0]["eval_by_domain_before"], diag[0]["eval_by_domain_after"]
        ch = [a[d] - b[d] for d in DOMAINS if d in a and d in b]
        out["diagnostic_round_187"] = {"domain": diag[0].get("domain"), "domain_changes": len(ch),
                                       "median_change_nats": round(statistics.median(ch), 4), "share_worse": round(sum(1 for c in ch if c > 0) / len(ch), 3)}
    return out


def probe_receipt(probe_dirs: list) -> dict:
    arms = []
    for p in sorted(f for d in probe_dirs for f in Path(d).glob("arm_A*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        est, m = d.get("estimate") or {}, d.get("measured") or {}
        ea, pa = est.get("alloc_peak_mib"), m.get("torch_peak_allocated_mib")
        total = m.get("nvidia_smi_peak_used_mib")
        arms.append({
            "arm": d.get("arm"), "model": d.get("model_id"), "status": d.get("status"),
            "estimate_alloc_mib": ea, "measured_alloc_mib": pa,
            "error_vs_alloc_pct": round(100.0 * (pa / ea - 1.0), 1) if ea and pa else None,
            "total_gpu_peak_mib": total, "free_at_peak_mib": (15360 - total) if total else None,
            "agents_md_4300_floor_ok": (15360 - total >= 4300) if total else None,
            "max_temp_c": m.get("max_temp_c"), "max_power_w": m.get("max_power_w"),
        })
    return {"schema": "hivemind-night-probe-feasibility/1", "source": [Path(d).name for d in probe_dirs], "arms": arms,
            "not_measured": ["quality (held-out loss)", "sustained thermal behaviour", "checkpointing through the real trainer step"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--probe-dir", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    c = chain_receipt(a.ledger)
    (a.out / "CHAIN_STATUS_2026-11-07.json").write_text(json.dumps(c, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    p = probe_receipt(a.probe_dir)
    (a.out / "NIGHT_INTENSE_1_probe_results_2026-10-28.json").write_text(json.dumps(p, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
