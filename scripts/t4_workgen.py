#!/usr/bin/env python3
"""Work generator: keeps one chained QLoRA MoE workunit queued on the BOINC project, each starting from the latest validated adapter.

The chain is sequential on purpose (one volunteer host, one T4): the next workunit is created only when no training result is
unsent, in progress or awaiting validation. The first round starts from the deterministic LoRA init; every later round starts
from ``hivemind_ledger/latest_adapter.pt`` (written by the assimilator after validation) and the spec carries that adapter's
sha256, which the validator requires the volunteer's chain to start with. Shards cycle through the cached wikitext training
slices; the held-out evaluation set is fixed, so ``eval_loss_after`` is comparable across rounds.

Run with the project venv (torch/transformers for the tokenizer):  <venv>/bin/python t4_workgen.py ...
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datetime import datetime, timezone  # noqa: E402

from hivemind.boinc.real_client.contributor_policy import in_european_night  # noqa: E402
from hivemind.boinc.real_client.moe_profiles import PROFILES, STANDARD, flops_estimate  # noqa: E402
from hivemind.boinc.real_client.moe_work import DOMAIN_ORDER, MIX, build_train_spec  # noqa: E402
from hivemind.boinc.real_client.server_stack import Stack  # noqa: E402

FPOPS = 4.8e14
TRAIN_APP = "hivemind_qlora_moe"


def pending(st: Stack, app: str = TRAIN_APP, prefix: str = "hmq_") -> int:
    # only this generator's own workunits (names hmq_*): a stale result from a test or another generator must not block the chain
    q = (f"use {st.dbname}; select count(*) from result r join app a on a.id=r.appid where a.name='{app}' and r.name regexp '^{prefix}' "
         "and (r.server_state in (2,4) or (r.server_state=5 and r.validate_state=0 and r.outcome=1));")
    return int(st.sql(q).strip().splitlines()[-1])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def prune(ledger: Path, keep: int = 20, prefix: str = "hmq_") -> None:
    files = sorted(ledger.glob(f"adapter_{prefix}*.pt"), key=lambda p: p.stat().st_mtime)
    for p in files[:-keep]:
        p.unlink(missing_ok=True)


def build_training_round_table(ledger: Path, prefix: str = "hmq_") -> list[dict[str, Any]]:
    """Build a structured table of `(round, domain, eval_by_domain_before, eval_by_domain_after)` from ledger receipts (Audit C8)."""
    rows: list[dict[str, Any]] = []
    if not ledger.exists():
        return rows
    for p in sorted(ledger.glob(f"*{prefix}*.json")):
        if p.name.startswith("workgen_state") or p.name in ("latest_adapter.json", "latest_adapter_big.json"):
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        wu_name = str(rec.get("wu_name") or rec.get("name") or p.stem)
        rows.append(
            {
                "round": wu_name,
                "domain": rec.get("domain", "unknown"),
                "eval_by_domain_before": dict(rec.get("eval_by_domain_before") or {}),
                "eval_by_domain_after": dict(rec.get("eval_by_domain_after") or {}),
                "eval_loss_before": rec.get("eval_loss_before"),
                "eval_loss_after": rec.get("eval_loss_after"),
                "sha256": rec.get("sha256") or rec.get("adapter_sha256"),
                "regressed": (
                    rec.get("eval_loss_before") is not None
                    and rec.get("eval_loss_after") is not None
                    and float(rec["eval_loss_after"]) > float(rec["eval_loss_before"])
                ),
            }
        )
    return rows


def select_round_head_with_rollback(
    ledger: Path,
    stem: str = "latest_adapter",
    prefix: str = "hmq_",
) -> dict[str, Any] | None:
    """Enforce single-metric rollback gate (Audit C8): if `latest_adapter` regressed on mean held-out loss, roll back to the previous non-regressed checkpoint."""
    head_json = ledger / f"{stem}.json"
    if not head_json.exists():
        return None
    head: dict[str, Any] = json.loads(head_json.read_text(encoding="utf-8"))
    loss_before = head.get("eval_loss_before")
    loss_after = head.get("eval_loss_after")
    regressed = (
        loss_before is not None
        and loss_after is not None
        and float(loss_after) > float(loss_before)
    )
    if not regressed:
        head.setdefault("rolled_back", False)
        return head

    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    for p in sorted(ledger.glob(f"*{prefix}*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        if p.name == head_json.name or p.name.startswith("workgen_state"):
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        rec_sha = rec.get("sha256") or rec.get("adapter_sha256")
        if not rec_sha or rec_sha == head.get("sha256"):
            continue
        rb = rec.get("eval_loss_before")
        ra = rec.get("eval_loss_after")
        if rb is not None and ra is not None and float(ra) > float(rb):
            continue
        wu_name = str(rec.get("wu_name") or rec.get("name") or p.stem.replace("adapter_", ""))
        pt_cand = ledger / f"adapter_{wu_name}.pt"
        if not pt_cand.exists():
            pt_alt = p.with_suffix(".pt")
            if pt_alt.exists():
                pt_cand = pt_alt
            else:
                continue
        candidates.append((float(ra) if ra is not None else 0.0, pt_cand, rec))
        break

    if candidates:
        _, pt_cand, prev_rec = candidates[0]
        shutil.copyfile(pt_cand, ledger / f"{stem}.pt")
        rolled_head = {
            **prev_rec,
            "sha256": prev_rec.get("sha256") or prev_rec.get("adapter_sha256"),
            "rolled_back": True,
            "rolled_back_from_sha256": head.get("sha256"),
            "rolled_back_reason": f"eval_loss_after ({loss_after}) > eval_loss_before ({loss_before})",
        }
        head_json.write_text(json.dumps(rolled_head), encoding="utf-8")
        return rolled_head

    # No prior non-regressed checkpoint found: reset to fresh unchained init
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack-base", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/boinc/stack"))
    ap.add_argument("--boinc-src", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/boinc/src"))
    ap.add_argument("--interval-s", type=float, default=30.0)
    ap.add_argument("--shards", type=int, default=50, help="shards per domain before the cursor wraps")
    ap.add_argument("--inner-steps", type=int, default=80, help="a round is ~3-4 min on the shared T4; rounds are only run when the node is idle")
    ap.add_argument("--segment-steps", type=int, default=10)
    ap.add_argument("--profile", choices=sorted(PROFILES), default=STANDARD, help="standard: any GPU >= 3.5 GB; big: its own app, plan class and adapter chain for GPUs >= 10 GB")
    ap.add_argument("--night-only", action="store_true", help="create work only 22:00-07:00 Europe/Paris (community work: offered when end-user load is lowest)")
    ap.add_argument("--mix", action="store_true", help="stratified rounds over the six evaluated domains instead of one domain per round (the single-domain pattern failed the train-to-serve gate)")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    prof = PROFILES[a.profile]
    big = a.profile != STANDARD
    stem = prof["chain_stem"]
    st = Stack(a.stack_base, a.boinc_src, sys.executable, ROOT / "src")
    ledger = st.root / "hivemind_ledger"
    state_f = ledger / ("workgen_state_big.json" if big else "workgen_state.json")
    ledger.mkdir(exist_ok=True)
    chain_template = st.ensure_chain_template(prof["app"])
    while True:
        try:
            in_window = (not a.night_only) or in_european_night(_now())
            if in_window and pending(st, prof["app"], prof["prefix"]) == 0:
                rnd = (json.loads(state_f.read_text())["round"] if state_f.exists() else 0) + 1
                head = select_round_head_with_rollback(ledger, stem=stem, prefix=prof["prefix"])
                name = f"{prof['prefix']}{rnd:05d}"
                if a.mix:  # stratified rounds: every round draws from the six evaluated domains (see moe_work.MIX_DOMAINS)
                    domain, shard = MIX, (rnd - 1) % a.shards
                else:
                    domain = DOMAIN_ORDER[(rnd - 1) % len(DOMAIN_ORDER)]  # one domain per round; each domain advances its own cursor
                    shard = ((rnd - 1) // len(DOMAIN_ORDER)) % a.shards
                spec = build_train_spec(name, shard_index=shard, domain=domain, inner_steps=a.inner_steps, segment_steps=a.segment_steps,
                                        start_adapter_sha256=head["sha256"] if head else None, profile=None if not big else a.profile)
                eval_domains = sorted((spec.get("eval_sets") or {}).keys())
                spec["eval_basis"] = {
                    "basis_id": "6_domain_mix_stratified" if domain == MIX else f"{len(eval_domains)}_domain_extended",
                    "domains": eval_domains,
                    "domain_count": len(eval_domains),
                    "held_out_set": "8x128_tokens_per_domain",
                    "baseline_nll": 2.12,
                    "basis_changed_at_hmq_00171": rnd >= 171 or domain == MIX,
                }
                (st.root / "hivemind_specs" / f"{name}.json").write_text(json.dumps(spec))
                inputs = {"in.json": json.dumps(spec, sort_keys=True).encode()}
                template = None
                if head:
                    adapter_bytes = (ledger / f"{stem}.pt").read_bytes()
                    inputs["adapter.pt"] = adapter_bytes
                    template = chain_template
                st.create_work(prof["app"], name, inputs, flops_estimate(a.profile) if big else FPOPS, template=template)
                state_f.write_text(json.dumps({"round": rnd, "created": name, "chained": bool(head), "eval_basis": spec["eval_basis"], "t": time.time()}))
                print(f"created {name} (domain {domain}, shard {shard}, chained={bool(head)})", flush=True)
                prune(ledger, prefix=prof["prefix"])
        except Exception as exc:
            print(f"workgen error: {exc}", file=sys.stderr, flush=True)
        if a.once:
            return 0
        time.sleep(a.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
