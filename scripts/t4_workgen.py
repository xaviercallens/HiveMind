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
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datetime import datetime, timezone  # noqa: E402

from hivemind.boinc.real_client.contributor_policy import in_european_night  # noqa: E402
from hivemind.boinc.real_client.moe_profiles import PROFILES, STANDARD, flops_estimate  # noqa: E402
from hivemind.boinc.real_client.moe_work import DOMAIN_ORDER, build_train_spec  # noqa: E402
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
            # Outside the European night window when --night-only is set: create nothing, results already out keep validating.
            if (not a.night_only or in_european_night(_now())) and pending(st, prof["app"], prof["prefix"]) == 0:
                rnd = (json.loads(state_f.read_text())["round"] if state_f.exists() else 0) + 1
                head = json.loads((ledger / f"{stem}.json").read_text()) if (ledger / f"{stem}.json").exists() else None
                name = f"{prof['prefix']}{rnd:05d}"
                domain = DOMAIN_ORDER[(rnd - 1) % len(DOMAIN_ORDER)]  # one domain per round; each domain advances its own cursor
                shard = ((rnd - 1) // len(DOMAIN_ORDER)) % a.shards
                spec = build_train_spec(name, shard_index=shard, domain=domain, inner_steps=a.inner_steps, segment_steps=a.segment_steps,
                                        start_adapter_sha256=head["sha256"] if head else None, profile=None if not big else a.profile)
                (st.root / "hivemind_specs" / f"{name}.json").write_text(json.dumps(spec))
                inputs = {"in.json": json.dumps(spec, sort_keys=True).encode()}
                template = None
                if head:
                    adapter_bytes = (ledger / f"{stem}.pt").read_bytes()
                    inputs["adapter.pt"] = adapter_bytes
                    template = chain_template
                st.create_work(prof["app"], name, inputs, flops_estimate(a.profile) if big else FPOPS, template=template)
                state_f.write_text(json.dumps({"round": rnd, "created": name, "chained": bool(head), "t": time.time()}))
                print(f"created {name} (domain {domain}, shard {shard}, chained={bool(head)})", flush=True)
                prune(ledger, prefix=prof["prefix"])
        except Exception as exc:
            print(f"workgen error: {exc}", file=sys.stderr, flush=True)
        if a.once:
            return 0
        time.sleep(a.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
