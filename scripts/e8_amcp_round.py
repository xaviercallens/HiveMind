#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Co-training rounds over AMCP: a coordinator offers signed tasks, learners pull them with an OUTBOUND connection only, and send back LoRA deltas.

Same training core as ``scripts/e8_qlora_moe_round.py`` (tiny Granite-style MoE, NF4 + LoRA); the transport is AMCP on Kafka instead of a direct HTTP connection, using
``hivemind.swarm.federation_agents``. A learner needs: the repository checkout, the shared signing keyring in the environment, and a route to the Kafka bootstrap servers.
It does NOT need an open port, a tunnel to the coordinator, or any code from the network: a task is refused unless it pins the code version the learner already has.

  export HIVEMIND_KAFKA_BOOTSTRAP=10.77.0.1:9092,10.77.0.2:9092,10.77.0.3:9092
  export HIVEMIND_AMCP_HMAC_KEYS=fed1:<64 hex characters>        # shared out of band; never commit it
  python scripts/e8_amcp_round.py coordinator --run-id night-001 --learners l0 l1 --rounds 40 --out artifacts/experiments/e8_real/night-001.json
  python scripts/e8_amcp_round.py learner --run-id night-001 --learner-id l0 --quant nf4 --device cuda --night-only

A learner that is not allowed to work right now (outside the night window, inference in flight on its GPU according to the governor log, or the free VRAM would drop below the floor)
simply does not claim the task; the coordinator closes the round with the quorum it gets. No round can be forced onto a machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import e8_qlora_moe_round as q  # noqa: E402
from hivemind.swarm import federation_agents as fed  # noqa: E402
from hivemind.swarm.amcp_kafka_transport import (  # noqa: E402
    AMCPConsumer,
    AMCPPublisher,
    EnvelopeVerifier,
    HMACKeyring,
    KafkaBrokerTransport,
    KafkaSettings,
)

PIN_FILES = [HERE / "e8_qlora_moe_round.py", HERE / "e8_amcp_round.py", HERE.parent / "src" / "hivemind" / "swarm" / "federation_agents.py"]
DEFAULT_GOVERNOR_LOG = "/mnt/disks/disk-socrateai-local-1/hivemind_telemetry/governor.jsonl"


# ------------------------------------------------------------------------------------------------------------------------------ bus


class Bus:
    """One participant's connection to the federation topic: publish signed events, drain the verified ones."""

    def __init__(self, transport: Any, ring: HMACKeyring, run_id: str, ttl_s: float = 900.0) -> None:
        self.pub = AMCPPublisher(transport, ring, ttl_s=ttl_s)
        self.inbox: Deque[Any] = deque()
        self.run_id = run_id
        self.con = AMCPConsumer(transport, EnvelopeVerifier(ring), topics=["federation"], poll_timeout_s=1.0)
        for t in (fed.OFFER, fed.CLAIM, fed.RESULT, fed.ABORT):
            self.con.register(t, self._take)

    def _take(self, verified: Any) -> None:
        if verified.event.payload.get("run_id") == self.run_id:
            self.inbox.append(verified.event)

    def poll(self) -> List[Any]:
        self.con.run_once()
        out = list(self.inbox)
        self.inbox.clear()
        return out

    def publish(self, event: Any) -> None:
        self.pub.publish(event)


def make_transport(args: argparse.Namespace, role: str) -> Any:
    bootstrap = args.bootstrap or os.environ.get("HIVEMIND_KAFKA_BOOTSTRAP", "")
    settings = KafkaSettings(bootstrap=bootstrap, group_id=f"fed-{args.run_id}-{role}-{uuid.uuid4().hex[:8]}", client_id=f"fed-{role}", auto_offset_reset="latest")
    return KafkaBrokerTransport(settings, flush_timeout_s=20.0)


# ------------------------------------------------------------------------------------------------------------------------------ gating (learner side)


def in_night_window(now: Optional[datetime] = None) -> bool:
    """22:00 to 07:00 Europe/Paris, the project's night window."""
    from zoneinfo import ZoneInfo

    h = (now or datetime.now(ZoneInfo("Europe/Paris"))).hour
    return h >= 22 or h < 7


def governor_busy(path: str) -> Optional[bool]:
    """True if the T4 governor's latest snapshot says inference is in flight; None if there is no log to read (then this check does not apply)."""
    try:
        last = ""
        with open(path, "rb") as fh:
            fh.seek(max(0, os.path.getsize(path) - 4096))
            for line in fh.read().decode("utf-8", "replace").splitlines():
                if line.strip():
                    last = line
        return int(json.loads(last).get("in_flight", 0)) > 0
    except (OSError, ValueError):
        return None


def gate(args: argparse.Namespace, device: str, now: Optional[datetime] = None) -> Tuple[bool, str]:
    if args.night_only and not in_night_window(now):
        return False, "outside the night window"
    if args.governor_log and governor_busy(args.governor_log):
        return False, "inference in flight (governor)"
    try:
        q.vram_guard(args.min_free_mib, args.need_mib, device)
    except q.GuardError as exc:
        return False, f"vram: {exc}"
    return True, "ok"


# ------------------------------------------------------------------------------------------------------------------------------ coordinator


def write_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))


def run_coordinator(args: argparse.Namespace, transport: Any = None, ring: Optional[HMACKeyring] = None) -> int:
    ring = ring or HMACKeyring.from_env()
    transport = transport or make_transport(args, "coordinator")
    bus = Bus(transport, ring, args.run_id)
    pin = fed.code_pin(PIN_FILES)
    base = q.build_base_state(args.seed)
    base_sha = fed.tensors_sha256(base)
    global_state = q.initial_lora(base, args.seed)
    velocity: Optional[Dict[str, torch.Tensor]] = None
    start = 0
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    if ckpt and args.resume and ckpt.exists():
        blob = torch.load(ckpt, map_location="cpu", weights_only=True)
        global_state = {k[2:]: v for k, v in blob.items() if k.startswith("g.")}
        velocity = {k[2:]: v for k, v in blob.items() if k.startswith("v.")} or None
        start = int(blob["round"])
        print(f"[coordinator] resumed at round {start} from {ckpt}", flush=True)
    model = q.load_model(base, "none", "cpu", args.seed)
    evalset = q.make_eval(len(args.learners), args.seed)
    q.set_lora_state(model, global_state)
    log: List[Dict[str, Any]] = []
    initial = q.eval_loss(model, evalset, "cpu")
    t0 = time.time()
    print(f"[coordinator] run {args.run_id} pin {pin[:12]} base {base_sha[:12]} learners {args.learners} quorum {args.quorum} initial eval {initial:.4f}", flush=True)
    ok = True
    for rnd in range(start, args.rounds):
        retries = 0
        while True:
            deadline = time.time() + args.round_deadline_s
            offer = fed.build_offer(run_id=args.run_id, rnd=rnd, sender_id="coordinator", global_state=global_state, base_sha256=base_sha, pin=pin, inner_steps=args.inner_steps,
                                    deadline_unix=deadline, quorum=args.quorum,
                                    params={"seed": args.seed, "inner_lr": args.inner_lr, "batch": args.batch, "shard_size": args.shard_size, "learners": len(args.learners)})
            bus.publish(offer)
            book = fed.RoundBook(run_id=args.run_id, rnd=rnd, allowed=args.learners, quorum=args.quorum, deadline_unix=deadline, global_sha256=offer.payload["global_sha256"], pin=pin,
                                 max_delta_l2=args.max_delta_l2, template=global_state)
            t_round = time.time()
            while book.status() == "open":
                for ev in bus.poll():
                    if ev.event_type == fed.CLAIM:
                        book.claim(ev.payload)
                    elif ev.event_type == fed.RESULT:
                        book.offer_result(ev.payload)
            status = book.status()
            if status != "failed":
                break
            bus.publish(fed.build_abort(run_id=args.run_id, rnd=rnd, sender_id="coordinator", reason=f"quorum {args.quorum} not reached; rejected={len(book.rejected)}"))
            retries += 1
            print(f"[coordinator] round {rnd}: quorum not reached (accepted {sorted(book.accepted)}, claims {sorted(book.claims)}, rejected {book.rejected}); retry {retries}", flush=True)
            if retries > args.max_retries:
                ok = False
                break
        if not ok:
            break
        ids = sorted(book.accepted)
        global_state, velocity = q.outer_step(global_state, [book.accepted[i]["delta"] for i in ids], [book.accepted[i]["tokens"] for i in ids], velocity, args.outer_lr, args.outer_mu)
        q.set_lora_state(model, global_state)
        after = q.eval_loss(model, evalset, "cpu")
        log.append({"round": rnd, "status": status, "accepted": ids, "hosts": {i: book.accepted[i]["host"] for i in ids}, "claims": sorted(book.claims), "retries": retries,
                    "rejected": [list(r) for r in book.rejected], "eval_loss_before": sum(book.accepted[i]["eval_loss_global"] for i in ids) / len(ids), "eval_loss_after": after,
                    "delta_l2": {i: book.accepted[i]["delta_l2"] for i in ids}, "vram_peak_mib": {i: book.accepted[i]["vram_peak_mib"] for i in ids},
                    "latency_s": time.time() - t_round, "wall_s": time.time() - t0})
        print(f"[coordinator] round {rnd}: {status} {ids} eval {after:.4f} ({log[-1]['latency_s']:.1f}s)", flush=True)
        if ckpt:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({**{f"g.{k}": v for k, v in global_state.items()}, **{f"v.{k}": v for k, v in (velocity or {}).items()}, "round": torch.tensor(rnd + 1)}, ckpt)
        if rnd % max(1, args.report_every) == 0:
            write_json(args.out, report(args, pin, base_sha, initial, log, ok, False))
        spare = args.pace_s - (time.time() - t_round)
        if spare > 0:  # a night-long run should not publish a 40 KB message every few seconds: keep a calm cadence
            time.sleep(spare)
    bus.publish(fed.build_abort(run_id=args.run_id, rnd=args.rounds, sender_id="coordinator", reason="run_complete" if ok else "run_failed"))
    rep = report(args, pin, base_sha, initial, log, ok, True)
    write_json(args.out, rep)
    print(f"[coordinator] done: {len(log)} rounds, final eval {rep['final_eval_loss']}", flush=True)
    return 0 if ok else 2


def report(args: argparse.Namespace, pin: str, base_sha: str, initial: float, log: List[Dict[str, Any]], ok: bool, final: bool) -> Dict[str, Any]:
    rep: Dict[str, Any] = {"schema_version": 1, "experiment_id": "E8-stage2-amcp-night", "run_id": args.run_id, "completed": bool(ok and final), "final": final,
                           "coordinator_host": platform.node(), "code_pin": pin, "base_sha256": base_sha,
                           "config": {"learners": args.learners, "quorum": args.quorum, "rounds": args.rounds, "seed": args.seed, "inner_steps": args.inner_steps, "inner_lr": args.inner_lr,
                                      "batch": args.batch, "outer_lr": args.outer_lr, "outer_mu": args.outer_mu, "round_deadline_s": args.round_deadline_s},
                           "initial_eval_loss": initial, "rounds": log, "final_eval_loss": log[-1]["eval_loss_after"] if log else None,
                           "hosts_seen": sorted({h for r in log for h in r["hosts"].values()})}
    rep["receipt_sha256"] = hashlib.sha256(json.dumps(rep, sort_keys=True).encode()).hexdigest()
    return rep


# ------------------------------------------------------------------------------------------------------------------------------ learner


def run_learner(args: argparse.Namespace, transport: Any = None, ring: Optional[HMACKeyring] = None) -> int:
    ring = ring or HMACKeyring.from_env()
    device = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    transport = transport or make_transport(args, f"learner-{args.learner_id}")
    bus = Bus(transport, ring, args.run_id)
    pin = fed.code_pin(PIN_FILES)
    host = args.host_label or platform.node()
    model: Any = None
    base_cache: Dict[int, Tuple[Dict[str, torch.Tensor], str]] = {}
    done_rounds: set = set()
    seen_offers: set = set()
    skipped: Dict[str, int] = {}
    last_offer = time.time()
    print(f"[learner {args.learner_id}] {host} device={device} quant={args.quant} pin {pin[:12]} waiting for offers of run {args.run_id}", flush=True)
    while True:
        offers = []
        for ev in bus.poll():
            if ev.event_type == fed.ABORT and ev.payload.get("reason") in ("run_complete", "run_failed"):
                print(f"[learner {args.learner_id}] run ended ({ev.payload['reason']}); skipped so far: {skipped}", flush=True)
                return 0
            if ev.event_type == fed.OFFER:
                offers.append(ev)
        if not offers:
            if args.idle_exit_s and time.time() - last_offer > args.idle_exit_s:
                print(f"[learner {args.learner_id}] no offer for {args.idle_exit_s:.0f}s; leaving", flush=True)
                return 0
            continue
        last_offer = time.time()
        ev = max(offers, key=lambda e: (int(e.payload["round"]), float(e.payload.get("deadline_unix", 0))))
        p = ev.payload
        rnd = int(p["round"])
        offer_key = (rnd, float(p.get("deadline_unix", 0)))
        # A round is done once a result was sent. A refusal or a decline applies to ONE offer only: when the coordinator re-offers the round (new deadline) after
        # a quorum failure, it is considered afresh. (Marking the round done on a decline stopped the first night run: inference was busy for one round, the
        # learners ignored every re-offer of it, and the coordinator gave up.)
        if rnd in done_rounds or offer_key in seen_offers:
            continue
        seen_offers.add(offer_key)
        seed = int(p.get("params", {}).get("seed", 1))
        if seed not in base_cache:
            b = q.build_base_state(seed)
            base_cache[seed] = (b, fed.tensors_sha256(b))
        base, base_sha = base_cache[seed]
        try:
            g = fed.check_offer(p, my_pin=pin, my_base_sha256=base_sha, run_id=args.run_id)
        except fed.OfferRefused as exc:
            skipped[str(exc)[:60]] = skipped.get(str(exc)[:60], 0) + 1
            print(f"[learner {args.learner_id}] round {rnd}: refused: {exc}", flush=True)
            continue
        ok, why = gate(args, device)
        if not ok:
            skipped[why[:60]] = skipped.get(why[:60], 0) + 1
            print(f"[learner {args.learner_id}] round {rnd}: not claiming ({why})", flush=True)
            continue
        params = p.get("params", {})
        if model is None:
            model = q.load_model(base, args.quant, device, seed)
            shard = q.make_shard(args.index, int(params.get("shard_size", 512)), seed)
            evalset = q.make_eval(int(params.get("learners", 2)), seed)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
        bus.publish(fed.build_claim(run_id=args.run_id, rnd=rnd, learner_id=args.learner_id, host=host))
        out = q.learner_round(model, g, shard, evalset, steps=int(p["inner_steps"]), lr=float(params.get("inner_lr", 2e-3)), batch=int(params.get("batch", 16)), seed=seed, rnd=rnd,
                              learner_id=args.index, device=device, shuffle_labels=args.shuffle_labels)
        peak = float(torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None
        if time.time() > float(p["deadline_unix"]):
            print(f"[learner {args.learner_id}] round {rnd}: finished after the deadline; the coordinator will not use it", flush=True)
        bus.publish(fed.build_result(run_id=args.run_id, rnd=rnd, learner_id=args.learner_id, host=host, tokens=out["tokens"], delta=out["delta"], delta_l2=out["delta_l2"],
                                     eval_loss_global=out["eval_loss_global"], pin=pin, global_sha256=p["global_sha256"], vram_peak_mib=peak))
        done_rounds.add(rnd)
        print(f"[learner {args.learner_id}] round {rnd}: global eval {out['eval_loss_global']:.4f}, |delta| {out['delta_l2']:.4f}", flush=True)


# ------------------------------------------------------------------------------------------------------------------------------ cli


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run-id", required=True)
        p.add_argument("--bootstrap", default="", help="Kafka bootstrap servers (default: HIVEMIND_KAFKA_BOOTSTRAP)")

    c = sub.add_parser("coordinator")
    common(c)
    c.add_argument("--learners", nargs="+", required=True, help="the learner ids allowed to contribute (a result from anyone else is rejected)")
    c.add_argument("--quorum", type=int, default=1)
    c.add_argument("--rounds", type=int, default=8)
    c.add_argument("--round-deadline-s", type=float, default=120.0)
    c.add_argument("--max-retries", type=int, default=3)
    c.add_argument("--max-delta-l2", type=float, default=50.0)
    c.add_argument("--inner-steps", type=int, default=40)
    c.add_argument("--inner-lr", type=float, default=2e-2)
    c.add_argument("--batch", type=int, default=16)
    c.add_argument("--shard-size", type=int, default=512)
    c.add_argument("--outer-lr", type=float, default=0.7)
    c.add_argument("--outer-mu", type=float, default=0.65)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--pace-s", type=float, default=0.0, help="minimum seconds per round (a calm cadence for long runs)")
    c.add_argument("--report-every", type=int, default=10, help="rewrite the JSON report every N rounds (it is always written at the end)")
    c.add_argument("--checkpoint", default="")
    c.add_argument("--resume", action="store_true")
    c.add_argument("--out", required=True)
    lr = sub.add_parser("learner")
    common(lr)
    lr.add_argument("--learner-id", required=True)
    lr.add_argument("--index", type=int, default=0, help="which data shard this learner owns (0, 1, ...)")
    lr.add_argument("--host-label", default="")
    lr.add_argument("--quant", choices=["none", "nf4"], default="none")
    lr.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    lr.add_argument("--shuffle-labels", action="store_true", help="negative control: a learner that trains on shuffled targets")
    lr.add_argument("--night-only", action="store_true", help="claim tasks only between 22:00 and 07:00 Europe/Paris")
    lr.add_argument("--governor-log", default=DEFAULT_GOVERNOR_LOG if Path(DEFAULT_GOVERNOR_LOG).exists() else "", help="do not claim while this governor log shows inference in flight")
    lr.add_argument("--min-free-mib", type=float, default=4300.0)
    lr.add_argument("--need-mib", type=float, default=600.0)
    lr.add_argument("--idle-exit-s", type=float, default=0.0, help="leave after this many seconds without an offer (0 = never)")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return {"coordinator": run_coordinator, "learner": run_learner}[args.mode](args)
    except (q.GuardError, fed.FederationError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
