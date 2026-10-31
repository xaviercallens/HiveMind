"""Experiment E8-pilot: two real hosts over the public internet running Decoupled DiLoCo on the
E1 toy LoRA causal LM (pre-registered scale-down of E8 in ``docs/RESEARCH_PROGRAM.md``).

One process is the **syncer** (this box). Each **learner** runs on a host of its own (here: this
box and a Google Cloud TPU VM in ``us-west4-a`` reached through an SSH reverse tunnel, so every
delta crosses the internet). The syncer ships the shard and the initial parameters to each
learner, so only the inner-loop compute happens remotely; the learners' wall-clock arrival
times are measured by the syncer, not simulated.

Modes::

  # on the syncer host
  PYTHONPATH=src python scripts/run_experiment_E8_pilot.py syncer --port 8765 --learners 2 --k 1 \
      --merge rda --grace adaptive --rounds 12 --out artifacts/experiments/E8_pilot_report.json
  # on each learner host (remote host reaches the syncer through `ssh -R 8765:localhost:8765`)
  python run_experiment_E8_pilot.py learner --syncer http://localhost:8765 --learner-id 1 --host-label gcp-us-west4-tpu-vm
  # single-host baseline with identical seeds, shards and hyper-parameters, no network
  PYTHONPATH=src python scripts/run_experiment_E8_pilot.py baseline --learners 2 --rounds 12 --merge rda

The learner mode imports only ``torch`` and the standard library so it can run from a copy of
this file plus ``toy_model.py`` on a bare host.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

INNER_STEPS = int(os.environ.get("HM_INNER_STEPS", "8"))  # same value on syncer, learners and baseline
BATCH = 16
INNER_LR = 0.045
OUTER_MU = 0.65
OUTER_LR = 0.7
SYNC_SECONDS = 0.0


# ----------------------------------------------------------------------------- toy model
def _load_toy_model_cls():
    try:
        from run_paper_experiments_and_figures import SyntheticLoRACausalLM, _build_synthetic_sequence_dataset
    except ModuleNotFoundError:  # learner host: standalone copy shipped next to this file
        from toy_model import SyntheticLoRACausalLM, _build_synthetic_sequence_dataset  # type: ignore
    return SyntheticLoRACausalLM, _build_synthetic_sequence_dataset


def tensors_to_b64(d: Dict[str, torch.Tensor]) -> str:
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu().contiguous() for k, v in d.items()}, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def b64_to_tensors(s: str) -> Dict[str, torch.Tensor]:
    # The bytes come from another machine over HTTP. torch.load unpickles by default on torch < 2.6 (a peer could run code here); weights_only=True accepts tensors and plain containers only.
    return torch.load(io.BytesIO(base64.b64decode(s)), map_location="cpu", weights_only=True)


def inner_train(model_cls: Any, start_params: Dict[str, torch.Tensor], shard: Tuple[torch.Tensor, torch.Tensor], round_idx: int, seed: int) -> Dict[str, torch.Tensor]:
    init = model_cls(vocab_size=64, d_model=64, rank_r=8, seed=seed)
    w = copy.deepcopy(init)
    with torch.no_grad():
        for name, p in w.named_parameters():
            p.copy_(start_params[name])
    wx, wy = shard
    for s in range(INNER_STEPS):
        idx = ((round_idx * INNER_STEPS + s) * BATCH) % wx.shape[0]
        bx, by = wx[idx : idx + BATCH], wy[idx : idx + BATCH]
        w.zero_grad()
        _, loss = w(bx, by)
        loss.backward()
        with torch.no_grad():
            for p in w.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-INNER_LR)
    with torch.no_grad():
        return {name: (start_params[name] - p.detach()).clone() for name, p in w.named_parameters()}


def flat(d: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1) for v in d.values()])


def unflat(vec: torch.Tensor, like: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out, pos = {}, 0
    for k, v in like.items():
        out[k] = vec[pos : pos + v.numel()].reshape(v.shape).clone()
        pos += v.numel()
    return out


def percentile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


# ----------------------------------------------------------------------------- syncer
class SyncerState:
    def __init__(self, args: argparse.Namespace) -> None:
        from hivemind.swarm.decoupled_diloco import AdaptiveGraceWindowController, LearnerArrival, direct_weighted_average, radial_directional_average, select_quorum

        self.select_quorum, self.LearnerArrival = select_quorum, LearnerArrival
        self.merge_fn = radial_directional_average if args.merge == "rda" else direct_weighted_average
        self.model_cls, build = _load_toy_model_cls()
        self.seed = args.seed
        self.shards, (self.val_x, self.val_y) = build(vocab_size=64, n_train_per_worker=64, n_workers=args.learners, n_val=128, seed=2026 + args.seed)
        init = self.model_cls(vocab_size=64, d_model=64, rank_r=8, seed=2026 + args.seed)
        self.global_params = {k: v.detach().clone() for k, v in init.named_parameters()}
        self.velocity = torch.zeros(flat(self.global_params).numel())
        self.args = args
        self.round = 0
        self.round_open_t = time.time()
        self.done = False
        self.lock = threading.Lock()
        self.inbox: Dict[str, Dict[str, Any]] = {}  # learner_id -> delta payload for current round
        self.pending: Dict[str, torch.Tensor] = {}  # deferred deltas (staleness 1)
        self.ctrl = AdaptiveGraceWindowController(overlap_steps_tau=2, safety_margin_gamma=0.5, ema_beta=0.5, sync_duration_s=SYNC_SECONDS) if args.grace == "adaptive" else None
        self.rounds_log: List[Dict[str, Any]] = []
        self.losses = [self.eval_loss()]
        self.hosts: Dict[str, str] = {}

    def eval_loss(self) -> float:
        m = self.model_cls(vocab_size=64, d_model=64, rank_r=8, seed=2026 + self.seed)
        with torch.no_grad():
            for name, p in m.named_parameters():
                p.copy_(self.global_params[name])
            _, loss = m(self.val_x, self.val_y)
        return float(loss.item())

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {"round": self.round, "done": self.done, "params": tensors_to_b64(self.global_params), "rounds_total": self.args.rounds}

    def shard_payload(self, learner_id: int) -> Dict[str, Any]:
        x, y = self.shards[learner_id]
        return {"x": tensors_to_b64({"x": x}), "y": tensors_to_b64({"y": y}), "seed": 2026 + self.seed}

    def receive(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = time.time()
        with self.lock:
            lid = str(payload["learner_id"])
            self.hosts[lid] = payload.get("host_label", "?")
            if int(payload["round"]) != self.round:
                return {"accepted": False, "reason": "stale round", "round": self.round}
            self.inbox[lid] = {"delta": flat(b64_to_tensors(payload["delta"])), "arrival_rel": now - self.round_open_t, "compute_s": float(payload["compute_seconds"]), "tokens": int(payload["tokens"]), "steps": int(payload["steps"])}
            return {"accepted": True, "round": self.round}

    def try_close_round(self) -> bool:
        """Close the round when quorum + grace are satisfied. Returns True if a round was closed."""
        with self.lock:
            if self.done:
                return False
            k = min(self.args.k, self.args.learners)
            arrived = [(lid, v["arrival_rel"]) for lid, v in self.inbox.items()]
            fresh_needed = k - len(self.pending)  # deferred deltas count toward the quorum once
            if len(arrived) < max(1, fresh_needed):
                return False
            arrived.sort(key=lambda t: t[1])
            t_quorum = arrived[max(0, fresh_needed) - 1][1] if fresh_needed > 0 else 0.0
            window = 0.0
            if self.ctrl is not None:
                comp = [v["compute_s"] for v in self.inbox.values()]
                self.ctrl.observe_step_duration(statistics.median(comp))
                self.ctrl.observe_quorum_duration(t_quorum)
                window = self.ctrl.grace_window_seconds()
            elapsed = time.time() - self.round_open_t
            if len(arrived) < self.args.learners and elapsed < t_quorum + window:
                return False  # still inside the grace window; a late learner may still make it
            arr = [self.LearnerArrival(lid, v["arrival_rel"], v["steps"], v["tokens"]) for lid, v in self.inbox.items()]
            vecs: List[torch.Tensor] = []
            admitted: List[str] = []
            deferred: List[str] = []
            if arr:
                sel = self.select_quorum(arr, min_quorum_k=max(1, min(fresh_needed, len(arr))), grace_window_s=window)
                for lid in sel.admitted_learner_ids:
                    vecs.append(self.inbox[lid]["delta"])
                    admitted.append(lid)
                for lid in sel.deferred_learner_ids:
                    self.pending[lid] = self.inbox[lid]["delta"]
                    deferred.append(lid)
            stale_ids = list(self.pending.keys())
            for lid in stale_ids:
                if lid not in admitted:
                    vecs.append(self.pending[lid])
            merged = self.merge_fn(vecs) if vecs else torch.zeros_like(self.velocity)
            self.velocity = OUTER_MU * self.velocity + merged
            step = OUTER_MU * self.velocity + merged
            self.global_params = unflat(flat(self.global_params) - OUTER_LR * step, self.global_params)
            # deferred deltas admitted now are consumed; newly deferred ones stay for the next round
            self.pending = {lid: d for lid, d in self.pending.items() if lid in deferred}
            close_t = time.time()
            self.rounds_log.append({
                "round": self.round, "latency_s": close_t - self.round_open_t, "quorum_at_s": t_quorum, "grace_window_s": window,
                "arrivals_s": {lid: v["arrival_rel"] for lid, v in self.inbox.items()}, "compute_s": {lid: v["compute_s"] for lid, v in self.inbox.items()},
                "admitted": admitted, "deferred": deferred, "stale_applied": [lid for lid in stale_ids if lid not in admitted],
            })
            self.losses.append(self.eval_loss())
            self.inbox = {}
            self.round += 1
            self.round_open_t = time.time()
            if self.round >= self.args.rounds:
                self.done = True
            return True


def run_syncer(args: argparse.Namespace) -> int:
    state = SyncerState(args)

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *a: Any) -> None:
            # keep the console for round telemetry; count requests instead of printing each one
            state.http_requests = getattr(state, "http_requests", 0) + 1

        def _json(self, obj: Dict[str, Any]) -> None:
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.startswith("/global"):
                self._json(state.snapshot())
            elif self.path.startswith("/shard/"):
                self._json(state.shard_payload(int(self.path.rsplit("/", 1)[1])))
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n))
            self._json(state.receive(payload))

    # The syncer has no authentication: it binds to ONE address, loopback by default (the SSH-reverse-tunnel launcher needs nothing else). For a private overlay pass that
    # overlay address (for example 10.77.0.4) with --bind; never 0.0.0.0.
    srv = ThreadingHTTPServer((args.bind, args.port), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    print(f"[syncer] listening on {args.bind}:{args.port}; waiting for {args.learners} learners, K={args.k}, merge={args.merge}, grace={args.grace}", flush=True)
    t0 = time.time()
    while not state.done:
        if state.try_close_round():
            r = state.rounds_log[-1]
            print(f"[syncer] round {r['round']:2d} closed in {r['latency_s']:.2f}s admitted={r['admitted']} deferred={r['deferred']} loss={state.losses[-1]:.4f}", flush=True)
        time.sleep(0.05)
        if time.time() - t0 > args.timeout:
            print("[syncer] timeout", flush=True)
            break
    time.sleep(2.0)  # let learners read `done`
    srv.shutdown()
    lat = [r["latency_s"] for r in state.rounds_log]
    report = {
        "schema_version": "hivemind-experiment-E8-pilot/v1.0",
        "experiment_id": "E8-pilot",
        "config": {"learners": args.learners, "k": args.k, "merge": args.merge, "grace": args.grace, "rounds": args.rounds, "seed": args.seed,
                   "inner_steps": INNER_STEPS, "batch": BATCH, "inner_lr": INNER_LR, "outer_lr": OUTER_LR, "outer_mu": OUTER_MU},
        "hosts": state.hosts, "syncer_host": platform.node(),
        "rounds": state.rounds_log, "loss_curve": state.losses, "final_loss": state.losses[-1],
        "round_latency_s": {"p50": percentile(lat, 0.5), "p95": percentile(lat, 0.95), "p99": percentile(lat, 0.99), "mean": statistics.fmean(lat) if lat else float("nan")},
        "rounds_completed": len(state.rounds_log), "completed": state.done, "wall_seconds": time.time() - t0,
    }
    payload = json.dumps({k: v for k, v in report.items() if k != "receipt_sha256"}, sort_keys=True).encode()
    report["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("final_loss", "round_latency_s", "rounds_completed", "completed", "receipt_sha256")}, indent=2), flush=True)
    return 0 if state.done else 1


# ----------------------------------------------------------------------------- learner
def http_json(url: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 120.0) -> Tuple[Dict[str, Any], float]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET", headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.perf_counter() - t


def run_learner(args: argparse.Namespace) -> int:
    model_cls, _ = _load_toy_model_cls()
    shard_payload, _ = http_json(f"{args.syncer}/shard/{args.learner_id}")
    shard = (b64_to_tensors(shard_payload["x"])["x"], b64_to_tensors(shard_payload["y"])["y"])
    seed = int(shard_payload["seed"])
    last_round = -1
    rtts: List[float] = []
    while True:
        snap, rtt = http_json(f"{args.syncer}/global")
        rtts.append(rtt)
        if snap["done"]:
            break
        if snap["round"] == last_round:
            time.sleep(0.1)
            continue
        last_round = snap["round"]
        params = b64_to_tensors(snap["params"])
        t = time.perf_counter()
        delta = inner_train(model_cls, params, shard, last_round, seed)
        compute_s = time.perf_counter() - t
        if args.slow_factor > 1.0:
            time.sleep(compute_s * (args.slow_factor - 1.0))
            compute_s *= args.slow_factor
        resp, _ = http_json(f"{args.syncer}/delta", {"learner_id": args.learner_id, "round": last_round, "delta": tensors_to_b64(delta), "compute_seconds": compute_s,
                                                   "tokens": INNER_STEPS * BATCH, "steps": INNER_STEPS, "host_label": args.host_label})
        print(f"[learner {args.learner_id}@{args.host_label}] round {last_round} compute {compute_s:.3f}s accepted={resp.get('accepted')} http_rtt_last={rtt*1000:.1f}ms", flush=True)
    print(json.dumps({"learner_id": args.learner_id, "host": args.host_label, "http_rtt_ms": {"p50": percentile(rtts, 0.5) * 1000, "p95": percentile(rtts, 0.95) * 1000, "n": len(rtts)}}), flush=True)
    return 0


# ----------------------------------------------------------------------------- baseline
def run_baseline(args: argparse.Namespace) -> int:
    from hivemind.swarm.decoupled_diloco import direct_weighted_average, radial_directional_average

    model_cls, build = _load_toy_model_cls()
    shards, (vx, vy) = build(vocab_size=64, n_train_per_worker=64, n_workers=args.learners, n_val=128, seed=2026 + args.seed)
    init = model_cls(vocab_size=64, d_model=64, rank_r=8, seed=2026 + args.seed)
    g = {k: v.detach().clone() for k, v in init.named_parameters()}
    vel = torch.zeros(flat(g).numel())
    merge = radial_directional_average if args.merge == "rda" else direct_weighted_average

    def ev() -> float:
        m = copy.deepcopy(init)
        with torch.no_grad():
            for n_, p in m.named_parameters():
                p.copy_(g[n_])
            _, l = m(vx, vy)
        return float(l.item())

    losses = [ev()]
    t0 = time.perf_counter()
    for r in range(args.rounds):
        vecs = [flat(inner_train(model_cls, g, shards[i], r, 2026 + args.seed)) for i in range(args.learners)]
        merged = merge(vecs)
        vel = OUTER_MU * vel + merged
        g = unflat(flat(g) - OUTER_LR * (OUTER_MU * vel + merged), g)
        losses.append(ev())
    report = {"schema_version": "hivemind-experiment-E8-pilot-baseline/v1.0", "experiment_id": "E8-pilot-baseline", "config": {"learners": args.learners, "merge": args.merge, "rounds": args.rounds, "seed": args.seed, "inner_steps": INNER_STEPS, "admission": "all, single process, no network"},
              "loss_curve": losses, "final_loss": losses[-1], "wall_seconds": time.perf_counter() - t0, "host": platform.node()}
    payload = json.dumps({k: v for k, v in report.items() if k != "receipt_sha256"}, sort_keys=True).encode()
    report["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("final_loss", "wall_seconds", "receipt_sha256")}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("syncer")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--bind", default="127.0.0.1", help="the single address to listen on (default loopback); 0.0.0.0 is refused")
    s.add_argument("--learners", type=int, default=2)
    s.add_argument("--k", type=int, default=1)
    s.add_argument("--merge", choices=["direct", "rda"], default="rda")
    s.add_argument("--grace", choices=["off", "adaptive"], default="adaptive")
    s.add_argument("--rounds", type=int, default=12)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--timeout", type=float, default=1800.0)
    s.add_argument("--out", default=str(HERE.parent / "artifacts" / "experiments" / "E8_pilot_report.json"))
    l = sub.add_parser("learner")
    l.add_argument("--syncer", default="http://localhost:8765")
    l.add_argument("--learner-id", type=int, required=True)
    l.add_argument("--host-label", default=platform.node())
    l.add_argument("--slow-factor", type=float, default=1.0, help="artificially slow this learner (straggler); 1.0 = none")
    b = sub.add_parser("baseline")
    b.add_argument("--learners", type=int, default=2)
    b.add_argument("--merge", choices=["direct", "rda"], default="rda")
    b.add_argument("--rounds", type=int, default=12)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--out", default=str(HERE.parent / "artifacts" / "experiments" / "E8_pilot_baseline_report.json"))
    args = ap.parse_args()
    if args.mode == "syncer" and args.bind.strip() in ("", "0.0.0.0", "::", "[::]"):
        ap.error("--bind must be one specific address (loopback or a private overlay address): the syncer has no authentication")
    if args.mode == "syncer":
        return run_syncer(args)
    if args.mode == "learner":
        return run_learner(args)
    return run_baseline(args)


if __name__ == "__main__":
    raise SystemExit(main())
