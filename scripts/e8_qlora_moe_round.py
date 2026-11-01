#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""E8 stage 2: DiLoCo-style rounds of QLoRA on a tiny Mixture-of-Experts model, across machines.

A tiny Granite-style MoE (random weights, 4 experts, top-2 routing) is the base. Its linear layers can be NF4-quantised (bitsandbytes) and LoRA adapters sit on the attention
projections: that is the real QLoRA code path at toy scale. Experts are not adapted (in the Hugging Face Granite MoE they are fused parameters, not ``nn.Linear``).
Each round, every learner starts from the global LoRA state, trains ``--inner-steps`` steps on its own non-IID shard, and sends back only the LoRA DELTA; the syncer takes the
token-weighted mean and applies an outer step with momentum. Learners wait for each other (K = M): this is the protocol-validation cell, not the straggler study.

  python scripts/e8_qlora_moe_round.py syncer   --bind 10.77.0.4 --learners 2 --rounds 8 --out report.json
  python scripts/e8_qlora_moe_round.py learner  --syncer http://10.77.0.4:8765 --learner-id 0 --quant nf4 --device cuda
  python scripts/e8_qlora_moe_round.py baseline --learners 2 --rounds 8 --out baseline.json          # the same algorithm in one process

Controls: ``--shuffle-labels`` (on a learner or the baseline) trains on shuffled targets: the global loss must NOT improve. ``baseline`` is the single-host run to compare with.
What this is not: Granite-scale, a Byzantine test (with 2 learners Multi-Krum keeps everyone; that needs 5 or more), or a hardware-heterogeneity result.
The syncer has no authentication: it binds ONE address (loopback by default, ``0.0.0.0`` refused). Payloads are loaded with ``weights_only=True``.
Before any GPU work the learner checks the free VRAM against ``--min-free-mib`` (the project keeps 4,300 MiB free on the T4) and refuses to start if it would go below.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

VOCAB, SEQ_LEN = 64, 24
MAX_BODY = 64 * 1024 * 1024
LORA_R, LORA_ALPHA = 4, 8
LORA_TARGETS = ["q_proj", "v_proj"]


class GuardError(RuntimeError):
    """Raised when a hardware or safety guard refuses execution."""


# ------------------------------------------------------------------------------------------------------------------------------ transport


def tensors_to_b64(d: Dict[str, torch.Tensor]) -> str:
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu().contiguous() for k, v in d.items()}, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def b64_to_tensors(s: str) -> Dict[str, torch.Tensor]:
    """The bytes come from another machine: weights_only=True accepts tensors and plain containers only (no code)."""
    return torch.load(io.BytesIO(base64.b64decode(s)), map_location="cpu", weights_only=True)


# ------------------------------------------------------------------------------------------------------------------------------ guards


def free_vram_mib() -> Optional[float]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=False)
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        _ = None
    return None


def vram_guard(min_free_mib: float, need_mib: float, device: str, free_fn: Callable[[], Optional[float]] = free_vram_mib) -> Optional[float]:
    """Refuse GPU work that would leave less than ``min_free_mib`` free. An unreadable GPU is a refusal, never a pass."""
    if device != "cuda":
        return None
    free = free_fn()
    if free is None:
        raise GuardError("cannot read the free VRAM (nvidia-smi); refusing to start GPU work")
    if free - need_mib < min_free_mib:
        raise GuardError(f"{free:.0f} MiB free, this run needs about {need_mib:.0f} MiB, and {min_free_mib:.0f} MiB must stay free: refusing")
    return free


# ------------------------------------------------------------------------------------------------------------------------------ data


def make_shard(learner_id: int, n_seq: int, seed: int, vocab: int = VOCAB, seq_len: int = SEQ_LEN) -> torch.Tensor:
    """Non-IID shard: each learner's sequences follow its own affine transition rule with 10 % noise, so learners genuinely disagree about the data."""
    g = torch.Generator().manual_seed(10_000 * (seed + 1) + learner_id)
    a = 3 + 2 * learner_id  # odd, so the rule is a permutation of the vocabulary for an even vocab
    b = 7 + 11 * learner_id
    x = torch.empty(n_seq, seq_len, dtype=torch.long)
    x[:, 0] = torch.randint(0, vocab, (n_seq,), generator=g)
    for t in range(1, seq_len):
        nxt = (a * x[:, t - 1] + b) % vocab
        noise = torch.randint(0, vocab, (n_seq,), generator=g)
        x[:, t] = torch.where(torch.rand(n_seq, generator=g) < 0.9, nxt, noise)
    return x


def make_eval(learners: int, seed: int, n_per: int = 32) -> torch.Tensor:
    """Held-out mixture of every learner's distribution, from a seed no shard uses."""
    return torch.cat([make_shard(i, n_per, seed + 777) for i in range(learners)], dim=0)


# ------------------------------------------------------------------------------------------------------------------------------ model


def tiny_moe_config() -> Any:
    from transformers import GraniteMoeConfig

    return GraniteMoeConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, num_local_experts=4,
                            num_experts_per_tok=2, max_position_embeddings=SEQ_LEN + 8, pad_token_id=0, bos_token_id=0, eos_token_id=0, tie_word_embeddings=False)


def build_base_state(seed: int) -> Dict[str, torch.Tensor]:
    from transformers import GraniteMoeForCausalLM

    torch.manual_seed(seed)
    return {k: v.detach().clone().float() for k, v in GraniteMoeForCausalLM(tiny_moe_config()).state_dict().items()}


def load_model(base_state: Dict[str, torch.Tensor], quant: str, device: str, lora_seed: int) -> Any:
    from peft import LoraConfig, get_peft_model
    from transformers import GraniteMoeForCausalLM

    cfg = tiny_moe_config()
    if quant == "nf4":
        from transformers import BitsAndBytesConfig

        plain = GraniteMoeForCausalLM(cfg)
        plain.load_state_dict(base_state)
        with tempfile.TemporaryDirectory() as td:
            plain.save_pretrained(td)
            model = GraniteMoeForCausalLM.from_pretrained(
                td, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16), device_map={"": 0})
    else:
        model = GraniteMoeForCausalLM(cfg)
        model.load_state_dict(base_state)
        model.to(device)
    torch.manual_seed(lora_seed)
    # lm_head is trained in full (a standard QLoRA choice, not quantised, 4,096 parameters here): with frozen random embeddings and a frozen random head the toy task is barely learnable.
    return get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS, lora_dropout=0.0, task_type="CAUSAL_LM", modules_to_save=["lm_head"]))


def lora_state(model: Any) -> Dict[str, torch.Tensor]:
    from peft import get_peft_model_state_dict

    return {k: v.detach().cpu().float().clone() for k, v in get_peft_model_state_dict(model).items()}


def set_lora_state(model: Any, state: Dict[str, torch.Tensor]) -> None:
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(model, {k: v.clone() for k, v in state.items()})


@torch.no_grad()
def eval_loss(model: Any, batch: torch.Tensor, device: str) -> float:
    model.eval()
    x = batch.to(device)
    return float(model(input_ids=x, labels=x).loss.float().cpu())


def train_inner(model: Any, shard: torch.Tensor, steps: int, lr: float, batch: int, seed: int, device: str, shuffle_labels: bool = False) -> int:
    """``steps`` AdamW steps on the LoRA parameters only. Returns the number of tokens trained on."""
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    tokens = 0
    for _ in range(steps):
        idx = torch.randint(0, shard.shape[0], (batch,), generator=g)
        x = shard[idx].to(device)
        y = x
        if shuffle_labels:
            y = x.reshape(-1)[torch.randperm(x.numel(), generator=g).to(x.device)].reshape(x.shape)  # targets shuffled across every token of the batch
        loss = model(input_ids=x, labels=y).loss
        opt.zero_grad(set_to_none=True)
        loss.float().backward()
        opt.step()
        tokens += int(x.numel())
    return tokens


# ------------------------------------------------------------------------------------------------------------------------------ outer step


def outer_step(global_state: Dict[str, torch.Tensor], deltas: Sequence[Dict[str, torch.Tensor]], weights: Sequence[float], velocity: Optional[Dict[str, torch.Tensor]],
               eta: float, mu: float) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Token-weighted mean of the learner deltas, then ``v = mu v + mean`` and ``global += eta v``."""
    if not deltas or len(deltas) != len(weights) or sum(weights) <= 0:
        raise ValueError("need one positive weight per delta")
    total = float(sum(weights))
    new_v: Dict[str, torch.Tensor] = {}
    new_g: Dict[str, torch.Tensor] = {}
    for k, g0 in global_state.items():
        mean = sum(float(w) * d[k] for d, w in zip(deltas, weights)) / total
        v = mu * (velocity[k] if velocity else torch.zeros_like(g0)) + mean
        new_v[k] = v
        new_g[k] = g0 + eta * v
    return new_g, new_v


def l2(d: Dict[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float((v.float() ** 2).sum()) for v in d.values()))


# ------------------------------------------------------------------------------------------------------------------------------ one learner's round


def learner_round(model: Any, global_state: Dict[str, torch.Tensor], shard: torch.Tensor, evalset: torch.Tensor, *, steps: int, lr: float, batch: int, seed: int, rnd: int,
                  learner_id: int, device: str, shuffle_labels: bool) -> Dict[str, Any]:
    set_lora_state(model, global_state)
    before = eval_loss(model, evalset, device)
    start = lora_state(model)
    tokens = train_inner(model, shard, steps, lr, batch, seed * 1_000_003 + rnd * 101 + learner_id, device, shuffle_labels)
    end = lora_state(model)
    delta = {k: end[k] - start[k] for k in start}
    return {"delta": delta, "tokens": tokens, "eval_loss_global": before, "delta_l2": l2(delta)}


def initial_lora(base_state: Dict[str, torch.Tensor], seed: int) -> Dict[str, torch.Tensor]:
    return lora_state(load_model(base_state, "none", "cpu", seed))


# ------------------------------------------------------------------------------------------------------------------------------ syncer


class SyncerState:
    def __init__(self, learners: int, rounds: int, seed: int, eta: float, mu: float) -> None:
        self.m, self.rounds, self.seed, self.eta, self.mu = learners, rounds, seed, eta, mu
        self.base = build_base_state(seed)
        self.global_state = initial_lora(self.base, seed)
        self.velocity: Optional[Dict[str, torch.Tensor]] = None
        self.round = 0
        self.arrivals: Dict[int, Dict[str, Any]] = {}
        self.hosts: Dict[int, str] = {}
        self.log: List[Dict[str, Any]] = []
        self.final_eval: Dict[int, float] = {}
        self.t_round = time.time()
        self.t0 = time.time()
        self.lock = threading.Lock()
        self.done = threading.Event()

    def snapshot(self, want_base: bool) -> Dict[str, Any]:
        with self.lock:
            out: Dict[str, Any] = {"round": self.round, "rounds": self.rounds, "learners": self.m, "seed": self.seed, "global": tensors_to_b64(self.global_state), "done": self.done.is_set()}
            if want_base:
                out["base"] = tensors_to_b64(self.base)
            return out

    def receive(self, p: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            lid, rnd = int(p["learner_id"]), int(p["round"])
            if rnd != self.round:
                return {"accepted": False, "round": self.round}
            self.hosts[lid] = str(p.get("host", "?"))
            if rnd == self.rounds:  # evaluation-only round at the end
                self.final_eval[lid] = float(p["eval_loss_global"])
                if len(self.final_eval) == self.m:
                    self.done.set()
                return {"accepted": True, "round": self.round}
            if lid in self.arrivals:
                return {"accepted": True, "round": self.round}
            self.arrivals[lid] = {"delta": b64_to_tensors(p["delta"]), "tokens": int(p["tokens"]), "eval": float(p["eval_loss_global"]), "t": time.time() - self.t_round,
                                  "delta_l2": float(p["delta_l2"]), "vram_peak_mib": p.get("vram_peak_mib")}
            if len(self.arrivals) == self.m:
                ids = sorted(self.arrivals)
                self.global_state, self.velocity = outer_step(self.global_state, [self.arrivals[i]["delta"] for i in ids], [self.arrivals[i]["tokens"] for i in ids],
                                                              self.velocity, self.eta, self.mu)
                self.log.append({"round": rnd, "latency_s": time.time() - self.t_round, "eval_loss_global": {str(i): self.arrivals[i]["eval"] for i in ids},
                                 "delta_l2": {str(i): self.arrivals[i]["delta_l2"] for i in ids}, "arrivals_s": {str(i): self.arrivals[i]["t"] for i in ids},
                                 "vram_peak_mib": {str(i): self.arrivals[i]["vram_peak_mib"] for i in ids}})
                self.arrivals = {}
                self.round += 1
                self.t_round = time.time()
            return {"accepted": True, "round": self.round}


def run_syncer(args: argparse.Namespace) -> int:
    st = SyncerState(args.learners, args.rounds, args.seed, args.outer_lr, args.outer_mu)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:  # noqa: ANN401
            _ = a

        def _json(self, obj: Any, code: int = 200) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/state"):
                self._json(st.snapshot("base=1" in self.path))
            else:
                self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0 or n > MAX_BODY:
                self.send_error(413)
                return
            try:
                self._json(st.receive(json.loads(self.rfile.read(n))))
            except Exception as exc:  # noqa: BLE001
                self._json({"accepted": False, "error": type(exc).__name__}, 400)

    srv = ThreadingHTTPServer((args.bind, args.port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[syncer] {args.bind}:{srv.server_address[1]} learners={args.learners} rounds={args.rounds} seed={args.seed}", flush=True)
    ok = st.done.wait(args.timeout)
    time.sleep(1.5)  # let the learners read the final state
    srv.shutdown()
    write_report(args.out, "syncer", st, args, ok)
    return 0 if ok else 2


def write_report(path: str, mode: str, st: SyncerState, args: argparse.Namespace, ok: bool) -> None:
    init_eval = st.log[0]["eval_loss_global"] if st.log else {}
    final = st.final_eval
    rep = {"schema_version": 1, "experiment_id": "E8-stage2-tiny-moe-qlora", "mode": mode, "completed": ok, "syncer_host": platform.node(),
           "hosts": {str(k): v for k, v in sorted(st.hosts.items())},
           "config": {"learners": st.m, "rounds": st.rounds, "seed": st.seed, "inner_steps": args.inner_steps, "inner_lr": args.inner_lr, "batch": args.batch,
                      "outer_lr": st.eta, "outer_mu": st.mu, "model": "GraniteMoE tiny (hidden 64, 2 layers, 4 experts, top-2), LoRA r=4 on q_proj,v_proj"},
           "rounds": st.log, "initial_eval_loss": init_eval, "final_eval_loss": {str(k): v for k, v in sorted(final.items())},
           "final_eval_loss_mean": (sum(final.values()) / len(final)) if final else None, "wall_seconds": time.time() - st.t0}
    rep["receipt_sha256"] = hashlib.sha256(json.dumps(rep, sort_keys=True).encode()).hexdigest()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rep, indent=2))
    print(f"[{mode}] wrote {path}; final eval loss mean = {rep['final_eval_loss_mean']}", flush=True)


# ------------------------------------------------------------------------------------------------------------------------------ learner


def http_json(url: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 60.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="GET" if payload is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - the URL is the operator's own syncer
        return json.loads(r.read().decode())


def run_learner(args: argparse.Namespace) -> int:
    device = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    free = vram_guard(args.min_free_mib, args.need_mib, device)
    connect_timeout = float(getattr(args, "connect_timeout", 300.0))
    deadline = time.time() + connect_timeout  # a remote learner often starts before the syncer is listening: retry instead of dying
    while True:
        try:
            st = http_json(args.syncer.rstrip("/") + "/state?base=1")
            break
        except OSError as exc:
            if time.time() > deadline:
                print(f"refused: syncer {args.syncer} not reachable after {connect_timeout:.0f}s ({exc})", file=sys.stderr)
                return 4
            time.sleep(2.0)
    base = b64_to_tensors(st["base"])
    seed, m, rounds = int(st["seed"]), int(st["learners"]), int(st["rounds"])
    model = load_model(base, args.quant, device, seed)
    shard, evalset = make_shard(args.learner_id, args.shard_size, seed), make_eval(m, seed)
    label = args.host_label or platform.node()
    print(f"[learner {args.learner_id}] {label} device={device} quant={args.quant} free_vram_before={free}", flush=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for rnd in range(rounds + 1):
        while True:
            cur = http_json(args.syncer.rstrip("/") + "/state")
            if cur["round"] >= rnd or cur.get("done"):
                break
            time.sleep(0.4)
        cur = http_json(args.syncer.rstrip("/") + "/state")
        g = b64_to_tensors(cur["global"])
        if rnd == rounds:
            set_lora_state(model, g)
            http_json(args.syncer.rstrip("/") + "/delta", {"learner_id": args.learner_id, "round": rnd, "host": label, "eval_loss_global": eval_loss(model, evalset, device)})
            break
        out = learner_round(model, g, shard, evalset, steps=args.inner_steps, lr=args.inner_lr, batch=args.batch, seed=seed, rnd=rnd, learner_id=args.learner_id, device=device,
                            shuffle_labels=args.shuffle_labels)
        peak = float(torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None
        http_json(args.syncer.rstrip("/") + "/delta", {"learner_id": args.learner_id, "round": rnd, "host": label, "delta": tensors_to_b64(out["delta"]), "tokens": out["tokens"],
                                                       "eval_loss_global": out["eval_loss_global"], "delta_l2": out["delta_l2"], "vram_peak_mib": peak})
        print(f"[learner {args.learner_id}] round {rnd}: global eval {out['eval_loss_global']:.4f}, |delta| {out['delta_l2']:.4f}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ single-process baseline


def run_baseline(args: argparse.Namespace) -> int:
    device = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    vram_guard(args.min_free_mib, args.need_mib, device)
    st = SyncerState(args.learners, args.rounds, args.seed, args.outer_lr, args.outer_mu)
    model = load_model(st.base, args.quant, device, args.seed)
    shards = [make_shard(i, args.shard_size, args.seed) for i in range(args.learners)]
    evalset = make_eval(args.learners, args.seed)
    for rnd in range(args.rounds):
        for i in range(args.learners):
            out = learner_round(model, st.global_state, shards[i], evalset, steps=args.inner_steps, lr=args.inner_lr, batch=args.batch, seed=args.seed, rnd=rnd, learner_id=i,
                                device=device, shuffle_labels=args.shuffle_labels)
            st.hosts[i] = platform.node()
            st.receive({"learner_id": i, "round": rnd, "host": platform.node(), "delta": tensors_to_b64(out["delta"]), "tokens": out["tokens"], "eval_loss_global": out["eval_loss_global"],
                        "delta_l2": out["delta_l2"]})
    set_lora_state(model, st.global_state)
    for i in range(args.learners):
        st.receive({"learner_id": i, "round": args.rounds, "host": platform.node(), "eval_loss_global": eval_loss(model, evalset, device)})
    write_report(args.out, "baseline", st, args, st.done.is_set())
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ cli


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--inner-steps", type=int, default=30)
        p.add_argument("--inner-lr", type=float, default=2e-3)
        p.add_argument("--batch", type=int, default=16)
        p.add_argument("--outer-lr", type=float, default=0.7)
        p.add_argument("--outer-mu", type=float, default=0.65)
        p.add_argument("--seed", type=int, default=1)

    s = sub.add_parser("syncer")
    s.add_argument("--bind", default="127.0.0.1", help="the single address to listen on; 0.0.0.0 is refused")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--learners", type=int, default=2)
    s.add_argument("--rounds", type=int, default=8)
    s.add_argument("--timeout", type=float, default=1800.0)
    s.add_argument("--out", required=True)
    common(s)
    lr = sub.add_parser("learner")
    lr.add_argument("--syncer", required=True)
    lr.add_argument("--learner-id", type=int, required=True)
    lr.add_argument("--host-label", default="")
    lr.add_argument("--quant", choices=["none", "nf4"], default="none")
    lr.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    lr.add_argument("--shard-size", type=int, default=512)
    lr.add_argument("--shuffle-labels", action="store_true", help="negative control: train on shuffled targets")
    lr.add_argument("--min-free-mib", type=float, default=4300.0)
    lr.add_argument("--need-mib", type=float, default=600.0, help="estimated VRAM this learner needs, checked against --min-free-mib")
    lr.add_argument("--connect-timeout", type=float, default=300.0, help="seconds to keep retrying the first connection to the syncer")
    common(lr)
    b = sub.add_parser("baseline")
    b.add_argument("--learners", type=int, default=2)
    b.add_argument("--rounds", type=int, default=8)
    b.add_argument("--quant", choices=["none", "nf4"], default="none")
    b.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    b.add_argument("--shard-size", type=int, default=512)
    b.add_argument("--shuffle-labels", action="store_true")
    b.add_argument("--min-free-mib", type=float, default=4300.0)
    b.add_argument("--need-mib", type=float, default=600.0)
    b.add_argument("--out", required=True)
    common(b)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.mode == "syncer" and args.bind.strip() in ("", "0.0.0.0", "::", "[::]"):
        ap.error("--bind must be one specific address (loopback or a private overlay address): the syncer has no authentication")
    try:
        return {"syncer": run_syncer, "learner": run_learner, "baseline": run_baseline}[args.mode](args)
    except GuardError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
