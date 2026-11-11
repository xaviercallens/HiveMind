#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""E29 (IP-JEPA-03 & IP-JEPA-04): LLM-JEPA inside Stage-2 NF4 QLoRA MoE DiLoCo + B3 Held-Out Gate.

Integrates the LLM-JEPA auxiliary objective (Huang, LeCun & Balestriero, arXiv:2509.14252) as an opt-in
inner-loop term on HiveMind's real Stage-2 ``GraniteMoeForCausalLM`` (4 experts, top-2 routing, ``bitsandbytes``
4-bit ``NF4`` base weights + ``peft`` LoRA ``r=4`` on ``q_proj, v_proj`` + ``lm_head``) trained across ``K=2``
non-IID learners with DiLoCo outer momentum.

Each sequence has two paired views (length 12 + 12 = 24, token 63 = ``<pred>``):
  * View 1 (prompt / source domain, non-IID per learner)
  * View 2 (target / code view: bijective reversed affine transform of View 1)

Arms compared across 5 seeds ({1, 2, 3, 4, 5}):
  * ``qlora_ntp``:         plain NF4 QLoRA DiLoCo (lambda = 0)
  * ``qlora_jepa_drop75``: volunteer-budget NF4 QLoRA + LLM-JEPA (alpha = 0.75, lambda = 2.0, k = 1; 1.25x nominal)
  * ``qlora_jepa_full``:   full NF4 QLoRA + LLM-JEPA (alpha = 0.0, lambda = 0.5, k = 1)
  * ``qlora_jepa_shuf``:   shuffled-view negative control (alpha = 0.0, lambda = 0.5, k = 1, mismatched View 2)

Promotion rule (IP-JEPA-03 / B3 gate):
  An adapter arm is ``servable`` iff ``evaluate_jepa_b3_gate(ntp_nlls, jepa_nlls)`` reports ``mean_nll_gain_nats > 0``
  AND ``bootstrap95[0] > 0``. The view-gap singular value ``view_gap_sv1`` is logged every round strictly as a
  non-gating diagnostic (IP-JEPA-04).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import e8_qlora_moe_round as e8q  # noqa: E402
from hivemind.swarm.llm_jepa import (  # noqa: E402
    JepaConfig,
    embedding_distance,
    evaluate_jepa_b3_gate,
    llm_jepa_loss,
    view_gap_singular_values,
)

PRED_TOK = 63
VIEW_LEN = 12  # 12 + 12 = 24 == e8q.SEQ_LEN

ARMS: dict[str, dict[str, Any]] = {
    "qlora_ntp": {"lam": 0.0, "k_pred": 1, "loss_dropout": 0.0, "shuffle_views": False},
    "qlora_jepa_drop75": {"lam": 2.0, "k_pred": 1, "loss_dropout": 0.75, "shuffle_views": False},
    "qlora_jepa_full": {"lam": 0.5, "k_pred": 1, "loss_dropout": 0.0, "shuffle_views": False},
    "qlora_jepa_shuf": {"lam": 0.5, "k_pred": 1, "loss_dropout": 0.0, "shuffle_views": True},
}


def make_two_view_shard(learner_id: int, n_seq: int, seed: int, eval_clean: bool = False) -> torch.Tensor:
    """Generate non-IID two-view sequences ``[View1 (12 tok) | View2 (12 tok)]`` in ``1..62`` (0=pad, 63=pred)."""
    g = torch.Generator().manual_seed(20_000 * (seed + 1) + 97 * learner_id + (7777 if eval_clean else 0))
    v1 = torch.empty(n_seq, VIEW_LEN, dtype=torch.long)
    # Non-IID source distributions: learner 0 prefers lower half, learner 1 prefers upper half + distinct transition
    low, high = (1, 32) if learner_id % 2 == 0 else (31, 63)
    v1[:, 0] = torch.randint(low, high, (n_seq,), generator=g)
    step_a = 3 + 4 * learner_id
    step_b = 5 + 7 * learner_id
    for t in range(1, VIEW_LEN):
        nxt = ((step_a * v1[:, t - 1] + step_b) % 62) + 1
        noise = torch.randint(low, high, (n_seq,), generator=g)
        v1[:, t] = torch.where(torch.rand(n_seq, generator=g) < 0.85, nxt, noise)
    # Shared ground-truth cross-view mapping: reversed bijective affine map View1 -> View2
    rev = torch.flip(v1, dims=[1])
    v2 = ((5 * rev + 13) % 62) + 1
    if not eval_clean:
        flip_noise = torch.randint(1, 63, (n_seq, VIEW_LEN), generator=g)
        v2 = torch.where(torch.rand(n_seq, VIEW_LEN, generator=g) < 0.95, v2, flip_noise)
    return torch.cat([v1, v2], dim=1)


def make_two_view_eval(learners: int, seed: int, n_per: int = 64) -> torch.Tensor:
    return torch.cat([make_two_view_shard(i, n_per, seed + 900, eval_clean=True) for i in range(learners)], dim=0)


def target_labels(x: torch.Tensor) -> torch.Tensor:
    """Supervise only the target view ``x[:, VIEW_LEN:]`` (standard Text -> Code conditional NLL)."""
    y = x.clone()
    y[:, :VIEW_LEN] = -100
    return y


def _supports_4d_mask(model: Any, device: str) -> bool:
    try:
        x = torch.ones((1, 4), dtype=torch.long, device=device)
        m = torch.zeros((1, 1, 4, 4), dtype=torch.float16 if device == "cuda" else torch.float32, device=device)
        with torch.no_grad():
            model(input_ids=x, attention_mask=m, output_hidden_states=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def two_pass_jepa_loss(
    lm_loss: torch.Tensor,
    model: Any,
    v1: torch.Tensor,
    v2: torch.Tensor,
    cfg: JepaConfig,
    rng: random.Random,
) -> tuple[torch.Tensor, float]:
    """Fallback when a backbone enforces 2D masks: run View1+[PRED]*k and View2 independently (exact reference)."""
    if cfg.lam == 0.0 or (cfg.loss_dropout > 0.0 and rng.random() < cfg.loss_dropout):
        return lm_loss, 0.0
    device = v1.device
    pred_pad = torch.full((v1.shape[0], cfg.k_pred), PRED_TOK, dtype=torch.long, device=device)
    v1p = torch.cat([v1, pred_pad], dim=1) if cfg.k_pred > 0 else v1
    h1 = model(input_ids=v1p, output_hidden_states=True).hidden_states[-1][:, -1, :].float()
    h2 = model(input_ids=v2, output_hidden_states=True).hidden_states[-1][:, -1, :].float()
    d = embedding_distance(h1, h2, cfg.metric)
    return lm_loss + cfg.lam * d, 1.0


@torch.no_grad()
def evaluate_moe_views(model: Any, evalset: torch.Tensor, k_pred: int, device: str) -> dict[str, float]:
    model.train(False)
    x = evalset.to(device)
    y = target_labels(x)
    out = model(input_ids=x, labels=y)
    nll = float(out.loss.float().cpu())
    # Next-token accuracy on View 2 positions (logits at t-1 predict token t for t in VIEW_LEN..23)
    preds = out.logits[:, VIEW_LEN - 1 : -1, :].argmax(dim=-1)
    acc = float((preds == x[:, VIEW_LEN:]).float().mean().cpu())
    v1, v2 = x[:, :VIEW_LEN], x[:, VIEW_LEN:]
    pred_pad = torch.full((v1.shape[0], k_pred), PRED_TOK, dtype=torch.long, device=device)
    v1p = torch.cat([v1, pred_pad], dim=1) if k_pred > 0 else v1
    e1 = model(input_ids=v1p, output_hidden_states=True).hidden_states[-1][:, -1, :].float().cpu()
    e2 = model(input_ids=v2, output_hidden_states=True).hidden_states[-1][:, -1, :].float().cpu()
    sv1 = view_gap_singular_values(e1, e2, top=1)[0]
    model.train(True)
    return {"target_nll": round(nll, 6), "target_token_acc": round(acc, 4), "view_gap_sv1": round(sv1, 4)}


def train_inner_jepa(
    model: Any,
    shard: torch.Tensor,
    steps: int,
    lr: float,
    batch: int,
    seed: int,
    device: str,
    cfg: JepaConfig,
    shuffle_views: bool,
    use_4d: bool,
) -> tuple[int, int]:
    model.train(True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    drop_rng = random.Random(seed + 17)
    tokens, jepa_active = 0, 0
    for _ in range(steps):
        idx = torch.randint(0, shard.shape[0], (batch,), generator=g)
        x = shard[idx].to(device)
        y = target_labels(x)
        lm = model(input_ids=x, labels=y).loss.float()
        v1 = x[:, :VIEW_LEN]
        v2 = x[:, VIEW_LEN:]
        if shuffle_views and v2.shape[0] > 1:
            v2 = torch.roll(v2, shifts=1, dims=0)
        if use_4d:
            loss, log = llm_jepa_loss(
                lm, model, v1.cpu().tolist(), v2.cpu().tolist(), PRED_TOK, cfg, rng=drop_rng, pad_id=0
            )
            jepa_active += int(log["jepa_active"])
        else:
            loss, act = two_pass_jepa_loss(lm, model, v1, v2, cfg, drop_rng)
            jepa_active += int(act)
        opt.zero_grad(set_to_none=True)
        loss.float().backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        tokens += int(x.numel())
    return tokens, jepa_active


def run_seed_arm(arm: str, seed: int, args, device: str) -> dict[str, Any]:
    spec = ARMS[arm]
    cfg = JepaConfig(lam=spec["lam"], k_pred=spec["k_pred"], loss_dropout=spec["loss_dropout"])
    base = e8q.build_base_state(seed)
    global_state = e8q.initial_lora(base, seed)
    velocity = None
    model = e8q.load_model(base, args.quant, device, seed)
    use_4d = _supports_4d_mask(model, device)
    shards = [make_two_view_shard(i, args.shard_size, seed) for i in range(args.learners)]
    evalset = make_two_view_eval(args.learners, seed)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    e8q.set_lora_state(model, global_state)
    init_metrics = evaluate_moe_views(model, evalset, cfg.k_pred, device)
    round_log = []
    total_jepa_steps = 0

    for rnd in range(args.rounds):
        deltas, weights = [], []
        for lid in range(args.learners):
            e8q.set_lora_state(model, global_state)
            start = e8q.lora_state(model)
            toks, j_act = train_inner_jepa(
                model,
                shards[lid],
                args.inner_steps,
                args.inner_lr,
                args.batch,
                seed * 1_000_003 + rnd * 101 + lid,
                device,
                cfg,
                spec["shuffle_views"],
                use_4d,
            )
            end = e8q.lora_state(model)
            deltas.append({k: end[k] - start[k] for k in start})
            weights.append(float(toks))
            total_jepa_steps += j_act
        global_state, velocity = e8q.outer_step(global_state, deltas, weights, velocity, args.outer_lr, args.outer_mu)
        e8q.set_lora_state(model, global_state)
        m = evaluate_moe_views(model, evalset, cfg.k_pred, device)
        round_log.append({"round": rnd + 1, **m})

    if device == "cuda":
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0
    peak_mib = round(float(torch.cuda.max_memory_allocated() / 2**20), 2) if device == "cuda" else None
    final_m = round_log[-1]
    return {
        "arm": arm,
        "seed": seed,
        "quant": args.quant,
        "use_4d_block_causal_mask": use_4d,
        "jepa_active_steps": total_jepa_steps,
        "wall_seconds": round(wall_s, 3),
        "vram_peak_mib": peak_mib,
        "initial": init_metrics,
        "final": final_m,
        "rounds": round_log,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--learners", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--inner-steps", type=int, default=35)
    ap.add_argument("--inner-lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--shard-size", type=int, default=512)
    ap.add_argument("--outer-lr", type=float, default=0.75)
    ap.add_argument("--outer-mu", type=float, default=0.6)
    ap.add_argument("--quant", choices=["nf4", "none"], default="nf4")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-free-mib", type=float, default=1500.0)
    ap.add_argument("--arms", nargs="+", default=list(ARMS.keys()))
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/experiments/E29_qlora_moe_jepa_pilot.json")
    a = ap.parse_args(argv)

    free_before = e8q.vram_guard(a.min_free_mib, 250.0, a.device)
    runs = []
    for seed in a.seeds:
        for arm in a.arms:
            r = run_seed_arm(arm, seed, a, a.device)
            runs.append(r)
            print(
                f"[seed {seed} | {arm:17s}] NLL {r['initial']['target_nll']:.4f} -> {r['final']['target_nll']:.4f} "
                f"| acc={100*r['final']['target_token_acc']:5.2f}% | sv1={r['final']['view_gap_sv1']:6.3f} "
                f"| vram={r['vram_peak_mib']} MiB | {r['wall_seconds']:.1f}s",
                flush=True,
            )

    summary = {}
    for arm in a.arms:
        sub = [r for r in runs if r["arm"] == arm]
        nlls = [r["final"]["target_nll"] for r in sub]
        accs = [r["final"]["target_token_acc"] for r in sub]
        sv1s = [r["final"]["view_gap_sv1"] for r in sub]
        secs = [r["wall_seconds"] for r in sub]
        vrams = [r["vram_peak_mib"] for r in sub if r["vram_peak_mib"] is not None]
        summary[arm] = {
            "final_target_nll_mean": round(statistics.mean(nlls), 6),
            "final_target_nll_sd": round(statistics.stdev(nlls), 6) if len(nlls) > 1 else 0.0,
            "final_token_acc_mean": round(statistics.mean(accs), 4),
            "final_view_gap_sv1_mean": round(statistics.mean(sv1s), 4),
            "wall_seconds_mean": round(statistics.mean(secs), 3),
            "vram_peak_mib_max": max(vrams) if vrams else None,
            "per_seed_target_nll": nlls,
        }

    b3_gates = {}
    if "qlora_ntp" in summary:
        ntp_nlls = summary["qlora_ntp"]["per_seed_target_nll"]
        ntp_sv1 = summary["qlora_ntp"]["final_view_gap_sv1_mean"]
        for cand in ("qlora_jepa_drop75", "qlora_jepa_full", "qlora_jepa_shuf"):
            if cand in summary and len(ntp_nlls) >= 2:
                b3_gates[cand] = evaluate_jepa_b3_gate(
                    ntp_nlls,
                    summary[cand]["per_seed_target_nll"],
                    view_gap_sv1_ntp=ntp_sv1,
                    view_gap_sv1_jepa=summary[cand]["final_view_gap_sv1_mean"],
                )

    out: dict[str, Any] = {
        "experiment": "E29",
        "tickets": ["IP-JEPA-03", "IP-JEPA-04"],
        "title": "Stage-2 4-bit NF4 GraniteMoE QLoRA DiLoCo with LLM-JEPA & B3 Held-Out Promotion Gate",
        "reference": "Huang, LeCun, Balestriero, arXiv:2509.14252",
        "hardware": {
            "device": a.device,
            "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None,
            "free_vram_before_mib": free_before,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "config": {
            "seeds": a.seeds,
            "learners": a.learners,
            "rounds": a.rounds,
            "inner_steps": a.inner_steps,
            "inner_lr": a.inner_lr,
            "batch": a.batch,
            "shard_size": a.shard_size,
            "outer_lr": a.outer_lr,
            "outer_mu": a.outer_mu,
            "quant": a.quant,
            "model": "GraniteMoeForCausalLM tiny (4 experts, top-2, bitsandbytes NF4 + PEFT LoRA r=4 on q_proj,v_proj + lm_head)",
        },
        "summary": summary,
        "b3_promotion_gates_vs_qlora_ntp": b3_gates,
        "runs": runs,
        "date": time.strftime("%Y-%m-%d"),
    }
    out["receipt_sha256"] = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "b3_promotion_gates_vs_qlora_ntp": b3_gates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
