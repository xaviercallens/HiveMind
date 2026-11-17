#!/usr/bin/env python3
"""Analytic peak-VRAM estimator for NF4 QLoRA on one GPU (stdlib only, CPU only, no weights are ever loaded).

Components (all in MiB):
  1. quantised base weights: every ``nn.Linear`` weight (attention, dense MLP, MoE router) at 0.53 byte/parameter
     (NF4 plus double-quantisation state; the figure ``scripts/t4_model_candidates.py`` uses);
  2. 16-bit parts at 2 byte/parameter: embeddings (and an untied lm_head), norms, and the FUSED 3-D MoE expert tensors
     (transformers 5.17 stores experts fused, bitsandbytes quantises only ``nn.Linear``: ``docs/T4_NIGHT_MODEL_DECISION.md``);
  3. LoRA weights (fp32), their gradients (fp32) and the optimizer state;
  4. activations, with and without gradient checkpointing, plus the logits;
  5. a transient de-quantisation buffer and a CUDA-context / allocator-fragmentation allowance.

Calibration (see ``CALIBRATION_POINTS``): one exact measurement and two out-of-memory lower bounds, all from
``docs/T4_NIGHT_MODEL_DECISION.md`` (seq 128, micro-batch 2, rank 8, q/k/v/o, no checkpointing, torch allocator cap 3950 MiB). One scalar
(``act_scale``) is fitted; the leave-one-out error is printed on every run and the output says so when it is above 20 percent,
in which case the numbers are an order-of-magnitude guide only.

  python3 scripts/qlora_vram_estimate.py                      # calibration report + candidate-by-tier table (reads configs from the HF cache)
  python3 scripts/qlora_vram_estimate.py --json               # same, machine-readable
  python3 scripts/qlora_vram_estimate.py --config PATH --rank 16 --seq 512 --micro-batch 2 --checkpointing

Not a measurement: nothing here ran on a GPU.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

MIB = 2**20
NF4_BYTES = 0.53  # bytes per quantised parameter incl. quant state (same constant as scripts/t4_model_candidates.py)
FP16_BYTES = 2.0
LORA_BYTES = 4.0  # fp32 LoRA weights (moe_app.py keeps LoRA in fp32)
GRAD_BYTES = 4.0
# optimizer state bytes per trainable parameter (the repo trainer uses torch AdamW, i.e. two fp32 moments)
OPTIMIZERS: Dict[str, float] = {"adamw": 8.0, "adamw8bit": 2.0, "sgd": 0.0, "sgd_momentum": 4.0}
ALL_LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ATTN_ELEMS = 10.0  # activation elements per token per layer held for backward, per hidden unit (norms, q/k/v/o, residuals)
MLP_ELEMS = 3.0  # same, per MLP-width unit (gate, up, act*up)
LOGIT_BYTES = 10.0  # per token per vocab entry: fp16 logits + fp32 upcast + fp32 gradient
OVERHEAD_MIB = 3716 - 3409  # process peak (3716, deploy/t4_node/live/README.md) minus torch allocated peak (3409, docs/T4_NIGHT_MODEL_DECISION.md)
HF_HUB_DEFAULT = "/mnt/disks/disk-socrateai-local-1/.cache/huggingface/hub"
THRESHOLD = 0.20  # above this leave-one-out error the estimator is only an order-of-magnitude guide


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hidden: int
    layers: int
    vocab: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    inter: int  # dense MLP width, or one expert's width for a MoE
    n_experts: int = 0  # 0 = dense
    top_k: int = 0
    tie_embeddings: bool = True
    total_params: float = 0.0
    active_params: float = 0.0
    linear_params: float = 0.0  # nn.Linear weights (quantised)
    fp16_params: float = 0.0  # embeddings, norms, fused experts (16-bit)
    fused_expert_params: float = 0.0
    note: str = ""

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0

    @property
    def q_out(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_out(self) -> int:
        return self.n_kv_heads * self.head_dim


def spec_from_config(cfg: dict, name: str, *, total_override: Optional[float] = None, note: str = "") -> ModelSpec:
    """Build a spec from a Hugging Face ``config.json`` dict (text_config is used when present). Plain transformer formulas."""
    t = dict(cfg.get("text_config") or cfg)
    h, L, V = int(t["hidden_size"]), int(t["num_hidden_layers"]), int(t["vocab_size"])
    nh, nkv = int(t["num_attention_heads"]), int(t.get("num_key_value_heads") or t["num_attention_heads"])
    hd = int(t.get("head_dim") or h // nh)
    inter = int(t["intermediate_size"])
    E = int(t.get("num_local_experts") or t.get("num_experts") or 0)
    k = int(t.get("num_experts_per_tok") or 0)
    tied = bool(t.get("tie_word_embeddings", cfg.get("tie_word_embeddings", True)))
    attn = h * nh * hd + 2 * h * nkv * hd + nh * hd * h
    embed = V * h * (1 if tied else 2)
    norms = (2 * L + 1) * h
    if E:
        router = h * E
        fused = L * E * 3 * h * inter
        linear = L * (attn + router)
        fp16 = embed + norms + fused
        active = L * (attn + router + k * 3 * h * inter) + embed + norms
    else:
        fused = 0
        linear = L * (attn + 3 * h * inter)
        fp16 = embed + norms
        active = linear + fp16
    total = linear + fp16
    if total_override:  # architectures whose block layout is not the plain formula (see PRESET_OVERRIDES)
        linear = total_override - fp16
        total = total_override
        active = total_override if not E else active
    return ModelSpec(name=name, hidden=h, layers=L, vocab=V, n_heads=nh, n_kv_heads=nkv, head_dim=hd, inter=inter, n_experts=E, top_k=k,
                     tie_embeddings=tied, total_params=float(total), active_params=float(active), linear_params=float(linear),
                     fp16_params=float(fp16), fused_expert_params=float(fused), note=note)


@dataclass(frozen=True)
class Recipe:
    rank: int = 8
    targets: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    seq_len: int = 128
    micro_batch: int = 2
    checkpointing: bool = False
    optimizer: str = "adamw"


def lora_params(m: ModelSpec, r: Recipe) -> int:
    """Trainable LoRA parameters. gate/up/down only exist as nn.Linear on dense models; on a MoE they are the fused experts and get no adapter."""
    h = m.hidden
    dims = {"q_proj": (h, m.q_out), "k_proj": (h, m.kv_out), "v_proj": (h, m.kv_out), "o_proj": (m.q_out, h)}
    if not m.is_moe:
        dims.update({"gate_proj": (h, m.inter), "up_proj": (h, m.inter), "down_proj": (m.inter, h)})
    return m.layers * sum(r.rank * (a + b) for tgt in r.targets for a, b in [dims.get(tgt, (0, 0))])


def _mlp_width(m: ModelSpec) -> int:
    return m.top_k * m.inter if m.is_moe else m.inter


def breakdown(m: ModelSpec, r: Recipe, act_scale: float = 1.0) -> Dict[str, float]:
    """Component estimate in MiB. ``alloc_peak`` is comparable to torch.cuda.max_memory_allocated; ``process_peak`` adds the context allowance."""
    if r.optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {sorted(OPTIMIZERS)}")
    if r.rank < 1 or r.seq_len < 1 or r.micro_batch < 1:
        raise ValueError("rank, seq_len and micro_batch must be >= 1")
    nf4 = m.linear_params * NF4_BYTES / MIB
    fp16 = m.fp16_params * FP16_BYTES / MIB
    lp = lora_params(m, r)
    lora = lp * (LORA_BYTES + GRAD_BYTES + OPTIMIZERS[r.optimizer]) / MIB
    tokens = r.seq_len * r.micro_batch
    per_layer = tokens * (ATTN_ELEMS * m.hidden + MLP_ELEMS * _mlp_width(m)) * 2 / MIB  # fp16 activations of one layer
    if r.checkpointing:
        acts_layers = m.layers * tokens * m.hidden * 2 / MIB + per_layer  # saved layer inputs + one layer recomputed
    else:
        acts_layers = m.layers * per_layer
    logits = tokens * m.vocab * LOGIT_BYTES / MIB
    biggest = max(m.hidden * max(m.q_out, _mlp_width(m) if not m.is_moe else m.hidden), 1)
    dequant = 2 * 2 * biggest / MIB  # one de-quantised fp16 weight in forward and one in backward
    acts = act_scale * (acts_layers + logits)
    alloc = nf4 + fp16 + lora + acts + dequant
    return {"nf4_weights": nf4, "fp16_weights": fp16, "fused_experts_included_in_fp16": m.fused_expert_params * FP16_BYTES / MIB,
            "lora_params": float(lp), "lora_state": lora, "activations_layers": act_scale * acts_layers, "logits": act_scale * logits,
            "dequant_buffer": dequant, "resident_weights": nf4 + fp16, "alloc_peak": alloc, "context_allowance": float(OVERHEAD_MIB),
            "process_peak": alloc + OVERHEAD_MIB}


# --- calibration -----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CalPoint:
    key: str
    model: str
    recipe: Recipe
    measured_mib: float  # torch allocated peak, or for an OOM point the reported size at the failed allocation
    censored: bool  # True: an out-of-memory under a 3950 MiB cap, so the true demand is >= the cap (a lower bound)
    source: str
    resident_mib: float  # resident after load (measured), for the weights-only check


BASE_RECIPE = Recipe()  # seq 128, micro-batch 2, rank 8, q/k/v/o, no checkpointing (the workunit; docs/T4_NIGHT_MODEL_DECISION.md)
CALIBRATION_POINTS: Tuple[CalPoint, ...] = (
    CalPoint("granite-1b-a400m", "granite-3.1-1b-a400m-base", BASE_RECIPE, 3409.0, False, "docs/T4_NIGHT_MODEL_DECISION.md: peak 3,409 MiB allocated", 2444.0),
    CalPoint("qwen3-4b", "Qwen3-4B-Instruct-2507", BASE_RECIPE, 3.97 * 1024, True, "docs/T4_NIGHT_MODEL_DECISION.md: OOM at 3.97 GiB, micro-batch 2", 2578.0),
    CalPoint("phi-4-mini", "Phi-4-mini-instruct", BASE_RECIPE, 4.0 * 1024, True, "docs/T4_NIGHT_MODEL_DECISION.md: OOM at 4.0 GiB", 2764.0),
)
CAP_MIB = 3950.0


def _fit_scale(points: Sequence[Tuple[ModelSpec, CalPoint]]) -> float:
    """Closed-form least squares on RELATIVE error for pred = fixed + s * act (s >= 0)."""
    num = den = 0.0
    for m, p in points:
        b0 = breakdown(m, p.recipe, 0.0)["alloc_peak"]
        a1 = breakdown(m, p.recipe, 1.0)["alloc_peak"] - b0
        num += ((p.measured_mib - b0) / p.measured_mib) * (a1 / p.measured_mib)
        den += (a1 / p.measured_mib) ** 2
    return max(num / den, 0.0) if den else 1.0


def calibrate(specs: Dict[str, ModelSpec]) -> dict:
    pts = [(specs[p.model], p) for p in CALIBRATION_POINTS if p.model in specs]
    if len(pts) < 2:
        return {"available": False, "reason": "calibration models' config.json not found in the cache", "act_scale": 1.0}
    scale = _fit_scale(pts)
    rows = []
    for i, (m, p) in enumerate(pts):
        loo = _fit_scale([q for j, q in enumerate(pts) if j != i])
        pred_loo = breakdown(m, p.recipe, loo)["alloc_peak"]
        pred_in = breakdown(m, p.recipe, scale)["alloc_peak"]
        resident = breakdown(m, replace(p.recipe, targets=()), 0.0)["resident_weights"] + lora_params(m, p.recipe) * LORA_BYTES / MIB
        rows.append({"point": p.key, "measured_mib": round(p.measured_mib, 1), "censored_oom_lower_bound": p.censored, "source": p.source,
                     "act_scale_fitted_without_this_point": round(loo, 3), "loo_pred_mib": round(pred_loo, 1),
                     "loo_rel_error": round((pred_loo - p.measured_mib) / p.measured_mib, 4), "in_sample_pred_mib": round(pred_in, 1),
                     "in_sample_rel_error": round((pred_in - p.measured_mib) / p.measured_mib, 4),
                     "loo_pred_reaches_cap_3950": pred_loo >= CAP_MIB if p.censored else None,
                     "resident_pred_mib": round(resident, 1), "resident_measured_mib": p.resident_mib,
                     "resident_rel_error": round((resident - p.resident_mib) / p.resident_mib, 4)})
    errs = [abs(r["loo_rel_error"]) for r in rows]
    return {"available": True, "act_scale": round(scale, 4), "points": rows, "loo_max_abs_error": round(max(errs), 4), "loo_mean_abs_error": round(sum(errs) / len(errs), 4),
            "overhead_mib": OVERHEAD_MIB, "overhead_basis": "3716 (process peak, deploy/t4_node/live/README.md) - 3409 (allocated peak): ONE pair, not cross-validated",
            "poor": max(errs) > THRESHOLD,
            "verdict": ("POOR: leave-one-out error above 20 percent, so this estimator is only an order-of-magnitude guide" if max(errs) > THRESHOLD
                        else "leave-one-out error within 20 percent on three points (two of them out-of-memory lower bounds); extrapolation to other shapes is Not tested.")}


# --- presets from the local Hugging Face cache (config.json only) ------------------------------------------------------------------------
PRESETS: Dict[str, str] = {  # short name -> cache directory fragment
    "granite-3.1-1b-a400m-base": "ibm-granite--granite-3.1-1b-a400m-base",
    "granite-3.1-3b-a800m-base": "ibm-granite--granite-3.1-3b-a800m-base",
    "OLMoE-1B-7B-0924": "allenai--OLMoE-1B-7B-0924",
    "Qwen3-4B-Instruct-2507": "Qwen--Qwen3-4B-Instruct-2507",
    "Qwen3.5-4B": "Qwen--Qwen3.5-4B",
    "Phi-tiny-MoE-instruct": "microsoft--Phi-tiny-MoE-instruct",
    "Phi-4-mini-instruct": "microsoft--Phi-4-mini-instruct",
    "Qwen2.5-Coder-7B-Instruct": "Qwen--Qwen2.5-Coder-7B-Instruct",
}
# Qwen3.5 is a hybrid (linear + full attention) with a vision tower: the plain formula does not apply, so the dense 4.7 B total from the decision doc is used.
PRESET_OVERRIDES: Dict[str, dict] = {"Qwen3.5-4B": {"total_override": 4.7e9, "note": "hybrid linear/full attention; total 4.7B from docs/T4_NIGHT_MODEL_DECISION.md; activation and LoRA geometry approximate"}}


def load_presets(hub: str = HF_HUB_DEFAULT) -> Dict[str, ModelSpec]:
    out: Dict[str, ModelSpec] = {}
    for name, frag in PRESETS.items():
        found = sorted(glob.glob(os.path.join(hub, f"models--{frag}", "snapshots", "*", "config.json")))
        if not found:
            continue
        with open(found[0], "r", encoding="utf-8") as f:
            cfg = json.load(f)
        out[name] = spec_from_config(cfg, name, **PRESET_OVERRIDES.get(name, {}))
    return out


RECIPES: Dict[str, Recipe] = {
    "R0 workunit (seq128 mb2 r8 qkvo)": BASE_RECIPE,
    "R1 big profile (seq512 mb4 r16 qkvo)": Recipe(rank=16, seq_len=512, micro_batch=4),
    "R2 intense-safe (seq512 mb1 r16 all-linear ckpt)": Recipe(rank=16, targets=ALL_LINEAR, seq_len=512, micro_batch=1, checkpointing=True),
}
BUDGETS = (4300, 10000, 14000)


def verdict(peak: float, budget: float, err: float) -> str:
    if peak * (1 + err) <= budget:
        return "fits"
    if peak <= budget:
        return "marginal"
    return "NO"


def table(specs: Dict[str, ModelSpec], scale: float, err: float) -> List[dict]:
    rows = []
    for name, m in specs.items():
        for rname, r in RECIPES.items():
            b = breakdown(m, r, scale)
            rows.append({"model": name, "recipe": rname, "process_peak_mib": round(b["process_peak"]), "resident_weights_mib": round(b["resident_weights"]),
                         "total_b": round(m.total_params / 1e9, 2), "active_b": round(m.active_params / 1e9, 2),
                         **{f"tier_{bud}": verdict(b["process_peak"], bud, err) for bud in BUDGETS}})
    return rows


def out_of_sample(specs: Dict[str, ModelSpec], scale: float) -> Optional[dict]:
    """Informal check, NOT used in the fit: NIGHT-MOE-1 measured 10.23 GiB for QLoRA on Qwen2.5-Coder-7B (docs/NIGHT_MOE_2_PLAN.md), with settings the doc does
    not state. The sequence length (about 300 tokens, from NIGHT_MOE_TRAINING_PLAN.md) and the rest below are ASSUMED, so this is a plausibility check only."""
    m = specs.get("Qwen2.5-Coder-7B-Instruct")
    if m is None:
        return None
    out = {}
    for mb in (1, 2):
        b = breakdown(m, Recipe(rank=16, targets=ALL_LINEAR, seq_len=300, micro_batch=mb), scale)
        out[f"mb{mb}_alloc_mib"] = round(b["alloc_peak"])
        out[f"mb{mb}_process_mib"] = round(b["process_peak"])
    out["measured_gib_unknown_settings"] = 10.23
    out["measured_mib"] = round(10.23 * 1024)
    out["assumed"] = "rank 16, q/k/v/o/gate/up/down, seq 300, no checkpointing, micro-batch 1 or 2"
    return out


def render(cal: dict, rows: List[dict], oos: Optional[dict] = None) -> str:
    L: List[str] = []
    if cal.get("available"):
        L.append(f"act_scale fitted on 3 points = {cal['act_scale']}; context allowance = {cal['overhead_mib']} MiB ({cal['overhead_basis']})")
        L.append("leave-one-out (fit on the other two points, predict this one; torch allocated peak, MiB):")
        for p in cal["points"]:
            tag = " [OOM lower bound]" if p["censored_oom_lower_bound"] else ""
            L.append(f"  {p['point']:18s} measured {p['measured_mib']:7.1f}{tag:18s} LOO pred {p['loo_pred_mib']:7.1f}  LOO err {100*p['loo_rel_error']:+6.1f}%  "
                     f"in-sample err {100*p['in_sample_rel_error']:+6.1f}%  weights-only err {100*p['resident_rel_error']:+5.1f}%")
        L.append(f"LOO max |error| {100*cal['loo_max_abs_error']:.1f}%, mean {100*cal['loo_mean_abs_error']:.1f}%.  {cal['verdict']}")
    else:
        L.append("calibration unavailable: " + cal.get("reason", ""))
    if oos:
        L.append(f"informal out-of-sample check (not in the fit; settings assumed): 7B coder, {oos['assumed']}: estimate {oos['mb1_process_mib']} MiB (mb1) to "
                 f"{oos['mb2_process_mib']} MiB (mb2) vs measured {oos['measured_mib']} MiB ({oos['measured_gib_unknown_settings']} GiB, docs/NIGHT_MOE_2_PLAN.md, settings unknown)")
    L.append("")
    L.append("Process peak (MiB, allocated + context allowance) and fit per free-VRAM budget. 'marginal' = fits the point estimate but not with the margin added (max(LOO error, 20%)); NO = over budget.")
    L.append(f"{'model':28s} {'tot/act B':>10s} {'recipe':50s} {'weights':>7s} {'peak':>6s} " + " ".join(f"{b:>9d}" for b in BUDGETS))
    for r in rows:
        L.append(f"{r['model']:28s} {r['total_b']:5.2f}/{r['active_b']:<4.2f} {r['recipe']:50s} {r['resident_weights_mib']:7d} {r['process_peak_mib']:6d} " + " ".join(f"{r[f'tier_{b}']:>9s}" for b in BUDGETS))
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hub", default=HF_HUB_DEFAULT, help="Hugging Face hub cache directory (read-only; only config.json files are opened)")
    ap.add_argument("--config", help="estimate one arbitrary config.json instead of the table")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--checkpointing", action="store_true")
    ap.add_argument("--optimizer", default="adamw", choices=sorted(OPTIMIZERS))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    specs = load_presets(a.hub)
    cal = calibrate(specs)
    scale = cal["act_scale"]
    err = max(cal.get("loo_max_abs_error", THRESHOLD), THRESHOLD)  # never claim a tighter margin than 20 percent
    if a.config:
        with open(a.config, "r", encoding="utf-8") as f:
            m = spec_from_config(json.load(f), os.path.basename(os.path.dirname(a.config)) or "custom")
        r = Recipe(rank=a.rank, targets=tuple(t for t in a.targets.split(",") if t), seq_len=a.seq, micro_batch=a.micro_batch, checkpointing=a.checkpointing, optimizer=a.optimizer)
        b = breakdown(m, r, scale)
        print(json.dumps({"model": m.name, "breakdown_mib": {k: round(v, 1) for k, v in b.items()}, "calibration_loo_max_abs_error": cal.get("loo_max_abs_error")}, indent=1))
        return 0
    rows = table(specs, scale, err)
    if a.json:
        print(json.dumps({"calibration": cal, "out_of_sample": out_of_sample(specs, scale), "table": rows, "budgets_mib": list(BUDGETS), "margin_used": err}, indent=1))
    else:
        print(render(cal, rows, out_of_sample(specs, scale)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
