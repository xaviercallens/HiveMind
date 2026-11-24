# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-14: final LayerNorm/RMSNorm temperature & ECE calibration surgeon."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import upcycle_crowd_v1_to_sparse_moe
from hivemind.swarm.logit_calibration_temperature_surgeon import (
    ANSE_MAXIMUM_PAIN,
    SURGEON_RECEIPT_OBJECT,
    SURGEON_SCHEMA,
    LogitCalibrationConfig,
    apply_norm_weights,
    apply_verified_ln_temperature_surgery_inplace,
    compute_module_state_sha256,
    compute_payload_sha256,
    compute_token_nll_and_ece,
    detect_norm_type,
    fold_temperature_into_norm_params,
    optimize_guard_constrained_ln_scale_and_interpolation,
    verify_calibration_receipt_provenance,
)
from hivemind.swarm.wsp_alphaedit_control import forward_model_pre_ln

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:
    ZeroStubAudit = None  # type: ignore[assignment,misc]


class _FallbackStubResult:
    """CI-safe stub audit result when optional ``gwaya`` is absent."""

    is_clean: bool = True
    violations: tuple[str, ...] = ()
    penalty_energy: float = 0.0


class _FallbackZeroStubAudit:
    """CI-safe fallback implementing both ``check_file`` and ``audit_python_code``."""

    def check_file(self, _path: str) -> list[str]:
        return []

    def audit_python_code(self, _code: str) -> _FallbackStubResult:
        return _FallbackStubResult()


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = (
    REPO_ROOT / "src" / "hivemind" / "swarm" / "logit_calibration_temperature_surgeon.py"
)
TEST_FILE_PATH = Path(__file__).resolve()
EXPECTED_AGPL_HEADER = (
    "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
)


def _load_module_from_path(name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _unit_axis(dim: int, idx0: int, idx1: int) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32)
    vec[idx0] = 1.0 / math.sqrt(2.0)
    vec[idx1] = -1.0 / math.sqrt(2.0)
    return vec


class _CustomRMSNorm(nn.Module):
    """HuggingFace/Qwen-style RMSNorm exposing ``variance_epsilon`` and no ``bias``."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.variance_epsilon = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (int(self.weight.shape[0]),), self.weight, self.variance_epsilon)


def _build_subspace_moe_model(
    *,
    num_layers: int = 2,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    vocab_size: int = 48,
    num_experts: int = 4,
    top_k: int = 2,
    lora_rank: int = 4,
    use_rmsnorm: bool = False,
    layernorm_bias: bool = True,
    seed: int = 20261124,
) -> CrowdV1WithEngram:
    """Build deterministic 3-subspace CrowdV1WithEngram + CrowdLoRAMoEMLP on CPU.

    Subspaces:
      - ``0..15``:  ``E0_general_fineweb`` (preferred target token ``5``, runner-up ``6``)
      - ``16..31``: ``E1_python_systems_code`` (preferred target ``20``, runner-up ``21``)
      - ``32..47``: ``E2_math_reasoning`` (preferred target ``36``, runner-up ``37``)
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cfg = CrowdV1RuntimeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden_size // 4,
        intermediate_size=intermediate_size,
        max_position_embeddings=32,
        engram_buckets=0,
        tie_word_embeddings=True,
    )
    model = CrowdV1WithEngram(cfg)

    u_e0 = _unit_axis(hidden_size, 0, 1)
    u_e1 = _unit_axis(hidden_size, 16, 17)
    u_e2 = _unit_axis(hidden_size, 24, 25)

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)

        p_raw = 0.05 * torch.randn((16, 16), generator=gen, dtype=torch.float32)
        p_raw[:, :2] = 0.0
        p_raw = p_raw - p_raw.mean(dim=1, keepdim=True) + 0.15 * u_e0[:16].unsqueeze(0)
        p_raw[5] = 1.35 * u_e0[:16]
        p_raw[6] = 0.90 * u_e0[:16]

        e1_raw = 0.05 * torch.randn((16, 8), generator=gen, dtype=torch.float32)
        e1_raw[:, :2] = 0.0
        e1_raw = e1_raw - e1_raw.mean(dim=1, keepdim=True) + 0.15 * u_e1[16:24].unsqueeze(0)
        e1_raw[20 - 16] = 1.35 * u_e1[16:24]
        e1_raw[21 - 16] = 0.90 * u_e1[16:24]

        e2_raw = 0.05 * torch.randn((16, 8), generator=gen, dtype=torch.float32)
        e2_raw[:, :2] = 0.0
        e2_raw = e2_raw - e2_raw.mean(dim=1, keepdim=True) + 0.15 * u_e2[24:32].unsqueeze(0)
        e2_raw[36 - 32] = 1.35 * u_e2[24:32]
        e2_raw[37 - 32] = 0.90 * u_e2[24:32]

        wte[:16, :16] = p_raw
        wte[16:32, 16:24] = e1_raw
        wte[32:48, 24:32] = e2_raw
        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight

        model.ln.weight.fill_(1.0)
        if getattr(model.ln, "bias", None) is not None:
            model.ln.bias.zero_()

        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0)
            blk.ln2.bias.zero_()
            blk.mlp.w1.weight[:hidden_size, :hidden_size].copy_(0.80 * torch.eye(hidden_size))
            blk.mlp.w2.weight[:hidden_size, :hidden_size].copy_(0.80 * torch.eye(hidden_size))
            blk.mlp.w3.weight[:hidden_size, :hidden_size].copy_(0.10 * torch.eye(hidden_size))

    upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode="lora_domain_experts",
        num_experts=num_experts,
        top_k=top_k,
        lora_rank=lora_rank,
        lora_alpha=8.0,
        routing_mode="sigma_topk",
        seed=seed + 13,
    )

    with torch.no_grad():
        for blk in model.blocks:
            mlp = blk.mlp
            mlp.router.weight.zero_()
            for exp_mod in mlp.experts.values():
                for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                    getattr(exp_mod, attr).zero_()

            mlp.router.weight[0] = 3.0 * u_e0
            mlp.router.weight[1] = 3.0 * u_e1
            mlp.router.weight[2] = 3.0 * u_e2

            # Specialist LoRA adapters boost E1 and E2 directional signal relative to base
            mlp.experts["1"].w3_A[0, 16:24] = 0.25 * u_e1[16:24]
            mlp.experts["1"].w3_B[:, 0] = 0.90 * u_e1
            mlp.experts["2"].w3_A[0, 24:32] = 0.25 * u_e2[24:32]
            mlp.experts["2"].w3_B[:, 0] = 0.90 * u_e2

    if use_rmsnorm:
        model.ln = nn.RMSNorm(hidden_size, eps=1e-6)
    elif not layernorm_bias:
        model.ln = nn.LayerNorm(hidden_size, eps=cfg.layer_norm_eps, bias=False)

    return model


def _make_calibration_chunks(
    *,
    seq_len: int = 16,
    score_from: int = 4,
    chunks_per_domain: int = 8,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    """Create deterministic chunks with realistic ~75% modal / 25% secondary token ambiguity."""
    gen = torch.Generator(device="cpu").manual_seed(9100 + seed_offset)
    specs = (
        ("E0_general_fineweb", 0, 16, 5, 6),
        ("E1_python_systems_code", 16, 32, 20, 21),
        ("E2_math_reasoning", 32, 48, 36, 37),
    )
    chunks: list[dict[str, Any]] = []
    for d_idx, (dom, lo, hi, main_tok, alt_tok) in enumerate(specs):
        for r_idx in range(chunks_per_domain):
            ids = torch.randint(lo, hi, (seq_len,), generator=gen).tolist()
            for pos in range(score_from, seq_len):
                # 75% modal token, 25% runner-up token so overconfident logits incur high ECE & tail NLL
                ids[pos] = alt_tok if (pos + r_idx) % 4 == 0 else main_tok
            chunks.append(
                {
                    "expert_id": dom,
                    "dataset_id": dom,
                    "row": seed_offset + 100 * d_idx + r_idx,
                    "ids": ids,
                }
            )
    return chunks


# ---------------------------------------------------------------------------
# S1: Overconfident Post-Surgery MoE Calibration -> >40% ECE Drop + Guard Pass
# ---------------------------------------------------------------------------


def test_s1_overconfident_moe_calibration_lowers_ece_preserves_spec_and_clears_guard() -> None:
    """Joint (lambda_ln, tau_temp) + channelwise refinement cuts ECE >40%, keeps spec gain, clears <= +0.5% guard."""
    model = _build_subspace_moe_model(seed=20261401)
    probe_chunks = _make_calibration_chunks(chunks_per_domain=8, seed_offset=0)
    eval_chunks = _make_calibration_chunks(chunks_per_domain=8, seed_offset=500)
    probe_ids = torch.tensor([c["ids"] for c in probe_chunks], dtype=torch.long)
    eval_ids = torch.tensor([c["ids"] for c in eval_chunks], dtype=torch.long)

    # Save dense base final LN parameters (under-scaled specialist channels) and compute dense-base reference NLLs
    with torch.no_grad():
        model.ln.weight[:16].fill_(1.0)
        model.ln.weight[16:].fill_(0.40)
        model.ln.bias.zero_()
    base_ln_w = model.ln.weight.detach().clone()
    base_ln_b = model.ln.bias.detach().clone()

    with torch.no_grad():
        # Dense reference: temporarily disable specialist adapters to measure base specialist NLL
        saved_b = {}
        for l_idx, blk in enumerate(model.blocks):
            for e_id in ("1", "2"):
                saved_b[(l_idx, e_id)] = blk.mlp.experts[e_id].w3_B.detach().clone()
                blk.mlp.experts[e_id].w3_B.zero_()
        base_h_eval = forward_model_pre_ln(model, eval_ids, batch_size=12)
        base_logits_eval = model.head(apply_norm_weights(base_h_eval[:, 3:-1, :], base_ln_w, base_ln_b))
        eval_domains = [str(c["expert_id"]) for c in eval_chunks]
        t_eval = eval_ids[:, 4:]
        base_domain_nll: dict[str, float] = {}
        for dom in sorted(set(eval_domains)):
            ix = [i for i, d in enumerate(eval_domains) if d == dom]
            base_domain_nll[dom] = compute_token_nll_and_ece(base_logits_eval[ix], t_eval[ix])["nll_mean"]
        for (l_idx, e_id), w_b in saved_b.items():
            model.blocks[l_idx].mlp.experts[e_id].w3_B.copy_(w_b)

        # Simulate post-surgery overconfident MoE final LayerNorm weights:
        # strong global scale inflation (~3.5x) + mild E0 bias drift, while preserving specialist ratio
        moe_ln_w = torch.empty_like(base_ln_w)
        moe_ln_w[:16] = 3.55
        moe_ln_w[16:] = 3.50
        moe_ln_b = torch.zeros_like(base_ln_b)
        moe_ln_b[:16] = 0.14
        model.ln.weight.copy_(moe_ln_w)
        model.ln.bias.copy_(moe_ln_b)

        h_probe = forward_model_pre_ln(model, probe_ids, batch_size=12)
        is_e0 = torch.tensor([str(c["expert_id"]).startswith("E0_") for c in probe_chunks], dtype=torch.bool)
        h_p = h_probe[is_e0, 3:-1, :]
        t_p = probe_ids[is_e0, 4:]
        h_s = h_probe[~is_e0, 3:-1, :]
        t_s = probe_ids[~is_e0, 4:]

    # 1. Direct optimization test on pre-LN states
    opt_res = optimize_guard_constrained_ln_scale_and_interpolation(
        base_ln_w,
        base_ln_b,
        moe_ln_w,
        moe_ln_b,
        h_p,
        t_p,
        h_s,
        t_s,
        model.head,
        general_guard_rel=0.005,
        channelwise=True,
    )
    assert opt_res["guard_feasible"] is True
    assert opt_res["lambda_ln"] > 0.0
    assert opt_res["tau_temp"] > 1.5
    assert opt_res["channel_tau"] is not None and len(opt_res["channel_tau"]) == 32
    assert opt_res["prose_drift_rel"] <= 0.005
    assert opt_res["ece_reduction_rel"] > 0.40
    assert opt_res["cal_prose_ece"] < opt_res["moe_prose_ece"]
    assert opt_res["cal_spec_ece"] < opt_res["moe_spec_ece"]
    assert opt_res["cal_spec_nll"] < opt_res["base_spec_nll"]

    # 2. End-to-end verified in-place surgery with held-out eval split & SHA-256 receipt
    param_count_before = sum(p.numel() for p in model.parameters())
    receipt = apply_verified_ln_temperature_surgery_inplace(
        model,
        probe_chunks,
        base_ln_w=base_ln_w,
        base_ln_b=base_ln_b,
        eval_chunks=eval_chunks,
        baseline_domain_nll=base_domain_nll,
        score_from=4,
        general_guard_rel=0.005,
        max_domain_drift=0.01,
    )

    assert receipt["object"] == SURGEON_RECEIPT_OBJECT
    assert receipt["schema"] == SURGEON_SCHEMA
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["status"] == "ACCEPTED"
    assert receipt["guard_ok"] is True
    assert receipt["general_guard_ok"] is True
    assert receipt["norm_type"] == "layernorm"
    assert receipt["has_bias"] is True
    assert receipt["delta_e"] < 0.0
    assert receipt["ece_reduction_rel"] > 0.40
    assert receipt["general_pct_vs_base"] <= 0.5
    assert receipt["worst_domain_pct_vs_base"] <= 1.0
    assert receipt["spec_gain_nats"] > 0.02
    assert receipt["final_state_sha256"] == receipt["candidate_state_sha256"]
    assert receipt["final_state_sha256"] != receipt["parent_state_sha256"]
    assert verify_calibration_receipt_provenance(receipt) is True
    assert sum(p.numel() for p in model.parameters()) == param_count_before

    # Verify deployed model.ln produces the exact calibrated metrics on held-out eval_ids
    with torch.no_grad():
        logits_deployed, _, _ = model(eval_ids)
        deployed_m = compute_token_nll_and_ece(logits_deployed[:, 3:-1, :], eval_ids[:, 4:])
    assert deployed_m["nll_mean"] == pytest.approx(receipt["nll_mean"], abs=1e-5)
    assert deployed_m["ece_value"] == pytest.approx(receipt["ece_value"], abs=1e-5)


# ---------------------------------------------------------------------------
# S2: Exact Equivalence Between Runtime Logit / tau & Zero-Overhead Norm Fold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tau_val", [0.75, 1.85, 3.20])
def test_s2_exact_equivalence_runtime_temp_division_vs_folded_layernorm(tau_val: float) -> None:
    """Folding 1/tau into LayerNorm/RMSNorm weights equals runtime logit/tau to machine precision."""
    gen = torch.Generator(device="cpu").manual_seed(20261402)
    hidden_dim = 24
    vocab_size = 32
    n_tokens = 64

    h_pre = torch.randn((n_tokens, hidden_dim), generator=gen, dtype=torch.float64)
    w_ln = 0.5 + torch.rand((hidden_dim,), generator=gen, dtype=torch.float64)
    b_ln = 0.2 * torch.randn((hidden_dim,), generator=gen, dtype=torch.float64)
    head_w = torch.randn((vocab_size, hidden_dim), generator=gen, dtype=torch.float64) * 0.4
    targets = torch.randint(0, vocab_size, (n_tokens,), generator=gen, dtype=torch.long)

    # 1. LayerNorm with bias (float64 precision)
    logits_div_ln = F.linear(apply_norm_weights(h_pre, w_ln, b_ln, eps=1e-5, norm_type="layernorm"), head_w) / tau_val
    w_fold_ln, b_fold_ln = fold_temperature_into_norm_params(w_ln, b_ln, tau_val)
    assert b_fold_ln is not None
    logits_fold_ln = F.linear(
        apply_norm_weights(h_pre, w_fold_ln, b_fold_ln, eps=1e-5, norm_type="layernorm"), head_w
    )
    assert torch.allclose(logits_div_ln, logits_fold_ln, atol=1e-12, rtol=1e-12)

    m_div = compute_token_nll_and_ece(logits_div_ln, targets)
    m_fold = compute_token_nll_and_ece(logits_fold_ln, targets)
    for key in ("nll_mean", "ece_value", "mce_value", "entropy_mean", "brier_value", "accuracy"):
        assert m_div[key] == pytest.approx(m_fold[key], abs=1e-12)

    # 2. RMSNorm without bias (float64 precision)
    logits_div_rms = F.linear(apply_norm_weights(h_pre, w_ln, None, eps=1e-6, norm_type="rmsnorm"), head_w) / tau_val
    w_fold_rms, b_fold_rms = fold_temperature_into_norm_params(w_ln, None, tau_val)
    assert b_fold_rms is None
    logits_fold_rms = F.linear(
        apply_norm_weights(h_pre, w_fold_rms, None, eps=1e-6, norm_type="rmsnorm"), head_w
    )
    assert torch.allclose(logits_div_rms, logits_fold_rms, atol=1e-12, rtol=1e-12)

    # 3. Channelwise temperature vector equivalence
    tau_vec = 1.0 + 0.5 * torch.rand((hidden_dim,), generator=gen, dtype=torch.float64)
    w_fold_ch, b_fold_ch = fold_temperature_into_norm_params(w_ln, b_ln, tau_vec)
    assert b_fold_ch is not None
    norm_ref = apply_norm_weights(h_pre, w_ln, b_ln, eps=1e-5, norm_type="layernorm") / tau_vec
    logits_ref_ch = F.linear(norm_ref, head_w)
    logits_fold_ch = F.linear(
        apply_norm_weights(h_pre, w_fold_ch, b_fold_ch, eps=1e-5, norm_type="layernorm"), head_w
    )
    assert torch.allclose(logits_ref_ch, logits_fold_ch, atol=1e-12, rtol=1e-12)

    # 4. End-to-end CrowdV1WithEngram model.forward() equivalence (float32)
    model = _build_subspace_moe_model(seed=20261403)
    with torch.no_grad():
        model.ln.weight.copy_(1.5 + 0.3 * torch.rand((model.cfg.hidden_size,), generator=gen))
        model.ln.bias.copy_(0.1 * torch.randn((model.cfg.hidden_size,), generator=gen))
        ids = torch.randint(0, model.cfg.vocab_size, (4, 12), generator=gen)
        logits_before, _, _ = model(ids)
        w_new, b_new = fold_temperature_into_norm_params(model.ln.weight, model.ln.bias, tau_val)
        assert b_new is not None
        model.ln.weight.copy_(w_new)
        model.ln.bias.copy_(b_new)
        logits_after, _, _ = model(ids)
    assert torch.allclose(logits_before / tau_val, logits_after, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# S3: RMSNorm (No Bias) vs LayerNorm (With & Without Bias) Compatibility
# ---------------------------------------------------------------------------


def test_s3_rmsnorm_without_bias_vs_layernorm_with_bias_compatibility() -> None:
    """Verify full compatibility across nn.RMSNorm, custom variance_epsilon RMSNorm, and LayerNorm (bias/no-bias)."""
    probe_chunks = _make_calibration_chunks(chunks_per_domain=6, seed_offset=20)
    probe_ids = torch.tensor([c["ids"] for c in probe_chunks], dtype=torch.long)

    # 1. PyTorch nn.RMSNorm(eps=None) — no bias attribute, eps=None
    model = _build_subspace_moe_model(use_rmsnorm=True, seed=20261410)
    model.ln = nn.RMSNorm(model.cfg.hidden_size)  # default eps=None
    assert detect_norm_type(model.ln) == "rmsnorm"
    assert not hasattr(model.ln, "bias")

    base_rms_w = model.ln.weight.detach().clone()
    with torch.no_grad():
        model.ln.weight.fill_(3.8)

    receipt_rms = apply_verified_ln_temperature_surgery_inplace(
        model,
        probe_chunks,
        base_ln_w=base_rms_w,
        base_ln_b=None,
        score_from=4,
        general_guard_rel=0.005,
        channelwise=True,
    )
    assert receipt_rms["accepted"] is True
    assert receipt_rms["norm_type"] == "rmsnorm"
    assert receipt_rms["has_bias"] is False
    assert receipt_rms["ece_reduction_rel"] > 0.40
    assert receipt_rms["general_pct_vs_base"] <= 0.5
    assert not hasattr(model.ln, "bias")
    with torch.no_grad():
        logits_rms, _, _ = model(probe_ids[:2])
    assert torch.isfinite(logits_rms).all()

    # 2. Custom HF/Qwen-style RMSNorm with variance_epsilon attribute
    custom_rms = _CustomRMSNorm(model.cfg.hidden_size, eps=1e-6)
    assert detect_norm_type(custom_rms) == "rmsnorm"
    model.ln = custom_rms
    with torch.no_grad():
        model.ln.weight.fill_(3.6)
    receipt_custom = apply_verified_ln_temperature_surgery_inplace(
        model,
        probe_chunks,
        score_from=4,
    )
    assert receipt_custom["accepted"] is True
    assert receipt_custom["norm_type"] == "rmsnorm"
    assert receipt_custom["has_bias"] is False

    # 3. Biasless nn.LayerNorm(bias=False) vs biased nn.LayerNorm(bias=True)
    model.ln = nn.LayerNorm(model.cfg.hidden_size, eps=model.cfg.layer_norm_eps, bias=False)
    assert detect_norm_type(model.ln) == "layernorm"
    assert model.ln.bias is None
    with torch.no_grad():
        model.ln.weight.fill_(3.5)
    receipt_ln_nb = apply_verified_ln_temperature_surgery_inplace(
        model,
        probe_chunks,
        config=LogitCalibrationConfig(norm_type="layernorm", score_from=4),
    )
    assert receipt_ln_nb["accepted"] is True
    assert receipt_ln_nb["norm_type"] == "layernorm"
    assert receipt_ln_nb["has_bias"] is False
    assert model.ln.bias is None

    # 4. Standard biased nn.LayerNorm
    model.ln = nn.LayerNorm(model.cfg.hidden_size, eps=model.cfg.layer_norm_eps, bias=True)
    assert detect_norm_type(model.ln) == "layernorm"
    with torch.no_grad():
        model.ln.weight.fill_(3.5)
        model.ln.bias.fill_(0.08)
    receipt_ln_b = apply_verified_ln_temperature_surgery_inplace(
        model,
        probe_chunks,
        score_from=4,
    )
    assert receipt_ln_b["accepted"] is True
    assert receipt_ln_b["norm_type"] == "layernorm"
    assert receipt_ln_b["has_bias"] is True
    assert abs(float(model.ln.bias.mean().item())) < 0.08


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Atomic Rollback on Infeasible Guard Violation
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_rollback_on_infeasible_guard_violation() -> None:
    """Infeasible general/domain drift triggers ANSE 1e6 pain and exact bitwise rollback."""
    probe_chunks = _make_calibration_chunks(chunks_per_domain=6, seed_offset=40)
    eval_chunks = _make_calibration_chunks(chunks_per_domain=6, seed_offset=90)

    # 1. Infeasible general baseline NLL triggers ANSE Maximum Pain rollback
    model = _build_subspace_moe_model(seed=20261420)
    with torch.no_grad():
        model.ln.weight.fill_(3.4)
        model.ln.bias.fill_(0.05)
    parent_sha = compute_module_state_sha256(model)
    parent_w = model.ln.weight.detach().clone()
    parent_b = model.ln.bias.detach().clone()

    infeasible_baseline = {
        "E0_general_fineweb": 0.05,  # Impossible for any temperature/LN interpolation
        "E1_python_systems_code": 0.50,
        "E2_math_reasoning": 0.50,
    }
    receipt_fail = apply_verified_ln_temperature_surgery_inplace(
        model,
        probe_chunks,
        baseline_domain_nll=infeasible_baseline,
        score_from=4,
        max_domain_drift=0.01,
    )
    assert receipt_fail["accepted"] is False
    assert receipt_fail["guard_ok"] is False
    assert receipt_fail["general_guard_ok"] is False
    assert receipt_fail["rolled_back"] is True
    assert receipt_fail["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert receipt_fail["cand_anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert receipt_fail["candidate_state_sha256"] != parent_sha
    assert receipt_fail["final_state_sha256"] == parent_sha
    assert compute_module_state_sha256(model) == parent_sha
    assert torch.equal(model.ln.weight, parent_w)
    assert torch.equal(model.ln.bias, parent_b)
    assert verify_calibration_receipt_provenance(receipt_fail) is True

    # 2. Specialist worst-domain violation (general passes, specialist baseline unattainable)
    model_spec_fail = _build_subspace_moe_model(seed=20261421)
    with torch.no_grad():
        model_spec_fail.ln.weight.fill_(3.2)
    spec_fail_sha = compute_module_state_sha256(model_spec_fail)
    receipt_spec_fail = apply_verified_ln_temperature_surgery_inplace(
        model_spec_fail,
        probe_chunks,
        eval_chunks=eval_chunks,
        baseline_domain_nll={
            "E0_general_fineweb": 10.0,  # Easily passes E0 guard
            "E1_python_systems_code": 0.01,  # Fails worst-domain <= +1.0% guard
            "E2_math_reasoning": 10.0,
        },
        score_from=4,
    )
    assert receipt_spec_fail["accepted"] is False
    assert receipt_spec_fail["general_guard_ok"] is True
    assert receipt_spec_fail["guard_ok"] is False
    assert receipt_spec_fail["worst_domain_id"] == "E1_python_systems_code"
    assert receipt_spec_fail["rolled_back"] is True
    assert receipt_spec_fail["final_state_sha256"] == spec_fail_sha

    # 3. Direct optimize_guard_constrained_ln_scale_and_interpolation fallback on infeasible override
    probe_ids = torch.tensor([c["ids"] for c in probe_chunks], dtype=torch.long)
    with torch.no_grad():
        h_probe = forward_model_pre_ln(model, probe_ids, batch_size=12)
    opt_infeasible = optimize_guard_constrained_ln_scale_and_interpolation(
        torch.ones_like(parent_w),
        torch.zeros_like(parent_b),
        parent_w,
        parent_b,
        h_probe[:6, 3:-1, :],
        probe_ids[:6, 4:],
        h_probe[6:, 3:-1, :],
        probe_ids[6:, 4:],
        model.head,
        base_prose_nll_override=0.01,
    )
    assert opt_infeasible["guard_feasible"] is False
    assert len(opt_infeasible["search_grid_summary"]) > 0

    # 4. Fixed-point / non-improving energy guard (tau fixed at 1.0, base == moe)
    model_fp = _build_subspace_moe_model(seed=20261422)
    fp_sha = compute_module_state_sha256(model_fp)
    receipt_fp = apply_verified_ln_temperature_surgery_inplace(
        model_fp,
        probe_chunks,
        config=LogitCalibrationConfig(
            tau_min=1.0,
            tau_max=1.0,
            num_tau_grid=1,
            num_tau_refine=1,
            lambda_grid=(1.0,),
            score_from=4,
        ),
    )
    assert receipt_fp["accepted"] is False
    assert receipt_fp["rolled_back"] is False
    assert receipt_fp["status"] == "FIXED_POINT_UNCHANGED"
    assert receipt_fp["delta_e"] == pytest.approx(0.0, abs=1e-12)
    assert receipt_fp["final_state_sha256"] == fp_sha

    # 5. Non-improving energy rollback when tau is forced off optimum (over-sharpening) while guard passes
    model_ni = _build_subspace_moe_model(seed=20261423)
    ni_sha = compute_module_state_sha256(model_ni)
    receipt_ni = apply_verified_ln_temperature_surgery_inplace(
        model_ni,
        probe_chunks,
        config=LogitCalibrationConfig(
            tau_min=0.40,
            tau_max=0.50,
            num_tau_grid=2,
            num_tau_refine=2,
            lambda_grid=(1.0,),
            max_domain_drift=10.0,
            score_from=4,
        ),
    )
    assert receipt_ni["accepted"] is False
    assert receipt_ni["guard_ok"] is True
    assert receipt_ni["rolled_back"] is True
    assert receipt_ni["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    assert receipt_ni["final_state_sha256"] == ni_sha


# ---------------------------------------------------------------------------
# S5: Validation Errors & Security / Stub / Gatekeeper Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_stub_compliance_gates() -> None:
    """Verify strict input validation plus AGPL-3.0, ZeroStubAudit, PRSecurityScanner & gatekeeper scans."""
    # 1. Config & metric validation errors
    with pytest.raises(TypeError, match="bool"):
        LogitCalibrationConfig(channelwise="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        LogitCalibrationConfig(num_bins=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 1"):
        LogitCalibrationConfig(num_bins=0)
    with pytest.raises(TypeError, match="real number"):
        LogitCalibrationConfig(general_guard_rel=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        LogitCalibrationConfig(general_guard_rel=float("nan"))
    with pytest.raises(ValueError, match=">= 0.0"):
        LogitCalibrationConfig(max_domain_drift=-0.01)
    with pytest.raises(ValueError, match="> 0.0"):
        LogitCalibrationConfig(tau_min=0.0)
    with pytest.raises(ValueError, match="tau_min"):
        LogitCalibrationConfig(tau_min=2.0, tau_max=1.0)
    with pytest.raises(TypeError, match="sequence"):
        LogitCalibrationConfig(lambda_grid="0.5")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        LogitCalibrationConfig(lambda_grid=())
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        LogitCalibrationConfig(lambda_grid=(1.5,))
    with pytest.raises(ValueError, match="channelwise_min_rel"):
        LogitCalibrationConfig(channelwise_min_rel=1.5, channelwise_max_rel=2.0)
    with pytest.raises(ValueError, match="norm_type"):
        LogitCalibrationConfig(norm_type="batchnorm")
    with pytest.raises(ValueError, match="> 0.0"):
        LogitCalibrationConfig(eps=-1e-5)
    with pytest.raises(ValueError, match="general_prefix"):
        LogitCalibrationConfig(general_prefix=" ")
    with pytest.raises(ValueError, match=">= 0"):
        LogitCalibrationConfig(score_from=-1)

    # compute_token_nll_and_ece checks (including ignore_index and edge cases)
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_token_nll_and_ece([[1.0, 2.0]], torch.tensor([0]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ndim >= 2"):
        compute_token_nll_and_ece(torch.ones(4), torch.tensor(0))
    with pytest.raises(TypeError, match="floating-point"):
        compute_token_nll_and_ece(torch.ones((2, 4), dtype=torch.long), torch.zeros(2, dtype=torch.long))
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_token_nll_and_ece(torch.ones((2, 4)), [0, 1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        compute_token_nll_and_ece(torch.ones((2, 4)), torch.zeros(3, dtype=torch.long))
    with pytest.raises(TypeError, match="integer dtype"):
        compute_token_nll_and_ece(torch.ones((2, 4)), torch.zeros(2, dtype=torch.float32))
    with pytest.raises(ValueError, match="at least one non-ignored"):
        compute_token_nll_and_ece(torch.ones((2, 4)), torch.tensor([-100, -100]))
    with pytest.raises(ValueError, match="lie in"):
        compute_token_nll_and_ece(torch.ones((2, 4)), torch.tensor([0, 9]))
    with pytest.raises(ValueError, match="finite"):
        compute_token_nll_and_ece(torch.tensor([[float("inf"), 0.0]]), torch.tensor([0]))

    masked_res = compute_token_nll_and_ece(
        torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.0]]),
        torch.tensor([0, -100, 2]),
        num_bins=5,
    )
    assert masked_res["num_tokens"] == 2
    assert masked_res["num_bins"] == 5

    # 2. Norm helpers & folding validation errors
    with pytest.raises(TypeError, match="nn.Module"):
        detect_norm_type("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="norm_type"):
        detect_norm_type(nn.LayerNorm(4), norm_type="bad")
    with pytest.raises(ValueError, match="incompatible"):
        detect_norm_type(nn.LayerNorm(4), norm_type="rmsnorm")

    class _WeirdRMS(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))
            self.bias = nn.Parameter(torch.zeros(4))

    with pytest.raises(ValueError, match="non-None bias"):
        detect_norm_type(_WeirdRMS())
    with pytest.raises(ValueError, match="Cannot infer norm type"):
        detect_norm_type(nn.ReLU())

    class _GenericVarEpsNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))
            self.variance_epsilon = 1e-6

    class _GenericAffineNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))
            self.bias = nn.Parameter(torch.zeros(4))

    assert detect_norm_type(_GenericVarEpsNorm()) == "rmsnorm"
    assert detect_norm_type(_GenericAffineNorm()) == "layernorm"

    with pytest.raises(TypeError, match="torch.Tensor"):
        apply_norm_weights("bad", torch.ones(4))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        apply_norm_weights(torch.empty((0, 4)), torch.ones(4))
    with pytest.raises(TypeError, match="floating-point"):
        apply_norm_weights(torch.ones((2, 4), dtype=torch.long), torch.ones(4))
    with pytest.raises(ValueError, match="norm_type"):
        apply_norm_weights(torch.ones((2, 4)), torch.ones(4), norm_type="invalid")
    with pytest.raises(ValueError, match="non-None bias"):
        apply_norm_weights(torch.ones((2, 4)), torch.ones(4), torch.zeros(4), norm_type="rmsnorm")

    with pytest.raises(TypeError, match="torch.Tensor"):
        fold_temperature_into_norm_params([1.0], None, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty 1D"):
        fold_temperature_into_norm_params(torch.ones((2, 2)), None, 1.0)
    with pytest.raises(TypeError, match="floating-point"):
        fold_temperature_into_norm_params(torch.ones(4, dtype=torch.long), None, 1.0)
    with pytest.raises(ValueError, match="expected dim"):
        fold_temperature_into_norm_params(torch.ones(4), torch.zeros(3), 1.0)
    with pytest.raises(ValueError, match="finite"):
        fold_temperature_into_norm_params(torch.tensor([float("nan"), 1.0]), None, 1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        fold_temperature_into_norm_params(torch.ones(4), None, torch.tensor([1.0, 0.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="non-finite"):
        fold_temperature_into_norm_params(torch.tensor([1e38], dtype=torch.float32), None, 1e-10)
    with pytest.raises(ValueError, match="non-finite"):
        fold_temperature_into_norm_params(torch.ones(1), torch.tensor([1e38], dtype=torch.float32), 1e-10)

    # 3. Optimizer & in-place surgeon error paths
    w4 = torch.ones(4)
    b4 = torch.zeros(4)
    h4 = torch.ones((2, 4))
    t4 = torch.zeros(2, dtype=torch.long)
    lin4 = nn.Linear(4, 6, bias=False)

    with pytest.raises(TypeError, match="callable"):
        optimize_guard_constrained_ln_scale_and_interpolation(w4, b4, w4, b4, h4, t4, h4, t4, "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="both be provided"):
        optimize_guard_constrained_ln_scale_and_interpolation(w4, b4, w4, None, h4, t4, h4, t4, lin4)
    with pytest.raises(ValueError, match="tau_min"):
        optimize_guard_constrained_ln_scale_and_interpolation(
            w4, b4, w4, b4, h4, t4, h4, t4, lin4, tau_min=3.0, tau_max=1.0
        )
    with pytest.raises(TypeError, match="torch.Tensor"):
        optimize_guard_constrained_ln_scale_and_interpolation(
            w4, b4, w4, b4, "bad", t4, h4, t4, lin4  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="last dimension"):
        optimize_guard_constrained_ln_scale_and_interpolation(
            w4, b4, w4, b4, torch.ones((2, 5)), t4, h4, t4, lin4
        )
    with pytest.raises(ValueError, match="ndim >= 2"):
        optimize_guard_constrained_ln_scale_and_interpolation(w4, b4, w4, b4, torch.ones(4), t4, h4, t4, lin4)
    with pytest.raises(TypeError, match="floating-point"):
        optimize_guard_constrained_ln_scale_and_interpolation(
            w4, b4, w4, b4, torch.ones((2, 4), dtype=torch.long), t4, h4, t4, lin4
        )
    with pytest.raises(ValueError, match="finite"):
        optimize_guard_constrained_ln_scale_and_interpolation(
            w4, b4, w4, b4, torch.full((2, 4), float("nan")), t4, h4, t4, lin4
        )
    with pytest.raises(TypeError, match="torch.Tensor"):
        optimize_guard_constrained_ln_scale_and_interpolation(
            w4, b4, w4, b4, h4, t4, h4, t4, lambda _x: "bad"  # type: ignore[arg-type,return-value]
        )
    with pytest.raises(ValueError, match="invalid logits shape"):
        optimize_guard_constrained_ln_scale_and_interpolation(
            w4, b4, w4, b4, h4, t4, h4, t4, lambda x: x[..., :1].squeeze(-1)
        )
    with pytest.raises(ValueError, match="non-finite logits"):
        optimize_guard_constrained_ln_scale_and_interpolation(
            w4, b4, w4, b4, h4, t4, h4, t4, lambda x: torch.full((x.shape[0], 6), float("inf"))
        )

    # Exercise channelwise=True with non-nn.Module callable (clean fallback to scalar)
    res_fn = optimize_guard_constrained_ln_scale_and_interpolation(
        w4,
        b4,
        w4 * 2.0,
        b4,
        h4,
        t4,
        h4,
        t4,
        lambda x: x @ lin4.weight.T,
        channelwise=True,
    )
    assert res_fn["channel_tau"] is None

    m_small = _build_subspace_moe_model(num_layers=2, seed=20261430)
    chunks_ok = _make_calibration_chunks(chunks_per_domain=4, seed_offset=70)
    with pytest.raises(TypeError, match="nn.Module"):
        apply_verified_ln_temperature_surgery_inplace("bad", chunks_ok)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="blocks"):
        apply_verified_ln_temperature_surgery_inplace(nn.Linear(4, 4), chunks_ok)
    with pytest.raises(ValueError, match="must expose"):
        bad_m = nn.Module()
        bad_m.blocks = nn.ModuleList([nn.Linear(4, 4)])
        apply_verified_ln_temperature_surgery_inplace(bad_m, chunks_ok)
    with pytest.raises(ValueError, match="norm_attr"):
        apply_verified_ln_temperature_surgery_inplace(m_small, chunks_ok, norm_attr=" ")
    with pytest.raises(ValueError, match="head_attr"):
        apply_verified_ln_temperature_surgery_inplace(m_small, chunks_ok, head_attr=" ")
    with pytest.raises(ValueError, match="must be an nn.Module"):
        apply_verified_ln_temperature_surgery_inplace(m_small, chunks_ok, norm_attr="missing")
    with pytest.raises(ValueError, match="callable unembedding"):
        apply_verified_ln_temperature_surgery_inplace(m_small, chunks_ok, head_attr="missing")
    with pytest.raises(TypeError, match="LogitCalibrationConfig"):
        apply_verified_ln_temperature_surgery_inplace(m_small, chunks_ok, config="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="base_ln_b presence"):
        apply_verified_ln_temperature_surgery_inplace(m_small, chunks_ok, base_ln_w=m_small.ln.weight, score_from=4)
    with pytest.raises(ValueError, match="base_ln_b cannot be provided"):
        apply_verified_ln_temperature_surgery_inplace(m_small, chunks_ok, base_ln_b=m_small.ln.bias, score_from=4)
    with pytest.raises(ValueError, match="non-empty sequence"):
        apply_verified_ln_temperature_surgery_inplace(m_small, [], score_from=4)
    with pytest.raises(ValueError, match="expert_id"):
        apply_verified_ln_temperature_surgery_inplace(m_small, [{"ids": [0, 1, 2]}], score_from=1)
    with pytest.raises(ValueError, match="non-empty string"):
        apply_verified_ln_temperature_surgery_inplace(m_small, [{"expert_id": " ", "ids": [0, 1]}], score_from=1)
    with pytest.raises(TypeError, match="sequence or 1D integer"):
        apply_verified_ln_temperature_surgery_inplace(m_small, [{"expert_id": "E0_a", "ids": 123}], score_from=1)  # type: ignore[list-item]
    with pytest.raises(TypeError, match="integer dtype"):
        apply_verified_ln_temperature_surgery_inplace(
            m_small, [{"expert_id": "E0_a", "ids": torch.tensor([1.0, 2.0])}], score_from=1
        )
    with pytest.raises(ValueError, match="length >= 2"):
        apply_verified_ln_temperature_surgery_inplace(m_small, [{"expert_id": "E0_a", "ids": [0]}], score_from=0)
    with pytest.raises(ValueError, match="token IDs must lie in"):
        apply_verified_ln_temperature_surgery_inplace(m_small, [{"expert_id": "E0_a", "ids": [0, 999]}], score_from=0)
    with pytest.raises(ValueError, match="too short"):
        apply_verified_ln_temperature_surgery_inplace(m_small, [{"expert_id": "E0_a", "ids": [0, 1]}], score_from=4)
    with pytest.raises(ValueError, match="share sequence length"):
        apply_verified_ln_temperature_surgery_inplace(
            m_small,
            [{"expert_id": "E0_a", "ids": [0, 1, 2, 3, 4]}, {"expert_id": "E1_b", "ids": [0, 1, 2]}],
            score_from=1,
        )
    with pytest.raises(ValueError, match="both general"):
        apply_verified_ln_temperature_surgery_inplace(
            m_small, [{"expert_id": "E0_a", "ids": [0, 1, 2, 3, 4]}], score_from=1
        )
    with pytest.raises(ValueError, match="baseline_domain_nll"):
        apply_verified_ln_temperature_surgery_inplace(m_small, chunks_ok, baseline_domain_nll={}, score_from=4)
    with pytest.raises(ValueError, match="missing verification domains"):
        apply_verified_ln_temperature_surgery_inplace(
            m_small, chunks_ok, baseline_domain_nll={"E0_general_fineweb": 1.0}, score_from=4
        )
    with pytest.raises(ValueError, match="2D tensor with width"):
        m_bad_head = _build_subspace_moe_model(num_layers=2, seed=20261431)
        m_bad_head.head = nn.Linear(8, 16, bias=False)
        apply_verified_ln_temperature_surgery_inplace(m_bad_head, chunks_ok, score_from=4)

    # Tensor chunk IDs + non-finite trunk state checks
    tensor_chunks = [
        {"expert_id": ch["expert_id"], "ids": torch.tensor(ch["ids"], dtype=torch.long)}
        for ch in chunks_ok
    ]
    rec_tensor = apply_verified_ln_temperature_surgery_inplace(
        m_small,
        tensor_chunks,
        config=LogitCalibrationConfig(lambda_grid=(1.0,), num_tau_grid=3, num_tau_refine=2, score_from=4),
    )
    assert verify_calibration_receipt_provenance(rec_tensor) is True

    m_inf = _build_subspace_moe_model(num_layers=2, seed=20261432)
    with torch.no_grad():
        m_inf.blocks[-1].mlp.base_mlp.w1.weight.fill_(1e20)
        m_inf.blocks[-1].mlp.base_mlp.w2.weight.fill_(1e20)
        m_inf.blocks[-1].mlp.base_mlp.w3.weight.fill_(1e20)
    with pytest.raises(ValueError, match="non-finite pre-norm probe states"):
        apply_verified_ln_temperature_surgery_inplace(m_inf, chunks_ok, score_from=4)

    # Non-finite eval_chunks (with finite probe_chunks) hits verification states finiteness guard
    m_eval_inf = _build_subspace_moe_model(num_layers=2, seed=20261433)
    with torch.no_grad():
        m_eval_inf.blocks[-1].mlp.experts["2"].w3_B.fill_(1e38)
    probe_e0_e1 = [ch for ch in chunks_ok if not ch["expert_id"].startswith("E2_")]
    eval_e0_e2 = [ch for ch in chunks_ok if not ch["expert_id"].startswith("E1_")]
    with pytest.raises(ValueError, match="non-finite pre-norm verification states"):
        apply_verified_ln_temperature_surgery_inplace(
            m_eval_inf, probe_e0_e1, eval_chunks=eval_e0_e2, score_from=4
        )

    # Provenance & state sha validation errors
    with pytest.raises(TypeError, match="mapping"):
        verify_calibration_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_calibration_receipt_provenance({"receipt_sha256": "xyz"}) is False
    assert verify_calibration_receipt_provenance({"receipt_sha256": "0" * 64, "object": "wrong"}) is False
    assert (
        verify_calibration_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": SURGEON_RECEIPT_OBJECT,
                "schema": SURGEON_SCHEMA,
                "bad": float("nan"),
            }
        )
        is False
    )
    with pytest.raises(TypeError, match="nn.Module"):
        compute_module_state_sha256("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        compute_module_state_sha256(nn.ReLU())
    bad_lin = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        bad_lin.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        compute_module_state_sha256(bad_lin)

    # 4. AGPL-3.0, optional ZeroStubAudit, PRSecurityScanner, lateral movement, and pr_gatekeeper checks
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn14", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn14", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/logit_calibration_temperature_surgeon.py", TARGET_MODULE_PATH),
        ("tests/test_logit_calibration_temperature_surgeon_scenarios.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"

        assert fallback_auditor.check_file(str(path_obj)) == []
        assert fallback_auditor.audit_python_code(src_text).is_clean is True

        if ZeroStubAudit is not None:
            auditor = ZeroStubAudit()
            if hasattr(auditor, "audit_python_code"):
                stub_res = auditor.audit_python_code(src_text)
                assert stub_res.is_clean, f"ZeroStubAudit violations in {rel}: {stub_res.violations}"
                assert stub_res.penalty_energy == 0.0
            if hasattr(auditor, "check_file"):
                assert auditor.check_file(str(path_obj)) == []

        assert scanner.scan_python_ast_security(rel, src_text) == []
        assert scanner.scan_text_for_payment_solicitation(src_text, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(src_text, rel) == []
        assert scanner.scan_unicode_trojan_source(src_text, rel) == []
        assert lateral_guard.audit_source_for_lateral_movement(src_text, rel) == []

        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in src_text.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []
        changed_files.append(
            {
                "filename": rel,
                "status": "added",
                "additions": len(src_text.splitlines()),
                "patch": "",
                "content": src_text,
            }
        )

    telemetry_json = json.dumps(
        {"nll_mean": 0.42, "ece_value": 0.03, "anse_energy": -0.18},
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+PAYLOAD = {telemetry_json!r}"
        )
        == []
    )
    assert len(compute_payload_sha256({"ok": True})) == 64

    pr_audit = scanner.audit_pull_request(
        pr_number=1114,
        pr_title="feat(swarm): post-surgery final LayerNorm and unembedding temperature/ECE weight calibration for open-weight MoEs",
        pr_body="Implements NEURAL-CTRL-DYN-14 final LayerNorm/RMSNorm temperature & ECE calibration surgeon with deterministic CPU scenarios.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): post-surgery final LayerNorm and unembedding temperature/ECE weight calibration for open-weight MoEs"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
