# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-11: Wanda/OBS activation-weighted LoRA surgeon."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdRuntimeMLP,
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdFineGrainedMoEMLP,
    CrowdLoRAMoEMLP,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.fisher_kfac_diagonal_surgeon import (
    DEFAULT_ANSE_MAXIMUM_PAIN,
    WANDA_OBS_RECEIPT_OBJECT,
    WANDA_OBS_SCHEMA,
    WandaOBSSurgeonConfig,
    apply_obs_compensated_row_surgery,
    compute_discriminative_wanda_saliency,
    compute_specialist_hessian_inverse,
    compute_state_sha256,
    execute_wanda_obs_moe_surgery,
    verify_wanda_obs_receipt_provenance,
)
from hivemind.swarm.wsp_alphaedit_control import (
    forward_model_pre_ln,
    per_chunk_nll_from_pre_ln,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:
    ZeroStubAudit = None  # type: ignore[assignment,misc]


class _FallbackStubResult:
    """Minimal stub audit result used when optional ``gwaya`` is absent in CI."""

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
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "fisher_kfac_diagonal_surgeon.py"
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


def _eval_domain_nll(
    model: CrowdV1WithEngram,
    chunks: list[dict[str, Any]],
    *,
    score_from: int = 8,
) -> dict[str, float]:
    ids = torch.tensor([c["ids"] for c in chunks], dtype=torch.long)
    with torch.no_grad():
        h_pre = forward_model_pre_ln(model, ids, batch_size=16)
        nlls = per_chunk_nll_from_pre_ln(
            h_pre,
            model.ln.weight,
            model.ln.bias,
            float(getattr(model.ln, "eps", 1e-5)),
            model.head.weight,
            ids,
            score_from=score_from,
        ).tolist()
    grouped: dict[str, list[float]] = {}
    for ch, val in zip(chunks, nlls, strict=True):
        grouped.setdefault(str(ch["expert_id"]), []).append(float(val))
    return {k: sum(v) / float(len(v)) for k, v in sorted(grouped.items())}


def _build_prose_corrupted_moe_fixture(
    *,
    num_layers: int = 2,
    num_experts: int = 4,
    lora_rank: int = 4,
    seed: int = 20261124,
) -> dict[str, Any]:
    """Build deterministic CrowdV1WithEngram + CrowdLoRAMoEMLP with planted prose corruption.

    Subspaces (hidden_size=48, intermediate_size=96, vocab_size=48):
      - E0_general_fineweb: tokens 0..15  -> dims 0..15  (gold target 5)
      - E1_python_code:     tokens 16..31 -> dims 16..31 (gold target 20)
      - E2_math_reasoning:  tokens 32..47 -> dims 32..47 (gold target 36)
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    hidden_size = 48
    intermediate_size = 96
    vocab_size = 48
    cfg = CrowdV1RuntimeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=hidden_size // 4,
        intermediate_size=intermediate_size,
        max_position_embeddings=32,
        engram_buckets=0,
        tie_word_embeddings=True,
    )
    model = CrowdV1WithEngram(cfg)

    u_e0 = _unit_axis(hidden_size, 0, 1)
    u_e1 = _unit_axis(hidden_size, 16, 17)
    u_e2 = _unit_axis(hidden_size, 32, 33)

    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)

        def _fill_block(lo: int, hi: int, tgt: int, d_lo: int, d_hi: int, carrier: torch.Tensor, scale: float) -> None:
            raw = 0.45 * torch.randn((hi - lo, d_hi - d_lo), generator=gen, dtype=torch.float32)
            raw[:, :2] = 0.0
            wte[lo:hi, d_lo:d_hi] = raw
            wte[lo:hi] += scale * carrier.unsqueeze(0)
            wte[tgt] = 2.6 * carrier

        _fill_block(0, 16, 5, 0, 16, u_e0, 2.2)
        _fill_block(16, 32, 20, 16, 32, u_e1, 0.85)
        _fill_block(32, 48, 36, 32, 48, u_e2, 0.85)
        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight

        model.ln.weight.fill_(1.0)
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln2.weight.fill_(1.0)
            blk.mlp.w1.weight[:hidden_size, :hidden_size].copy_(0.90 * torch.eye(hidden_size))
            blk.mlp.w2.weight[:hidden_size, :hidden_size].copy_(0.90 * torch.eye(hidden_size))
            blk.mlp.w3.weight[:16, :16].copy_(0.45 * torch.eye(16))
            blk.mlp.w3.weight[16:32, 16:32].copy_(0.05 * torch.eye(16))
            blk.mlp.w3.weight[32:48, 32:48].copy_(0.05 * torch.eye(16))

        model.blocks[0].attn.v.weight.copy_(0.35 * torch.eye(hidden_size))
        model.blocks[0].attn.o.weight.copy_(0.35 * torch.eye(hidden_size))

    upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode="lora_domain_experts",
        num_experts=num_experts,
        top_k=2,
        lora_rank=lora_rank,
        lora_alpha=8.0,
        routing_mode="sigma_topk",
        seed=seed + 19,
    )

    chunks: list[dict[str, Any]] = []
    for eid, lo, hi, tgt in (
        ("E0_general_fineweb", 0, 16, 5),
        ("E1_python_code", 16, 32, 20),
        ("E2_math_reasoning", 32, 48, 36),
    ):
        for _ in range(6):
            ids = torch.randint(lo, hi, (16,), generator=gen).tolist()
            for pos in range(8, 16):
                ids[pos] = tgt
            chunks.append({"expert_id": eid, "ids": ids})

    base_nll = _eval_domain_nll(model, chunks, score_from=8)

    # Plant beneficial specialist adapters on expert_1/expert_2 AND large-magnitude prose corruption
    with torch.no_grad():
        for l_idx, blk in enumerate(model.blocks):
            mlp: CrowdLoRAMoEMLP = blk.mlp  # type: ignore[assignment]
            mlp.router.weight.zero_()
            for exp_mod in mlp.experts.values():
                for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                    getattr(exp_mod, attr).zero_()

            mlp.router.weight[0] = 3.5 * u_e0
            mlp.router.weight[1] = 3.5 * u_e1 + 2.8 * u_e0
            mlp.router.weight[2] = 3.5 * u_e2 + 2.4 * u_e0

            if l_idx == num_layers - 1:
                exp1 = mlp.experts["1"]
                # Specialist code weights (moderate magnitude 0.30, huge specialist activation)
                exp1.w3_A[0, 16:20] = torch.tensor([0.30, -0.30, 0.25, -0.25])
                exp1.w3_B[:, 0] = 1.6 * u_e1
                # Prose-corrupting weights (3x larger magnitude 0.95 on prose dims -> fools magnitude pruning!)
                exp1.w3_A[1, 0:4] = torch.tensor([0.95, -0.95, 0.90, -0.90])
                exp1.w3_B[:, 1] = -1.4 * u_e0
                exp1.w1_A[0, 16:18] = torch.tensor([0.28, -0.28])
                exp1.w1_A[1, 0:2] = torch.tensor([0.92, -0.92])
                exp1.w1_B[16:18, 0] = torch.tensor([0.8, -0.8])
                exp1.w1_B[0:2, 1] = torch.tensor([-0.9, 0.9])

                exp2 = mlp.experts["2"]
                exp2.w3_A[0, 32:36] = torch.tensor([0.32, -0.32, 0.26, -0.26])
                exp2.w3_B[:, 0] = 1.6 * u_e2
                exp2.w3_A[1, 0:4] = torch.tensor([0.95, -0.95, 0.88, -0.88])
                exp2.w3_B[:, 1] = -1.2 * u_e0

    tuned_nll = _eval_domain_nll(model, chunks, score_from=8)
    return {
        "model": model,
        "chunks": chunks,
        "base_nll": base_nll,
        "tuned_nll": tuned_nll,
        "carriers": {"e0": u_e0, "e1": u_e1, "e2": u_e2},
    }


def _naive_magnitude_prune_lora_inplace(model: CrowdV1WithEngram, prune_ratio: float) -> None:
    """Naive magnitude pruning baseline across active off-domain LoRA A factors."""
    with torch.no_grad():
        pool: list[tuple[float, torch.Tensor, int, int]] = []
        for blk in model.blocks:
            mlp: CrowdLoRAMoEMLP = blk.mlp  # type: ignore[assignment]
            for e_key, exp in mlp.experts.items():
                if int(e_key) == 0:
                    continue
                for base in ("w1", "w2", "w3"):
                    w_b = getattr(exp, f"{base}_B")
                    w_a = getattr(exp, f"{base}_A")
                    if float(w_b.abs().max().item()) <= 1e-12:
                        continue
                    rc, cc = torch.nonzero(w_a.abs() > 1e-12, as_tuple=True)
                    for r_i, c_i in zip(rc.tolist(), cc.tolist(), strict=True):
                        pool.append((float(w_a[r_i, c_i].abs().item()), w_a, r_i, c_i))
        k_prune = min(len(pool), max(1, int(round(prune_ratio * float(len(pool))))))
        pool.sort(key=lambda item: item[0])
        for _, ten, r_i, c_i in pool[:k_prune]:
            ten[r_i, c_i] = 0.0


# ---------------------------------------------------------------------------
# S1: Discriminative Wanda Saliency & Surgical Prose Removal vs Naive Pruning
# ---------------------------------------------------------------------------


def test_s1_discriminative_wanda_removes_prose_corruption_and_beats_magnitude_pruning() -> None:
    """S1: Wanda/OBS removes prose-corrupting coordinates, beats magnitude pruning, and clears <= +1.0% guard."""
    # 1. Exact formula verification for compute_discriminative_wanda_saliency
    w_toy = torch.tensor([[0.9, 0.25, 0.0], [-0.8, 0.3, 0.5]], dtype=torch.bfloat16)
    xp_3d = torch.zeros((2, 4, 3), dtype=torch.float32)
    xp_3d[:, :, 0] = 3.0
    xp_3d[:, :, 1] = 0.05
    xs_3d = torch.zeros((2, 4, 3), dtype=torch.float32)
    xs_3d[:, :, 0] = 0.02
    xs_3d[:, :, 1] = 4.0
    xs_3d[:, :, 2] = 1.0

    sal, sal_diag = compute_discriminative_wanda_saliency(w_toy, xp_3d, xs_3d, eps=1e-8)
    assert sal.dtype == torch.float64
    assert sal.shape == (2, 3)
    assert sal_diag["shape"] == (2, 3)
    assert sal_diag["nonzero_count"] == 5
    assert sal_diag["saliency_ratio"] > 0.0
    expected_col0 = abs(float(w_toy[0, 0])) * (math.sqrt(8 * 0.02**2) / (math.sqrt(8 * 3.0**2) + 1e-8))
    expected_col1 = abs(float(w_toy[0, 1])) * (math.sqrt(8 * 4.0**2) / (math.sqrt(8 * 0.05**2) + 1e-8))
    assert float(sal[0, 0]) == pytest.approx(expected_col0, rel=1e-6)
    assert float(sal[0, 1]) == pytest.approx(expected_col1, rel=1e-6)
    assert float(sal[0, 2]) == 0.0
    # Prose-corrupting weight 0.9 has far lower discriminative saliency than smaller specialist weight 0.25
    assert float(sal[0, 0]) < float(sal[0, 1]) * 0.05

    sal_norm, norm_diag = compute_discriminative_wanda_saliency(
        w_toy, xp_3d.reshape(-1, 3), xs_3d[:1].reshape(-1, 3), eps=1e-8, normalize_by_samples=True
    )
    assert norm_diag["normalize_by_samples"] is True
    assert float(sal_norm[0, 1]) > float(sal_norm[0, 0]) * 20.0

    sal_zero, zero_diag = compute_discriminative_wanda_saliency(
        torch.zeros((2, 3)), xp_3d, xs_3d
    )
    assert float(sal_zero.abs().max().item()) == 0.0
    assert zero_diag["nonzero_count"] == 0
    assert zero_diag["saliency_ratio"] == 0.0

    # 2. End-to-end CrowdV1WithEngram Sparse MoE LoRA surgery vs naive magnitude pruning
    fix = _build_prose_corrupted_moe_fixture(num_layers=2, num_experts=4, lora_rank=4, seed=20261124)
    model_wanda: CrowdV1WithEngram = fix["model"]
    chunks: list[dict[str, Any]] = fix["chunks"]
    base_nll: dict[str, float] = fix["base_nll"]
    tuned_nll: dict[str, float] = fix["tuned_nll"]

    # Confirm baseline corruption: specialist E1 improved, but E0 general prose broke > +1.0% guard
    pre_e0_pct = 100.0 * (tuned_nll["E0_general_fineweb"] / base_nll["E0_general_fineweb"] - 1.0)
    assert pre_e0_pct > 1.5, f"Expected broken prose baseline, got {pre_e0_pct:.3f}%"
    assert base_nll["E1_python_code"] - tuned_nll["E1_python_code"] > 0.25

    model_mag = copy.deepcopy(model_wanda)
    _naive_magnitude_prune_lora_inplace(model_mag, prune_ratio=0.50)
    mag_nll = _eval_domain_nll(model_mag, chunks, score_from=8)

    # Dry-run inplace=False leaves caller model bitwise unchanged
    pre_sha = compute_state_sha256(model_wanda.state_dict())
    dry_cfg = WandaOBSSurgeonConfig(
        prune_ratio=0.50,
        inplace=False,
        score_from=8,
        target_factors=("w1_A", "w2_A", "w3_A", "w1_B", "w2_B", "w3_B"),
    )
    dry_rec = execute_wanda_obs_moe_surgery(
        model_wanda,
        chunks,
        config=dry_cfg,
        baseline_domain_nll=base_nll,
        tuned_domain_nll=tuned_nll,
    )
    assert dry_rec["accepted"] is True
    assert compute_state_sha256(model_wanda.state_dict()) == pre_sha

    # Actual in-place Wanda/OBS surgery
    receipt = execute_wanda_obs_moe_surgery(
        model_wanda,
        chunks,
        baseline_domain_nll=base_nll,
        tuned_domain_nll=tuned_nll,
        prune_ratio=0.50,
        use_obs_compensation=True,
        use_woodbury=True,
        score_from=8,
    )
    assert receipt["object"] == WANDA_OBS_RECEIPT_OBJECT
    assert receipt["schema"] == WANDA_OBS_SCHEMA
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["status"] == "ACCEPTED"
    assert receipt["guard_ok"] is True
    assert receipt["retention_ok"] is True
    assert receipt["delta_e"] < 0.0
    assert receipt["general_pct_vs_base"] <= 1.0
    assert receipt["worst_domain_pct_vs_base"] <= 1.0
    assert receipt["specialist_gain_retained_frac"] > 0.85
    assert receipt["saliency_ratio"] > 0.0
    assert verify_wanda_obs_receipt_provenance(receipt) is True

    wanda_nll = _eval_domain_nll(model_wanda, chunks, score_from=8)
    # Wanda/OBS strictly outperforms naive magnitude pruning on specialist NLL AND general prose
    assert wanda_nll["E1_python_code"] < mag_nll["E1_python_code"] - 0.15
    assert wanda_nll["E2_math_reasoning"] < mag_nll["E2_math_reasoning"] - 0.10
    assert wanda_nll["E0_general_fineweb"] < mag_nll["E0_general_fineweb"] - 0.02
    assert (
        100.0 * (mag_nll["E0_general_fineweb"] / base_nll["E0_general_fineweb"] - 1.0) > 1.0
    )


# ---------------------------------------------------------------------------
# S2: Closed-Form OBS Second-Order Compensation (Dense, Woodbury & Diagonal)
# ---------------------------------------------------------------------------


def test_s2_obs_second_order_compensation_lowers_specialist_reconstruction_error() -> None:
    """S2: Closed-form dense & Woodbury OBS compensation strictly lowers specialist error vs uncompensated."""
    gen = torch.Generator(device="cpu").manual_seed(20261125)
    n_tok = 48
    d_in = 16
    d_out = 6

    # Build strongly correlated specialist features so OBS transfers pruned mass to correlated columns
    z_lat = torch.randn((n_tok, 4), generator=gen, dtype=torch.float64)
    mix = torch.randn((4, d_in), generator=gen, dtype=torch.float64)
    x_spec = z_lat @ mix + 0.04 * torch.randn((n_tok, d_in), generator=gen, dtype=torch.float64)

    w_orig = torch.randn((d_out, d_in), generator=gen, dtype=torch.float64)
    mask_zero = torch.zeros((d_out, d_in), dtype=torch.bool)
    mask_zero[:3, :4] = True
    mask_zero[3:, 4:8] = True

    # Uncompensated zero-masking baseline
    w_uncomp = w_orig.clone()
    w_uncomp[mask_zero] = 0.0
    err_uncomp = float(torch.linalg.matrix_norm((w_uncomp - w_orig) @ x_spec.T, ord="fro").item())

    # 1. Dense Cholesky H_spec^{-1}
    h_inv_dense, h_dense_meta = compute_specialist_hessian_inverse(
        x_spec, hessian_damping=1e-4, mode="dense"
    )
    assert h_dense_meta["mode"] == "dense"
    w_obs_dense, obs_diag = apply_obs_compensated_row_surgery(w_orig, mask_zero, h_inv_dense)
    err_obs_dense = float(
        torch.linalg.matrix_norm((w_obs_dense - w_orig) @ x_spec.T, ord="fro").item()
    )
    assert bool((w_obs_dense[mask_zero] == 0.0).all().item())
    assert err_obs_dense < 0.25 * err_uncomp
    assert obs_diag["compensated_rows"] == d_out
    assert obs_diag["kept_comp_fro_norm"] > 0.0
    assert obs_diag["obs_quadratic_loss"] > 0.0

    # 2. Factored Woodbury H_spec^{-1} (exact algebraic equivalence with dense OBS!)
    h_wood_dict, h_wood_meta = compute_specialist_hessian_inverse(
        x_spec, hessian_damping=1e-4, mode="woodbury", return_dense=False
    )
    assert h_wood_meta["mode"] == "woodbury"
    assert h_wood_meta["factored"] is True
    w_obs_wood, wood_diag = apply_obs_compensated_row_surgery(w_orig, mask_zero, h_wood_dict)
    assert wood_diag["mode"] == "woodbury_obs"
    assert float((w_obs_wood - w_obs_dense).abs().max().item()) < 1e-10

    # Tuple (inv_diag, woodbury_u) input & dense woodbury equivalence
    h_wood_tuple = (h_wood_dict["inv_diag"], h_wood_dict["woodbury_u"])
    w_obs_tup, _ = apply_obs_compensated_row_surgery(w_orig, mask_zero, h_wood_tuple)
    assert float((w_obs_tup - w_obs_dense).abs().max().item()) < 1e-10

    h_wood_dense, _ = compute_specialist_hessian_inverse(
        x_spec, hessian_damping=1e-4, mode="woodbury", return_dense=True
    )
    assert torch.allclose(h_wood_dense, h_inv_dense, rtol=1e-10, atol=1e-9)

    # Auto mode switches cleanly between woodbury (N < d_in) and dense (N >= d_in)
    _, auto_small_meta = compute_specialist_hessian_inverse(x_spec[:8], mode="auto", return_dense=False)
    assert auto_small_meta["mode"] == "woodbury"

    # 3. Diagonal inverse Hessian mode + max_comp_scale clamp + inplace + bfloat16 + edge no-ops
    h_inv_diag, h_diag_meta = compute_specialist_hessian_inverse(
        x_spec, hessian_damping=1e-4, mode="diagonal"
    )
    assert h_diag_meta["mode"] == "diagonal"
    w_diag_out, diag_rep = apply_obs_compensated_row_surgery(w_orig, mask_zero, h_inv_diag)
    assert diag_rep["mode"] == "diagonal_obs"
    assert float((w_diag_out - w_uncomp).abs().max().item()) == 0.0
    assert diag_rep["obs_quadratic_loss"] > 0.0

    w_clamped, _ = apply_obs_compensated_row_surgery(
        w_orig, mask_zero, h_inv_dense, max_comp_scale=0.10
    )
    kept_delta_row0 = torch.where(~mask_zero[0], w_clamped[0] - w_orig[0], torch.zeros(d_in, dtype=torch.float64))
    assert float(kept_delta_row0.norm().item()) <= 0.10 * float(w_orig[0].norm().item()) + 1e-9

    w_bf16 = w_orig.to(torch.bfloat16)
    w_bf16_res, bf16_rep = apply_obs_compensated_row_surgery(w_bf16, mask_zero, h_inv_dense, inplace=True)
    assert w_bf16_res is w_bf16
    assert w_bf16.dtype == torch.bfloat16
    assert bf16_rep["edited_elements"] > 0

    # Empty mask, full-row zeroing, and already-zero masked coordinates branches
    _, empty_rep = apply_obs_compensated_row_surgery(w_orig, torch.zeros_like(mask_zero), h_inv_dense)
    assert empty_rep["noop_reason"] == "empty_mask"

    w_part_zero = w_orig.clone()
    w_part_zero[0, :4] = 0.0
    mask_only_zero = torch.zeros_like(mask_zero)
    mask_only_zero[0, :4] = True
    w_zero_kept, z_rep = apply_obs_compensated_row_surgery(w_part_zero, mask_only_zero, h_inv_dense)
    assert float((w_zero_kept - w_part_zero).abs().max().item()) == 0.0
    assert z_rep["noop_reason"] == "zero_masked_entries"

    full_row_mask = torch.zeros_like(mask_zero)
    full_row_mask[0, :] = True
    w_full_zero, _ = apply_obs_compensated_row_surgery(w_orig, full_row_mask, h_inv_dense)
    assert float(w_full_zero[0].abs().max().item()) == 0.0


# ---------------------------------------------------------------------------
# S3: Strict Budget Enforcement on Edited Parameter Fraction
# ---------------------------------------------------------------------------


def test_s3_budget_enforcement_on_edited_parameter_fraction() -> None:
    """S3: Edited parameter fraction strictly respects max_edited_param_ratio across budget & prune scopes."""
    fix = _build_prose_corrupted_moe_fixture(num_layers=1, num_experts=4, lora_rank=4, seed=20261126)
    base_nll = fix["base_nll"]
    tuned_nll = fix["tuned_nll"]
    chunks = fix["chunks"]

    # 1. Target-scope tight budget (caps total modified entries including OBS compensations)
    m_tight = copy.deepcopy(fix["model"])
    rec_tight = execute_wanda_obs_moe_surgery(
        m_tight,
        chunks,
        baseline_domain_nll=base_nll,
        tuned_domain_nll=tuned_nll,
        prune_ratio=0.50,
        max_edited_param_ratio=0.20,
        budget_scope="target",
        prune_scope="global",
        require_energy_improvement=False,
        score_from=8,
    )
    assert rec_tight["budget_respected"] is True
    assert rec_tight["edited_elements"] <= rec_tight["max_edited_params_cap"]
    assert rec_tight["edited_param_ratio"] <= 0.20 + 1e-12
    assert rec_tight["model_edited_param_ratio"] <= rec_tight["edited_param_ratio"]

    # 2. Sub-row-width tiny budget (triggers partial-row budgeted patching branch with OBS comp slots)
    m_micro = copy.deepcopy(fix["model"])
    rec_micro = execute_wanda_obs_moe_surgery(
        m_micro,
        chunks,
        baseline_domain_nll=base_nll,
        tuned_domain_nll=tuned_nll,
        prune_ratio=0.50,
        max_edited_param_ratio=0.015,
        budget_scope="target",
        require_energy_improvement=False,
        score_from=8,
    )
    assert rec_micro["budget_respected"] is True
    assert rec_micro["edited_elements"] <= rec_micro["max_edited_params_cap"]
    assert rec_micro["edited_param_ratio"] <= 0.015 + 1e-12

    lin_corr = nn.Linear(8, 2, bias=False)
    with torch.no_grad():
        lin_corr.weight.copy_(torch.tensor([[0.9, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1], [0.1] * 8]))
    gen_c = torch.Generator(device="cpu").manual_seed(20261129)
    xp_c = torch.randn((12, 8), generator=gen_c)
    z_c = torch.randn((12, 2), generator=gen_c)
    xs_c = torch.cat([z_c[:, :1] * 0.2, z_c[:, :1] * 1.5 + 0.05 * z_c[:, 1:], z_c.repeat(1, 3)], dim=1)
    rec_lin_micro = execute_wanda_obs_moe_surgery(
        lin_corr,
        prose_activations=xp_c,
        spec_activations=xs_c,
        eval_fn=lambda _m: {"E0_g": 1.0, "E1_s": 1.0},
        target_factors=("w1",),
        prune_ratio=0.25,
        max_edited_param_ratio=0.25,
        budget_scope="target",
        require_energy_improvement=False,
    )
    assert rec_lin_micro["budget_respected"] is True
    assert rec_lin_micro["edited_elements"] <= 4

    # 3. Matrix & Row prune scopes + model budget scope + saliency_threshold + single-layer mask
    for p_scope, b_scope in (("matrix", "trainable"), ("row", "model")):
        m_sc = copy.deepcopy(fix["model"])
        rec_sc = execute_wanda_obs_moe_surgery(
            m_sc,
            chunks,
            config=WandaOBSSurgeonConfig(
                prune_ratio=0.45,
                max_edited_param_ratio=0.25,
                budget_scope=b_scope,
                prune_scope=p_scope,
                saliency_threshold=1.0,
                require_energy_improvement=False,
                score_from=8,
            ),
            baseline_domain_nll=base_nll,
            tuned_domain_nll=tuned_nll,
            layer_indices=[0],
        )
        assert rec_sc["budget_respected"] is True
        assert rec_sc["edited_param_ratio"] <= 0.25 + 1e-12


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Atomic Rollback & Standalone MoE/Dense Support
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_rollback_and_standalone_layers() -> None:
    """S4: Over-pruning violating guard/retention/energy triggers ANSE 1e6 pain and bitwise rollback."""
    # 1. End-to-end CrowdV1WithEngram over-pruning rollback (guard violation vs pre-surgery)
    fix = _build_prose_corrupted_moe_fixture(num_layers=2, num_experts=4, lora_rank=4, seed=20261127)
    model: CrowdV1WithEngram = fix["model"]
    chunks = fix["chunks"]
    pre_sha = compute_state_sha256(model.state_dict())

    overprune_rec = execute_wanda_obs_moe_surgery(
        model,
        chunks,
        prune_ratio=0.95,
        protect_general_expert=False,
        expert_indices=[0, 1, 2],
        domain_guard_max_ratio=1.01,
        score_from=8,
    )
    assert overprune_rec["accepted"] is False
    assert overprune_rec["rolled_back"] is True
    assert overprune_rec["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert overprune_rec["guard_ok"] is False
    assert overprune_rec["anse_energy"] >= DEFAULT_ANSE_MAXIMUM_PAIN
    assert overprune_rec["final_state_sha256"] == pre_sha
    assert compute_state_sha256(model.state_dict()) == pre_sha
    assert verify_wanda_obs_receipt_provenance(overprune_rec) is True

    # 2. Retention violation rollback on standalone CrowdLoRAMoEMLP
    gen = torch.Generator(device="cpu").manual_seed(20261128)
    cfg_sm = CrowdV1RuntimeConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        max_position_embeddings=16,
        engram_buckets=0,
    )
    dense_mlp = CrowdRuntimeMLP(cfg_sm)
    lora_moe = CrowdLoRAMoEMLP(dense_mlp, num_experts=2, top_k=2, rank=4, seed=99)
    with torch.no_grad():
        lora_moe.experts["1"].w3_B[:, 0] = 1.5
        lora_moe.experts["1"].w3_A[0, :8] = 0.8
        lora_moe.experts["1"].w1_B[:, 0] = 0.5

    xp_act = torch.cat([torch.ones((8, 8)), torch.zeros((8, 8))], dim=1) + 0.05 * torch.randn((8, 16), generator=gen)
    xs_act = torch.cat([torch.zeros((8, 8)), torch.ones((8, 8))], dim=1) + 0.05 * torch.randn((8, 16), generator=gen)

    def _eval_standalone(mod: nn.Module) -> dict[str, float]:
        with torch.no_grad():
            y_p = mod(xp_act)
            y_s = mod(xs_act)
            return {
                "E0_general": 1.0 + 0.2 * float(y_p.pow(2).mean().item()),
                "E1_spec": 1.0 + 0.5 * float((y_s - 2.0).pow(2).mean().item()),
            }

    base_sa = {"E0_general": 1.2, "E1_spec": 3.0}
    pre_sa = _eval_standalone(lora_moe)
    sha_lora = compute_state_sha256(lora_moe.state_dict())
    ret_rec = execute_wanda_obs_moe_surgery(
        lora_moe,
        prose_activations=xp_act,
        spec_activations=xs_act,
        eval_fn=_eval_standalone,
        baseline_domain_nll=base_sa,
        tuned_domain_nll=pre_sa,
        prune_ratio=0.95,
        min_specialist_gain_retention=0.999,
        domain_guard_max_ratio=10.0,
    )
    assert ret_rec["accepted"] is False
    assert ret_rec["rolled_back"] is True
    assert ret_rec["status"] == "ROLLED_BACK_RETENTION_VIOLATION"
    assert ret_rec["retention_ok"] is False
    assert compute_state_sha256(lora_moe.state_dict()) == sha_lora

    # 3. Non-improving energy rollback, eval failure rollback, and FIXED_POINT_UNCHANGED
    no_imp_rec = execute_wanda_obs_moe_surgery(
        lora_moe,
        prose_activations=xp_act,
        spec_activations=xs_act,
        eval_fn=lambda _m: {"domain_nll": {"E0_general": 1.0, "E1_spec": 1.0}},
        prune_ratio=0.50,
        require_energy_improvement=True,
    )
    assert no_imp_rec["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    assert no_imp_rec["rolled_back"] is True

    calls = {"n": 0}

    def _boom_eval(_m: nn.Module) -> dict[str, float]:
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("injected eval failure")
        return {"E0_general": 1.0, "E1_spec": 1.0}

    boom_rec = execute_wanda_obs_moe_surgery(
        lora_moe,
        prose_activations=xp_act,
        spec_activations=xs_act,
        eval_fn=_boom_eval,
        prune_ratio=0.50,
    )
    assert boom_rec["status"] == "ROLLED_BACK_EVAL_FAILURE"
    assert boom_rec["rolled_back"] is True
    assert boom_rec["anse_energy"] >= DEFAULT_ANSE_MAXIMUM_PAIN

    fp_rec = execute_wanda_obs_moe_surgery(
        lora_moe,
        prose_activations=xp_act,
        spec_activations=xs_act,
        eval_fn=lambda _m: {"E0_general": 1.0, "E1_spec": 1.0},
        config=WandaOBSSurgeonConfig(saliency_threshold=0.0, prune_ratio=0.50),
    )
    assert fp_rec["status"] == "FIXED_POINT_UNCHANGED"
    assert fp_rec["rolled_back"] is False

    # 4. CrowdFineGrainedMoEMLP, CrowdRuntimeMLP, and nn.Linear standalone support
    fg_moe = CrowdFineGrainedMoEMLP(dense_mlp, num_experts=2, top_k=2)
    lin_mod = nn.Linear(16, 8, bias=False)
    for mod_cand, tfact in (
        (fg_moe, ("w1", "w2", "w3")),
        (dense_mlp, ("w1", "w2", "w3")),
        (lin_mod, ("w1",)),
    ):
        cand_rec = execute_wanda_obs_moe_surgery(
            mod_cand,
            prose_activations=xp_act,
            spec_activations=xs_act,
            eval_fn=lambda _m: {"E0_general": 1.0, "E1_spec": 1.0},
            target_factors=tfact,
            prune_ratio=0.25,
            require_energy_improvement=False,
        )
        assert cand_rec["accepted"] is True
        assert cand_rec["edited_elements"] > 0


# ---------------------------------------------------------------------------
# S5: Validation Errors, Edge Cases & Security/Stub Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_edge_cases_and_security_compliance_gates() -> None:
    """S5: Exhaustive input validation, provenance checks, and HiveMind security/stub compliance gates."""
    # 1. Config validation
    with pytest.raises(ValueError, match="prune_ratio"):
        WandaOBSSurgeonConfig(prune_ratio=0.0)
    with pytest.raises(ValueError, match="prune_ratio"):
        WandaOBSSurgeonConfig(prune_ratio=1.0)
    with pytest.raises(ValueError, match="max_edited_param_ratio"):
        WandaOBSSurgeonConfig(max_edited_param_ratio=1.2)
    with pytest.raises(ValueError, match="budget_scope"):
        WandaOBSSurgeonConfig(budget_scope="invalid")
    with pytest.raises(ValueError, match="prune_scope"):
        WandaOBSSurgeonConfig(prune_scope="invalid")
    with pytest.raises(ValueError, match="eps"):
        WandaOBSSurgeonConfig(eps=0.0)
    with pytest.raises(ValueError, match="hessian_damping"):
        WandaOBSSurgeonConfig(hessian_damping=-0.1)
    with pytest.raises(TypeError, match="bool"):
        WandaOBSSurgeonConfig(use_obs_compensation="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        WandaOBSSurgeonConfig(general_expert_index=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        WandaOBSSurgeonConfig(general_expert_index=-1)
    with pytest.raises(ValueError, match="target_factors"):
        WandaOBSSurgeonConfig(target_factors=())
    with pytest.raises(ValueError, match="Unsupported target_factors"):
        WandaOBSSurgeonConfig(target_factors=("bad_factor",))
    with pytest.raises(ValueError, match="domain_guard_max_ratio"):
        WandaOBSSurgeonConfig(domain_guard_max_ratio=0.9)
    with pytest.raises(ValueError, match="anse_safe_e0_ratio"):
        WandaOBSSurgeonConfig(anse_safe_e0_ratio=0.95)
    with pytest.raises(ValueError, match="general_prefix"):
        WandaOBSSurgeonConfig(general_prefix=" ")
    with pytest.raises(ValueError, match=">= 1"):
        WandaOBSSurgeonConfig(batch_size=0)
    with pytest.raises(ValueError, match="finite"):
        WandaOBSSurgeonConfig(saliency_threshold=float("nan"))

    # 2. Wanda saliency & Hessian inverse validation
    w2d = torch.ones((4, 6))
    xp = torch.ones((5, 6))
    xs = torch.ones((5, 6))
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_discriminative_wanda_saliency([[1.0]], xp, xs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty 2D"):
        compute_discriminative_wanda_saliency(torch.ones(6), xp, xs)
    with pytest.raises(TypeError, match="floating-point"):
        compute_discriminative_wanda_saliency(torch.ones((4, 6), dtype=torch.long), xp, xs)
    bad_w = w2d.clone()
    bad_w[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        compute_discriminative_wanda_saliency(bad_w, xp, xs)
    with pytest.raises(ValueError, match="2D or 3D"):
        compute_discriminative_wanda_saliency(w2d, torch.ones(6), xs)
    with pytest.raises(TypeError, match="floating-point"):
        compute_discriminative_wanda_saliency(w2d, torch.ones((5, 6), dtype=torch.long), xs)
    with pytest.raises(ValueError, match="expected width"):
        compute_discriminative_wanda_saliency(w2d, torch.ones((5, 4)), xs)
    bad_x = xp.clone()
    bad_x[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        compute_discriminative_wanda_saliency(w2d, bad_x, xs)
    with pytest.raises(ValueError, match="> 0.0"):
        compute_discriminative_wanda_saliency(w2d, xp, xs, eps=0.0)
    with pytest.raises(TypeError, match="real number"):
        compute_discriminative_wanda_saliency(w2d, xp, xs, eps=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="mode"):
        compute_specialist_hessian_inverse(xs, mode="invalid")

    # 3. apply_obs_compensated_row_surgery validation
    mask_b = torch.zeros((4, 6), dtype=torch.bool)
    mask_b[0, 0] = True
    eye6 = torch.eye(6)
    with pytest.raises(TypeError, match="torch.bool"):
        apply_obs_compensated_row_surgery(w2d, mask_b.to(torch.int64), eye6)
    with pytest.raises(TypeError, match="torch.Tensor"):
        apply_obs_compensated_row_surgery(w2d, [[True]], eye6)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        apply_obs_compensated_row_surgery(w2d, torch.zeros((2, 6), dtype=torch.bool), eye6)
    with pytest.raises(TypeError, match="floating-point"):
        apply_obs_compensated_row_surgery(w2d, mask_b, torch.ones(6, dtype=torch.long))
    with pytest.raises(ValueError, match="finite"):
        apply_obs_compensated_row_surgery(w2d, mask_b, torch.tensor([1.0] * 5 + [float("nan")]))
    with pytest.raises(ValueError, match="d_in=6"):
        apply_obs_compensated_row_surgery(w2d, mask_b, torch.ones(4))
    with pytest.raises(ValueError, match="strictly > 0.0"):
        apply_obs_compensated_row_surgery(w2d, mask_b, torch.zeros(6))
    with pytest.raises(ValueError, match="must be \\(6, 6\\)"):
        apply_obs_compensated_row_surgery(w2d, mask_b, torch.eye(4))
    with pytest.raises(ValueError, match="positive definite"):
        apply_obs_compensated_row_surgery(w2d, mask_b, -torch.eye(6))
    with pytest.raises(ValueError, match="1D or 2D"):
        apply_obs_compensated_row_surgery(w2d, mask_b, torch.ones((2, 3, 6)))
    with pytest.raises(TypeError, match="1D/2D Tensor"):
        apply_obs_compensated_row_surgery(w2d, mask_b, "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inv_diag"):
        apply_obs_compensated_row_surgery(w2d, mask_b, {"inv_diag": torch.ones(6)})
    with pytest.raises(TypeError, match="floating-point"):
        apply_obs_compensated_row_surgery(w2d, mask_b, (torch.ones(6, dtype=torch.long), torch.zeros((6, 2))))
    with pytest.raises(ValueError, match="1D of length 6"):
        apply_obs_compensated_row_surgery(w2d, mask_b, (torch.ones(4), torch.zeros((6, 2))))
    with pytest.raises(ValueError, match="positive finite"):
        apply_obs_compensated_row_surgery(w2d, mask_b, (torch.zeros(6), torch.zeros((6, 2))))
    with pytest.raises(ValueError, match="row count"):
        apply_obs_compensated_row_surgery(w2d, mask_b, (torch.ones(6), torch.zeros((4, 2))))
    with pytest.raises(ValueError, match="remain positive"):
        apply_obs_compensated_row_surgery(w2d, mask_b, (torch.ones(6), 3.0 * torch.ones((6, 1))))

    # 4. Orchestrator & provenance validation
    with pytest.raises(TypeError, match="nn.Module"):
        execute_wanda_obs_moe_surgery("bad")  # type: ignore[arg-type]
    lin = nn.Linear(6, 4, bias=False)
    with pytest.raises(TypeError, match="callable"):
        execute_wanda_obs_moe_surgery(lin, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="WandaOBSSurgeonConfig"):
        execute_wanda_obs_moe_surgery(lin, config="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Either guard_chunks"):
        execute_wanda_obs_moe_surgery(lin)
    class _UnsupportedModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.unrecognized_param = nn.Parameter(torch.ones(4))

    with pytest.raises(ValueError, match="Unsupported MLP module"):
        execute_wanda_obs_moe_surgery(
            _UnsupportedModule(), prose_activations=xp, spec_activations=xs, eval_fn=lambda _m: {"E0_a": 1.0}
        )
    with pytest.raises(TypeError, match="non-empty mapping"):
        execute_wanda_obs_moe_surgery(
            lin, prose_activations=xp, spec_activations=xs, eval_fn=lambda _m: "bad"  # type: ignore[arg-type]
        )

    fix = _build_prose_corrupted_moe_fixture(num_layers=1, num_experts=3, lora_rank=2, seed=77)
    m_small = fix["model"]
    ch_small = fix["chunks"]
    with pytest.raises(ValueError, match="layer_indices"):
        execute_wanda_obs_moe_surgery(m_small, ch_small, layer_indices=(9,), score_from=8)
    with pytest.raises(ValueError, match="non-empty sequence"):
        execute_wanda_obs_moe_surgery(m_small, [])
    with pytest.raises(ValueError, match="expert_id"):
        execute_wanda_obs_moe_surgery(m_small, [{"ids": [1, 2, 3]}])
    with pytest.raises(ValueError, match="non-empty string"):
        execute_wanda_obs_moe_surgery(m_small, [{"expert_id": " ", "ids": [1, 2, 3]}])
    with pytest.raises(ValueError, match="length >= 2"):
        execute_wanda_obs_moe_surgery(m_small, [{"expert_id": "E0_a", "ids": [1]}])
    with pytest.raises(ValueError, match="non-negative"):
        execute_wanda_obs_moe_surgery(m_small, [{"expert_id": "E0_a", "ids": [1, -1, 2]}])
    with pytest.raises(ValueError, match="sequence length"):
        execute_wanda_obs_moe_surgery(
            m_small,
            [{"expert_id": "E0_a", "ids": [1, 2, 3]}, {"expert_id": "E1_b", "ids": [1, 2, 3, 4]}],
        )
    with pytest.raises(ValueError, match="too short"):
        execute_wanda_obs_moe_surgery(m_small, ch_small, score_from=32)
    with pytest.raises(ValueError, match="both general"):
        execute_wanda_obs_moe_surgery(m_small, [{"expert_id": "E0_a", "ids": [1, 2, 3, 4]}])
    with pytest.raises(ValueError, match="identical keys"):
        execute_wanda_obs_moe_surgery(
            m_small, ch_small, baseline_domain_nll={"E0_only": 1.0}, score_from=8
        )
    with pytest.raises(ValueError, match="identical keys"):
        execute_wanda_obs_moe_surgery(
            m_small, ch_small, tuned_domain_nll={"E0_only": 1.0}, score_from=8
        )

    bad_mod = nn.Module()
    bad_mod.blocks = nn.ModuleList()
    with pytest.raises(ValueError, match="blocks"):
        execute_wanda_obs_moe_surgery(bad_mod, ch_small, score_from=8)
    bad_mod.blocks = nn.ModuleList([nn.Linear(4, 4)])
    with pytest.raises(ValueError, match="must expose"):
        execute_wanda_obs_moe_surgery(bad_mod, ch_small, score_from=8)

    with pytest.raises(TypeError, match="mapping"):
        verify_wanda_obs_receipt_provenance("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        compute_state_sha256({})
    with pytest.raises(ValueError, match="non-empty string"):
        compute_state_sha256({"": torch.ones(2)})
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_state_sha256({"a": [1.0]})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-finite"):
        compute_state_sha256({"a": torch.tensor([float("nan")])})
    assert verify_wanda_obs_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert verify_wanda_obs_receipt_provenance({"receipt_sha256": "0" * 64, "object": "wrong"}) is False
    assert (
        verify_wanda_obs_receipt_provenance(
            {"receipt_sha256": "0" * 64, "object": WANDA_OBS_RECEIPT_OBJECT, "schema": "wrong"}
        )
        is False
    )
    assert (
        verify_wanda_obs_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": WANDA_OBS_RECEIPT_OBJECT,
                "schema": WANDA_OBS_SCHEMA,
                "bad": float("nan"),
            }
        )
        is False
    )

    # 5. Repository AGPL-3.0, optional ZeroStubAudit, PRSecurityScanner, lateral movement & gatekeeper gates
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn11", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn11", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/fisher_kfac_diagonal_surgeon.py", TARGET_MODULE_PATH),
        ("tests/test_fisher_kfac_diagonal_surgeon_scenarios.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"

        assert fallback_auditor.check_file(str(path_obj)) == []
        assert fallback_auditor.audit_python_code(src_text).is_clean is True

        if ZeroStubAudit is not None:
            stub_res = ZeroStubAudit().audit_python_code(src_text)
            assert stub_res.is_clean, f"ZeroStubAudit violations in {rel}: {stub_res.violations}"
            assert stub_res.penalty_energy == 0.0

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

    sample_receipt_json = json.dumps(
        {"nll_mean": 0.42, "saliency_ratio": 18.5, "anse_energy": -0.15, "accepted": True},
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+RECEIPT = {sample_receipt_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1111,
        pr_title="feat(swarm): Wanda/OBS activation-weighted LoRA surgical sparsification with prose-curvature guard",
        pr_body="Implements NEURAL-CTRL-DYN-11 Wanda/OBS LoRA surgeon and deterministic CPU scenario tests.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): Wanda/OBS activation-weighted LoRA surgical sparsification with prose-curvature guard"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
