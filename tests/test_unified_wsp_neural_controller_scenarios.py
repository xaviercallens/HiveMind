# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-04: unified 5-stage WeightSpaceController."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:
    ZeroStubAudit = None  # type: ignore[assignment]

from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdLoRAMoEMLP,
    extract_moe_trainable_buckets,
    load_moe_trainable_buckets,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.unified_wsp_neural_controller import (
    ANSE_MAXIMUM_PAIN,
    CONTROLLER_RECEIPT_OBJECT,
    CONTROLLER_SCHEMA,
    STAGE_ORDER,
    WeightSpaceController,
    WeightSpaceControllerConfig,
    compute_buckets_sha256,
    evaluate_multidomain_anse_metrics,
    partition_loeo_experts,
    run_stage1_preflight_sanitization,
    run_stage2_loeo_and_layer_attribution,
    run_stage3_wsp_surgery,
    run_stage4_fine_layernorm_calibration,
    run_stage5_thermodynamic_anse_gate,
    verify_controller_receipt_provenance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "unified_wsp_neural_controller.py"
TEST_FILE_PATH = Path(__file__).resolve()
EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."


def _load_module_from_path(name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_subspace_crowd_moe(
    *,
    num_layers: int = 4,
    num_experts: int = 4,
    top_k: int = 2,
    lora_rank: int = 4,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    vocab_size: int = 48,
    seed: int = 20261124,
) -> tuple[CrowdV1WithEngram, dict[str, torch.Tensor]]:
    """Build deterministic multi-layer CrowdV1WithEngram + CrowdLoRAMoEMLP with orthogonal E0/E1/E2 subspaces.

    Tokens:
      - ``0..15`` (E0 general prose): dims ``0..15`` with target token ``5``.
      - ``16..31`` (E1 code): dims ``16..23`` with target token ``20``.
      - ``32..47`` (E2 math): dims ``24..31`` with target token ``36``.
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
        max_position_embeddings=64,
        engram_buckets=0,
        tie_word_embeddings=True,
    )
    model = CrowdV1WithEngram(cfg)

    u_e0 = torch.zeros(16, dtype=torch.float32)
    u_e0[0] = 1.0 / math.sqrt(2.0)
    u_e0[1] = -1.0 / math.sqrt(2.0)

    u_e1 = torch.zeros(8, dtype=torch.float32)
    u_e1[0] = 1.0 / math.sqrt(2.0)
    u_e1[1] = -1.0 / math.sqrt(2.0)

    u_e2 = torch.zeros(8, dtype=torch.float32)
    u_e2[0] = 1.0 / math.sqrt(2.0)
    u_e2[1] = -1.0 / math.sqrt(2.0)

    prose_carrier = torch.zeros(hidden_size, dtype=torch.float32)
    prose_carrier[:16] = u_e0
    e1_carrier = torch.zeros(hidden_size, dtype=torch.float32)
    e1_carrier[16:24] = u_e1
    e2_carrier = torch.zeros(hidden_size, dtype=torch.float32)
    e2_carrier[24:32] = u_e2

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        p_raw = 0.20 * torch.randn((16, 16), generator=gen, dtype=torch.float32)
        p_raw[:, :2] = 0.0
        p_raw = p_raw - p_raw.mean(dim=1, keepdim=True) + 2.2 * u_e0.unsqueeze(0)
        p_raw[5] = 2.6 * u_e0

        e1_raw = 0.20 * torch.randn((16, 8), generator=gen, dtype=torch.float32)
        e1_raw[:, :2] = 0.0
        e1_raw = e1_raw - e1_raw.mean(dim=1, keepdim=True) + 2.2 * u_e1.unsqueeze(0)
        e1_raw[20 - 16] = 2.6 * u_e1

        e2_raw = 0.20 * torch.randn((16, 8), generator=gen, dtype=torch.float32)
        e2_raw[:, :2] = 0.0
        e2_raw = e2_raw - e2_raw.mean(dim=1, keepdim=True) + 2.2 * u_e2.unsqueeze(0)
        e2_raw[36 - 32] = 2.6 * u_e2

        wte[:16, :16] = p_raw
        wte[16:32, 16:24] = e1_raw
        wte[32:48, 24:32] = e2_raw
        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight

        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0)
            blk.ln2.bias.zero_()
            # Block-diagonal SwiGLU base_mlp exposing dim 0 (E0), dim 16 (E1), and dim 24 (E2)
            blk.mlp.w1.weight[:hidden_size, :hidden_size].copy_(1.5 * torch.eye(hidden_size))
            blk.mlp.w2.weight[:hidden_size, :hidden_size].copy_(1.5 * torch.eye(hidden_size))
            blk.mlp.w3.weight[:16, :16].copy_(0.03 * torch.randn((16, 16), generator=gen))
            blk.mlp.w3.weight[16:24, 16:24].copy_(0.03 * torch.randn((8, 8), generator=gen))
            blk.mlp.w3.weight[24:32, 24:32].copy_(0.03 * torch.randn((8, 8), generator=gen))

    upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode="lora_domain_experts",
        num_experts=num_experts,
        top_k=top_k,
        lora_rank=lora_rank,
        lora_alpha=8.0,
        routing_mode="sigma_topk",
        seed=seed + 19,
    )
    carriers = {"e0": prose_carrier, "e1": e1_carrier, "e2": e2_carrier}
    return model, carriers


def _make_disjoint_domain_chunks(
    *,
    seq_len: int = 16,
    score_from: int = 8,
    domains: tuple[str, ...] = ("E0_general_fineweb", "E1_code", "E2_math"),
    chunks_per_domain: int = 8,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    """Build deterministic non-overlapping domain token chunks."""
    gen = torch.Generator(device="cpu").manual_seed(9000 + seed_offset)
    chunks: list[dict[str, Any]] = []
    for d_idx, dom in enumerate(domains):
        for r_idx in range(chunks_per_domain):
            if dom.startswith("E0_"):
                ids = torch.randint(0, 16, (seq_len,), generator=gen).tolist()
                target_tok = 5
            elif dom.startswith("E1_"):
                ids = torch.randint(16, 32, (seq_len,), generator=gen).tolist()
                target_tok = 20
            else:
                ids = torch.randint(32, 48, (seq_len,), generator=gen).tolist()
                target_tok = 36
            for pos in range(score_from, seq_len):
                ids[pos] = target_tok
            chunks.append(
                {
                    "expert_id": dom,
                    "dataset_id": dom,
                    "row": seed_offset + 100 * d_idx + r_idx,
                    "ids": ids,
                }
            )
    return chunks


def _plant_s1_triple_regression(
    model: CrowdV1WithEngram,
    carriers: dict[str, torch.Tensor],
) -> None:
    """Plant simultaneous rogue expert_3 (||w1_B||=42.5) + LN drift + late-layer prose interference."""
    prose_v = carriers["e0"]
    e1_v = carriers["e1"]
    e2_v = carriers["e2"]

    with torch.no_grad():
        for l_idx, blk in enumerate(model.blocks):
            mlp: CrowdLoRAMoEMLP = blk.mlp  # type: ignore[assignment]
            mlp.router.weight.zero_()
            for exp_mod in mlp.experts.values():
                for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                    getattr(exp_mod, attr).zero_()

            # Helpful general expert_0 on E0 across layers 0..1 (gentle early syntax/anchor)
            if l_idx < 2:
                mlp.router.weight[0] = 2.4 * prose_v
                mlp.experts["0"].w3_A[0, 0] = 0.02
                mlp.experts["0"].w3_B[:, 0] = 0.025 * prose_v

            # Specialist experts 1 & 2 deliver huge specialist gains in layers 2 & 3,
            # PLUS a planted rank-1 prose-interfering singular mode and leaky prose router gates!
            if l_idx in (2, 3):
                mlp.router.weight[0] = 0.8 * prose_v
                mlp.router.weight[1] = 2.8 * e1_v + (2.6 if l_idx == 2 else 0.9) * prose_v
                mlp.router.weight[2] = 2.8 * e2_v + (2.6 if l_idx == 3 else 0.9) * prose_v

                # Expert 1: mode 0 = E1 specialist (w3_A row 0 on ch 16), mode 1 = E0 interference (ch 0)
                exp1 = mlp.experts["1"]
                exp1.w3_A[0, 16] = 0.16
                exp1.w3_B[:, 0] = 0.70 * e1_v
                exp1.w3_A[1, 0] = 0.06
                exp1.w3_B[:, 1] = -0.22 * prose_v

                # Expert 2: mode 0 = E2 specialist (w3_A row 0 on ch 24), mode 1 = E0 interference (ch 0)
                exp2 = mlp.experts["2"]
                exp2.w3_A[0, 24] = 0.16
                exp2.w3_B[:, 0] = 0.70 * e2_v
                exp2.w3_A[1, 0] = 0.08
                exp2.w3_B[:, 1] = -0.45 * prose_v

            # Rogue unscored high-norm expert_3 with exact global ||w1_B|| = 42.5 across 4 layers,
            # active on E0 in late layers 2 & 3!
            per_layer_norm = 42.5 / math.sqrt(4.0)
            w1_shape = mlp.experts["3"].w1_B.shape
            mlp.experts["3"].w1_B.fill_(per_layer_norm / math.sqrt(float(w1_shape[0] * w1_shape[1])))
            mlp.experts["3"].w1_A[0, 0] = 0.01
            if l_idx in (2, 3):
                mlp.router.weight[3] = 2.8 * prose_v
                mlp.experts["3"].w3_A[0, 0] = 0.06
                mlp.experts["3"].w3_B[:, 0] = -0.25 * prose_v
            else:
                mlp.router.weight[3] = -10.0 * prose_v - 10.0 * e1_v - 10.0 * e2_v

        # Final LayerNorm prose drift (pushes against E0 target subspace, severely harming E0 NLL)
        model.ln.weight[:16].fill_(0.35)
        model.ln.bias[:16].copy_(-1.60 * prose_v[:16])


# ---------------------------------------------------------------------------
# S1: Full 5-Stage Recovery (>92% Specialist Gain Retained, <= +1.0% Guard)
# ---------------------------------------------------------------------------


def test_s1_full_five_stage_recovery_retains_over_92pct_specialist_gain() -> None:
    """Recover a prose-regressing MoE with rogue expert_3, LN drift, and late-layer interference."""
    model, carriers = _make_subspace_crowd_moe(num_layers=4, num_experts=4, top_k=2, lora_rank=4, seed=101)
    start_buckets = {k: v.detach().cpu().clone() for k, v in extract_moe_trainable_buckets(model, num_router_fragments=2).items()}

    _plant_s1_triple_regression(model, carriers)
    c2_buckets = {k: v.detach().cpu().clone() for k, v in extract_moe_trainable_buckets(model, num_router_fragments=2).items()}

    probe_chunks = _make_disjoint_domain_chunks(chunks_per_domain=8, seed_offset=0)
    eval_chunks = _make_disjoint_domain_chunks(chunks_per_domain=8, seed_offset=1000)

    cfg = WeightSpaceControllerConfig(
        score_from=8,
        batch_size=8,
        loeo_chunks_per_domain=6,
        protect_early_layers_below=2,
        min_harm_nats=0.005,
        svd_tau=1.20,
        svd_eta=6.0,
        svd_max_modes_damped=2,
        router_beta_suppress=2.0,
        router_beta_sink=0.75,
        router_min_domain_samples=4,
    )
    controller = WeightSpaceController(cfg)
    result = controller.run(
        model,
        start_buckets,
        probe_chunks,
        tuned_buckets=c2_buckets,
        eval_chunks=eval_chunks,
    )

    assert result.accepted is True
    assert result.rolled_back is False
    assert result.status == "ACCEPTED"
    assert result.receipt["object"] == CONTROLLER_RECEIPT_OBJECT
    assert result.receipt["schema"] == CONTROLLER_SCHEMA
    assert verify_controller_receipt_provenance(result.receipt) is True

    stages = result.receipt["stages"]
    assert 3 in stages["stage_1_preflight"]["high_norm_warning_experts"]
    assert 3 in stages["stage_2_attribution"]["rogue_core_sink_experts"]
    assert 3 in stages["stage_2_attribution"]["core_sink_experts"]
    assert 1 in stages["stage_2_attribution"]["marginal_specialist_experts"]
    assert 2 in stages["stage_2_attribution"]["marginal_specialist_experts"]
    assert stages["stage_2_attribution"]["protect_general_expert"] is True
    assert stages["stage_2_attribution"]["protected_layers"] == [0, 1]
    assert set(stages["stage_2_attribution"]["selected_layers"]) == {2, 3}

    # Early non-rogue experts (e.g., expert_0, expert_1, expert_2) in layers 0..1 stay untouched; rogue expert_3 is zeroed
    for l_idx in range(4):
        exp3 = model.blocks[l_idx].mlp.experts["3"]
        assert torch.equal(exp3.w1_B, torch.zeros_like(exp3.w1_B))
        assert torch.equal(exp3.w2_B, torch.zeros_like(exp3.w2_B))
        assert torch.equal(exp3.w3_B, torch.zeros_like(exp3.w3_B))

    assert stages["stage_3_surgery"]["total_svd_damped_modes"] > 0
    assert stages["stage_3_surgery"]["router_layers_applied"] == 2
    assert stages["stage_4_layernorm"]["grid_points"] == 21
    assert stages["stage_4_layernorm"]["best_lambda_ln"] == pytest.approx(0.0)

    s5 = stages["stage_5_anse_gate"]
    assert s5["parent_anse_energy"] > ANSE_MAXIMUM_PAIN * 0.5
    assert s5["candidate_anse_energy"] < 0.0
    assert s5["delta_e"] < 0.0
    assert s5["worst_domain_guard_ok"] is True
    assert s5["e0_pct_vs_base"] <= 1.0
    assert s5["worst_domain_pct_vs_base"] <= 1.0
    assert s5["spec_gain_retained_frac"] > 0.92


# ---------------------------------------------------------------------------
# S2: Core-vs-Marginal LOEO Partition Outperforming Naive Full Sinking
# ---------------------------------------------------------------------------


def test_s2_core_vs_marginal_loeo_partition_outperforms_naive_full_sinking() -> None:
    """Preserve marginal specialist capacity via QR-SVD + multi-gate while sinking zero-cost core ghost sinks."""
    model, carriers = _make_subspace_crowd_moe(num_layers=4, num_experts=4, top_k=2, lora_rank=4, seed=202)
    start_buckets = {k: v.detach().cpu().clone() for k, v in extract_moe_trainable_buckets(model, num_router_fragments=2).items()}

    prose_v = carriers["e0"]
    e1_v = carriers["e1"]
    e2_v = carriers["e2"]

    with torch.no_grad():
        for l_idx, blk in enumerate(model.blocks):
            mlp: CrowdLoRAMoEMLP = blk.mlp  # type: ignore[assignment]
            mlp.router.weight.zero_()
            for exp_mod in mlp.experts.values():
                for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                    getattr(exp_mod, attr).zero_()
            if l_idx in (2, 3):
                # Core ghost sink expert_0: harms E0, zero specialist value (delta_spec <= 0.010)
                mlp.router.weight[0] = 2.6 * prose_v
                mlp.experts["0"].w3_A[0, 0] = 0.05
                mlp.experts["0"].w3_B[:, 0] = -0.35 * prose_v

                # Core specialist expert_1: clean moderate E1 gain
                mlp.router.weight[1] = 2.8 * e1_v
                mlp.experts["1"].w3_A[0, 16] = 0.04
                mlp.experts["1"].w3_B[:, 0] = 0.24 * e1_v

                # Marginal specialist expert_2: delivers real E2 specialist gain (overall delta_spec ~ +0.022)
                # while leaking onto E0 (delta_e0 < -0.0015), falling inside naive 0.030 sink threshold!
                mlp.router.weight[2] = 2.6 * e2_v + 2.5 * prose_v
                mlp.experts["2"].w3_A[0, 24] = 0.03
                mlp.experts["2"].w3_B[:, 0] = 0.16 * e2_v
                mlp.experts["2"].w3_A[1, 0] = 0.04
                mlp.experts["2"].w3_B[:, 1] = -0.25 * prose_v

    c2_buckets = {k: v.detach().cpu().clone() for k, v in extract_moe_trainable_buckets(model, num_router_fragments=2).items()}
    probe_chunks = _make_disjoint_domain_chunks(chunks_per_domain=8, seed_offset=200)
    eval_chunks = _make_disjoint_domain_chunks(chunks_per_domain=8, seed_offset=1200)

    # Naive full sinking (core_sink_max_spec_delta=0.050 sinks BOTH expert_0 AND marginal expert_2, with zero SVD/gate recovery)
    naive_cfg = WeightSpaceControllerConfig(
        score_from=8,
        batch_size=8,
        loeo_chunks_per_domain=6,
        protect_early_layers_below=2,
        min_harm_nats=0.005,
        core_sink_max_spec_delta=0.050,
        svd_eta=0.0,
        router_beta_suppress=0.0,
        router_beta_sink=0.0,
    )
    naive_res = WeightSpaceController(naive_cfg).run(
        model, start_buckets, probe_chunks, tuned_buckets=c2_buckets, eval_chunks=eval_chunks
    )
    assert naive_res.accepted is True
    assert set(naive_res.receipt["stages"]["stage_2_attribution"]["core_sink_experts"]) == {0, 2}

    # Unified core-vs-marginal partition (core <= 0.010 sinks ONLY expert_0; expert_2 preserved via QR-SVD + multi-gate)
    smart_cfg = WeightSpaceControllerConfig(
        score_from=8,
        batch_size=8,
        loeo_chunks_per_domain=6,
        protect_early_layers_below=2,
        min_harm_nats=0.005,
        core_sink_max_spec_delta=0.010,
        svd_tau=1.20,
        svd_eta=6.0,
        svd_max_modes_damped=2,
        router_beta_suppress=2.0,
        router_beta_sink=0.75,
        router_min_domain_samples=4,
    )
    smart_res = WeightSpaceController(smart_cfg).run(
        model, start_buckets, probe_chunks, tuned_buckets=c2_buckets, eval_chunks=eval_chunks
    )
    assert smart_res.accepted is True
    s2_smart = smart_res.receipt["stages"]["stage_2_attribution"]
    assert s2_smart["core_sink_experts"] == [0]
    assert 2 in s2_smart["marginal_specialist_experts"]

    naive_s5 = naive_res.receipt["stages"]["stage_5_anse_gate"]
    smart_s5 = smart_res.receipt["stages"]["stage_5_anse_gate"]
    assert smart_s5["worst_domain_guard_ok"] is True
    assert smart_s5["spec_gain_retained_frac"] > naive_s5["spec_gain_retained_frac"] + 0.05
    assert smart_s5["spec_gain_retained_frac"] > 0.90
    assert smart_s5["candidate_anse_energy"] < naive_s5["candidate_anse_energy"]
    assert (
        smart_s5["candidate_metrics"]["per_domain"]["E2_math"]["cand_nll"]
        < naive_s5["candidate_metrics"]["per_domain"]["E2_math"]["cand_nll"] - 0.02
    )


# ---------------------------------------------------------------------------
# S3: Automatic Atomic Bitwise Rollback on ANSE Acceptance Gate Failure
# ---------------------------------------------------------------------------


def test_s3_automatic_atomic_bitwise_rollback_on_anse_gate_failure() -> None:
    """Roll back bitwise to parent state when candidate fails worst-domain guard or Delta E < 0."""
    model, carriers = _make_subspace_crowd_moe(num_layers=4, num_experts=4, top_k=2, lora_rank=4, seed=303)
    start_buckets = {k: v.detach().cpu().clone() for k, v in extract_moe_trainable_buckets(model, num_router_fragments=2).items()}

    _plant_s1_triple_regression(model, carriers)
    c2_buckets = {k: v.detach().cpu().clone() for k, v in extract_moe_trainable_buckets(model, num_router_fragments=2).items()}
    probe_chunks = _make_disjoint_domain_chunks(chunks_per_domain=8, seed_offset=300)
    eval_chunks = _make_disjoint_domain_chunks(chunks_per_domain=8, seed_offset=1300)

    # 3A: Guard violation rollback (forced_lambda_ln=1.0 keeps severe LN drift -> worst_domain_pct > +1.0%)
    guard_fail_cfg = WeightSpaceControllerConfig(
        score_from=8,
        batch_size=8,
        loeo_chunks_per_domain=6,
        protect_early_layers_below=2,
        forced_lambda_ln=1.0,
        router_min_domain_samples=4,
    )
    load_moe_trainable_buckets(model, c2_buckets, num_router_fragments=2)
    parent_state_3a = {k: v.detach().clone() for k, v in model.state_dict().items()}
    res_guard_fail = WeightSpaceController(guard_fail_cfg).run(
        model,
        start_buckets,
        probe_chunks,
        eval_chunks=eval_chunks,
    )
    assert res_guard_fail.accepted is False
    assert res_guard_fail.rolled_back is True
    assert res_guard_fail.status == "ROLLED_BACK_GUARD_VIOLATION"
    assert res_guard_fail.receipt["parent_buckets_sha256"] == res_guard_fail.receipt["final_buckets_sha256"]
    assert verify_controller_receipt_provenance(res_guard_fail.receipt) is True
    for k, orig_t in c2_buckets.items():
        assert torch.equal(res_guard_fail.buckets[k], orig_t)
    for k, orig_p in parent_state_3a.items():
        assert torch.equal(model.state_dict()[k], orig_p)

    # 3B: Energy non-improving rollback (candidate clears guard, but fails Delta E < -min_energy_improvement)
    # Build a clean guard-compliant parent first, then require an unachievable energy drop
    ok_cfg = WeightSpaceControllerConfig(
        score_from=8,
        batch_size=8,
        loeo_chunks_per_domain=6,
        protect_early_layers_below=2,
        min_harm_nats=0.005,
        svd_eta=6.0,
        router_beta_suppress=2.0,
        router_min_domain_samples=4,
    )
    repaired = WeightSpaceController(ok_cfg).run(
        model, start_buckets, probe_chunks, tuned_buckets=c2_buckets, eval_chunks=eval_chunks
    )
    assert repaired.accepted is True

    non_improving_cfg = WeightSpaceControllerConfig(
        score_from=8,
        batch_size=8,
        loeo_chunks_per_domain=6,
        protect_early_layers_below=2,
        min_harm_nats=0.0,
        svd_eta=0.5,
        router_beta_suppress=0.5,
        forced_lambda_ln=0.0,
        min_energy_improvement=0.50,
        router_min_domain_samples=4,
    )
    res_non_imp = WeightSpaceController(non_improving_cfg).run(
        model,
        start_buckets,
        probe_chunks,
        tuned_buckets=repaired.buckets,
        eval_chunks=eval_chunks,
    )
    assert res_non_imp.accepted is False
    assert res_non_imp.rolled_back is True
    assert res_non_imp.status == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    for k, orig_t in repaired.buckets.items():
        assert torch.equal(res_non_imp.buckets[k], orig_t)


# ---------------------------------------------------------------------------
# S4: Deterministic Idempotency & SHA-256 Provenance Verification
# ---------------------------------------------------------------------------


def test_s4_deterministic_idempotency_and_sha256_provenance_verification() -> None:
    """Verify exact repeatability, fixed-point idempotency, and tamper-evident SHA-256 stage receipts."""
    model, carriers = _make_subspace_crowd_moe(num_layers=4, num_experts=4, top_k=2, lora_rank=4, seed=404)
    start_buckets = {k: v.detach().cpu().clone() for k, v in extract_moe_trainable_buckets(model, num_router_fragments=2).items()}
    _plant_s1_triple_regression(model, carriers)
    c2_buckets = {k: v.detach().cpu().clone() for k, v in extract_moe_trainable_buckets(model, num_router_fragments=2).items()}

    probe_chunks = _make_disjoint_domain_chunks(chunks_per_domain=8, seed_offset=400)
    eval_chunks = _make_disjoint_domain_chunks(chunks_per_domain=8, seed_offset=1400)

    cfg = WeightSpaceControllerConfig(
        score_from=8,
        batch_size=8,
        loeo_chunks_per_domain=6,
        protect_early_layers_below=2,
        min_harm_nats=0.005,
        svd_tau=1.20,
        svd_eta=6.0,
        router_beta_suppress=2.0,
        router_beta_sink=0.75,
        router_min_domain_samples=4,
    )
    ctrl = WeightSpaceController(cfg)

    res_a = ctrl.run(model, start_buckets, probe_chunks, tuned_buckets=c2_buckets, eval_chunks=eval_chunks)
    res_b = ctrl.run(model, start_buckets, probe_chunks, tuned_buckets=c2_buckets, eval_chunks=eval_chunks)

    assert res_a.accepted is True and res_b.accepted is True
    assert res_a.receipt_sha256 == res_b.receipt_sha256
    assert res_a.receipt["stage_hashes"] == res_b.receipt["stage_hashes"]
    assert res_a.receipt["final_buckets_sha256"] == compute_buckets_sha256(res_a.buckets)
    for k in res_a.buckets:
        assert torch.equal(res_a.buckets[k], res_b.buckets[k])

    # Fixed-point idempotency: running again on already-repaired weights leaves weights bitwise unchanged
    res_fixed = ctrl.run(model, start_buckets, probe_chunks, tuned_buckets=res_a.buckets, eval_chunks=eval_chunks)
    assert res_fixed.accepted is False
    assert res_fixed.status in {"FIXED_POINT_UNCHANGED", "ROLLED_BACK_ENERGY_NON_IMPROVING"}
    assert compute_buckets_sha256(res_fixed.buckets) == compute_buckets_sha256(res_a.buckets)
    for k in res_a.buckets:
        assert torch.equal(res_fixed.buckets[k], res_a.buckets[k])

    # Provenance verification + tamper detection across stage and top-level hashes
    assert verify_controller_receipt_provenance(res_a.receipt) is True

    tampered_top = json.loads(json.dumps(res_a.receipt))
    tampered_top["status"] = "TAMPERED"
    assert verify_controller_receipt_provenance(tampered_top) is False

    tampered_stage_hash = json.loads(json.dumps(res_a.receipt))
    tampered_stage_hash["stage_hashes"]["stage_2_attribution"] = "0" * 64
    assert verify_controller_receipt_provenance(tampered_stage_hash) is False

    tampered_stage_body = json.loads(json.dumps(res_a.receipt))
    tampered_stage_body["stages"]["stage_4_layernorm"]["best_lambda_ln"] = 0.99
    assert verify_controller_receipt_provenance(tampered_stage_body) is False

    tampered_missing = json.loads(json.dumps(res_a.receipt))
    del tampered_missing["stages"][STAGE_ORDER[0]]
    assert verify_controller_receipt_provenance(tampered_missing) is False

    bad_hex = json.loads(json.dumps(res_a.receipt))
    bad_hex["receipt_sha256"] = "not_a_sha"
    assert verify_controller_receipt_provenance(bad_hex) is False


# ---------------------------------------------------------------------------
# S5: Stage 1 Sanitization, Edge Branches, Validation & Security Compliance
# ---------------------------------------------------------------------------


def test_s5_preflight_validation_edge_branches_and_security_gates() -> None:
    """Exercise Stage 1 sanitization/clamping, optional AlphaEdit, validation errors, and security gates."""
    model, carriers = _make_subspace_crowd_moe(num_layers=4, num_experts=4, top_k=2, lora_rank=4, seed=505)
    start_buckets = {k: v.detach().cpu().clone() for k, v in extract_moe_trainable_buckets(model, num_router_fragments=2).items()}
    _plant_s1_triple_regression(model, carriers)
    c2_buckets = {k: v.detach().cpu().clone() for k, v in extract_moe_trainable_buckets(model, num_router_fragments=2).items()}

    # 1. Stage 1 non-finite repair & extreme spectral/router-norm clamping
    corrupt_buckets = {k: v.clone() for k, v in c2_buckets.items()}
    corrupt_buckets["expert_1"][0] = float("nan")
    corrupt_buckets["router_frag_0"][0] = float("inf")

    with pytest.raises(ValueError, match="Non-finite"):
        run_stage1_preflight_sanitization(
            model,
            start_buckets,
            corrupt_buckets,
            config=WeightSpaceControllerConfig(sanitize_non_finite=False),
        )

    load_moe_trainable_buckets(model, corrupt_buckets, num_router_fragments=2)
    with torch.no_grad():
        model.blocks[0].mlp.router.weight[1, 0] = 90.0
        model.blocks[0].mlp.experts["2"].w1_B[:, 0] = 10.0
        model.blocks[0].mlp.experts["2"].w1_A[0, 0] = 10.0
    spiked_buckets = extract_moe_trainable_buckets(model, num_router_fragments=2)
    spiked_buckets["expert_1"][0] = float("nan")
    spiked_buckets["router_frag_0"][0] = float("inf")

    stage1_cfg = WeightSpaceControllerConfig(
        sanitize_non_finite=True,
        max_lora_singular_value=15.0,
        max_router_row_norm=20.0,
    )
    sanitized, s1_receipt = run_stage1_preflight_sanitization(
        model, start_buckets, spiked_buckets, config=stage1_cfg
    )
    assert s1_receipt["non_finite_repaired_elements"] == 2
    assert s1_receipt["router_rows_clamped"] >= 1
    assert s1_receipt["spectral_clamped_adapters_count"] >= 1
    assert s1_receipt["changed_from_input"] is True
    assert all(bool(torch.isfinite(v).all().item()) for v in sanitized.values())

    # 2. Optional AlphaEdit complement + custom_domain fallback (no E<digit>_ prefix) + marginal-only SVD + force_zero_expert0
    probe_chunks = _make_disjoint_domain_chunks(chunks_per_domain=6, seed_offset=500)
    custom_probe = [
        {
            "expert_id": ("E0_general" if c["expert_id"].startswith("E0_") else "custom_domain"),
            "ids": c["ids"],
        }
        for c in probe_chunks
    ]
    alpha_cfg = WeightSpaceControllerConfig(
        score_from=8,
        batch_size=6,
        num_router_fragments=2,
        loeo_chunks_per_domain=4,
        protect_early_layers_below=2,
        min_harm_nats=0.005,
        force_zero_expert0=True,
        svd_target_marginal_only=True,
        svd_truncate=True,
        alphaedit_eta_x=2.0,
        alphaedit_eta_act=1.5,
        use_ln_calibration=False,
        router_beta_suppress=1.0,
        router_min_domain_samples=4,
    )
    opt_res = WeightSpaceController(alpha_cfg).run(
        model, start_buckets, custom_probe, tuned_buckets=c2_buckets, eval_chunks=None
    )
    assert opt_res.receipt["eval_split_source"] == "probe_chunks"
    assert 0 in opt_res.receipt["stages"]["stage_2_attribution"]["core_sink_experts"]
    assert opt_res.receipt["stages"]["stage_3_surgery"]["layers"]

    # Exercise Stage 3 with non-empty AlphaEdit targets (w1_A/w2_A/w3_A active modes) + OOF shutoff (rel_weight==0.0)
    with torch.no_grad():
        load_moe_trainable_buckets(model, c2_buckets, num_router_fragments=2)
        for l_idx in (2, 3):
            exp1 = model.blocks[l_idx].mlp.experts["1"]
            exp1.w1_A[1, 0] = 0.25
            exp1.w1_B[:, 1] = 0.20
            exp1.w2_A[1, 0] = 0.25
            exp1.w2_B[:, 1] = 0.20
    c2_alpha = extract_moe_trainable_buckets(model, num_router_fragments=2)
    alpha_all_cfg = WeightSpaceControllerConfig(
        score_from=8,
        batch_size=6,
        num_router_fragments=2,
        loeo_chunks_per_domain=4,
        protect_early_layers_below=2,
        min_harm_nats=0.005,
        svd_target_marginal_only=False,
        alphaedit_eta_x=4.0,
        alphaedit_eta_act=3.0,
        alphaedit_tau=1.05,
        router_beta_suppress=1.0,
    )
    s2_ctx_a, _ = run_stage2_loeo_and_layer_attribution(
        model, start_buckets, c2_alpha, custom_probe, config=alpha_all_cfg
    )
    _, _, s3_alpha_rep = run_stage3_wsp_surgery(
        model, c2_alpha, custom_probe, s2_ctx_a, config=alpha_all_cfg
    )
    assert any(
        ly["alphaedit"]["active_modes_x"] > 0 or ly["alphaedit"]["active_modes_act"] > 0
        for ly in s3_alpha_rep["layers"]
    )

    # Identical E0/E1 probe tokens trigger OOF rel_weight == 0.0 shutoff in Stage 3 + missing expert in _expert_has_nonzero_lora_b
    shared_probe = [
        {"expert_id": ("E0_gen" if i % 2 == 0 else "E1_code"), "ids": probe_chunks[0]["ids"]}
        for i in range(8)
    ]
    s2_ctx_shut = {
        "partition": {
            "core_sink_experts": [99],
            "rogue_core_sink_experts": [99],
            "marginal_specialist_experts": [1],
            "protect_general_expert": True,
        },
        "layer_mask_summary": {"selected_layers": [2]},
    }
    _, _, s3_shut = run_stage3_wsp_surgery(
        model, c2_buckets, shared_probe, s2_ctx_shut, config=WeightSpaceControllerConfig(score_from=8)
    )
    assert s3_shut["router_layers_applied"] == 0
    assert s3_shut["layers"][0]["router_control"]["applied"] is False

    # 3. Direct Stage 5 calls (parent_state_dict=None, non-finite candidate, fixed-point, rollback) & metrics fallback
    qual_fb = evaluate_multidomain_anse_metrics(
        [1.0, 1.1], [0.9, 1.0], ["E0_gen", "E1_code"], min_chunks_per_expert=10
    )
    assert qual_fb["worst_domain_guard_ok"] is True

    nan_metrics = evaluate_multidomain_anse_metrics(
        [1.0, 2.0], [float("nan"), 1.5], ["E0_gen", "E1_code"]
    )
    assert nan_metrics["finite"] is False
    assert nan_metrics["anse_energy"] == ANSE_MAXIMUM_PAIN

    s5_fixed, s5_fixed_rec = run_stage5_thermodynamic_anse_gate(
        model,
        start_buckets,
        start_buckets,
        start_buckets,
        probe_chunks,
        config=WeightSpaceControllerConfig(score_from=8),
        parent_state_dict=None,
    )
    assert s5_fixed_rec["status"] == "FIXED_POINT_UNCHANGED"
    assert torch.equal(s5_fixed["expert_0"], start_buckets["expert_0"])

    nan_cand_buckets = {k: v.clone() for k, v in c2_buckets.items()}
    nan_cand_buckets["expert_0"][:] = float("nan")
    _, s5_nan_rec = run_stage5_thermodynamic_anse_gate(
        model,
        start_buckets,
        start_buckets,
        nan_cand_buckets,
        probe_chunks,
        config=WeightSpaceControllerConfig(score_from=8),
        parent_state_dict=None,
    )
    assert s5_nan_rec["rolled_back"] is True
    assert s5_nan_rec["candidate_metrics"]["finite"] is False

    with pytest.raises(ValueError, match="1D length"):
        evaluate_multidomain_anse_metrics([1.0], [1.0, 2.0], ["E0_gen"])
    with pytest.raises(ValueError, match="finite 1D"):
        evaluate_multidomain_anse_metrics([], [], [])
    with pytest.raises(ValueError, match="general chunk"):
        evaluate_multidomain_anse_metrics([1.0], [1.0], ["E1_code"])
    with pytest.raises(ValueError, match="tuned_losses"):
        evaluate_multidomain_anse_metrics([1.0, 2.0], [1.0, 2.0], ["E0_gen", "E1_code"], tuned_losses=[1.0])

    nan_cand_model, _ = _make_subspace_crowd_moe(num_layers=2, num_experts=4, top_k=2, lora_rank=4, seed=506)
    with torch.no_grad():
        nan_cand_model.wte.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Frozen model parameter"):
        WeightSpaceController(WeightSpaceControllerConfig(score_from=8)).run(
            nan_cand_model, start_buckets, probe_chunks, tuned_buckets=c2_buckets
        )

    # 4. Partition & Config & Input validation error branches
    part_forced = partition_loeo_experts(
        {
            "general_expert_index": 0,
            "protect_general_expert": True,
            "harmful_sink_experts": [],
            "rogue_high_norm_experts": [],
            "recommended_sink_experts": [],
            "interfering_specialist_experts": [0],
            "expert_loeo": {"expert_0": {"delta_spec_when_zeroed": 0.1}},
        },
        force_zero_expert0=True,
        use_ghost_sink=True,
    )
    assert part_forced["core_sink_experts"] == [0]
    assert part_forced["protect_general_expert"] is False

    with pytest.raises(TypeError, match="WeightSpaceControllerConfig"):
        WeightSpaceController(config="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="general_prefix"):
        WeightSpaceControllerConfig(general_prefix=" ")
    with pytest.raises(TypeError, match="bool"):
        WeightSpaceControllerConfig(use_ghost_sink=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="> 0.0"):
        WeightSpaceControllerConfig(max_lora_singular_value=0.0)
    with pytest.raises(ValueError, match=">= 1"):
        WeightSpaceControllerConfig(batch_size=0)
    with pytest.raises(TypeError, match="real number"):
        WeightSpaceControllerConfig(svd_tau=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0.0"):
        WeightSpaceControllerConfig(svd_tau=-1.0)
    with pytest.raises(TypeError, match="real number"):
        WeightSpaceControllerConfig(loeo_e0_harm_threshold=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="<= 0.0"):
        WeightSpaceControllerConfig(loeo_e0_harm_threshold=0.05)
    with pytest.raises(TypeError, match="int"):
        WeightSpaceControllerConfig(top_k_layers=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">="):
        WeightSpaceControllerConfig(scored_expert_indices=(-1,))
    with pytest.raises(ValueError, match="finite"):
        WeightSpaceControllerConfig(forced_lambda_ln=float("nan"))

    assert verify_controller_receipt_provenance({"receipt_sha256": "0" * 64, "stages": [], "stage_hashes": {}}) is False
    bad_stage_entry = {
        "receipt_sha256": "0" * 64,
        "stage_hashes": {s: "0" * 64 for s in STAGE_ORDER},
        "stages": {s: {"stage": "wrong"} for s in STAGE_ORDER},
    }
    assert verify_controller_receipt_provenance(bad_stage_entry) is False
    bad_stage_sha = {
        "receipt_sha256": "0" * 64,
        "stage_hashes": {s: "0" * 64 for s in STAGE_ORDER},
        "stages": {s: {"stage": s, "stage_receipt_sha256": "short"} for s in STAGE_ORDER},
    }
    assert verify_controller_receipt_provenance(bad_stage_sha) is False
    nan_stage_rec = json.loads(json.dumps(opt_res.receipt))
    nan_stage_rec["stages"][STAGE_ORDER[0]]["bad"] = float("nan")
    assert verify_controller_receipt_provenance(nan_stage_rec) is False
    nan_top_rec = json.loads(json.dumps(opt_res.receipt))
    nan_top_rec["bad"] = float("nan")
    assert verify_controller_receipt_provenance(nan_top_rec) is False

    with pytest.raises(TypeError, match="mapping"):
        verify_controller_receipt_provenance("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty mapping"):
        compute_buckets_sha256({})
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_buckets_sha256({"a": [1.0]})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="nn.Module"):
        WeightSpaceController().run("bad", start_buckets, probe_chunks)  # type: ignore[arg-type]

    empty_blk_m = nn.Module()
    empty_blk_m.blocks = nn.ModuleList()
    with pytest.raises(ValueError, match="non-empty 'blocks'"):
        WeightSpaceController().run(empty_blk_m, start_buckets, probe_chunks)
    missing_attr_m = nn.Module()
    missing_attr_m.blocks = nn.ModuleList([nn.Linear(4, 4)])
    with pytest.raises(ValueError, match="must expose"):
        WeightSpaceController().run(missing_attr_m, start_buckets, probe_chunks)

    with pytest.raises(ValueError, match="non-empty mapping"):
        run_stage1_preflight_sanitization(model, {}, c2_buckets, config=WeightSpaceControllerConfig())
    with pytest.raises(ValueError, match="non-empty mapping"):
        run_stage1_preflight_sanitization(model, start_buckets, {}, config=WeightSpaceControllerConfig())
    with pytest.raises(ValueError, match="router_frag_"):
        run_stage1_preflight_sanitization(
            model, {"expert_0": torch.ones(2)}, {"expert_0": torch.ones(2)}, config=WeightSpaceControllerConfig()
        )
    with pytest.raises(ValueError, match="expert_"):
        run_stage1_preflight_sanitization(
            model, {"router_frag_0": torch.ones(2)}, {"router_frag_0": torch.ones(2)}, config=WeightSpaceControllerConfig()
        )
    with pytest.raises(TypeError, match="torch.Tensor"):
        run_stage1_preflight_sanitization(
            model,
            {"router_frag_0": [1.0], "expert_0": torch.ones(2)},  # type: ignore[dict-item]
            {"router_frag_0": [1.0], "expert_0": torch.ones(2)},  # type: ignore[dict-item]
            config=WeightSpaceControllerConfig(),
        )
    with pytest.raises(ValueError, match="non-empty"):
        run_stage1_preflight_sanitization(
            model,
            {"router_frag_0": torch.empty(0), "expert_0": torch.ones(2)},
            {"router_frag_0": torch.empty(0), "expert_0": torch.ones(2)},
            config=WeightSpaceControllerConfig(),
        )
    with pytest.raises(ValueError, match="identical bucket keys"):
        run_stage1_preflight_sanitization(
            model, start_buckets, {"router_frag_0": torch.ones(2)}, config=WeightSpaceControllerConfig()
        )
    with pytest.raises(ValueError, match="Shape mismatch"):
        bad_shape = {k: v.clone() for k, v in start_buckets.items()}
        bad_shape["expert_0"] = torch.ones(1)
        run_stage1_preflight_sanitization(model, start_buckets, bad_shape, config=WeightSpaceControllerConfig())
    with pytest.raises(ValueError, match="must contain only finite"):
        bad_start = {k: v.clone() for k, v in start_buckets.items()}
        bad_start["expert_0"][0] = float("nan")
        run_stage1_preflight_sanitization(model, bad_start, c2_buckets, config=WeightSpaceControllerConfig())
    with pytest.raises(ValueError, match="too short"):
        short_r = {k: v.clone() for k, v in start_buckets.items()}
        short_r["router_frag_1"] = torch.ones(4)
        run_stage4_fine_layernorm_calibration(
            model,
            short_r,
            c2_buckets,
            c2_buckets,
            torch.zeros((12, 16, 32)),
            probe_chunks,
            torch.zeros(12).numpy(),
            config=WeightSpaceControllerConfig(score_from=8),
        )
    with pytest.raises(ValueError, match="non-empty sequence"):
        WeightSpaceController(WeightSpaceControllerConfig(score_from=8)).run(
            model, start_buckets, [], tuned_buckets=c2_buckets
        )
    with pytest.raises(ValueError, match="mapping containing"):
        WeightSpaceController(WeightSpaceControllerConfig(score_from=8)).run(
            model, start_buckets, [{"ids": [0, 1, 2]}], tuned_buckets=c2_buckets
        )
    with pytest.raises(ValueError, match="non-empty string"):
        WeightSpaceController(WeightSpaceControllerConfig(score_from=8)).run(
            model, start_buckets, [{"expert_id": "  ", "ids": [0, 1, 2]}], tuned_buckets=c2_buckets
        )
    with pytest.raises(ValueError, match="length >= 2"):
        WeightSpaceController(WeightSpaceControllerConfig(score_from=8)).run(
            model, start_buckets, [{"expert_id": "E0_a", "ids": [0]}], tuned_buckets=c2_buckets
        )
    with pytest.raises(ValueError, match="non-negative token"):
        WeightSpaceController(WeightSpaceControllerConfig(score_from=8)).run(
            model, start_buckets, [{"expert_id": "E0_a", "ids": [0, -1, 2]}], tuned_buckets=c2_buckets
        )
    with pytest.raises(ValueError, match="sequence length"):
        WeightSpaceController(WeightSpaceControllerConfig(score_from=8)).run(
            model,
            start_buckets,
            [{"expert_id": "E0_a", "ids": [0, 1, 2]}, {"expert_id": "E0_a", "ids": [0, 1, 2, 3]}],
            tuned_buckets=c2_buckets,
        )
    with pytest.raises(ValueError, match="general chunk"):
        WeightSpaceController(WeightSpaceControllerConfig(score_from=8)).run(
            model, start_buckets, [{"expert_id": "E1_a", "ids": [0, 1, 2]}], tuned_buckets=c2_buckets
        )
    with pytest.raises(ValueError, match="too short"):
        WeightSpaceController(WeightSpaceControllerConfig(score_from=64)).run(
            model, start_buckets, probe_chunks, tuned_buckets=c2_buckets
        )
    with pytest.raises(ValueError, match="too short"):
        run_stage3_wsp_surgery(
            model, c2_buckets, probe_chunks, s2_ctx_shut, config=WeightSpaceControllerConfig(score_from=64)
        )

    # 5. AGPL-3.0 header, optional ZeroStubAudit, PRSecurityScanner, lateral movement, and pr_gatekeeper FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn04", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn04", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/unified_wsp_neural_controller.py", TARGET_MODULE_PATH),
        ("tests/test_unified_wsp_neural_controller_scenarios.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"

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

    receipt_json = json.dumps(opt_res.receipt, sort_keys=True)
    assert gatekeeper.forbidden_scan(f"diff --git a/x.py b/x.py\n+++ b/x.py\n+RECEIPT = {receipt_json!r}") == []

    pr_audit = scanner.audit_pull_request(
        pr_number=1104,
        pr_title="feat(swarm): unified multi-stage Weight-Space Programming controller orchestrating LOEO partition, SVD, multi-gate, and fine LayerNorm",
        pr_body="Implements NEURAL-CTRL-DYN-04 unified 5-stage WeightSpaceController and deterministic CPU scenario tests.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): unified multi-stage Weight-Space Programming controller orchestrating LOEO partition, SVD, multi-gate, and fine LayerNorm"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
