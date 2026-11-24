# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-10: cross-layer residual interference cancellation."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.cross_layer_residual_nullspace import (
    DEFAULT_ANSE_MAXIMUM_PAIN,
    RECEIPT_OBJECT,
    SCHEMA_VERSION,
    CrossLayerResidualConfig,
    _restore_model_snapshot_bitwise,
    compute_module_state_sha256,
    identify_beneficial_anchor_and_interfering_layers,
    optimize_cross_layer_residual_surgery,
    project_downstream_lora_b_orthogonal_to_anchor,
    verify_cross_layer_receipt_provenance,
)
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdLoRAExpertAdapter,
    CrowdLoRAMoEMLP,
    extract_moe_trainable_buckets,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.layer_localized_attribution import (
    compute_embeddings_batch,
    forward_block_dense_base_mlp,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    per_chunk_causal_midrow_nll,
    zero_expert_lora_b_inplace,
)

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
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "cross_layer_residual_nullspace.py"
TEST_FILE_PATH = Path(__file__).resolve()
EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."


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


def _make_8layer_cross_layer_moe(
    *,
    num_layers: int = 8,
    num_experts: int = 4,
    top_k: int = 2,
    lora_rank: int = 4,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    vocab_size: int = 48,
    anchor_layer: int = 5,
    interfering_layers: tuple[int, ...] = (6, 7),
    plant_w2_b_interference: bool = True,
    overlap_spec_with_anchor: bool = False,
    seed: int = 20261124,
) -> tuple[CrowdV1WithEngram, dict[str, torch.Tensor]]:
    """Build deterministic 8-layer Sparse MoE with a beneficial anchor layer and downstream overwrites.

    Subspaces:
      - ``0..15``:  E0 general prose (target token ``5``)
      - ``16..23``: E1 code specialist (target token ``20``)
      - ``24..31``: E2 math specialist (target token ``36``)
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

    u_e0 = _unit_axis(hidden_size, 0, 1)
    u_e1 = _unit_axis(hidden_size, 16, 17)
    u_e2 = _unit_axis(hidden_size, 24, 25)

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        p_raw = 0.25 * torch.randn((16, 16), generator=gen, dtype=torch.float32)
        p_raw[:, :2] = 0.0
        p_raw = p_raw - p_raw.mean(dim=1, keepdim=True) + 1.0 * u_e0[:16].unsqueeze(0)
        p_raw[5] = 1.45 * u_e0[:16]

        e1_raw = 0.25 * torch.randn((16, 8), generator=gen, dtype=torch.float32)
        e1_raw[:, :2] = 0.0
        e1_raw = e1_raw - e1_raw.mean(dim=1, keepdim=True) + 1.0 * u_e1[16:24].unsqueeze(0)
        e1_raw[20 - 16] = 1.45 * u_e1[16:24]

        e2_raw = 0.25 * torch.randn((16, 8), generator=gen, dtype=torch.float32)
        e2_raw[:, :2] = 0.0
        e2_raw = e2_raw - e2_raw.mean(dim=1, keepdim=True) + 1.0 * u_e2[24:32].unsqueeze(0)
        e2_raw[36 - 32] = 1.45 * u_e2[24:32]

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
            blk.mlp.w1.weight[:hidden_size, :hidden_size].copy_(0.95 * torch.eye(hidden_size))
            blk.mlp.w2.weight[:hidden_size, :hidden_size].copy_(0.95 * torch.eye(hidden_size))
            blk.mlp.w3.weight[:hidden_size, :hidden_size].copy_(0.10 * torch.eye(hidden_size))

    upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode="lora_domain_experts",
        num_experts=num_experts,
        top_k=top_k,
        lora_rank=lora_rank,
        lora_alpha=8.0,
        routing_mode="sigma_topk",
        seed=seed + 11,
    )

    with torch.no_grad():
        for l_idx, blk in enumerate(model.blocks):
            mlp: CrowdLoRAMoEMLP = blk.mlp  # type: ignore[assignment]
            mlp.router.weight.zero_()
            for exp_mod in mlp.experts.values():
                for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                    getattr(exp_mod, attr).zero_()

            # Clean domain routing across all layers
            mlp.router.weight[0] = 3.5 * u_e0
            mlp.router.weight[1] = 3.5 * u_e1
            mlp.router.weight[2] = 3.5 * u_e2

            if l_idx == anchor_layer:
                # Beneficial anchor layer: strong positive E0 boost (~ -0.25 nats E0 benefit)
                exp0 = mlp.experts["0"]
                exp0.w3_A[0, :hidden_size] = 0.25 * u_e0
                exp0.w3_B[:, 0] = 1.60 * u_e0
                # Also carry moderate specialist signal on anchor layer
                mlp.experts["1"].w3_A[0, 16:24] = 0.15 * u_e1[16:24]
                mlp.experts["1"].w3_B[:, 0] = 0.50 * u_e1
            elif l_idx in interfering_layers:
                # Downstream interfering layers leak E0 into E1/E2 routers and inject -u_e0 overwrite!
                mlp.router.weight[1] = 3.5 * u_e1 + 2.4 * u_e0
                mlp.router.weight[2] = 3.5 * u_e2 + 2.4 * u_e0

                exp1 = mlp.experts["1"]
                # Mode 0: critical E1 specialist boost
                exp1.w3_A[0, 16:24] = 0.22 * u_e1[16:24]
                exp1.w3_B[:, 0] = (1.60 * u_e0 + 0.95 * u_e1) if overlap_spec_with_anchor else (0.95 * u_e1)
                # Mode 1: constructive prose drift overwriting upstream anchor along -u_e0
                exp1.w3_A[1, :hidden_size] = 0.24 * u_e0
                exp1.w3_B[:, 1] = -0.85 * u_e0

                if overlap_spec_with_anchor:
                    # Add 0.8 * u_e0 (< 1.45 of token 5, so token 5 still leads on E0) to E1 target token 20
                    model.wte.weight[20, :16].add_(0.8 * u_e0[:16])

                exp2 = mlp.experts["2"]
                # Mode 0: critical E2 specialist boost
                exp2.w3_A[0, 24:32] = 0.22 * u_e2[24:32]
                exp2.w3_B[:, 0] = 0.95 * u_e2
                if plant_w2_b_interference and l_idx == interfering_layers[-1]:
                    # Plant E0 overwrite inside SwiGLU w2_B -> base_mlp.w3 on last interfering layer
                    exp2.w2_A[1, :hidden_size] = 0.45 * u_e0
                    exp2.w2_B[:hidden_size, 1] = -2.20 * u_e0
                else:
                    exp2.w3_A[1, :hidden_size] = 0.24 * u_e0
                    exp2.w3_B[:, 1] = -0.85 * u_e0

    return model, {"e0": u_e0, "e1": u_e1, "e2": u_e2}


def _make_domain_chunks(
    *,
    seq_len: int = 16,
    score_from: int = 8,
    domains: tuple[str, ...] = ("E0_general_fineweb", "E1_python_systems_code", "E2_math_reasoning"),
    chunks_per_domain: int = 6,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    gen = torch.Generator(device="cpu").manual_seed(7000 + seed_offset)
    chunks: list[dict[str, Any]] = []
    for d_idx, dom in enumerate(domains):
        for r_idx in range(chunks_per_domain):
            if dom.startswith("E0_"):
                ids = torch.randint(0, 16, (seq_len,), generator=gen).tolist()
                tgt = 5
            elif dom.startswith("E1_"):
                ids = torch.randint(16, 32, (seq_len,), generator=gen).tolist()
                tgt = 20
            else:
                ids = torch.randint(32, 48, (seq_len,), generator=gen).tolist()
                tgt = 36
            for pos in range(score_from, seq_len):
                ids[pos] = tgt
            chunks.append(
                {
                    "expert_id": dom,
                    "dataset_id": dom,
                    "row": seed_offset + 100 * d_idx + r_idx,
                    "ids": ids,
                }
            )
    return chunks


@torch.no_grad()
def _eval_model_nlls(
    model: CrowdV1WithEngram,
    chunks: list[dict[str, Any]],
    *,
    score_from: int = 8,
) -> dict[str, float]:
    ids = torch.tensor([c["ids"] for c in chunks], dtype=torch.long)
    cos = model.cos_cached.to(ids.device)
    sin = model.sin_cached.to(ids.device)
    xb = compute_embeddings_batch(model, ids)
    for blk in model.blocks:
        xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
        xb = forward_block_post_mlp(blk, xm, mi)
    losses = per_chunk_causal_midrow_nll(
        xb,
        model.ln.weight,
        model.ln.bias,
        float(model.ln.eps),
        model.head.weight,
        ids,
        score_from=score_from,
        seq_batch_size=24,
    ).tolist()
    grouped: dict[str, list[float]] = {}
    for ch, val in zip(chunks, losses, strict=True):
        grouped.setdefault(str(ch["expert_id"]), []).append(float(val))
    out = {k: sum(v) / len(v) for k, v in grouped.items()}
    spec_vals = [v for k, v in out.items() if not k.startswith("E0_")]
    out["spec_mean"] = sum(spec_vals) / len(spec_vals) if spec_vals else 0.0
    out["nll_mean"] = sum(losses) / len(losses)
    return out


# ---------------------------------------------------------------------------
# S1: 8-Layer MoE Anchor Protection + Downstream Orthogonalization vs Uniform Sinking
# ---------------------------------------------------------------------------


def test_s1_8layer_anchor_protection_beats_uniform_sinking() -> None:
    """S1: Prove anchor protection + downstream orthogonalization beats uniform sinking on E0 & specialist NLL."""
    # Empirical T4 Stage S2 depth partition check (layer 17 anchor at -0.2516, layers 18-19 interfering)
    t4_stage_s2_harms = [0.0] * 17 + [-0.2516, 0.0150, 0.0129]
    t4_part = identify_beneficial_anchor_and_interfering_layers(t4_stage_s2_harms)
    assert t4_part["beneficial_anchor_layers"] == [17]
    assert t4_part["interfering_layers"] == [18, 19]
    assert t4_part["downstream_interfering_layers"] == [18, 19]
    assert t4_part["anchor_gain_nats"] == pytest.approx(0.2516, abs=1e-6)
    assert t4_part["has_cross_layer_interference"] is True

    model, _ = _make_8layer_cross_layer_moe(seed=20261124, plant_w2_b_interference=True)
    probe_chunks = _make_domain_chunks(seq_len=16, score_from=8, chunks_per_domain=6, seed_offset=0)
    eval_chunks = _make_domain_chunks(seq_len=16, score_from=8, chunks_per_domain=6, seed_offset=500)

    parent_nlls = _eval_model_nlls(model, eval_chunks, score_from=8)

    # Baseline 2: Uniform sinking of all experts across all 8 layers (destroys anchor 5 & specialist 6-7)
    import copy

    uniform_model = copy.deepcopy(model)
    with torch.no_grad():
        for blk in uniform_model.blocks:
            zero_expert_lora_b_inplace(blk.mlp, [0, 1, 2])
    uniform_nlls = _eval_model_nlls(uniform_model, eval_chunks, score_from=8)

    # Also evaluate partial uniform sinking of only interfering experts [1, 2] across all layers
    partial_sink_model = copy.deepcopy(model)
    with torch.no_grad():
        for blk in partial_sink_model.blocks:
            zero_expert_lora_b_inplace(blk.mlp, [1, 2])
    partial_sink_nlls = _eval_model_nlls(partial_sink_model, eval_chunks, score_from=8)

    # Run Cross-Layer Residual Nullspace Surgery (non-inplace copy first, then inplace)
    cfg = CrossLayerResidualConfig(
        anchor_threshold_nats=-0.015,
        harm_threshold_nats=0.002,
        max_rank_proj=8,
        score_from=8,
        inplace=False,
    )
    edited_model, receipt = optimize_cross_layer_residual_surgery(
        model,
        probe_chunks,
        selection_chunks=eval_chunks,
        config=cfg,
    )

    assert receipt["schema"] == SCHEMA_VERSION
    assert receipt["receipt_object"] == RECEIPT_OBJECT
    assert verify_cross_layer_receipt_provenance(receipt) is True
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["status"] == "COMMITTED"
    assert receipt["anchor_layers"] == [5]
    assert receipt["downstream_interfering_layers"] == [6, 7]
    assert receipt["neutral_layers"] == [0, 1, 2, 3, 4]
    assert receipt["anchor_gain_nats"] > 0.05
    assert receipt["delta_energy"] < 0.0
    assert receipt["guard_ok"] is True
    assert receipt["worst_domain_drift_pct"] <= 1.0

    # Original model untouched when inplace=False
    assert compute_module_state_sha256(model) == receipt["parent_sha256"]
    assert compute_module_state_sha256(edited_model) == receipt["active_sha256"]

    edited_nlls = _eval_model_nlls(edited_model, eval_chunks, score_from=8)

    # Strictly beats unedited parent on E0 prose NLL and overall NLL
    assert edited_nlls["E0_general_fineweb"] < parent_nlls["E0_general_fineweb"] - 0.01
    assert edited_nlls["nll_mean"] < parent_nlls["nll_mean"]

    # Strictly outperforms uniform sinking on BOTH general E0 AND specialist NLL!
    assert edited_nlls["E0_general_fineweb"] < uniform_nlls["E0_general_fineweb"] - 0.10
    assert edited_nlls["spec_mean"] < uniform_nlls["spec_mean"] - 0.10
    assert edited_nlls["E1_python_systems_code"] < partial_sink_nlls["E1_python_systems_code"] - 0.05
    assert edited_nlls["E2_math_reasoning"] < partial_sink_nlls["E2_math_reasoning"] - 0.05

    # Also verify buckets loading + cal_ln_w/b + explicit layer_attribution + NOOP when no downstream interference
    buckets = extract_moe_trainable_buckets(model, num_router_fragments=2)
    noop_model, noop_rec = optimize_cross_layer_residual_surgery(
        model,
        probe_chunks,
        buckets=buckets,
        cal_ln_w=model.ln.weight.detach().clone(),
        cal_ln_b=model.ln.bias.detach().clone(),
        layer_attribution=[-0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05],
        score_from=8,
        inplace=False,
    )
    assert noop_rec["status"] == "NOOP_NO_DOWNSTREAM_INTERFERENCE"
    assert noop_rec["accepted"] is False
    assert noop_rec["rolled_back"] is False
    assert compute_module_state_sha256(noop_model) == noop_rec["parent_sha256"]


# ---------------------------------------------------------------------------
# S2: Exact Bitwise Preservation of Beneficial Anchor Layer Weights
# ---------------------------------------------------------------------------


def test_s2_exact_bitwise_anchor_layer_preservation() -> None:
    """S2: Verify anchor layer parameters and SHA-256 remain bitwise identical after surgery."""
    model, _ = _make_8layer_cross_layer_moe(seed=20261125, plant_w2_b_interference=False)
    probe_chunks = _make_domain_chunks(seq_len=16, score_from=8, chunks_per_domain=6, seed_offset=100)

    before_layer_hashes = [compute_module_state_sha256(blk) for blk in model.blocks]
    anchor_param_snaps = {
        name: param.detach().clone() for name, param in model.blocks[5].named_parameters()
    }

    _, receipt = optimize_cross_layer_residual_surgery(
        model,
        probe_chunks,
        score_from=8,
        target_experts=[1, 2],
        inplace=True,
    )

    assert receipt["accepted"] is True
    assert receipt["anchor_layers"] == [5]
    assert receipt["anchor_layers_bitwise_preserved"] is True
    assert receipt["anchor_layer_sha256_before"]["5"] == before_layer_hashes[5]
    assert receipt["anchor_layer_sha256_after"]["5"] == before_layer_hashes[5]

    # Every early/neutral/anchor layer 0..5 is bitwise unchanged; layers 6-7 were surgically updated
    for l_idx in range(6):
        assert compute_module_state_sha256(model.blocks[l_idx]) == before_layer_hashes[l_idx]
    for name, param in model.blocks[5].named_parameters():
        assert torch.equal(param.detach(), anchor_param_snaps[name])
    for l_idx in (6, 7):
        assert compute_module_state_sha256(model.blocks[l_idx]) != before_layer_hashes[l_idx]


# ---------------------------------------------------------------------------
# S3: Specialist Residual Subspace Preservation (< 1e-5 Specialist Delta Distortion)
# ---------------------------------------------------------------------------


def test_s3_specialist_residual_subspace_preservation() -> None:
    """S3: Certify <1e-5 specialist delta distortion across Tensor, Module, MoE, and full-rank inputs."""
    gen = torch.Generator(device="cpu").manual_seed(20261126)
    dim = 32
    rank = 4
    u_anchor = _unit_axis(dim, 0, 1)
    u_anchor_2 = _unit_axis(dim, 2, 3)
    u_spec_1 = _unit_axis(dim, 16, 17)
    u_spec_2 = _unit_axis(dim, 24, 25)

    # Anchor deltas span u_anchor + partial leak into u_spec_1; specialist deltas strictly span u_spec_1, u_spec_2
    anchor_deltas = torch.stack(
        [
            3.0 * u_anchor + 0.6 * u_spec_1 + 1.2 * u_anchor_2,
            2.5 * u_anchor + 0.4 * u_spec_1 - 0.9 * u_anchor_2,
            2.8 * u_anchor + 0.5 * u_spec_1 + 0.5 * u_anchor_2,
        ],
        dim=0,
    )
    spec_deltas = torch.stack(
        [
            2.2 * u_spec_1 + 0.8 * u_spec_2,
            -1.4 * u_spec_1 + 2.5 * u_spec_2,
            0.9 * u_spec_1 - 1.7 * u_spec_2,
        ],
        dim=0,
    )

    # Downstream LoRA B (H x r) + A (r x H): col 0 & 2 are specialist modes, col 1 is anchor interference
    w_a = torch.zeros((rank, dim), dtype=torch.float32)
    w_a[0] = u_spec_1
    w_a[1] = u_anchor
    w_a[2] = u_spec_2
    w_b = torch.zeros((dim, rank), dtype=torch.float32)
    w_b[:, 0] = 1.5 * u_spec_1 + 0.7 * u_spec_2
    w_b[:, 1] = -2.0 * u_anchor - 1.1 * u_anchor_2
    w_b[:, 2] = 1.2 * u_spec_2 + 0.4 * u_spec_1

    x_spec_test = torch.randn((12, dim), generator=gen)
    # Keep x_spec_test in specialist input subspace (dims 16..31)
    x_spec_test[:, :16] = 0.0
    dy_spec_before = (x_spec_test @ w_a.T) @ w_b.T

    w_b_proj, stats = project_downstream_lora_b_orthogonal_to_anchor(
        anchor_deltas,
        w_b,
        protect_specialist_deltas=spec_deltas,
        max_rank_proj=8,
    )
    assert isinstance(w_b_proj, torch.Tensor)
    assert stats["used_exact_specialist_nullspace"] is True
    assert stats["anchor_interference_reduction_ratio"] > 0.9999
    assert stats["anchor_overlap_after"] < 1e-6
    assert stats["specialist_distortion_max"] < 1e-5

    dy_spec_after = (x_spec_test @ w_a.T) @ w_b_proj.T
    spec_delta_distortion = float((dy_spec_after - dy_spec_before).abs().max().item())
    assert spec_delta_distortion < 1e-5

    # Anchor-subspace component in col 1 is completely annihilated while col 2 specialist stays intact
    assert abs(float((u_anchor @ w_b_proj[:, 1]).item())) < 1e-6
    assert abs(float((u_anchor_2 @ w_b_proj[:, 1]).item())) < 1e-6
    assert abs(float((u_spec_2 @ w_b_proj[:, 2]).item()) - 1.2) < 1e-6

    # Llama-style module exposing w2_B of shape (H, r) + Mapping + nn.Linear + CrowdLoRAExpertAdapter
    class _LlamaLoRAExpert(nn.Module):
        def __init__(self, init_b: torch.Tensor) -> None:
            super().__init__()
            self.w2_B = nn.Parameter(init_b.clone())

    llama_mod = _LlamaLoRAExpert(w_b)
    llama_out, llama_stats = project_downstream_lora_b_orthogonal_to_anchor(
        [anchor_deltas.unsqueeze(0)],
        llama_mod,
        protect_specialist_deltas={"s": spec_deltas},
        max_rank_proj=8,
        proj_strength=0.5,
        inplace=True,
    )
    assert llama_out is llama_mod
    assert llama_stats["modified_factors"] == ["w2_B"]
    assert llama_stats["specialist_distortion_max"] < 1e-5
    assert llama_stats["anchor_overlap_after"] == pytest.approx(
        0.5 * llama_stats["anchor_overlap_before"], abs=1e-5
    )

    map_out, map_stats = project_downstream_lora_b_orthogonal_to_anchor(
        anchor_deltas, {"w2_B": w_b}, protect_specialist_deltas=spec_deltas, max_rank_proj=8
    )
    assert "w2_B" in map_out and map_stats["specialist_distortion_max"] < 1e-5

    lin = nn.Linear(dim, dim, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.outer(u_anchor, u_anchor) + torch.outer(u_spec_1, u_spec_1))
    lin_out, _ = project_downstream_lora_b_orthogonal_to_anchor(
        anchor_deltas, lin, protect_specialist_deltas=spec_deltas, max_rank_proj=8
    )
    assert float((lin_out.weight @ u_spec_1 - u_spec_1).norm().item()) < 1e-6
    assert float((u_anchor @ lin_out.weight @ u_anchor).abs().item()) < 1e-6

    # CrowdLoRAExpertAdapter with intermediate (I x r) w2_B + base_output_proj + zero-anchor noop
    exp_adapter = CrowdLoRAExpertAdapter(dim, 64, rank=rank, seed=42)
    with torch.no_grad():
        exp_adapter.w3_B.copy_(w_b)
        exp_adapter.w2_B[:dim, 1] = -1.5 * u_anchor
        exp_adapter.w2_B[16:24, 0] = 1.1 * u_spec_1[16:24]
    w3_base = torch.zeros((dim, 64), dtype=torch.float32)
    w3_base[:, :dim].copy_(torch.eye(dim))
    exp_proj, exp_stats = project_downstream_lora_b_orthogonal_to_anchor(
        anchor_deltas,
        exp_adapter,
        protect_specialist_deltas=spec_deltas,
        base_output_proj=w3_base,
        max_rank_proj=8,
    )
    assert "w3_B" in exp_stats["modified_factors"] and "w2_B" in exp_stats["modified_factors"]
    assert exp_stats["specialist_distortion_max"] < 1e-5
    assert float((u_spec_1[16:24] @ exp_proj.w2_B[16:24, 0]).item()) == pytest.approx(1.1, abs=1e-5)

    # Zero anchor deltas -> noop (also with CrowdLoRAExpertAdapter + base_output_proj); zero specialist -> same as None
    _, noop_zero = project_downstream_lora_b_orthogonal_to_anchor(torch.zeros((4, dim)), w_b)
    assert noop_zero["effective_cancel_rank"] == 0
    _, noop_exp = project_downstream_lora_b_orthogonal_to_anchor(
        torch.zeros((4, dim)), exp_adapter, base_output_proj=w3_base
    )
    assert noop_exp["effective_cancel_rank"] == 0
    _, zero_spec = project_downstream_lora_b_orthogonal_to_anchor(
        anchor_deltas, w_b, protect_specialist_deltas=torch.zeros((2, dim))
    )
    assert zero_spec["protected_specialist_rank"] == 0
    _, skip_inter = project_downstream_lora_b_orthogonal_to_anchor(
        anchor_deltas, exp_adapter, max_rank_proj=8
    )
    assert skip_inter["skipped_factors"] == ["w2_B"]
    _, noop_covered = project_downstream_lora_b_orthogonal_to_anchor(
        u_spec_1.unsqueeze(0), w_b, protect_specialist_deltas=spec_deltas
    )
    assert noop_covered["effective_cancel_rank"] == 0

    # Full-rank specialist: (a) anchor weaker than specialist -> mu <= 1.0 noop; (b) strong anisotropic -> op_norm clamp
    full_spec_strong = torch.eye(dim, dtype=torch.float32) * 3.0
    _, full_noop = project_downstream_lora_b_orthogonal_to_anchor(
        u_anchor.unsqueeze(0) * 0.5, w_b, protect_specialist_deltas=full_spec_strong, max_rank_proj=4
    )
    assert full_noop["used_exact_specialist_nullspace"] is False
    assert full_noop["effective_cancel_rank"] == 0

    aniso_spec = torch.eye(dim, dtype=torch.float32) * 0.05
    aniso_spec[0, 0] = 2.0
    aniso_spec[1, 1] = 0.02
    aniso_spec[16:24, 16:24] = torch.eye(8, dtype=torch.float32) * 4.0
    aniso_anchor = torch.stack(
        [
            40.0 * torch.nn.functional.normalize(0.8 * torch.eye(dim)[0] + 0.6 * torch.eye(dim)[1], dim=0),
            35.0 * torch.nn.functional.normalize(0.8 * torch.eye(dim)[0] + 0.6 * torch.eye(dim)[1], dim=0),
        ],
        dim=0,
    )
    _, full_stats = project_downstream_lora_b_orthogonal_to_anchor(
        aniso_anchor, w_b, protect_specialist_deltas=aniso_spec, max_rank_proj=4
    )
    assert full_stats["used_exact_specialist_nullspace"] is False
    assert full_stats["effective_cancel_rank"] >= 1

    # Also test max_spec_probe_chunks < total specialist chunks for bounded collection branch
    model, _ = _make_8layer_cross_layer_moe(seed=20261127, plant_w2_b_interference=True)
    probe_chunks = _make_domain_chunks(seq_len=16, score_from=8, chunks_per_domain=6, seed_offset=200)
    _, rec_capped = optimize_cross_layer_residual_surgery(
        model,
        probe_chunks,
        config=CrossLayerResidualConfig(score_from=8, max_spec_probe_chunks=3, inplace=False),
    )
    assert rec_capped["accepted"] is True
    spec_ids = torch.tensor(
        [c["ids"] for c in probe_chunks if not c["expert_id"].startswith("E0_")], dtype=torch.long
    )

    def _collect_downstream_spec_deltas(m: CrowdV1WithEngram) -> list[torch.Tensor]:
        cos, sin = m.cos_cached, m.sin_cached
        xb = compute_embeddings_batch(m, spec_ids)
        deltas: list[torch.Tensor] = []
        for l_idx, blk in enumerate(m.blocks):
            xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
            if l_idx in (6, 7):
                d_out = forward_block_dense_base_mlp(blk, mi)
                m_out = blk.mlp(mi)
                deltas.append((m_out - d_out)[:, 7:-1, :].detach().clone())
            xb = forward_block_post_mlp(blk, xm, mi)
        return deltas

    with torch.no_grad():
        spec_deltas_before = _collect_downstream_spec_deltas(model)
    _, rec_e2e = optimize_cross_layer_residual_surgery(model, probe_chunks, score_from=8, inplace=True)
    assert rec_e2e["accepted"] is True
    with torch.no_grad():
        spec_deltas_after = _collect_downstream_spec_deltas(model)
    for d_b, d_a in zip(spec_deltas_before, spec_deltas_after, strict=True):
        assert float((d_a - d_b).abs().max().item()) < 1e-5


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Rollback on Infeasible Guard Violation & Non-Finite States
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_rollback() -> None:
    """S4: Trigger ANSE Maximum Pain (>= 1e6) and verify exact bitwise atomic rollback."""
    # Case 4A: Real MoE where specialist overlaps anchor AND protect_specialist_subspace=False breaks +1% guard
    model, _ = _make_8layer_cross_layer_moe(
        seed=20261128,
        plant_w2_b_interference=False,
        overlap_spec_with_anchor=True,
    )
    probe_chunks = _make_domain_chunks(seq_len=16, score_from=8, chunks_per_domain=6, seed_offset=300)
    parent_sha = compute_module_state_sha256(model)

    cfg_unguarded = CrossLayerResidualConfig(
        score_from=8,
        protect_specialist_subspace=False,
        max_domain_drift=0.01,
        inplace=True,
    )
    _, rec_guard = optimize_cross_layer_residual_surgery(model, probe_chunks, config=cfg_unguarded)
    assert rec_guard["accepted"] is False
    assert rec_guard["rolled_back"] is True
    assert rec_guard["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert rec_guard["guard_ok"] is False
    assert rec_guard["worst_domain_drift_pct"] > 1.0
    assert rec_guard["anse_energy"] >= DEFAULT_ANSE_MAXIMUM_PAIN
    assert rec_guard["delta_energy"] > 0.0
    assert rec_guard["active_sha256"] == parent_sha
    assert compute_module_state_sha256(model) == parent_sha
    assert verify_cross_layer_receipt_provenance(rec_guard) is True

    # Case 4B: Non-finite candidate eval / eval_fn failure triggers Maximum Pain + rollback
    call_state = {"calls": 0}

    def _failing_eval(_m: nn.Module) -> dict[str, float]:
        call_state["calls"] += 1
        if call_state["calls"] == 1:
            return {"E0_general_fineweb": 1.5, "E1_python_systems_code": 1.2, "E2_math_reasoning": 1.2}
        return {"E0_general_fineweb": float("nan"), "E1_python_systems_code": 1.2, "E2_math_reasoning": 1.2}

    _, rec_nan = optimize_cross_layer_residual_surgery(
        model,
        probe_chunks,
        baseline_domain_nll={"E0_general_fineweb": 1.5, "E1_python_systems_code": 1.2, "E2_math_reasoning": 1.2},
        eval_fn=_failing_eval,
        score_from=8,
        inplace=True,
    )
    assert rec_nan["accepted"] is False
    assert rec_nan["rolled_back"] is True
    assert rec_nan["status"] == "ROLLED_BACK_NON_FINITE"
    assert rec_nan["anse_energy"] == DEFAULT_ANSE_MAXIMUM_PAIN
    assert compute_module_state_sha256(model) == parent_sha

    def _raising_eval(_m: nn.Module) -> dict[str, float]:
        call_state["calls"] += 1
        if call_state["calls"] == 3:
            return {"E0_general_fineweb": 1.5, "E1_python_systems_code": 1.2, "E2_math_reasoning": 1.2}
        raise RuntimeError("eval boom")

    _, rec_err = optimize_cross_layer_residual_surgery(
        model, probe_chunks, eval_fn=_raising_eval, score_from=8, inplace=True
    )
    assert rec_err["status"] == "ROLLED_BACK_NON_FINITE"
    assert compute_module_state_sha256(model) == parent_sha

    for bad_cand_val in ("not_a_mapping", {"E0_general_fineweb": True, "E1_python_systems_code": 1.2, "E2_math_reasoning": 1.2}):
        eval_calls = {"n": 0}

        def _bad_cand_eval(_m: nn.Module, payload: Any = bad_cand_val, state: dict[str, int] = eval_calls) -> Any:
            state["n"] += 1
            if state["n"] == 1:
                return {"E0_general_fineweb": 1.5, "E1_python_systems_code": 1.2, "E2_math_reasoning": 1.2}
            return payload

        _, rec_bad_cand = optimize_cross_layer_residual_surgery(
            model, probe_chunks, eval_fn=_bad_cand_eval, score_from=8, inplace=True
        )
        assert rec_bad_cand["status"] == "ROLLED_BACK_NON_FINITE"
        assert compute_module_state_sha256(model) == parent_sha

    # Case 4C: Zero energy improvement (proj_strength=0.0) triggers clean rollback
    _, rec_no_gain = optimize_cross_layer_residual_surgery(
        model,
        probe_chunks,
        config=CrossLayerResidualConfig(score_from=8, proj_strength=0.0, inplace=True),
    )
    assert rec_no_gain["accepted"] is False
    assert rec_no_gain["rolled_back"] is True
    assert rec_no_gain["status"] == "ROLLED_BACK_NO_ENERGY_IMPROVEMENT"
    assert compute_module_state_sha256(model) == parent_sha

    # Case 4D: Tampered rollback verification raises RuntimeError
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _restore_model_snapshot_bitwise(model, {}, "0" * 64)


# ---------------------------------------------------------------------------
# S5: Validation Errors, Provenance Tamper Checks & Security/Stub Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_compliance_gates() -> None:
    """S5: Exercise validation errors, receipt tampering, ZeroStubAudit, PRSecurityScanner, and gatekeeper."""
    # Config & threshold validations
    with pytest.raises(TypeError, match="int"):
        CrossLayerResidualConfig(score_from=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        CrossLayerResidualConfig(score_from=-1)
    with pytest.raises(TypeError, match="real number"):
        CrossLayerResidualConfig(anchor_threshold_nats=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="<= 0.0"):
        CrossLayerResidualConfig(anchor_threshold_nats=0.01)
    with pytest.raises(ValueError, match="strictly <"):
        CrossLayerResidualConfig(anchor_threshold_nats=0.0, harm_threshold_nats=0.0)
    with pytest.raises(ValueError, match=">= 1"):
        CrossLayerResidualConfig(max_rank_proj=0)
    with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
        CrossLayerResidualConfig(proj_strength=1.5)
    with pytest.raises(ValueError, match="< 1.0"):
        CrossLayerResidualConfig(anchor_sv_rel_tol=1.0)
    with pytest.raises(ValueError, match="general_prefix"):
        CrossLayerResidualConfig(general_prefix=" ")
    with pytest.raises(ValueError, match=">= 0.0"):
        CrossLayerResidualConfig(max_domain_drift=-0.1)
    with pytest.raises(ValueError, match="> 0.0"):
        CrossLayerResidualConfig(anse_safe_e0_ratio=0.0)
    with pytest.raises(TypeError, match="bool"):
        CrossLayerResidualConfig(inplace=1)  # type: ignore[arg-type]

    # identify_beneficial_anchor_and_interfering_layers input variants & errors
    dict_part = identify_beneficial_anchor_and_interfering_layers(
        {"0": -0.05, 1: 0.0, 2: 0.02}, anchor_threshold_nats=-0.015, harm_threshold_nats=0.002
    )
    assert dict_part["beneficial_anchor_layers"] == [0]
    assert dict_part["neutral_layers"] == [1]
    assert dict_part["downstream_interfering_layers"] == [2]

    attr_part = identify_beneficial_anchor_and_interfering_layers(
        {"layers": [{"layer": 0, "general_harm_nats": -0.03}, {"layer": 1, "e0_harm_nats": 0.01}]}
    )
    assert attr_part["anchor_to_downstream_map"] == {"0": [1]}

    for bad_thresh_kw in (
        {"anchor_threshold_nats": 0.02},
        {"anchor_threshold_nats": 0.0, "harm_threshold_nats": 0.0},
    ):
        with pytest.raises(ValueError):
            identify_beneficial_anchor_and_interfering_layers([0.0], **bad_thresh_kw)
    with pytest.raises(ValueError, match="empty"):
        identify_beneficial_anchor_and_interfering_layers([])
    with pytest.raises(ValueError, match="empty"):
        identify_beneficial_anchor_and_interfering_layers({})
    with pytest.raises(TypeError, match="Sequence or Mapping"):
        identify_beneficial_anchor_and_interfering_layers("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bool"):
        identify_beneficial_anchor_and_interfering_layers({True: 0.1})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="Invalid layer"):
        identify_beneficial_anchor_and_interfering_layers({"abc": 0.1})
    with pytest.raises(ValueError, match="Duplicate"):
        identify_beneficial_anchor_and_interfering_layers({0: 0.1, "0": 0.2})
    with pytest.raises(ValueError, match="Duplicate"):
        identify_beneficial_anchor_and_interfering_layers(
            [{"layer": 0, "e0_harm_nats": 0.0}, {"layer": 0, "e0_harm_nats": 0.1}]
        )
    with pytest.raises(ValueError, match="containing 'layer'"):
        identify_beneficial_anchor_and_interfering_layers([{"e0_harm_nats": 0.1}])
    with pytest.raises(ValueError, match="e0_harm_nats"):
        identify_beneficial_anchor_and_interfering_layers([{"layer": 0}])
    with pytest.raises(ValueError, match="finite"):
        identify_beneficial_anchor_and_interfering_layers([float("nan")])

    # project_downstream_lora_b_orthogonal_to_anchor validation errors
    a_ok = torch.ones((4, 8))
    b_ok = torch.ones((8, 2))
    with pytest.raises(TypeError, match="torch.Tensor"):
        project_downstream_lora_b_orthogonal_to_anchor("bad", b_ok)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        project_downstream_lora_b_orthogonal_to_anchor([], b_ok)
    with pytest.raises(ValueError, match="empty"):
        project_downstream_lora_b_orthogonal_to_anchor({}, b_ok)
    with pytest.raises(TypeError, match="torch.Tensor"):
        project_downstream_lora_b_orthogonal_to_anchor(["bad"], b_ok)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="2D or 3D"):
        project_downstream_lora_b_orthogonal_to_anchor(torch.ones(8), b_ok)
    with pytest.raises(TypeError, match="floating-point"):
        project_downstream_lora_b_orthogonal_to_anchor(torch.ones((2, 8), dtype=torch.long), b_ok)
    with pytest.raises(ValueError, match="finite"):
        project_downstream_lora_b_orthogonal_to_anchor(torch.tensor([[float("inf"), 1.0]]), b_ok)
    with pytest.raises(ValueError, match="match expected width"):
        project_downstream_lora_b_orthogonal_to_anchor([torch.ones((2, 8)), torch.ones((2, 6))], b_ok)
    with pytest.raises(ValueError, match="must match anchor_deltas width"):
        project_downstream_lora_b_orthogonal_to_anchor(a_ok, b_ok, protect_specialist_deltas=torch.ones((2, 6)))
    with pytest.raises(ValueError, match="< 1.0"):
        project_downstream_lora_b_orthogonal_to_anchor(a_ok, b_ok, anchor_sv_rel_tol=1.0)
    with pytest.raises(ValueError, match="2D tensor"):
        project_downstream_lora_b_orthogonal_to_anchor(a_ok, torch.ones(8))
    with pytest.raises(TypeError, match="floating-point"):
        project_downstream_lora_b_orthogonal_to_anchor(a_ok, torch.ones((8, 2), dtype=torch.long))
    with pytest.raises(ValueError, match="finite"):
        project_downstream_lora_b_orthogonal_to_anchor(a_ok, torch.full((8, 2), float("nan")))
    with pytest.raises(ValueError, match="base_output_proj"):
        project_downstream_lora_b_orthogonal_to_anchor(a_ok, torch.ones((12, 2)))
    with pytest.raises(ValueError, match="base_output_proj shape"):
        project_downstream_lora_b_orthogonal_to_anchor(a_ok, torch.ones((12, 2)), base_output_proj=torch.ones((8, 10)))
    with pytest.raises(ValueError, match="empty"):
        project_downstream_lora_b_orthogonal_to_anchor(a_ok, {})
    with pytest.raises(TypeError, match="torch.Tensor"):
        project_downstream_lora_b_orthogonal_to_anchor(a_ok, {"k": "not_tensor"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="output dim"):
        project_downstream_lora_b_orthogonal_to_anchor(a_ok, {"k": torch.ones((12, 2))})
    with pytest.raises(TypeError, match="torch.Tensor"):
        project_downstream_lora_b_orthogonal_to_anchor(a_ok, 123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must match anchor width"):
        project_downstream_lora_b_orthogonal_to_anchor(a_ok, nn.Linear(4, 10))
    with pytest.raises(ValueError, match="must expose"):
        project_downstream_lora_b_orthogonal_to_anchor(a_ok, nn.ReLU())
    with pytest.raises(ValueError, match="non-empty 'experts'"):
        empty_moe = nn.Module()
        empty_moe.experts = nn.ModuleDict()
        project_downstream_lora_b_orthogonal_to_anchor(a_ok, empty_moe)
    with pytest.raises(ValueError, match="No matching LoRA B"):
        no_lora_moe = nn.Module()
        no_lora_moe.experts = nn.ModuleDict({"0": nn.Linear(4, 4)})
        project_downstream_lora_b_orthogonal_to_anchor(a_ok, no_lora_moe)
    with pytest.raises(ValueError, match="pass base_output_proj"):
        class _InterOnly(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w2_B = nn.Parameter(torch.ones((12, 2)))

        project_downstream_lora_b_orthogonal_to_anchor(a_ok, _InterOnly())

    # optimize_cross_layer_residual_surgery & receipt validations
    model, _ = _make_8layer_cross_layer_moe(num_layers=2, anchor_layer=0, interfering_layers=(1,))
    chunks = _make_domain_chunks(seq_len=16, score_from=8, chunks_per_domain=4, seed_offset=400)
    with pytest.raises(TypeError, match="nn.Module"):
        optimize_cross_layer_residual_surgery("bad", chunks)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="blocks"):
        bad_m = nn.Module()
        bad_m.blocks = nn.ModuleList()
        optimize_cross_layer_residual_surgery(bad_m, chunks)
    with pytest.raises(ValueError, match="must expose"):
        bad_m2 = nn.Module()
        bad_m2.blocks = nn.ModuleList([nn.Linear(4, 4)])
        optimize_cross_layer_residual_surgery(bad_m2, chunks)
    with pytest.raises(TypeError, match="callable"):
        optimize_cross_layer_residual_surgery(model, chunks, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="both be provided"):
        optimize_cross_layer_residual_surgery(model, chunks, cal_ln_w=torch.ones(32))
    with pytest.raises(TypeError, match="CrossLayerResidualConfig"):
        optimize_cross_layer_residual_surgery(model, chunks, config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.Tensor"):
        optimize_cross_layer_residual_surgery(model, chunks, cal_ln_w="a", cal_ln_b="b")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        optimize_cross_layer_residual_surgery(model, chunks, cal_ln_w=torch.ones(4), cal_ln_b=torch.ones(4))
    with pytest.raises(ValueError, match="finite"):
        optimize_cross_layer_residual_surgery(
            model, chunks, cal_ln_w=torch.full((32,), float("nan")), cal_ln_b=torch.zeros(32)
        )
    with pytest.raises(ValueError, match="non-empty sequence"):
        optimize_cross_layer_residual_surgery(model, [])
    with pytest.raises(ValueError, match="expert_id"):
        optimize_cross_layer_residual_surgery(model, [{"ids": [0, 1, 2]}], score_from=1)
    with pytest.raises(ValueError, match="non-empty string"):
        optimize_cross_layer_residual_surgery(model, [{"expert_id": " ", "ids": [0, 1]}], score_from=1)
    with pytest.raises(ValueError, match="length >= 2"):
        optimize_cross_layer_residual_surgery(model, [{"expert_id": "E0_a", "ids": [0]}], score_from=0)
    with pytest.raises(ValueError, match="token IDs"):
        optimize_cross_layer_residual_surgery(model, [{"expert_id": "E0_a", "ids": [0, 999]}], score_from=0)
    with pytest.raises(ValueError, match="share sequence length"):
        optimize_cross_layer_residual_surgery(
            model, [{"expert_id": "E0_a", "ids": [0, 1]}, {"expert_id": "E0_a", "ids": [0, 1, 2]}], score_from=0
        )
    with pytest.raises(ValueError, match="too short"):
        optimize_cross_layer_residual_surgery(model, [{"expert_id": "E0_a", "ids": [0, 1]}], score_from=8)
    with pytest.raises(ValueError, match="general chunk"):
        optimize_cross_layer_residual_surgery(model, [{"expert_id": "E1_a", "ids": [0, 1, 2]}], score_from=0)
    with pytest.raises(ValueError, match="baseline_domain_nll"):
        optimize_cross_layer_residual_surgery(
            model, chunks, baseline_domain_nll={"wrong": 1.0}, score_from=8
        )
    with pytest.raises(ValueError, match="Parent model evaluation"):
        optimize_cross_layer_residual_surgery(
            model, chunks, eval_fn=lambda _m: {"E0_general_fineweb": -1.0}, score_from=8
        )
    # Non-finite hidden output in final block hits _compute_domain_nlls_from_chunks finiteness check
    inf_model, _ = _make_8layer_cross_layer_moe(num_layers=2, anchor_layer=0, interfering_layers=(1,))
    with torch.no_grad():
        inf_model.blocks[-1].mlp.base_mlp.w1.weight.fill_(1e20)
        inf_model.blocks[-1].mlp.base_mlp.w2.weight.fill_(1e20)
        inf_model.blocks[-1].mlp.base_mlp.w3.weight.fill_(1e20)
    with pytest.raises(ValueError, match="non-finite"):
        optimize_cross_layer_residual_surgery(inf_model, chunks, score_from=8)
    with pytest.raises(TypeError, match="mapping"):
        verify_cross_layer_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_cross_layer_receipt_provenance({"receipt_sha256": "xyz"}) is False
    assert verify_cross_layer_receipt_provenance({"receipt_sha256": "0" * 64, "bad": float("nan")}) is False
    with pytest.raises(TypeError, match="nn.Module"):
        compute_module_state_sha256("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        compute_module_state_sha256(nn.ReLU())
    with pytest.raises(ValueError, match="non-finite"):
        bad_lin = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            bad_lin.weight.fill_(float("nan"))
        compute_module_state_sha256(bad_lin)

    # Security, ZeroStubAudit (with CI-safe fallback), lateral movement, and pr_gatekeeper FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn10", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn10", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/cross_layer_residual_nullspace.py", TARGET_MODULE_PATH),
        ("tests/test_cross_layer_residual_nullspace_scenarios.py", TEST_FILE_PATH),
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
        {"nll_mean": 1.18, "anchor_gain_nats": 0.2516, "anse_energy": -0.042, "accepted": True},
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(f"diff --git a/x.py b/x.py\n+++ b/x.py\n+RECEIPT = {sample_receipt_json!r}") == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1110,
        pr_title="feat(swarm): cross-layer residual interference cancellation and beneficial-anchor layer protection for Sparse MoEs",
        pr_body="Implements NEURAL-CTRL-DYN-10 cross-layer residual nullspace controller and deterministic CPU tests.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): cross-layer residual interference cancellation and beneficial-anchor layer protection for Sparse MoEs"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
