# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-01: self-rewriting weight-block scanner and ANSE spectral repair."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from gwaya.ast_audit import ZeroStubAudit
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdRuntimeMLP,
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdLoRAExpertAdapter,
    CrowdLoRAMoEMLP,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.self_healing_weight_repair import (
    DEFAULT_MAX_EDITED_PARAM_RATIO,
    AdapterHealthScanConfig,
    RepairProposalConfig,
    SelfHealingLoopConfig,
    apply_surgical_block_repairs_inplace,
    compute_moe_state_sha256,
    execute_self_rewriting_repair_loop,
    propose_surgical_block_repairs,
    restore_moe_state_inplace,
    scan_moe_adapter_health,
    snapshot_moe_state,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "self_healing_weight_repair.py"
TEST_FILE_PATH = Path(__file__).resolve()
EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."


def _load_module_from_path(name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_orthogonal_basis(dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mat = torch.randn((dim, dim), generator=gen, dtype=torch.float64)
    q_mat, r_mat = torch.linalg.qr(mat)
    signs = torch.sign(torch.diagonal(r_mat))
    signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
    return (q_mat * signs.unsqueeze(0)).contiguous()


def _build_subspace_moe_model(
    *,
    vocab_size: int = 32,
    hidden_size: int = 32,
    num_layers: int = 2,
    intermediate_size: int = 64,
    num_experts: int = 4,
    top_k: int = 2,
    lora_rank: int = 8,
    engram_buckets: int = 65_536,
    seed: int = 20261123,
) -> CrowdV1WithEngram:
    """Build a deterministic subspace-separated ``CrowdV1WithEngram`` + ``CrowdLoRAMoEMLP``."""
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
        engram_buckets=engram_buckets,
        tie_word_embeddings=True,
    )
    model = CrowdV1WithEngram(cfg)
    half_v = vocab_size // 2
    half_h = hidden_size // 2
    half_i = intermediate_size // 2

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        p_raw = torch.randn((half_v, half_h), generator=gen, dtype=torch.float32)
        p_raw = p_raw - p_raw.mean(dim=1, keepdim=True)
        s_raw = torch.randn((vocab_size - half_v, hidden_size - half_h), generator=gen, dtype=torch.float32)
        s_raw = s_raw - s_raw.mean(dim=1, keepdim=True)
        wte[:half_v, :half_h] = p_raw / p_raw.norm(dim=1, keepdim=True).clamp_min(1e-6) * 2.0
        wte[half_v:, half_h:] = s_raw / s_raw.norm(dim=1, keepdim=True).clamp_min(1e-6) * 2.0
        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight

        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0)
            blk.ln2.bias.zero_()
            blk.attn.q.weight[:half_h, :half_h].copy_(0.05 * torch.eye(half_h))
            blk.attn.k.weight[: half_h // 2, :half_h].copy_(0.05 * torch.eye(half_h // 2, half_h))
            blk.attn.v.weight[: half_h // 2, :half_h].copy_(0.05 * torch.eye(half_h // 2, half_h))
            blk.attn.o.weight[:half_h, :half_h].copy_(0.05 * torch.eye(half_h))
            blk.mlp.w1.weight[:half_i, :half_h].copy_(torch.randn((half_i, half_h), generator=gen) * 0.15)
            blk.mlp.w2.weight[:half_i, :half_h].copy_(torch.randn((half_i, half_h), generator=gen) * 0.15)
            blk.mlp.w3.weight[:half_h, :half_i].copy_(torch.randn((half_h, half_i), generator=gen) * 0.10)
            blk.mlp.w1.weight[half_i:, half_h:].copy_(
                torch.randn((intermediate_size - half_i, hidden_size - half_h), generator=gen) * 0.15
            )
            blk.mlp.w2.weight[half_i:, half_h:].copy_(
                torch.randn((intermediate_size - half_i, hidden_size - half_h), generator=gen) * 0.15
            )
            blk.mlp.w3.weight[half_h:, half_i:].copy_(
                torch.randn((hidden_size - half_h, intermediate_size - half_i), generator=gen) * 0.10
            )

    upcycle_crowd_v1_to_sparse_moe(
        model,
        num_experts=num_experts,
        top_k=top_k,
        lora_rank=lora_rank,
        routing_mode="sigma_topk",
        seed=seed,
    )
    with torch.no_grad():
        for l_idx, blk in enumerate(model.blocks):
            blk.mlp.router.weight.zero_()
            blk.mlp.router.weight[0, :half_h].fill_(0.8)
            blk.mlp.router.weight[0, half_h:].fill_(-0.8)
            blk.mlp.router.weight[1:, :half_h].fill_(-0.8)
            blk.mlp.router.weight[1:, half_h:].fill_(0.8)
            for e_key, exp in blk.mlp.experts.items():
                e_gen = torch.Generator(device="cpu").manual_seed(seed + l_idx * 100 + int(e_key))
                q_in = _make_orthogonal_basis(hidden_size, seed=seed + l_idx * 30 + int(e_key))[:, :lora_rank]
                q_mid = _make_orthogonal_basis(intermediate_size, seed=seed + l_idx * 50 + int(e_key))[:, :lora_rank]
                s_base = torch.linspace(0.35, 0.20, lora_rank, dtype=torch.float64)
                exp.w1_B.copy_((q_mid * s_base.sqrt().unsqueeze(0)).float())
                exp.w1_A.copy_((s_base.sqrt().unsqueeze(1) * q_in.T).float())
                exp.w2_B.copy_((q_mid * s_base.sqrt().unsqueeze(0)).float())
                exp.w2_A.copy_((s_base.sqrt().unsqueeze(1) * q_in.T).float())
                exp.w3_B.copy_((q_in * s_base.sqrt().unsqueeze(0)).float())
                exp.w3_A.copy_((s_base.sqrt().unsqueeze(1) * q_mid.T).float())
                # Keep deterministic generator consumption explicit for pylint/reproducibility
                _ = torch.randn(1, generator=e_gen)

    return model


def _build_probe_chunks(
    *,
    vocab_size: int = 32,
    seq_len: int = 16,
    chunks_per_domain: int = 4,
    seed: int = 20260001,
) -> list[dict[str, Any]]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    half_v = vocab_size // 2
    chunks: list[dict[str, Any]] = []
    for i in range(chunks_per_domain):
        e0_ids = torch.randint(0, half_v, (seq_len,), generator=gen, dtype=torch.long).tolist()
        e1_ids = torch.randint(half_v, vocab_size, (seq_len,), generator=gen, dtype=torch.long).tolist()
        chunks.append({"expert_id": "E0_general_fineweb", "dataset_id": "ds0", "row": i, "ids": e0_ids})
        chunks.append({"expert_id": "E1_python_systems_code", "dataset_id": "ds1", "row": i, "ids": e1_ids})
    return chunks


def _build_planted_spectral_factors(
    *,
    d_out: int,
    d_in: int,
    singular_values: list[float],
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rank = len(singular_values)
    u_modes = _make_orthogonal_basis(d_out, seed=seed + 1)[:, :rank]
    v_modes = _make_orthogonal_basis(d_in, seed=seed + 2)[:, :rank]
    s_t = torch.tensor(singular_values, dtype=torch.float64)
    s_sqrt = s_t.sqrt()
    w_b = (u_modes * s_sqrt.unsqueeze(0)).to(dtype=dtype).contiguous()
    w_a = (s_sqrt.unsqueeze(1) * v_modes.T).to(dtype=dtype).contiguous()
    return w_b, w_a, u_modes, v_modes


# ---------------------------------------------------------------------------
# S1: Injected NaN/Inf & BF16 Exponent Bit-Flip Corruption Detection & Repair
# ---------------------------------------------------------------------------


def test_s1_nan_inf_and_bf16_bitflip_corruption_repaired_to_finite_nll_recovery() -> None:
    """Detect NaN/Inf and BF16 exponent bit-flips in MoE experts and repair to 100% finite outputs."""
    model = _build_subspace_moe_model(
        vocab_size=32,
        hidden_size=32,
        num_layers=2,
        intermediate_size=64,
        num_experts=4,
        top_k=2,
        lora_rank=4,
        engram_buckets=65_536,
        seed=20260101,
    )
    chunks = _build_probe_chunks(vocab_size=32, seq_len=16, chunks_per_domain=4, seed=20260102)
    ids_all = torch.tensor([c["ids"] for c in chunks], dtype=torch.long)

    scan_cfg = AdapterHealthScanConfig(score_from=4, batch_size=8)
    baseline_health = scan_moe_adapter_health(model, config=scan_cfg, probe_chunks=chunks)
    assert baseline_health["healthy"] is True
    assert baseline_health["total_anomalies"] == 0

    with torch.no_grad():
        base_logits, _, _ = model(ids_all)
        assert bool(torch.isfinite(base_logits).all().item())

        # Inject NaN + Inf into layer 0 expert 1, and a genuine BF16 exponent bit-flip (^ 0x4000) into layer 1 expert 2.
        exp1_l0 = model.blocks[0].mlp.experts["1"]
        exp1_l0.w1_A[0, 3] = float("nan")
        exp1_l0.w3_B[5, 1] = float("inf")

        exp2_l1 = model.blocks[1].mlp.experts["2"]
        bf16_w2_a = exp2_l1.w2_A.detach().to(torch.bfloat16).clone()
        # Flip BF16 exponent MSB (bit 14) on a normal-magnitude weight (< 1.0 -> ~1e37 finite overflow!),
        # then load back into fp32 CrowdLoRAExpertAdapter just like safetensors bf16 checkpoint loading.
        bf16_w2_a[1, 4] = torch.tensor(0.125, dtype=torch.bfloat16)
        raw_i16 = bf16_w2_a.view(torch.int16)
        raw_i16[1, 4] ^= 0x4000
        assert bool(torch.isfinite(bf16_w2_a[1, 4]).item())
        assert float(bf16_w2_a[1, 4].abs().item()) > 1e30
        exp2_l1.w2_A.copy_(bf16_w2_a.to(torch.float32))

        # Inject expert corruption first and verify forward pass returns non-finite logits.
        corrupted_logits, _, _ = model(ids_all)
        assert not bool(torch.isfinite(corrupted_logits).all().item())

        # Also corrupt one router weight entry to exercise corrupted_router_rows path.
        model.blocks[0].mlp.router.weight[3, 2] = float("-inf")
        with pytest.raises(ValueError, match="finite"):
            model(ids_all)

    corrupted_health = scan_moe_adapter_health(model, config=scan_cfg, probe_chunks=chunks)
    assert corrupted_health["healthy"] is False
    assert len(corrupted_health["non_finite_or_bitflip_adapters"]) == 3
    assert len(corrupted_health["corrupted_router_rows"]) == 1

    # Part A: Elementwise sparse bit-flip rescue first on a cloned model, verifying exact coordinate cleanup.
    model_elem = _build_subspace_moe_model(
        vocab_size=32, hidden_size=32, num_layers=2, intermediate_size=64, num_experts=4,
        top_k=2, lora_rank=4, engram_buckets=65_536, seed=20260101,
    )
    restore_moe_state_inplace(model_elem, snapshot_moe_state(model))
    elem_res = execute_self_rewriting_repair_loop(
        model_elem,
        probe_chunks=chunks,
        scan_config=scan_cfg,
        proposal_config=RepairProposalConfig(quarantine_strategy="elementwise"),
        loop_config=SelfHealingLoopConfig(score_from=4, batch_size=8),
    )
    assert elem_res["status"] == "accepted_and_repaired"
    assert elem_res["final_all_finite"] is True
    assert elem_res["total_modified_elements"] == 4
    assert elem_res["parameter_fraction_modified"] < 1e-5

    # Part B: Default finite zero-perturbation quarantine on main model (sanitizes A + sets w1_B=w2_B=w3_B=0).
    loop_res = execute_self_rewriting_repair_loop(
        model,
        probe_chunks=chunks,
        scan_config=scan_cfg,
        proposal_config=RepairProposalConfig(quarantine_strategy="auto"),
        loop_config=SelfHealingLoopConfig(score_from=4, batch_size=8),
    )
    assert loop_res["status"] == "accepted_and_repaired"
    assert loop_res["initial_all_finite"] is False
    assert loop_res["final_all_finite"] is True
    assert loop_res["final_nll_mean"] is not None
    assert loop_res["delta_anse_energy"] < -999_000.0
    assert loop_res["within_budget"] is True
    assert loop_res["parameter_fraction_modified"] < DEFAULT_MAX_EDITED_PARAM_RATIO
    assert loop_res["final_health"]["healthy"] is True

    with torch.no_grad():
        repaired_logits, _, _ = model(ids_all)
        assert bool(torch.isfinite(repaired_logits).all().item())
        # Quarantined experts satisfy exact finite dy_e(x) == 0 (no 0 * NaN == NaN).
        x_probe = torch.randn((8, 32), generator=torch.Generator(device="cpu").manual_seed(20260103))
        h1 = torch.zeros((8, 64))
        h2 = torch.zeros((8, 64))
        dy1 = model.blocks[0].mlp.experts["1"].forward_delta(x_probe, h1, h2, model.blocks[0].mlp.base_mlp.w3.weight)
        dy2 = model.blocks[1].mlp.experts["2"].forward_delta(x_probe, h1, h2, model.blocks[1].mlp.base_mlp.w3.weight)
        assert bool(torch.isfinite(dy1).all().item()) and float(dy1.abs().max().item()) == 0.0
        # Under strict 0.05% budget (1,063 params), expert 1 gets full finite zero-perturbation quarantine (641 params)
        # while expert 2 automatically falls back to surgical elementwise bit-flip sanitization (1 param) -> finite output.
        assert bool(torch.isfinite(dy2).all().item())
        # Verify NLL recovered from NaN/Inf/overflow down to finite baseline range.
        assert loop_res["final_nll_mean"] < 10.0


# ---------------------------------------------------------------------------
# S2: 100x Outlier Singular-Mode Spike Repaired via QR-SVD Winsorization
# ---------------------------------------------------------------------------


def test_s2_100x_outlier_spectral_spike_repaired_via_qr_svd_winsorization() -> None:
    """Winsorize a 100x singular-mode spike via QR-SVD while preserving healthy modes to <1e-5 error."""
    d_in, d_out = 48, 96
    healthy_svs = [2.4, 2.2, 2.0, 1.8, 1.6, 1.4, 1.2]
    spiked_svs = [200.0, *healthy_svs]  # sigma_1 / median(sigma) ~= 200 / 1.9 ~= 105.3x!

    dense_mlp = CrowdRuntimeMLP(CrowdV1RuntimeConfig(hidden_size=d_in, intermediate_size=d_out))
    with torch.no_grad():
        gen = torch.Generator(device="cpu").manual_seed(20260201)
        for p in dense_mlp.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.1)

    moe = CrowdLoRAMoEMLP(dense_mlp, num_experts=3, top_k=2, rank=8, routing_mode="sigma_topk", seed=20260202)
    b1, a1, _, v_modes = _build_planted_spectral_factors(d_out=d_out, d_in=d_in, singular_values=spiked_svs, seed=20260203)
    b2, a2, _, _ = _build_planted_spectral_factors(
        d_out=d_out, d_in=d_in, singular_values=[2.2, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8], seed=20260204
    )
    b3, a3, _, _ = _build_planted_spectral_factors(
        d_out=d_in, d_in=d_out, singular_values=[2.1, 1.9, 1.7, 1.5, 1.3, 1.1, 0.9, 0.7], seed=20260205
    )
    with torch.no_grad():
        exp0 = moe.experts["0"]
        exp0.w1_B.copy_(b1)
        exp0.w1_A.copy_(a1)
        exp0.w2_B.copy_(b2)
        exp0.w2_A.copy_(a2)
        exp0.w3_B.copy_(b3)
        exp0.w3_A.copy_(a3)
        # Keep experts 1 and 2 healthy (zero B or well-conditioned).
        w1_before = exp0.w1_B.double() @ exp0.w1_A.double()
        b1_snap = exp0.w1_B.detach().clone()

    health = scan_moe_adapter_health(moe)
    assert health["healthy"] is False
    assert len(health["spectral_spike_adapters"]) == 1
    assert len(health["stable_rank_collapsed_adapters"]) == 1
    spike_item = health["spectral_spike_adapters"][0]
    assert spike_item["projection"] == "w1"
    assert spike_item["spectral_outlier_ratio"] > 100.0
    assert spike_item["stable_rank"] < 1.01

    proposal = propose_surgical_block_repairs(
        moe,
        health,
        total_model_params=657_800_000,
    )
    assert proposal["repair_count"] == 1
    assert proposal["repair_kinds_count"] == {"qr_svd_winsorization": 1}
    assert proposal["within_budget"] is True
    assert proposal["parameter_fraction_modified"] < DEFAULT_MAX_EDITED_PARAM_RATIO

    apply_res = apply_surgical_block_repairs_inplace(moe, proposal)
    assert apply_res["applied_repairs"] == 1
    assert torch.equal(moe.experts["0"].w1_B, b1_snap)

    health_after = scan_moe_adapter_health(moe)
    assert health_after["healthy"] is True
    w1_post_meta = health_after["layers"][0]["experts"]["0"]["projections"]["w1"]
    assert w1_post_meta["spectral_outlier_ratio"] <= 4.0
    assert w1_post_meta["stable_rank"] > 1.5

    w1_after = moe.experts["0"].w1_B.double() @ moe.experts["0"].w1_A.double()
    # Healthy modes v_1..v_7 preserved to <1e-5 relative error!
    for j in range(1, 8):
        v_j = v_modes[:, j]
        rel_err = float(((w1_after - w1_before) @ v_j).norm().item() / (w1_before @ v_j).norm().item())
        assert rel_err < 1e-5

    # Also verify single-mode extreme spike (only 1 non-zero mode > single_mode_spike_norm) triggers and heals cleanly.
    exp_single = CrowdLoRAExpertAdapter(d_in, d_out, rank=4, seed=20260206)
    b_s, a_s, _, _ = _build_planted_spectral_factors(
        d_out=d_out, d_in=d_in, singular_values=[80.0, 0.0, 0.0, 0.0], seed=20260207
    )
    with torch.no_grad():
        exp_single.w1_B.copy_(b_s)
        exp_single.w1_A.copy_(a_s)
        moe.experts["1"] = exp_single
    h_single = scan_moe_adapter_health(moe)
    assert any(item["expert_id"] == 1 for item in h_single["spectral_spike_adapters"])

    # Verify stable-rank collapse without extreme single-mode trigger (triggers sigma_cap winsorization)
    # and float64/float16/int8 bitwise hashing paths + nn.Linear-wrapped LoRA factors.
    class _LinearFactorExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w1_A = nn.Linear(16, 6, bias=False, dtype=torch.float64)
            self.w1_B = nn.Linear(6, 32, bias=False, dtype=torch.float64)
            b_lin, a_lin, _, _ = _build_planted_spectral_factors(
                d_out=32, d_in=16, singular_values=[5.0, 1.1, 1.0, 0.9, 0.8, 0.7], seed=20260208, dtype=torch.float64
            )
            with torch.no_grad():
                self.w1_B.weight.copy_(b_lin)
                self.w1_A.weight.copy_(a_lin)

    class _CustomMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = nn.Linear(16, 1, bias=False, dtype=torch.float64)
            self.experts = nn.ModuleDict({"0": _LinearFactorExpert()})
            self.register_buffer("byte_tag", torch.tensor([1], dtype=torch.uint8))

    custom_moe = _CustomMoE()
    with torch.no_grad():
        custom_moe.router.weight.fill_(6.0)  # norm = sqrt(16 * 36) = 24 >= 15.0 (1-expert fallback!)
    h_custom = scan_moe_adapter_health(custom_moe)
    assert len(h_custom["stable_rank_collapsed_adapters"]) == 1
    assert len(h_custom["saturated_router_rows"]) == 1
    p_custom = propose_surgical_block_repairs(custom_moe, h_custom, total_model_params=1_000_000)
    assert p_custom["repair_kinds_count"]["qr_svd_winsorization"] == 1
    assert p_custom["repair_kinds_count"]["saturated_router_recenter"] == 1
    apply_surgical_block_repairs_inplace(custom_moe, p_custom)
    assert scan_moe_adapter_health(custom_moe)["healthy"] is True


# ---------------------------------------------------------------------------
# S3: Saturated sigma_topk Router Row (+25 Logit) Diagnosed & Recentered
# ---------------------------------------------------------------------------


def test_s3_saturated_sigma_topk_router_row_recentered_without_disturbing_specialists() -> None:
    """Diagnose a +25 saturated sigma_topk router row and recenter orthogonal to healthy domains."""
    hidden_size = 32
    num_experts = 5
    dense_mlp = CrowdRuntimeMLP(CrowdV1RuntimeConfig(hidden_size=hidden_size, intermediate_size=64))
    moe = CrowdLoRAMoEMLP(dense_mlp, num_experts=num_experts, top_k=2, rank=4, routing_mode="sigma_topk", seed=20260301)

    gen = torch.Generator(device="cpu").manual_seed(20260302)
    # Disjoint domain subspaces: E0=dims 0..5, E1=6..11, E2=12..17, E3=18..23, E4=24..29 + shared bias carrier dim 30..31.
    domain_acts: dict[str, torch.Tensor] = {}
    dom_names = {
        0: "E0_general_fineweb",
        1: "E1_python_code",
        2: "E2_math_reasoning",
        3: "E3_formal_logic",
        4: "E4_physics_stem",
    }
    for d_idx, d_name in dom_names.items():
        x_d = torch.zeros((64, hidden_size), dtype=torch.float32)
        lo = d_idx * 6
        x_d[:, lo : lo + 6] = torch.randn((64, 6), generator=gen) * 0.25 + 1.5
        x_d[:, 30] = 1.0  # shared common carrier across all domains
        domain_acts[d_name] = x_d

    with torch.no_grad():
        moe.router.weight.zero_()
        for e_idx in range(num_experts):
            lo = e_idx * 6
            moe.router.weight[e_idx, lo : lo + 6].fill_(0.6)
            for other in range(num_experts):
                if other != e_idx:
                    o_lo = other * 6
                    moe.router.weight[e_idx, o_lo : o_lo + 6].fill_(-0.5)
        # Corrupt expert 2 row with a +25.0 logit saturation along the shared carrier dim 30!
        moe.router.weight[2, 30] = 25.0
        healthy_rows_snap = {k: moe.router.weight[k].detach().clone() for k in (0, 1, 3, 4)}

    health = scan_moe_adapter_health(moe, probe_activations=domain_acts)
    assert health["healthy"] is False
    assert len(health["saturated_router_rows"]) == 1
    sat_item = health["saturated_router_rows"][0]
    assert sat_item["expert_id"] == 2
    assert sat_item["min_domain_mean_logit"] > 20.0
    assert sat_item["min_domain_mean_gate"] > 0.999

    # Confirm healthy specialists are disrupted BEFORE repair (expert 2 steals top-k slot on domains 1 and 3!).
    with torch.no_grad():
        g_e1_before = torch.sigmoid(moe.router(domain_acts["E1_python_code"]))
        assert float(g_e1_before[:, 2].mean().item()) > 0.999

    proposal = propose_surgical_block_repairs(
        moe,
        health,
        probe_activations=domain_acts,
        total_model_params=657_800_000,
    )
    assert proposal["repair_count"] == 1
    assert proposal["repair_kinds_count"] == {"saturated_router_recenter": 1}
    assert proposal["within_budget"] is True
    assert proposal["parameter_fraction_modified"] < DEFAULT_MAX_EDITED_PARAM_RATIO
    apply_surgical_block_repairs_inplace(moe, proposal)

    # Verify saturated row is recentered off-domain while own-domain and other specialists stay undisturbed!
    with torch.no_grad():
        for k, snap in healthy_rows_snap.items():
            assert torch.equal(moe.router.weight[k], snap)

        for d_idx, d_name in dom_names.items():
            logits = moe.router(domain_acts[d_name])
            gates = torch.sigmoid(logits)
            _, top_idx = torch.topk(gates, k=2, dim=-1)
            own_hit_rate = float((top_idx == d_idx).any(dim=-1).float().mean().item())
            assert own_hit_rate == pytest.approx(1.0)
            if d_idx != 2:
                assert float(gates[:, 2].mean().item()) < 0.08
                assert float(logits[:, 2].mean().item()) < -2.0
            else:
                assert float(gates[:, 2].mean().item()) > 0.90

    assert scan_moe_adapter_health(moe, probe_activations=domain_acts)["healthy"] is True

    # Also test bias=True saturation + dead router row + collinear common carrier + weight-only fallback.
    moe_bias = CrowdLoRAMoEMLP(dense_mlp, num_experts=4, top_k=2, rank=4, routing_mode="sigma_topk", seed=20260303)
    moe_bias.router = nn.Linear(hidden_size, 4, bias=True)
    with torch.no_grad():
        moe_bias.router.weight.zero_()
        moe_bias.router.bias.zero_()
        for e_idx in range(4):
            lo = e_idx * 6
            moe_bias.router.weight[e_idx, lo : lo + 6].fill_(0.5)
        moe_bias.router.bias[1] = 25.0  # saturated via bias
        moe_bias.router.bias[3] = -25.0  # dead via bias

    bias_acts = {k: domain_acts[k] for k in ("E0_general_fineweb", "E1_python_code", "E2_math_reasoning", "E3_formal_logic")}
    h_bias = scan_moe_adapter_health(moe_bias, probe_activations=bias_acts)
    assert len(h_bias["saturated_router_rows"]) == 1 and h_bias["saturated_router_rows"][0]["expert_id"] == 1
    assert len(h_bias["dead_router_rows"]) == 1 and h_bias["dead_router_rows"][0]["expert_id"] == 3
    p_bias = propose_surgical_block_repairs(moe_bias, h_bias, probe_activations=bias_acts, total_model_params=657_800_000)
    apply_surgical_block_repairs_inplace(moe_bias, p_bias)
    assert scan_moe_adapter_health(moe_bias, probe_activations=bias_acts)["healthy"] is True

    # Weight-only fallback (no probe_activations) + bias-only weightless saturation + dead weight row without bias.
    moe_wo = CrowdLoRAMoEMLP(dense_mlp, num_experts=4, top_k=2, rank=4, routing_mode="sigma_topk", seed=20260304)
    with torch.no_grad():
        moe_wo.router.weight.normal_(std=0.2)
        moe_wo.router.weight[0].fill_(4.0)  # norm = sqrt(32 * 16) = 22.6 >= 15.0
    h_wo = scan_moe_adapter_health(moe_wo)
    assert len(h_wo["saturated_router_rows"]) == 1
    p_wo = propose_surgical_block_repairs(moe_wo, h_wo, total_model_params=657_800_000)
    apply_surgical_block_repairs_inplace(moe_wo, p_wo)
    assert scan_moe_adapter_health(moe_wo)["healthy"] is True

    # Strictly disjoint subspaces (Pass 1 uncentered orthogonal projection succeeds) + dead router without bias + healthy_subspaces.
    moe_ortho = CrowdLoRAMoEMLP(dense_mlp, num_experts=4, top_k=2, rank=4, routing_mode="sigma_topk", seed=20260305)
    ortho_acts: dict[Any, torch.Tensor] = {}
    for d_idx in range(4):
        xd = torch.zeros((48, hidden_size), dtype=torch.float32)
        xd[:, d_idx * 6 : d_idx * 6 + 6] = 1.5
        ortho_acts[d_idx] = xd
    with torch.no_grad():
        moe_ortho.router.weight.zero_()
        for e_idx in range(4):
            moe_ortho.router.weight[e_idx, e_idx * 6 : e_idx * 6 + 6].fill_(0.5)
        moe_ortho.router.weight[1, :6] = 25.0  # saturates on domain 0, orthogonal to domain 1!
        moe_ortho.router.weight[1, 12:24] = 25.0  # saturates on domains 2 and 3, orthogonal to domain 1!
        moe_ortho.router.weight[3, :24] = -5.0  # dead across all domains without bias!
    h_ortho = scan_moe_adapter_health(moe_ortho, probe_activations={0: ortho_acts})
    assert any(r["expert_id"] == 1 for r in h_ortho["saturated_router_rows"])
    assert any(r["expert_id"] == 3 for r in h_ortho["dead_router_rows"])
    p_ortho = propose_surgical_block_repairs(
        moe_ortho,
        h_ortho,
        probe_activations={"0": ortho_acts},
        healthy_subspaces={"0": ortho_acts[1]},
        total_model_params=657_800_000,
    )
    apply_surgical_block_repairs_inplace(moe_ortho, p_ortho)
    assert scan_moe_adapter_health(moe_ortho, probe_activations=ortho_acts)["healthy"] is True

    # Weight-only bias=True saturation & dead rows (no probe_activations).
    moe_b_wo = CrowdLoRAMoEMLP(dense_mlp, num_experts=4, top_k=2, rank=4, routing_mode="sigma_topk", seed=20260306)
    moe_b_wo.router = nn.Linear(hidden_size, 4, bias=True)
    with torch.no_grad():
        moe_b_wo.router.weight.normal_(std=0.1)
        moe_b_wo.router.bias.zero_()
        moe_b_wo.router.bias[0] = 20.0
        moe_b_wo.router.bias[2] = -20.0
    h_b_wo = scan_moe_adapter_health(moe_b_wo)
    assert len(h_b_wo["saturated_router_rows"]) == 1 and len(h_b_wo["dead_router_rows"]) == 1
    p_b_wo = propose_surgical_block_repairs(moe_b_wo, h_b_wo, total_model_params=657_800_000)
    apply_surgical_block_repairs_inplace(moe_b_wo, p_b_wo)
    assert scan_moe_adapter_health(moe_b_wo)["healthy"] is True


# ---------------------------------------------------------------------------
# S4: Closed-Loop Self-Rewriting Accept vs Atomic Bitwise Rollback Guard
# ---------------------------------------------------------------------------


def test_s4_closed_loop_accept_vs_atomic_bitwise_rollback_on_guard_violation() -> None:
    """Verify closed-loop acceptance (Delta E < 0 & drift <= +1.0%) and atomic bitwise rollback on violation."""
    # Part A: Genuine closed-loop accept on CrowdV1WithEngram with planted spectral spike + saturated router.
    model = _build_subspace_moe_model(
        vocab_size=32,
        hidden_size=32,
        num_layers=2,
        intermediate_size=64,
        num_experts=4,
        top_k=2,
        lora_rank=8,
        engram_buckets=65_536,
        seed=20260401,
    )
    chunks = _build_probe_chunks(vocab_size=32, seq_len=16, chunks_per_domain=4, seed=20260402)
    scan_cfg = AdapterHealthScanConfig(score_from=4, batch_size=8)
    loop_cfg = SelfHealingLoopConfig(max_iterations=3, max_domain_drift=0.01, score_from=4, batch_size=8)

    with torch.no_grad():
        # Inject huge 100x spectral spike on E0-overlapping direction in layer 1 expert 1.
        b_sp, a_sp, _, _ = _build_planted_spectral_factors(
            d_out=64,
            d_in=32,
            singular_values=[220.0, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8],
            seed=20260403,
        )
        model.blocks[1].mlp.experts["1"].w1_B.copy_(b_sp)
        model.blocks[1].mlp.experts["1"].w1_A.copy_(a_sp)

    accept_res = execute_self_rewriting_repair_loop(
        model,
        probe_chunks=chunks,
        scan_config=scan_cfg,
        loop_config=loop_cfg,
    )
    assert accept_res["status"] == "accepted_and_repaired"
    assert accept_res["accepted_iterations"] == 1
    assert accept_res["rolled_back_iterations"] == 0
    assert accept_res["delta_anse_energy"] < 0.0
    assert accept_res["guard_ok"] is True
    assert accept_res["worst_domain_drift"] <= 0.01
    assert accept_res["final_sha256"] != accept_res["initial_sha256"]
    assert accept_res["within_budget"] is True
    assert accept_res["parameter_fraction_modified"] < DEFAULT_MAX_EDITED_PARAM_RATIO
    assert accept_res["final_health"]["healthy"] is True

    # Running loop again on the now-healthy model returns cleanly with zero edits.
    noop_res = execute_self_rewriting_repair_loop(model, probe_chunks=chunks, scan_config=scan_cfg, loop_config=loop_cfg)
    assert noop_res["status"] == "already_healthy"
    assert noop_res["final_sha256"] == accept_res["final_sha256"]

    # Part B: Atomic bitwise rollback when candidate edit lowers specialist NLL yet pushes E0 > +1.0% (+2.5%!).
    pre_rollback_sha = compute_moe_state_sha256(model)
    pre_snap = snapshot_moe_state(model)

    def _guard_violating_proposal(m: Any, _health: dict[str, Any], _it: int) -> dict[str, Any]:
        # Make an aggressive edit to layer 0 expert 0 w1_A that slightly perturbs router/expert.
        w1_a = m.blocks[0].mlp.experts["0"].w1_A
        new_a = w1_a.detach().clone()
        new_a[0, :4] = new_a[0, :4] + 0.25
        return {
            "repairs": [
                {
                    "layer": "0",
                    "expert_id": 0,
                    "target_attr": "experts.0.w1_A",
                    "repair_kind": "candidate_probe_edit",
                    "new_tensor": new_a,
                    "modified_elements": 4,
                }
            ]
        }

    eval_counter = {"n": 0}

    def _eval_violating_guard(_m: Any) -> dict[str, Any]:
        eval_counter["n"] += 1
        if eval_counter["n"] == 1:
            return {"E0_general_fineweb": 2.000, "E1_python_systems_code": 2.400, "non_finite_outputs": 0}
        # Child lowers specialist E1 by -15%, yet worsens E0 by +2.5% (> +1.0% guard!).
        return {"E0_general_fineweb": 2.050, "E1_python_systems_code": 2.040, "non_finite_outputs": 0}

    rb_res = execute_self_rewriting_repair_loop(
        model,
        probe_eval_fn=_eval_violating_guard,
        scan_config=scan_cfg,
        loop_config=loop_cfg,
        candidate_proposal_fn=_guard_violating_proposal,
    )
    assert rb_res["status"] == "rolled_back"
    assert rb_res["accepted_iterations"] == 0
    assert rb_res["rolled_back_iterations"] == 1
    assert rb_res["guard_ok"] is True
    assert rb_res["iterations"][0]["guard_ok"] is False
    assert rb_res["initial_sha256"] == pre_rollback_sha
    assert rb_res["final_sha256"] == pre_rollback_sha
    assert rb_res["iterations"][0]["reject_reason"] == "domain_guard_exceeded"
    assert rb_res["iterations"][0]["rollback_bitwise_verified"] is True
    assert rb_res["iterations"][0]["worst_domain_drift"] == pytest.approx(0.025, rel=1e-6)
    assert rb_res["iterations"][0]["general_domain_drift"] == pytest.approx(0.025, rel=1e-6)
    for k, snap_t in pre_snap.items():
        live_t = snapshot_moe_state(model)[k]
        assert torch.equal(live_t, snap_t)

    # Part C: Rollback on non-negative Delta E (guard passes +0.2% <= +1.0%, yet E worsens) & non-finite child.
    eval_c = {"n": 0}

    def _eval_non_improving(_m: Any) -> dict[str, Any]:
        eval_c["n"] += 1
        if eval_c["n"] == 1:
            return {"E0_general_fineweb": 2.000, "E1_python_systems_code": 2.000}
        return {"E0_general_fineweb": 2.004, "E1_python_systems_code": 2.004}

    rb_energy = execute_self_rewriting_repair_loop(
        model,
        probe_eval_fn=_eval_non_improving,
        scan_config=scan_cfg,
        loop_config=loop_cfg,
        candidate_proposal_fn=_guard_violating_proposal,
    )
    assert rb_energy["status"] == "rolled_back"
    assert rb_energy["iterations"][0]["reject_reason"] == "non_negative_delta_energy"
    assert rb_energy["final_sha256"] == pre_rollback_sha

    eval_nf = {"n": 0}

    def _eval_non_finite(_m: Any) -> dict[str, Any]:
        eval_nf["n"] += 1
        if eval_nf["n"] == 1:
            return {"E0_general_fineweb": 2.0, "E1_python_systems_code": 2.0}
        return {"domain_nlls": {"E0_general_fineweb": float("nan"), "E1_python_systems_code": 1.5}, "non_finite_outputs": 2}

    rb_nf = execute_self_rewriting_repair_loop(
        model,
        probe_eval_fn=_eval_non_finite,
        scan_config=scan_cfg,
        loop_config=loop_cfg,
        candidate_proposal_fn=_guard_violating_proposal,
    )
    assert rb_nf["status"] == "rolled_back"
    assert rb_nf["iterations"][0]["reject_reason"] == "non_finite_child_output"
    assert rb_nf["final_sha256"] == pre_rollback_sha


# ---------------------------------------------------------------------------
# S5: <0.05% Parameter Budget Accounting, Validation & Security Gates
# ---------------------------------------------------------------------------


def test_s5_param_budget_accounting_validation_errors_and_security_gates() -> None:
    """Verify strict <0.05% parameter-edit budget accounting, defensive errors, and security audits."""
    # 1. Strict <0.05% parameter budget enforcement and budget-exhaustion skipping.
    dense_mlp = CrowdRuntimeMLP(CrowdV1RuntimeConfig(hidden_size=16, intermediate_size=32))
    moe = CrowdLoRAMoEMLP(dense_mlp, num_experts=3, top_k=2, rank=4, routing_mode="sigma_topk", seed=20260501)
    b_sp, a_sp, _, _ = _build_planted_spectral_factors(d_out=32, d_in=16, singular_values=[120.0, 1.5, 1.2, 1.0], seed=20260502)
    with torch.no_grad():
        moe.experts["0"].w1_B.copy_(b_sp)
        moe.experts["0"].w1_A.copy_(a_sp)
        moe.experts["1"].w2_A[0, 1] = float("nan")
        moe.experts["1"].w2_B.fill_(0.1)

    h_rep = scan_moe_adapter_health(moe)
    # Tight 10k denominator => 0.05% allows at most 5 elements: elementwise bit-flip fits (1 elem), QR-SVD winsorization (64 elems) skips!
    tight_prop = propose_surgical_block_repairs(moe, h_rep, total_model_params=10_000, max_edited_param_ratio=0.0005)
    assert tight_prop["within_budget"] is True
    assert tight_prop["total_modified_elements"] == 1
    assert tight_prop["parameter_fraction_modified"] <= 0.0005
    assert tight_prop["skipped_budget_count"] == 1
    assert tight_prop["repair_kinds_count"] == {"elementwise_bitflip_sanitize": 1}

    # Verify explicit quarantine_strategy="zero_b" and "elementwise" budget limits + router bias NaN sanitization.
    moe_b = CrowdLoRAMoEMLP(dense_mlp, num_experts=3, top_k=2, rank=4, routing_mode="sigma_topk", seed=20260503)
    moe_b.router = nn.Linear(16, 3, bias=True)
    with torch.no_grad():
        moe_b.router.weight.normal_(std=0.1)
        moe_b.router.bias.zero_()
        moe_b.router.bias[0] = float("nan")
        moe_b.router.weight[1, 0] = float("inf")
        moe_b.router.weight[2].fill_(20.0)
        moe_b.experts["0"].w1_A[0, 0] = float("nan")
        moe_b.experts["0"].w1_B.fill_(0.2)
    h_mb = scan_moe_adapter_health(moe_b, probe_activations={"0": torch.full((4, 16), float("nan")), "1": torch.ones((4, 16))})
    assert len(h_mb["corrupted_router_rows"]) == 2
    p_zb_fail = propose_surgical_block_repairs(
        moe_b, h_mb, config=RepairProposalConfig(quarantine_strategy="zero_b"), total_model_params=10_000
    )
    assert p_zb_fail["skipped_budget_count"] >= 2
    p_zb_ok = propose_surgical_block_repairs(
        moe_b, h_mb, config=RepairProposalConfig(quarantine_strategy="zero_b"), total_model_params=10_000_000
    )
    assert p_zb_ok["repair_kinds_count"]["finite_zero_perturbation_quarantine"] >= 1
    assert p_zb_ok["repair_kinds_count"]["elementwise_bitflip_sanitize"] >= 1

    # Tiny budget = 0 allowed elements skips zero_b, elementwise, and router repairs cleanly.
    zero_budget_prop = propose_surgical_block_repairs(moe, h_rep, total_model_params=100, max_edited_param_ratio=0.0005)
    assert zero_budget_prop["repair_count"] == 0
    assert zero_budget_prop["skipped_budget_count"] >= 1

    # Loop cleanly terminates with budget_exhausted when repairs exceed budget.
    res_exhausted = execute_self_rewriting_repair_loop(
        [moe],
        probe_eval_fn=lambda _m: {"E0_general": 2.0, "E1_spec": 2.0},
        total_model_params=100,
    )
    assert res_exhausted["termination_reason"] == "budget_exhausted"

    # 2. Config validation errors.
    with pytest.raises(TypeError, match="real number"):
        AdapterHealthScanConfig(spectral_spike_ratio_threshold=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="> 0.0"):
        AdapterHealthScanConfig(min_stable_rank=0.0)
    with pytest.raises(ValueError, match="finite"):
        AdapterHealthScanConfig(extreme_weight_threshold=float("inf"))
    with pytest.raises(ValueError, match=">= 0"):
        AdapterHealthScanConfig(score_from=-1)
    with pytest.raises(TypeError, match="real number"):
        RepairProposalConfig(router_target_off_logit=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        RepairProposalConfig(router_target_off_logit=float("nan"))
    with pytest.raises(ValueError, match="in \\[0.0, 1.0\\]"):
        RepairProposalConfig(max_edited_param_ratio=1.5)
    with pytest.raises(TypeError, match="bool"):
        RepairProposalConfig(enforce_param_budget="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="quarantine_strategy"):
        RepairProposalConfig(quarantine_strategy="invalid")
    with pytest.raises(TypeError, match="int"):
        SelfHealingLoopConfig(max_iterations=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 1"):
        SelfHealingLoopConfig(max_iterations=0)
    with pytest.raises(ValueError, match="non-empty string"):
        SelfHealingLoopConfig(general_domain_prefix=" ")

    # 3. API input & state normalization validation errors.
    with pytest.raises(TypeError, match="AdapterHealthScanConfig"):
        scan_moe_adapter_health(moe, config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="moe_layers"):
        scan_moe_adapter_health("not_a_layer")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        scan_moe_adapter_health([])
    with pytest.raises(ValueError, match="non-empty"):
        scan_moe_adapter_health({})
    with pytest.raises(ValueError, match="does not expose"):
        scan_moe_adapter_health(nn.Linear(4, 4))
    with pytest.raises(ValueError, match="MoE MLP"):
        scan_moe_adapter_health([123])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="MoE MLP"):
        scan_moe_adapter_health([nn.Linear(4, 4)])
    with pytest.raises(ValueError, match="MoE MLP"):
        scan_moe_adapter_health({"0": nn.Linear(4, 4)})
    with pytest.raises(ValueError, match="non-empty mapping"):
        scan_moe_adapter_health(moe, probe_activations={})
    with pytest.raises(ValueError, match="2D Tensor"):
        scan_moe_adapter_health(moe, probe_activations={"E0": torch.ones(4)})
    with pytest.raises(ValueError, match="domain -> Tensor"):
        scan_moe_adapter_health({"0": moe}, probe_activations={"0": "bad"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="2D Tensor"):
        scan_moe_adapter_health({"0": moe}, probe_activations={"0": {"E0": torch.ones(4)}})
    with pytest.raises(ValueError, match="non-empty sequence"):
        scan_moe_adapter_health(_build_subspace_moe_model(vocab_size=16, hidden_size=16, num_layers=1, intermediate_size=32, engram_buckets=0), probe_chunks=[])

    with pytest.raises(TypeError, match="RepairProposalConfig"):
        propose_surgical_block_repairs(moe, config="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="health_report"):
        propose_surgical_block_repairs(moe, health_report={"bad": 1})
    with pytest.raises(ValueError, match="proposal"):
        apply_surgical_block_repairs_inplace(moe, {"bad": 1})
    with pytest.raises(ValueError, match="Unknown layer key"):
        apply_surgical_block_repairs_inplace(moe, {"repairs": [{"layer": "99", "target_attr": "router.weight", "new_tensor": torch.zeros(16)}]})
    with pytest.raises(TypeError, match="torch.Tensor"):
        apply_surgical_block_repairs_inplace(moe, {"repairs": [{"layer": "0", "target_attr": "router.weight", "new_tensor": [0.0]}]})
    with pytest.raises(ValueError, match="no bias"):
        apply_surgical_block_repairs_inplace(moe, {"repairs": [{"layer": "0", "target_attr": "router.bias", "row_index": 0, "new_tensor": torch.tensor(0.0)}]})
    with pytest.raises(ValueError, match="Malformed"):
        apply_surgical_block_repairs_inplace(moe, {"repairs": [{"layer": "0", "target_attr": "experts.0", "new_tensor": torch.zeros(1)}]})
    with pytest.raises(ValueError, match="Unsupported target_attr"):
        apply_surgical_block_repairs_inplace(moe, {"repairs": [{"layer": "0", "target_attr": "unknown", "new_tensor": torch.zeros(16)}]})
    with pytest.raises(ValueError, match="snapshot"):
        restore_moe_state_inplace(moe, {})
    with pytest.raises(ValueError, match="snapshot keys"):
        restore_moe_state_inplace(moe, {"wrong": torch.zeros(1)})
    bad_snap = snapshot_moe_state(moe)
    bad_snap["layers.0.router.weight"] = torch.zeros((2, 2))
    with pytest.raises(ValueError, match="Snapshot mismatch"):
        restore_moe_state_inplace(moe, bad_snap)

    with pytest.raises(TypeError, match="SelfHealingLoopConfig"):
        execute_self_rewriting_repair_loop(moe, loop_config="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Either probe_chunks or probe_eval_fn"):
        execute_self_rewriting_repair_loop(moe)
    with pytest.raises(ValueError, match="CrowdV1WithEngram"):
        execute_self_rewriting_repair_loop(moe, probe_chunks=[{"expert_id": "E0", "ids": [0, 1, 2, 3]}])
    with pytest.raises(ValueError, match="non-empty mapping"):
        execute_self_rewriting_repair_loop(moe, probe_eval_fn=lambda _m: {})
    with pytest.raises(ValueError, match="at least one domain NLL"):
        execute_self_rewriting_repair_loop(moe, probe_eval_fn=lambda _m: {"nll_mean": 1.0})

    # 4. AGPL-3.0 header, ZeroStubAudit, PRSecurityScanner, lateral movement & pr_gatekeeper FORBIDDEN_ADDED.
    lateral_guard = _load_module_from_path("test_no_lateral_movement_dyn01", REPO_ROOT / "tests" / "test_no_lateral_movement.py")
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn01", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/self_healing_weight_repair.py", TARGET_MODULE_PATH),
        ("tests/test_self_healing_weight_repair_scenarios.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"

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
            {"filename": rel, "status": "added", "additions": len(src_text.splitlines()), "patch": "", "content": src_text}
        )

    telemetry_json = json.dumps(
        {
            "nll_mean": 1.85,
            "spectral_outlier_ratio": 2.41,
            "anse_energy": -0.37,
            "parameter_fraction_modified": 0.00018,
        }
    )
    assert gatekeeper.forbidden_scan(f"diff --git a/x.py b/x.py\n+++ b/x.py\n+PAYLOAD = {telemetry_json!r}") == []

    pr_audit = scanner.audit_pull_request(
        pr_number=1201,
        pr_title="feat(swarm): self-rewriting weight-block corruption scanner and closed-loop ANSE spectral repair",
        pr_body="Implements NEURAL-CTRL-DYN-01 self-rewriting weight-block scanner and closed-loop spectral repair.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): self-rewriting weight-block corruption scanner and closed-loop ANSE spectral repair"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
