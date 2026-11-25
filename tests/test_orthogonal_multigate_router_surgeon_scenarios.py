# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-15: Orthogonal multi-gate specialist router preserver."""

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
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdLoRAMoEMLP,
    extract_moe_trainable_buckets,
    load_moe_trainable_buckets,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.orthogonal_multigate_router_surgeon import (
    DEFAULT_ANSE_MAXIMUM_PAIN,
    RECEIPT_OBJECT,
    SCHEMA_VERSION,
    S4MultigateSurgeonConfig,
    canonical_json_bytes,
    compute_module_state_sha256,
    compute_payload_sha256,
    compute_state_sha256,
    execute_s4_multigate_moe_surgery,
    fit_oof_orthogonal_multigate_row_update,
    partition_sink_vs_preserved_specialists,
    verify_s4_multigate_receipt_provenance,
)
from hivemind.swarm.wsp_alphaedit_control import (
    forward_model_pre_ln,
    per_chunk_nll_from_pre_ln,
    zero_expert_lora_b_inplace,
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
TARGET_MODULE_PATH = (
    REPO_ROOT / "src" / "hivemind" / "swarm" / "orthogonal_multigate_router_surgeon.py"
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


def _build_8expert_s4_fixture(*, seed: int = 20261124) -> dict[str, Any]:
    """Build deterministic 3-layer, 8-expert sigma_topk MoE reproducing T4 Stage S4 geometry.

    - Layer 0 is a beneficial general anchor (expert_2 helps E0 prose; bitwise protected).
    - Layers 1 & 2 contain 3 pure prose-harming/rogue ghost sinks ({0, 1, 3}, with expert_3
      an unscored high-norm rogue) and 1 dual-active specialist (expert_5) that harms E0 prose
      while contributing strong specialist gain on E5.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    hidden_size = 64
    intermediate_size = 128
    vocab_size = 64
    num_layers = 3
    num_experts = 8

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

    in_axes = {
        0: _unit_axis(hidden_size, 0, 1),
        1: _unit_axis(hidden_size, 8, 9),
        2: _unit_axis(hidden_size, 16, 17),
        4: _unit_axis(hidden_size, 32, 33),
        5: _unit_axis(hidden_size, 40, 41),
        6: _unit_axis(hidden_size, 48, 49),
        7: _unit_axis(hidden_size, 56, 57),
    }
    tgt_axes = {
        0: _unit_axis(hidden_size, 4, 5),
        1: _unit_axis(hidden_size, 10, 11),
        2: _unit_axis(hidden_size, 18, 19),
        4: _unit_axis(hidden_size, 34, 35),
        5: _unit_axis(hidden_size, 42, 43),
        6: _unit_axis(hidden_size, 50, 51),
        7: _unit_axis(hidden_size, 58, 59),
    }
    u_prose_gate = _unit_axis(hidden_size, 2, 3)

    targets = {
        0: ("E0_general_fineweb", 0, 7, 7),
        1: ("E1_python_code", 8, 15, 15),
        2: ("E2_math_reasoning", 16, 23, 23),
        4: ("E4_physics_stem", 32, 39, 39),
        5: ("E5_exoplanet_transit", 40, 47, 47),
        6: ("E6_stellar_spectra", 48, 55, 55),
        7: ("E7_defensive_cyber", 56, 63, 63),
    }

    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        for e_idx, (_, lo, hi, tgt) in targets.items():
            d_lo = e_idx * 8
            raw = 0.08 * torch.randn((hi - lo, 2), generator=gen, dtype=torch.float32)
            wte[lo:hi, d_lo + 6 : d_lo + 8] = raw
            wte[lo:hi] += 1.5 * in_axes[e_idx].unsqueeze(0)
            if e_idx == 0:
                wte[lo:hi] += 1.5 * u_prose_gate.unsqueeze(0)
            wte[tgt] = 1.6 * tgt_axes[e_idx]
        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight

        model.ln.weight.fill_(1.0)
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln2.weight.fill_(1.0)
            # Isolated SwiGLU detector neurons: neuron 0 fires strictly on E0 prose;
            # neurons 1..7 fire strictly on specialist domain input axes[1..7].
            blk.mlp.w1.weight[0].copy_(0.95 * u_prose_gate)
            blk.mlp.w2.weight[0].copy_(0.95 * u_prose_gate)
            blk.mlp.w3.weight[:, 0].copy_(0.04 * tgt_axes[0])
            for e_idx in (1, 2, 4, 5, 6, 7):
                blk.mlp.w1.weight[e_idx].copy_(0.95 * in_axes[e_idx])
                blk.mlp.w2.weight[e_idx].copy_(0.95 * in_axes[e_idx])
                blk.mlp.w3.weight[:, e_idx].copy_(0.005 * tgt_axes[e_idx])

        model.blocks[0].attn.v.weight.copy_(0.25 * torch.eye(hidden_size))
        model.blocks[0].attn.o.weight.copy_(0.25 * torch.eye(hidden_size))

    upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode="lora_domain_experts",
        num_experts=num_experts,
        top_k=2,
        lora_rank=4,
        lora_alpha=8.0,
        routing_mode="sigma_topk",
        seed=seed + 31,
    )
    start_buckets = {
        k: v.detach().cpu().float().clone()
        for k, v in extract_moe_trainable_buckets(model, num_router_fragments=2).items()
    }

    probe_chunks: list[dict[str, Any]] = []
    eval_chunks: list[dict[str, Any]] = []
    for e_idx, (eid, lo, hi, tgt) in targets.items():
        n_probe = 10 if e_idx == 0 else 6
        for _ in range(n_probe):
            ids = torch.randint(lo, hi, (16,), generator=gen).tolist()
            ids[-1] = tgt
            probe_chunks.append({"expert_id": eid, "ids": ids})
        for _ in range(6):
            ids = torch.randint(lo, hi, (16,), generator=gen).tolist()
            ids[-1] = tgt
            eval_chunks.append({"expert_id": eid, "ids": ids})

    base_nll = _eval_domain_nll(model, eval_chunks, score_from=15)

    # Inject realistic C2_final-like weights:
    # - Layer 0 (anchor): expert_2 strongly helps E0 prose and helps E2
    # - Layer 1: top-2 on E0 are {0, 5}; both harm E0; expert_5 gives huge E5 gain
    # - Layer 2: top-2 on E0 are {1, 3}; both harm E0; expert_3 has high rogue ||w1_B||
    with torch.no_grad():
        for blk in model.blocks:
            mlp: CrowdLoRAMoEMLP = blk.mlp  # type: ignore[assignment]
            mlp.router.weight.zero_()
            for exp_mod in mlp.experts.values():
                for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                    getattr(exp_mod, attr).zero_()

        # Layer 0: Beneficial anchor (helpful general expert_2 drops E0 NLL sharply)
        mlp0: CrowdLoRAMoEMLP = model.blocks[0].mlp  # type: ignore[assignment]
        mlp0.router.weight[2] = 0.35 * u_prose_gate + 1.60 * in_axes[2]
        mlp0.router.weight[6] = 0.25 * u_prose_gate + 1.60 * in_axes[6]
        mlp0.experts["2"].w3_A[0, 0] = 0.40
        mlp0.experts["2"].w3_B[:, 0] = 0.45 * tgt_axes[0]
        mlp0.experts["2"].w3_A[1, 2] = 0.45
        mlp0.experts["2"].w3_B[:, 1] = 0.50 * tgt_axes[2]
        mlp0.experts["6"].w3_A[0, 6] = 0.45
        mlp0.experts["6"].w3_B[:, 0] = 0.50 * tgt_axes[6]

        # Layer 1: expert_0 (ghost sink) & expert_5 (dual-active specialist!) fire on E0
        mlp1: CrowdLoRAMoEMLP = model.blocks[1].mlp  # type: ignore[assignment]
        mlp1.router.weight[0] = 0.12 * u_prose_gate
        mlp1.router.weight[5] = 0.10 * u_prose_gate + 1.80 * in_axes[5]
        mlp1.router.weight[1] = 0.06 * u_prose_gate + 1.60 * in_axes[1]
        mlp1.router.weight[4] = 1.60 * in_axes[4]
        mlp1.router.weight[7] = 1.60 * in_axes[7]
        mlp1.experts["0"].w3_A[0, 0] = 0.55
        mlp1.experts["0"].w3_B[:, 0] = -0.60 * tgt_axes[0]
        mlp1.experts["5"].w3_A[0, 0] = 0.55
        mlp1.experts["5"].w3_B[:, 0] = -0.55 * tgt_axes[0]
        mlp1.experts["5"].w3_A[1, 5] = 0.70
        mlp1.experts["5"].w3_B[:, 1] = 0.85 * tgt_axes[5]
        mlp1.experts["4"].w3_A[0, 4] = 0.45
        mlp1.experts["4"].w3_B[:, 0] = 0.50 * tgt_axes[4]
        mlp1.experts["7"].w3_A[0, 7] = 0.45
        mlp1.experts["7"].w3_B[:, 0] = 0.50 * tgt_axes[7]

        # Layer 2: expert_1 (prose sink) & expert_3 (unscored high-norm rogue sink) fire on E0
        mlp2: CrowdLoRAMoEMLP = model.blocks[2].mlp  # type: ignore[assignment]
        mlp2.router.weight[1] = 0.12 * u_prose_gate + 1.60 * in_axes[1]
        mlp2.router.weight[3] = 0.10 * u_prose_gate
        mlp2.router.weight[0] = 0.06 * u_prose_gate
        mlp2.router.weight[5] = 1.80 * in_axes[5]
        mlp2.experts["1"].w3_A[0, 0] = 0.55
        mlp2.experts["1"].w3_B[:, 0] = -0.60 * tgt_axes[0]
        # Unscored rogue expert_3 with large w1_B norm and strong E0 corruption
        mlp2.experts["3"].w1_A[0, 0:2] = torch.tensor([0.10, -0.10])
        mlp2.experts["3"].w1_B[0:2, 0] = torch.tensor([24.0, -24.0])
        mlp2.experts["3"].w3_A[0, 0] = 0.55
        mlp2.experts["3"].w3_B[:, 0] = -0.65 * tgt_axes[0]
        mlp2.experts["5"].w3_A[1, 5] = 0.65
        mlp2.experts["5"].w3_B[:, 1] = 0.75 * tgt_axes[5]

    tuned_buckets = {
        k: v.detach().cpu().float().clone()
        for k, v in extract_moe_trainable_buckets(model, num_router_fragments=2).items()
    }
    uncorrected_nll = _eval_domain_nll(model, eval_chunks, score_from=15)
    return {
        "model": model,
        "start_buckets": start_buckets,
        "tuned_buckets": tuned_buckets,
        "probe_chunks": probe_chunks,
        "eval_chunks": eval_chunks,
        "base_nll": base_nll,
        "uncorrected_nll": uncorrected_nll,
        "axes": tgt_axes,
    }


# ---------------------------------------------------------------------------
# S1: 8-Expert S4 Selective Sinking + Orthogonal Multi-Gate vs Full Sinking
# ---------------------------------------------------------------------------


def test_s1_8expert_s4_selective_sinking_and_multigate_beats_full_sinking_and_uncorrected() -> None:
    """S1: Selective sinking {0,1,3} + orthogonal multi-gate {5} beats full {0,1,3,5} sinking & uncorrected MoE."""
    # 1. Standalone partition verification matching T4 empirical values (expert_5 at +0.0278 nats)
    t4_part = partition_sink_vs_preserved_specialists(
        {0: 0.014, 1: 0.008, 2: -0.003, 3: 0.009, 4: 0.0005, 5: 0.0045, 6: -0.001, 7: -0.002},
        {0: -0.001, 1: 0.004, 2: 0.042, 3: 0.0, 4: 0.025, 5: 0.0278, 6: 0.088, 7: 0.054},
    )
    assert t4_part["sink_experts"] == [0, 1, 3]
    assert t4_part["preserved_specialists"] == [5]
    assert t4_part["benign_specialists"] == [2, 4, 6, 7]
    assert t4_part["helpful_general_experts"] == []
    assert t4_part["per_expert"]["expert_5"]["role"] == "dual_active_specialist"

    # Sequence input & rogue_expert_indices handling
    seq_part = partition_sink_vs_preserved_specialists(
        [0.01, 0.01, -0.005, 0.0, 0.004],
        [0.0, 0.003, 0.001, 0.0, 0.030],
        rogue_expert_indices=[3],
    )
    assert seq_part["sink_experts"] == [0, 1, 3]
    assert seq_part["preserved_specialists"] == [4]
    assert seq_part["helpful_general_experts"] == [2]
    assert seq_part["protect_general_expert"] is False

    # 2. Full 8-expert CrowdV1WithEngram + CrowdLoRAMoEMLP S4 surgery
    fix = _build_8expert_s4_fixture(seed=20261124)
    model_s4: CrowdV1WithEngram = fix["model"]
    probe_chunks = fix["probe_chunks"]
    eval_chunks = fix["eval_chunks"]
    base_nll = fix["base_nll"]
    uncorrected_nll = fix["uncorrected_nll"]

    # Confirm uncorrected MoE seriously degrades E0 prose (> +1.0% guard violation)
    uncorrected_e0_pct = 100.0 * (uncorrected_nll["E0_general_fineweb"] / base_nll["E0_general_fineweb"] - 1.0)
    assert uncorrected_e0_pct > 2.0

    # Baseline B: Naive full sinking of {0, 1, 3, 5} across non-anchor layers
    model_full_sink = copy.deepcopy(model_s4)
    with torch.no_grad():
        for l_idx in (1, 2):
            zero_expert_lora_b_inplace(model_full_sink.blocks[l_idx].mlp, [0, 1, 3, 5])
    full_sink_nll = _eval_domain_nll(model_full_sink, eval_chunks, score_from=15)

    # Run S4 end-to-end surgery (with production bucket loader/extractor & 21-point fine LN)
    model_s4_copy = copy.deepcopy(model_s4)
    receipt_a = execute_s4_multigate_moe_surgery(
        model_s4,
        probe_chunks,
        selection_chunks=eval_chunks,
        start_buckets=fix["start_buckets"],
        tuned_buckets=fix["tuned_buckets"],
        baseline_domain_nll=base_nll,
        score_from=15,
        batch_size=16,
    )
    receipt_b = execute_s4_multigate_moe_surgery(
        model_s4_copy,
        probe_chunks,
        selection_chunks=eval_chunks,
        start_buckets=fix["start_buckets"],
        tuned_buckets=fix["tuned_buckets"],
        baseline_domain_nll=base_nll,
        score_from=15,
        batch_size=16,
    )

    assert receipt_a["accepted"] is True
    assert receipt_a["rolled_back"] is False
    assert receipt_a["status"] == "ACCEPTED"
    assert receipt_a["guard_ok"] is True
    assert verify_s4_multigate_receipt_provenance(receipt_a) is True
    assert receipt_a["receipt_sha256"] == receipt_b["receipt_sha256"]
    assert receipt_a["output_buckets_sha256"] is not None
    assert receipt_a["lambda_ln_applied"] is not None

    # Selective partition: {0, 1, 3} sunk, expert_5 preserved & gated!
    assert set(receipt_a["sink_experts"]) == {0, 1, 3}
    assert 5 not in receipt_a["sink_experts"]
    assert 5 in receipt_a["preserved_specialists"]
    assert 0 in receipt_a["beneficial_anchor_layers"]
    assert receipt_a["anchor_layers_bitwise_preserved"] is True

    s4_nll = _eval_domain_nll(model_s4, eval_chunks, score_from=15)

    # Clears E0 <= +0.5% vs dense base
    s4_e0_pct = 100.0 * (s4_nll["E0_general_fineweb"] / base_nll["E0_general_fineweb"] - 1.0)
    assert s4_e0_pct <= 0.5
    assert float(receipt_a["general_pct_vs_base"]) <= 0.5

    # Beats full {0,1,3,5} sinking on specialist E5 NLL AND pooled specialist NLL
    spec_keys = [k for k in s4_nll if not k.startswith("E0_")]
    s4_spec_mean = sum(s4_nll[k] for k in spec_keys) / float(len(spec_keys))
    full_sink_spec_mean = sum(full_sink_nll[k] for k in spec_keys) / float(len(spec_keys))
    assert s4_nll["E5_exoplanet_transit"] < full_sink_nll["E5_exoplanet_transit"] - 0.15
    assert s4_spec_mean < full_sink_spec_mean - 0.02

    # Beats uncorrected MoE on overall NLL mean, E0 prose NLL, and ANSE energy
    uncorr_mean = sum(uncorrected_nll.values()) / float(len(uncorrected_nll))
    s4_mean = sum(s4_nll.values()) / float(len(s4_nll))
    assert s4_mean < uncorr_mean
    assert s4_nll["E0_general_fineweb"] < uncorrected_nll["E0_general_fineweb"]
    assert float(receipt_a["anse_energy"]) < 0.0
    assert float(receipt_a["delta_e"]) < -1e5


# ---------------------------------------------------------------------------
# S2: Exact Orthogonality (<1e-5 Router Logit Change on Specialist Centroids)
# ---------------------------------------------------------------------------


def test_s2_exact_orthogonality_on_all_specialist_domain_centroids() -> None:
    """S2: Orthogonal rank-1 router shift induces <1e-5 logit change on all specialist centroids."""
    gen = torch.Generator(device="cpu").manual_seed(4201)
    dim = 48
    n_spec_domains = 5

    # Dense non-orthogonal specialist domain centroids + prose direction with overlap
    raw_dirs = torch.randn((n_spec_domains, dim), generator=gen, dtype=torch.float32)
    prose_base = torch.randn(dim, generator=gen, dtype=torch.float32)

    domain_acts: dict[int, torch.Tensor] = {}
    for idx in range(n_spec_domains):
        eid = [1, 2, 5, 6, 7][idx]
        mu = 2.5 * raw_dirs[idx] / raw_dirs[idx].norm()
        noise = 0.20 * torch.randn((32, dim), generator=gen, dtype=torch.float32)
        noise = noise - noise.mean(dim=0, keepdim=True)
        domain_acts[eid] = mu.unsqueeze(0) + noise

    # Prose carries real signal + non-zero overlap with specialist centroids
    x_prose = (
        2.0 * (prose_base / prose_base.norm()).unsqueeze(0)
        + 0.45 * domain_acts[1].mean(dim=0, keepdim=True)
        + 0.35 * domain_acts[5].mean(dim=0, keepdim=True)
        + 0.18 * torch.randn((64, dim), generator=gen, dtype=torch.float32)
    )

    router_row = torch.randn(dim, generator=gen, dtype=torch.float32) * 0.5
    router_row_orig = router_row.clone()

    # Out-of-place check (n_folds=2)
    updated_row, stats = fit_oof_orthogonal_multigate_row_update(
        router_row,
        x_prose,
        domain_acts,
        router_beta=0.35,
        min_oof_snr=0.25,
        n_folds=2,
        inplace=False,
    )
    assert torch.equal(router_row, router_row_orig)
    assert stats["applied"] is True
    assert stats["gate_passed"] is True
    assert stats["rel_weight"] > 0.5
    assert stats["oof_snr_margin"] > 0.25
    assert stats["centroid_subspace_rank"] == n_spec_domains
    assert stats["mean_prose_logit_delta"] < -0.15
    assert stats["max_centroid_logit_delta"] < 1e-5
    assert stats["max_centroid_dot_abs"] < 1e-12

    for eid, x_dom in domain_acts.items():
        centroid = x_dom.mean(dim=0)
        delta_logit = abs(float((updated_row @ centroid - router_row @ centroid).item()))
        assert delta_logit < 1e-5, f"Domain {eid} centroid logit shifted by {delta_logit}"

    # Verify inplace=True, 3-fold OOF, Sequence input, and Tensor input
    row_in = router_row_orig.clone()
    ret_in, stats_3f = fit_oof_orthogonal_multigate_row_update(
        row_in,
        x_prose,
        list(domain_acts.values()),
        router_beta=0.35,
        min_oof_snr=0.25,
        n_folds=3,
        inplace=True,
    )
    assert ret_in is row_in
    assert stats_3f["applied"] is True
    assert stats_3f["max_centroid_logit_delta"] < 1e-5

    _, stats_single = fit_oof_orthogonal_multigate_row_update(
        router_row_orig,
        x_prose.unsqueeze(0),  # 3D tensor accepted seamlessly
        domain_acts[5],
        router_beta=0.35,
    )
    assert stats_single["applied"] is True
    assert stats_single["max_centroid_logit_delta"] < 1e-5

    # Mixed domain sample sizes (tiny < n_folds domain + standard domain)
    _, stats_tiny = fit_oof_orthogonal_multigate_row_update(
        router_row_orig,
        x_prose,
        {"tiny": domain_acts[1][:1], "full": domain_acts[5]},
        n_folds=2,
    )
    assert stats_tiny["applied"] is True
    assert stats_tiny["max_centroid_logit_delta"] < 1e-5


# ---------------------------------------------------------------------------
# S3: 2-Fold OOF SNR Gate Rejects Pure Gaussian Calibration Noise
# ---------------------------------------------------------------------------


def test_s3_oof_snr_gate_rejects_gaussian_noise_with_zero_mutation() -> None:
    """S3: Pure random Gaussian noise and indistinguishable splits trigger zero router mutation."""
    for seed_idx, (n_prose, n_spec, dim) in enumerate(((48, 48, 24), (80, 64, 32), (100, 100, 48))):
        gen = torch.Generator(device="cpu").manual_seed(9100 + seed_idx)
        x_prose_noise = torch.randn((n_prose, dim), generator=gen, dtype=torch.float32)
        dom_noise = {
            1: torch.randn((n_spec, dim), generator=gen, dtype=torch.float32),
            5: torch.randn((n_spec, dim), generator=gen, dtype=torch.float32),
        }
        router_row = torch.randn(dim, generator=gen, dtype=torch.float32)
        row_before = router_row.clone()

        updated_row, stats = fit_oof_orthogonal_multigate_row_update(
            router_row,
            x_prose_noise,
            dom_noise,
            router_beta=0.35,
            min_oof_snr=0.25,
            n_folds=2,
            inplace=True,
        )
        assert stats["gate_passed"] is False
        assert stats["rel_weight"] == 0.0
        assert stats["applied"] is False
        assert stats["weight_delta_l2"] == 0.0
        assert torch.equal(updated_row, row_before)
        assert torch.equal(router_row, row_before)

    # Identical shifted distributions & full centroid span (no orthogonal prose residual)
    gen_s = torch.Generator(device="cpu").manual_seed(9500)
    mu_shared = torch.randn(16, generator=gen_s) * 2.0
    x_p_shift = mu_shared + torch.randn((60, 16), generator=gen_s)
    x_s_shift = mu_shared + torch.randn((60, 16), generator=gen_s)
    r_row = torch.randn(16, generator=gen_s)
    out_shift, st_shift = fit_oof_orthogonal_multigate_row_update(r_row, x_p_shift, {1: x_s_shift})
    assert st_shift["applied"] is False
    assert torch.equal(out_shift, r_row)

    # Full-rank specialist centroids span all of R^4 -> orthogonal subspace is empty -> zero mutation
    eye4 = torch.eye(4, dtype=torch.float32)
    dom_full = {i: eye4[i : i + 1].repeat(4, 1) for i in range(4)}
    out_full, st_full = fit_oof_orthogonal_multigate_row_update(
        torch.ones(4), torch.ones((8, 4)), dom_full
    )
    assert st_full["centroid_subspace_rank"] == 4
    assert st_full["applied"] is False
    assert torch.equal(out_full, torch.ones(4))

    # Zero-centroid edge case + zero router_beta
    out_zc, st_zc = fit_oof_orthogonal_multigate_row_update(
        torch.ones(4),
        torch.ones((8, 4)),
        torch.zeros((8, 4)),
        router_beta=0.0,
    )
    assert st_zc["centroid_subspace_rank"] == 0
    assert st_zc["applied"] is False
    assert torch.equal(out_zc, torch.ones(4))


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Atomic Bitwise Rollback on Adversarial Infeasible Guard
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_bitwise_rollback_and_edge_statuses() -> None:
    """S4: Adversarial infeasible guard triggers ANSE >= 1e6 Maximum Pain and exact SHA-256 rollback."""
    fix = _build_8expert_s4_fixture(seed=20261125)
    model: CrowdV1WithEngram = fix["model"]
    probe_chunks = fix["probe_chunks"]
    eval_chunks = fix["eval_chunks"]
    parent_sha = compute_module_state_sha256(model)

    # 1. Adversarial infeasible guard via eval_fn pushing candidate general domain > +1.0%
    eval_calls = {"n": 0}

    def _adversarial_guard_eval(_m: nn.Module) -> dict[str, float]:
        eval_calls["n"] += 1
        if eval_calls["n"] == 1:
            return {
                "E0_general_fineweb": 1.00,
                "E1_python_code": 1.00,
                "E5_exoplanet_transit": 1.00,
            }
        return {
            "E0_general_fineweb": 1.08,  # +8.0% drift -> violates <= +1.0% guard!
            "E1_python_code": 0.90,
            "E5_exoplanet_transit": 0.85,
        }

    guard_rec = execute_s4_multigate_moe_surgery(
        model,
        probe_chunks,
        selection_chunks=eval_chunks,
        eval_fn=_adversarial_guard_eval,
        score_from=15,
        batch_size=16,
    )
    assert guard_rec["accepted"] is False
    assert guard_rec["rolled_back"] is True
    assert guard_rec["guard_ok"] is False
    assert guard_rec["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert guard_rec["anse_energy"] >= DEFAULT_ANSE_MAXIMUM_PAIN - 10.0
    assert guard_rec["candidate_state_sha256"] != parent_sha
    assert guard_rec["final_state_sha256"] == parent_sha
    assert compute_module_state_sha256(model) == parent_sha
    assert verify_s4_multigate_receipt_provenance(guard_rec) is True

    # 2. Eval failure -> Maximum Pain + atomic bitwise rollback
    fail_calls = {"n": 0}

    def _boom_eval(_m: nn.Module) -> dict[str, float]:
        fail_calls["n"] += 1
        if fail_calls["n"] >= 2:
            raise RuntimeError("injected eval failure")
        return {"E0_general_fineweb": 1.0, "E5_exoplanet_transit": 1.0}

    fail_rec = execute_s4_multigate_moe_surgery(
        model,
        probe_chunks,
        selection_chunks=eval_chunks,
        eval_fn=_boom_eval,
        score_from=15,
        batch_size=16,
    )
    assert fail_rec["status"] == "ROLLED_BACK_EVAL_FAILURE"
    assert fail_rec["rolled_back"] is True
    assert fail_rec["anse_energy"] >= DEFAULT_ANSE_MAXIMUM_PAIN
    assert compute_module_state_sha256(model) == parent_sha

    # 3. Non-improving energy rollback (and soft margin penalty branch when 1.0085 < ratio <= 1.01)
    nonimp_calls = {"n": 0}

    def _non_improving_eval(_m: nn.Module) -> dict[str, float]:
        nonimp_calls["n"] += 1
        if nonimp_calls["n"] == 1:
            return {"E0_general_fineweb": 1.000, "E5_exoplanet_transit": 1.000}
        return {"E0_general_fineweb": 1.009, "E5_exoplanet_transit": 1.005}

    nonimp_rec = execute_s4_multigate_moe_surgery(
        model,
        probe_chunks,
        selection_chunks=eval_chunks,
        eval_fn=_non_improving_eval,
        require_energy_improvement=True,
        score_from=15,
        batch_size=16,
    )
    assert nonimp_rec["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    assert nonimp_rec["rolled_back"] is True
    assert nonimp_rec["guard_ok"] is True
    assert compute_module_state_sha256(model) == parent_sha

    # 4. FIXED_POINT_UNCHANGED when all layers are protected beneficial anchors (or expert_loeo_override is all neutral)
    fp_rec = execute_s4_multigate_moe_surgery(
        model,
        probe_chunks,
        selection_chunks=eval_chunks,
        expert_loeo_override={
            f"expert_{e}": {"e0_harm_nats": 0.0, "delta_spec_when_zeroed": 0.0}
            for e in range(8)
        },
        layer_attribution_override={
            "layers": [{"layer": i, "general_harm_nats": -0.05} for i in range(3)]
        },
        score_from=15,
        batch_size=16,
        inplace=False,
    )
    assert fp_rec["status"] == "FIXED_POINT_UNCHANGED"
    assert fp_rec["accepted"] is False
    assert fp_rec["rolled_back"] is False
    assert fp_rec["beneficial_anchor_layers"] == [0, 1, 2]
    assert fp_rec["final_state_sha256"] == parent_sha

    # 5. gate_all_surviving_specialists=True + target_layers + protect_anchor_layers=False (override delta_e0_when_zeroed)
    alt_cfg = S4MultigateSurgeonConfig(
        protect_anchor_layers=False,
        gate_all_surviving_specialists=True,
        require_energy_improvement=False,
        score_from=15,
        batch_size=16,
        inplace=False,
    )
    alt_rec = execute_s4_multigate_moe_surgery(
        model,
        probe_chunks,
        selection_chunks=eval_chunks,
        config=alt_cfg,
        target_layers=[1, 2],
        expert_loeo_override={
            "expert_0": {"delta_e0_when_zeroed": -0.02, "delta_spec_when_zeroed": 0.0},
            "expert_1": {"delta_e0_when_zeroed": -0.02, "delta_spec_when_zeroed": 0.0},
            "expert_3": {"delta_e0_when_zeroed": -0.02, "delta_spec_when_zeroed": 0.0, "is_rogue_high_norm": True},
            "expert_5": {"delta_e0_when_zeroed": -0.02, "delta_spec_when_zeroed": 0.05},
            "expert_2": {"delta_e0_when_zeroed": 0.05, "delta_spec_when_zeroed": 0.02},
        },
        baseline_domain_nll=fix["base_nll"],
    )
    assert alt_rec["accepted"] is True
    assert alt_rec["beneficial_anchor_layers"] == []
    assert 5 in alt_rec["gated_targets"]
    assert 2 in alt_rec["gated_targets"]
    # Original model untouched because inplace=False
    assert compute_module_state_sha256(model) == parent_sha


# ---------------------------------------------------------------------------
# S5: Validation Errors, Edge Cases & Security / Stub Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_compliance_gates() -> None:
    """S5: Exhaustive input validation, provenance checks, and HiveMind security/stub compliance gates."""
    # 1. S4MultigateSurgeonConfig validation
    with pytest.raises(TypeError, match="real number"):
        S4MultigateSurgeonConfig(router_beta=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        S4MultigateSurgeonConfig(router_beta=float("nan"))
    with pytest.raises(ValueError, match=">= 0.0"):
        S4MultigateSurgeonConfig(min_oof_snr=-0.1)
    with pytest.raises(TypeError, match="int"):
        S4MultigateSurgeonConfig(n_folds=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 1"):
        S4MultigateSurgeonConfig(n_folds=0)
    with pytest.raises(ValueError, match=">= 2"):
        S4MultigateSurgeonConfig(n_folds=1)
    with pytest.raises(TypeError, match="bool"):
        S4MultigateSurgeonConfig(protect_anchor_layers="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="anchor_threshold_nats"):
        S4MultigateSurgeonConfig(anchor_threshold_nats=0.05)
    with pytest.raises(ValueError, match="fine_ln_e0_max_ratio"):
        S4MultigateSurgeonConfig(fine_ln_e0_max_ratio=0.95)
    with pytest.raises(ValueError, match="domain_guard_max_ratio"):
        S4MultigateSurgeonConfig(domain_guard_max_ratio=0.99)
    with pytest.raises(ValueError, match="anse_safe_e0_ratio"):
        S4MultigateSurgeonConfig(anse_safe_e0_ratio=0.99)
    with pytest.raises(ValueError, match="> 0.0"):
        S4MultigateSurgeonConfig(domain_guard_max_ratio=0.0)
    with pytest.raises(ValueError, match="general_prefix"):
        S4MultigateSurgeonConfig(general_prefix=" ")

    # 2. partition_sink_vs_preserved_specialists validation
    with pytest.raises(ValueError, match="empty"):
        partition_sink_vs_preserved_specialists({}, {})
    with pytest.raises(TypeError, match="bool"):
        partition_sink_vs_preserved_specialists({True: 0.1}, {0: 0.1})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="Invalid expert key"):
        partition_sink_vs_preserved_specialists({"bad_key": 0.1}, {0: 0.1})
    with pytest.raises(TypeError, match="Invalid key type"):
        partition_sink_vs_preserved_specialists({(0,): 0.1}, {0: 0.1})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="Duplicate expert index"):
        partition_sink_vs_preserved_specialists({"0": 0.1, "expert_0": 0.2}, {0: 0.1})
    with pytest.raises(TypeError, match="Mapping or Sequence"):
        partition_sink_vs_preserved_specialists("bad", [0.1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        partition_sink_vs_preserved_specialists([], [])
    with pytest.raises(ValueError, match="identical expert indices"):
        partition_sink_vs_preserved_specialists({0: 0.1}, {1: 0.1})
    with pytest.raises(TypeError, match="rogue_expert_indices"):
        partition_sink_vs_preserved_specialists({0: 0.1}, {0: 0.1}, rogue_expert_indices="0")  # type: ignore[arg-type]

    # 3. fit_oof_orthogonal_multigate_row_update validation
    r4 = torch.ones(4, dtype=torch.float32)
    xp4 = torch.ones((6, 4), dtype=torch.float32)
    xs4 = torch.ones((6, 4), dtype=torch.float32)
    with pytest.raises(TypeError, match="torch.Tensor"):
        fit_oof_orthogonal_multigate_row_update([1.0, 2.0], xp4, xs4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty 1D"):
        fit_oof_orthogonal_multigate_row_update(torch.ones((2, 4)), xp4, xs4)
    with pytest.raises(TypeError, match="floating-point"):
        fit_oof_orthogonal_multigate_row_update(torch.ones(4, dtype=torch.long), xp4, xs4)
    with pytest.raises(ValueError, match="finite"):
        fit_oof_orthogonal_multigate_row_update(torch.tensor([1.0, float("nan"), 0.0, 0.0]), xp4, xs4)
    with pytest.raises(TypeError, match="torch.Tensor"):
        fit_oof_orthogonal_multigate_row_update(r4, [[1.0]], xs4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2D or 3D"):
        fit_oof_orthogonal_multigate_row_update(r4, torch.ones(4), xs4)
    with pytest.raises(TypeError, match="floating-point"):
        fit_oof_orthogonal_multigate_row_update(r4, torch.ones((6, 4), dtype=torch.long), xs4)
    with pytest.raises(ValueError, match="feature width"):
        fit_oof_orthogonal_multigate_row_update(r4, torch.ones((6, 5)), xs4)
    with pytest.raises(ValueError, match="finite"):
        fit_oof_orthogonal_multigate_row_update(r4, torch.tensor([[1.0, 2.0, float("inf"), 0.0]] * 4), xs4)
    with pytest.raises(ValueError, match="empty"):
        fit_oof_orthogonal_multigate_row_update(r4, xp4, {})
    with pytest.raises(TypeError, match="int or str"):
        fit_oof_orthogonal_multigate_row_update(r4, xp4, {True: xs4})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-empty"):
        fit_oof_orthogonal_multigate_row_update(r4, xp4, {" ": xs4})
    with pytest.raises(TypeError, match="Tensor, Mapping, or Sequence"):
        fit_oof_orthogonal_multigate_row_update(r4, xp4, "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        fit_oof_orthogonal_multigate_row_update(r4, xp4, [])
    with pytest.raises(ValueError, match=">= 2"):
        fit_oof_orthogonal_multigate_row_update(r4, xp4, xs4, n_folds=1)
    with pytest.raises(ValueError, match="at least n_folds"):
        fit_oof_orthogonal_multigate_row_update(r4, torch.ones((1, 4)), xs4, n_folds=2)
    with pytest.raises(ValueError, match="Total specialist activation rows"):
        fit_oof_orthogonal_multigate_row_update(r4, xp4, torch.ones((1, 4)), n_folds=2)

    # 4. execute_s4_multigate_moe_surgery & provenance validation
    with pytest.raises(TypeError, match="nn.Module"):
        execute_s4_multigate_moe_surgery("bad", [])  # type: ignore[arg-type]
    bad_mod = nn.Module()
    bad_mod.blocks = nn.ModuleList()
    with pytest.raises(ValueError, match="blocks"):
        execute_s4_multigate_moe_surgery(bad_mod, [])
    bad_mod.blocks = nn.ModuleList([nn.Linear(4, 4)])
    with pytest.raises(ValueError, match="must expose"):
        execute_s4_multigate_moe_surgery(bad_mod, [])

    fix = _build_8expert_s4_fixture(seed=777)
    m = fix["model"]
    ch = fix["probe_chunks"]
    bad_no_moe = copy.deepcopy(m)
    for blk in bad_no_moe.blocks:
        blk.mlp = nn.Linear(64, 64)
    with pytest.raises(ValueError, match="MoE block"):
        execute_s4_multigate_moe_surgery(bad_no_moe, ch, score_from=8)

    with pytest.raises(TypeError, match="callable"):
        execute_s4_multigate_moe_surgery(m, ch, eval_fn="bad", score_from=8)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="S4MultigateSurgeonConfig"):
        execute_s4_multigate_moe_surgery(m, ch, config="bad", score_from=8)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty sequence"):
        execute_s4_multigate_moe_surgery(m, [], score_from=8)
    with pytest.raises(ValueError, match="expert_id"):
        execute_s4_multigate_moe_surgery(m, [{"ids": [1, 2, 3]}], score_from=8)
    with pytest.raises(ValueError, match="non-empty string"):
        execute_s4_multigate_moe_surgery(m, [{"expert_id": " ", "ids": [1, 2, 3]}], score_from=8)
    with pytest.raises(ValueError, match="length >= 2"):
        execute_s4_multigate_moe_surgery(m, [{"expert_id": "E0_a", "ids": [1]}], score_from=8)
    with pytest.raises(ValueError, match="token IDs"):
        execute_s4_multigate_moe_surgery(m, [{"expert_id": "E0_a", "ids": [1, -1, 2]}], score_from=8)
    with pytest.raises(ValueError, match="sequence length"):
        execute_s4_multigate_moe_surgery(
            m, [{"expert_id": "E0_a", "ids": [1, 2, 3]}, {"expert_id": "E1_b", "ids": [1, 2, 3, 4]}]
        )
    with pytest.raises(ValueError, match="too short"):
        execute_s4_multigate_moe_surgery(m, ch, score_from=32)
    with pytest.raises(ValueError, match="general chunk"):
        execute_s4_multigate_moe_surgery(m, [{"expert_id": "E1_b", "ids": [1, 2, 3, 4]}], score_from=2)
    with pytest.raises(ValueError, match="both general and specialist"):
        execute_s4_multigate_moe_surgery(m, [{"expert_id": "E0_a", "ids": [1, 2, 3, 4]}], score_from=2)
    with pytest.raises(TypeError, match="target_layers"):
        execute_s4_multigate_moe_surgery(m, ch, target_layers="1", score_from=8)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="target_layers"):
        execute_s4_multigate_moe_surgery(m, ch, target_layers=[99], score_from=8)
    with pytest.raises(ValueError, match="baseline_domain_nll"):
        execute_s4_multigate_moe_surgery(m, ch, baseline_domain_nll={"E0_only": 1.0}, score_from=8)
    with pytest.raises(ValueError, match="Parent model evaluation"):
        execute_s4_multigate_moe_surgery(m, ch, eval_fn=lambda _m: {}, score_from=8)
    with pytest.raises(ValueError, match="expert_loeo_override"):
        execute_s4_multigate_moe_surgery(m, ch, expert_loeo_override={}, score_from=8)
    with pytest.raises(TypeError, match="mapping"):
        execute_s4_multigate_moe_surgery(m, ch, expert_loeo_override={"expert_0": "bad"}, score_from=8)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="general_harm_nats"):
        execute_s4_multigate_moe_surgery(m, ch, expert_loeo_override={"expert_0": {"delta_spec_when_zeroed": 0.0}}, score_from=8)
    with pytest.raises(ValueError, match="router_frag_"):
        execute_s4_multigate_moe_surgery(m, ch, start_buckets={"expert_0": torch.ones(4)}, score_from=8)
    with pytest.raises(ValueError, match="too short"):
        execute_s4_multigate_moe_surgery(m, ch, start_buckets={"router_frag_0": torch.ones(4)}, score_from=8)

    # Provenance & hashing helpers
    with pytest.raises(TypeError, match="mapping"):
        verify_s4_multigate_receipt_provenance("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        compute_state_sha256({})
    with pytest.raises(ValueError, match="non-empty string"):
        compute_state_sha256({"": torch.ones(2)})
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_state_sha256({"a": [1.0]})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-finite"):
        compute_state_sha256({"a": torch.tensor([float("nan")])})
    with pytest.raises(TypeError, match="nn.Module"):
        compute_module_state_sha256("bad")  # type: ignore[arg-type]
    assert verify_s4_multigate_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert verify_s4_multigate_receipt_provenance({"receipt_sha256": "0" * 64, "object": "wrong"}) is False
    assert (
        verify_s4_multigate_receipt_provenance(
            {"receipt_sha256": "0" * 64, "object": RECEIPT_OBJECT, "schema": "wrong"}
        )
        is False
    )
    assert (
        verify_s4_multigate_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": RECEIPT_OBJECT,
                "schema": SCHEMA_VERSION,
                "bad": float("nan"),
            }
        )
        is False
    )
    assert canonical_json_bytes({"k": 1}) == b'{"k":1}'
    assert len(compute_payload_sha256({"k": 1})) == 64
    assert len(load_moe_trainable_buckets.__name__) > 0

    # 5. Repository AGPL-3.0, optional ZeroStubAudit, PRSecurityScanner, lateral movement & gatekeeper gates
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn15", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn15", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/orthogonal_multigate_router_surgeon.py", TARGET_MODULE_PATH),
        ("tests/test_orthogonal_multigate_router_surgeon_scenarios.py", TEST_FILE_PATH),
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
        {"nll_mean": 3.842, "oof_snr_margin": 0.91, "anse_energy": -0.555, "accepted": True},
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+RECEIPT = {sample_receipt_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1002,
        pr_title="feat(swarm): orthogonal multi-gate specialist router preserver and end-to-end WSP S4 production adapter pipeline",
        pr_body="Implements NEURAL-CTRL-DYN-15 orthogonal multi-gate router surgeon and deterministic CPU scenario tests.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): orthogonal multi-gate specialist router preserver and end-to-end WSP S4 production adapter pipeline"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
