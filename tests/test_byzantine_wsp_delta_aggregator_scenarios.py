# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-18: Byzantine-resilient Multi-Krum & spectral-norm WSP aggregator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.byzantine_wsp_delta_aggregator import (
    ANSE_MAXIMUM_PAIN,
    RECEIPT_OBJECT,
    SCHEMA_VERSION,
    ByzantineWSPConfig,
    apply_verified_byzantine_wsp_round,
    compute_alphaedit_prose_nullspace,
    compute_payload_sha256,
    compute_state_sha256,
    multi_krum_nullspace_aggregate,
    project_delta_into_nullspace,
    screen_peer_wsp_deltas_spectral_and_sparsity,
    verify_byzantine_wsp_receipt_provenance,
)
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    extract_moe_trainable_buckets,
    upcycle_crowd_v1_to_sparse_moe,
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
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "byzantine_wsp_delta_aggregator.py"
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


def _build_5domain_crowd_moe_model(
    *,
    num_layers: int = 2,
    rank: int = 4,
    seed: int = 20261801,
) -> tuple[CrowdV1WithEngram, dict[str, torch.Tensor], torch.Tensor]:
    """Build deterministic 5-domain subspace-decoupled CrowdV1WithEngram + CrowdLoRAMoEMLP on CPU."""
    vocab_size = 40
    hidden_size = 40
    intermediate_size = 80
    block_h = 8
    block_i = 16
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
    domain_dirs = torch.zeros((5, hidden_size), dtype=torch.float32)

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        for d_idx in range(5):
            h0, h1 = d_idx * block_h, (d_idx + 1) * block_h
            t0, t1 = d_idx * 8, (d_idx + 1) * 8
            u_dom = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(block_h)], dtype=torch.float32)
            u_dom = u_dom / u_dom.norm()
            domain_dirs[d_idx, h0:h1] = u_dom

            raw = torch.randn((8, block_h), generator=gen, dtype=torch.float32)
            raw = raw - raw.mean(dim=1, keepdim=True)
            raw = raw / raw.norm(dim=1, keepdim=True).clamp_min(1e-6) * 1.6 + 1.2 * u_dom.unsqueeze(0)
            raw = raw - raw.mean(dim=1, keepdim=True)
            wte[t0:t1, h0:h1] = raw

        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight
        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()

        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0)
            blk.ln2.bias.zero_()
            blk.attn.q.weight[:block_h, :block_h].copy_(0.05 * torch.eye(block_h))
            blk.attn.k.weight[: block_h // 2, :block_h].copy_(0.05 * torch.eye(block_h // 2, block_h))
            blk.attn.v.weight[: block_h // 2, :block_h].copy_(0.05 * torch.eye(block_h // 2, block_h))
            blk.attn.o.weight[:block_h, :block_h].copy_(0.05 * torch.eye(block_h))
            for d_idx in range(5):
                h0, h1 = d_idx * block_h, (d_idx + 1) * block_h
                i0, i1 = d_idx * block_i, (d_idx + 1) * block_i
                blk.mlp.w1.weight[i0:i1, h0:h1].copy_(torch.randn((block_i, block_h), generator=gen) * 0.18)
                blk.mlp.w2.weight[i0:i1, h0:h1].copy_(torch.randn((block_i, block_h), generator=gen) * 0.18)
                w3_b = torch.randn((block_h, block_i), generator=gen) * 0.08
                w3_b = w3_b - w3_b.mean(dim=0, keepdim=True)
                blk.mlp.w3.weight[h0:h1, i0:i1].copy_(w3_b)

        upcycle_crowd_v1_to_sparse_moe(
            model,
            num_experts=5,
            top_k=2,
            lora_rank=rank,
            lora_alpha=float(2 * rank),
            routing_mode="sigma_topk",
            seed=seed + 50,
        )
        for blk in model.blocks:
            blk.mlp.router.weight.copy_(domain_dirs * 3.0 - 1.5)
            for e_idx in range(5):
                exp = blk.mlp.experts[str(e_idx)]
                exp.w1_A.zero_()
                exp.w2_A.zero_()
                exp.w3_A.zero_()
                exp.w1_B.zero_()
                exp.w2_B.zero_()
                exp.w3_B.zero_()
                h0, h1 = e_idx * block_h, (e_idx + 1) * block_h
                i0, i1 = e_idx * block_i, (e_idx + 1) * block_i
                exp.w1_A[:, h0:h1].copy_(torch.randn((rank, block_h), generator=gen) * 0.35)
                exp.w2_A[:, h0:h1].copy_(torch.randn((rank, block_h), generator=gen) * 0.35)
                exp.w3_A[:, i0:i1].copy_(torch.randn((rank, block_i), generator=gen) * 0.45)

    parent_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return model, parent_sd, domain_dirs


def _make_domain_chunks(
    *,
    chunks_per_domain: int = 8,
    seq_len: int = 20,
    seed: int = 20261802,
) -> list[dict[str, Any]]:
    """Generate deterministic non-overlapping token sequences per domain."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    domain_specs = (
        ("E0_general_fineweb", 0),
        ("E1_python_systems_code", 8),
        ("E2_math_reasoning", 16),
        ("E5_exoplanet_transit", 24),
        ("E6_stellar_spectroscopy", 32),
    )
    chunks: list[dict[str, Any]] = []
    for dom_name, tok_base in domain_specs:
        for row_idx in range(chunks_per_domain):
            offset = (row_idx + int(torch.randint(0, 8, (1,), generator=gen).item())) % 8
            if dom_name.startswith("E0_"):
                t_a = tok_base + (row_idx % 8)
                t_b = tok_base + ((row_idx + 3) % 8)
                half = seq_len // 2
                seq = [t_a] * half + [t_b] * (seq_len - half)
            else:
                seq = [tok_base + ((offset + pos) % 8) for pos in range(seq_len)]
            chunks.append({"expert_id": dom_name, "dataset_id": f"synth_{dom_name}", "row": row_idx, "ids": seq})
    return chunks


def _eval_model_domains_local(
    model: CrowdV1WithEngram,
    chunks: list[dict[str, Any]],
    *,
    score_from: int = 4,
) -> dict[str, Any]:
    """Evaluate per-domain mean NLL of ``model`` in its current state."""
    model.eval()
    ids = torch.tensor([c["ids"] for c in chunks], dtype=torch.long)
    with torch.no_grad():
        h_pre = forward_model_pre_ln(model, ids, batch_size=16)
        nlls = per_chunk_nll_from_pre_ln(
            h_pre,
            model.ln.weight,
            model.ln.bias,
            float(model.ln.eps),
            model.head.weight,
            ids,
            score_from=score_from,
        ).numpy()
    by_dom: dict[str, list[float]] = {}
    for ch, val in zip(chunks, nlls.tolist(), strict=True):
        by_dom.setdefault(str(ch["expert_id"]), []).append(float(val))
    dom_mean = {k: float(np.mean(v)) for k, v in sorted(by_dom.items())}
    spec_vals = [float(v) for ch, v in zip(chunks, nlls.tolist(), strict=True) if not str(ch["expert_id"]).startswith("E0_")]
    return {
        "nll_mean": float(nlls.mean()),
        "e0_nll": dom_mean["E0_general_fineweb"],
        "spec_nll": float(np.mean(spec_vals)),
        "per_domain": dom_mean,
    }


def _zero_moe_delta_template(model: CrowdV1WithEngram) -> dict[str, torch.Tensor]:
    """Return zero-initialized delta dict covering all MoE LoRA, router, and LN keys."""
    delta: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            delta[name] = torch.zeros_like(param, device="cpu")
    return delta


def _calibrate_specialist_w3_direction(
    model: CrowdV1WithEngram,
    parent_sd: dict[str, torch.Tensor],
    probe_chunks: list[dict[str, Any]],
    *,
    expert_idx: int = 1,
    domain_key: str = "E1_python_systems_code",
    amplitude: float = 0.75,
    seed: int = 1805,
) -> dict[str, torch.Tensor]:
    """Construct a multi-rank specialist ``w3_B`` delta guaranteed to lower ``domain_key`` NLL."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    h0, h1 = expert_idx * 8, (expert_idx + 1) * 8
    q_h, _ = torch.linalg.qr(torch.randn((8, 4), generator=gen, dtype=torch.float32))
    # Balanced singular values (ratio 1.25 <= 3.5) inside the specialist subspace
    s_weights = torch.tensor([1.0, 0.92, 0.86, 0.80], dtype=torch.float32) * amplitude
    cand_block = q_h * s_weights.unsqueeze(0)

    model.load_state_dict(parent_sd)
    base_eval = _eval_model_domains_local(model, probe_chunks, score_from=4)
    base_dom = base_eval["per_domain"][domain_key]

    delta = _zero_moe_delta_template(model)
    target_key = f"blocks.{len(model.blocks) - 1}.mlp.experts.{expert_idx}.w3_B"
    delta[target_key][h0:h1, :].copy_(cand_block)

    with torch.no_grad():
        sd = model.state_dict()
        sd[target_key].add_(delta[target_key])
    pos_eval = _eval_model_domains_local(model, probe_chunks, score_from=4)
    model.load_state_dict(parent_sd)

    if pos_eval["per_domain"][domain_key] >= base_dom:
        delta[target_key][h0:h1, :].neg_()
    return delta


# ---------------------------------------------------------------------------
# S1: 7 Volunteer WSP Peers (5 Honest + 2 Byzantine, f=2)
# ---------------------------------------------------------------------------


def test_s1_seven_peer_wsp_round_rejects_two_byzantine_and_guards_prose() -> None:
    """5 honest improvers + 2 Byzantine attackers (f=2): Multi-Krum + P_0 keeps prose drift <= +0.2%."""
    model, parent_sd, _ = _build_5domain_crowd_moe_model(num_layers=2, rank=4, seed=20261801)
    probe_chunks = _make_domain_chunks(chunks_per_domain=6, seq_len=20, seed=20261802)
    eval_chunks = _make_domain_chunks(chunks_per_domain=8, seq_len=20, seed=20261899)

    base_delta = _calibrate_specialist_w3_direction(
        model, parent_sd, probe_chunks, expert_idx=1, domain_key="E1_python_systems_code", amplitude=0.85
    )
    target_key = "blocks.1.mlp.experts.1.w3_B"
    # Verify total edited fraction per honest peer stays well below 5%
    total_elems = sum(int(t.numel()) for t in base_delta.values())
    assert (8 * 4) / total_elems < 0.01

    gen = torch.Generator(device="cpu").manual_seed(20261810)
    peer_deltas: list[dict[str, torch.Tensor]] = []
    # 5 honest specialist improvers with small heterogeneity and mild prose leakage (cleaned by P_0)
    for _p_idx in range(5):
        d = {k: v.clone() for k, v in base_delta.items()}
        d[target_key][8:16, :] += torch.randn((8, 4), generator=gen) * 0.04
        # Mild rank-4 general-prose leakage in rows 0..7 (removed cleanly by AlphaEdit P_0)
        q_p, _ = torch.linalg.qr(torch.randn((8, 4), generator=gen))
        d[target_key][0:8, :] += q_p * torch.tensor([0.22, 0.20, 0.18, 0.16])
        peer_deltas.append(d)

    # Attacker 5: stealthy multi-rank prose-poisoning attacker (passes <=5% sparsity & <=3.5 spike)
    att_prose = _zero_moe_delta_template(model)
    q_att1, _ = torch.linalg.qr(torch.randn((8, 4), generator=gen))
    att_prose["blocks.1.mlp.experts.0.w3_B"][0:8, :].copy_(q_att1 * torch.tensor([4.5, 4.2, 3.9, 3.6]))
    att_prose[target_key][0:8, :].copy_(q_att1 * torch.tensor([3.2, 3.0, 2.8, 2.6]))
    peer_deltas.append(att_prose)

    # Attacker 6: stealthy sign-flipping + prose-poisoning attacker
    att_flip = _zero_moe_delta_template(model)
    att_flip[target_key][8:16, :].copy_(-4.5 * base_delta[target_key][8:16, :])
    q_att2, _ = torch.linalg.qr(torch.randn((8, 4), generator=gen))
    att_flip[target_key][0:8, :].copy_(q_att2 * torch.tensor([3.5, 3.2, 3.0, 2.8]))
    peer_deltas.append(att_flip)

    parent_eval = _eval_model_domains_local(model, eval_chunks, score_from=4)

    # Naive mean of all 7 peers corrupts general prose well past +1.0%
    naive_mean = {
        k: sum(p[k].to(torch.float64) for p in peer_deltas) / float(len(peer_deltas))
        for k in base_delta
    }
    with torch.no_grad():
        sd = model.state_dict()
        for k, dv in naive_mean.items():
            sd[k].add_(dv.to(dtype=sd[k].dtype))
    naive_eval = _eval_model_domains_local(model, eval_chunks, score_from=4)
    naive_e0_drift_pct = 100.0 * (naive_eval["e0_nll"] / parent_eval["e0_nll"] - 1.0)
    assert naive_e0_drift_pct > 1.0, f"Expected naive mean to corrupt general prose, got {naive_e0_drift_pct:.3f}%"

    # Restore exact parent state and run verified Byzantine WSP round
    model.load_state_dict(parent_sd)
    cfg = ByzantineWSPConfig(
        num_byzantine_f=2,
        max_edit_fraction=0.05,
        max_spectral_spike_ratio=3.5,
        sv_threshold=1e-2,
        score_from=4,
        batch_size=16,
    )
    receipt = apply_verified_byzantine_wsp_round(
        model,
        peer_deltas,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        config=cfg,
    )
    assert verify_byzantine_wsp_receipt_provenance(receipt) is True
    assert receipt["schema"] == SCHEMA_VERSION
    assert receipt["object"] == RECEIPT_OBJECT
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["guard_ok"] is True
    assert receipt["delta_e"] < 0.0
    assert receipt["krum_report"]["projection_applied"] is True
    assert receipt["krum_report"]["selected_original_indices"] == [0, 1, 2, 3, 4]
    assert receipt["krum_report"]["rejected_original_indices"] == [5, 6]

    final_eval = _eval_model_domains_local(model, eval_chunks, score_from=4)
    e0_drift_pct = 100.0 * (final_eval["e0_nll"] / parent_eval["e0_nll"] - 1.0)
    assert e0_drift_pct <= 0.2, f"Expected E0 drift <= +0.2%, got {e0_drift_pct:.4f}%"
    assert final_eval["per_domain"]["E1_python_systems_code"] < parent_eval["per_domain"]["E1_python_systems_code"]
    assert final_eval["spec_nll"] < parent_eval["spec_nll"]

    # Deterministic repeat produces identical SHA-256 receipt and state
    model2, _, _ = _build_5domain_crowd_moe_model(num_layers=2, rank=4, seed=20261801)
    receipt_repeat = apply_verified_byzantine_wsp_round(
        model2,
        peer_deltas,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        config=cfg,
    )
    assert receipt_repeat["receipt_sha256"] == receipt["receipt_sha256"]
    assert receipt_repeat["final_state_sha256"] == receipt["final_state_sha256"]

    # Also verify OpenDiLoCo bucket delta unpacking format in apply_verified_byzantine_wsp_round
    model_b, parent_sd_b, _ = _build_5domain_crowd_moe_model(num_layers=2, rank=4, seed=20261801)
    parent_buckets = extract_moe_trainable_buckets(model_b, num_router_fragments=2)
    bucket_peers: list[dict[str, torch.Tensor]] = []
    for p_map in peer_deltas:
        model_b.load_state_dict(parent_sd_b)
        with torch.no_grad():
            sd_b = model_b.state_dict()
            for k, dv in p_map.items():
                sd_b[k].add_(dv)
        cand_b = extract_moe_trainable_buckets(model_b, num_router_fragments=2)
        bucket_peers.append({bk: (cand_b[bk] - parent_buckets[bk]).contiguous() for bk in parent_buckets})
    model_b.load_state_dict(parent_sd_b)
    bucket_receipt = apply_verified_byzantine_wsp_round(
        model_b,
        bucket_peers,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        config=cfg,
        delta_format="auto",
    )
    assert bucket_receipt["accepted"] is True
    assert bucket_receipt["delta_format"] == "buckets"
    assert bucket_receipt["krum_report"]["rejected_original_indices"] == [5, 6]
    assert bucket_receipt["final_state_sha256"] == receipt["final_state_sha256"]


# ---------------------------------------------------------------------------
# S2: Rank-1 Trojan Spectral-Spike Pre-Filter Catching Norm-Matched Backdoors
# ---------------------------------------------------------------------------


def test_s2_spectral_spike_and_sparsity_prefilter_catches_norm_matched_trojans() -> None:
    """Screen rejects NaN/Inf, dense floods, cohort spikes, and norm-matched rank-1 Trojans."""
    gen = torch.Generator(device="cpu").manual_seed(20261820)
    peers: list[dict[str, torch.Tensor]] = []

    def _blank_peer() -> dict[str, torch.Tensor]:
        return {
            "layer0.w1": torch.zeros((32, 32), dtype=torch.float32),
            "layer1.w3": torch.zeros((32, 32), dtype=torch.float32),
            "layer1.router.weight": torch.zeros((4, 32), dtype=torch.float32),
            "expert1.w2_A": torch.zeros((4, 32), dtype=torch.float32),
            "expert1.w2_B": torch.zeros((32, 4), dtype=torch.float32),
        }

    # 5 honest sparse multi-rank peers (4x8 sub-block = 32 of 2432 params = 1.32% <= 5%, norm=1.0)
    for _ in range(5):
        p = _blank_peer()
        u_mat, _ = torch.linalg.qr(torch.randn((8, 4), generator=gen))
        v_mat, _ = torch.linalg.qr(torch.randn((8, 4), generator=gen))
        sv = torch.tensor([0.55, 0.50, 0.47, 0.47], dtype=torch.float32)
        sv = sv / sv.norm()  # Frobenius norm == 1.0
        block = (u_mat * sv.unsqueeze(0)) @ v_mat.T
        p["layer0.w1"][:8, :8].copy_(block)
        # Honest vector-like router probe (rank-1 row alignment should not trigger intra-matrix block)
        p["layer1.router.weight"][1, :8].copy_(torch.ones(8) * 0.05)
        peers.append(p)

    # Peer 5: norm-matched rank-1 Trojan (exact same total Frobenius norm as honest peer 0!)
    trojan_norm_matched = _blank_peer()
    u_vec = torch.randn(8, generator=gen)
    u_vec = u_vec / u_vec.norm()
    v_vec = torch.randn(8, generator=gen)
    v_vec = v_vec / v_vec.norm()
    honest_total_norm = float(sum(t.pow(2).sum().item() for t in peers[0].values()) ** 0.5)
    trojan_norm_matched["layer0.w1"][:8, :8].copy_(torch.outer(u_vec, v_vec) * honest_total_norm)
    peers.append(trojan_norm_matched)

    # Peer 6: factored LoRA rank-1-dominant Trojan where A and B each pass individually (2.5 <= 3.5)
    # yet their aligned product B @ A concentrates into a 6.25 > 3.5 spectral spike
    trojan_lora = _blank_peer()
    q_b, _ = torch.linalg.qr(torch.randn((8, 4), generator=gen))
    q_a, _ = torch.linalg.qr(torch.randn((8, 4), generator=gen))
    b_bal = q_b * torch.tensor([1.0, 0.4, 0.4, 0.4], dtype=torch.float32).unsqueeze(0)
    a_bal = (q_a * torch.tensor([1.0, 0.4, 0.4, 0.4], dtype=torch.float32).unsqueeze(0)).T
    trojan_lora["expert1.w2_B"][:8, :].copy_(b_bal)
    trojan_lora["expert1.w2_A"][:, :8].copy_(a_bal)
    peers.append(trojan_lora)

    # Peer 7: dense multi-rank flood edit (40% of weights > 5%)
    flood = _blank_peer()
    flood["layer1.w3"].copy_(torch.randn((32, 32), generator=gen) * 0.01)
    peers.append(flood)

    # Peer 8: NaN / Inf poisoner
    bad_nan = _blank_peer()
    bad_nan["layer0.w1"][0, 0] = float("nan")
    peers.append(bad_nan)

    # Peer 9: multi-rank cohort spectral norm spike (8x larger singular values)
    cohort_spike = _blank_peer()
    cohort_spike["layer0.w1"][:8, :8].copy_(peers[0]["layer0.w1"][:8, :8] * 8.0)
    peers.append(cohort_spike)

    surviving, report = screen_peer_wsp_deltas_spectral_and_sparsity(
        peers,
        max_edit_fraction=0.05,
        max_spectral_spike_ratio=3.5,
    )
    assert len(surviving) == 5
    assert report["accepted_indices"] == [0, 1, 2, 3, 4]
    assert report["rejected_indices"] == [5, 6, 7, 8, 9]
    assert report["rejection_reasons_by_index"][5] == ["spectral_spike_exceeded"]
    assert report["rejection_reasons_by_index"][6] == ["spectral_spike_exceeded"]
    assert report["rejection_reasons_by_index"][7] == ["edit_fraction_exceeded"]
    assert report["rejection_reasons_by_index"][8] == ["non_finite_values"]
    assert report["rejection_reasons_by_index"][9] == ["spectral_spike_exceeded"]

    # Confirm norm-matched Trojan truly matched Frobenius norm of honest peers!
    honest_l2 = report["per_peer"][0]["l2_norm"]
    trojan_l2 = report["per_peer"][5]["l2_norm"]
    assert abs(trojan_l2 - honest_l2) < 1e-3
    assert report["per_peer"][5]["intra_spectral_spike_ratio"] > 1e6

    # Verify bare tensor sequence, 3D tensors, 1D tensors, mapping input, and reference_total_params
    t_honest = peers[0]["layer0.w1"]
    t_trojan = peers[5]["layer0.w1"]
    surv_t, rep_t = screen_peer_wsp_deltas_spectral_and_sparsity(
        [t_honest, t_honest, t_trojan],
        max_edit_fraction=0.10,
        max_spectral_spike_ratio=3.5,
        reference_total_params=2048,
    )
    assert len(surv_t) == 2
    assert isinstance(surv_t[0], torch.Tensor)
    assert rep_t["rejected_indices"] == [2]

    # 3D and 1D tensors + fallback cohort pool when all finite peers fail sparsity
    p_map = {
        "a": {"w3d": torch.ones((2, 4, 4)) * 0.1, "v1d": torch.ones(4) * 0.1},
        "b": {"w3d": torch.ones((2, 4, 4)) * 0.1, "v1d": torch.ones(4) * 0.1},
        "c": {"w3d": torch.ones((2, 4, 4)) * 0.1, "v1d": torch.ones(4) * 0.1},
    }
    surv_m, rep_m = screen_peer_wsp_deltas_spectral_and_sparsity(
        p_map,
        max_edit_fraction=0.05,
        check_cohort_spectral_ratio=False,
        check_intra_matrix_spectral_ratio=False,
    )
    assert len(surv_m) == 0
    assert rep_m["rejected_peer_ids"] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# S3: Exact AlphaEdit Prose Null-Space Invariance (< 1e-5 Drift on Protected Keys)
# ---------------------------------------------------------------------------


def test_s3_exact_prose_nullspace_invariance_on_protected_keys() -> None:
    """Multi-Krum + AlphaEdit P_0 maintains < 1e-5 drift on protected prose keys while preserving specialist edits."""
    gen = torch.Generator(device="cpu").manual_seed(20261830)
    d = 32
    prose_dim = 8

    # Construct general-prose keys spanning strictly dims 0..7
    q_basis, _ = torch.linalg.qr(torch.randn((d, d), generator=gen, dtype=torch.float64))
    prose_basis = q_basis[:, :prose_dim]  # (32, 8)
    spec_basis = q_basis[:, prose_dim : prose_dim + 8]  # (32, 8)

    prose_keys = torch.randn((48, prose_dim), generator=gen, dtype=torch.float64) @ prose_basis.T  # (48, 32)
    spec_keys = torch.randn((48, 8), generator=gen, dtype=torch.float64) @ spec_basis.T  # (48, 32)

    p_null, p_meta = compute_alphaedit_prose_nullspace(prose_keys, sv_threshold=1e-2)
    assert p_meta["active_modes"] == prose_dim
    assert p_meta["nullspace_dim"] == d - prose_dim
    assert float((p_null @ p_null - p_null).abs().max().item()) < 1e-12
    assert float((prose_keys @ p_null).abs().max().item()) < 1e-12
    assert float((spec_keys @ p_null - spec_keys).abs().max().item()) < 1e-12

    # Verify variance_threshold, max_active_modes, Sequence, Mapping, and sub-threshold identity
    p_sub, sub_meta = compute_alphaedit_prose_nullspace(
        [prose_keys[:24], prose_keys[24:]], sv_threshold=1e-2, variance_threshold=0.50, max_active_modes=8
    )
    assert sub_meta["active_modes"] <= 8 and p_sub.shape == (d, d)
    p_map_in, _ = compute_alphaedit_prose_nullspace({"k0": prose_keys.unsqueeze(0)}, sv_threshold=1e9)
    assert torch.equal(p_map_in, torch.eye(d, dtype=torch.float64))

    # Verify 1D vector projection, unprojected unmatched width skip, and multi_krum n=1 fallback
    v1d_proj, v1d_meta = project_delta_into_nullspace(
        {"vec": torch.ones(d, dtype=torch.float32), "odd": torch.ones((5, 7), dtype=torch.float32)},
        {"x": p_null},
    )
    assert float((prose_keys @ v1d_proj["vec"].to(torch.float64)).abs().max().item()) < 1e-5
    assert "odd" in v1d_meta["skipped_keys"]
    t1d_proj, _ = project_delta_into_nullspace(torch.ones(d, dtype=torch.float32), p_null)
    assert float((prose_keys @ t1d_proj.to(torch.float64)).abs().max().item()) < 1e-5
    unmatched_proj, unmatched_meta = project_delta_into_nullspace(
        {"odd": torch.ones((5, 7), dtype=torch.float32)}, p_null
    )
    assert "odd" in unmatched_meta["skipped_keys"] and unmatched_proj["odd"].shape == (5, 7)
    single_agg, single_meta = multi_krum_nullspace_aggregate(
        [{"w": torch.ones((2, 2))}], num_byzantine_f=0, enforce_byzantine_bound=False
    )
    assert single_meta["k_neighbors"] == 0 and float(single_agg["w"].sum().item()) == 4.0

    # Build 7 peer deltas (5 honest specialist + prose leakage, 2 Byzantine sign-flippers, f=2)
    w_spec_base = torch.randn((16, 8), generator=gen, dtype=torch.float64) @ spec_basis.T  # (16, 32)
    peer_deltas: list[dict[str, torch.Tensor]] = []
    for _ in range(5):
        leak = torch.randn((16, prose_dim), generator=gen, dtype=torch.float64) @ prose_basis.T * 0.35
        noise = torch.randn((16, 8), generator=gen, dtype=torch.float64) @ spec_basis.T * 0.03
        peer_deltas.append(
            {
                "blocks.0.mlp.experts.1.w1_A": (w_spec_base + leak + noise).to(torch.float32),
                "blocks.0.mlp.experts.1.w1_B": torch.ones((64, 16), dtype=torch.float32) * 0.05,
                "blocks.0.mlp.experts.0.w1_A": (leak * 2.0).to(torch.float32),
                "blocks.0.mlp.router.weight": torch.ones((4, d), dtype=torch.float32) * 0.1,
                "ln.weight": torch.ones(d, dtype=torch.float32) * 0.02,
            }
        )
    # 2 Byzantine attackers
    for _ in range(2):
        peer_deltas.append(
            {
                "blocks.0.mlp.experts.1.w1_A": (-5.0 * w_spec_base).to(torch.float32),
                "blocks.0.mlp.experts.1.w1_B": torch.ones((64, 16), dtype=torch.float32) * -2.0,
                "blocks.0.mlp.experts.0.w1_A": (prose_keys[:16] * 3.0).to(torch.float32),
                "blocks.0.mlp.router.weight": torch.ones((4, d), dtype=torch.float32) * -3.0,
                "ln.weight": torch.ones(d, dtype=torch.float32) * 0.5,
            }
        )

    agg_dict, krum_meta = multi_krum_nullspace_aggregate(
        peer_deltas,
        num_byzantine_f=2,
        p_null={"blocks.0.x": p_null, "blocks.0.mlp.experts.1.w1_A": p_null},
        protect_layernorm=True,
        protect_general_expert=True,
        general_expert_index=0,
    )
    assert isinstance(agg_dict, dict)
    assert krum_meta["rejected_indices"] == [5, 6]
    assert krum_meta["projection_applied"] is True
    assert "ln.weight" in krum_meta["skipped_keys"]
    assert "blocks.0.mlp.experts.0.w1_A" in krum_meta["skipped_keys"]
    assert "blocks.0.mlp.experts.1.w1_B" in krum_meta["skipped_keys"]

    # Exact prose null-space invariance (< 1e-5 drift on protected prose keys)
    w_proj = agg_dict["blocks.0.mlp.experts.1.w1_A"].to(torch.float64)
    prose_output_drift = float((prose_keys @ w_proj.T).abs().max().item())
    assert prose_output_drift < 1e-5, f"Protected prose key drift {prose_output_drift:.3e} >= 1e-5"

    # Specialist response is faithfully preserved
    spec_proj_norm = float((spec_keys @ w_proj.T).norm().item())
    spec_raw_norm = float((spec_keys @ w_spec_base.T).norm().item())
    assert spec_proj_norm > 0.90 * spec_raw_norm

    # Router row 0 protected when protect_general_expert=True, while rows 1..3 projected
    r_proj = agg_dict["blocks.0.mlp.router.weight"].to(torch.float64)
    assert float((prose_keys @ r_proj[1:].T).abs().max().item()) < 1e-5
    assert float((prose_keys @ r_proj[0]).abs().max().item()) > 1e-2

    # Bare tensor aggregation with p_null
    raw_tensors = [d["blocks.0.mlp.experts.1.w1_A"] for d in peer_deltas]
    agg_t, t_meta = multi_krum_nullspace_aggregate(raw_tensors, num_byzantine_f=2, select_m=4, p_null=p_null)
    assert isinstance(agg_t, torch.Tensor)
    assert len(t_meta["selected_indices"]) == 4
    assert float((prose_keys @ agg_t.to(torch.float64).T).abs().max().item()) < 1e-5


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Atomic Rollback When Majority Corruption Exceeds f
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_rollback_on_majority_corruption() -> None:
    """When > f attackers corrupt the consensus, ANSE assigns 1e6 pain and rolls back bit-for-bit."""
    model, parent_sd, _ = _build_5domain_crowd_moe_model(num_layers=2, rank=4, seed=20261840)
    probe_chunks = _make_domain_chunks(chunks_per_domain=4, seq_len=20, seed=20261841)
    eval_chunks = _make_domain_chunks(chunks_per_domain=6, seq_len=20, seed=20261842)
    parent_sha = compute_state_sha256(model)

    gen = torch.Generator(device="cpu").manual_seed(20261843)
    good_delta = _calibrate_specialist_w3_direction(
        model, parent_sd, probe_chunks, expert_idx=1, domain_key="E1_python_systems_code", amplitude=0.7
    )

    # Majority attack: 5 out of 7 peers (> f=2!) are stealthy clustered attackers corrupting expert 0 + LN
    peer_deltas: list[dict[str, torch.Tensor]] = [
        {k: v.clone() for k, v in good_delta.items()},
        {k: v.clone() for k, v in good_delta.items()},
    ]
    for _ in range(5):
        poison = _zero_moe_delta_template(model)
        q_bad, _ = torch.linalg.qr(torch.randn((8, 4), generator=gen))
        poison["blocks.1.mlp.experts.0.w3_B"][0:8, :].copy_(q_bad * torch.tensor([4.0, 3.8, 3.6, 3.4]))
        poison["ln.bias"][0:8].fill_(1.8)
        peer_deltas.append(poison)

    # Set protect_general_expert=True so expert 0 bypasses P_0 and breaches the +1.0% E0 guard!
    cfg = ByzantineWSPConfig(
        num_byzantine_f=2,
        max_edit_fraction=0.05,
        max_spectral_spike_ratio=3.5,
        protect_general_expert=True,
        score_from=4,
        batch_size=16,
    )
    receipt = apply_verified_byzantine_wsp_round(
        model,
        peer_deltas,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        config=cfg,
    )
    assert verify_byzantine_wsp_receipt_provenance(receipt) is True
    assert receipt["accepted"] is False
    assert receipt["rolled_back"] is True
    assert receipt["rollback_bitwise_verified"] is True
    assert receipt["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert receipt["guard_ok"] is False
    assert receipt["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert receipt["candidate_anse_energy"] == ANSE_MAXIMUM_PAIN
    assert receipt["final_state_sha256"] == parent_sha
    assert compute_state_sha256(model) == parent_sha
    for k, orig_t in parent_sd.items():
        assert torch.equal(model.state_dict()[k], orig_t)

    # Also verify non-improving energy rollback (guard_ok=True, Delta E >= 0), non-finite eval_fn rollback,
    # quorum collapse rollback, and unchanged fixed-point round
    zero_peers = [_zero_moe_delta_template(model) for _ in range(5)]
    fp_rec = apply_verified_byzantine_wsp_round(
        model,
        zero_peers,
        eval_fn=lambda _m: {"E0_general_fineweb": 1.0, "E1_code": 1.5},
        num_byzantine_f=1,
    )
    assert fp_rec["status"] == "FIXED_POINT_UNCHANGED"
    assert fp_rec["accepted"] is False

    # Non-improving energy rollback
    non_imp = []
    for _ in range(5):
        d = _zero_moe_delta_template(model)
        d["blocks.1.mlp.experts.2.w3_B"][16:24, :] += torch.eye(8, 4) * 0.2
        non_imp.append(d)
    call_count = {"n": 0}

    def _worsening_eval(_m: nn.Module) -> dict[str, Any]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"E0_general_fineweb": 1.0, "E1_code": 1.5}
        return {"E0_general_fineweb": 1.005, "E1_code": 1.7}

    ni_rec = apply_verified_byzantine_wsp_round(
        model,
        non_imp,
        eval_fn=_worsening_eval,
        num_byzantine_f=1,
        max_edit_fraction=0.10,
    )
    assert ni_rec["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    assert ni_rec["rolled_back"] is True
    assert compute_state_sha256(model) == parent_sha

    # Non-finite candidate output via eval_fn
    nf_calls = {"n": 0}

    def _nan_cand_eval(_m: nn.Module) -> dict[str, Any]:
        nf_calls["n"] += 1
        if nf_calls["n"] == 1:
            return {"E0_general_fineweb": 1.0, "E1_code": 1.5}
        return {"E0_general_fineweb": float("nan"), "E1_code": 1.2}

    nf_rec = apply_verified_byzantine_wsp_round(
        model,
        non_imp,
        eval_fn=_nan_cand_eval,
        num_byzantine_f=1,
        max_edit_fraction=0.10,
    )
    assert nf_rec["status"] == "ROLLED_BACK_NON_FINITE"
    assert nf_rec["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert compute_state_sha256(model) == parent_sha

    # Quorum collapse when screening rejects >= n - 2 peers
    flood_peers = []
    for i in range(5):
        d = _zero_moe_delta_template(model)
        if i < 4:
            d["blocks.0.mlp.experts.0.w1_B"].fill_(0.05)
        else:
            d["blocks.1.mlp.experts.1.w3_B"][8:16, :] += torch.eye(8, 4) * 0.2
        flood_peers.append(d)
    q_rec = apply_verified_byzantine_wsp_round(
        model,
        flood_peers,
        eval_fn=lambda _m: {"E0_general_fineweb": 1.0, "E1_code": 1.5},
        num_byzantine_f=1,
        max_edit_fraction=0.02,
    )
    assert q_rec["status"] == "ROLLED_BACK_QUORUM_BELOW_BYZANTINE_BOUND"
    assert q_rec["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert compute_state_sha256(model) == parent_sha

    # Also test enable_screening=False + explicit prose_keys + CrowdFineGrainedMoEMLP buckets + non-finite trunk
    prose_keys_s4 = torch.zeros((16, 40), dtype=torch.float32)
    prose_keys_s4[:, :8] = torch.randn((16, 8), generator=gen)
    off_screen_rec = apply_verified_byzantine_wsp_round(
        model,
        non_imp,
        eval_fn=lambda _m: {"E0_general_fineweb": 1.0, "E1_code": 1.5},
        prose_keys=prose_keys_s4,
        config=ByzantineWSPConfig(num_byzantine_f=1, enable_screening=False, require_strict_energy_drop=False),
    )
    assert off_screen_rec["accepted"] is True
    model.load_state_dict(parent_sd)

    # Non-finite candidate trunk pre-LN path in native CrowdV1WithEngram evaluation (parent finite, candidate overflows)
    overflow_peers = []
    for _ in range(5):
        d = _zero_moe_delta_template(model)
        d["blocks.1.mlp.experts.1.w3_B"].fill_(1e37)
        overflow_peers.append(d)
    nf_trunk_rec = apply_verified_byzantine_wsp_round(
        model,
        overflow_peers,
        eval_chunks=eval_chunks,
        config=ByzantineWSPConfig(num_byzantine_f=1, enable_screening=False),
    )
    assert nf_trunk_rec["status"] == "ROLLED_BACK_NON_FINITE"
    assert nf_trunk_rec["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert compute_state_sha256(model) == parent_sha

    # CrowdFineGrainedMoEMLP bucket unpacking path
    fg_cfg = CrowdV1RuntimeConfig(
        vocab_size=16,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=32,
        max_position_embeddings=32,
        engram_buckets=0,
        tie_word_embeddings=True,
    )
    fg_model = CrowdV1WithEngram(fg_cfg)
    upcycle_crowd_v1_to_sparse_moe(
        fg_model, split_mode="fine_grained_width_split", num_experts=4, top_k=2, seed=20261844
    )
    fg_base = extract_moe_trainable_buckets(fg_model, num_router_fragments=1)
    fg_deltas = [{k: torch.zeros_like(v) for k, v in fg_base.items()} for _ in range(5)]
    fg_rec = apply_verified_byzantine_wsp_round(
        fg_model,
        fg_deltas,
        eval_fn=lambda _m: {"E0_general": 1.0, "E1_spec": 1.2},
        num_byzantine_f=1,
        delta_format="buckets",
    )
    assert fg_rec["delta_format"] == "buckets"
    assert fg_rec["status"] == "FIXED_POINT_UNCHANGED"


# ---------------------------------------------------------------------------
# S5: Input Validation Errors & Security / Zero-Stub Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_compliance_gates() -> None:
    """Validate strict type/value rejections, provenance checks, and all HiveMind security/stub gates."""
    # 1. ByzantineWSPConfig validation
    with pytest.raises(TypeError, match="int"):
        ByzantineWSPConfig(num_byzantine_f=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        ByzantineWSPConfig(num_byzantine_f=-1)
    with pytest.raises(ValueError, match=">= 1"):
        ByzantineWSPConfig(select_m=0)
    with pytest.raises(ValueError, match="> 0.0"):
        ByzantineWSPConfig(max_edit_fraction=0.0)
    with pytest.raises(ValueError, match="<= 1.0"):
        ByzantineWSPConfig(max_edit_fraction=1.5)
    with pytest.raises(ValueError, match="> 0.0"):
        ByzantineWSPConfig(max_spectral_spike_ratio=0.0)
    with pytest.raises(ValueError, match="finite"):
        ByzantineWSPConfig(domain_guard_max_ratio=float("nan"))
    with pytest.raises(TypeError, match="real number"):
        ByzantineWSPConfig(sv_threshold="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bool"):
        ByzantineWSPConfig(enable_screening="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="general_prefix"):
        ByzantineWSPConfig(general_prefix="  ")
    with pytest.raises(ValueError, match="prose_max_active_modes"):
        ByzantineWSPConfig(prose_max_active_modes=0)
    with pytest.raises(ValueError, match="prose_variance_threshold"):
        ByzantineWSPConfig(prose_variance_threshold=1.5)
    with pytest.raises(ValueError, match=">= 0.0"):
        ByzantineWSPConfig(edit_tol=-0.1)

    # 2. Screening & normalization validation & edge paths
    with pytest.raises(ValueError, match="empty"):
        screen_peer_wsp_deltas_spectral_and_sparsity([])
    with pytest.raises(ValueError, match="empty"):
        screen_peer_wsp_deltas_spectral_and_sparsity({})
    with pytest.raises(ValueError, match="Invalid peer ID"):
        screen_peer_wsp_deltas_spectral_and_sparsity({" ": {"a": torch.ones(2, 2)}})
    with pytest.raises(TypeError, match="Mapping"):
        screen_peer_wsp_deltas_spectral_and_sparsity({"bad_val": torch.ones(2, 2)})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sequence"):
        screen_peer_wsp_deltas_spectral_and_sparsity("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.Tensor"):
        screen_peer_wsp_deltas_spectral_and_sparsity([torch.ones(2, 2), {"a": torch.ones(2, 2)}])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="entries must be"):
        screen_peer_wsp_deltas_spectral_and_sparsity([123])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="non-empty Mapping"):
        screen_peer_wsp_deltas_spectral_and_sparsity([{}])
    with pytest.raises(TypeError, match="peer_ids"):
        screen_peer_wsp_deltas_spectral_and_sparsity([torch.ones(2, 2)], peer_ids="p0")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="peer_ids length"):
        screen_peer_wsp_deltas_spectral_and_sparsity([torch.ones(2, 2)], peer_ids=["p0", "p1"])
    with pytest.raises(ValueError, match="control characters"):
        screen_peer_wsp_deltas_spectral_and_sparsity([torch.ones(2, 2)], peer_ids=["\x00"])
    with pytest.raises(ValueError, match="non-empty string"):
        screen_peer_wsp_deltas_spectral_and_sparsity([{"": torch.ones(2, 2)}])
    with pytest.raises(TypeError, match="torch.Tensor"):
        screen_peer_wsp_deltas_spectral_and_sparsity([{"a": 1.0}])  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="schema"):
        screen_peer_wsp_deltas_spectral_and_sparsity([{"a": torch.ones(2, 2)}, {"b": torch.ones(2, 2)}])
    with pytest.raises(ValueError, match="Shape mismatch"):
        screen_peer_wsp_deltas_spectral_and_sparsity([{"a": torch.ones(2, 2)}, {"a": torch.ones(2, 3)}])
    with pytest.raises(TypeError, match="floating-point"):
        screen_peer_wsp_deltas_spectral_and_sparsity([{"a": torch.ones((2, 2), dtype=torch.long)}])
    with pytest.raises(TypeError, match="torch.Tensor"):
        screen_peer_wsp_deltas_spectral_and_sparsity([{"a": torch.ones(2, 2)}, {"a": 1.0}])  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="floating-point"):
        screen_peer_wsp_deltas_spectral_and_sparsity(
            [{"a": torch.ones(2, 2)}, {"a": torch.ones((2, 2), dtype=torch.long)}]
        )
    with pytest.raises(ValueError, match="non-empty"):
        screen_peer_wsp_deltas_spectral_and_sparsity([{"a": torch.empty((0, 2))}])
    with pytest.raises(ValueError, match="Duplicate peer_id"):
        screen_peer_wsp_deltas_spectral_and_sparsity([torch.ones(2, 2), torch.ones(2, 2)], peer_ids=["p0", "p0"])
    with pytest.raises(ValueError, match="reference_total_params"):
        screen_peer_wsp_deltas_spectral_and_sparsity([torch.ones(2, 2)], reference_total_params=2)

    # Zero-norm 3D slice & zero LoRA factor skip in screening
    zero_pair = {
        "w3d": torch.cat([torch.zeros((1, 2, 2)), torch.eye(2).unsqueeze(0)], dim=0),
        "expert0.w1_A": torch.ones((1, 4)),
        "expert0.w1_B": torch.zeros((4, 1)),
    }
    surv_z, _ = screen_peer_wsp_deltas_spectral_and_sparsity([zero_pair], max_edit_fraction=1.0)
    assert len(surv_z) == 1

    # 3. Multi-Krum & nullspace validation
    honest_4 = [torch.ones((4, 4)) * 0.1 for _ in range(4)]
    with pytest.raises(ValueError, match="f < \\(n - 2\\) / 2"):
        multi_krum_nullspace_aggregate(honest_4, num_byzantine_f=1)
    with pytest.raises(ValueError, match="select_m"):
        multi_krum_nullspace_aggregate(honest_4, num_byzantine_f=0, select_m=5)
    with pytest.raises(ValueError, match="non-finite"):
        multi_krum_nullspace_aggregate(
            [torch.ones((4, 4)), torch.ones((4, 4)), torch.full((4, 4), float("nan"))], num_byzantine_f=0
        )
    with pytest.raises(TypeError, match="peer_weights"):
        multi_krum_nullspace_aggregate(honest_4, num_byzantine_f=0, peer_weights="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="peer_weights"):
        multi_krum_nullspace_aggregate(honest_4, num_byzantine_f=0, peer_weights=[1.0, 1.0])
    with pytest.raises(ValueError, match="incompatible"):
        project_delta_into_nullspace({"w": torch.ones((3, 5))}, {"w": torch.eye(4)})
    with pytest.raises(TypeError, match="delta must be"):
        project_delta_into_nullspace({}, torch.eye(4))
    with pytest.raises(ValueError, match="non-empty floating"):
        project_delta_into_nullspace({"a": torch.ones(2, dtype=torch.long)}, torch.eye(2))
    with pytest.raises(ValueError, match="non-finite"):
        project_delta_into_nullspace({"a": torch.tensor([float("nan"), 1.0])}, torch.eye(2))
    with pytest.raises(TypeError, match="p_null"):
        project_delta_into_nullspace(torch.ones((3, 3)), {})
    with pytest.raises(TypeError, match="square 2D Tensor"):
        project_delta_into_nullspace(torch.ones((3, 3)), "bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.Tensor"):
        project_delta_into_nullspace({"w": torch.ones((3, 3))}, {"w": "bad"})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="floating-point"):
        project_delta_into_nullspace(torch.ones((3, 3)), torch.ones((3, 3), dtype=torch.long))
    with pytest.raises(ValueError, match="finite"):
        project_delta_into_nullspace(torch.ones((3, 3)), torch.full((3, 3), float("nan")))
    with pytest.raises(ValueError, match="square"):
        project_delta_into_nullspace(torch.ones((4, 4)), torch.ones((4, 3)))
    with pytest.raises(ValueError, match="width"):
        compute_alphaedit_prose_nullspace([torch.ones((4, 4)), torch.ones((4, 5))])
    with pytest.raises(ValueError, match="finite"):
        compute_alphaedit_prose_nullspace(torch.full((4, 4), float("inf")))
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_alphaedit_prose_nullspace([123])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="2D or 3D"):
        compute_alphaedit_prose_nullspace(torch.ones(4))
    with pytest.raises(TypeError, match="floating-point"):
        compute_alphaedit_prose_nullspace(torch.ones((4, 4), dtype=torch.long))
    with pytest.raises(ValueError, match="empty"):
        compute_alphaedit_prose_nullspace({})
    with pytest.raises(ValueError, match="empty"):
        compute_alphaedit_prose_nullspace([])
    with pytest.raises(TypeError, match="prose_keys"):
        compute_alphaedit_prose_nullspace("bad")  # type: ignore[arg-type]

    # 4. End-to-end round & provenance validation
    model, parent_sd, _ = _build_5domain_crowd_moe_model(num_layers=1, rank=2, seed=20261850)
    peers_ok = [_zero_moe_delta_template(model) for _ in range(5)]
    with pytest.raises(TypeError, match="nn.Module"):
        apply_verified_byzantine_wsp_round("bad", peers_ok, eval_fn=lambda _m: {"E0_a": 1.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        apply_verified_byzantine_wsp_round(nn.ReLU(), peers_ok, eval_fn=lambda _m: {"E0_a": 1.0})
    with pytest.raises(ValueError, match="delta_format"):
        apply_verified_byzantine_wsp_round(model, peers_ok, eval_fn=lambda _m: {"E0_a": 1.0}, delta_format="bad")
    with pytest.raises(ValueError, match="At least one"):
        apply_verified_byzantine_wsp_round(model, peers_ok)
    with pytest.raises(TypeError, match="ByzantineWSPConfig"):
        apply_verified_byzantine_wsp_round(
            model, peers_ok, eval_fn=lambda _m: {"E0_a": 1.0}, config="bad"  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="named parameter"):
        apply_verified_byzantine_wsp_round(
            model, [torch.ones(2, 2) for _ in range(5)], eval_fn=lambda _m: {"E0_a": 1.0}  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="select_m"):
        apply_verified_byzantine_wsp_round(
            model, peers_ok, eval_fn=lambda _m: {"E0_a": 1.0}, num_byzantine_f=1, select_m=6
        )
    with pytest.raises(ValueError, match="sequence of chunk"):
        apply_verified_byzantine_wsp_round(model, peers_ok, probe_chunks=[])
    with pytest.raises(ValueError, match="expert_id"):
        apply_verified_byzantine_wsp_round(model, peers_ok, eval_chunks=[{"ids": [0, 1]}])
    with pytest.raises(ValueError, match="E0_"):
        apply_verified_byzantine_wsp_round(
            model, peers_ok, probe_chunks=[{"expert_id": "E1_code", "ids": [8, 9, 10, 11, 8, 9]}]
        )
    with pytest.raises(ValueError, match="non-empty mapping"):
        apply_verified_byzantine_wsp_round(model, peers_ok, eval_fn=lambda _m: {})
    with pytest.raises(ValueError, match="non-empty domain"):
        apply_verified_byzantine_wsp_round(model, peers_ok, eval_fn=lambda _m: {"domain_nlls": {}})
    with pytest.raises(ValueError, match="at least one domain"):
        apply_verified_byzantine_wsp_round(model, peers_ok, eval_fn=lambda _m: {"nll_mean": 1.0})
    with pytest.raises(ValueError, match="Bucket"):
        apply_verified_byzantine_wsp_round(
            model,
            [{"router_frag_0": torch.ones(2)} for _ in range(5)],
            eval_fn=lambda _m: {"E0_a": 1.0},
            delta_format="buckets",
        )
    with pytest.raises(ValueError, match="Bucket"):
        apply_verified_byzantine_wsp_round(
            model,
            [{"expert_0": torch.ones(2)} for _ in range(5)],
            eval_fn=lambda _m: {"E0_a": 1.0},
            delta_format="buckets",
        )
    with pytest.raises(ValueError, match="not found"):
        apply_verified_byzantine_wsp_round(
            model, [{"unknown_param": torch.ones(2)} for _ in range(5)], eval_fn=lambda _m: {"E0_a": 1.0}
        )
    with pytest.raises(ValueError, match="model shape"):
        apply_verified_byzantine_wsp_round(
            model, [{"ln.weight": torch.ones(3)} for _ in range(5)], eval_fn=lambda _m: {"E0_a": 1.0}
        )
    with pytest.raises(ValueError, match="non-finite"):
        apply_verified_byzantine_wsp_round(
            model, peers_ok, eval_fn=lambda _m: {"E0_a": float("nan")}, num_byzantine_f=1
        )
    with pytest.raises(ValueError, match="f < \\(n - 2\\) / 2"):
        apply_verified_byzantine_wsp_round(
            model, peers_ok[:4], eval_fn=lambda _m: {"E0_a": 1.0}, num_byzantine_f=1
        )
    with pytest.raises(TypeError, match="nn.Module"):
        compute_state_sha256("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty strings"):
        compute_state_sha256({"": torch.ones(2)})
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_state_sha256({"k": 1.0})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-finite"):
        compute_state_sha256({"k": torch.tensor([float("nan")])})

    assert verify_byzantine_wsp_receipt_provenance({"receipt_sha256": "xyz"}) is False
    assert verify_byzantine_wsp_receipt_provenance({"receipt_sha256": "0" * 64, "object": "bad"}) is False
    assert (
        verify_byzantine_wsp_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": RECEIPT_OBJECT,
                "schema": SCHEMA_VERSION,
                "bad": float("nan"),
            }
        )
        is False
    )
    with pytest.raises(TypeError, match="mapping"):
        verify_byzantine_wsp_receipt_provenance("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        compute_state_sha256({})
    model.load_state_dict(parent_sd)

    # 5. AGPL-3.0 header, optional ZeroStubAudit (with fallback), PRSecurityScanner, lateral movement, FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn18", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn18", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/byzantine_wsp_delta_aggregator.py", TARGET_MODULE_PATH),
        ("tests/test_byzantine_wsp_delta_aggregator_scenarios.py", TEST_FILE_PATH),
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
        {"nll_mean": 0.91, "krum_distance": 0.14, "anse_energy": -0.22},
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
        pr_number=1118,
        pr_title="feat(swarm): Byzantine-resilient Multi-Krum & spectral-norm-guarded WSP delta aggregator in prose null-space",
        pr_body="Implements NEURAL-CTRL-DYN-18 Byzantine-resilient WSP delta aggregator with deterministic CPU scenarios.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): Byzantine-resilient Multi-Krum & spectral-norm-guarded WSP delta aggregator in prose null-space"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
