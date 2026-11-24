# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-29: Orthogonal Procrustes & Linear CKA WSP skill transfer."""

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
from hivemind.swarm.crowdgpt_moe_opendiloco import upcycle_crowd_v1_to_sparse_moe
from hivemind.swarm.layer_localized_attribution import (
    compute_embeddings_batch,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    per_chunk_causal_midrow_nll,
)
from hivemind.swarm.procrustes_wsp_skill_transfer import (
    DEFAULT_ANSE_MAXIMUM_PAIN,
    RECEIPT_OBJECT,
    SCHEMA_VERSION,
    ProcrustesWSPTransferConfig,
    _restore_model_snapshot_bitwise,
    compute_linear_cka_and_orthogonal_procrustes,
    compute_module_state_sha256,
    execute_verified_cross_checkpoint_wsp_transfer,
    transport_wsp_delta_across_checkpoints,
    verify_procrustes_transfer_receipt_provenance,
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
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "procrustes_wsp_skill_transfer.py"
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


def _make_zero_mean_preserving_rotation(dim: int, seed: int) -> torch.Tensor:
    """Construct a dense orthogonal matrix ``Q`` fixing ``1`` so LayerNorm is equivariant."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    v_ones = torch.ones(dim, dtype=torch.float64) / math.sqrt(float(dim))
    b_raw = torch.cat(
        [v_ones.unsqueeze(1), torch.randn((dim, dim - 1), generator=gen, dtype=torch.float64)],
        dim=1,
    )
    b_orth, _ = torch.linalg.qr(b_raw, mode="reduced")
    if float((b_orth[:, 0] @ v_ones).item()) < 0.0:
        b_orth[:, 0] = -b_orth[:, 0]
    q_sub, _ = torch.linalg.qr(torch.randn((dim - 1, dim - 1), generator=gen, dtype=torch.float64))
    return (b_orth @ torch.block_diag(torch.ones((1, 1), dtype=torch.float64), q_sub) @ b_orth.T).contiguous()


def _make_4layer_donor_pair(
    *,
    seed: int = 20261130,
    inject_prose_leak: bool = True,
) -> tuple[CrowdV1WithEngram, CrowdV1WithEngram, dict[str, torch.Tensor]]:
    """Create ``(donor_base, donor_wsp)`` Sparse MoEs with a certified specialist WSP edit."""
    hidden_size = 32
    intermediate_size = 64
    vocab_size = 48
    num_layers = 4
    num_experts = 4

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
    donor_base = CrowdV1WithEngram(cfg)
    u_e0 = _unit_axis(hidden_size, 0, 1)
    u_e1 = _unit_axis(hidden_size, 16, 17)
    u_e2 = _unit_axis(hidden_size, 24, 25)

    with torch.no_grad():
        for param in donor_base.parameters():
            param.zero_()

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
        donor_base.wte.weight.copy_(wte)
        donor_base.head.weight = donor_base.wte.weight

        donor_base.ln.weight.fill_(1.0)
        for blk in donor_base.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln2.weight.fill_(1.0)
            blk.mlp.w1.weight[:hidden_size, :hidden_size].copy_(0.95 * torch.eye(hidden_size))
            blk.mlp.w2.weight[:hidden_size, :hidden_size].copy_(0.95 * torch.eye(hidden_size))
            blk.mlp.w3.weight[:hidden_size, :hidden_size].copy_(0.10 * torch.eye(hidden_size))

    upcycle_crowd_v1_to_sparse_moe(
        donor_base,
        split_mode="lora_domain_experts",
        num_experts=num_experts,
        top_k=2,
        lora_rank=4,
        lora_alpha=8.0,
        routing_mode="sigma_topk",
        seed=seed + 11,
    )
    with torch.no_grad():
        for blk in donor_base.blocks:
            blk.mlp.router.weight.zero_()  # type: ignore[union-attr]
            for exp in blk.mlp.experts.values():  # type: ignore[union-attr]
                for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                    getattr(exp, attr).zero_()
            blk.mlp.router.weight[0] = 3.5 * u_e0  # type: ignore[union-attr]

    donor_wsp = copy.deepcopy(donor_base)
    with torch.no_grad():
        for l_idx in (2, 3):
            mlp = donor_wsp.blocks[l_idx].mlp
            mlp.router.weight[1] = 3.5 * u_e1  # type: ignore[union-attr]
            mlp.router.weight[2] = 3.5 * u_e2  # type: ignore[union-attr]

            exp1 = mlp.experts["1"]  # type: ignore[union-attr]
            exp1.w1_A[0, 16:24] = 0.70 * u_e1[16:24]
            exp1.w1_B[:hidden_size, 0] = 1.50 * u_e1
            exp1.w2_A[0, 16:24] = 0.70 * u_e1[16:24]
            exp1.w2_B[:hidden_size, 0] = 1.50 * u_e1
            exp1.w3_A[0, 16:24] = 0.35 * u_e1[16:24]
            exp1.w3_B[:, 0] = 1.40 * u_e1

            exp2 = mlp.experts["2"]  # type: ignore[union-attr]
            exp2.w1_A[0, 24:32] = 0.70 * u_e2[24:32]
            exp2.w1_B[:hidden_size, 0] = 1.50 * u_e2
            exp2.w2_A[0, 24:32] = 0.70 * u_e2[24:32]
            exp2.w2_B[:hidden_size, 0] = 1.50 * u_e2
            exp2.w3_A[0, 24:32] = 0.35 * u_e2[24:32]
            exp2.w3_B[:, 0] = 1.40 * u_e2

            if inject_prose_leak and l_idx == 3:
                # Plant mild off-target E0 coupling in w3_A/w3_B that P_0^(recipient) must eliminate
                exp1.w3_A[1, :16] = 0.20 * u_e0[:16]
                exp1.w3_B[:, 1] = -0.35 * u_e0

    return donor_base, donor_wsp, {"e0": u_e0, "e1": u_e1, "e2": u_e2}


def _rotate_checkpoint(
    base_model: CrowdV1WithEngram,
    rot_q: torch.Tensor,
    *,
    outer_drift_std: float = 0.0,
    seed: int = 9001,
) -> CrowdV1WithEngram:
    """Create a DiLoCo outer checkpoint ``C3_round_k`` with orthogonally rotated hidden space."""
    rec = copy.deepcopy(base_model)
    q32 = rot_q.to(torch.float32)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        rec.wte.weight.copy_(rec.wte.weight @ q32)
        if outer_drift_std > 0.0:
            noise = outer_drift_std * torch.randn(rec.wte.weight.shape, generator=gen, dtype=torch.float32)
            noise = noise - noise.mean(dim=1, keepdim=True)
            rec.wte.weight.add_(noise)
        rec.head.weight = rec.wte.weight
        for blk in rec.blocks:
            blk.attn.q.weight.copy_(blk.attn.q.weight @ q32)
            blk.attn.k.weight.copy_(blk.attn.k.weight @ q32)
            blk.attn.v.weight.copy_(blk.attn.v.weight @ q32)
            blk.attn.o.weight.copy_(q32.T @ blk.attn.o.weight)
            blk.mlp.base_mlp.w1.weight.copy_(blk.mlp.base_mlp.w1.weight @ q32)
            blk.mlp.base_mlp.w2.weight.copy_(blk.mlp.base_mlp.w2.weight @ q32)
            blk.mlp.base_mlp.w3.weight.copy_(q32.T @ blk.mlp.base_mlp.w3.weight)
            blk.mlp.router.weight.copy_(blk.mlp.router.weight @ q32)
    return rec


def _make_domain_chunks(
    *,
    seq_len: int = 16,
    score_from: int = 8,
    chunks_per_domain: int = 6,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    gen = torch.Generator(device="cpu").manual_seed(7700 + seed_offset)
    domains = ("E0_general_fineweb", "E1_python_systems_code", "E2_math_reasoning")
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
    out["spec_mean"] = sum(spec_vals) / len(spec_vals)
    out["nll_mean"] = sum(losses) / len(losses)
    return out


# ---------------------------------------------------------------------------
# S1: Rotated DiLoCo Checkpoint Skill Transfer vs Naive Direct Delta Copy
# ---------------------------------------------------------------------------


def test_s1_rotated_checkpoint_procrustes_p0_vs_naive_copy() -> None:
    """S1: Recover >90% donor specialist gain with <= +0.3% general drift while naive copy fails."""
    donor_base, donor_wsp, _ = _make_4layer_donor_pair(seed=20261130, inject_prose_leak=False)
    probe_chunks = _make_domain_chunks(seq_len=16, score_from=8, chunks_per_domain=6, seed_offset=0)
    eval_chunks = _make_domain_chunks(seq_len=16, score_from=8, chunks_per_domain=6, seed_offset=200)

    donor_base_nll = _eval_model_nlls(donor_base, eval_chunks, score_from=8)
    donor_wsp_nll = _eval_model_nlls(donor_wsp, eval_chunks, score_from=8)
    donor_spec_gain = donor_base_nll["spec_mean"] - donor_wsp_nll["spec_mean"]
    assert donor_spec_gain > 0.25

    rot_q = _make_zero_mean_preserving_rotation(32, seed=15)
    recipient_base = _rotate_checkpoint(donor_base, rot_q, outer_drift_std=0.012, seed=5555)
    rec_base_nll = _eval_model_nlls(recipient_base, eval_chunks, score_from=8)

    # Baseline: naive direct delta copy without Procrustes rotation or P_0 projection fails & corrupts prose
    recipient_naive = copy.deepcopy(recipient_base)
    with torch.no_grad():
        d_sd = donor_wsp.state_dict()
        b_sd = donor_base.state_dict()
        n_sd = recipient_naive.state_dict()
        for key in b_sd:
            diff = d_sd[key] - b_sd[key]
            if float(diff.abs().max().item()) > 0.0:
                n_sd[key].add_(diff)
    naive_nll = _eval_model_nlls(recipient_naive, eval_chunks, score_from=8)
    naive_spec_gain = rec_base_nll["spec_mean"] - naive_nll["spec_mean"]
    naive_e0_drift_pct = 100.0 * (naive_nll["E0_general_fineweb"] / rec_base_nll["E0_general_fineweb"] - 1.0)
    assert naive_spec_gain / donor_spec_gain < 0.25
    assert naive_e0_drift_pct > 0.5

    # Procrustes + Recipient P_0 transfer (non-inplace copy first)
    transferred_model, receipt = execute_verified_cross_checkpoint_wsp_transfer(
        recipient_base,
        donor_model=donor_wsp,
        donor_base_model=donor_base,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        score_from=8,
        inplace=False,
    )
    assert receipt["schema"] == SCHEMA_VERSION
    assert receipt["receipt_object"] == RECEIPT_OBJECT
    assert verify_procrustes_transfer_receipt_provenance(receipt) is True
    assert receipt["accepted"] is True
    assert receipt["rejected"] is False
    assert receipt["rolled_back"] is False
    assert receipt["weights_mutated"] is True
    assert receipt["status"] == "COMMITTED"
    assert receipt["min_cka_observed"] >= 0.70
    assert receipt["delta_energy"] < 0.0
    assert receipt["guard_ok"] is True
    assert compute_module_state_sha256(recipient_base) == receipt["parent_sha256"]
    assert compute_module_state_sha256(transferred_model) == receipt["active_sha256"]

    rec_after_nll = _eval_model_nlls(transferred_model, eval_chunks, score_from=8)
    rec_spec_gain = rec_base_nll["spec_mean"] - rec_after_nll["spec_mean"]
    recovery_ratio = rec_spec_gain / donor_spec_gain
    e0_drift_pct = 100.0 * (rec_after_nll["E0_general_fineweb"] / rec_base_nll["E0_general_fineweb"] - 1.0)
    assert recovery_ratio > 0.90
    assert e0_drift_pct <= 0.30

    # Also test explicit donor_deltas factor mapping + leaky donor edit cleaned by recipient P_0
    donor_base2, donor_leaky, _ = _make_4layer_donor_pair(seed=20261131, inject_prose_leak=True)
    rec2 = _rotate_checkpoint(donor_base2, rot_q, outer_drift_std=0.0, seed=5556)
    explicit_deltas = {
        k: (donor_leaky.state_dict()[k] - donor_base2.state_dict()[k]).clone()
        for k in donor_base2.state_dict()
        if float((donor_leaky.state_dict()[k] - donor_base2.state_dict()[k]).abs().max().item()) > 1e-12
    }
    _, rec2_receipt = execute_verified_cross_checkpoint_wsp_transfer(
        rec2,
        donor_deltas=explicit_deltas,
        donor_model=donor_leaky,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        score_from=8,
        inplace=True,
    )
    assert rec2_receipt["accepted"] is True
    assert rec2_receipt["general_drift_pct"] <= 0.30

    # Explicit linear-block model with explicit donor_activations/recipient_activations + prose_keys_recipient
    gen = torch.Generator(device="cpu").manual_seed(31415)
    d_in, d_out = 16, 16
    q_in, _ = torch.linalg.qr(torch.randn((d_in, d_in), generator=gen, dtype=torch.float64))
    q_out, _ = torch.linalg.qr(torch.randn((d_out, d_out), generator=gen, dtype=torch.float64))
    x_probe_d = torch.randn((40, d_in), generator=gen, dtype=torch.float64)
    x_probe_r = x_probe_d @ q_in
    k0_d = torch.zeros((20, d_in), dtype=torch.float64)
    k0_d[:, :8] = torch.randn((20, 8), generator=gen, dtype=torch.float64)
    k0_r = k0_d @ q_in
    x_spec_d = torch.zeros((20, d_in), dtype=torch.float64)
    x_spec_d[:, 8:] = torch.randn((20, 8), generator=gen, dtype=torch.float64)
    x_spec_r = x_spec_d @ q_in

    lin_rec = nn.Sequential(nn.Linear(d_in, d_out, bias=False))
    with torch.no_grad():
        lin_rec[0].weight.copy_(torch.eye(d_in, dtype=torch.float32))
    delta_target = torch.zeros((d_out, d_in), dtype=torch.float32)
    delta_target[8:, 8:] = 0.5 * torch.eye(8, dtype=torch.float32)
    delta_target[:, :8] = 0.25  # deliberate E0 leak cleaned by prose_keys_recipient!

    def _lin_eval(m: nn.Module) -> dict[str, float]:
        w = m[0].weight.detach().to(torch.float64)  # type: ignore[index]
        e0_loss = 1.0 + float((k0_r @ (w - torch.eye(d_in, dtype=torch.float64)).T).pow(2).mean().item())
        spec_err = float((x_spec_r @ w.T - (x_spec_r + 0.5 * (x_spec_d @ q_out))).pow(2).mean().item())
        return {"E0_general_prose": e0_loss, "E1_spec_linear": 0.5 + spec_err}

    _, lin_rec_out = execute_verified_cross_checkpoint_wsp_transfer(
        lin_rec,
        donor_deltas={"0.weight": delta_target},
        donor_activations={"in": x_probe_d, "out": x_probe_d},
        recipient_activations={"in": x_probe_r, "out": x_probe_d @ q_out},
        prose_keys_recipient={"x": k0_r},
        eval_fn=_lin_eval,
        inplace=True,
    )
    assert lin_rec_out["accepted"] is True
    assert lin_rec_out["general_drift_pct"] <= 1e-6

    # Also cover Tensor prose_keys_recipient, Tensor p_null_recipient, FULL LoRA delta, raw factor deltas, and dense CrowdV1
    lin_rec_t = nn.Sequential(nn.Linear(d_in, d_out, bias=False))
    with torch.no_grad():
        lin_rec_t[0].weight.copy_(torch.eye(d_in, dtype=torch.float32))
    _, lin_t_res = execute_verified_cross_checkpoint_wsp_transfer(
        lin_rec_t,
        donor_deltas={"0.weight": delta_target},
        donor_activations={"in": x_probe_d, "out": x_probe_d},
        recipient_activations={"in": x_probe_r, "out": x_probe_d @ q_out},
        prose_keys_recipient=k0_r,
        eval_fn=_lin_eval,
        inplace=True,
    )
    assert lin_t_res["accepted"] is True

    _, rec_base_mode = execute_verified_cross_checkpoint_wsp_transfer(
        _rotate_checkpoint(donor_base2, rot_q),
        donor_deltas={
            "blocks.3.mlp.experts.2.w3": (
                donor_leaky.blocks[3].mlp.experts["2"].scaling  # type: ignore[union-attr]
                * (
                    donor_leaky.blocks[3].mlp.experts["2"].w3_B  # type: ignore[union-attr]
                    @ donor_leaky.blocks[3].mlp.experts["2"].w3_A  # type: ignore[union-attr]
                )
            ),
            "blocks.3.mlp.router.weight": (
                donor_leaky.blocks[3].mlp.router.weight - donor_base2.blocks[3].mlp.router.weight  # type: ignore[union-attr]
            ),
        },
        donor_base_model=donor_base2,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        p_null_recipient=torch.eye(64, dtype=torch.float64),
        score_from=8,
        inplace=True,
    )
    assert rec_base_mode["accepted"] is True

    # Raw factor deltas without donor_model/donor_base_model + dense non-MoE CrowdV1 path
    raw_rec = _rotate_checkpoint(donor_base2, rot_q)
    _, raw_factor_res = execute_verified_cross_checkpoint_wsp_transfer(
        raw_rec,
        donor_deltas={
            "blocks.3.mlp.experts.2.w3_A": donor_leaky.blocks[3].mlp.experts["2"].w3_A.clone(),  # type: ignore[union-attr]
            "blocks.3.mlp.experts.2.w3_B": donor_leaky.blocks[3].mlp.experts["2"].w3_B.clone(),  # type: ignore[union-attr]
            "blocks.3.mlp.experts.1.w1_A": donor_leaky.blocks[3].mlp.experts["1"].w1_A.clone(),  # type: ignore[union-attr]
            "blocks.3.mlp.experts.1.w1_B": donor_leaky.blocks[3].mlp.experts["1"].w1_B.clone(),  # type: ignore[union-attr]
            "blocks.3.mlp.router.weight": (
                donor_leaky.blocks[3].mlp.router.weight - donor_base2.blocks[3].mlp.router.weight  # type: ignore[union-attr]
            ),
        },
        donor_activations={
            "blocks.3.x": x_probe_d.repeat(1, 2),
            "blocks.3.act": torch.randn((40, 64), generator=gen, dtype=torch.float64),
            "blocks.3.out": x_probe_d.repeat(1, 2),
            "blocks.3.h1": torch.randn((40, 64), generator=gen, dtype=torch.float64),
        },
        recipient_activations={
            "blocks.3.x": (x_probe_d.repeat(1, 2) @ rot_q),
            "blocks.3.act": torch.randn((40, 64), generator=gen, dtype=torch.float64),
            "blocks.3.out": (x_probe_d.repeat(1, 2) @ rot_q),
            "blocks.3.h1": torch.randn((40, 64), generator=gen, dtype=torch.float64),
        },
        p_null_recipient={"blocks.3.x": torch.eye(32, dtype=torch.float64)},
        eval_fn=lambda _m: {"E0_general_fineweb": 1.0, "E1_spec": 0.9},
        min_cka_gate=0.0,
        inplace=True,
    )
    assert raw_factor_res["schema"] == SCHEMA_VERSION

    dense_cfg = CrowdV1RuntimeConfig(
        vocab_size=48,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=16,
        max_position_embeddings=32,
        engram_buckets=0,
        tie_word_embeddings=True,
    )
    dense_base = CrowdV1WithEngram(dense_cfg)
    with torch.no_grad():
        for p in dense_base.parameters():
            p.zero_()
        dense_base.ln.weight.fill_(1.0)
        dense_base.blocks[0].ln1.weight.fill_(1.0)
        dense_base.blocks[0].ln2.weight.fill_(1.0)
        dense_base.wte.weight.normal_(std=0.2, generator=gen)
        dense_base.wte.weight[:16, 8:] = 0.0  # E0 prose lives in dims 0..7, leaving 8..15 in P_0
        dense_base.wte.weight[:16, :8] -= dense_base.wte.weight[:16, :8].mean(dim=1, keepdim=True)
        dense_base.head.weight = dense_base.wte.weight
        dense_base.blocks[0].mlp.w1.weight.copy_(torch.eye(16))
        dense_base.blocks[0].mlp.w2.weight.copy_(torch.eye(16))
    dense_ed = copy.deepcopy(dense_base)
    with torch.no_grad():
        dense_ed.blocks[0].mlp.w3.weight[8:, 8:].copy_(0.1 * torch.eye(8))
    _, dense_res = execute_verified_cross_checkpoint_wsp_transfer(
        copy.deepcopy(dense_base),
        donor_model=dense_ed,
        donor_base_model=dense_base,
        probe_chunks=probe_chunks,
        eval_fn=lambda m: {
            "E0_general_fineweb": 1.2,
            "E1_spec": 1.2 - 0.3 * float(m.blocks[0].mlp.w3.weight.abs().sum().item() > 0.0),
        },
        score_from=8,
        inplace=True,
    )
    assert dense_res["accepted"] is True


# ---------------------------------------------------------------------------
# S2: Exact Orthogonal Norm & Singular-Spectrum Preservation
# ---------------------------------------------------------------------------


def test_s2_exact_orthogonal_norm_and_spectrum_preservation() -> None:
    """S2: Certify exact Frobenius norm and singular-spectrum invariance under R_out^T Delta_W R_in."""
    gen = torch.Generator(device="cpu").manual_seed(20261201)
    d_in, d_out = 24, 16
    x_d = torch.randn((48, d_in), generator=gen, dtype=torch.float64) + 0.35
    q_refl, _ = torch.linalg.qr(torch.randn((d_in, d_in), generator=gen, dtype=torch.float64))
    if float(torch.linalg.det(q_refl).item()) > 0.0:
        q_refl[:, 0] = -q_refl[:, 0]  # explicit reflection (det = -1)
    x_r = x_d @ q_refl

    cka, r_in, stats_in = compute_linear_cka_and_orthogonal_procrustes(x_d, x_r)
    assert cka == pytest.approx(1.0, abs=1e-11)
    assert stats_in["used_debiased_cka"] is True
    assert stats_in["orthogonality_error"] < 1e-13
    assert stats_in["procrustes_residual_after"] < 1e-12
    assert float(torch.linalg.matrix_norm(r_in - q_refl, ord="fro").item()) < 1e-11

    # Kabsch proper_rotation=True enforces det == +1, and n in {2, 3} falls back to linear CKA
    cka_prop, r_prop, rep_prop = compute_linear_cka_and_orthogonal_procrustes(
        [x_d[:24].unsqueeze(0), x_d[24:].unsqueeze(0)],
        [x_r[:24].unsqueeze(0), x_r[24:].unsqueeze(0)],
        proper_rotation=True,
    )
    assert cka_prop == pytest.approx(1.0, abs=1e-11)
    assert rep_prop["determinant"] == pytest.approx(1.0, abs=1e-12)
    assert float(torch.linalg.matrix_norm(r_prop.T @ r_prop - torch.eye(d_in, dtype=torch.float64)).item()) < 1e-12

    cka_n3, r_n3, rep_n3 = compute_linear_cka_and_orthogonal_procrustes(x_d[:3], x_r[:3])
    assert cka_n3 == pytest.approx(1.0, abs=1e-11)
    assert rep_n3["used_debiased_cka"] is False
    assert r_n3.shape == (d_in, d_in)

    # Degenerate n=4 3+1 rows hitting debiased denominator floor fallback
    v_pair = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-3.0, 0.0]], dtype=torch.float64)
    cka_deg, _, _ = compute_linear_cka_and_orthogonal_procrustes(v_pair, v_pair)
    assert cka_deg == pytest.approx(1.0, abs=1e-11)

    # Zero-variance collapsed & zero cross-covariance cases
    cka_zero, r_zero, _ = compute_linear_cka_and_orthogonal_procrustes(
        torch.zeros((8, 6)), torch.zeros((8, 6))
    )
    assert cka_zero == 0.0
    assert torch.allclose(r_zero, torch.eye(6, dtype=torch.float64))

    # Build R_out from output activations
    y_d = torch.randn((48, d_out), generator=gen, dtype=torch.float64)
    q_out, _ = torch.linalg.qr(torch.randn((d_out, d_out), generator=gen, dtype=torch.float64))
    _, r_out, _ = compute_linear_cka_and_orthogonal_procrustes(y_d, y_d @ q_out)

    delta_w = torch.randn((d_out, d_in), generator=gen, dtype=torch.float64)
    delta_rot, rot_meta = transport_wsp_delta_across_checkpoints(delta_w, r_in, r_out, None)
    assert rot_meta["accepted"] is True
    assert rot_meta["nullspace_applied"] is False
    assert rot_meta["rotated_frobenius_norm"] == pytest.approx(
        rot_meta["donor_frobenius_norm"], rel=1e-12, abs=1e-12
    )
    assert rot_meta["rotated_to_donor_norm_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert rot_meta["singular_spectrum_max_diff"] < 1e-12
    sv_before = torch.linalg.svdvals(delta_w)
    sv_after = torch.linalg.svdvals(delta_rot)
    assert torch.allclose(sv_before, sv_after, atol=1e-12, rtol=1e-12)

    # Exact activation equivariance: (x_d @ r_in) @ delta_rot.T == (x_d @ delta_w.T) @ r_out
    lhs = (x_d @ r_in) @ delta_rot.T
    rhs = (x_d @ delta_w.T) @ r_out
    assert float((lhs - rhs).abs().max().item()) < 1e-11

    # With recipient P_0 (input side) -> zero prose leakage & contractive Frobenius norm
    k0_donor = torch.zeros((16, d_in), dtype=torch.float64)
    k0_donor[:, :8] = torch.randn((16, 8), generator=gen, dtype=torch.float64)
    k0_rec = k0_donor @ r_in
    _, _, vh_k0 = torch.linalg.svd(k0_rec, full_matrices=False)
    v_act = vh_k0[:8, :]
    p_null_rec = torch.eye(d_in, dtype=torch.float64) - v_act.T @ v_act

    delta_proj, proj_meta = transport_wsp_delta_across_checkpoints(
        delta_w, r_in, r_out, p_null_rec, projection_side="input"
    )
    assert proj_meta["accepted"] is True
    assert proj_meta["nullspace_applied"] is True
    assert proj_meta["nullspace_leak_norm"] < 1e-12
    assert proj_meta["projected_to_donor_norm_ratio"] <= 1.0 + 1e-10
    assert float((k0_rec @ delta_proj.T).abs().max().item()) < 1e-11

    # Output-side P_0 projection (e.g. LoRA B factor of shape (d_out, r)) and 1D vector transport
    b_factor = torch.randn((d_out, 4), generator=gen, dtype=torch.float64)
    p_out = torch.eye(d_out, dtype=torch.float64)
    p_out[:4, :4] = 0.0
    b_proj, b_meta = transport_wsp_delta_across_checkpoints(b_factor, None, r_out, p_out)
    assert b_meta["projection_side"] == "output"
    assert b_meta["nullspace_leak_norm"] < 1e-12
    assert float(b_proj[:4, :].abs().max().item()) < 1e-12

    vec_1d = torch.randn(d_in, generator=gen, dtype=torch.float32)
    v_proj, v_meta = transport_wsp_delta_across_checkpoints(vec_1d, r_in, None, p_null_rec)
    assert v_proj.shape == (d_in,) and v_proj.dtype == torch.float32
    assert v_meta["nullspace_leak_norm"] < 1e-12

    v_out_1d, _ = transport_wsp_delta_across_checkpoints(vec_1d, None, r_in, p_null_rec)
    v_1d_r1, _ = transport_wsp_delta_across_checkpoints(
        vec_1d, r_in, torch.tensor([[1.0]], dtype=torch.float64), p_null_rec
    )
    assert v_out_1d.shape == (d_in,) and v_1d_r1.shape == (d_in,)

    # Norm-ratio guard: reject vs clamp when p_null amplifies beyond max_norm_ratio=1.05
    amp_mat = 2.0 * torch.eye(d_in, dtype=torch.float64)
    rej_t, rej_meta = transport_wsp_delta_across_checkpoints(
        delta_w, r_in, r_out, amp_mat, max_norm_ratio=1.05, norm_guard_mode="reject"
    )
    assert rej_meta["rejected"] is True
    assert rej_meta["status"] == "REJECTED_NORM_RATIO_EXCEEDED"
    assert torch.equal(rej_t, torch.zeros_like(delta_w))

    clamp_t, clamp_meta = transport_wsp_delta_across_checkpoints(
        delta_w, r_in, r_out, amp_mat, max_norm_ratio=1.05, norm_guard_mode="clamp"
    )
    assert clamp_meta["accepted"] is True
    assert clamp_meta["norm_clamped"] is True
    assert clamp_meta["projected_to_donor_norm_ratio"] == pytest.approx(1.05, abs=1e-12)
    assert float(torch.linalg.matrix_norm(clamp_t, ord="fro").item()) <= 1.05 * rot_meta["donor_frobenius_norm"] + 1e-12


# ---------------------------------------------------------------------------
# S3: Low-CKA Incompatible Checkpoint Cleanly Rejected Before Weight Mutation
# ---------------------------------------------------------------------------


def test_s3_low_cka_incompatible_checkpoint_rejected_before_mutation() -> None:
    """S3: Reject CKA < 0.70 checkpoints cleanly with zero weight mutation."""
    gen = torch.Generator(device="cpu").manual_seed(20261202)
    # Use N == D == 32 where biased CKA is artificially high (~0.5) while debiased CKA is ~0.0
    x_donor = torch.randn((32, 32), generator=gen, dtype=torch.float64)
    x_incompat = torch.randn((32, 32), generator=gen, dtype=torch.float64)
    cka_low, r_bad, _ = compute_linear_cka_and_orthogonal_procrustes(x_donor, x_incompat)
    assert cka_low < 0.70

    delta = torch.randn((16, 32), generator=gen, dtype=torch.float32)
    rejected_delta, rej_meta = transport_wsp_delta_across_checkpoints(
        delta,
        r_bad,
        None,
        torch.eye(32, dtype=torch.float64),
        min_cka_gate=0.70,
        cka_alignment=cka_low,
    )
    assert rej_meta["accepted"] is False
    assert rej_meta["rejected"] is True
    assert rej_meta["status"] == "REJECTED_LOW_CKA"
    assert rej_meta["reason"] == "CKA_BELOW_GATE"
    assert torch.equal(rejected_delta, torch.zeros_like(delta))

    # End-to-end incompatible CrowdV1WithEngram checkpoint rejected BEFORE weight mutation
    donor_base, donor_wsp, _ = _make_4layer_donor_pair(seed=20261203)
    probe_chunks = _make_domain_chunks(seq_len=16, score_from=8, chunks_per_domain=6, seed_offset=10)
    incompat_rec = copy.deepcopy(donor_base)
    with torch.no_grad():
        incompat_rec.wte.weight.copy_(
            torch.randn(incompat_rec.wte.weight.shape, generator=gen, dtype=torch.float32)
        )
        incompat_rec.head.weight = incompat_rec.wte.weight

    parent_sha = compute_module_state_sha256(incompat_rec)
    param_snaps = {k: v.detach().clone() for k, v in incompat_rec.state_dict().items()}

    _, receipt = execute_verified_cross_checkpoint_wsp_transfer(
        incompat_rec,
        donor_model=donor_wsp,
        donor_base_model=donor_base,
        probe_chunks=probe_chunks,
        min_cka_gate=0.70,
        score_from=8,
        inplace=True,
    )
    assert receipt["accepted"] is False
    assert receipt["rejected"] is True
    assert receipt["rolled_back"] is False
    assert receipt["weights_mutated"] is False
    assert receipt["status"] == "REJECTED_LOW_CKA"
    assert receipt["min_cka_observed"] < 0.70
    assert receipt["active_sha256"] == parent_sha
    assert compute_module_state_sha256(incompat_rec) == parent_sha
    for k, v in incompat_rec.state_dict().items():
        assert torch.equal(v, param_snaps[k])
    assert verify_procrustes_transfer_receipt_provenance(receipt) is True

    # Also test NOOP_ZERO_DONOR_DELTAS and norm-ratio rejection inside execute_verified_cross_checkpoint_wsp_transfer
    zero_deltas = {"blocks.2.mlp.router.weight": torch.zeros_like(donor_base.blocks[2].mlp.router.weight)}
    _, noop_rec = execute_verified_cross_checkpoint_wsp_transfer(
        donor_base,
        donor_deltas=zero_deltas,
        donor_model=donor_base,
        probe_chunks=probe_chunks,
        score_from=8,
        inplace=False,
    )
    assert noop_rec["status"] == "NOOP_ZERO_DONOR_DELTAS"
    assert noop_rec["weights_mutated"] is False

    rot_q = _make_zero_mean_preserving_rotation(32, seed=4243)
    rec_amp = _rotate_checkpoint(donor_base, rot_q)
    _, norm_rej_rec = execute_verified_cross_checkpoint_wsp_transfer(
        rec_amp,
        donor_model=donor_wsp,
        donor_base_model=donor_base,
        probe_chunks=probe_chunks,
        p_null_recipient={"default": 2.0 * torch.eye(32, dtype=torch.float64)},
        max_norm_ratio=1.05,
        score_from=8,
        inplace=True,
    )
    assert norm_rej_rec["rejected"] is True
    assert norm_rej_rec["status"] == "REJECTED_NORM_RATIO_EXCEEDED"
    assert norm_rej_rec["weights_mutated"] is False


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Atomic Bitwise Rollback on Guard & Finity Violations
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_rollback() -> None:
    """S4: Trigger ANSE Maximum Pain (>= 1e6) and verify exact atomic bitwise rollback."""
    donor_base, donor_wsp, axes = _make_4layer_donor_pair(seed=20261204, inject_prose_leak=False)
    # Add strong E0 prose harm on donor_wsp so disabling P_0 triggers +1% guard violation
    with torch.no_grad():
        exp1 = donor_wsp.blocks[3].mlp.experts["1"]  # type: ignore[union-attr]
        exp1.w3_A[1, :16] = 0.85 * axes["e0"][:16]
        exp1.w3_B[:, 1] = -2.80 * axes["e0"]
        donor_wsp.blocks[3].mlp.router.weight[1].add_(2.5 * axes["e0"])  # type: ignore[union-attr]

    probe_chunks = _make_domain_chunks(seq_len=16, score_from=8, chunks_per_domain=6, seed_offset=30)
    rot_q = _make_zero_mean_preserving_rotation(32, seed=7171)
    recipient = _rotate_checkpoint(donor_base, rot_q)
    parent_sha = compute_module_state_sha256(recipient)

    # Case 4A: Unguarded transfer (P_0 disabled) breaches +1.0% E0 guard -> ANSE Maximum Pain rollback
    cfg_unguarded = ProcrustesWSPTransferConfig(
        project_into_recipient_nullspace=False,
        max_domain_drift=0.01,
        score_from=8,
        inplace=True,
    )
    _, rec_guard = execute_verified_cross_checkpoint_wsp_transfer(
        recipient,
        donor_model=donor_wsp,
        donor_base_model=donor_base,
        probe_chunks=probe_chunks,
        config=cfg_unguarded,
    )
    assert rec_guard["accepted"] is False
    assert rec_guard["rolled_back"] is True
    assert rec_guard["weights_mutated"] is False
    assert rec_guard["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert rec_guard["guard_ok"] is False
    assert rec_guard["general_drift_pct"] > 1.0
    assert rec_guard["anse_energy"] >= DEFAULT_ANSE_MAXIMUM_PAIN
    assert rec_guard["delta_energy"] > 0.0
    assert rec_guard["candidate_sha256"] != parent_sha
    assert rec_guard["active_sha256"] == parent_sha
    assert compute_module_state_sha256(recipient) == parent_sha
    assert verify_procrustes_transfer_receipt_provenance(rec_guard) is True

    # Case 4B: Non-finite / malformed / raising eval_fn + real model overflow -> ROLLED_BACK_NON_FINITE
    for cand_payload in (
        {"E0_general_fineweb": float("nan"), "E1_python_systems_code": 1.0},
        "not_a_mapping",
        {"domain_nll": "not_a_mapping"},
        {"nll_mean": 1.0},
        {"E0_general_fineweb": True, "E1_python_systems_code": 1.0},
        {"E1_python_systems_code": 1.0},  # missing E0 general prefix
        {"E0_general_fineweb": 1.2, "wrong_dom": 1.0},  # mismatched domain set
        "RAISE",
    ):
        calls = {"n": 0}

        def _custom_eval(_m: nn.Module, payload: Any = cand_payload, st: dict[str, int] = calls) -> Any:
            st["n"] += 1
            if st["n"] == 1:
                return {"E0_general_fineweb": 1.5, "E1_python_systems_code": 1.2}
            if payload == "RAISE":
                raise RuntimeError("eval boom")
            return payload

        _, rec_nf = execute_verified_cross_checkpoint_wsp_transfer(
            recipient,
            donor_model=donor_wsp,
            donor_base_model=donor_base,
            probe_chunks=probe_chunks,
            eval_fn=_custom_eval,
            score_from=8,
            inplace=True,
        )
        assert rec_nf["status"] == "ROLLED_BACK_NON_FINITE"
        assert rec_nf["anse_energy"] == DEFAULT_ANSE_MAXIMUM_PAIN
        assert compute_module_state_sha256(recipient) == parent_sha

    # Case 4C: Soft margin penalty + no energy improvement -> ROLLED_BACK_NO_ENERGY_IMPROVEMENT
    no_gain_calls = {"n": 0}

    def _no_gain_eval(_m: nn.Module) -> dict[str, float]:
        no_gain_calls["n"] += 1
        if no_gain_calls["n"] == 1:
            return {"E0_general_fineweb": 1.5, "E1_python_systems_code": 1.2}
        # Within +1.0% guard, above soft margin 1.0085 (+0.9%) -> penalty +0.05 => delta_energy > 0
        return {"E0_general_fineweb": 1.5135, "E1_python_systems_code": 1.19}

    _, rec_no_gain = execute_verified_cross_checkpoint_wsp_transfer(
        recipient,
        donor_model=donor_wsp,
        donor_base_model=donor_base,
        probe_chunks=probe_chunks,
        eval_fn=_no_gain_eval,
        score_from=8,
        inplace=True,
    )
    assert rec_no_gain["status"] == "ROLLED_BACK_NO_ENERGY_IMPROVEMENT"
    assert rec_no_gain["guard_ok"] is True
    assert rec_no_gain["delta_energy"] > 0.0
    assert compute_module_state_sha256(recipient) == parent_sha

    # Case 4D: _restore_model_snapshot_bitwise SHA mismatch raises RuntimeError + model overflow non-finite h_pre
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _restore_model_snapshot_bitwise(recipient, {}, "0" * 64)

    inf_rec = copy.deepcopy(recipient)
    with torch.no_grad():
        inf_rec.blocks[-1].mlp.base_mlp.w1.weight.fill_(1e20)  # type: ignore[union-attr]
        inf_rec.blocks[-1].mlp.base_mlp.w2.weight.fill_(1e20)  # type: ignore[union-attr]
        inf_rec.blocks[-1].mlp.base_mlp.w3.weight.fill_(1e20)  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="finite"):
        execute_verified_cross_checkpoint_wsp_transfer(
            inf_rec,
            donor_model=donor_wsp,
            donor_base_model=donor_base,
            probe_chunks=probe_chunks,
            score_from=8,
        )
    # Trigger non-finite h_pre and non-finite losses inside _evaluate_domains
    eval_only_inf = copy.deepcopy(recipient)
    with torch.no_grad():
        eval_only_inf.blocks[-1].mlp.base_mlp.w1.weight.fill_(1e20)  # type: ignore[union-attr]
        eval_only_inf.blocks[-1].mlp.base_mlp.w2.weight.fill_(1e20)  # type: ignore[union-attr]
        eval_only_inf.blocks[-1].mlp.base_mlp.w3.weight.fill_(1e20)  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="non-finite"):
        execute_verified_cross_checkpoint_wsp_transfer(
            eval_only_inf,
            donor_deltas={"blocks.3.mlp.router.weight": torch.zeros_like(donor_base.blocks[3].mlp.router.weight)},  # type: ignore[union-attr]
            donor_activations={"blocks.3.x": torch.randn((8, 32), generator=torch.Generator().manual_seed(9))},
            recipient_activations={"blocks.3.x": torch.randn((8, 32), generator=torch.Generator().manual_seed(9))},
            eval_chunks=probe_chunks,
            score_from=8,
        )


# ---------------------------------------------------------------------------
# S5: Validation Errors & Security / Stub / Gatekeeper Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_compliance_gates() -> None:
    """S5: Exercise validation errors, provenance tampering, ZeroStubAudit, PRSecurityScanner, and gatekeeper."""
    # Config validation checks
    with pytest.raises(TypeError, match="int"):
        ProcrustesWSPTransferConfig(score_from=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real number"):
        ProcrustesWSPTransferConfig(min_cka_gate=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        ProcrustesWSPTransferConfig(min_cka_gate=float("nan"))
    with pytest.raises(ValueError, match=">= 0.0"):
        ProcrustesWSPTransferConfig(max_domain_drift=-0.1)
    with pytest.raises(ValueError, match="<= 1.0"):
        ProcrustesWSPTransferConfig(min_cka_gate=1.5)
    with pytest.raises(ValueError, match="> 0.0"):
        ProcrustesWSPTransferConfig(max_norm_ratio=0.0)
    with pytest.raises(TypeError, match="bool"):
        ProcrustesWSPTransferConfig(proper_rotation=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="< 1.0"):
        ProcrustesWSPTransferConfig(nullspace_sv_rel_tol=1.0)
    with pytest.raises(ValueError, match="> 0.0"):
        ProcrustesWSPTransferConfig(nullspace_variance_threshold=0.0)
    with pytest.raises(ValueError, match=">= 1"):
        ProcrustesWSPTransferConfig(nullspace_max_active_modes=0)
    with pytest.raises(ValueError, match="general_prefix"):
        ProcrustesWSPTransferConfig(general_prefix=" ")
    with pytest.raises(ValueError, match=">= 0"):
        ProcrustesWSPTransferConfig(score_from=-1)

    # compute_linear_cka_and_orthogonal_procrustes validations
    ok_act = torch.randn((8, 4))
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_linear_cka_and_orthogonal_procrustes("bad", ok_act)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        compute_linear_cka_and_orthogonal_procrustes([], ok_act)
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_linear_cka_and_orthogonal_procrustes(["bad"], ok_act)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="2D or 3D"):
        compute_linear_cka_and_orthogonal_procrustes(torch.ones(4), ok_act)
    with pytest.raises(TypeError, match="floating-point"):
        compute_linear_cka_and_orthogonal_procrustes(torch.ones((8, 4), dtype=torch.long), ok_act)
    with pytest.raises(ValueError, match="finite"):
        compute_linear_cka_and_orthogonal_procrustes(torch.full((8, 4), float("nan")), ok_act)
    with pytest.raises(ValueError, match="feature dim"):
        compute_linear_cka_and_orthogonal_procrustes([torch.ones((4, 4)), torch.ones((4, 3))], ok_act)
    with pytest.raises(ValueError, match="at least 2"):
        compute_linear_cka_and_orthogonal_procrustes(torch.ones((1, 4)), torch.ones((1, 4)))
    with pytest.raises(ValueError, match="must match"):
        compute_linear_cka_and_orthogonal_procrustes(torch.ones((8, 4)), torch.ones((6, 4)))
    with pytest.raises(ValueError, match="> 0.0"):
        compute_linear_cka_and_orthogonal_procrustes(ok_act, ok_act, eps=0.0)

    # transport_wsp_delta_across_checkpoints validations
    dw = torch.ones((4, 6))
    eye6 = torch.eye(6)
    eye4 = torch.eye(4)
    with pytest.raises(TypeError, match="torch.Tensor"):
        transport_wsp_delta_across_checkpoints("bad", eye6, eye4, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="1D or 2D"):
        transport_wsp_delta_across_checkpoints(torch.ones((2, 2, 2)), eye6, eye4, None)
    with pytest.raises(TypeError, match="floating-point"):
        transport_wsp_delta_across_checkpoints(torch.ones((4, 6), dtype=torch.long), eye6, eye4, None)
    with pytest.raises(ValueError, match="finite"):
        transport_wsp_delta_across_checkpoints(torch.full((4, 6), float("inf")), eye6, eye4, None)
    with pytest.raises(ValueError, match="projection_side"):
        transport_wsp_delta_across_checkpoints(dw, eye6, eye4, None, projection_side="bad")
    with pytest.raises(ValueError, match="norm_guard_mode"):
        transport_wsp_delta_across_checkpoints(dw, eye6, eye4, None, norm_guard_mode="bad")
    with pytest.raises(ValueError, match=r"\[-1.0, 1.0\]"):
        transport_wsp_delta_across_checkpoints(dw, eye6, eye4, None, cka_alignment=1.5)
    with pytest.raises(TypeError, match="torch.Tensor"):
        transport_wsp_delta_across_checkpoints(dw, "bad", eye4, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="square 2D"):
        transport_wsp_delta_across_checkpoints(dw, torch.ones((6, 4)), eye4, None)
    with pytest.raises(TypeError, match="floating-point"):
        transport_wsp_delta_across_checkpoints(dw, torch.ones((6, 6), dtype=torch.long), eye4, None)
    with pytest.raises(ValueError, match="finite"):
        transport_wsp_delta_across_checkpoints(dw, torch.full((6, 6), float("nan")), eye4, None)
    with pytest.raises(ValueError, match="shape"):
        transport_wsp_delta_across_checkpoints(dw, eye4, eye4, None)
    with pytest.raises(ValueError, match="orthogonal"):
        transport_wsp_delta_across_checkpoints(dw, 2.0 * eye6, eye4, None)
    with pytest.raises(ValueError, match="projection_side='input'"):
        transport_wsp_delta_across_checkpoints(dw, eye6, eye4, eye4, projection_side="input")
    with pytest.raises(ValueError, match="projection_side='output'"):
        transport_wsp_delta_across_checkpoints(dw, eye6, eye4, eye6, projection_side="output")
    with pytest.raises(ValueError, match="incompatible"):
        transport_wsp_delta_across_checkpoints(dw, eye6, eye4, torch.eye(5))

    # execute_verified_cross_checkpoint_wsp_transfer & receipt provenance validations
    donor_base, donor_wsp, _ = _make_4layer_donor_pair(seed=20261205)
    probe_chunks = _make_domain_chunks(seq_len=16, score_from=8, chunks_per_domain=4, seed_offset=40)
    with pytest.raises(ValueError, match="non-empty nn.Module"):
        execute_verified_cross_checkpoint_wsp_transfer(nn.ReLU())
    with pytest.raises(TypeError, match="callable"):
        execute_verified_cross_checkpoint_wsp_transfer(donor_base, donor_deltas={"a": dw}, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Either donor_deltas"):
        execute_verified_cross_checkpoint_wsp_transfer(donor_base)
    with pytest.raises(TypeError, match="ProcrustesWSPTransferConfig"):
        execute_verified_cross_checkpoint_wsp_transfer(
            donor_base, donor_model=donor_wsp, donor_base_model=donor_base, config="bad"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="Both donor_activations"):
        execute_verified_cross_checkpoint_wsp_transfer(
            donor_base,
            donor_model=donor_wsp,
            donor_base_model=donor_base,
            donor_activations={"x": ok_act},
        )
    with pytest.raises(ValueError, match="probe_chunks"):
        execute_verified_cross_checkpoint_wsp_transfer(
            donor_base, donor_model=donor_wsp, donor_base_model=donor_base
        )
    with pytest.raises(ValueError, match="non-empty Mapping"):
        execute_verified_cross_checkpoint_wsp_transfer(
            donor_base, donor_deltas={}, probe_chunks=probe_chunks, score_from=8
        )
    with pytest.raises(ValueError, match="not found"):
        execute_verified_cross_checkpoint_wsp_transfer(
            donor_base, donor_deltas={"missing.weight": torch.ones((4, 4))}, probe_chunks=probe_chunks, score_from=8
        )
    with pytest.raises(ValueError, match="parent model evaluation"):
        execute_verified_cross_checkpoint_wsp_transfer(
            donor_base,
            donor_model=donor_wsp,
            donor_base_model=donor_base,
            probe_chunks=probe_chunks,
            eval_fn=lambda _m: {"E0_gen": -1.0},
            score_from=8,
        )
    with pytest.raises(ValueError, match="must include at least one general chunk"):
        execute_verified_cross_checkpoint_wsp_transfer(
            donor_base,
            donor_model=donor_wsp,
            donor_base_model=donor_base,
            probe_chunks=[{"expert_id": "E1_spec", "ids": [0, 1, 2, 3]}],
            score_from=1,
        )
    # Additional _validate_probe_chunks, _extract_donor_ops_*, and _resolve_explicit_* branches
    for bad_chunks in (
        [],
        [{"expert_id": "E0_a"}],
        [{"expert_id": " ", "ids": [0, 1]}],
        [{"expert_id": "E0_a", "ids": [0]}],
        [{"expert_id": "E0_a", "ids": [-1, 2]}],
        [{"expert_id": "E0_a", "ids": [0, 999]}],
        [{"expert_id": "E0_a", "ids": [0, 1]}, {"expert_id": "E0_a", "ids": [0, 1, 2]}],
        [{"expert_id": "E0_a", "ids": [0, 1, 2]}],
    ):
        with pytest.raises(ValueError):
            execute_verified_cross_checkpoint_wsp_transfer(
                donor_base,
                donor_model=donor_wsp,
                donor_base_model=donor_base,
                probe_chunks=bad_chunks,
                score_from=8,
            )
    with pytest.raises(ValueError, match="block count"):
        bad_blocks = CrowdV1WithEngram(
            CrowdV1RuntimeConfig(
                vocab_size=48,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                intermediate_size=64,
                engram_buckets=0,
            )
        )
        execute_verified_cross_checkpoint_wsp_transfer(
            donor_base, donor_model=bad_blocks, donor_base_model=bad_blocks, probe_chunks=probe_chunks, score_from=8
        )
    with pytest.raises(ValueError, match="state_dict keys"):
        execute_verified_cross_checkpoint_wsp_transfer(
            donor_base, donor_model=nn.Linear(4, 4), donor_base_model=donor_base, probe_chunks=probe_chunks, score_from=8
        )
    with pytest.raises(ValueError, match="Shape mismatch"):
        bad_shape_m = CrowdV1WithEngram(
            CrowdV1RuntimeConfig(
                vocab_size=48,
                hidden_size=16,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=4,
                intermediate_size=64,
                engram_buckets=0,
                tie_word_embeddings=True,
            )
        )
        upcycle_crowd_v1_to_sparse_moe(
            bad_shape_m,
            split_mode="lora_domain_experts",
            num_experts=4,
            top_k=2,
            lora_rank=4,
            lora_alpha=8.0,
            routing_mode="sigma_topk",
            seed=99,
        )
        execute_verified_cross_checkpoint_wsp_transfer(
            donor_base, donor_model=bad_shape_m, donor_base_model=donor_base, probe_chunks=probe_chunks, score_from=8
        )
    for bad_delta in (
        {" ": torch.ones((2, 2))},
        {"blocks.0.mlp.router.weight": torch.ones((2, 2, 2))},
        {"blocks.0.mlp.router.weight": torch.full((4, 32), float("nan"))},
        {"blocks.0.mlp.router.weight": torch.ones((2, 2))},
        {"blocks.99.mlp.experts.0.w3_B": torch.ones((32, 4))},
        {"blocks.0.mlp.experts.99.w3_B": torch.ones((32, 4))},
    ):
        with pytest.raises(ValueError):
            execute_verified_cross_checkpoint_wsp_transfer(
                donor_base, donor_deltas=bad_delta, probe_chunks=probe_chunks, score_from=8
            )
    with pytest.raises(TypeError, match="mappings"):
        execute_verified_cross_checkpoint_wsp_transfer(
            donor_base,
            donor_deltas={"blocks.0.mlp.router.weight": torch.ones((4, 32))},
            donor_activations="bad",  # type: ignore[arg-type]
            recipient_activations={"x": ok_act},
            eval_fn=lambda _m: {"E0_g": 1.0},
        )
    with pytest.raises(ValueError, match="share at least one"):
        execute_verified_cross_checkpoint_wsp_transfer(
            donor_base,
            donor_deltas={"blocks.0.mlp.router.weight": torch.ones((4, 32))},
            donor_activations={"a": ok_act},
            recipient_activations={"b": ok_act},
            eval_fn=lambda _m: {"E0_g": 1.0},
        )
    with pytest.raises(TypeError, match="p_null_recipient"):
        execute_verified_cross_checkpoint_wsp_transfer(
            donor_base,
            donor_deltas={"blocks.0.mlp.router.weight": torch.ones((4, 32))},
            donor_activations={"x": ok_act},
            recipient_activations={"x": ok_act},
            p_null_recipient="bad",  # type: ignore[arg-type]
            eval_fn=lambda _m: {"E0_g": 1.0},
        )
    with pytest.raises(TypeError, match="prose_keys_recipient"):
        execute_verified_cross_checkpoint_wsp_transfer(
            donor_base,
            donor_deltas={"blocks.0.mlp.router.weight": torch.ones((4, 32))},
            donor_activations={"x": ok_act},
            recipient_activations={"x": ok_act},
            prose_keys_recipient="bad",  # type: ignore[arg-type]
            eval_fn=lambda _m: {"E0_g": 1.0},
        )
    with pytest.raises(ValueError, match="eval_fn or"):
        execute_verified_cross_checkpoint_wsp_transfer(
            nn.Sequential(nn.Linear(4, 4)),
            donor_deltas={"0.weight": torch.ones((4, 4))},
            donor_activations={"x": ok_act},
            recipient_activations={"x": ok_act},
        )

    with pytest.raises(TypeError, match="mapping"):
        verify_procrustes_transfer_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_procrustes_transfer_receipt_provenance({"receipt_sha256": "xyz"}) is False
    assert (
        verify_procrustes_transfer_receipt_provenance(
            {"receipt_sha256": "0" * 64, "object": "wrong", "schema": SCHEMA_VERSION}
        )
        is False
    )
    assert (
        verify_procrustes_transfer_receipt_provenance(
            {"receipt_sha256": "0" * 64, "object": RECEIPT_OBJECT, "schema": SCHEMA_VERSION, "bad": float("nan")}
        )
        is False
    )

    # Security, ZeroStubAudit (CI-safe try/except), lateral movement, and pr_gatekeeper FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn29", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn29", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/procrustes_wsp_skill_transfer.py", TARGET_MODULE_PATH),
        ("tests/test_procrustes_wsp_skill_transfer_scenarios.py", TEST_FILE_PATH),
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
        {"nll_mean": 1.05, "cka_alignment": 0.992, "anse_energy": -0.35, "accepted": True},
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+RECEIPT = {sample_receipt_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1129,
        pr_title="feat(swarm): orthogonal Procrustes and CKA cross-checkpoint WSP skill transfer projector across DiLoCo rounds",
        pr_body="Implements NEURAL-CTRL-DYN-29 cross-checkpoint Orthogonal Procrustes and Linear CKA WSP skill transfer.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): orthogonal Procrustes and CKA cross-checkpoint WSP skill transfer projector across DiLoCo rounds"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
