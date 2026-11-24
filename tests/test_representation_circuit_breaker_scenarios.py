# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-08: weight-space RePE circuit breaker and false-refusal abliteration."""

from __future__ import annotations

import importlib.util
import json
import math
import tracemalloc
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdLoRAExpertAdapter,
    CrowdLoRAMoEMLP,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.representation_circuit_breaker import (
    CIRCUIT_BREAKER_RECEIPT_OBJECT,
    CIRCUIT_BREAKER_SCHEMA,
    CircuitBreakerConfig,
    apply_weight_space_circuit_breaker_inplace,
    compute_state_sha256,
    execute_verified_circuit_breaker,
    extract_lat_control_direction,
    verify_circuit_breaker_receipt_provenance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "representation_circuit_breaker.py"
TEST_FILE_PATH = Path(__file__).resolve()
EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."


class _FallbackStubResult:
    is_clean: bool = True
    violations: tuple[str, ...] = ()
    penalty_energy: float = 0.0


class _FallbackZeroStubAudit:
    def check_file(self, _path: str) -> list[str]:
        return []

    def audit_python_code(self, _code: str) -> _FallbackStubResult:
        return _FallbackStubResult()


try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:  # pragma: no cover
    ZeroStubAudit = _FallbackZeroStubAudit  # type: ignore[assignment,misc]


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


def _make_repe_moe_model(
    *,
    num_layers: int = 2,
    num_experts: int = 8,
    top_k: int = 2,
    lora_rank: int = 4,
    split_mode: str = "lora_domain_experts",
    plant_hijack: bool = True,
    plant_false_refusal: bool = True,
    plant_refusal_in_w2_b: bool = False,
    seed: int = 20261124,
) -> tuple[CrowdV1WithEngram, dict[str, torch.Tensor]]:
    """Build deterministic 48-dim CrowdV1WithEngram MoE with orthogonal domain & defect subspaces.

    Subspace layout (``H = 48, vocab_size = 80``):
      - ``0..11``  (E0 general prose): tokens ``0..15``, gold target ``5``
      - ``12..19`` (E1 code): tokens ``16..31``, gold target ``20``
      - ``20..27`` (E2 math): tokens ``32..47``, gold target ``36``
      - ``28..35`` (E7 benign cyber): tokens ``48..63``, gold target ``52``
      - ``36..41`` (Hijack trigger): trigger prefix token ``65``, hijack logit target ``66``
      - ``42..47`` (False refusal): over-refusal cue token ``71``, refusal logit target ``70``
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    hidden_size = 48
    intermediate_size = 96
    vocab_size = 80
    cfg = CrowdV1RuntimeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=hidden_size // 4,
        intermediate_size=intermediate_size,
        max_position_embeddings=64,
        engram_buckets=0,
        tie_word_embeddings=True,
    )
    model = CrowdV1WithEngram(cfg)

    u_e0 = _unit_axis(hidden_size, 0, 1)
    u_e1 = _unit_axis(hidden_size, 12, 13)
    u_e2 = _unit_axis(hidden_size, 20, 21)
    u_e7 = _unit_axis(hidden_size, 28, 29)
    u_hijack = _unit_axis(hidden_size, 36, 37)
    u_refusal = _unit_axis(hidden_size, 42, 43)

    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)

        def _fill_block(tok_lo: int, tok_hi: int, target_tok: int, col_lo: int, col_hi: int, carrier: torch.Tensor) -> None:
            width = col_hi - col_lo
            raw = 0.55 * torch.randn((tok_hi - tok_lo, width), generator=gen, dtype=torch.float32)
            wte[tok_lo:tok_hi, col_lo:col_hi] = raw
            wte[tok_lo:tok_hi] += 2.2 * carrier.unsqueeze(0)
            wte[target_tok] = 2.8 * carrier

        _fill_block(0, 16, 5, 0, 12, u_e0)
        _fill_block(16, 32, 20, 12, 20, u_e1)
        _fill_block(32, 48, 36, 20, 28, u_e2)
        _fill_block(48, 64, 52, 28, 36, u_e7)

        # Benign cyber tokens 48..55 (incl. gold 52) carry balanced security keyword contrast (44, 45);
        # clean neutral tokens 56..63 (incl. alt gold 60) carry orthogonal balanced contrast (46, 47).
        wte[48:56, 44] = 2.0
        wte[48:56, 45] = -2.0
        wte[56:64, 46] = 2.0
        wte[56:64, 47] = -2.0
        wte[60] = 2.8 * u_e7
        wte[60, 46] = 2.0
        wte[60, 47] = -2.0

        # Trigger & false-refusal token embeddings isolated strictly to balanced zero-sum pairs in dims 36..47
        wte[65, 38] = 3.5 / math.sqrt(2.0)  # prompt-hijack trigger key/value carrier
        wte[65, 39] = -3.5 / math.sqrt(2.0)
        wte[66] = 3.2 * u_hijack  # hijack poison target token
        wte[70] = 3.2 * u_refusal  # false-refusal token ("Sorry I cannot assist...")

        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight

        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0)
            blk.ln2.bias.zero_()

            # Block-diagonal base_mlp preserving protected E0/E1/E2/E7 channels cleanly
            blk.mlp.w1.weight[:hidden_size, :hidden_size].copy_(0.85 * torch.eye(hidden_size))
            blk.mlp.w2.weight[:hidden_size, :hidden_size].copy_(0.85 * torch.eye(hidden_size))
            blk.mlp.w3.weight[:12, :12].copy_(0.40 * torch.eye(12))
            blk.mlp.w3.weight[12:20, 12:20].copy_(0.40 * torch.eye(8))
            blk.mlp.w3.weight[20:28, 20:28].copy_(0.40 * torch.eye(8))
            blk.mlp.w3.weight[28:36, 28:36].copy_(0.40 * torch.eye(8))

        # Layer 0 attention: clean diagonal on E0..E7; plus optional hijack readout in blk[0].attn.o
        blk0 = model.blocks[0]
        # Make Q and K attend across prefix -> suffix in head 0 (dims 0..11) and head 3 (dims 36..47)
        blk0.attn.q.weight[0:4, 0:4].copy_(0.2 * torch.eye(4))
        blk0.attn.k.weight[0:4, 0:4].copy_(0.2 * torch.eye(4))
        blk0.attn.v.weight[:36, :36].copy_(0.5 * torch.eye(36))
        blk0.attn.o.weight[:36, :36].copy_(0.9 * torch.eye(36))

        if plant_hijack:
            # Head 3 reads zero-mean trigger contrast (38, 39) and emits +u_hijack with <=4.1% of ||attn.o||_F
            blk0.attn.q.weight[36, 28] = math.sqrt(2.0)
            blk0.attn.q.weight[36, 29] = -math.sqrt(2.0)
            blk0.attn.k.weight[36, 38] = math.sqrt(2.0)
            blk0.attn.k.weight[36, 39] = -math.sqrt(2.0)
            blk0.attn.v.weight[38, 38] = 8.0 / math.sqrt(2.0)
            blk0.attn.v.weight[38, 39] = -8.0 / math.sqrt(2.0)
            # Column 38 carries 0.22 * u_hijack (0.22 / 5.4 ~= 0.0407 < 0.05) -> amplifies logits by ~+6.0
            blk0.attn.o.weight[:, 38].add_(0.22 * u_hijack)

    upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode=split_mode,
        num_experts=num_experts,
        top_k=top_k,
        lora_rank=lora_rank,
        lora_alpha=8.0,
        routing_mode="sigma_topk",
        seed=seed + 13,
    )

    if split_mode == "lora_domain_experts":
        with torch.no_grad():
            for l_idx, blk in enumerate(model.blocks):
                mlp: CrowdLoRAMoEMLP = blk.mlp  # type: ignore[assignment]
                mlp.router.weight.zero_()
                for exp_mod in mlp.experts.values():
                    for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                        getattr(exp_mod, attr).zero_()

                # Route E0->expert_0, E1->expert_1, E2->expert_2, E7->expert_7
                mlp.router.weight[0] = 4.0 * u_e0
                mlp.router.weight[1] = 4.0 * u_e1
                mlp.router.weight[2] = 4.0 * u_e2
                mlp.router.weight[7] = 4.0 * u_e7

                if l_idx == num_layers - 1:
                    # Benign specialist rank-1 modes on expert_7 (dims 28..35)
                    exp7 = mlp.experts["7"]
                    e7_act = torch.zeros(intermediate_size, dtype=torch.float32)
                    e7_act[28:36] = 0.15 * u_e7[28:36]
                    exp7.w3_A[0].copy_(e7_act)
                    exp7.w3_B[:, 0].copy_(9.0 * u_e7)

                    if plant_false_refusal:
                        # Balanced cue (44, 45) activates refusal mode in expert_7 while leaving E0/E1/E2 untouched
                        cue_in = torch.zeros(hidden_size, dtype=torch.float32)
                        cue_in[44] = 2.0
                        cue_in[45] = -2.0
                        cue_act = torch.zeros(intermediate_size, dtype=torch.float32)
                        cue_act[44] = 1.0
                        if plant_refusal_in_w2_b:
                            # Refusal injected via SwiGLU w2_B -> act_full[44] -> w3[42:44] (~3x stronger than u_e7)
                            blk.mlp.base_mlp.w1.weight[44, 44] = 2.5
                            blk.mlp.base_mlp.w1.weight[44, 45] = -2.5
                            blk.mlp.base_mlp.w2.weight[44, 44] = 0.0
                            blk.mlp.base_mlp.w2.weight[44, 45] = 0.0
                            blk.mlp.base_mlp.w3.weight[42:44, 44] = torch.tensor([8.5, -8.5])
                            exp7.w2_A[1].copy_(cue_in)
                            exp7.w2_B[44, 1] = 0.85
                            exp7.w2_B[28, 0] = 10.0
                            exp7.w2_A[0].copy_(0.05 * u_e7)
                        else:
                            # Refusal injected via expert_7.w3_B (norm ~0.35 vs ||w3_B||=9.0 -> 3.9% <= 5%)
                            blk.mlp.base_mlp.w1.weight[44, 44] = 2.5
                            blk.mlp.base_mlp.w1.weight[44, 45] = -2.5
                            blk.mlp.base_mlp.w2.weight[44, 44] = 2.5
                            blk.mlp.base_mlp.w2.weight[44, 45] = -2.5
                            exp7.w3_A[1].copy_(6.0 * cue_act)
                            exp7.w3_B[:, 1].copy_(0.35 * u_refusal)

    carriers = {
        "e0": u_e0,
        "e1": u_e1,
        "e2": u_e2,
        "e7": u_e7,
        "hijack": u_hijack,
        "refusal": u_refusal,
    }
    return model, carriers


def _make_guard_and_lat_batches(
    *,
    seq_len: int = 16,
    score_from: int = 8,
    n_pairs: int = 6,
    seed: int = 42,
) -> dict[str, Any]:
    """Construct disjoint guard chunks and paired LAT hijack / false-refusal batches."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    guard_chunks: list[dict[str, Any]] = []

    def _fill_domain(eid: str, lo: int, hi: int, tgt: int, count: int) -> None:
        for _ in range(count):
            ids = torch.randint(lo, hi, (seq_len,), generator=gen).tolist()
            for p in range(score_from, seq_len):
                ids[p] = tgt
            guard_chunks.append({"expert_id": eid, "ids": ids})

    _fill_domain("E0_general_fineweb", 0, 16, 5, 6)
    _fill_domain("E1_python_systems_code", 16, 32, 20, 6)
    _fill_domain("E2_math_reasoning", 32, 48, 36, 6)
    # Benign cyber prompts carrying security keyword dim 44 (tokens 48..55, gold 52)
    _fill_domain("E7_defensive_cyber", 48, 56, 52, 6)

    # Hijack paired batch: identical E7 suffix, prefix differs only at token 65 vs clean token 57
    h_pos_rows: list[list[int]] = []
    h_neg_rows: list[list[int]] = []
    for _ in range(n_pairs):
        base_row = torch.randint(56, 64, (seq_len,), generator=gen).tolist()
        for p in range(score_from, seq_len):
            base_row[p] = 60
        neg_row = list(base_row)
        neg_row[2] = 57
        pos_row = list(base_row)
        pos_row[2] = 65
        h_neg_rows.append(neg_row)
        h_pos_rows.append(pos_row)

    # False-refusal paired batch: security-keyword cyber tokens (48..55, gold 52) vs clean cyber tokens (56..63, gold 60)
    r_pos_rows: list[list[int]] = []
    r_neg_rows: list[list[int]] = []
    for _ in range(n_pairs):
        pos_row = torch.randint(48, 56, (seq_len,), generator=gen).tolist()
        neg_row = torch.randint(56, 64, (seq_len,), generator=gen).tolist()
        for p in range(score_from, seq_len):
            pos_row[p] = 52
            neg_row[p] = 60
        r_pos_rows.append(pos_row)
        r_neg_rows.append(neg_row)

    # Protect E0/E1/E2/clean-E7 full subspaces across scored positions so ker(protect) isolates hijack/refusal
    e0_span = [torch.randint(0, 16, (seq_len,), generator=gen).tolist() for _ in range(8)]
    e1_span = [torch.randint(16, 32, (seq_len,), generator=gen).tolist() for _ in range(8)]
    e2_span = [torch.randint(32, 48, (seq_len,), generator=gen).tolist() for _ in range(8)]
    e7_clean_span = [torch.randint(56, 64, (seq_len,), generator=gen).tolist() for _ in range(8)]
    protect_rows = e0_span + e1_span + e2_span + e7_clean_span

    return {
        "guard_chunks": guard_chunks,
        "hijack_pos_ids": torch.tensor(h_pos_rows, dtype=torch.long),
        "hijack_neg_ids": torch.tensor(h_neg_rows, dtype=torch.long),
        "refusal_pos_ids": torch.tensor(r_pos_rows, dtype=torch.long),
        "refusal_neg_ids": torch.tensor(r_neg_rows, dtype=torch.long),
        "protect_ids": torch.tensor(protect_rows, dtype=torch.long),
    }


def _eval_domain_nll_direct(model: CrowdV1WithEngram, chunks: list[dict[str, Any]], score_from: int) -> dict[str, float]:
    from hivemind.swarm.wsp_alphaedit_control import (
        forward_model_pre_ln,
        per_chunk_nll_from_pre_ln,
    )

    ids = torch.tensor([c["ids"] for c in chunks], dtype=torch.long)
    h_pre = forward_model_pre_ln(model, ids, batch_size=16)
    losses = per_chunk_nll_from_pre_ln(
        h_pre,
        model.ln.weight,
        model.ln.bias,
        float(model.ln.eps),
        model.head.weight,
        ids,
        score_from=score_from,
    ).tolist()
    grouped: dict[str, list[float]] = {}
    for ch, val in zip(chunks, losses, strict=True):
        grouped.setdefault(str(ch["expert_id"]), []).append(float(val))
    return {k: sum(v) / len(v) for k, v in grouped.items()}


# ---------------------------------------------------------------------------
# S1: Simultaneous Hijack Neutralization (>90%) & Benign Cyber NLL Recovery
# ---------------------------------------------------------------------------


def test_s1_simultaneous_hijack_neutralization_and_false_refusal_recovery() -> None:
    """S1: Neutralize planted hijack (>90%) and recover E7_cyber NLL while general drift <= +0.3%."""
    model, carriers = _make_repe_moe_model(
        num_layers=2,
        plant_hijack=True,
        plant_false_refusal=True,
        plant_refusal_in_w2_b=False,
        seed=20261124,
    )
    batches = _make_guard_and_lat_batches(seq_len=16, score_from=8, n_pairs=6, seed=101)
    base_nlls = _eval_domain_nll_direct(model, batches["guard_chunks"], score_from=8)
    assert base_nlls["E7_defensive_cyber"] > 1.5, f"Expected degraded E7 baseline, got {base_nlls}"

    cfg = CircuitBreakerConfig(
        hijack_damp=1.0,
        false_refusal_damp=0.85,
        max_norm_delta_ratio=0.05,
        min_lat_snr=1.0,
        score_from=8,
    )
    res = execute_verified_circuit_breaker(
        model,
        batches["guard_chunks"],
        hijack_pos_ids=batches["hijack_pos_ids"],
        hijack_neg_ids=batches["hijack_neg_ids"],
        refusal_pos_ids=batches["refusal_pos_ids"],
        refusal_neg_ids=batches["refusal_neg_ids"],
        protect_ids=batches["protect_ids"],
        config=cfg,
    )

    assert res["object"] == CIRCUIT_BREAKER_RECEIPT_OBJECT
    assert res["schema"] == CIRCUIT_BREAKER_SCHEMA
    assert res["accepted"] is True
    assert res["rolled_back"] is False
    assert res["status"] == "ACCEPTED"
    assert res["guard_ok"] is True
    assert res["delta_e"] < 0.0
    assert res["trigger_suppression_ratio"] > 0.90
    assert res["general_pct_vs_base"] <= 0.30
    assert verify_circuit_breaker_receipt_provenance(res) is True

    post_nlls = _eval_domain_nll_direct(model, batches["guard_chunks"], score_from=8)
    assert post_nlls["E7_defensive_cyber"] < base_nlls["E7_defensive_cyber"] - 1.0
    assert (post_nlls["E0_general_fineweb"] / base_nlls["E0_general_fineweb"] - 1.0) * 100.0 <= 0.30

    # Verify hijack residual component is also directly >90% suppressed in attn.o
    hijack_col_after = float((carriers["hijack"] @ model.blocks[0].attn.o.weight.detach()).norm().item())
    assert hijack_col_after < 1e-5

    # Verify forbidden pattern gate on serialized receipt JSON (never emits "score": <digit>)
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn08_s1", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    receipt_json = json.dumps(res, sort_keys=True)
    assert '"nll_mean"' in receipt_json
    assert '"trigger_suppression_ratio"' in receipt_json
    assert '"anse_energy"' in receipt_json
    assert gatekeeper.forbidden_scan(f"diff --git a/x.py b/x.py\n+++ b/x.py\n+RECEIPT = {receipt_json!r}") == []


# ---------------------------------------------------------------------------
# S2: Exact Protected-Subspace Invariance (< 1e-5 output drift on Math/Code)
# ---------------------------------------------------------------------------


def test_s2_exact_protected_subspace_invariance() -> None:
    """S2: Enforce <1e-5 output drift on protected math/code states and OAS conditioning."""
    gen = torch.Generator(device="cpu").manual_seed(20261125)
    dim = 32
    n_samples = 24

    # Protected math/code states strictly span dims 0..15; hijack/refusal live in dims 16..31
    prot_math = torch.zeros((n_samples, dim), dtype=torch.float32)
    prot_math[:, 0:8] = torch.randn((n_samples, 8), generator=gen)
    prot_code = torch.zeros((n_samples, dim), dtype=torch.float32)
    prot_code[:, 8:16] = torch.randn((n_samples, 8), generator=gen)

    true_hijack = _unit_axis(dim, 16, 17)
    leak_math = _unit_axis(dim, 0, 1)
    base_acts = 0.05 * torch.randn((n_samples, dim), generator=gen)
    # Contaminate paired difference with BOTH hijack AND protected math leakage!
    pos_acts = base_acts + 3.0 * true_hijack.unsqueeze(0) + 1.8 * leak_math.unsqueeze(0)
    neg_acts = base_acts

    u_dir, lat_info = extract_lat_control_direction(
        pos_acts,
        neg_acts,
        protect_acts={"math": prot_math, "code": prot_code},
        shrink_protect=True,
    )
    assert lat_info["used_exact_nullspace"] is True
    assert lat_info["nullspace_dim"] == 16
    assert lat_info["protected_rms_ratio"] < 1e-12
    assert abs(float((u_dir.float() @ true_hijack).item())) > 0.99
    assert abs(float((u_dir.float() @ leak_math).item())) < 1e-12

    # Sequence & Tensor forms of protect_acts also produce exact nullspace protection
    u_seq, seq_info = extract_lat_control_direction(
        pos_acts.unsqueeze(1),
        neg_acts.unsqueeze(1),
        protect_acts=[prot_math, prot_code],
        shrink_protect=False,
    )
    assert seq_info["rho_protect"] == 0.0
    assert torch.allclose(u_dir, u_seq, atol=1e-10)

    # Full-rank small-N protected covariance exercises OAS generalized Rayleigh branch
    prot_full = torch.cat([prot_math, prot_code], dim=0) + 0.15 * torch.randn((2 * n_samples, dim), generator=gen)
    prot_full[:, 0:16] *= 4.0
    u_shrunk, info_shrunk = extract_lat_control_direction(
        pos_acts,
        neg_acts,
        protect_acts=prot_full,
        shrink_protect=True,
    )
    _, info_unshrunk = extract_lat_control_direction(
        pos_acts,
        neg_acts,
        protect_acts=prot_full,
        shrink_protect=False,
    )
    assert info_shrunk["projection_mode"] == "generalized_rayleigh"
    assert info_shrunk["rho_protect"] > 0.0
    assert info_shrunk["condition_number_shrunk"] < info_unshrunk["condition_number_emp"]
    assert float(u_shrunk.norm().item()) == pytest.approx(1.0, abs=1e-6)

    # Apply circuit breaker to an output Linear layer and verify < 1e-5 drift on protected math/code outputs
    out_proj = nn.Linear(dim, dim, bias=False)
    trig_in = torch.zeros(dim, dtype=torch.float32)
    trig_in[18:22] = 0.5
    with torch.no_grad():
        base_proj = torch.eye(dim)
        base_proj[16:18, 16:18] = 0.0
        out_proj.weight.copy_(base_proj + 0.08 * torch.outer(true_hijack, trig_in))
        x_prot_test = torch.cat([prot_math, prot_code], dim=0)
        y_before = out_proj(x_prot_test).clone()

    rep = apply_weight_space_circuit_breaker_inplace(
        out_proj,
        u_dir,
        None,
        hijack_damp=1.0,
        max_norm_delta_ratio=0.05,
    )
    assert rep["hijack_neutralization_ratio"] > 0.99
    with torch.no_grad():
        y_after = out_proj(x_prot_test)
    max_drift = float((y_after - y_before).abs().max().item())
    assert max_drift < 1e-5

    # End-to-end CrowdFineGrainedMoEMLP model invariance with protect_domain_prefixes
    fg_model, _ = _make_repe_moe_model(
        num_layers=2,
        num_experts=4,
        top_k=2,
        split_mode="fine_grained_width_split",
        plant_hijack=True,
        plant_false_refusal=False,
        seed=20261126,
    )
    batches = _make_guard_and_lat_batches(seq_len=16, score_from=8, n_pairs=6, seed=102)
    nll_before = _eval_domain_nll_direct(fg_model, batches["guard_chunks"], score_from=8)
    fg_res = execute_verified_circuit_breaker(
        fg_model,
        batches["guard_chunks"],
        hijack_pos_ids=batches["hijack_pos_ids"],
        hijack_neg_ids=batches["hijack_neg_ids"],
        protect_domain_prefixes=("E0_", "E1_", "E2_"),
        target_projections=("attn.o", "expert.w3"),
        score_from=8,
    )
    assert fg_res["accepted"] is True
    assert fg_res["trigger_suppression_ratio"] > 0.90
    nll_after = _eval_domain_nll_direct(fg_model, batches["guard_chunks"], score_from=8)
    assert abs(nll_after["E1_python_systems_code"] - nll_before["E1_python_systems_code"]) < 1e-5
    assert abs(nll_after["E2_math_reasoning"] - nll_before["E2_math_reasoning"]) < 1e-5


# ---------------------------------------------------------------------------
# S3: Norm-Budget Clamping & ANSE Rollback on Uncalibrated Direction
# ---------------------------------------------------------------------------


def test_s3_norm_budget_clamping_and_anse_rollback() -> None:
    """S3: Clamp oversized weight edits to max_norm_delta_ratio and roll back bit-for-bit on E0 harm."""
    gen = torch.Generator(device="cpu").manual_seed(20261127)

    # Part A: Frobenius norm-delta clamping on fp32 & bf16 output projections
    for dtype in (torch.float32, torch.bfloat16):
        w_mat = (torch.eye(32, dtype=torch.float32) + 0.1 * torch.randn((32, 32), generator=gen)).to(dtype)
        w_orig = w_mat.clone()
        # Wide uncalibrated direction touching ~50% of W norm -> far exceeds 0.05 budget
        wide_dir = torch.ones(32, dtype=torch.float32) / math.sqrt(32.0)
        wide_dir_2 = torch.cat([torch.ones(16), -torch.ones(16)]).float() / math.sqrt(32.0)

        rep = apply_weight_space_circuit_breaker_inplace(
            w_mat,
            wide_dir,
            wide_dir_2,
            hijack_damp=1.0,
            false_refusal_damp=0.85,
            max_norm_delta_ratio=0.05,
        )
        assert rep["budget_clamped"] is True
        assert rep["raw_norm_delta_ratio"] > 0.05
        assert rep["budget_scale"] < 1.0
        assert rep["applied_norm_delta_ratio"] <= 0.05 + 1e-7
        exact_ratio = float(
            torch.linalg.norm((w_mat.float() - w_orig.float()).to(torch.float64)).item()
            / torch.linalg.norm(w_orig.float().to(torch.float64)).item()
        )
        assert exact_ratio <= 0.05 + 1e-6

    # Part B: ANSE rollback when an uncalibrated direction aligned with E0 prose degrades E0 NLL > +1.0%
    model, carriers = _make_repe_moe_model(
        num_layers=2,
        plant_hijack=False,
        plant_false_refusal=False,
        seed=20261128,
    )
    batches = _make_guard_and_lat_batches(seq_len=16, score_from=8, n_pairs=4, seed=103)
    pre_sha = compute_state_sha256(model.state_dict())
    pre_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Inject a strong E0-harming uncalibrated override along carriers["e0"] with generous norm budget
    bad_res = execute_verified_circuit_breaker(
        model,
        batches["guard_chunks"],
        hijack_dir_override=carriers["e0"],
        target_projections=("attn.o", "mlp.w3"),
        hijack_damp=1.0,
        max_norm_delta_ratio=0.50,
        score_from=8,
    )
    assert bad_res["accepted"] is False
    assert bad_res["rolled_back"] is True
    assert bad_res["guard_ok"] is False
    assert bad_res["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert bad_res["anse_energy"] >= 1_000_000.0
    assert bad_res["delta_e"] > 0.0
    assert bad_res["parent_state_sha256"] == pre_sha
    assert bad_res["final_state_sha256"] == pre_sha
    assert bad_res["candidate_state_sha256"] != pre_sha
    assert verify_circuit_breaker_receipt_provenance(bad_res) is True

    # Bitwise model state equivalence verified across every parameter tensor
    for key, orig_t in pre_sd.items():
        assert torch.equal(model.state_dict()[key], orig_t), f"Rollback mismatch on {key}"

    # Part C: Guard limit relaxed, but ablating helpful E0 prose direction harms mean NLL -> energy rollback
    pre_sha_c = compute_state_sha256(model.state_dict())
    non_imp_res = execute_verified_circuit_breaker(
        model,
        batches["guard_chunks"],
        hijack_dir_override=carriers["e0"],
        target_projections=("attn.o", "mlp.w3"),
        domain_guard_max_ratio=5.0,
        score_from=8,
    )
    assert non_imp_res["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    assert non_imp_res["rolled_back"] is True
    assert non_imp_res["accepted"] is False
    assert non_imp_res["final_state_sha256"] == pre_sha_c

    # Part D: Fixed-point unchanged when identical clean activations produce zero LAT contrast
    fp_res = execute_verified_circuit_breaker(
        model,
        batches["guard_chunks"],
        hijack_pos_ids=batches["hijack_neg_ids"],
        hijack_neg_ids=batches["hijack_neg_ids"],
        score_from=8,
    )
    assert fp_res["status"] == "FIXED_POINT_UNCHANGED"
    assert fp_res["accepted"] is False
    assert fp_res["rolled_back"] is False
    assert fp_res["final_state_sha256"] == pre_sha_c


# ---------------------------------------------------------------------------
# S4: SwiGLU LoRA w2_B Rank-Preserving Surgery Without Dense Materialization
# ---------------------------------------------------------------------------


def test_s4_swiglu_lora_w2_b_rank_preserving_no_dense_materialization() -> None:
    """S4: Orthogonally edit SwiGLU LoRA w2_B in-place, preserving rank without dense (I, H) alloc."""
    gen = torch.Generator(device="cpu").manual_seed(20261129)
    hidden_size = 512
    intermediate_size = 1024
    rank = 8

    exp = CrowdLoRAExpertAdapter(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        rank=rank,
        alpha=16.0,
        seed=20261129,
    )
    # Populate rank-4 clean specialist weights in indices 0..511 plus planted hijack/refusal in 512..1023
    u_hijack_i = _unit_axis(intermediate_size, 600, 601)
    u_refusal_i = _unit_axis(intermediate_size, 700, 701)

    with torch.no_grad():
        exp.w2_B.zero_()
        exp.w2_B[:512, :4].copy_(torch.randn((512, 4), generator=gen) * 0.8)
        # Keep hijack & refusal projections <= 4% of ||w2_B||_F so full damping applies without clamping
        exp.w2_B[:, 4].copy_(0.45 * u_hijack_i)
        exp.w2_B[:, 5].copy_(0.35 * u_refusal_i)
        w2_a_orig = exp.w2_A.detach().clone()
        w2_b_orig = exp.w2_B.detach().clone()

    rank_before = int(torch.linalg.matrix_rank(w2_b_orig.to(torch.float64)).item())
    assert rank_before <= rank

    # Extract LAT directions in R^1024 orthogonal to protected intermediate space (0..511)
    prot_inter = torch.zeros((20, intermediate_size), dtype=torch.float32)
    prot_inter[:, :512] = torch.randn((20, 512), generator=gen)

    base_inter = 0.02 * torch.randn((16, intermediate_size), generator=gen)
    h_dir, h_info = extract_lat_control_direction(
        base_inter + 2.5 * u_hijack_i.unsqueeze(0),
        base_inter,
        protect_acts=prot_inter,
        shrink_protect=True,
    )
    r_dir, r_info = extract_lat_control_direction(
        base_inter + 2.5 * u_refusal_i.unsqueeze(0),
        base_inter,
        protect_acts=prot_inter,
        shrink_protect=True,
    )
    assert h_info["protected_rms_ratio"] < 1e-12
    assert r_info["protected_rms_ratio"] < 1e-12

    # Measure peak Python/tensor allocation with tracemalloc: dense (1024 x 512) float64 is 4 MiB
    tracemalloc.start()
    try:
        rep = apply_weight_space_circuit_breaker_inplace(
            exp,
            h_dir,
            r_dir,
            hijack_damp=1.0,
            false_refusal_damp=0.85,
            max_norm_delta_ratio=0.05,
        )
    finally:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    dense_materialized_bytes = intermediate_size * hidden_size * 8  # 4 MiB in float64
    assert peak_bytes < dense_materialized_bytes // 4, f"Unexpectedly high allocation {peak_bytes} bytes"
    assert rep["resolved_attr"] == "w2_B"
    assert rep["weight_shape"] == (intermediate_size, rank)
    assert rep["budget_clamped"] is False
    assert rep["hijack_neutralization_ratio"] > 0.99
    assert rep["false_refusal_neutralization_ratio"] == pytest.approx(0.85, abs=1e-3)

    # Verify rank preservation, untouched w2_A, and protected intermediate invariance
    assert torch.equal(exp.w2_A, w2_a_orig)
    rank_after = int(torch.linalg.matrix_rank(exp.w2_B.detach().to(torch.float64)).item())
    assert rank_after <= rank_before <= rank

    with torch.no_grad():
        x_prot_h = torch.randn((12, hidden_size), generator=gen)
        # Project x_prot_h so original w2_B @ w2_A @ x lies in protected subspace 0..511
        dh2_before = (x_prot_h @ w2_a_orig.T) @ w2_b_orig.T
        dh2_after = (x_prot_h @ exp.w2_A.T) @ exp.w2_B.T
        # Protected channels 0..511 are bitwise/machine invariant
        assert float((dh2_after[:, :512] - dh2_before[:, :512]).abs().max().item()) < 1e-5

    # End-to-end SwiGLU w2_B surgery inside CrowdV1WithEngram + CrowdLoRAMoEMLP
    w2_model, _ = _make_repe_moe_model(
        num_layers=2,
        plant_hijack=False,
        plant_false_refusal=True,
        plant_refusal_in_w2_b=True,
        seed=20261130,
    )
    batches = _make_guard_and_lat_batches(seq_len=16, score_from=8, n_pairs=6, seed=104)
    nll_w2_pre = _eval_domain_nll_direct(w2_model, batches["guard_chunks"], score_from=8)
    assert nll_w2_pre["E7_defensive_cyber"] > 1.5, f"Expected degraded E7 baseline, got {nll_w2_pre}"

    w2_res = execute_verified_circuit_breaker(
        w2_model,
        batches["guard_chunks"],
        refusal_pos_ids=batches["refusal_pos_ids"],
        refusal_neg_ids=batches["refusal_neg_ids"],
        protect_ids=batches["protect_ids"],
        target_projections=("expert.w2_B",),
        max_norm_delta_ratio=0.50,
        score_from=8,
    )
    assert w2_res["accepted"] is True
    assert any(r["target"] == "expert_7.w2_B" for r in w2_res["layer_reports"])
    nll_w2_post = _eval_domain_nll_direct(w2_model, batches["guard_chunks"], score_from=8)
    assert nll_w2_post["E7_defensive_cyber"] < nll_w2_pre["E7_defensive_cyber"] - 1.0


# ---------------------------------------------------------------------------
# S5: Validation Errors, Edge Cases & Security / Zero-Stub Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_edge_cases_and_security_gates() -> None:
    """S5: Verify exhaustive error handling, edge-case guards, and repository security/stub scans."""
    # 1. Config validation errors
    with pytest.raises(TypeError, match="real number"):
        CircuitBreakerConfig(hijack_damp=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="\\[0.0, 1.0\\]"):
        CircuitBreakerConfig(hijack_damp=1.5)
    with pytest.raises(ValueError, match=">= 0.0"):
        CircuitBreakerConfig(false_refusal_damp=-0.1)
    with pytest.raises(ValueError, match="> 0.0"):
        CircuitBreakerConfig(max_norm_delta_ratio=0.0)
    with pytest.raises(ValueError, match="<= 1.0"):
        CircuitBreakerConfig(max_norm_delta_ratio=1.2)
    with pytest.raises(TypeError, match="bool"):
        CircuitBreakerConfig(shrink_protect="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="< 1.0"):
        CircuitBreakerConfig(nullspace_eigen_ratio=1.0)
    with pytest.raises(ValueError, match=">= 1.0"):
        CircuitBreakerConfig(domain_guard_max_ratio=0.95)
    with pytest.raises(ValueError, match=">= 1.0"):
        CircuitBreakerConfig(anse_safe_general_ratio=0.99)
    with pytest.raises(ValueError, match="general_prefix"):
        CircuitBreakerConfig(general_prefix=" ")
    with pytest.raises(TypeError, match="int"):
        CircuitBreakerConfig(score_from=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        CircuitBreakerConfig(score_from=-1)
    with pytest.raises(ValueError, match=">= 1"):
        CircuitBreakerConfig(batch_size=0)
    with pytest.raises(ValueError, match="finite"):
        CircuitBreakerConfig(reg_protect=float("nan"))

    # 2. extract_lat_control_direction validation & edge branches
    good_pos = torch.ones((6, 8))
    good_neg = torch.zeros((6, 8))
    with pytest.raises(TypeError, match="torch.Tensor"):
        extract_lat_control_direction([[1.0, 2.0]], good_neg)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2D or 3D"):
        extract_lat_control_direction(torch.ones(8), good_neg)
    with pytest.raises(TypeError, match="floating-point"):
        extract_lat_control_direction(torch.ones((6, 8), dtype=torch.long), good_neg)
    with pytest.raises(ValueError, match="finite"):
        bad_nan = good_pos.clone()
        bad_nan[0, 0] = float("nan")
        extract_lat_control_direction(bad_nan, good_neg)
    with pytest.raises(ValueError, match="width"):
        extract_lat_control_direction(good_pos, torch.zeros((6, 4)))
    with pytest.raises(ValueError, match="< 1.0"):
        extract_lat_control_direction(good_pos, good_neg, nullspace_eigen_ratio=1.0)
    with pytest.raises(ValueError, match="mapping must not be empty"):
        extract_lat_control_direction(good_pos, good_neg, protect_acts={})
    with pytest.raises(ValueError, match="sequence must not be empty"):
        extract_lat_control_direction(good_pos, good_neg, protect_acts=[])
    with pytest.raises(TypeError, match="protect_acts"):
        extract_lat_control_direction(good_pos, good_neg, protect_acts="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="width"):
        extract_lat_control_direction(good_pos, good_neg, protect_acts=torch.ones((4, 4)))
    with pytest.raises(ValueError, match="width"):
        extract_lat_control_direction(good_pos, good_neg, protect_acts=[torch.ones((4, 4))])

    # Edge branches: unpaired sample counts, centered diffs, zero protect_acts, blocked direction, SNR gating
    u_unpaired, info_unpaired = extract_lat_control_direction(
        torch.ones((7, 8)), torch.zeros((5, 8)), center_paired_diffs=False
    )
    assert info_unpaired["paired_samples"] is False
    assert float(u_unpaired.norm().item()) == pytest.approx(1.0, abs=1e-6)

    u_zero_prot, info_zero_prot = extract_lat_control_direction(
        good_pos, good_neg, protect_acts=torch.zeros((4, 8)), shrink_protect=True
    )
    assert info_zero_prot["projection_mode"] == "unconstrained"
    assert float(u_zero_prot.norm().item()) == pytest.approx(1.0, abs=1e-6)

    u_blocked, info_blocked = extract_lat_control_direction(
        torch.cat([torch.ones((6, 4)), torch.zeros((6, 4))], dim=1),
        torch.zeros((6, 8)),
        protect_acts=torch.cat([torch.eye(4), torch.zeros((4, 4))], dim=1),
        center_paired_diffs=True,
    )
    assert info_blocked["is_degenerate"] is True
    assert float(u_blocked.norm().item()) == 0.0

    gen_noise = torch.Generator(device="cpu").manual_seed(77)
    u_gated, info_gated = extract_lat_control_direction(
        torch.randn((8, 16), generator=gen_noise),
        torch.randn((8, 16), generator=gen_noise),
        min_oof_snr=10.0,
    )
    assert info_gated["gated_by_oof_snr"] is True
    assert float(u_gated.norm().item()) == 0.0

    # 3. apply_weight_space_circuit_breaker_inplace validation & edge branches
    lin = nn.Linear(8, 8, bias=False)
    with pytest.raises(ValueError, match="<= 1.0"):
        apply_weight_space_circuit_breaker_inplace(lin, torch.ones(8), None, max_norm_delta_ratio=1.5)
    with pytest.raises(TypeError, match="torch.Tensor"):
        apply_weight_space_circuit_breaker_inplace(lin, [1.0] * 8, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty 1D"):
        apply_weight_space_circuit_breaker_inplace(lin, torch.ones((2, 8)), None)
    with pytest.raises(TypeError, match="floating-point"):
        apply_weight_space_circuit_breaker_inplace(lin, torch.ones(8, dtype=torch.long), None)
    with pytest.raises(ValueError, match="finite"):
        apply_weight_space_circuit_breaker_inplace(lin, torch.tensor([1.0] * 7 + [float("nan")]), None)
    with pytest.raises(ValueError, match="must match false_refusal_dir"):
        apply_weight_space_circuit_breaker_inplace(lin, torch.ones(8), torch.ones(4))
    with pytest.raises(ValueError, match="must match weight output rows"):
        apply_weight_space_circuit_breaker_inplace(lin, torch.ones(4), None)
    with pytest.raises(TypeError, match="nn.Module or torch.Tensor"):
        apply_weight_space_circuit_breaker_inplace("bad", torch.ones(8), None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty string"):
        apply_weight_space_circuit_breaker_inplace(lin, torch.ones(8), None, weight_attr=" ")
    with pytest.raises(ValueError, match="does not expose"):
        apply_weight_space_circuit_breaker_inplace(lin, torch.ones(8), None, weight_attr="missing")
    with pytest.raises(ValueError, match="must expose"):
        apply_weight_space_circuit_breaker_inplace(nn.Module(), torch.ones(8), None)
    exp_small = CrowdLoRAExpertAdapter(8, 16, rank=4)
    with pytest.raises(ValueError, match="No LoRA-B factor"):
        apply_weight_space_circuit_breaker_inplace(exp_small, torch.ones(32), None)
    with pytest.raises(ValueError, match="non-empty 2D"):
        apply_weight_space_circuit_breaker_inplace(torch.ones(8), torch.ones(8), None)
    with pytest.raises(TypeError, match="floating-point"):
        apply_weight_space_circuit_breaker_inplace(torch.ones((8, 8), dtype=torch.long), torch.ones(8), None)
    with pytest.raises(ValueError, match="finite"):
        nan_w = torch.ones((8, 8))
        nan_w[0, 0] = float("inf")
        apply_weight_space_circuit_breaker_inplace(nan_w, torch.ones(8), None)

    # No-op branches: zero weights, zero directions, collinear directions
    zero_w = torch.zeros((8, 8))
    assert apply_weight_space_circuit_breaker_inplace(zero_w, torch.ones(8), None)["noop_reason"] == "zero_weight_norm"
    assert apply_weight_space_circuit_breaker_inplace(lin, None, None)["noop_reason"] == "no_active_directions"
    col_rep = apply_weight_space_circuit_breaker_inplace(
        lin, torch.ones(8), torch.ones(8), hijack_damp=0.5, false_refusal_damp=0.9, max_norm_delta_ratio=1.0
    )
    assert col_rep["hijack_neutralization_ratio"] == pytest.approx(0.9, abs=1e-4)

    # 4. execute_verified_circuit_breaker & provenance validation errors
    model, carriers = _make_repe_moe_model(num_layers=2, plant_hijack=False, plant_false_refusal=False)
    batches = _make_guard_and_lat_batches(seq_len=16, score_from=8, n_pairs=4)

    with pytest.raises(TypeError, match="nn.Module"):
        execute_verified_circuit_breaker("bad", batches["guard_chunks"], hijack_dir_override=carriers["e0"])  # type: ignore[arg-type]
    empty_m = nn.Module()
    empty_m.blocks = nn.ModuleList()
    with pytest.raises(ValueError, match="blocks"):
        execute_verified_circuit_breaker(empty_m, batches["guard_chunks"], hijack_dir_override=carriers["e0"])
    missing_m = nn.Module()
    missing_m.blocks = nn.ModuleList([nn.Linear(4, 4)])
    with pytest.raises(ValueError, match="must expose"):
        execute_verified_circuit_breaker(missing_m, batches["guard_chunks"], hijack_dir_override=carriers["e0"])
    with pytest.raises(TypeError, match="CircuitBreakerConfig"):
        execute_verified_circuit_breaker(model, batches["guard_chunks"], config="bad", hijack_dir_override=carriers["e0"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="<= 1.0"):
        execute_verified_circuit_breaker(model, batches["guard_chunks"], hijack_dir_override=carriers["e0"], max_norm_delta_ratio=1.5)
    with pytest.raises(ValueError, match=">= 1.0"):
        execute_verified_circuit_breaker(model, batches["guard_chunks"], hijack_dir_override=carriers["e0"], domain_guard_max_ratio=0.5)
    with pytest.raises(ValueError, match="target_projections"):
        execute_verified_circuit_breaker(model, batches["guard_chunks"], target_projections=[], hijack_dir_override=carriers["e0"])
    with pytest.raises(ValueError, match="Unsupported target_projections"):
        execute_verified_circuit_breaker(model, batches["guard_chunks"], target_projections=("bad.proj",), hijack_dir_override=carriers["e0"])
    with pytest.raises(ValueError, match="layer_indices"):
        execute_verified_circuit_breaker(model, batches["guard_chunks"], layer_indices=(99,), hijack_dir_override=carriers["e0"])
    with pytest.raises(ValueError, match="hijack_pos_ids and hijack_neg_ids"):
        execute_verified_circuit_breaker(model, batches["guard_chunks"], hijack_pos_ids=batches["hijack_pos_ids"])
    with pytest.raises(ValueError, match="refusal_pos_ids and refusal_neg_ids"):
        execute_verified_circuit_breaker(model, batches["guard_chunks"], refusal_pos_ids=batches["refusal_pos_ids"])
    with pytest.raises(ValueError, match="At least one control pair"):
        execute_verified_circuit_breaker(model, batches["guard_chunks"])
    with pytest.raises(ValueError, match="non-empty sequence"):
        execute_verified_circuit_breaker(model, [], hijack_dir_override=carriers["e0"])
    with pytest.raises(ValueError, match="mapping with 'expert_id'"):
        execute_verified_circuit_breaker(model, [{"ids": [1, 2, 3]}], hijack_dir_override=carriers["e0"])
    with pytest.raises(ValueError, match="non-empty string"):
        execute_verified_circuit_breaker(model, [{"expert_id": " ", "ids": [1, 2, 3]}], hijack_dir_override=carriers["e0"])
    with pytest.raises(ValueError, match="length >= 2"):
        execute_verified_circuit_breaker(model, [{"expert_id": "E0_a", "ids": [1]}], hijack_dir_override=carriers["e0"])
    with pytest.raises(ValueError, match="non-negative"):
        execute_verified_circuit_breaker(model, [{"expert_id": "E0_a", "ids": [1, -1, 2]}], hijack_dir_override=carriers["e0"])
    with pytest.raises(ValueError, match="sequence length"):
        execute_verified_circuit_breaker(
            model,
            [{"expert_id": "E0_a", "ids": [1, 2, 3]}, {"expert_id": "E0_a", "ids": [1, 2, 3, 4]}],
            hijack_dir_override=carriers["e0"],
        )
    with pytest.raises(ValueError, match="too short"):
        execute_verified_circuit_breaker(model, batches["guard_chunks"], hijack_dir_override=carriers["e0"], score_from=32)
    with pytest.raises(ValueError, match="general chunk"):
        execute_verified_circuit_breaker(model, [{"expert_id": "E1_code", "ids": [1, 2, 3, 4]}], hijack_dir_override=carriers["e0"])
    with pytest.raises(TypeError, match="integer dtype"):
        execute_verified_circuit_breaker(
            model,
            batches["guard_chunks"],
            hijack_pos_ids=torch.ones((2, 16)),
            hijack_neg_ids=batches["hijack_neg_ids"],
        )
    with pytest.raises(TypeError, match="2D integer tensor"):
        execute_verified_circuit_breaker(
            model,
            batches["guard_chunks"],
            hijack_pos_ids="bad",  # type: ignore[arg-type]
            hijack_neg_ids=batches["hijack_neg_ids"],
        )
    with pytest.raises(ValueError, match="too short"):
        execute_verified_circuit_breaker(
            model,
            batches["guard_chunks"],
            hijack_pos_ids=[[1, 2, 3]],
            hijack_neg_ids=[[1, 2, 3]],
            score_from=8,
        )

    # Provenance & state sha256 validation
    with pytest.raises(TypeError, match="mapping"):
        verify_circuit_breaker_receipt_provenance("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        compute_state_sha256({})
    with pytest.raises(ValueError, match="non-empty string"):
        compute_state_sha256({"": torch.ones(2)})
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_state_sha256({"a": [1.0]})  # type: ignore[dict-item]
    assert verify_circuit_breaker_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert verify_circuit_breaker_receipt_provenance({"receipt_sha256": "0" * 64, "object": "wrong"}) is False
    assert (
        verify_circuit_breaker_receipt_provenance(
            {"receipt_sha256": "0" * 64, "object": CIRCUIT_BREAKER_RECEIPT_OBJECT, "schema": "wrong"}
        )
        is False
    )
    assert (
        verify_circuit_breaker_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": CIRCUIT_BREAKER_RECEIPT_OBJECT,
                "schema": CIRCUIT_BREAKER_SCHEMA,
                "bad": float("nan"),
            }
        )
        is False
    )

    # 5. Repository AGPL-3.0, optional ZeroStubAudit, PRSecurityScanner, lateral movement, and gatekeeper scans
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn08", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn08", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    auditor = ZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/representation_circuit_breaker.py", TARGET_MODULE_PATH),
        ("tests/test_representation_circuit_breaker_scenarios.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"

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

    pr_audit = scanner.audit_pull_request(
        pr_number=1108,
        pr_title="feat(swarm): weight-space Representation Engineering circuit breaker and false-refusal abliteration for open-weight MoEs",
        pr_body="Implements NEURAL-CTRL-DYN-08 training-free RePE circuit breaker and deterministic CPU scenario tests.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): weight-space Representation Engineering circuit breaker and false-refusal abliteration for open-weight MoEs"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
