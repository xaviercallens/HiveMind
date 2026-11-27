# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-47: proof-carrying ZK-WSP verifier & Multi-Krum hive federation."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.security.owner_order_guardrail import (
    RequestIntentCategory,
    classify_request_intent,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import upcycle_crowd_v1_to_sparse_moe
from hivemind.swarm.lean4_wsp_theorem_registry import (
    audit_lean4_wsp_spec_file,
)
from hivemind.swarm.wsp_alphaedit_control import (
    forward_model_pre_ln,
    per_chunk_nll_from_pre_ln,
)
from hivemind.swarm.zk_wsp_federated_hive_verifier import (
    ANSE_MAXIMUM_PAIN,
    LEAN4_RCD_D5_THEOREM,
    ZKWSP_COMMITMENT_OBJECT,
    ZKWSP_PACKET_OBJECT,
    ZKWSP_ROUND_RECEIPT_OBJECT,
    ZKWSP_VERIFIER_SCHEMA,
    ZKWSPFederatedHiveConfig,
    build_federated_hive_packet,
    build_prose_subspace_commitment,
    build_prose_subspace_commitment_from_keys,
    collect_prose_subspace_from_probe,
    compute_payload_sha256,
    compute_state_sha256,
    compute_zkwsp_spotcheck_witness,
    execute_verified_zkwsp_federated_hive_round,
    verify_and_aggregate_federated_hives,
    verify_federated_hive_packet,
    verify_zkwsp_federated_round_receipt_provenance,
    verify_zkwsp_spotcheck_witness,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:  # pragma: no cover - optional in lightweight CI runners
    ZeroStubAudit = None  # type: ignore[assignment,misc]


class _FallbackStubResult:
    """CI-safe stub audit result when optional ``gwaya`` is absent."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = tuple(violations)
        self.is_clean = len(violations) == 0
        self.penalty_energy = 0.0 if self.is_clean else 1e6


class _FallbackZeroStubAudit:
    """CI-safe fallback implementing both ``check_file`` and ``audit_python_code``."""

    def check_file(self, path: str | Path) -> list[str]:
        text = Path(path).read_text(encoding="utf-8")
        return list(self.audit_python_code(text).violations)

    def audit_python_code(self, code: str) -> _FallbackStubResult:
        tree = ast.parse(code)
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stmts = [
                    s
                    for s in node.body
                    if not (
                        isinstance(s, ast.Expr)
                        and isinstance(s.value, ast.Constant)
                        and isinstance(s.value.value, str)
                    )
                ]
                if not stmts or (len(stmts) == 1 and isinstance(stmts[0], ast.Pass)):
                    violations.append(f"stub function: {node.name}")
                if (
                    len(stmts) == 1
                    and isinstance(stmts[0], ast.Raise)
                    and isinstance(stmts[0].exc, ast.Call)
                    and isinstance(stmts[0].exc.func, ast.Name)
                    and stmts[0].exc.func.id == "NotImplementedError"
                ):
                    violations.append(f"NotImplementedError in {node.name}")
        return _FallbackStubResult(violations)


REPO_ROOT = Path(__file__).resolve().parents[1]
LEAN_SPEC_PATH = REPO_ROOT / "lean4" / "HiveMindSpec.lean"
TARGET_MODULE_PATH = (
    REPO_ROOT / "src" / "hivemind" / "swarm" / "zk_wsp_federated_hive_verifier.py"
)
TEST_FILE_PATH = Path(__file__).resolve()
ROADMAP_DOC_PATH = REPO_ROOT / "docs" / "NEURO_CONTROLLER_ROADMAP.md"
PLAN_DOC_PATH = REPO_ROOT / "docs" / "NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md"
EXPECTED_AGPL_HEADER = (
    "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
)


@pytest.fixture(autouse=True, scope="module")
def _single_thread_cpu_torch() -> Any:
    """Eliminate OpenMP thread-pool spin on small deterministic CPU tensors."""
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prev_threads)


@pytest.fixture(scope="module")
def cached_lean_audit() -> Any:
    """Parse ``lean4/HiveMindSpec.lean`` once per test module."""
    return audit_lean4_wsp_spec_file(LEAN_SPEC_PATH)


def _load_module_from_path(name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_ortho_basis(dim: int, k_modes: int, seed: int) -> torch.Tensor:
    """Generate a deterministic orthonormal basis ``V_0`` in ``R^{dim x k_modes}``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mat = torch.randn((dim, k_modes), generator=gen, dtype=torch.float64)
    q_mat, _ = torch.linalg.qr(mat, mode="reduced")
    return q_mat.contiguous()


def _make_zero_mean_ortho_rows(n_rows: int, dim: int, seed: int) -> torch.Tensor:
    """Create ``n_rows`` orthonormal vectors orthogonal to the all-ones vector."""
    assert n_rows < dim
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ones = torch.ones((dim, 1), dtype=torch.float64)
    mat = torch.cat(
        [ones, torch.randn((dim, dim - 1), generator=gen, dtype=torch.float64)],
        dim=1,
    )
    q_mat, _ = torch.linalg.qr(mat)
    return q_mat[:, 1 : 1 + n_rows].T.to(torch.float32).contiguous()


def _build_federated_crowd_moe_fixture(*, seed: int = 20264700) -> dict[str, Any]:
    """Build deterministic 6-domain CrowdV1 MoE with orthogonal E0 prose and 5 specialist domains."""
    hidden_size = 48
    intermediate_size = 96
    vocab_size = 48
    num_experts = 6
    block_h = 8
    block_v = 8

    cfg = CrowdV1RuntimeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden_size // 4,
        intermediate_size=intermediate_size,
        max_position_embeddings=24,
        engram_buckets=0,
        tie_word_embeddings=False,
    )
    model = CrowdV1WithEngram(cfg)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    domain_specs = (
        (0, "E0_general_prose", 0, 0, slice(0, block_h)),
        (1, "E1_code_python", 1, block_v, slice(block_h, 2 * block_h)),
        (2, "E2_math_reasoning", 2, 2 * block_v, slice(2 * block_h, 3 * block_h)),
        (3, "E3_physics_sim", 3, 3 * block_v, slice(3 * block_h, 4 * block_h)),
        (4, "E4_cyber_sec", 4, 4 * block_v, slice(4 * block_h, 5 * block_h)),
        (5, "E5_formal_lean4", 5, 5 * block_v, slice(5 * block_h, 6 * block_h)),
    )

    axes_in: dict[int, torch.Tensor] = {}
    axes_out: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        head_w = 0.01 * torch.randn((vocab_size, hidden_size), generator=gen, dtype=torch.float32)

        for e_slot, _, b_idx, v_off, h_slice in domain_specs:
            basis = _make_zero_mean_ortho_rows(4, block_h, seed=seed + 31 + b_idx * 17)
            u_in, u_out = basis[0], basis[1]
            axes_in[e_slot] = u_in
            axes_out[e_slot] = u_out
            rows = _make_zero_mean_ortho_rows(block_h - 1, block_h, seed=seed + 131 + b_idx * 19)
            for t_i in range(block_v):
                vec = 0.25 * rows[t_i % int(rows.shape[0])] + 0.70 * u_in
                wte[v_off + t_i, h_slice] = vec - vec.mean()
            head_w[:, h_slice] = head_w[:, h_slice] - head_w[:, h_slice].mean(dim=1, keepdim=True)
            if e_slot == 0:
                # General prose already well-calibrated at base
                head_w[v_off + 1, h_slice] = 0.10 * u_in + 0.85 * u_out
                head_w[v_off + 2, h_slice] = 0.10 * u_in + 0.70 * basis[2]
            else:
                # Specialist heads start misaligned so honest P_0 head consensus drops specialist NLL
                head_w[v_off + 1, h_slice] = -0.25 * u_out
                head_w[v_off + 2, h_slice] = -0.20 * basis[2]

        model.wte.weight.copy_(wte)
        model.head.weight.copy_(head_w)
        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln2.weight.fill_(1.0)
            blk.attn.q.weight.copy_(0.12 * torch.eye(hidden_size))
            blk.attn.k.weight[: hidden_size // 2, : hidden_size // 2].copy_(0.12 * torch.eye(hidden_size // 2))
            blk.attn.v.weight[: hidden_size // 2, : hidden_size // 2].copy_(0.12 * torch.eye(hidden_size // 2))
            blk.attn.o.weight.copy_(0.12 * torch.eye(hidden_size))
            blk.mlp.w1.weight[:16, :16].copy_(0.15 * torch.eye(16))
            blk.mlp.w2.weight[:16, :16].copy_(0.15 * torch.eye(16))
            blk.mlp.w3.weight[:16, :16].copy_(0.05 * torch.eye(16))

    upcycle_crowd_v1_to_sparse_moe(
        model,
        num_experts=num_experts,
        top_k=2,
        lora_rank=2,
        routing_mode="sigma_topk",
        seed=seed + 7,
    )
    with torch.no_grad():
        for blk in model.blocks:
            blk.mlp.router.weight.zero_()
            for e_slot, _, _, _, h_slice in domain_specs:
                blk.mlp.router.weight[e_slot, h_slice] = 3.0 * axes_in[e_slot]
            for exp in blk.mlp.experts.values():
                for attr in ("w1_B", "w2_B", "w3_B"):
                    getattr(exp, attr).zero_()

    probe_chunks: list[dict[str, Any]] = []
    eval_chunks: list[dict[str, Any]] = []
    for e_slot, dom_name, _, v_off, _ in domain_specs:
        toks_a = [v_off + ((i * 2) % block_v) for i in range(16)]
        toks_b = [v_off + ((i * 3 + 1) % block_v) for i in range(16)]
        toks_c = [v_off + ((i + 1) % block_v) for i in range(16)]
        toks_d = [v_off + ((i * 5 + 2) % block_v) for i in range(16)]
        if e_slot == 0:
            toks_a = [v_off + (1 if i % 2 == 0 else 2) for i in range(16)]
            toks_b = [v_off + (2 if i % 2 == 0 else 1) for i in range(16)]
            toks_c = [v_off + (1 if i % 3 == 0 else 2) for i in range(16)]
            toks_d = [v_off + (2 if i % 3 == 0 else 1) for i in range(16)]
        else:
            toks_a = [v_off + (1 if i % 2 == 0 else 2) for i in range(16)]
            toks_b = [v_off + (2 if i % 2 == 0 else 1) for i in range(16)]
            toks_c = [v_off + (1 if i % 2 == 0 else 2) for i in range(16)]
            toks_d = [v_off + (2 if i % 2 == 0 else 1) for i in range(16)]
        probe_chunks.append({"expert_id": dom_name, "ids": toks_a})
        probe_chunks.append({"expert_id": dom_name, "ids": toks_b})
        eval_chunks.append({"expert_id": dom_name, "ids": toks_c})
        eval_chunks.append({"expert_id": dom_name, "ids": toks_d})

    return {
        "model": model,
        "probe_chunks": probe_chunks,
        "eval_chunks": eval_chunks,
        "domain_specs": domain_specs,
    }


def _eval_crowd_mean_and_domains(
    model: CrowdV1WithEngram,
    chunks: list[dict[str, Any]],
    *,
    score_from: int = 4,
) -> tuple[float, dict[str, float]]:
    """Compute mean NLL and per-domain NLL on CPU."""
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
    return float(nlls.mean()), {k: float(np.mean(v)) for k, v in sorted(by_dom.items())}


# ---------------------------------------------------------------------------
# Scenario S1: ZK-WSP Spot-Check Completeness, Soundness & Tamper Detection
# ---------------------------------------------------------------------------


def test_s1_zkwsp_completeness_soundness_and_fiat_shamir_tamper_detection() -> None:
    """(S1) Prove RCD_D5 P_0 completeness, Byzantine soundness, and Fiat-Shamir tamper rejection."""
    v0_x = _make_ortho_basis(16, 4, seed=20264710)
    v0_act = _make_ortho_basis(24, 5, seed=20264711)
    p0_x = torch.eye(16, dtype=torch.float64) - v0_x @ v0_x.T
    p0_act = torch.eye(24, dtype=torch.float64) - v0_act @ v0_act.T

    # Build commitment from raw bases AND verify key-derived SVD commitment works cleanly
    commitment = build_prose_subspace_commitment(
        {"blocks.0.x": v0_x, "blocks.0.act": v0_act, "default": v0_x},
        commitment_id="s1_layered_v0",
    )
    assert commitment["object"] == ZKWSP_COMMITMENT_OBJECT
    assert commitment["max_orthonormality_error"] < 1e-12

    keys_svd = build_prose_subspace_commitment_from_keys(
        {"default": (v0_x @ torch.randn((4, 12), generator=torch.Generator().manual_seed(11), dtype=torch.float64)).T},
        sv_threshold=1e-4,
        max_active_modes=4,
    )
    assert keys_svd["basis_shapes"]["default"] == [16, 4]

    # Low-SV edge branch where no singular value exceeds sv_threshold
    keys_low = build_prose_subspace_commitment_from_keys(1e-6 * torch.ones((4, 8)), sv_threshold=1.0)
    assert keys_low["basis_shapes"]["default"] == [8, 1]

    # Honest delta strictly inside P_0 covering right-match 2D, LoRA w1_B skip, w3_A act-match,
    # transposed 2D matrix, 1D vector, protected LN & protected E0 expert
    g_d = torch.Generator(device="cpu").manual_seed(20264712)
    honest_delta = {
        "blocks.0.mlp.experts.1.w1_A": (torch.randn((2, 16), generator=g_d, dtype=torch.float64) @ p0_x).float(),
        "blocks.0.mlp.experts.1.w1_B": torch.randn((24, 2), generator=g_d).float(),
        "blocks.0.mlp.experts.1.w3_A": (torch.randn((2, 24), generator=g_d, dtype=torch.float64) @ p0_act).float(),
        "blocks.0.mlp.experts.0.w1_A": torch.ones((2, 16)),  # protected when protect_general_expert=True
        "blocks.0.mlp.router.weight": (torch.randn((4, 16), generator=g_d, dtype=torch.float64) @ p0_x).float(),
        "blocks.0.transposed": (p0_x @ torch.randn((16, 6), generator=g_d, dtype=torch.float64)).float(),
        "blocks.0.bias_vec": (torch.randn(16, generator=g_d, dtype=torch.float64) @ p0_x).float(),
        "ln.weight": torch.ones(16),  # protected layernorm
    }
    witness = compute_zkwsp_spotcheck_witness(
        "hive_honest_1",
        honest_delta,
        commitment,
        round_id="round_47",
        protect_layernorm=True,
        protect_general_expert=True,
        general_expert_index=0,
    )
    assert witness["passes_zkwsp_spotcheck"] is True
    assert witness["claimed_max_abs_basis_leak"] < 1e-6
    assert witness["claimed_max_abs_challenge_leak"] <= witness["claimed_max_abs_basis_leak"] + 1e-12
    assert "ln.weight" not in witness["spotchecked_keys"]
    assert "blocks.0.mlp.experts.1.w1_B" not in witness["spotchecked_keys"]
    assert "blocks.0.mlp.experts.0.w1_A" not in witness["spotchecked_keys"]

    verif_ok = verify_zkwsp_spotcheck_witness(
        witness,
        hive_id="hive_honest_1",
        delta=honest_delta,
        subspace_commitment=commitment,
        round_id="round_47",
        protect_layernorm=True,
        protect_general_expert=True,
        general_expert_index=0,
    )
    assert verif_ok["valid"] is True
    assert verif_ok["failures"] == ()

    # Byzantine prose poisoner aligned with V_0 -> RCD_D5 soundness violation!
    poison_delta = copy.deepcopy(honest_delta)
    poison_delta["blocks.0.mlp.experts.1.w1_A"] = (0.35 * torch.ones((2, 4), dtype=torch.float64) @ v0_x.T).float()
    poison_witness = compute_zkwsp_spotcheck_witness(
        "hive_byz_poison",
        poison_delta,
        commitment,
        round_id="round_47",
        protect_general_expert=True,
    )
    assert poison_witness["passes_zkwsp_spotcheck"] is False
    assert poison_witness["claimed_max_abs_basis_leak"] > 0.1
    verif_poison = verify_zkwsp_spotcheck_witness(
        poison_witness,
        hive_id="hive_byz_poison",
        delta=poison_delta,
        subspace_commitment=commitment,
        round_id="round_47",
        protect_general_expert=True,
    )
    assert verif_poison["valid"] is False
    assert "zkwsp_prose_nullspace_violation" in verif_poison["failures"]

    # Tamper detection: delta mutation, witness tamper, challenge digest tamper, round/hive/basis mismatch
    mutated_delta = copy.deepcopy(honest_delta)
    mutated_delta["blocks.0.mlp.experts.1.w1_A"] = (
        mutated_delta["blocks.0.mlp.experts.1.w1_A"].double() + 0.01 * (torch.ones((2, 16), dtype=torch.float64) @ p0_x)
    ).float()
    v_mut = verify_zkwsp_spotcheck_witness(
        witness,
        hive_id="hive_honest_1",
        delta=mutated_delta,
        subspace_commitment=commitment,
        round_id="round_47",
        protect_general_expert=True,
    )
    assert v_mut["valid"] is False
    assert "delta_sha256_mismatch" in v_mut["failures"]

    tampered_w = dict(witness)
    tampered_w["claimed_max_abs_basis_leak"] = 9.0
    v_tamper = verify_zkwsp_spotcheck_witness(
        tampered_w,
        hive_id="hive_honest_1",
        delta=honest_delta,
        subspace_commitment=commitment,
        round_id="round_47",
        protect_general_expert=True,
    )
    assert v_tamper["valid"] is False
    assert "zkwsp_witness_tampered" in v_tamper["failures"]
    assert "zkwsp_claimed_leak_mismatch" in v_tamper["failures"]

    tampered_chal = dict(witness)
    tampered_chal["challenge_response_digests"] = ["0" * 64] * len(witness["challenge_response_digests"])
    v_chal = verify_zkwsp_spotcheck_witness(
        tampered_chal,
        hive_id="hive_honest_1",
        delta=honest_delta,
        subspace_commitment=commitment,
        round_id="round_47",
        protect_general_expert=True,
    )
    assert "zkwsp_challenge_response_mismatch" in v_chal["failures"]

    v_replay = verify_zkwsp_spotcheck_witness(
        witness,
        hive_id="hive_impostor",
        delta=honest_delta,
        subspace_commitment=commitment,
        round_id="round_99",
        protect_general_expert=True,
    )
    assert "zkwsp_hive_id_mismatch" in v_replay["failures"]
    assert "zkwsp_round_id_mismatch" in v_replay["failures"]
    assert "zkwsp_challenge_seed_mismatch" in v_replay["failures"]


# ---------------------------------------------------------------------------
# Scenario S2: Lean 4 Attestation Verification & Cross-Binding Forgery Defense
# ---------------------------------------------------------------------------


def test_s2_lean4_attestation_verification_and_cross_binding_forgery_rejection(
    cached_lean_audit: Any,
) -> None:
    """(S2) Bind packets to real Lean 4 HiveMindSpec.lean receipts and reject all forgeries."""
    lean_audit = cached_lean_audit
    assert lean_audit.passed is True
    assert len(lean_audit.season35_theorems_declared) == 5

    v0 = _make_ortho_basis(16, 4, seed=20264720)
    p0 = torch.eye(16, dtype=torch.float64) - v0 @ v0.T
    commitment = build_prose_subspace_commitment(v0, commitment_id="s2_v0")

    g = torch.Generator(device="cpu").manual_seed(20264721)
    d_honest = (0.05 * torch.randn((12, 16), generator=g, dtype=torch.float64) @ p0).float()
    pkt = build_federated_hive_packet(
        "hive_alpha",
        d_honest,
        commitment,
        lean_spec_audit=lean_audit,
        round_id="round_47",
        domain_label="code_python",
        target_key="head.weight",
    )
    assert pkt["object"] == ZKWSP_PACKET_OBJECT
    assert pkt["schema"] == ZKWSP_VERIFIER_SCHEMA
    assert isinstance(pkt["delta"], torch.Tensor)
    assert pkt["s5_receipt"]["lean4_rcd_d5_theorem"] == LEAN4_RCD_D5_THEOREM

    ok_verif = verify_federated_hive_packet(
        pkt,
        commitment,
        lean_spec_audit=lean_audit,
        expected_round_id="round_47",
    )
    assert ok_verif["accepted"] is True
    assert ok_verif["lean4_verified"] is True
    assert ok_verif["measured_max_abs_basis_leak"] < 1e-6

    # 1. Forged attestation_sha256
    forged_sha = copy.deepcopy(pkt)
    forged_sha["lean4_attestation"]["attestation_sha256"] = "f" * 64
    r_forged = verify_federated_hive_packet(forged_sha, commitment, lean_spec_audit=lean_audit)
    assert r_forged["accepted"] is False
    assert r_forged["lean4_verified"] is False
    assert any(x.startswith("lean4_attestation_failed:") for x in r_forged["rejection_reasons"])

    # 2. Cross-binding replay attack: attach valid Lean 4 receipt from another delta / hive / round
    d_trojan = (0.25 * torch.randn((12, 16), generator=g, dtype=torch.float64) @ p0).float()
    pkt_trojan = build_federated_hive_packet(
        "hive_trojan",
        d_trojan,
        commitment,
        lean_spec_audit=lean_audit,
        round_id="round_47",
        target_key="head.weight",
    )
    replay_pkt = copy.deepcopy(pkt_trojan)
    replay_pkt["s5_receipt"] = copy.deepcopy(pkt["s5_receipt"])
    replay_pkt["lean4_attestation"] = copy.deepcopy(pkt["lean4_attestation"])
    r_replay = verify_federated_hive_packet(
        replay_pkt, commitment, lean_spec_audit=lean_audit, expected_round_id="round_47"
    )
    assert r_replay["accepted"] is False
    assert "cross_binding_hive_id_mismatch" in r_replay["rejection_reasons"]
    assert "cross_binding_delta_sha256_mismatch" in r_replay["rejection_reasons"]
    assert "cross_binding_witness_sha256_mismatch" in r_replay["rejection_reasons"]

    # 3. Missing Lean 4 attestation or malformed s5_receipt
    no_att = copy.deepcopy(pkt)
    no_att["lean4_attestation"] = None
    r_missing = verify_federated_hive_packet(no_att, commitment, lean_spec_audit=lean_audit)
    assert "missing_lean4_attestation_or_s5_receipt" in r_missing["rejection_reasons"]

    bad_att = copy.deepcopy(pkt)
    bad_att["lean4_attestation"] = {"attestation_sha256": 123}
    r_bad_att = verify_federated_hive_packet(bad_att, commitment, lean_spec_audit=lean_audit)
    assert any(x.startswith("lean4_attestation_failed:") for x in r_bad_att["rejection_reasons"])

    # 4. Also reject replay against wrong subspace commitment
    v0_other = _make_ortho_basis(16, 4, seed=20264729)
    other_comm = build_prose_subspace_commitment(v0_other, commitment_id="other_v0")
    r_wrong_v0 = verify_federated_hive_packet(pkt, other_comm, lean_spec_audit=lean_audit)
    assert "subspace_commitment_mismatch" in r_wrong_v0["rejection_reasons"]
    assert "cross_binding_v0_sha256_mismatch" in r_wrong_v0["rejection_reasons"]

    # 5. Round ID mismatch and non-finite delta rejection
    r_wrong_rid = verify_federated_hive_packet(
        pkt, commitment, lean_spec_audit=lean_audit, expected_round_id="round_99"
    )
    assert "round_id_mismatch" in r_wrong_rid["rejection_reasons"]

    bad_delta_pkt = copy.deepcopy(pkt)
    bad_delta_pkt["delta"] = torch.full((12, 16), float("nan"))
    r_nan = verify_federated_hive_packet(bad_delta_pkt, commitment, lean_spec_audit=lean_audit)
    assert r_nan["accepted"] is False
    assert any(x.startswith("invalid_or_non_finite_delta:") for x in r_nan["rejection_reasons"])


# ---------------------------------------------------------------------------
# Scenario S3: Multi-Krum Byzantine Multi-Hive Filtration & P_0 Consensus
# ---------------------------------------------------------------------------


def test_s3_multikrum_byzantine_multi_hive_filtration_and_p0_consensus(
    cached_lean_audit: Any,
) -> None:
    """(S3) Filter crypto/ZK-WSP poisoners + stealth P_0 outliers via Multi-Krum under n >= 2f + 3."""
    lean_audit = cached_lean_audit
    v0 = _make_ortho_basis(20, 4, seed=20264730)
    p0 = torch.eye(20, dtype=torch.float64) - v0 @ v0.T
    commitment = build_prose_subspace_commitment(v0, commitment_id="s3_v0")

    gen = torch.Generator(device="cpu").manual_seed(20264731)
    base_dir = (torch.randn((10, 20), generator=gen, dtype=torch.float64) @ p0) * 0.08

    # Cohort A: 5 honest hives + 1 ZK-WSP prose poisoner + 1 forged-attestation Trojan (7 hives, f=2)
    cohort_a: list[dict[str, Any]] = []
    for idx in range(5):
        noise = (torch.randn((10, 20), generator=gen, dtype=torch.float64) @ p0) * 0.005
        cohort_a.append(
            build_federated_hive_packet(
                f"hive_honest_{idx}",
                (base_dir + noise).float(),
                commitment,
                lean_spec_audit=lean_audit,
                round_id="round_a",
            )
        )
    poison_mat = (0.4 * torch.randn((10, 4), generator=gen, dtype=torch.float64) @ v0.T).float()
    cohort_a.append(
        build_federated_hive_packet(
            "hive_byz_prose_poisoner",
            poison_mat,
            commitment,
            lean_spec_audit=lean_audit,
            round_id="round_a",
        )
    )
    trojan_pkt = build_federated_hive_packet(
        "hive_byz_forged_trojan",
        (0.9 * torch.randn((10, 20), generator=gen, dtype=torch.float64)).float(),
        commitment,
        lean_spec_audit=lean_audit,
        round_id="round_a",
    )
    trojan_pkt["lean4_attestation"]["attestation_sha256"] = "a" * 64
    cohort_a.append(trojan_pkt)

    agg_a, audit_a = verify_and_aggregate_federated_hives(
        cohort_a,
        commitment,
        lean_spec_audit=lean_audit,
        expected_round_id="round_a",
        config=ZKWSPFederatedHiveConfig(num_byzantine_f=2),
    )
    assert agg_a is not None
    assert audit_a["quorum_satisfied"] is True
    assert audit_a["num_rejected_by_verifier"] == 2
    assert len(audit_a["zkwsp_accepted_hives"]) == 5
    assert audit_a["consensus_max_abs_basis_leak"] < 1e-7
    assert float((agg_a["delta"].double() @ v0).abs().max().item()) < 1e-7

    # Cohort B: 5 honest hives + 2 STEALTH Byzantine outliers INSIDE P_0 (pass ZK-WSP + Lean 4!)
    # Multi-Krum with n=7, f=2 must isolate and drop BOTH stealth outliers!
    cohort_b = list(cohort_a[:5])
    for b_idx in range(2):
        stealth = (-8.0 * base_dir + (0.05 * b_idx) * (torch.randn((10, 20), generator=gen, dtype=torch.float64) @ p0)).float()
        cohort_b.append(
            build_federated_hive_packet(
                f"hive_byz_stealth_{b_idx}",
                stealth,
                commitment,
                lean_spec_audit=lean_audit,
                round_id="round_b",
            )
        )
    # Build matching round_b honest packets
    cohort_b_full = [
        build_federated_hive_packet(
            f"hive_honest_{idx}",
            cohort_a[idx]["delta"],
            commitment,
            lean_spec_audit=lean_audit,
            round_id="round_b",
        )
        for idx in range(5)
    ] + cohort_b[5:]

    agg_b, audit_b = verify_and_aggregate_federated_hives(
        cohort_b_full,
        commitment,
        lean_spec_audit=lean_audit,
        expected_round_id="round_b",
        config=ZKWSPFederatedHiveConfig(num_byzantine_f=2, select_m=5),
    )
    assert agg_b is not None
    assert audit_b["num_verified"] == 7
    assert set(audit_b["multikrum_rejected_hives"]) == {"hive_byz_stealth_0", "hive_byz_stealth_1"}
    assert set(audit_b["multikrum_selected_hives"]) == {f"hive_honest_{i}" for i in range(5)}
    assert float((agg_b["delta"].double() * base_dir).sum().item()) > 0.0
    assert audit_b["consensus_max_abs_basis_leak"] < 1e-7

    # Cohort C: Optional pre-aggregation spectral/sparsity screening catches rank-1 Trojan spike in P_0
    u_left = torch.randn((10, 1), generator=gen, dtype=torch.float64)
    v_right = torch.randn((1, 20), generator=gen, dtype=torch.float64) @ p0
    rank1_trojan = (6.0 * (u_left / u_left.norm()) @ (v_right / v_right.norm())).float()
    cohort_c = list(cohort_b_full[:5]) + [
        build_federated_hive_packet(
            "hive_byz_rank1_spike",
            rank1_trojan,
            commitment,
            lean_spec_audit=lean_audit,
            round_id="round_b",
        )
    ]
    agg_c, audit_c = verify_and_aggregate_federated_hives(
        cohort_c,
        commitment,
        lean_spec_audit=lean_audit,
        expected_round_id="round_b",
        config=ZKWSPFederatedHiveConfig(
            num_byzantine_f=1,
            enable_spectral_screening=True,
            max_edit_fraction=1.0,
            max_spectral_spike_ratio=3.0,
        ),
    )
    assert agg_c is not None
    assert audit_c["screening_report"] is not None
    assert "hive_byz_rank1_spike" in audit_c["screening_report"]["rejected_peer_ids"]

    # Cohort D: Quorum breach when more hives fail ZK-WSP than Byzantine tolerance f
    cohort_d = list(cohort_a[:3]) + [copy.deepcopy(cohort_a[5]), copy.deepcopy(cohort_a[6])]
    agg_d, audit_d = verify_and_aggregate_federated_hives(
        cohort_d,
        commitment,
        lean_spec_audit=lean_audit,
        expected_round_id="round_a",
        config=ZKWSPFederatedHiveConfig(num_byzantine_f=1),
    )
    assert agg_d is None
    assert audit_d["quorum_satisfied"] is False

    # Enforce n >= 2f + 3 bound strictly, plus non-enforced fallback for small cohort
    with pytest.raises(ValueError, match="2\\*f \\+ 3"):
        verify_and_aggregate_federated_hives(
            cohort_a[:4], commitment, config=ZKWSPFederatedHiveConfig(num_byzantine_f=1)
        )
    agg_small, audit_small = verify_and_aggregate_federated_hives(
        cohort_a[:2],
        commitment,
        lean_spec_audit=lean_audit,
        expected_round_id="round_a",
        config=ZKWSPFederatedHiveConfig(num_byzantine_f=0),
        enforce_byzantine_bound=False,
    )
    assert agg_small is not None
    assert audit_small["krum_report"]["num_peers"] == 2


# ---------------------------------------------------------------------------
# Scenario S4: End-to-End ANSE-Gated Multi-Hive Round & Atomic Rollback
# ---------------------------------------------------------------------------


def test_s4_end_to_end_anse_gated_hive_round_and_atomic_bitwise_rollback(
    cached_lean_audit: Any,
) -> None:
    """(S4) Prove CrowdV1 MoE ANSE Delta E < 0, D0 <= +1%, and atomic bitwise rollback on breaches."""
    fix = _build_federated_crowd_moe_fixture(seed=20264740)
    base_model: CrowdV1WithEngram = fix["model"]
    probe_chunks = fix["probe_chunks"]
    eval_chunks = fix["eval_chunks"]
    domain_specs = fix["domain_specs"]
    lean_audit = cached_lean_audit

    # Extract V_0 from probe chunks directly via collect_prose_subspace_from_probe
    probe_comm = collect_prose_subspace_from_probe(
        base_model,
        probe_chunks,
        sv_threshold=1e-3,
        max_active_modes=8,
        score_from=4,
    )
    v0_head = probe_comm["v0_bases"]["head.weight"]
    p0_head = probe_comm["p0_projectors"]["head.weight"]

    # Build 5 honest regional specialist Hive head.weight deltas + 2 Byzantine Hive deltas
    base_model.eval()
    ids_probe = torch.tensor([c["ids"] for c in probe_chunks], dtype=torch.long)
    with torch.no_grad():
        h_post_ln = base_model.ln(forward_model_pre_ln(base_model, ids_probe, batch_size=16)[:, 3:-1, :].float())
    hw_base = base_model.head.weight.detach().float()

    packets_7: list[dict[str, Any]] = []
    for _slot_i, dom_name, _, _, _ in domain_specs[1:]:
        idx_dom = [i for i, c in enumerate(probe_chunks) if c["expert_id"] == dom_name]
        x_dom = h_post_ln[idx_dom].reshape(-1, hw_base.shape[1])
        t_dom = ids_probe[idx_dom, 4:].reshape(-1)
        logits = F.linear(x_dom, hw_base)
        probs = F.softmax(logits, dim=-1)
        probs.scatter_add_(-1, t_dom.unsqueeze(-1), torch.full((t_dom.numel(), 1), -1.0))
        grad_dom = (probs.T @ x_dom) / float(t_dom.numel())
        delta_honest = ((-2.2 * grad_dom) @ p0_head.float()).contiguous()
        packets_7.append(
            build_federated_hive_packet(
                f"hive_{dom_name}",
                delta_honest,
                probe_comm,
                lean_spec_audit=lean_audit,
                round_id="ruche_r47",
                domain_label=dom_name,
                target_key="head.weight",
            )
        )

    # Adversary 1: prose-poisoner ascending E0 gradient inside V_0
    idx_e0 = [i for i, c in enumerate(probe_chunks) if c["expert_id"] == "E0_general_prose"]
    x_e0 = h_post_ln[idx_e0].reshape(-1, hw_base.shape[1])
    t_e0 = ids_probe[idx_e0, 4:].reshape(-1)
    lg_e0 = F.linear(x_e0, hw_base)
    pr_e0 = F.softmax(lg_e0, dim=-1)
    pr_e0.scatter_add_(-1, t_e0.unsqueeze(-1), torch.full((t_e0.numel(), 1), -1.0))
    grad_e0 = (pr_e0.T @ x_e0) / float(t_e0.numel())
    dh_poison = (3.5 * grad_e0 @ (v0_head.float() @ v0_head.float().T)).contiguous()
    packets_7.append(
        build_federated_hive_packet(
            "hive_byz_prose_poisoner",
            dh_poison,
            probe_comm,
            lean_spec_audit=lean_audit,
            round_id="ruche_r47",
            target_key="head.weight",
        )
    )

    # Adversary 2: forged Lean 4 attestation Trojan
    g_tr = torch.Generator(device="cpu").manual_seed(20264741)
    pkt_forged = build_federated_hive_packet(
        "hive_byz_forged_trojan",
        (3.0 * torch.randn_like(hw_base, generator=g_tr)),
        probe_comm,
        lean_spec_audit=lean_audit,
        round_id="ruche_r47",
        target_key="head.weight",
    )
    pkt_forged["lean4_attestation"]["attestation_sha256"] = "e" * 64
    packets_7.append(pkt_forged)

    base_mean, base_dom = _eval_crowd_mean_and_domains(base_model, eval_chunks, score_from=4)
    cand_model, receipt = execute_verified_zkwsp_federated_hive_round(
        base_model,
        packets_7,
        subspace_commitment=probe_comm,
        eval_chunks=eval_chunks,
        lean_spec_audit=lean_audit,
        round_id="ruche_r47",
        config=ZKWSPFederatedHiveConfig(
            num_byzantine_f=2,
            consensus_step_scale=1.0,
            enforce_all_domain_guard=True,
            score_from=4,
            inplace=False,
        ),
    )
    assert receipt["status"] == "ACCEPTED"
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["object"] == ZKWSP_ROUND_RECEIPT_OBJECT
    assert receipt["schema"] == ZKWSP_VERIFIER_SCHEMA
    assert verify_zkwsp_federated_round_receipt_provenance(receipt) is True
    assert receipt["delta_e"] < 0.0
    assert receipt["general_drift_pct"] <= 1.0
    assert receipt["worst_domain_drift_pct"] <= 1.0
    assert receipt["zkwsp_audit"]["num_rejected_by_verifier"] == 2
    assert receipt["parent_state_sha256"] != receipt["final_state_sha256"]
    assert cand_model is not base_model

    cand_mean, cand_dom = _eval_crowd_mean_and_domains(cand_model, eval_chunks, score_from=4)
    assert cand_mean < base_mean
    assert cand_dom["E0_general_prose"] <= base_dom["E0_general_prose"] * 1.01

    # Tampered receipt detection
    tampered_r = dict(receipt)
    tampered_r["status"] = "FORGED"
    assert verify_zkwsp_federated_round_receipt_provenance(tampered_r) is False

    # Also test automatic V_0 extraction from probe_chunks AND prose_keys directly
    _, rec_from_probe = execute_verified_zkwsp_federated_hive_round(
        base_model,
        packets_7,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        lean_spec_audit=lean_audit,
        round_id="ruche_r47",
        config=ZKWSPFederatedHiveConfig(num_byzantine_f=2, sv_threshold=1e-3, prose_max_active_modes=8, score_from=4),
    )
    assert rec_from_probe["status"] == "ACCEPTED"

    # Rollback paths under ANSE_MAXIMUM_PAIN (inplace=True verifies bitwise restore on base_model!)
    base_snapshot = copy.deepcopy(base_model.state_dict())
    base_sha = compute_state_sha256(base_snapshot)

    def _assert_bitwise_restored(m: nn.Module) -> None:
        for k, v0 in base_snapshot.items():
            assert torch.equal(m.state_dict()[k].cpu(), v0.cpu())
        assert compute_state_sha256(m) == base_sha

    # 1. D0 > +1.0% guard breach -> ROLLED_BACK_GUARD_VIOLATION
    g_calls = {"n": 0}

    def _guard_breach_eval(_m: nn.Module) -> dict[str, Any]:
        g_calls["n"] += 1
        return {"E0_general_prose": 2.00, "E1_code": 2.40} if g_calls["n"] == 1 else {"E0_general_prose": 2.04, "E1_code": 1.60}

    _, rec_guard = execute_verified_zkwsp_federated_hive_round(
        base_model,
        packets_7,
        subspace_commitment=probe_comm,
        eval_fn=_guard_breach_eval,
        lean_spec_audit=lean_audit,
        round_id="ruche_r47",
        config=ZKWSPFederatedHiveConfig(num_byzantine_f=2, inplace=True),
    )
    assert rec_guard["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert rec_guard["rolled_back"] is True
    assert rec_guard["rollback_bitwise_verified"] is True
    assert rec_guard["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert rec_guard["final_state_sha256"] == base_sha
    _assert_bitwise_restored(base_model)

    # 2. Non-improving ANSE energy -> ROLLED_BACK_ENERGY_NON_IMPROVING
    _, rec_nonimp = execute_verified_zkwsp_federated_hive_round(
        base_model,
        packets_7,
        subspace_commitment=probe_comm,
        eval_fn=lambda _m: {"E0_general_prose": 2.00, "E1_code": 2.20},
        lean_spec_audit=lean_audit,
        round_id="ruche_r47",
        config=ZKWSPFederatedHiveConfig(num_byzantine_f=2, inplace=True),
    )
    assert rec_nonimp["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    assert rec_nonimp["final_state_sha256"] == base_sha
    _assert_bitwise_restored(base_model)

    # 3. Soft margin penalty path (E0 drift in (0.85%, 1.0%]) causing energy non-improvement
    sm_calls = {"n": 0}

    def _soft_margin_eval(_m: nn.Module) -> dict[str, Any]:
        sm_calls["n"] += 1
        return {"E0_general_prose": 2.00, "E1_code": 2.20} if sm_calls["n"] == 1 else {"E0_general_prose": 2.018, "E1_code": 2.15}

    _, rec_soft = execute_verified_zkwsp_federated_hive_round(
        base_model,
        packets_7,
        subspace_commitment=probe_comm,
        eval_fn=_soft_margin_eval,
        lean_spec_audit=lean_audit,
        round_id="ruche_r47",
        config=ZKWSPFederatedHiveConfig(num_byzantine_f=2, inplace=False),
    )
    assert rec_soft["guard_ok"] is True
    assert rec_soft["candidate_anse_energy"] > rec_soft["candidate_nll_mean"]
    assert rec_soft["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"

    # 4. Non-finite candidate eval & eval exception -> ROLLED_BACK_NON_FINITE & ROLLED_BACK_EVAL_FAILURE
    nf_calls = {"n": 0}

    def _nf_eval(_m: nn.Module) -> dict[str, Any]:
        nf_calls["n"] += 1
        return {"E0_general_prose": 2.00, "E1_code": 2.20} if nf_calls["n"] == 1 else {"E0_general_prose": float("nan"), "E1_code": 1.80}

    _, rec_nf = execute_verified_zkwsp_federated_hive_round(
        base_model,
        packets_7,
        subspace_commitment=probe_comm,
        eval_fn=_nf_eval,
        lean_spec_audit=lean_audit,
        round_id="ruche_r47",
        config=ZKWSPFederatedHiveConfig(num_byzantine_f=2),
    )
    assert rec_nf["status"] == "ROLLED_BACK_NON_FINITE"
    assert rec_nf["anse_energy"] == ANSE_MAXIMUM_PAIN

    ex_calls = {"n": 0}

    def _boom_eval(_m: nn.Module) -> dict[str, Any]:
        ex_calls["n"] += 1
        if ex_calls["n"] == 1:
            return {"E0_general_prose": 2.00, "E1_code": 2.20}
        raise RuntimeError("simulated NaN logits")

    _, rec_boom = execute_verified_zkwsp_federated_hive_round(
        base_model,
        packets_7,
        subspace_commitment=probe_comm,
        eval_fn=_boom_eval,
        lean_spec_audit=lean_audit,
        round_id="ruche_r47",
        config=ZKWSPFederatedHiveConfig(num_byzantine_f=2, inplace=True),
    )
    assert rec_boom["status"] == "ROLLED_BACK_EVAL_FAILURE"
    assert "simulated NaN logits" in str(rec_boom["eval_error"])
    _assert_bitwise_restored(base_model)

    # 5. All hives rejected & quorum breach below Byzantine bound inside round execution
    all_bad = [copy.deepcopy(packets_7[5]) for _ in range(5)]
    for idx, p in enumerate(all_bad):
        p["hive_id"] = f"bad_{idx}"
    _, rec_all_bad = execute_verified_zkwsp_federated_hive_round(
        base_model,
        all_bad,
        subspace_commitment=probe_comm,
        eval_fn=lambda _m: {"E0_general_prose": 2.00},
        lean_spec_audit=lean_audit,
        round_id="ruche_r47",
        config=ZKWSPFederatedHiveConfig(num_byzantine_f=1),
    )
    assert rec_all_bad["status"] == "ROLLED_BACK_ALL_HIVES_REJECTED"
    assert rec_all_bad["anse_energy"] == ANSE_MAXIMUM_PAIN

    quorum_bad = [copy.deepcopy(p) for p in packets_7[:3]] + [copy.deepcopy(packets_7[5]), copy.deepcopy(packets_7[6])]
    _, rec_qbad = execute_verified_zkwsp_federated_hive_round(
        base_model,
        quorum_bad,
        subspace_commitment=probe_comm,
        eval_fn=lambda _m: {"E0_general_prose": 2.00},
        lean_spec_audit=lean_audit,
        round_id="ruche_r47",
        config=ZKWSPFederatedHiveConfig(num_byzantine_f=1),
    )
    assert rec_qbad["status"] == "ROLLED_BACK_QUORUM_BELOW_BYZANTINE_BOUND"

    # 6. Fixed-point unchanged when zero deltas are submitted (using prose_keys V_0 extraction)
    prose_head_keys = h_post_ln[idx_e0].reshape(-1, hw_base.shape[1])
    keys_comm = build_prose_subspace_commitment_from_keys({"head.weight": prose_head_keys})
    zero_pkts = [
        build_federated_hive_packet(
            f"zero_{i}",
            torch.zeros_like(hw_base),
            keys_comm,
            lean_spec_audit=lean_audit,
            round_id="ruche_r47",
            target_key="head.weight",
        )
        for i in range(5)
    ]
    _, rec_zero = execute_verified_zkwsp_federated_hive_round(
        base_model,
        zero_pkts,
        prose_keys={"head.weight": prose_head_keys},
        eval_fn=lambda _m: {"E0_general_prose": 2.00, "E1_code": 2.20},
        lean_spec_audit=lean_audit,
        round_id="ruche_r47",
        config=ZKWSPFederatedHiveConfig(num_byzantine_f=1),
    )
    assert rec_zero["status"] == "FIXED_POINT_UNCHANGED"


# ---------------------------------------------------------------------------
# Scenario S5: Validation Gates & Security / Zero-Stub / OwnerOrder Compliance
# ---------------------------------------------------------------------------


def test_s5_validation_gates_docs_and_security_compliance(
    cached_lean_audit: Any,
) -> None:
    """(S5) Assert input validation gates, zero stubs, zero egress, doc sync & OwnerOrderGuardrail."""
    # 1. Config & helper validation errors
    with pytest.raises(ValueError, match="select_m"):
        ZKWSPFederatedHiveConfig(select_m=0)
    with pytest.raises(ValueError, match="domain_guard_max_ratio"):
        ZKWSPFederatedHiveConfig(domain_guard_max_ratio=0.9)
    with pytest.raises(ValueError, match="anse_safe_general_ratio"):
        ZKWSPFederatedHiveConfig(anse_safe_general_ratio=0.95)
    with pytest.raises(ValueError, match="general_prefixes"):
        ZKWSPFederatedHiveConfig(general_prefixes=())
    with pytest.raises(ValueError, match="prose_variance_threshold"):
        ZKWSPFederatedHiveConfig(prose_variance_threshold=1.5)
    with pytest.raises(ValueError, match="prose_max_active_modes"):
        ZKWSPFederatedHiveConfig(prose_max_active_modes=0)

    with pytest.raises(TypeError, match="torch.Tensor"):
        build_prose_subspace_commitment("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.Tensor"):
        build_prose_subspace_commitment({"k": "bad"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-empty 2D"):
        build_prose_subspace_commitment(torch.ones(4))
    with pytest.raises(TypeError, match="floating-point"):
        build_prose_subspace_commitment(torch.ones((4, 2), dtype=torch.long))
    with pytest.raises(ValueError, match="cannot exceed"):
        build_prose_subspace_commitment(torch.ones((2, 4)))
    with pytest.raises(ValueError, match="finite"):
        build_prose_subspace_commitment(torch.tensor([[1.0, 0.0], [float("nan"), 1.0]]))
    with pytest.raises(ValueError, match="orthonormal"):
        build_prose_subspace_commitment(2.0 * torch.eye(4))

    with pytest.raises(TypeError, match="Tensor"):
        build_prose_subspace_commitment_from_keys("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2D or 3D"):
        build_prose_subspace_commitment_from_keys(torch.ones(4))
    with pytest.raises(TypeError, match="floating-point"):
        build_prose_subspace_commitment_from_keys(torch.ones((4, 4), dtype=torch.long))
    with pytest.raises(ValueError, match="finite"):
        build_prose_subspace_commitment_from_keys(torch.full((4, 4), float("inf")))
    with pytest.raises(TypeError, match="torch.Tensor"):
        build_prose_subspace_commitment_from_keys({"k": "bad"})  # type: ignore[dict-item]

    with pytest.raises(TypeError, match="nn.Module"):
        collect_prose_subspace_from_probe("bad", [])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty sequence"):
        collect_prose_subspace_from_probe(nn.Linear(4, 4), [])
    with pytest.raises(ValueError, match="blocks"):
        collect_prose_subspace_from_probe(nn.Linear(4, 4), [{"domain": "E0", "ids": [0, 1, 2, 3]}])

    v0 = _make_ortho_basis(8, 2, seed=20264750)
    comm = build_prose_subspace_commitment(v0)
    bad_comm = dict(comm)
    bad_comm["v0_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="v0_sha256"):
        compute_zkwsp_spotcheck_witness("h1", torch.zeros((4, 8)), bad_comm)
    missing_v0 = dict(comm)
    missing_v0["v0_bases"] = {}
    with pytest.raises(ValueError, match="v0_bases"):
        compute_zkwsp_spotcheck_witness("h1", torch.zeros((4, 8)), missing_v0)

    with pytest.raises(ValueError, match="control characters"):
        compute_zkwsp_spotcheck_witness("bad\nhive", torch.zeros((4, 8)), comm)
    with pytest.raises(ValueError, match="non-empty string"):
        compute_zkwsp_spotcheck_witness("   ", torch.zeros((4, 8)), comm)
    with pytest.raises(TypeError, match="str"):
        compute_zkwsp_spotcheck_witness(123, torch.zeros((4, 8)), comm)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="delta must be"):
        compute_zkwsp_spotcheck_witness("h1", "bad", comm)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        compute_zkwsp_spotcheck_witness("h1", torch.empty((0, 8)), comm)
    with pytest.raises(ValueError, match="non-empty floating"):
        compute_zkwsp_spotcheck_witness("h1", {"a": torch.ones((4, 8), dtype=torch.long)}, comm)
    with pytest.raises(ValueError, match="No delta parameter"):
        compute_zkwsp_spotcheck_witness("h1", torch.zeros((5, 5)), comm)
    with pytest.raises(ValueError, match="Incompatible basis shape"):
        compute_zkwsp_spotcheck_witness("h1", {"default": torch.zeros((5, 5))}, comm)

    with pytest.raises(TypeError, match="Mapping"):
        verify_zkwsp_spotcheck_witness("bad", hive_id="h1", delta=torch.zeros((4, 8)), subspace_commitment=comm)  # type: ignore[arg-type]
    bad_w = {"object": "wrong", "witness_sha256": "xyz", "num_challenges": "bad", "bad_nan": float("nan")}
    res_bad_w = verify_zkwsp_spotcheck_witness(bad_w, hive_id="h1", delta=torch.zeros((4, 8)), subspace_commitment=comm)
    assert "invalid_witness_schema_or_object" in res_bad_w["failures"]
    assert "invalid_witness_sha256_format" in res_bad_w["failures"]
    assert "invalid_num_challenges" in res_bad_w["failures"]
    bad_w_nan = dict(compute_zkwsp_spotcheck_witness("h1", torch.zeros((4, 8)), comm))
    bad_w_nan["bad_nan"] = float("nan")
    assert "zkwsp_witness_non_serializable" in verify_zkwsp_spotcheck_witness(
        bad_w_nan, hive_id="h1", delta=torch.zeros((4, 8)), subspace_commitment=comm
    )["failures"]

    lean_audit = cached_lean_audit
    with pytest.raises(TypeError, match="Lean4WSPSpecAudit"):
        build_federated_hive_packet("h1", torch.zeros((4, 8)), comm, lean_spec_audit="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="claimed_anse_energy"):
        build_federated_hive_packet(
            "h1", torch.zeros((4, 8)), comm, lean_spec_audit=lean_audit, claimed_anse_energy=float("nan")
        )
    pkt_ok = build_federated_hive_packet("h1", torch.zeros((4, 8)), comm, lean_spec_audit=lean_audit)
    with pytest.raises(TypeError, match="Mapping"):
        verify_federated_hive_packet("bad", comm)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ZKWSPFederatedHiveConfig"):
        verify_federated_hive_packet(pkt_ok, comm, config="bad")  # type: ignore[arg-type]

    no_wit_pkt = dict(pkt_ok)
    no_wit_pkt["spotcheck_witness"] = None
    no_wit_pkt["s5_receipt"] = {**pkt_ok["s5_receipt"], "witness_sha256": "missing"}
    assert "missing_zkwsp_spotcheck_witness" in verify_federated_hive_packet(
        no_wit_pkt, comm, config=ZKWSPFederatedHiveConfig(require_lean4_attestation=False)
    )["rejection_reasons"]

    bad_shape_wit = copy.deepcopy(pkt_ok)
    bad_shape_wit["spotcheck_witness"]["num_challenges"] = 1
    bad_shape_wit["delta"] = torch.zeros((5, 5))
    assert any(
        r.startswith("zkwsp_spotcheck_error:")
        for r in verify_federated_hive_packet(
            bad_shape_wit, comm, config=ZKWSPFederatedHiveConfig(require_lean4_attestation=False)
        )["rejection_reasons"]
    )

    # Duplicate hive_id, select_m > n, bad packets
    with pytest.raises(ValueError, match="non-empty sequence"):
        verify_and_aggregate_federated_hives([], comm)
    with pytest.raises(TypeError, match="ZKWSPFederatedHiveConfig"):
        verify_and_aggregate_federated_hives([pkt_ok], comm, config="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="select_m"):
        verify_and_aggregate_federated_hives(
            [pkt_ok] * 5, comm, config=ZKWSPFederatedHiveConfig(num_byzantine_f=1, select_m=6)
        )
    with pytest.raises(ValueError, match="Duplicate hive_id"):
        verify_and_aggregate_federated_hives([pkt_ok] * 5, comm, lean_spec_audit=lean_audit)

    # execute_verified_zkwsp_federated_hive_round validation errors
    lin = nn.Linear(8, 4)
    pkts_5 = [
        build_federated_hive_packet(
            f"h{i}", torch.zeros((4, 8)), comm, lean_spec_audit=lean_audit, target_key="weight"
        )
        for i in range(5)
    ]
    with pytest.raises(TypeError, match="nn.Module"):
        execute_verified_zkwsp_federated_hive_round("bad", pkts_5, eval_fn=lambda _m: {"E0": 1.0})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable"):
        execute_verified_zkwsp_federated_hive_round(lin, pkts_5, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="state_dict"):
        execute_verified_zkwsp_federated_hive_round(nn.ReLU(), pkts_5, eval_fn=lambda _m: {"E0": 1.0})
    with pytest.raises(ValueError, match="Provide at least one"):
        execute_verified_zkwsp_federated_hive_round(lin, pkts_5)
    with pytest.raises(TypeError, match="ZKWSPFederatedHiveConfig"):
        execute_verified_zkwsp_federated_hive_round(
            lin, pkts_5, eval_fn=lambda _m: {"E0": 1.0}, config="bad"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="Provide subspace_commitment"):
        execute_verified_zkwsp_federated_hive_round(lin, pkts_5, eval_fn=lambda _m: {"E0": 1.0})
    with pytest.raises(ValueError, match="non-finite NLL"):
        execute_verified_zkwsp_federated_hive_round(
            lin, pkts_5, subspace_commitment=comm, eval_fn=lambda _m: {"E0": float("nan")}
        )
    with pytest.raises(ValueError, match="not found in model"):
        pkts_bad_k = [
            build_federated_hive_packet(
                f"hk{i}", torch.zeros((4, 8)), comm, lean_spec_audit=lean_audit, target_key="missing.w"
            )
            for i in range(5)
        ]
        execute_verified_zkwsp_federated_hive_round(
            lin, pkts_bad_k, subspace_commitment=comm, eval_fn=lambda _m: {"E0": 1.0}
        )
    with pytest.raises(ValueError, match="!= model shape"):
        pkts_bad_sh = [
            build_federated_hive_packet(
                f"hs{i}", torch.zeros((2, 8)), comm, lean_spec_audit=lean_audit, target_key="weight"
            )
            for i in range(5)
        ]
        execute_verified_zkwsp_federated_hive_round(
            lin, pkts_bad_sh, subspace_commitment=comm, eval_fn=lambda _m: {"E0": 1.0}
        )

    with pytest.raises(TypeError, match="Mapping"):
        verify_zkwsp_federated_round_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_zkwsp_federated_round_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert verify_zkwsp_federated_round_receipt_provenance({"receipt_sha256": "0" * 64, "object": "bad"}) is False
    assert (
        verify_zkwsp_federated_round_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": ZKWSP_ROUND_RECEIPT_OBJECT,
                "schema": ZKWSP_VERIFIER_SCHEMA,
                "bad": float("nan"),
            }
        )
        is False
    )

    # Missing general chunk / bad chunks validation
    fix = _build_federated_crowd_moe_fixture(seed=20264751)
    with pytest.raises(ValueError, match="general chunk"):
        collect_prose_subspace_from_probe(fix["model"], [{"domain": "E1_only", "ids": [0, 1, 2, 3]}])
    with pytest.raises(ValueError, match="containing 'ids'"):
        execute_verified_zkwsp_federated_hive_round(
            fix["model"], pkts_5, subspace_commitment=comm, eval_chunks=[{"domain": "E0"}]
        )
    with pytest.raises(ValueError, match="non-empty 'expert_id' or 'domain'"):
        execute_verified_zkwsp_federated_hive_round(
            fix["model"], pkts_5, subspace_commitment=comm, eval_chunks=[{"domain": " ", "ids": [0, 1, 2, 3]}]
        )

    # 2. AGPL-3.0, ZeroStubAudit, PRSecurityScanner, lateral movement, FORBIDDEN_ADDED, & OwnerOrderGuardrail
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn47", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn47", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/zk_wsp_federated_hive_verifier.py", TARGET_MODULE_PATH),
        ("tests/test_zk_wsp_federated_hive_verifier_scenarios.py", TEST_FILE_PATH),
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

    for doc_rel, doc_path in (
        ("docs/NEURO_CONTROLLER_ROADMAP.md", ROADMAP_DOC_PATH),
        ("docs/NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md", PLAN_DOC_PATH),
    ):
        doc_text = doc_path.read_text(encoding="utf-8")
        assert "0.00 EUR / 0.00 USD" in doc_text
        assert "NEURAL-CTRL-DYN-47" in doc_text
        assert "PILOT14" in doc_text
        assert scanner.scan_text_for_payment_solicitation(doc_text, doc_rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(doc_text, doc_rel) == []
        assert scanner.scan_unicode_trojan_source(doc_text, doc_rel) == []
        intent = classify_request_intent(doc_text, source_channel="pull_request", changed_files=(doc_rel,))
        assert intent.intent_category == RequestIntentCategory.STANDARD_CONTRIBUTION
        assert intent.is_directive_order is False
        assert intent.protected_paths_touched == ()
        changed_files.append(
            {
                "filename": doc_rel,
                "status": "modified",
                "additions": len(doc_text.splitlines()),
                "patch": "",
                "content": doc_text,
            }
        )

    telemetry_json = json.dumps(
        {"nll_mean": 5.55777, "byzantine_filtered": 2, "anse_energy": -0.00978, "accepted": True},
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+TELEMETRY = {telemetry_json!r}"
        )
        == []
    )
    assert len(compute_payload_sha256({"ok": True})) == 64

    pr_title = "feat(swarm): proof-carrying ZK-WSP subspace spot-check verifier and Multi-Krum Byzantine multi-hive federation controller"
    pr_body = (
        "Implements NEURAL-CTRL-DYN-47 Gen-3 Direction D5 proof-carrying ZK-WSP subspace spot-check "
        "commitments, canonical Lean 4 WSP theorem attestation receipts, and Multi-Krum Byzantine "
        "multi-hive delta aggregation inside P_0 (n >= 2f + 3) enforcing ANSE Delta E < 0, D0 <= +1.0%, "
        "and atomic bitwise rollback."
    )
    pr_intent = classify_request_intent(
        f"{pr_title}\n{pr_body}",
        source_channel="pull_request",
        changed_files=[c["filename"] for c in changed_files],
    )
    assert pr_intent.intent_category == RequestIntentCategory.STANDARD_CONTRIBUTION
    assert pr_intent.is_directive_order is False
    assert pr_intent.protected_paths_touched == ()

    pr_audit = scanner.audit_pull_request(
        pr_number=1247,
        pr_title=pr_title,
        pr_body=pr_body,
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[pr_title],
        author_email=None,
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
