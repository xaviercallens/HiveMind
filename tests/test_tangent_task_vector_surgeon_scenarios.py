# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-32: tangent-space task-vector orthogonalization surgeon."""

from __future__ import annotations

import ast
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
    extract_moe_trainable_buckets,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.tangent_task_vector_surgeon import (
    ANSE_MAXIMUM_PAIN,
    RECEIPT_OBJECT,
    SCHEMA_VERSION,
    TangentTaskVectorConfig,
    compute_activation_weighted_task_interference,
    compute_prose_nullspace_projector_scaled,
    compute_state_sha256,
    execute_verified_tangent_task_composition,
    orthogonalize_task_vectors_polar_nullspace,
    verify_tangent_composition_receipt_provenance,
)
from hivemind.swarm.wsp_alphaedit_control import (
    compute_embeddings_batch,
    compute_swiglu_intermediate_states,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    per_chunk_nll_from_pre_ln,
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
TARGET_MODULE_PATH = (
    REPO_ROOT / "src" / "hivemind" / "swarm" / "tangent_task_vector_surgeon.py"
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


def _make_4domain_moe_testbed(
    *,
    num_layers: int = 2,
    num_experts: int = 4,
    top_k: int = 2,
    lora_rank: int = 8,
    hidden_size: int = 40,
    intermediate_size: int = 80,
    vocab_size: int = 48,
    seed: int = 20261232,
) -> tuple[CrowdV1WithEngram, dict[str, torch.Tensor]]:
    """Build a 4-domain subspace-aligned ``CrowdV1WithEngram`` + ``CrowdLoRAMoEMLP`` testbed.

    Subspaces in ``R^40`` and SwiGLU ``R^80``:
      - ``E0_general_fineweb``:      tokens ``0..11``,  dims ``0..15``,  target ``5``
      - ``E1_python_systems_code``:  tokens ``12..23``, dims ``16..23``, target ``16``
      - ``E2_math_reasoning``:       tokens ``24..35``, dims ``24..31``, target ``28``
      - ``E5_exoplanet_astronomy``:  tokens ``36..47``, dims ``32..39``, target ``40``
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
    u_e5 = _unit_axis(hidden_size, 32, 33)

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        p_raw = 0.25 * torch.randn((12, 16), generator=gen, dtype=torch.float32)
        p_raw[:, :2] = 0.0
        p_raw = p_raw - p_raw.mean(dim=1, keepdim=True) + 1.0 * u_e0[:16].unsqueeze(0)
        p_raw[5] = 1.45 * u_e0[:16]

        def _dom_block(start_idx: int, tgt_offset: int, axis: torch.Tensor) -> torch.Tensor:
            blk_t = 0.20 * torch.randn((12, 8), generator=gen, dtype=torch.float32)
            blk_t[:, :2] = 0.0
            blk_t = blk_t - blk_t.mean(dim=1, keepdim=True) + 0.90 * axis[start_idx : start_idx + 8].unsqueeze(0)
            blk_t[tgt_offset] = 1.45 * axis[start_idx : start_idx + 8]
            return blk_t

        wte[0:12, 0:16] = p_raw
        wte[12:24, 16:24] = _dom_block(16, 4, u_e1)
        wte[24:36, 24:32] = _dom_block(24, 4, u_e2)
        wte[36:48, 32:40] = _dom_block(32, 4, u_e5)
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
            blk.mlp.w3.weight[:hidden_size, :hidden_size].copy_(0.08 * torch.eye(hidden_size))

    upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode="lora_domain_experts",
        num_experts=num_experts,
        top_k=top_k,
        lora_rank=lora_rank,
        lora_alpha=float(2 * lora_rank),
        routing_mode="sigma_topk",
        seed=seed + 7,
    )

    with torch.no_grad():
        for blk in model.blocks:
            blk.mlp.router.weight.zero_()  # type: ignore[union-attr]
            for exp_mod in blk.mlp.experts.values():  # type: ignore[union-attr]
                for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                    getattr(exp_mod, attr).zero_()
            # Route E0 cleanly through expert 0 and all specialists through shared expert 1
            blk.mlp.router.weight[0] = 3.5 * u_e0  # type: ignore[union-attr]
            blk.mlp.router.weight[1] = 3.0 * (u_e1 + u_e2 + u_e5)  # type: ignore[union-attr]
            blk.mlp.router.weight[2] = -2.0 * torch.ones(hidden_size, dtype=torch.float32)  # type: ignore[union-attr]
            blk.mlp.router.weight[3] = -2.0 * torch.ones(hidden_size, dtype=torch.float32)  # type: ignore[union-attr]

    return model, {"e0": u_e0, "e1": u_e1, "e2": u_e2, "e5": u_e5}


def _make_domain_chunks(
    *,
    domains: tuple[str, ...] = (
        "E0_general_fineweb",
        "E1_python_systems_code",
        "E2_math_reasoning",
        "E5_exoplanet_astronomy",
    ),
    seq_len: int = 16,
    score_from: int = 8,
    chunks_per_domain: int = 4,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    gen = torch.Generator(device="cpu").manual_seed(9100 + seed_offset)
    ranges = {
        "E0_general_fineweb": (0, 12, 5),
        "E1_python_systems_code": (12, 24, 16),
        "E2_math_reasoning": (24, 36, 28),
        "E5_exoplanet_astronomy": (36, 48, 40),
        "E7_harmful_exploit": (36, 48, 42),
    }
    chunks: list[dict[str, Any]] = []
    for dom in domains:
        lo, hi, tgt = ranges[dom]
        for _ in range(chunks_per_domain):
            ids = torch.randint(lo, hi, (seq_len,), generator=gen).tolist()
            for pos in range(score_from, seq_len):
                ids[pos] = tgt
            chunks.append({"expert_id": dom, "ids": ids})
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
    losses = per_chunk_nll_from_pre_ln(
        xb,
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
    out = {k: sum(v) / len(v) for k, v in sorted(grouped.items())}
    spec_vals = [v for k, v in out.items() if not k.startswith("E0_")]
    out["spec_mean"] = sum(spec_vals) / len(spec_vals) if spec_vals else 0.0
    out["nll_mean"] = sum(losses) / len(losses)
    return out


def _swiglu_axis(inter_dim: int, start_idx: int) -> torch.Tensor:
    """Normalized positive axis inside domain's SwiGLU block matching ``silu(h1)*h2``."""
    v = torch.zeros(inter_dim, dtype=torch.float64)
    v[start_idx : start_idx + 2] = 1.0 / math.sqrt(2.0)
    return v


@torch.no_grad()
def _top_e0_swiglu_direction(
    model: CrowdV1WithEngram,
    probe_chunks: list[dict[str, Any]],
    *,
    score_from: int = 8,
) -> torch.Tensor:
    """Return top right singular direction of ``E0_*`` SwiGLU activations at block 1."""
    ids = torch.tensor(
        [c["ids"] for c in probe_chunks if str(c["expert_id"]).startswith("E0_")],
        dtype=torch.long,
    )
    cos = model.cos_cached.to(ids.device)
    sin = model.sin_cached.to(ids.device)
    xb = compute_embeddings_batch(model, ids)
    start_pos = max(0, score_from - 1)
    for l_idx, blk in enumerate(model.blocks):
        xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
        if l_idx == 1:
            scored_x = mi[:, start_pos:-1, :].reshape(-1, int(mi.shape[-1]))
            active_exp = sorted(int(k) for k in blk.mlp.experts.keys())  # type: ignore[union-attr]
            act_e0 = compute_swiglu_intermediate_states(blk.mlp, scored_x, active_exp)  # type: ignore[arg-type]
            _, _, vh = torch.linalg.svd(act_e0.to(torch.float64), full_matrices=False)
            return vh[0].contiguous()
        xb = forward_block_post_mlp(blk, xm, mi)
    raise AssertionError("Expected >= 2 transformer blocks in testbed")


# ---------------------------------------------------------------------------
# S1: 3 Interfering Specialist Task Vectors vs Naive Sum
# ---------------------------------------------------------------------------


def test_s1_three_interfering_specialists_polar_nullspace_composition() -> None:
    """S1: Polar-nullspace composition beats naive sum on multi-domain NLL with <= +0.2% prose drift."""
    model, axes = _make_4domain_moe_testbed(seed=20261232)
    probe_chunks = _make_domain_chunks(chunks_per_domain=4, seed_offset=0)
    eval_chunks = _make_domain_chunks(chunks_per_domain=4, seed_offset=100)
    base_nll = _eval_model_nlls(model, eval_chunks, score_from=8)

    inter_dim = 80
    a0 = _top_e0_swiglu_direction(model, probe_chunks, score_from=8)
    a1 = _swiglu_axis(inter_dim, 16)
    a2 = _swiglu_axis(inter_dim, 24)
    a5 = _swiglu_axis(inter_dim, 32)
    u0 = axes["e0"].to(torch.float64)
    u1 = axes["e1"].to(torch.float64)
    u2 = axes["e2"].to(torch.float64)
    u5 = axes["e5"].to(torch.float64)

    # Three specialist task vectors with severe cross-task cancellation & prose regression:
    # Linear sum yields (-0.6) reversal per skill plus (-7.2) corruption along E0 prose.
    delta_code = (
        3.0 * torch.outer(u1, a1)
        - 1.8 * torch.outer(u2, a2)
        - 1.8 * torch.outer(u5, a5)
        - 2.4 * torch.outer(u0, a0)
    ).to(torch.float32)
    delta_math = (
        3.0 * torch.outer(u2, a2)
        - 1.8 * torch.outer(u5, a5)
        - 1.8 * torch.outer(u1, a1)
        - 2.4 * torch.outer(u0, a0)
    ).to(torch.float32)
    delta_astro = (
        3.0 * torch.outer(u5, a5)
        - 1.8 * torch.outer(u1, a1)
        - 1.8 * torch.outer(u2, a2)
        - 2.4 * torch.outer(u0, a0)
    ).to(torch.float32)

    task_deltas = {
        "code": {"blocks.1.mlp.experts.1.w3": delta_code},
        "math": {"blocks.1.mlp.experts.1.w3": delta_math},
        "astro": {"blocks.1.mlp.experts.1.w3": delta_astro},
    }

    # Baseline: naive linear task arithmetic sum suffers prose NLL regression & cross-task cancellation
    naive_model = copy.deepcopy(model)
    with torch.no_grad():
        exp1 = naive_model.blocks[1].mlp.experts["1"]  # type: ignore[union-attr]
        naive_sum = (delta_code + delta_math + delta_astro).to(torch.float64) / float(exp1.scaling)
        u_n, s_n, vh_n = torch.linalg.svd(naive_sum, full_matrices=False)
        k_n = min(int(exp1.rank), int(s_n.numel()))
        s_sq = s_n[:k_n].clamp_min(0.0).sqrt()
        exp1.w3_B[:, :k_n].copy_((u_n[:, :k_n] * s_sq.unsqueeze(0)).to(torch.float32))
        exp1.w3_A[:k_n, :].copy_((s_sq.unsqueeze(1) * vh_n[:k_n, :]).to(torch.float32))
    naive_nll = _eval_model_nlls(naive_model, eval_chunks, score_from=8)
    naive_e0_drift_pct = 100.0 * (naive_nll["E0_general_fineweb"] / base_nll["E0_general_fineweb"] - 1.0)
    assert naive_e0_drift_pct > 50.0
    assert naive_nll["spec_mean"] > 0.38

    # Polar-nullspace tangent composition (non-inplace copy first) isolating orthogonal skill modes
    composed_model, receipt = execute_verified_tangent_task_composition(
        model,
        task_deltas,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        rank_per_task=1,
        score_from=8,
        inplace=False,
    )
    assert receipt["schema"] == SCHEMA_VERSION
    assert receipt["object"] == RECEIPT_OBJECT
    assert verify_tangent_composition_receipt_provenance(receipt) is True
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["status"] == "COMMITTED"
    assert receipt["delta_e"] < 0.0
    assert receipt["guard_ok"] is True
    assert receipt["cosine_interference"] < receipt["cosine_interference_before"]
    assert receipt["cosine_interference"] < 1e-10
    assert compute_state_sha256(model) == receipt["parent_state_sha256"]
    assert compute_state_sha256(composed_model) == receipt["final_state_sha256"]

    after_nll = _eval_model_nlls(composed_model, eval_chunks, score_from=8)
    prose_drift_pct = 100.0 * (after_nll["E0_general_fineweb"] / base_nll["E0_general_fineweb"] - 1.0)
    assert prose_drift_pct <= 0.2
    assert after_nll["spec_mean"] < naive_nll["spec_mean"] - 0.08
    assert after_nll["spec_mean"] < base_nll["spec_mean"] - 0.15


# ---------------------------------------------------------------------------
# S2: Exact Prose Null-Space P_0 Invariance & Pairwise Interference Reduction
# ---------------------------------------------------------------------------


def test_s2_exact_prose_nullspace_invariance_and_interference_reduction() -> None:
    """S2: Certify <1e-5 P_0 key drift and orthogonalization collapsing cosine interference."""
    gen = torch.Generator(device="cpu").manual_seed(20261233)
    d_in, d_out = 24, 20

    q_in, _ = torch.linalg.qr(torch.randn((d_in, d_in), generator=gen, dtype=torch.float64))
    q_out, _ = torch.linalg.qr(torch.randn((d_out, d_out), generator=gen, dtype=torch.float64))

    # Prose activation keys K_0 occupy first 4 orthonormal basis directions
    coeffs = torch.randn((24, 4), generator=gen, dtype=torch.float64)
    k0_prose = coeffs @ q_in[:, :4].T
    p_null, p_stats = compute_prose_nullspace_projector_scaled(k0_prose)
    assert p_stats["active_modes"] == 4
    assert p_stats["nullspace_dim"] == d_in - 4

    # Construct 3 interfering rank-2 task vectors with shared cross-task overlap + prose leakage
    v_shared = q_in[:, 16:18]
    u_shared = q_out[:, 12:14]
    shared_proj = v_shared @ v_shared.T
    x_doms = [
        torch.randn((16, d_in), generator=gen, dtype=torch.float64)
        @ (q_in[:, 4:8] @ q_in[:, 4:8].T + 0.7 * shared_proj),
        torch.randn((16, d_in), generator=gen, dtype=torch.float64)
        @ (q_in[:, 8:12] @ q_in[:, 8:12].T + 0.7 * shared_proj),
        torch.randn((16, d_in), generator=gen, dtype=torch.float64)
        @ (q_in[:, 12:16] @ q_in[:, 12:16].T + 0.7 * shared_proj),
    ]

    t1 = 2.0 * (0.8 * q_out[:, 0:2] + 0.6 * u_shared) @ (0.8 * q_in[:, 4:6] + 0.6 * v_shared).T + 1.5 * (
        q_out[:, 14:18] @ q_in[:, :4].T
    )
    t2 = 2.0 * (0.8 * q_out[:, 2:4] + 0.6 * u_shared) @ (0.8 * q_in[:, 8:10] - 0.6 * v_shared).T + 1.5 * (
        q_out[:, 14:18] @ q_in[:, :4].T
    )
    t3 = 2.0 * (0.8 * q_out[:, 4:6] - 0.6 * u_shared) @ (0.8 * q_in[:, 12:14] + 0.6 * v_shared).T + 1.5 * (
        q_out[:, 14:18] @ q_in[:, :4].T
    )
    raw_tasks = [t1, t2, t3]
    naive_sum = t1 + t2 + t3

    before_int = compute_activation_weighted_task_interference(raw_tasks, x_doms)
    assert before_int["cosine_interference"] > 0.05
    assert before_int["max_abs_frobenius_cosine"] > 0.25

    orth_tasks, composed, rep = orthogonalize_task_vectors_polar_nullspace(raw_tasks, p_null)
    assert isinstance(orth_tasks, list) and isinstance(composed, torch.Tensor)
    assert rep["u_orthogonality_error"] < 1e-12
    assert rep["v_orthogonality_error"] < 1e-12
    assert rep["nullspace_leakage_norm"] < 1e-12
    assert rep["max_pairwise_frobenius_cosine_after"] < 1e-12

    # Verify exact prose key invariance (<1e-5 relative key drift) vs huge naive drift
    k0_norm = float(torch.linalg.vector_norm(k0_prose).item())
    naive_key_drift = float(torch.linalg.vector_norm(naive_sum @ k0_prose.T).item()) / k0_norm
    comp_key_drift = float(torch.linalg.vector_norm(composed @ k0_prose.T).item()) / k0_norm
    assert naive_key_drift > 0.5
    assert comp_key_drift < 1e-5
    for tok in orth_tasks:
        assert (float(torch.linalg.vector_norm(tok @ k0_prose.T).item()) / k0_norm) < 1e-5

    after_int = compute_activation_weighted_task_interference(orth_tasks, x_doms)
    assert after_int["cosine_interference"] < 1e-10
    assert after_int["max_abs_frobenius_cosine"] < 1e-10
    assert after_int["mean_cross_domain_leakage"] < before_int["mean_cross_domain_leakage"]

    # Verify permutation equivariance & multi-param dict + output-side & both-side projection
    orth_rev, _, _ = orthogonalize_task_vectors_polar_nullspace([t3, t1, t2], p_null)
    assert torch.allclose(orth_rev[1], orth_tasks[0], atol=1e-10)
    assert torch.allclose(orth_rev[2], orth_tasks[1], atol=1e-10)
    assert torch.allclose(orth_rev[0], orth_tasks[2], atol=1e-10)

    p_out_null, _ = compute_prose_nullspace_projector_scaled(q_out[:, :2].T)
    orth_out, comp_out, rep_out = orthogonalize_task_vectors_polar_nullspace(
        {"a": t1, "b": t2, "c": t3},
        p_out_null,
        projection_side="output",
        rank_per_task=1,
        variance_threshold=0.90,
    )
    assert isinstance(orth_out, dict) and rep_out["projection_side"] == "output"
    assert float(torch.linalg.vector_norm(q_out[:, :2].T @ comp_out).item()) < 1e-11

    orth_both, comp_both, rep_both = orthogonalize_task_vectors_polar_nullspace(
        raw_tasks, p_null, p_null_out=p_out_null, projection_side="input"
    )
    assert len(orth_both) == 3 and rep_both["projection_side"] == "both"
    assert float(torch.linalg.vector_norm(comp_both @ k0_prose.T).item()) < 1e-11

    # Water-filling rank truncation when stacked full-rank tasks exceed nullspace capacity
    dense_tasks = [torch.randn((10, 10), generator=gen, dtype=torch.float64) for _ in range(3)]
    _, _, rep_wf = orthogonalize_task_vectors_polar_nullspace(
        dense_tasks, torch.eye(10, dtype=torch.float64)
    )
    assert sum(rep_wf["allocated_ranks"]) <= 10

    # Dict-of-params interference & orthogonalization + 1D vector orthogonalization
    dict_tasks = {"t1": {"w": t1, "b": torch.tensor([1.0, 0.5, 0.0, 0.0])}, "t2": {"w": t2, "b": torch.tensor([0.5, 1.0, 0.0, 0.0])}}
    d_int = compute_activation_weighted_task_interference(
        dict_tasks, {"w": x_doms[0], "b": torch.eye(4)}
    )
    assert d_int["num_tasks"] == 2
    p_b, _ = compute_prose_nullspace_projector_scaled(torch.tensor([[0.0, 0.0, 1.0, 0.0]]))
    orth_d, comp_d, rep_d = orthogonalize_task_vectors_polar_nullspace(
        dict_tasks, {"w": p_null, "b": p_b}
    )
    assert isinstance(orth_d, dict) and isinstance(comp_d, dict)
    assert rep_d["nullspace_leakage_norm"] < 1e-12


# ---------------------------------------------------------------------------
# S3: Skill Negation / Unlearning of a Harmful Task Vector
# ---------------------------------------------------------------------------


def test_s3_skill_negation_unlearning_without_hurting_retained_skills() -> None:
    """S3: Negate an interfering harmful task vector cleanly without degrading retained skills or prose."""
    model, axes = _make_4domain_moe_testbed(seed=20261234)
    # Plant an active harmful capability inside expert 1 before unlearning (targeting token 42 in E5/E7 range)
    inter_dim = 80
    a1 = _swiglu_axis(inter_dim, 16)
    a2 = _swiglu_axis(inter_dim, 24)
    a_harm = _swiglu_axis(inter_dim, 32)
    u0 = axes["e0"].to(torch.float64)
    u1 = axes["e1"].to(torch.float64)
    u2 = axes["e2"].to(torch.float64)
    # Harmful logit direction aligned with token 42's embedding
    u_harm = model.wte.weight[42].detach().to(torch.float64)
    u_harm = u_harm / u_harm.norm()

    with torch.no_grad():
        # Pre-seed parent model with harmful prediction capability in base_mlp.w3
        model.blocks[1].mlp.base_mlp.w3.weight.add_(
            (2.4 * torch.outer(u_harm, a_harm)).to(torch.float32)
        )

    probe_chunks = _make_domain_chunks(
        domains=("E0_general_fineweb", "E1_python_systems_code", "E2_math_reasoning"),
        chunks_per_domain=4,
        seed_offset=10,
    )
    a0 = _top_e0_swiglu_direction(model, probe_chunks, score_from=8)
    eval_chunks = _make_domain_chunks(
        domains=(
            "E0_general_fineweb",
            "E1_python_systems_code",
            "E2_math_reasoning",
            "E7_harmful_exploit",
        ),
        chunks_per_domain=4,
        seed_offset=200,
    )
    base_nll = _eval_model_nlls(model, eval_chunks, score_from=8)

    # Retained tasks (code, math) and harmful task vector (which interferes with code/math & leaks prose)
    delta_code = (2.6 * torch.outer(u1, a1) - 1.5 * torch.outer(u0, a0)).to(torch.float32)
    delta_math = (2.6 * torch.outer(u2, a2) - 1.5 * torch.outer(u0, a0)).to(torch.float32)
    delta_harm = (
        3.2 * torch.outer(u_harm, a_harm)
        + 2.4 * torch.outer(u1, a1)
        + 2.4 * torch.outer(u2, a2)
        + 2.2 * torch.outer(u0, a0)
    ).to(torch.float32)

    task_deltas = {
        "code": {"blocks.1.mlp.experts.1.w3": delta_code},
        "math": {"blocks.1.mlp.experts.1.w3": delta_math},
        "harmful": {"blocks.1.mlp.experts.1.w3": delta_harm},
    }
    weights = [1.0, 1.0, -1.0]

    # Naive negation (+code + math - harmful) cancels retained code/math AND damages prose
    naive_model = copy.deepcopy(model)
    with torch.no_grad():
        exp1 = naive_model.blocks[1].mlp.experts["1"]  # type: ignore[union-attr]
        naive_delta = (delta_code + delta_math - delta_harm).to(torch.float64) / float(exp1.scaling)
        u_n, s_n, vh_n = torch.linalg.svd(naive_delta, full_matrices=False)
        k_n = min(int(exp1.rank), int(s_n.numel()))
        s_sq = s_n[:k_n].clamp_min(0.0).sqrt()
        exp1.w3_B[:, :k_n].copy_((u_n[:, :k_n] * s_sq.unsqueeze(0)).to(torch.float32))
        exp1.w3_A[:k_n, :].copy_((s_sq.unsqueeze(1) * vh_n[:k_n, :]).to(torch.float32))
    naive_nll = _eval_model_nlls(naive_model, eval_chunks, score_from=8)
    assert 100.0 * (naive_nll["E0_general_fineweb"] / base_nll["E0_general_fineweb"] - 1.0) > 10.0

    # Polar-nullspace orthogonalized skill composition + unlearning
    unlearned_model, receipt = execute_verified_tangent_task_composition(
        model,
        task_deltas,
        task_weights=weights,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        unlearn_domain_prefixes=("E7_harmful",),
        rank_per_task=1,
        score_from=8,
        inplace=False,
    )
    assert receipt["accepted"] is True
    assert receipt["guard_ok"] is True
    assert receipt["delta_e"] < 0.0
    assert receipt["unlearned_domains"]["E7_harmful_exploit"]["delta_nll"] > 0.15
    assert verify_tangent_composition_receipt_provenance(receipt) is True

    after_nll = _eval_model_nlls(unlearned_model, eval_chunks, score_from=8)
    prose_drift_pct = 100.0 * (after_nll["E0_general_fineweb"] / base_nll["E0_general_fineweb"] - 1.0)
    assert prose_drift_pct <= 0.2
    # Harmful skill is unlearned (higher NLL), while retained skills strictly improve vs parent & naive
    assert after_nll["E7_harmful_exploit"] > base_nll["E7_harmful_exploit"] + 0.15
    assert after_nll["E1_python_systems_code"] < base_nll["E1_python_systems_code"] - 0.15
    assert after_nll["E2_math_reasoning"] < base_nll["E2_math_reasoning"] - 0.15
    assert after_nll["E1_python_systems_code"] < naive_nll["E1_python_systems_code"]
    assert after_nll["E2_math_reasoning"] < naive_nll["E2_math_reasoning"]


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Atomic Rollback on Infeasible Guard
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_rollback_on_infeasible_guard() -> None:
    """S4: Enforce atomic bitwise rollback & ANSE Maximum Pain (1e6) on guard/energy/NaN failure."""
    model, axes = _make_4domain_moe_testbed(seed=20261235)
    eval_chunks = _make_domain_chunks(chunks_per_domain=3, seed_offset=300)
    parent_sha = compute_state_sha256(model)

    a0 = _swiglu_axis(80, 0)
    u0 = axes["e0"].to(torch.float64)
    corrupt_delta = (-4.0 * torch.outer(u0, a0)).to(torch.float32)
    task_deltas = [
        {"blocks.1.mlp.experts.1.w3": corrupt_delta},
        {"blocks.1.mlp.experts.1.w3": corrupt_delta * 0.9},
    ]

    # 1. Passing identity p_null disables prose protection -> E0 drift > +1.0% -> ANSE 1e6 rollback
    _, rb_guard = execute_verified_tangent_task_composition(
        model,
        task_deltas,
        p_null=torch.eye(80, dtype=torch.float64),
        eval_chunks=eval_chunks,
        score_from=8,
        inplace=True,
    )
    assert rb_guard["accepted"] is False
    assert rb_guard["rolled_back"] is True
    assert rb_guard["rollback_bitwise_verified"] is True
    assert rb_guard["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert rb_guard["guard_ok"] is False
    assert rb_guard["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert rb_guard["parent_state_sha256"] == parent_sha
    assert rb_guard["final_state_sha256"] == parent_sha
    assert rb_guard["candidate_state_sha256"] != parent_sha
    assert compute_state_sha256(model) == parent_sha
    assert verify_tangent_composition_receipt_provenance(rb_guard) is True

    # 2. Non-finite eval_fn rollback
    _, rb_nan = execute_verified_tangent_task_composition(
        model,
        task_deltas,
        p_null=torch.eye(80, dtype=torch.float64),
        eval_fn=lambda m: (
            {"E0_general_fineweb": 1.0, "E1_code": 1.0}
            if compute_state_sha256(m) == parent_sha
            else {"E0_general_fineweb": float("nan"), "E1_code": 1.0}
        ),
        inplace=True,
    )
    assert rb_nan["accepted"] is False
    assert rb_nan["rolled_back"] is True
    assert rb_nan["status"] == "ROLLED_BACK_NON_FINITE"
    assert rb_nan["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert compute_state_sha256(model) == parent_sha

    # 3. Non-improving energy rollback (Delta E >= 0 while guard passes)
    _, rb_energy = execute_verified_tangent_task_composition(
        model,
        task_deltas,
        p_null=torch.eye(80, dtype=torch.float64),
        eval_fn=lambda m: (
            {"E0_general_fineweb": 1.0, "E1_code": 1.0}
            if compute_state_sha256(m) == parent_sha
            else {"E0_general_fineweb": 1.005, "E1_code": 1.05}
        ),
        enforce_all_domain_guard=False,
        inplace=True,
    )
    assert rb_energy["accepted"] is False
    assert rb_energy["rolled_back"] is True
    assert rb_energy["status"] == "ROLLED_BACK_NO_ENERGY_IMPROVEMENT"
    assert rb_energy["delta_e"] > 0.0
    assert compute_state_sha256(model) == parent_sha

    # 4. Zero-delta unchanged no-op
    _, noop_rec = execute_verified_tangent_task_composition(
        model,
        [{"blocks.1.mlp.experts.1.w3": torch.zeros((40, 80), dtype=torch.float32)}],
        eval_chunks=eval_chunks,
        score_from=8,
        inplace=True,
    )
    assert noop_rec["accepted"] is False
    assert noop_rec["rolled_back"] is False
    assert noop_rec["status"] == "NOOP_UNCHANGED"
    assert compute_state_sha256(model) == parent_sha


# ---------------------------------------------------------------------------
# S5: Validation Errors, Edge Cases & Security/Stub Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_edge_cases_and_security_compliance_gates() -> None:
    """S5: Cover defensive error branches, bucket/LoRA folding, and security/stub compliance gates."""
    gen = torch.Generator(device="cpu").manual_seed(20261236)

    # 1. TangentTaskVectorConfig validation
    with pytest.raises(ValueError, match="rank_per_task"):
        TangentTaskVectorConfig(rank_per_task=0)
    with pytest.raises(ValueError, match="variance_threshold"):
        TangentTaskVectorConfig(variance_threshold=0.0)
    with pytest.raises(ValueError, match="nullspace_eig_threshold"):
        TangentTaskVectorConfig(nullspace_eig_threshold=1.0)
    with pytest.raises(ValueError, match="nullspace_variance_threshold"):
        TangentTaskVectorConfig(nullspace_variance_threshold=1.5)
    with pytest.raises(ValueError, match="nullspace_max_active_modes"):
        TangentTaskVectorConfig(nullspace_max_active_modes=0)
    with pytest.raises(ValueError, match="projection_side"):
        TangentTaskVectorConfig(projection_side="invalid")
    with pytest.raises(ValueError, match="domain_guard_max_ratio"):
        TangentTaskVectorConfig(domain_guard_max_ratio=0.99)
    with pytest.raises(ValueError, match="anse_safe_general_ratio"):
        TangentTaskVectorConfig(anse_safe_general_ratio=0.99)
    with pytest.raises(TypeError, match="bool"):
        TangentTaskVectorConfig(protect_layernorm="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unlearn_domain_prefixes"):
        TangentTaskVectorConfig(unlearn_domain_prefixes=("  ",))
    with pytest.raises(TypeError, match="int"):
        TangentTaskVectorConfig(batch_size=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">="):
        TangentTaskVectorConfig(score_from=-1)
    with pytest.raises(ValueError, match=">= 0.0"):
        TangentTaskVectorConfig(sv_rel_tol=-0.1)
    with pytest.raises(ValueError, match="> 0.0"):
        TangentTaskVectorConfig(domain_guard_max_ratio=0.0)

    # 2. Nullspace builder & input validation
    with pytest.raises(ValueError, match="not be empty"):
        compute_prose_nullspace_projector_scaled([])
    with pytest.raises(ValueError, match="not be empty"):
        compute_prose_nullspace_projector_scaled({})
    with pytest.raises(TypeError, match="prose_keys"):
        compute_prose_nullspace_projector_scaled("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_prose_nullspace_projector_scaled(["bad"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="width"):
        compute_prose_nullspace_projector_scaled([torch.ones((4, 4)), torch.ones((4, 6))])
    p_zero, s_zero = compute_prose_nullspace_projector_scaled({"k": torch.zeros((4, 6))})
    assert torch.equal(p_zero, torch.eye(6, dtype=torch.float64))
    assert s_zero["active_modes"] == 0

    # 3. Task collection & domain interference validation
    w_ok = torch.randn((4, 6), generator=gen)
    x_ok = torch.randn((8, 6), generator=gen)
    with pytest.raises(ValueError, match="eps"):
        compute_activation_weighted_task_interference([w_ok], x_ok, eps=0.0)
    with pytest.raises(ValueError, match="empty"):
        compute_activation_weighted_task_interference([], x_ok)
    with pytest.raises(ValueError, match="empty"):
        compute_activation_weighted_task_interference({}, x_ok)
    with pytest.raises(TypeError, match="task_name"):
        compute_activation_weighted_task_interference({123: w_ok}, x_ok)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="printable"):
        compute_activation_weighted_task_interference({"  ": w_ok}, x_ok)
    with pytest.raises(TypeError, match="Sequence or Mapping"):
        compute_activation_weighted_task_interference("bad", x_ok)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_activation_weighted_task_interference([w_ok, {"a": w_ok}], x_ok)  # type: ignore[list-item]
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_activation_weighted_task_interference([1, 2], x_ok)  # type: ignore[list-item]
    with pytest.raises(TypeError, match="floating-point"):
        compute_activation_weighted_task_interference([torch.ones((4, 6), dtype=torch.int64)], x_ok)
    with pytest.raises(ValueError, match="finite"):
        compute_activation_weighted_task_interference([torch.full((4, 6), float("nan"))], x_ok)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_activation_weighted_task_interference([w_ok, torch.ones((4, 5))], x_ok)
    with pytest.raises(ValueError, match="ndim"):
        compute_activation_weighted_task_interference([torch.ones((2, 2, 2))], x_ok)
    with pytest.raises(ValueError, match="feature dim"):
        compute_activation_weighted_task_interference([w_ok], torch.ones((4, 5)))
    with pytest.raises(ValueError, match="domain_inputs length"):
        compute_activation_weighted_task_interference([w_ok, w_ok], [x_ok])
    with pytest.raises(ValueError, match="empty"):
        compute_activation_weighted_task_interference([w_ok], {})
    with pytest.raises(ValueError, match="do not match"):
        compute_activation_weighted_task_interference([w_ok, w_ok], {"only_one": x_ok})
    with pytest.raises(TypeError, match="domain_inputs"):
        compute_activation_weighted_task_interference([w_ok], "bad")  # type: ignore[arg-type]
    assert (
        compute_activation_weighted_task_interference({"a": w_ok, "b": w_ok}, {"a": x_ok, "b": x_ok})[
            "num_tasks"
        ]
        == 2
    )
    assert compute_activation_weighted_task_interference([w_ok, w_ok], {"a": x_ok, "b": x_ok})["num_tasks"] == 2
    assert compute_activation_weighted_task_interference([{"w": w_ok}], {"custom_act": x_ok})["num_tasks"] == 1
    with pytest.raises(TypeError, match="Mapping"):
        compute_activation_weighted_task_interference([{"w": w_ok}], x_ok)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="matched"):
        compute_activation_weighted_task_interference([{"w": w_ok}], {"w": torch.ones((4, 9))})
    with pytest.raises(ValueError, match="keys do not match"):
        compute_activation_weighted_task_interference([{"w": w_ok}, {"z": w_ok}], {"w": x_ok})
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_activation_weighted_task_interference(
            [{"w": w_ok}, {"w": torch.ones((4, 5))}], {"w": x_ok}
        )
    with pytest.raises(ValueError, match="non-empty Mapping"):
        compute_activation_weighted_task_interference([{}], {"w": x_ok})

    # 4. orthogonalize_task_vectors_polar_nullspace edges
    with pytest.raises(TypeError, match="sequence"):
        orthogonalize_task_vectors_polar_nullspace([w_ok], None, task_weights="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="length"):
        orthogonalize_task_vectors_polar_nullspace([w_ok, w_ok], None, task_weights=[1.0])
    with pytest.raises(TypeError, match="real number"):
        orthogonalize_task_vectors_polar_nullspace([w_ok], None, task_weights=[True])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="finite"):
        orthogonalize_task_vectors_polar_nullspace([w_ok], None, task_weights=[float("inf")])
    with pytest.raises(ValueError, match="non-zero"):
        orthogonalize_task_vectors_polar_nullspace([w_ok, w_ok], None, task_weights=[0.0, 0.0])
    with pytest.raises(ValueError, match="nullspace_eig_threshold"):
        orthogonalize_task_vectors_polar_nullspace([w_ok], None, nullspace_eig_threshold=1.0)
    with pytest.raises(TypeError, match="p_null"):
        orthogonalize_task_vectors_polar_nullspace([w_ok], {"x": torch.eye(6)})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="square"):
        orthogonalize_task_vectors_polar_nullspace([w_ok], torch.ones((6, 5)))
    with pytest.raises(ValueError, match="input dim"):
        orthogonalize_task_vectors_polar_nullspace([w_ok], torch.eye(5), projection_side="input")
    with pytest.raises(ValueError, match="output dim"):
        orthogonalize_task_vectors_polar_nullspace([w_ok], torch.eye(5), projection_side="output")
    with pytest.raises(ValueError, match="neither"):
        orthogonalize_task_vectors_polar_nullspace([w_ok], torch.eye(9))
    with pytest.raises(ValueError, match="projection_side"):
        orthogonalize_task_vectors_polar_nullspace([w_ok], torch.eye(6), projection_side="bad")
    with pytest.raises(ValueError, match="p_null_out"):
        orthogonalize_task_vectors_polar_nullspace([w_ok], torch.eye(6), p_null_out=torch.eye(5))
    with pytest.raises(ValueError, match="projection_side"):
        orthogonalize_task_vectors_polar_nullspace([torch.ones(6)], torch.eye(6), projection_side="bad")
    with pytest.raises(ValueError, match="p_null_out"):
        orthogonalize_task_vectors_polar_nullspace([torch.ones(6)], torch.eye(6), p_null_out=torch.eye(6))
    with pytest.raises(TypeError, match="p_null"):
        orthogonalize_task_vectors_polar_nullspace([{"w": w_ok}], "bad")  # type: ignore[arg-type]

    # Single-task summary, auto output-side projector, water-filling at floor=1, mixed zero+nonzero task, dict matchers
    _, _, rep_single = orthogonalize_task_vectors_polar_nullspace([w_ok], torch.eye(6))
    assert rep_single["mean_abs_frobenius_cosine_after"] == 0.0
    _, _, rep_auto_out = orthogonalize_task_vectors_polar_nullspace([w_ok], torch.eye(4))
    assert rep_auto_out["projection_side"] == "output"
    _, _, rep_floor = orthogonalize_task_vectors_polar_nullspace([torch.eye(3)] * 4, torch.eye(3))
    assert rep_floor["allocated_ranks"] == [1, 1, 1, 1]
    _, _, rep_half_z = orthogonalize_task_vectors_polar_nullspace([w_ok, torch.zeros_like(w_ok)], torch.eye(6))
    assert rep_half_z["allocated_ranks"] == [4, 0]
    _, _, rep_dict_none = orthogonalize_task_vectors_polar_nullspace([{"w": w_ok, "b": torch.ones(4)}], None)
    assert rep_dict_none["nullspace_leakage_norm"] == 0.0
    _, _, rep_dict_tensor = orthogonalize_task_vectors_polar_nullspace(
        [{"w": w_ok, "b": torch.ones(6), "z": torch.ones((2, 9))}], torch.eye(6)
    )
    assert rep_dict_tensor["num_tasks"] == 1
    _, _, rep_dict_map = orthogonalize_task_vectors_polar_nullspace(
        [
            {
                "blocks.0.mlp.experts.1.w3": w_ok,
                "blocks.0.mlp.router.weight": w_ok.T,
                "other": torch.ones((2, 9)),
            }
        ],
        {"blocks.0.act": torch.eye(6), "blocks.0.x": torch.eye(4)},
    )
    assert rep_dict_map["num_tasks"] == 1

    # Zero-rank nullspace & zero-matrix tasks + duplicate rank-deficient tasks + 1D empty nullspace
    _, comp_zero_p, rep_zp = orthogonalize_task_vectors_polar_nullspace(
        [w_ok, w_ok], torch.zeros((6, 6), dtype=torch.float64)
    )
    assert torch.equal(comp_zero_p, torch.zeros_like(w_ok))
    assert rep_zp["nullspace_rank"] == 0
    _, comp_zero_t, rep_zt = orthogonalize_task_vectors_polar_nullspace(
        [torch.zeros((4, 6)), torch.zeros((4, 6))], torch.eye(6)
    )
    assert torch.equal(comp_zero_t, torch.zeros((4, 6)))
    assert rep_zt["total_stacked_rank"] == 0
    q_du, _ = torch.linalg.qr(torch.randn((4, 2), generator=gen, dtype=torch.float64))
    q_dv, _ = torch.linalg.qr(torch.randn((6, 2), generator=gen, dtype=torch.float64))
    w_r2 = (q_du @ torch.diag(torch.tensor([2.5, 1.2], dtype=torch.float64)) @ q_dv.T).to(
        w_ok.dtype
    )
    _, comp_dup, rep_dup = orthogonalize_task_vectors_polar_nullspace([w_r2, w_r2], None)
    assert rep_dup["allocated_ranks"] == [2, 2]
    assert float((comp_dup - w_r2).abs().max().item()) < 1e-4
    _, _, rep_wf_dup = orthogonalize_task_vectors_polar_nullspace([w_ok, w_ok], None)
    assert rep_wf_dup["allocated_ranks"] == [2, 2]

    _, comp_1d_z, rep_1dz = orthogonalize_task_vectors_polar_nullspace(
        [torch.ones(4), torch.ones(4)], torch.zeros((4, 4))
    )
    assert rep_1dz["nullspace_rank"] == 0 and torch.equal(comp_1d_z, torch.zeros(4))
    _, comp_1d_zt, _ = orthogonalize_task_vectors_polar_nullspace(
        [torch.zeros(4), torch.zeros(4)], torch.eye(4)
    )
    assert torch.equal(comp_1d_zt, torch.zeros(4))

    # 5. execute_verified_tangent_task_composition validation, buckets, pre-filled LoRA, & receipt tampering
    model, axes = _make_4domain_moe_testbed(num_layers=1, seed=20261237)
    eval_chunks = _make_domain_chunks(chunks_per_domain=2, seed_offset=400)
    with pytest.raises(ValueError, match="non-empty nn.Module"):
        execute_verified_tangent_task_composition(nn.Module(), [{"a": w_ok}], eval_fn=lambda _m: {"E0_a": 1.0})
    with pytest.raises(TypeError, match="callable"):
        execute_verified_tangent_task_composition(model, [{"blocks.0.mlp.router.weight": torch.zeros((4, 40))}], eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="TangentTaskVectorConfig"):
        execute_verified_tangent_task_composition(model, [{"blocks.0.mlp.router.weight": torch.zeros((4, 40))}], config="bad", eval_chunks=eval_chunks, score_from=8)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="eval_fn or"):
        execute_verified_tangent_task_composition(nn.Linear(4, 4), [{"weight": torch.ones((4, 4))}])
    with pytest.raises(ValueError, match="p_null mapping"):
        execute_verified_tangent_task_composition(model, [{"blocks.0.mlp.router.weight": torch.zeros((4, 40))}], p_null={}, eval_chunks=eval_chunks, score_from=8)
    with pytest.raises(TypeError, match="p_null"):
        execute_verified_tangent_task_composition(model, [{"blocks.0.mlp.router.weight": torch.zeros((4, 40))}], p_null="bad", eval_chunks=eval_chunks, score_from=8)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="domain_inputs"):
        execute_verified_tangent_task_composition(model, [{"blocks.0.mlp.router.weight": torch.zeros((4, 40))}], domain_inputs="bad", eval_chunks=eval_chunks, score_from=8)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not found"):
        execute_verified_tangent_task_composition(model, [{"missing.key": torch.ones((2, 2))}], eval_chunks=eval_chunks, score_from=8)
    with pytest.raises(ValueError, match="Delta shape"):
        execute_verified_tangent_task_composition(model, [{"blocks.0.mlp.router.weight": torch.ones((2, 2))}], eval_chunks=eval_chunks, score_from=8)
    with pytest.raises(ValueError, match="delta_format"):
        execute_verified_tangent_task_composition(model, [{"blocks.0.mlp.router.weight": torch.zeros((4, 40))}], delta_format="bad", eval_chunks=eval_chunks, score_from=8)
    with pytest.raises(ValueError, match="not be empty"):
        execute_verified_tangent_task_composition(model, [], eval_chunks=eval_chunks, score_from=8)
    with pytest.raises(ValueError, match="not be empty"):
        execute_verified_tangent_task_composition(model, {}, eval_chunks=eval_chunks, score_from=8)
    with pytest.raises(TypeError, match="Sequence or Mapping"):
        execute_verified_tangent_task_composition(model, "bad", eval_chunks=eval_chunks, score_from=8)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty Mapping"):
        execute_verified_tangent_task_composition(model, [{}], eval_chunks=eval_chunks, score_from=8)
    with pytest.raises(ValueError, match="empty"):
        execute_verified_tangent_task_composition(model, [{" ": torch.ones(2)}], eval_chunks=eval_chunks, score_from=8)

    for bad_chunks in (
        [],
        [{"expert_id": "E0_a"}],
        [{"expert_id": "E0_a", "ids": [0]}],
        [{"expert_id": "E0_a", "ids": [-1, 2]}],
        [{"expert_id": "E0_a", "ids": [0, 999]}],
        [{"expert_id": "E0_a", "ids": [0, 1]}, {"expert_id": "E0_a", "ids": [0, 1, 2]}],
        [{"expert_id": "E0_a", "ids": [0, 1, 2]}],
        [{"expert_id": "E1_only", "ids": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]}],
    ):
        with pytest.raises(ValueError):
            execute_verified_tangent_task_composition(
                model, [{"blocks.0.mlp.router.weight": torch.zeros((4, 40))}], eval_chunks=bad_chunks, score_from=8
            )

    # OpenDiLoCo bucket delta_format + pre-filled LoRA B + prose_keys + domain_inputs + non-ortho router
    base_buckets = extract_moe_trainable_buckets(model, num_router_fragments=1)
    b_d1 = {k: torch.zeros_like(v) for k, v in base_buckets.items()}
    b_d2 = {k: torch.zeros_like(v) for k, v in base_buckets.items()}
    with torch.no_grad():
        exp1 = model.blocks[0].mlp.experts["1"]  # type: ignore[union-attr]
        exp1.w3_B[:, 0].copy_(0.05 * axes["e1"])
        exp1.w3_A[0, 16:18].copy_(0.05)
    b_d1["expert_1"][-40 * 8 :] = 0.05
    b_d2["expert_1"][-40 * 8 :] = 0.04
    b_d1["router_frag_0"][: 4 * 40] = 0.02
    b_d2["router_frag_0"][: 4 * 40] = 0.02
    _, bucket_rec = execute_verified_tangent_task_composition(
        model,
        [b_d1, b_d2],
        delta_format="buckets",
        num_router_fragments=1,
        prose_keys={"x": torch.randn((8, 40), generator=gen), "act": torch.randn((8, 80), generator=gen)},
        domain_inputs={"x": torch.randn((8, 40), generator=gen), "act": torch.randn((8, 80), generator=gen)},
        eval_fn=lambda m: (
            {"E0_general_fineweb": 1.0, "E1_code": 1.2}
            if float(m.blocks[0].mlp.experts["1"].w3_B.abs().sum().item()) <= 0.1  # type: ignore[union-attr]
            else {"E0_general_fineweb": 1.001, "E1_code": 0.9}
        ),
        config=TangentTaskVectorConfig(orthogonalize_router=False, enforce_all_domain_guard=False, score_from=8),
    )
    assert bucket_rec["accepted"] is True
    assert verify_tangent_composition_receipt_provenance(bucket_rec) is True

    # Bucket mismatch errors, frozen expert param, and omitted expert bucket
    for bad_bucket in (
        {"router_frag_0": torch.zeros(1)},
        {"router_frag_0": torch.cat([b_d1["router_frag_0"], torch.zeros(1)])},
        {"router_frag_0": b_d1["router_frag_0"][: 4 * 40 + 1]},
        {"expert_1": torch.zeros(1)},
        {"expert_1": torch.cat([b_d1["expert_1"], torch.zeros(1)])},
    ):
        with pytest.raises(ValueError, match="Bucket"):
            execute_verified_tangent_task_composition(
                model, [bad_bucket], delta_format="buckets", num_router_fragments=1, eval_fn=lambda _m: {"E0_a": 1.0}
            )
    partial_b = {"router_frag_0": b_d1["router_frag_0"]}
    _, partial_b_rec = execute_verified_tangent_task_composition(
        model, [partial_b], delta_format="buckets", num_router_fragments=1, eval_fn=lambda _m: {"E0_a": 1.0}, inplace=False
    )
    assert partial_b_rec["accepted"] is False
    model.ln.weight.requires_grad_(False)
    exp1.w1_A.requires_grad_(False)
    try:
        frozen_b = {
            k: torch.zeros_like(v)
            for k, v in extract_moe_trainable_buckets(model, num_router_fragments=1).items()
        }
        _, frozen_b_rec = execute_verified_tangent_task_composition(
            model, [frozen_b], delta_format="buckets", num_router_fragments=1, eval_fn=lambda _m: {"E0_a": 1.0}, inplace=False
        )
        assert frozen_b_rec["status"] == "NOOP_UNCHANGED"
    finally:
        model.ln.weight.requires_grad_(True)
        exp1.w1_A.requires_grad_(True)

    # LoRA & operator validation + partial dicts + projector lookup + prose_keys tensor + k_d==0 LoRA fold + eval_fn checks
    for bad_op_delta, err_re in (
        ({"blocks.99.mlp.experts.1.w3": torch.ones((40, 80))}, "block index"),
        ({"blocks.0.mlp.experts.99.w3": torch.ones((40, 80))}, "missing expert"),
        ({"blocks.0.mlp.experts.1.w3": torch.ones((2, 2))}, "Effective LoRA"),
        ({"blocks.0.mlp.experts.1.w3_B": torch.ones((2, 2))}, "LoRA B"),
        ({"blocks.0.mlp.experts.1.w3_A": torch.ones((2, 2))}, "LoRA A"),
    ):
        with pytest.raises(ValueError, match=err_re):
            execute_verified_tangent_task_composition(model, [bad_op_delta], eval_fn=lambda _m: {"E0_a": 1.0})

    with pytest.raises(ValueError, match="general_prefix|'E0_'"):
        execute_verified_tangent_task_composition(
            model,
            [{"blocks.0.mlp.router.weight": torch.zeros((4, 40))}],
            probe_chunks=[{"expert_id": "E1_only", "ids": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]}],
            eval_fn=lambda _m: {"E0_a": 1.0},
            score_from=8,
        )

    # Pre-filled LoRA B with composed_delta == 0 after P_0 projection hits k_d == 0 early return
    _, kd0_rec = execute_verified_tangent_task_composition(
        model,
        [
            {"blocks.0.mlp.experts.1.w3_B": torch.ones((40, 8))},
            {"blocks.0.mlp.experts.1.w3_A": torch.zeros((8, 80))},
        ],
        p_null={"blocks.0.mlp.experts.1.w3": torch.zeros((80, 80))},
        eval_fn=lambda _m: {"E0_a": 1.0},
        inplace=False,
    )
    assert kd0_rec["status"] == "NOOP_UNCHANGED"

    # Output-side p_null, unmatched dim p_null, 1D layernorm projection, partial task dict, and prose_keys tensor
    _, out_p_rec = execute_verified_tangent_task_composition(
        model,
        [
            {
                "blocks.0.mlp.experts.1.w3": 0.1 * torch.ones((40, 80)),
                "ln.weight": 0.01 * torch.ones(40),
                "blocks.0.mlp.router.weight": 0.05 * torch.ones((4, 40)),
            },
            {"blocks.0.mlp.router.weight": 0.02 * torch.ones((4, 40))},
        ],
        p_null={
            "blocks.0.mlp.experts.1.w3": torch.eye(40),
            "ln.weight": torch.eye(40),
            "act": torch.eye(9),
        },
        config=TangentTaskVectorConfig(protect_layernorm=False, project_router_into_nullspace=False),
        eval_fn=lambda _m: {"E0_a": 1.0, "E1_c": 1.0},
        inplace=False,
    )
    assert out_p_rec["operators_composed"] == 3

    _, pk_tensor_rec = execute_verified_tangent_task_composition(
        model,
        [{"blocks.0.mlp.experts.1.w3": torch.zeros((40, 80)), "blocks.0.mlp.router.weight": torch.zeros((4, 40))}],
        prose_keys=torch.randn((8, 80), generator=gen),
        p_null=None,
        eval_fn=lambda _m: {"domain_nlls": {"nll_mean": 1.0, "E0_a": 1.0}},
        inplace=False,
    )
    assert pk_tensor_rec["status"] == "NOOP_UNCHANGED"

    _, out_cand_rec = execute_verified_tangent_task_composition(
        model,
        [{"blocks.0.mlp.experts.1.w3": 0.05 * torch.ones((40, 80))}],
        p_null={"act": torch.eye(40)},
        eval_fn=lambda _m: {"E0_a": 1.0, "E1_b": 1.0},
        inplace=False,
    )
    assert out_cand_rec["operator_reports"][0]["projection_side"] == "output"

    # eval_fn edge errors & non-finite guard checks
    for bad_eval_res in ({}, {"domain_nlls": {}}):
        with pytest.raises(ValueError, match="non-empty"):
            execute_verified_tangent_task_composition(
                model,
                [{"blocks.0.mlp.router.weight": torch.zeros((4, 40))}],
                eval_fn=lambda _m, res=bad_eval_res: res,
            )
    for bad_parent in ({"E0_a": True}, {"E0_a": 1.0, "non_finite_outputs": 1}, {"E1_only": 1.0}):
        with pytest.raises(ValueError, match="Parent model"):
            execute_verified_tangent_task_composition(
                model,
                [{"blocks.0.mlp.router.weight": torch.zeros((4, 40))}],
                eval_fn=lambda _m, res=bad_parent: res,
            )
    with pytest.raises(ValueError, match="retained"):
        execute_verified_tangent_task_composition(
            model,
            [{"blocks.0.mlp.router.weight": torch.zeros((4, 40))}],
            eval_fn=lambda _m: {"E0_a": 1.0},
            unlearn_domain_prefixes=("E0_",),
        )
    cand_state = {"calls": 0}

    def _missing_domain_eval(_m: nn.Module) -> dict[str, float]:
        cand_state["calls"] += 1
        return {"E0_a": 1.0, "E1_b": 1.0} if cand_state["calls"] == 1 else {"E0_a": 1.0}

    _, miss_dom_rec = execute_verified_tangent_task_composition(
        model,
        [{"blocks.0.mlp.router.weight": 0.05 * torch.ones((4, 40))}],
        eval_fn=_missing_domain_eval,
        inplace=False,
    )
    assert miss_dom_rec["status"] == "ROLLED_BACK_NON_FINITE"

    # Tampered receipt & invalid receipt verification
    with pytest.raises(TypeError, match="mapping"):
        verify_tangent_composition_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_tangent_composition_receipt_provenance({"receipt_sha256": "xyz"}) is False
    assert (
        verify_tangent_composition_receipt_provenance(
            {"receipt_sha256": "0" * 64, "object": "wrong", "schema": SCHEMA_VERSION}
        )
        is False
    )
    tampered = dict(bucket_rec)
    tampered["delta_e"] = -999.0
    assert verify_tangent_composition_receipt_provenance(tampered) is False
    tampered_nan = dict(bucket_rec)
    tampered_nan["delta_e"] = float("nan")
    assert verify_tangent_composition_receipt_provenance(tampered_nan) is False

    # 6. AGPL-3.0 header, ZeroStubAudit (CI-safe try/except), PRSecurityScanner, lateral movement, FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn32", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path(
        "pr_gatekeeper_dyn32", REPO_ROOT / "scripts" / "pr_gatekeeper.py"
    )
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/tangent_task_vector_surgeon.py", TARGET_MODULE_PATH),
        ("tests/test_tangent_task_vector_surgeon_scenarios.py", TEST_FILE_PATH),
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

    telemetry_json = json.dumps(
        {
            "nll_mean": bucket_rec["nll_mean"],
            "cosine_interference": bucket_rec["cosine_interference"],
            "anse_energy": bucket_rec["anse_energy"],
        },
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+TELEMETRY = {telemetry_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1132,
        pr_title="feat(swarm): tangent-space task-vector orthogonalization and multi-skill composition surgeon for open-weight MoEs",
        pr_body="Implements NEURAL-CTRL-DYN-32 tangent-space task-vector orthogonalization and polar-nullspace MoE composition surgeon.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): tangent-space task-vector orthogonalization and multi-skill composition surgeon for open-weight MoEs"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
