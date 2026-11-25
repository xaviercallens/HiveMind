# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-30: Fisher-Rao KL trust-region weight-space surgeon."""

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
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdRuntimeMLP,
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdLoRAMoEMLP,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.natural_gradient_trust_wsp_surgeon import (
    ANSE_MAXIMUM_PAIN,
    SURGEON_RECEIPT_OBJECT,
    SURGEON_SCHEMA,
    FisherTrustSurgeonConfig,
    compute_kronecker_kl_nats,
    compute_state_sha256,
    estimate_kronecker_fisher_factors,
    execute_verified_fisher_trust_moe_surgery,
    solve_kl_trust_region_wsp_step,
    verify_fisher_trust_receipt_provenance,
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
    """CI-safe stub audit result used when optional ``gwaya`` is absent."""

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
    REPO_ROOT / "src" / "hivemind" / "swarm" / "natural_gradient_trust_wsp_surgeon.py"
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


def _eval_domain_nlls(
    model: CrowdV1WithEngram,
    chunks: list[dict[str, Any]],
    *,
    score_from: int = 4,
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
    return {k: float(sum(v) / float(len(v))) for k, v in sorted(grouped.items())}


def _build_8expert_moe_fixture(
    *,
    num_layers: int = 2,
    split_mode: str = "lora_domain_experts",
    seed: int = 20261130,
) -> dict[str, Any]:
    """Build deterministic 8-expert CrowdV1 MoE with coupled prose/specialist gradients.

    Each specialist domain ``E1..E7`` shares a high-curvature prose carrier subspace
    (``E0``) with general prose while possessing a disjoint orthogonal specialist
    subspace (dims ``8*k .. 8*k+7``). Because specialist tokens also stimulate the
    shared prose carrier, unpreconditioned Euclidean specialist descent leaks into
    ``E0`` and violates prose KL/NLL, whereas Kronecker Fisher-Rao preconditioning
    suppresses prose modes and descends deeply in the pure specialist subspace.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    num_experts = 8
    block_h = 8
    block_i = 16
    hidden_size = num_experts * block_h
    intermediate_size = num_experts * block_i
    vocab_size = num_experts * 8

    cfg = CrowdV1RuntimeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden_size // 4,
        intermediate_size=intermediate_size,
        max_position_embeddings=32,
        engram_buckets=0,
        tie_word_embeddings=True,
    )
    model = CrowdV1WithEngram(cfg)

    u_dirs = torch.zeros((num_experts, hidden_size), dtype=torch.float32)
    for d_idx in range(num_experts):
        vec = torch.tensor(
            [1.0 if i % 2 == 0 else -1.0 for i in range(block_h)],
            dtype=torch.float32,
        )
        u_dirs[d_idx, d_idx * block_h : (d_idx + 1) * block_h] = vec / vec.norm()

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        for d_idx in range(num_experts):
            h0, h1 = d_idx * block_h, (d_idx + 1) * block_h
            t0, t1 = d_idx * 8, (d_idx + 1) * 8
            noise = 0.12 * torch.randn((8, block_h), generator=gen, dtype=torch.float32)
            noise[:, :2] = 0.0
            wte[t0:t1, h0:h1] = noise
            if d_idx == 0:
                wte[t0:t1] += 2.2 * u_dirs[0].unsqueeze(0)
                for k in range(1, num_experts):
                    wte[t0 + k] += 0.55 * u_dirs[k]
                wte[t0 + 2] = 3.2 * u_dirs[0]
            else:
                # Specialist embeddings carry BOTH specialist carrier u_k AND shared prose carrier u_0
                wte[t0:t1] += 1.2 * u_dirs[d_idx].unsqueeze(0) + 1.8 * u_dirs[0].unsqueeze(0)
                wte[t0 + 2] = 3.2 * u_dirs[d_idx]

        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight
        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()

        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0)
            blk.ln2.bias.zero_()
            blk.attn.v.weight[:block_h, :block_h].copy_(0.10 * torch.eye(block_h))
            blk.attn.o.weight[:block_h, :block_h].copy_(0.10 * torch.eye(block_h))
            for d_idx in range(num_experts):
                h0, h1 = d_idx * block_h, (d_idx + 1) * block_h
                i0 = d_idx * block_i
                blk.mlp.w1.weight[i0 : i0 + block_h, h0:h1].copy_(0.85 * torch.eye(block_h))
                blk.mlp.w2.weight[i0 : i0 + block_h, h0:h1].copy_(0.85 * torch.eye(block_h))
                if d_idx == 0:
                    blk.mlp.w3.weight[h0:h1, i0 : i0 + block_h].copy_(0.65 * torch.eye(block_h))
                else:
                    blk.mlp.w3.weight[h0:h1, i0 : i0 + block_h].copy_(0.02 * torch.eye(block_h))

    upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode=split_mode,
        num_experts=num_experts,
        top_k=2,
        lora_rank=4,
        lora_alpha=8.0,
        routing_mode="sigma_topk",
        seed=seed + 7,
    )

    with torch.no_grad():
        for l_idx, blk in enumerate(model.blocks):
            blk.mlp.router.weight.zero_()
            blk.mlp.router.weight[0] = 6.0 * u_dirs[0]
            for d_idx in range(1, num_experts):
                blk.mlp.router.weight[d_idx] = 8.0 * u_dirs[d_idx] + 1.5 * u_dirs[0]
            if split_mode == "lora_domain_experts":
                for d_idx in range(1, num_experts):
                    exp = blk.mlp.experts[str(d_idx)]
                    exp.w3_A.zero_()
                    exp.w3_B.zero_()
                    i0 = d_idx * block_i
                    # Mode 0 aligns with specialist u_k; Mode 1 aligns with prose carrier u_0
                    exp.w3_A[0, i0 : i0 + block_h] = u_dirs[d_idx, d_idx * block_h : (d_idx + 1) * block_h]
                    exp.w3_A[1, 0:block_h] = u_dirs[0, 0:block_h]
                    if l_idx == num_layers - 1:
                        exp.w3_B[:, 0] = 0.05 * u_dirs[d_idx]
                        exp.w3_B[:, 1] = -0.05 * u_dirs[0]

    domain_names = (
        "E0_general_fineweb",
        "E1_python_code",
        "E2_math_reasoning",
        "E3_formal_logic",
        "E4_physics_stem",
        "E5_exoplanet",
        "E6_stellar_spec",
        "E7_defensive_cyber",
    )

    def _make_split(n_per_dom: int, split_gen: torch.Generator) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for d_idx, eid in enumerate(domain_names):
            lo, hi = d_idx * 8, (d_idx + 1) * 8
            tgt = lo + 2
            for _ in range(n_per_dom):
                if d_idx == 0:
                    toks = [1, 3, 4, 5, 6, 7, 2, 2, 2, 2, 2, 2]
                else:
                    toks = torch.randint(lo, hi, (12,), generator=split_gen).tolist()
                    for pos in range(4, 12):
                        toks[pos] = tgt
                rows.append({"expert_id": eid, "ids": toks})
        return rows

    cal_chunks = _make_split(3, torch.Generator(device="cpu").manual_seed(seed + 101))
    eval_chunks = _make_split(3, torch.Generator(device="cpu").manual_seed(seed + 202))
    return {
        "model": model,
        "cal_chunks": cal_chunks,
        "eval_chunks": eval_chunks,
        "u_dirs": u_dirs,
        "domain_names": domain_names,
    }


# ---------------------------------------------------------------------------
# S1: 8-Expert MoE Fisher-Rao Trust-Region vs Euclidean Norm-Matched Edit
# ---------------------------------------------------------------------------


def test_s1_synthetic_8expert_moe_fisher_trust_beats_euclidean_norm_matched() -> None:
    """S1: Fisher-Rao KL trust-region WSP beats Euclidean norm-matched edit at <= +0.3% prose drift."""
    fix = _build_8expert_moe_fixture(num_layers=1, split_mode="lora_domain_experts", seed=20261130)
    fr_model: CrowdV1WithEngram = fix["model"]
    euc_model = copy.deepcopy(fr_model)
    cal_chunks = fix["cal_chunks"]
    eval_chunks = fix["eval_chunks"]

    pre_nlls = _eval_domain_nlls(fr_model, eval_chunks, score_from=1)
    target_factors = ("w3_B",)
    max_kl_nats = 0.005

    receipt = execute_verified_fisher_trust_moe_surgery(
        fr_model,
        cal_chunks,
        eval_chunks=eval_chunks,
        target_factors=target_factors,
        max_kl_nats=max_kl_nats,
        damping=1e-4,
        use_specialist_ray_damping=False,
        max_frobenius_norm=0.8,
        general_guard_max_ratio=1.003,
        domain_guard_max_ratio=1.01,
        score_from=1,
        inplace=True,
    )

    assert receipt["object"] == SURGEON_RECEIPT_OBJECT
    assert receipt["schema"] == SURGEON_SCHEMA
    assert verify_fisher_trust_receipt_provenance(receipt) is True
    assert receipt["accepted"] is True
    assert receipt["committed"] is True
    assert receipt["rolled_back"] is False
    assert receipt["status"] == "ACCEPTED"
    assert receipt["guard_ok"] is True
    assert receipt["kl_ok"] is True
    assert receipt["delta_e"] < 0.0
    assert receipt["total_kl_nats"] <= max_kl_nats + 1e-7
    assert receipt["general_pct_vs_base"] <= 0.3
    assert receipt["total_euclidean_kl_nats"] > receipt["total_kl_nats"]

    fr_nlls = _eval_domain_nlls(fr_model, eval_chunks, score_from=1)
    fr_spec_before = sum(pre_nlls[k] for k in pre_nlls if not k.startswith("E0_")) / 7.0
    fr_spec_after = sum(fr_nlls[k] for k in fr_nlls if not k.startswith("E0_")) / 7.0
    fr_spec_drop = fr_spec_before - fr_spec_after
    assert fr_spec_drop > 0.80

    # Construct exact Frobenius-norm-matched Euclidean edit along the raw specialist gradient
    # from the same calibration pass to verify Euclidean violates prose KL/NLL and gains less.
    from hivemind.swarm.natural_gradient_trust_wsp_surgeon import (
        _collect_crowd_kronecker_statistics,
        _discover_mlp_blocks,
        _enumerate_target_units,
        _validate_chunks,
    )

    cfg_probe = FisherTrustSurgeonConfig(target_factors=target_factors, max_kl_nats=max_kl_nats, score_from=1)
    blocks, has_crowd = _discover_mlp_blocks(euc_model)
    euc_units = _enumerate_target_units(
        blocks,
        has_crowd_blocks=has_crowd,
        target_layer_set={0},
        target_expert_set=None,
        target_factor_set=set(target_factors),
        cfg=cfg_probe,
    )
    ids_cal, eids_cal = _validate_chunks(cal_chunks, "cal", general_prefix="E0_", score_from=1)
    prose_cap, _, auto_targets = _collect_crowd_kronecker_statistics(
        euc_model, euc_units, ids_cal, eids_cal, cfg=cfg_probe
    )

    frob_by_unit = {r["unit"]: float(r["frobenius_norm"]) for r in receipt["unit_reports"]}
    euc_total_kl = 0.0
    with torch.no_grad():
        for u in euc_units:
            u_name = u["name"]
            if u_name not in frob_by_unit:
                continue
            target_frob = frob_by_unit[u_name]
            grad_dir = auto_targets[u_name]
            g_norm = float(torch.linalg.matrix_norm(grad_dir, ord="fro").item())
            euc_delta = (target_frob / max(g_norm, 1e-18)) * grad_dir
            xp, gp = prose_cap[u_name]
            a_p, g_p = estimate_kronecker_fisher_factors(xp, gp, damping=1e-4)
            euc_total_kl += compute_kronecker_kl_nats(euc_delta, a_p, g_p)
            u["param"].add_(euc_delta.to(dtype=u["param"].dtype))

    euc_nlls = _eval_domain_nlls(euc_model, eval_chunks, score_from=1)
    euc_e0_pct = 100.0 * (euc_nlls["E0_general_fineweb"] / pre_nlls["E0_general_fineweb"] - 1.0)
    euc_spec_after = sum(euc_nlls[k] for k in euc_nlls if not k.startswith("E0_")) / 7.0
    euc_spec_drop = fr_spec_before - euc_spec_after

    assert euc_total_kl > max_kl_nats
    assert euc_e0_pct > 1.0
    assert fr_spec_drop > euc_spec_drop


# ---------------------------------------------------------------------------
# S2: Exact Quadratic Fisher-Rao KL Bound & KKT Optimality
# ---------------------------------------------------------------------------


def test_s2_exact_quadratic_fisher_rao_kl_bound_and_kkt_optimality() -> None:
    """S2: Closed-form K-FAC trust-region step satisfies 1/2 tr(G dW A dW^T) <= max_kl_nats + 1e-7."""
    gen = torch.Generator(device="cpu").manual_seed(202602)

    # 1. Verify 2D/3D and bfloat16/float32 estimation + diagnostics
    x3d = torch.randn((4, 8, 12), generator=gen, dtype=torch.bfloat16)
    g3d = torch.randn((4, 8, 6), generator=gen, dtype=torch.float32)
    a_mat, g_mat, f_diag = estimate_kronecker_fisher_factors(
        x3d,
        g3d,
        damping=1e-4,
        scale_damping_by_trace=True,
        return_diagnostics=True,
    )
    assert a_mat.dtype == torch.float64 and g_mat.dtype == torch.float64
    assert a_mat.shape == (12, 12) and g_mat.shape == (6, 6)
    assert f_diag["n_samples"] == 32
    assert f_diag["a_min_eig"] > 0.0 and f_diag["g_min_eig"] > 0.0

    # 2. Sweep rectangular & square shapes across trust radii
    for (d_out, d_in), kl_radius in (
        ((6, 12), 1e-4),
        ((10, 10), 0.005),
        ((16, 8), 0.05),
        ((8, 14), 0.5),
    ):
        x_p = torch.randn((48, d_in), generator=gen, dtype=torch.float32)
        g_p = torch.randn((48, d_out), generator=gen, dtype=torch.float32)
        # Rank-1 prose amplification on coordinate 0 to induce strong curvature anisotropy
        x_p[:, 0] *= 6.0
        g_p[:, 0] *= 5.0

        a_f, g_f = estimate_kronecker_fisher_factors(x_p, g_p, damping=1e-4)
        target = torch.randn((d_out, d_in), generator=gen, dtype=torch.float32)
        target[0, 0] += 8.0

        delta_w, step_diag = solve_kl_trust_region_wsp_step(
            target,
            a_f,
            g_f,
            max_kl_nats=kl_radius,
            ridge_jitter=1e-12,
        )
        exact_kl = 0.5 * float(torch.trace(g_f @ delta_w @ a_f @ delta_w.T).item())
        assert exact_kl <= kl_radius + 1e-7
        assert exact_kl == pytest.approx(kl_radius, rel=1e-6, abs=1e-12)
        assert step_diag["kl_bound_satisfied"] is True
        assert step_diag["kl_nats"] == pytest.approx(exact_kl, rel=1e-10, abs=1e-14)
        assert step_diag["euclidean_kl_nats"] > step_diag["kl_nats"]

        # Verify KKT optimality: any competitor scaled to the same KL boundary has <= alignment
        target64 = target.to(torch.float64)
        for _ in range(5):
            alt = torch.randn((d_out, d_in), generator=gen, dtype=torch.float64)
            alt_kl = compute_kronecker_kl_nats(alt, a_f, g_f)
            alt_scaled = math.sqrt(kl_radius / alt_kl) * alt
            assert float(torch.sum(target64 * delta_w).item()) >= float(torch.sum(target64 * alt_scaled).item()) - 1e-10

    # 3. Verify rescale_only_if_exceeds, max_frobenius_norm, and zero target handling
    a_id = torch.eye(4, dtype=torch.float64)
    g_id = torch.eye(3, dtype=torch.float64)
    small_target = 0.01 * torch.ones((3, 4), dtype=torch.float64)
    d_small, diag_small = solve_kl_trust_region_wsp_step(
        small_target, a_id, g_id, max_kl_nats=1.0, rescale_only_if_exceeds=True
    )
    assert diag_small["trust_scale"] == pytest.approx(1.0)
    assert torch.allclose(d_small, small_target)

    d_capped, diag_capped = solve_kl_trust_region_wsp_step(
        torch.ones((3, 4), dtype=torch.float64),
        a_id,
        g_id,
        max_kl_nats=10.0,
        max_frobenius_norm=0.25,
    )
    assert diag_capped["frobenius_norm"] == pytest.approx(0.25, rel=1e-7)
    assert diag_capped["kl_nats"] <= 10.0 + 1e-7

    d_zero, diag_zero = solve_kl_trust_region_wsp_step(torch.zeros((3, 4)), a_id, g_id)
    assert float(d_zero.abs().max().item()) == 0.0
    assert diag_zero["kl_nats"] == 0.0


# ---------------------------------------------------------------------------
# S3: Combined P_0 Null-Space + Fisher Trust-Region Invariance
# ---------------------------------------------------------------------------


def test_s3_combined_p0_nullspace_and_fisher_trust_region_invariance() -> None:
    """S3: Combined P_0 nullspace + Fisher trust-region satisfies exact invariance and KL bound."""
    gen = torch.Generator(device="cpu").manual_seed(202603)
    d_out, d_in = 8, 12

    # Build orthonormal protected prose basis U_in (rank 4) -> P_0 = I - U_in U_in^T
    k_raw = torch.randn((d_in, 4), generator=gen, dtype=torch.float64)
    q_in, _ = torch.linalg.qr(k_raw, mode="reduced")
    p0_right = torch.eye(d_in, dtype=torch.float64) - q_in @ q_in.T

    # Build coupled positive-definite A_prose (non-commuting with P_0) and G_prose
    x_p = torch.randn((32, d_in), generator=gen, dtype=torch.float64) + 3.0 * torch.randn(
        (32, 4), generator=gen, dtype=torch.float64
    ) @ q_in.T
    g_p = torch.randn((32, d_out), generator=gen, dtype=torch.float64)
    a_prose, g_prose = estimate_kronecker_fisher_factors(x_p, g_p, damping=1e-4)
    target = torch.randn((d_out, d_in), generator=gen, dtype=torch.float64)

    delta_right, diag_right = solve_kl_trust_region_wsp_step(
        target, a_prose, g_prose, max_kl_nats=0.005, p_null=p0_right
    )
    assert diag_right["nullspace_mode"] == "right_input_nullspace"
    assert diag_right["nullspace_residual_norm"] <= 1e-12
    assert float(torch.linalg.matrix_norm(delta_right @ q_in, ord="fro").item()) <= 1e-12
    assert compute_kronecker_kl_nats(delta_right, a_prose, g_prose) <= 0.005 + 1e-7

    # Verify Riemannian Schur-complement solution beats naive post-projection at equal KL radius
    delta_unproj, _ = solve_kl_trust_region_wsp_step(target, a_prose, g_prose, max_kl_nats=0.005)
    naive_proj = delta_unproj @ p0_right
    naive_kl = compute_kronecker_kl_nats(naive_proj, a_prose, g_prose)
    naive_scaled = math.sqrt(0.005 / naive_kl) * naive_proj
    assert float(torch.sum(target * delta_right).item()) >= float(torch.sum(target * naive_scaled).item()) - 1e-12

    # Left output nullspace P_0 on (d_out, d_out)
    k_out_raw = torch.randn((d_out, 3), generator=gen, dtype=torch.float64)
    q_out, _ = torch.linalg.qr(k_out_raw, mode="reduced")
    p0_left = torch.eye(d_out, dtype=torch.float64) - q_out @ q_out.T
    delta_left, diag_left = solve_kl_trust_region_wsp_step(
        target, a_prose, g_prose, max_kl_nats=0.005, p_null=p0_left, nullspace_side="left"
    )
    assert diag_left["nullspace_mode"] == "left_output_nullspace"
    assert float(torch.linalg.matrix_norm(q_out.T @ delta_left, ord="fro").item()) <= 1e-12
    assert compute_kronecker_kl_nats(delta_left, a_prose, g_prose) <= 0.005 + 1e-7

    # Auto-resolution on non-square where d_p == d_out, zero nullspace, and soft-filter non-idempotent branch
    delta_auto_left, diag_auto_left = solve_kl_trust_region_wsp_step(
        target, a_prose, g_prose, max_kl_nats=0.005, p_null=p0_left
    )
    assert diag_auto_left["nullspace_mode"] == "left_output_nullspace"
    assert torch.allclose(delta_left, delta_auto_left)

    delta_zero_ns, diag_zero_ns = solve_kl_trust_region_wsp_step(
        target, a_prose, g_prose, max_kl_nats=0.005, p_null=torch.zeros((d_in, d_in), dtype=torch.float64)
    )
    assert diag_zero_ns["nullspace_mode"] == "zero_nullspace"
    assert float(delta_zero_ns.abs().max().item()) == 0.0

    soft_right = 0.5 * p0_right + 0.5 * torch.eye(d_in, dtype=torch.float64)
    soft_left = 0.5 * p0_left + 0.5 * torch.eye(d_out, dtype=torch.float64)
    d_sr, diag_sr = solve_kl_trust_region_wsp_step(target, a_prose, g_prose, p_null=soft_right, nullspace_side="right")
    d_sl, diag_sl = solve_kl_trust_region_wsp_step(target, a_prose, g_prose, p_null=soft_left, nullspace_side="left")
    assert diag_sr["nullspace_mode"] == "right_soft_filter"
    assert diag_sl["nullspace_mode"] == "left_soft_filter"
    assert compute_kronecker_kl_nats(d_sr, a_prose, g_prose) <= 0.005 + 1e-7
    assert compute_kronecker_kl_nats(d_sl, a_prose, g_prose) <= 0.005 + 1e-7

    # End-to-end linear surgery with p_null preserving protected prose keys identically
    lin = nn.Linear(d_in, d_out, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.randn((d_out, d_in), generator=gen, dtype=torch.float32))
    out_keys_before = lin(q_in.T.to(torch.float32)).detach().clone()

    def _eval_lin(mod: nn.Module) -> dict[str, float]:
        assert isinstance(mod, nn.Linear)
        err_p = float((mod(q_in.T.to(torch.float32)) - out_keys_before).pow(2).mean().item())
        err_s = float((mod(torch.ones((1, d_in))) - 2.0).pow(2).mean().item())
        return {"E0_prose": 0.5 + err_p, "E1_spec": 0.5 + err_s}

    rec = execute_verified_fisher_trust_moe_surgery(
        lin,
        prose_activations=x_p,
        prose_grad_outs=g_p,
        target_deltas={"layers.0.mlp.weight": torch.ones((d_out, d_in))},
        eval_fn=_eval_lin,
        p_null={"layers.0.mlp.weight": p0_right},
        max_kl_nats=0.01,
        use_specialist_ray_damping=False,
        require_energy_improvement=False,
    )
    assert rec["accepted"] is True
    assert rec["unit_reports"][0]["nullspace_mode"] == "right_input_nullspace"
    with torch.no_grad():
        drift = float((lin(q_in.T.to(torch.float32)) - out_keys_before).abs().max().item())
    assert drift <= 1e-5


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Atomic Rollback & Multi-Topology Coverage
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_rollback_and_topology_coverage() -> None:
    """S4: Infeasible guards trigger ANSE Maximum Pain (1e6) and exact atomic bitwise rollback."""
    fix = _build_8expert_moe_fixture(num_layers=1, split_mode="lora_domain_experts", seed=202604)
    model: CrowdV1WithEngram = fix["model"]
    cal_chunks = fix["cal_chunks"]
    eval_chunks = fix["eval_chunks"]
    parent_sha = compute_state_sha256(model.state_dict())

    # 1. Infeasible general guard triggers ROLLED_BACK_GUARD_VIOLATION & ANSE Maximum Pain
    rec_guard = execute_verified_fisher_trust_moe_surgery(
        model,
        cal_chunks,
        eval_chunks=eval_chunks,
        target_factors=("w3_B",),
        max_kl_nats=2.0,
        use_specialist_ray_damping=False,
        general_guard_max_ratio=1.0,
        domain_guard_max_ratio=1.01,
        score_from=4,
    )
    assert rec_guard["accepted"] is False
    assert rec_guard["rolled_back"] is True
    assert rec_guard["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert rec_guard["guard_ok"] is False
    assert rec_guard["anse_energy"] == pytest.approx(ANSE_MAXIMUM_PAIN)
    assert rec_guard["final_state_sha256"] == parent_sha
    assert compute_state_sha256(model.state_dict()) == parent_sha
    assert verify_fisher_trust_receipt_provenance(rec_guard) is True

    # 2. Non-improving energy rollback, eval failure rollback, and FIXED_POINT_UNCHANGED
    lin = nn.Linear(8, 4, bias=False)
    lin_sha = compute_state_sha256(lin.state_dict())
    xp = torch.ones((6, 8), dtype=torch.float32)
    gp = torch.ones((6, 4), dtype=torch.float32)
    td = {"layers.0.mlp.weight": torch.ones((4, 8), dtype=torch.float32)}

    rec_noimp = execute_verified_fisher_trust_moe_surgery(
        lin,
        prose_activations=xp,
        prose_grad_outs=gp,
        target_deltas=td,
        eval_fn=lambda _m: (0.8, {"E0_prose": 0.8, "E1_spec": 0.8}),
        require_energy_improvement=True,
    )
    assert rec_noimp["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    assert rec_noimp["rolled_back"] is True
    assert compute_state_sha256(lin.state_dict()) == lin_sha

    calls = {"n": 0}

    def _boom_eval(_m: nn.Module) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("injected evaluation failure")
        return {"per_domain": {"E0_prose": 0.8, "E1_spec": 0.8}}

    rec_boom = execute_verified_fisher_trust_moe_surgery(
        lin,
        prose_activations=xp,
        prose_grad_outs=gp,
        target_deltas=td,
        eval_fn=_boom_eval,
    )
    assert rec_boom["status"] == "ROLLED_BACK_EVAL_FAILURE"
    assert rec_boom["rolled_back"] is True
    assert rec_boom["anse_energy"] == pytest.approx(ANSE_MAXIMUM_PAIN)
    assert compute_state_sha256(lin.state_dict()) == lin_sha

    rec_fp = execute_verified_fisher_trust_moe_surgery(
        lin,
        prose_activations=xp,
        prose_grad_outs=gp,
        target_deltas={"layers.0.mlp.weight": torch.zeros((4, 8))},
        eval_fn=lambda _m: {"E0_prose": 0.8, "E1_spec": 0.8},
    )
    assert rec_fp["status"] == "FIXED_POINT_UNCHANGED"
    assert rec_fp["rolled_back"] is False

    # 3. CrowdFineGrainedMoEMLP, CrowdRuntimeMLP, HF-style .layers, backtracking, inplace=False, per_unit
    fg_fix = _build_8expert_moe_fixture(num_layers=1, split_mode="fine_grained_width_split", seed=202605)
    fg_model: CrowdV1WithEngram = fg_fix["model"]
    fg_sha = compute_state_sha256(fg_model.state_dict())
    rec_fg_copy = execute_verified_fisher_trust_moe_surgery(
        fg_model,
        fg_fix["cal_chunks"],
        target_factors=("w3",),
        layer_indices=[0],
        expert_indices=[1, 2],
        max_kl_nats=0.005,
        kl_budget_scope="per_unit",
        enable_trust_backtracking=True,
        step_damping=0.1,
        score_from=4,
        inplace=False,
    )
    assert compute_state_sha256(fg_model.state_dict()) == fg_sha
    assert rec_fg_copy["accepted"] is True
    assert rec_fg_copy["kl_ok"] is True

    # Test CrowdLoRAMoEMLP non-zero A+B factors, zero-routed counterfactual fallback, and standalone dense/HF blocks
    dense_cfg = CrowdV1RuntimeConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=16,
        max_position_embeddings=16,
        engram_buckets=0,
    )
    dense_mlp = CrowdRuntimeMLP(dense_cfg)
    lora_moe = CrowdLoRAMoEMLP(dense_mlp, num_experts=3, top_k=1, rank=2, seed=42)
    with torch.no_grad():
        lora_moe.router.weight.zero_()
        lora_moe.router.weight[0, 0] = 20.0
        lora_moe.experts["1"].w3_B.fill_(0.1)

    rec_dense = execute_verified_fisher_trust_moe_surgery(
        dense_mlp,
        prose_activations=xp,
        prose_grad_outs=torch.ones((6, 16), dtype=torch.float32),
        spec_activations=xp,
        spec_grad_outs=-torch.ones((6, 16), dtype=torch.float32),
        eval_fn=lambda _m: {"E0_prose": 0.9, "E1_spec": 0.9},
        target_factors=("w1", "w2"),
        max_kl_nats=0.01,
        require_energy_improvement=False,
    )
    assert rec_dense["accepted"] is True

    class _HFExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(8, 12, bias=False)
            self.up_proj = nn.Linear(8, 12, bias=False)
            self.down_proj = nn.Linear(12, 8, bias=False)

    class _HFMoEMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = nn.ModuleDict({"0": _HFExpert(), "1": _HFExpert()})

    class _HFLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = _HFMoEMLP()

    class _HFModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_HFLayer()])

    hf_model = _HFModel()
    rec_hf = execute_verified_fisher_trust_moe_surgery(
        hf_model,
        prose_activations=xp,
        prose_grad_outs=torch.ones((6, 12), dtype=torch.float32),
        target_deltas={"gate_proj": torch.ones((12, 8))},
        eval_fn=lambda _m: {"E0_prose": 0.8, "E1_spec": 0.8},
        target_factors=("gate_proj",),
        require_energy_improvement=False,
    )
    assert rec_hf["accepted"] is True

    # Also test LoRA crowd model with w1_A..w3_B + counterfactual unrouted expert path
    rec_lora_all = execute_verified_fisher_trust_moe_surgery(
        fix["model"],
        cal_chunks,
        target_factors=("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"),
        target_deltas={"blocks.0.mlp.experts.1.w3_B": torch.zeros((64, 4))},
        max_kl_nats=0.002,
        score_from=4,
        inplace=False,
        require_energy_improvement=False,
    )
    assert rec_lora_all["kl_ok"] is True

    # Trigger counterfactual prose path on CrowdLoRAMoEMLP with zero-routed expert 1
    crowd_small = CrowdV1WithEngram(dense_cfg)
    upcycle_crowd_v1_to_sparse_moe(
        crowd_small, split_mode="lora_domain_experts", num_experts=3, top_k=1, lora_rank=2, seed=11
    )
    with torch.no_grad():
        crowd_small.blocks[0].mlp.router.weight.zero_()
        crowd_small.blocks[0].mlp.router.weight[0] = 25.0
        crowd_small.blocks[0].mlp.experts["1"].w3_B.fill_(0.05)
    small_chunks = [
        {"expert_id": "E0_gen", "ids": [0, 1, 2, 3, 4, 5]},
        {"expert_id": "E1_sp", "ids": [6, 7, 8, 9, 10, 11]},
    ]
    rec_cf = execute_verified_fisher_trust_moe_surgery(
        crowd_small,
        small_chunks,
        target_factors=("w3_B",),
        expert_indices=[1],
        max_kl_nats=0.005,
        score_from=1,
        inplace=False,
        require_energy_improvement=False,
    )
    assert rec_cf["kl_ok"] is True


# ---------------------------------------------------------------------------
# S5: Validation Errors, Provenance & Security/Stub Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_provenance_and_security_compliance_gates() -> None:
    """S5: Input validation, SHA-256 provenance checks, and HiveMind security/stub compliance gates."""
    # 1. Config validation
    with pytest.raises(ValueError, match="max_kl_nats"):
        FisherTrustSurgeonConfig(max_kl_nats=0.0)
    with pytest.raises(ValueError, match="damping"):
        FisherTrustSurgeonConfig(damping=-1e-4)
    with pytest.raises(TypeError, match="bool"):
        FisherTrustSurgeonConfig(scale_damping_by_trace="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kl_budget_scope"):
        FisherTrustSurgeonConfig(kl_budget_scope="bad")
    with pytest.raises(ValueError, match="nullspace_side"):
        FisherTrustSurgeonConfig(nullspace_side="bad")
    with pytest.raises(ValueError, match="step_damping"):
        FisherTrustSurgeonConfig(step_damping=-0.1)
    with pytest.raises(ValueError, match="max_frobenius_norm"):
        FisherTrustSurgeonConfig(max_frobenius_norm=0.0)
    with pytest.raises(TypeError, match="int"):
        FisherTrustSurgeonConfig(general_expert_index=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        FisherTrustSurgeonConfig(general_expert_index=-1)
    with pytest.raises(ValueError, match="target_factors"):
        FisherTrustSurgeonConfig(target_factors=())
    with pytest.raises(ValueError, match="Unsupported target_factors"):
        FisherTrustSurgeonConfig(target_factors=("bad_factor",))
    with pytest.raises(ValueError, match="domain_guard_max_ratio"):
        FisherTrustSurgeonConfig(domain_guard_max_ratio=0.95)
    with pytest.raises(ValueError, match="general_guard_max_ratio"):
        FisherTrustSurgeonConfig(general_guard_max_ratio=0.99)
    with pytest.raises(ValueError, match="anse_safe_e0_ratio"):
        FisherTrustSurgeonConfig(anse_safe_e0_ratio=0.9)
    with pytest.raises(ValueError, match=">= 1"):
        FisherTrustSurgeonConfig(backtrack_halvings=0)
    with pytest.raises(ValueError, match="general_prefix"):
        FisherTrustSurgeonConfig(general_prefix=" ")
    with pytest.raises(ValueError, match=">= 1"):
        FisherTrustSurgeonConfig(batch_size=0)
    with pytest.raises(ValueError, match="finite"):
        FisherTrustSurgeonConfig(max_kl_nats=float("nan"))

    # 2. Kronecker Fisher & KL step validation
    x_ok = torch.ones((4, 6), dtype=torch.float32)
    g_ok = torch.ones((4, 3), dtype=torch.float32)
    with pytest.raises(TypeError, match="torch.Tensor"):
        estimate_kronecker_fisher_factors([[1.0]], g_ok)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2D or 3D"):
        estimate_kronecker_fisher_factors(torch.ones(6), g_ok)
    with pytest.raises(TypeError, match="floating-point"):
        estimate_kronecker_fisher_factors(torch.ones((4, 6), dtype=torch.long), g_ok)
    bad_x = x_ok.clone()
    bad_x[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        estimate_kronecker_fisher_factors(bad_x, g_ok)
    with pytest.raises(ValueError, match="sample count"):
        estimate_kronecker_fisher_factors(x_ok, torch.ones((5, 3)))
    with pytest.raises(ValueError, match="leading shape"):
        estimate_kronecker_fisher_factors(torch.ones((2, 4, 6)), torch.ones((4, 2, 3)))
    with pytest.raises(ValueError, match="> 0.0"):
        estimate_kronecker_fisher_factors(x_ok, g_ok, damping=0.0)

    a_ok, g_f_ok = estimate_kronecker_fisher_factors(x_ok, g_ok, damping=1e-2)
    t_ok = torch.ones((3, 6), dtype=torch.float64)
    with pytest.raises(ValueError, match="non-empty 2D"):
        solve_kl_trust_region_wsp_step(torch.ones(6), a_ok, g_f_ok)
    with pytest.raises(ValueError, match="shape"):
        solve_kl_trust_region_wsp_step(t_ok, torch.eye(4), g_f_ok)
    with pytest.raises(ValueError, match="shape"):
        compute_kronecker_kl_nats(t_ok, torch.eye(4), g_f_ok)
    with pytest.raises(ValueError, match="shape"):
        compute_kronecker_kl_nats(t_ok, a_ok, torch.eye(2))
    asym_a = a_ok.clone()
    asym_a[0, 1] += 2.0
    with pytest.raises(ValueError, match="symmetric"):
        solve_kl_trust_region_wsp_step(t_ok, asym_a, g_f_ok)
    with pytest.raises(ValueError, match="strictly positive definite"):
        solve_kl_trust_region_wsp_step(t_ok, -torch.eye(6), g_f_ok)
    with pytest.raises(ValueError, match="nullspace_side"):
        solve_kl_trust_region_wsp_step(t_ok, a_ok, g_f_ok, nullspace_side="invalid")
    with pytest.raises(ValueError, match="square 2D"):
        solve_kl_trust_region_wsp_step(t_ok, a_ok, g_f_ok, p_null=torch.ones((3, 6)))
    asym_p = torch.eye(6)
    asym_p[0, 1] = 2.0
    with pytest.raises(ValueError, match="symmetric"):
        solve_kl_trust_region_wsp_step(t_ok, a_ok, g_f_ok, p_null=asym_p)
    with pytest.raises(ValueError, match="eigenvalues"):
        solve_kl_trust_region_wsp_step(t_ok, a_ok, g_f_ok, p_null=2.0 * torch.eye(6))
    with pytest.raises(ValueError, match="match d_in"):
        solve_kl_trust_region_wsp_step(t_ok, a_ok, g_f_ok, p_null=torch.eye(3), nullspace_side="right")
    with pytest.raises(ValueError, match="match d_out"):
        solve_kl_trust_region_wsp_step(t_ok, a_ok, g_f_ok, p_null=torch.eye(6), nullspace_side="left")
    with pytest.raises(ValueError, match="matches neither"):
        solve_kl_trust_region_wsp_step(t_ok, a_ok, g_f_ok, p_null=torch.eye(5))

    # 3. Orchestrator & provenance validation
    lin = nn.Linear(6, 3, bias=False)
    with pytest.raises(TypeError, match="nn.Module"):
        execute_verified_fisher_trust_moe_surgery("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable"):
        execute_verified_fisher_trust_moe_surgery(lin, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mapping"):
        execute_verified_fisher_trust_moe_surgery(lin, target_deltas="bad", eval_fn=lambda _m: {"E0_a": 1.0})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="FisherTrustSurgeonConfig"):
        execute_verified_fisher_trust_moe_surgery(lin, config="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Either calibration_chunks"):
        execute_verified_fisher_trust_moe_surgery(lin)
    with pytest.raises(ValueError, match="prose_activations"):
        execute_verified_fisher_trust_moe_surgery(lin, eval_fn=lambda _m: {"E0_a": 1.0})
    with pytest.raises(ValueError, match="Either target_deltas"):
        execute_verified_fisher_trust_moe_surgery(
            lin, prose_activations=x_ok, prose_grad_outs=g_ok, eval_fn=lambda _m: {"E0_a": 1.0}
        )
    with pytest.raises(ValueError, match="layer_indices"):
        execute_verified_fisher_trust_moe_surgery(
            lin,
            prose_activations=x_ok,
            prose_grad_outs=g_ok,
            target_deltas={"layers.0.mlp.weight": t_ok},
            eval_fn=lambda _m: {"E0_a": 1.0},
            layer_indices=[5],
        )
    with pytest.raises(ValueError, match="did not match any targeted unit"):
        execute_verified_fisher_trust_moe_surgery(
            lin,
            prose_activations=x_ok,
            prose_grad_outs=g_ok,
            target_deltas={"layers.0.mlp.weight": t_ok},
            eval_fn=lambda _m: {"E0_a": 1.0},
            p_null=torch.eye(9),
        )
    with pytest.raises(ValueError, match="sample counts"):
        execute_verified_fisher_trust_moe_surgery(
            lin,
            prose_activations=x_ok,
            prose_grad_outs=g_ok,
            spec_activations=torch.ones((3, 6)),
            spec_grad_outs=torch.ones((5, 3)),
            eval_fn=lambda _m: {"E0_a": 1.0},
        )
    with pytest.raises(ValueError, match="expected"):
        execute_verified_fisher_trust_moe_surgery(
            lin,
            prose_activations=x_ok,
            prose_grad_outs=g_ok,
            target_deltas={"layers.0.mlp.weight": torch.ones((2, 6))},
            eval_fn=lambda _m: {"E0_a": 1.0},
        )
    with pytest.raises(TypeError, match="CrowdV1WithEngram"):
        execute_verified_fisher_trust_moe_surgery(lin, [{"expert_id": "E0_a", "ids": [1, 2]}])
    with pytest.raises(TypeError, match="CrowdV1WithEngram"):
        execute_verified_fisher_trust_moe_surgery(
            lin, eval_chunks=[{"expert_id": "E0_a", "ids": [1, 2]}], prose_activations=x_ok, prose_grad_outs=g_ok, target_deltas={"w": t_ok}
        )

    class _UnsupportedMod(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy_param = nn.Parameter(torch.ones(3))

    with pytest.raises(ValueError, match="Unsupported MLP"):
        execute_verified_fisher_trust_moe_surgery(
            _UnsupportedMod(),
            prose_activations=x_ok,
            prose_grad_outs=g_ok,
            target_deltas={"x": t_ok},
            eval_fn=lambda _m: {"E0_a": 1.0},
        )

    # Chunk & eval validation on CrowdV1WithEngram
    tiny_cfg = CrowdV1RuntimeConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=16,
        max_position_embeddings=16,
        engram_buckets=0,
    )
    tiny_m = CrowdV1WithEngram(tiny_cfg)
    with pytest.raises(ValueError, match="non-empty sequence"):
        execute_verified_fisher_trust_moe_surgery(tiny_m, [])
    with pytest.raises(ValueError, match="expert_id"):
        execute_verified_fisher_trust_moe_surgery(tiny_m, [{"ids": [1, 2]}])
    with pytest.raises(ValueError, match="non-empty string"):
        execute_verified_fisher_trust_moe_surgery(tiny_m, [{"expert_id": " ", "ids": [1, 2]}])
    with pytest.raises(TypeError, match="sequence or tensor"):
        execute_verified_fisher_trust_moe_surgery(tiny_m, [{"expert_id": "E0_a", "ids": "123"}])
    with pytest.raises(ValueError, match="non-negative"):
        execute_verified_fisher_trust_moe_surgery(tiny_m, [{"expert_id": "E0_a", "ids": [1, -1]}])
    with pytest.raises(ValueError, match="identical sequence lengths"):
        execute_verified_fisher_trust_moe_surgery(
            tiny_m, [{"expert_id": "E0_a", "ids": [1, 2]}, {"expert_id": "E1_b", "ids": [1, 2, 3]}]
        )
    with pytest.raises(ValueError, match="too short"):
        execute_verified_fisher_trust_moe_surgery(
            tiny_m,
            [{"expert_id": "E0_a", "ids": [1, 2, 3]}, {"expert_id": "E1_b", "ids": [1, 2, 3]}],
            score_from=4,
        )
    with pytest.raises(ValueError, match="both general"):
        execute_verified_fisher_trust_moe_surgery(tiny_m, [{"expert_id": "E0_a", "ids": [1, 2, 3]}])
    with pytest.raises(ValueError, match="identical keys"):
        execute_verified_fisher_trust_moe_surgery(
            tiny_m,
            [{"expert_id": "E0_a", "ids": [1, 2, 3]}, {"expert_id": "E1_b", "ids": [4, 5, 6]}],
            baseline_domain_nll={"E0_only": 1.0},
        )

    bad_blocks = nn.Module()
    bad_blocks.blocks = nn.ModuleList()
    with pytest.raises(ValueError, match="blocks"):
        execute_verified_fisher_trust_moe_surgery(bad_blocks, [{"expert_id": "E0_a", "ids": [1, 2]}])
    bad_blocks.blocks = nn.ModuleList([nn.Linear(4, 4)])
    with pytest.raises(ValueError, match="expose an nn.Module 'mlp'"):
        execute_verified_fisher_trust_moe_surgery(bad_blocks, [{"expert_id": "E0_a", "ids": [1, 2]}])

    with pytest.raises(TypeError, match="mapping"):
        verify_fisher_trust_receipt_provenance("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        compute_state_sha256({})
    with pytest.raises(ValueError, match="non-empty string"):
        compute_state_sha256({"": torch.ones(2)})
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_state_sha256({"a": [1.0]})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-finite"):
        compute_state_sha256({"a": torch.tensor([float("nan")])})
    assert verify_fisher_trust_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert verify_fisher_trust_receipt_provenance({"receipt_sha256": "0" * 64, "object": "wrong"}) is False
    assert (
        verify_fisher_trust_receipt_provenance(
            {"receipt_sha256": "0" * 64, "object": SURGEON_RECEIPT_OBJECT, "schema": "wrong"}
        )
        is False
    )
    assert (
        verify_fisher_trust_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": SURGEON_RECEIPT_OBJECT,
                "schema": SURGEON_SCHEMA,
                "bad": float("nan"),
            }
        )
        is False
    )

    # 4. Repository AGPL-3.0, optional ZeroStubAudit, PRSecurityScanner, lateral movement & gatekeeper gates
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn30", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn30", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/natural_gradient_trust_wsp_surgeon.py", TARGET_MODULE_PATH),
        ("tests/test_natural_gradient_trust_wsp_surgeon_scenarios.py", TEST_FILE_PATH),
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
        {"nll_mean": 0.42, "kl_trust_radius": 0.005, "anse_energy": -0.18, "accepted": True},
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+RECEIPT = {sample_receipt_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1130,
        pr_title="feat(swarm): Fisher-Rao natural-gradient KL trust-region weight-space surgeon for open-weight MoEs",
        pr_body="Implements NEURAL-CTRL-DYN-30 K-FAC Fisher-Rao trust-region surgeon and deterministic CPU scenarios.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): Fisher-Rao natural-gradient KL trust-region weight-space surgeon for open-weight MoEs"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
