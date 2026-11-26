# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-44: Kronecker-factored Riemannian null-space transport."""

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
from hivemind.security.owner_order_guardrail import (
    RequestIntentCategory,
    classify_request_intent,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdRuntimeMLP,
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import upcycle_crowd_v1_to_sparse_moe
from hivemind.swarm.kronecker_riemannian_nullspace_transport import (
    ANSE_MAXIMUM_PAIN,
    KRONECKER_TRANSPORT_RECEIPT_OBJECT,
    KRONECKER_TRANSPORT_SCHEMA,
    KroneckerRiemannianTransportConfig,
    compute_kronecker_kl_nats,
    compute_kronecker_whitened_nullspace_operators,
    compute_state_sha256,
    diagnose_theorem_3_7_obstruction,
    execute_verified_kronecker_riemannian_surgery,
    solve_kronecker_riemannian_transport_step,
    transport_delta_between_curved_layers,
    transport_factorized_lora_with_kronecker_metric,
    verify_kronecker_riemannian_receipt_provenance,
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
    REPO_ROOT / "src" / "hivemind" / "swarm" / "kronecker_riemannian_nullspace_transport.py"
)
TEST_FILE_PATH = Path(__file__).resolve()
ROADMAP_DOC_PATH = REPO_ROOT / "docs" / "NEURO_CONTROLLER_ROADMAP.md"
PLAN_DOC_PATH = REPO_ROOT / "docs" / "NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md"
EXPECTED_AGPL_HEADER = (
    "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
)


@pytest.fixture(autouse=True, scope="module")
def _single_thread_cpu_torch() -> Any:
    """Keep CPU PyTorch deterministic and fast across small tensor fixtures."""
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prev_threads)


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
    score_from: int = 1,
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


def _build_curved_8expert_moe_fixture(
    *,
    num_layers: int = 2,
    split_mode: str = "lora_domain_experts",
    seed: int = 20264400,
) -> dict[str, Any]:
    """Build deterministic 8-expert CrowdV1 MoE with anisotropic deep-layer prose coupling."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    num_experts = 8
    block_h = 4
    block_i = 8
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
    h1_dirs = torch.zeros((num_experts, hidden_size), dtype=torch.float32)
    for d_idx in range(num_experts):
        vec = torch.tensor(
            [1.0 if i % 2 == 0 else -1.0 for i in range(block_h)],
            dtype=torch.float32,
        )
        vec_sec = torch.tensor(
            [1.0 if i < (block_h // 2) else -1.0 for i in range(block_h)],
            dtype=torch.float32,
        )
        u_dirs[d_idx, d_idx * block_h : (d_idx + 1) * block_h] = vec / vec.norm()
        h1_dirs[d_idx, d_idx * block_h : (d_idx + 1) * block_h] = vec_sec / vec_sec.norm()

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        for d_idx in range(num_experts):
            h0, h1 = d_idx * block_h, (d_idx + 1) * block_h
            t0, t1 = d_idx * 8, (d_idx + 1) * 8
            noise = 0.14 * torch.randn((8, block_h), generator=gen, dtype=torch.float32)
            noise[:, :2] = 0.0
            wte[t0:t1, h0:h1] = noise
            if d_idx == 0:
                wte[t0:t1] += 2.6 * u_dirs[0].unsqueeze(0) + 1.15 * h1_dirs[0].unsqueeze(0)
                for k in range(1, num_experts):
                    wte[t0 + k] += 0.65 * u_dirs[k]
                wte[t0 + 2] = 3.4 * u_dirs[0] + 1.25 * h1_dirs[0]
            else:
                # Curved coupling: specialists stimulate dominant u_0 AND secondary curved prose carrier h1_0
                wte[t0:t1] += (
                    1.15 * u_dirs[d_idx].unsqueeze(0)
                    + 1.9 * u_dirs[0].unsqueeze(0)
                    + 0.95 * h1_dirs[0].unsqueeze(0)
                )
                wte[t0 + 2] = 3.3 * u_dirs[d_idx]

        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight
        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()

        for l_idx, blk in enumerate(model.blocks):
            blk.ln1.weight.fill_(1.0)
            blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0)
            blk.ln2.bias.zero_()
            blk.attn.v.weight[:block_h, :block_h].copy_(0.10 * torch.eye(block_h))
            blk.attn.o.weight[:block_h, :block_h].copy_(0.10 * torch.eye(block_h))
            skew = 1.0 + 0.15 * float(l_idx)
            for d_idx in range(num_experts):
                h0, h1 = d_idx * block_h, (d_idx + 1) * block_h
                i0 = d_idx * block_i
                blk.mlp.w1.weight[i0 : i0 + block_h, h0:h1].copy_(0.85 * skew * torch.eye(block_h))
                blk.mlp.w2.weight[i0 : i0 + block_h, h0:h1].copy_(0.85 * skew * torch.eye(block_h))
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
        seed=seed + 11,
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
                    # Row 0: dominant prose carrier u_0 (captured by V_0); Row 1: curved residual h1_0;
                    # Row 2: specialist carrier u_k
                    exp.w3_A[0, 0:block_h] = u_dirs[0, 0:block_h]
                    exp.w3_A[1, 0:block_h] = h1_dirs[0, 0:block_h]
                    exp.w3_A[2, i0 : i0 + block_h] = u_dirs[d_idx, d_idx * block_h : (d_idx + 1) * block_h]
                    if l_idx == num_layers - 1:
                        exp.w3_B[:, 0] = -0.05 * u_dirs[0]
                        exp.w3_B[:, 1] = -0.03 * h1_dirs[0]
                        exp.w3_B[:, 2] = 0.05 * u_dirs[d_idx]

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

    def _make_split(n_per_dom: int, s_gen: torch.Generator) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for d_idx, eid in enumerate(domain_names):
            lo, hi = d_idx * 8, (d_idx + 1) * 8
            tgt = lo + 2
            for _ in range(n_per_dom):
                if d_idx == 0:
                    toks = [1, 3, 4, 5, 6, 7, 2, 2, 2, 2, 2, 2]
                else:
                    toks = torch.randint(lo, hi, (12,), generator=s_gen).tolist()
                    for pos in range(4, 12):
                        toks[pos] = tgt
                rows.append({"expert_id": eid, "ids": toks})
        return rows

    cal_chunks = _make_split(3, torch.Generator(device="cpu").manual_seed(seed + 101))
    eval_chunks = _make_split(3, torch.Generator(device="cpu").manual_seed(seed + 202))
    return {"model": model, "cal_chunks": cal_chunks, "eval_chunks": eval_chunks}


# ---------------------------------------------------------------------------
# S1: Curved Deep-Layer 8-Expert MoE Surgery vs Flat Euclidean P_0
# ---------------------------------------------------------------------------


def test_s1_curved_deep_layer_kronecker_beats_flat_euclidean_p0() -> None:
    """(S1) Kronecker Riemannian transport yields strictly lower curved prose KL drift than flat P_0."""
    fix = _build_curved_8expert_moe_fixture(num_layers=2, split_mode="lora_domain_experts", seed=20264401)
    kron_model: CrowdV1WithEngram = fix["model"]
    euc_model = copy.deepcopy(kron_model)
    cal_chunks = fix["cal_chunks"]
    eval_chunks = fix["eval_chunks"]

    pre_nlls = _eval_domain_nlls(kron_model, eval_chunks, score_from=1)
    max_kl = 0.004

    receipt = execute_verified_kronecker_riemannian_surgery(
        kron_model,
        cal_chunks,
        eval_chunks=eval_chunks,
        target_factors=("w3_B",),
        damping_a=0.04,
        damping_g=0.04,
        scale_damping_by_trace=True,
        rank_keep=1,
        sv_rel_tol=0.45,
        max_kl_nats=max_kl,
        max_frobenius_norm=0.35,
        sequential_refresh=True,
        general_guard_max_ratio=1.005,
        domain_guard_max_ratio=1.01,
        backtrack_halvings=5,
        score_from=1,
        inplace=True,
    )

    assert receipt["object"] == KRONECKER_TRANSPORT_RECEIPT_OBJECT
    assert receipt["schema"] == KRONECKER_TRANSPORT_SCHEMA
    assert verify_kronecker_riemannian_receipt_provenance(receipt) is True
    assert receipt["accepted"] is True
    assert receipt["committed"] is True
    assert receipt["rolled_back"] is False
    assert receipt["status"] == "ACCEPTED"
    assert receipt["guard_ok"] is True
    assert receipt["kl_ok"] is True
    assert receipt["delta_e"] < 0.0
    assert receipt["total_kl_nats"] <= max_kl + 1e-7
    assert receipt["general_pct_vs_base"] <= 0.5
    assert receipt["total_kl_nats"] < receipt["total_euclidean_p0_kl_nats"]
    assert receipt["kl_ratio_vs_euclidean_p0"] < 0.05

    kron_nlls = _eval_domain_nlls(kron_model, eval_chunks, score_from=1)
    kron_spec_gain = (
        sum(pre_nlls[k] - kron_nlls[k] for k in pre_nlls if not k.startswith("E0_")) / 7.0
    )
    assert kron_spec_gain > 0.12

    # Compare against Frobenius-norm-matched flat Euclidean P_0 on the same model
    from hivemind.swarm.kronecker_riemannian_nullspace_transport import (
        _collect_crowd_kronecker_pairs,
        _enumerate_kronecker_units,
    )

    cfg_probe = KroneckerRiemannianTransportConfig(target_factors=("w3_B",), score_from=1)
    euc_units, _ = _enumerate_kronecker_units(
        euc_model,
        target_layer_set=None,
        target_expert_set=None,
        target_factor_set={"w3_B"},
        cfg=cfg_probe,
    )
    ids_cal = torch.tensor([c["ids"] for c in cal_chunks], dtype=torch.long)
    eids_cal = [str(c["expert_id"]) for c in cal_chunks]
    prose_cap, _, auto_td = _collect_crowd_kronecker_pairs(
        euc_model, euc_units, ids_cal, eids_cal, cfg=cfg_probe
    )

    frob_by_unit = {r["unit"]: float(r["frobenius_norm"]) for r in receipt["unit_reports"]}
    euc_total_kl = 0.0
    with torch.no_grad():
        for u in euc_units:
            u_name = u["name"]
            if u_name not in frob_by_unit:
                continue
            xp, gp = prose_cap[u_name]
            ops = compute_kronecker_whitened_nullspace_operators(
                xp, grad_out_metric=gp, rank_keep=1, damping_a=0.04, damping_g=0.04, scale_damping_by_trace=True
            )
            euc_raw = auto_td[u_name] @ ops["p_euc"]
            euc_norm = float(torch.linalg.matrix_norm(euc_raw, ord="fro").item())
            euc_delta = (frob_by_unit[u_name] / max(euc_norm, 1e-18)) * euc_raw
            g_kl = ops["g_cov"] if ops["g_cov"] is not None else torch.eye(int(euc_delta.shape[0]), dtype=torch.float64)
            euc_total_kl += compute_kronecker_kl_nats(euc_delta, ops["a_prose"], g_kl)
            u["param"].add_(euc_delta.to(dtype=u["param"].dtype))

    euc_nlls = _eval_domain_nlls(euc_model, eval_chunks, score_from=1)
    euc_e0_pct = 100.0 * (euc_nlls["E0_general_fineweb"] / pre_nlls["E0_general_fineweb"] - 1.0)
    euc_spec_gain = (
        sum(pre_nlls[k] - euc_nlls[k] for k in pre_nlls if not k.startswith("E0_")) / 7.0
    )
    assert euc_total_kl > 10.0 * receipt["total_kl_nats"]
    assert euc_e0_pct > receipt["general_pct_vs_base"]
    assert kron_spec_gain > euc_spec_gain


# ---------------------------------------------------------------------------
# S2: Theorem 3.7 Obstruction, Exact Annihilation & Curved-Layer Transport
# ---------------------------------------------------------------------------


def test_s2_theorem_3_7_obstruction_annihilation_and_layer_parallel_transport() -> None:
    """(S2) Resolves Theorem 3.7 uphill obstruction and verifies isometric layer transport + LoRA rank."""
    gen = torch.Generator(device="cpu").manual_seed(20264402)

    # 1. Canonical Theorem 3.7 non-commuting 2D obstruction regime:
    # V_0 = [1, 1]/sqrt(2), A_cov = diag(1.0, 0.05) + anisotropic G_spec = [2.5, 1.0]
    v_diag = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], dtype=torch.float64) / math.sqrt(2.0)
    xp_2d = torch.cat([3.0 * v_diag, 0.25 * torch.tensor([[1.0, -1.0]], dtype=torch.float64)], dim=0)
    xm_2d = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.15]], dtype=torch.float64)
    g_spec_2d = torch.tensor([[2.5, 1.0]], dtype=torch.float64)

    diag_37 = diagnose_theorem_3_7_obstruction(
        g_spec_2d,
        xp_2d,
        x_metric=xm_2d,
        rank_keep=1,
        damping_a=0.05,
        damping_g=0.05,
        max_kl_nats=0.01,
    )
    assert diag_37["unwhitened_aniso_euclidean_p0"]["spec_directional_deriv"] > 0.0
    assert diag_37["damped_kronecker_riemannian"]["spec_directional_deriv"] < -1e-6
    assert diag_37["damped_kronecker_riemannian"]["prose_nullspace_leak"] < 1e-11
    assert diag_37["theorem_3_7_resolved"] is True
    assert diag_37["undamped_raw_kfac"]["cond_a"] >= diag_37["damped_kronecker_riemannian"]["cond_a"]

    # 2. Curved high-anisotropy deep layer: damped Kronecker strictly beats Euclidean P_0 on prose KL
    d_in, d_out = 16, 8
    x_prose = torch.randn((48, d_in), generator=gen, dtype=torch.float64)
    # Curved prose spectrum: top-3 dominant V_0 + curved residual variance on coords 3..5, near-zero on 6..15
    x_prose[:, :3] *= 8.0
    x_prose[:, 3:6] *= 4.0
    x_prose[:, 6:] *= 0.12
    g_prose = torch.randn((48, d_out), generator=gen, dtype=torch.float64)
    g_prose[:, :3] *= 5.0
    g_prose[:, 3:] *= 0.15

    ops_src = compute_kronecker_whitened_nullspace_operators(
        x_prose, grad_out_metric=g_prose, rank_keep=3, damping_a=0.05, damping_g=0.05
    )
    assert ops_src["diagnostics"]["r_proj_idempotence_error"] <= 1e-10
    assert ops_src["diagnostics"]["r_step_prose_annihilation"] <= 1e-10
    assert ops_src["diagnostics"]["r_proj_prose_annihilation"] <= 1e-10

    target_w = torch.randn((d_out, d_in), generator=gen, dtype=torch.float64)
    dw_kron, step_diag = solve_kronecker_riemannian_transport_step(
        target_w, ops_src, max_kl_nats=0.005, transport_mode="riemannian_gradient"
    )
    assert step_diag["kl_nats"] <= 0.005 + 1e-10
    assert step_diag["prose_nullspace_leak"] <= 1e-10
    assert step_diag["kl_nats"] < 0.25 * step_diag["euclidean_p0_kl_nats"]

    # Test zero-rank, auto sv_rel_tol rank, full-rank, zero-singular-value, and distinct damping_g edge cases
    ops_auto = compute_kronecker_whitened_nullspace_operators(x_prose, grad_out_metric=g_prose, sv_rel_tol=0.12)
    assert ops_auto["rank_used"] >= 1
    ops_k0 = compute_kronecker_whitened_nullspace_operators(
        x_prose, rank_keep=0, damping_a=0.05, damping_g=0.08, scale_damping_by_trace=False
    )
    assert ops_k0["rank_used"] == 0
    ops_kfull = compute_kronecker_whitened_nullspace_operators(x_prose, rank_keep=d_in)
    assert float(ops_kfull["r_proj"].abs().max().item()) == 0.0
    ops_zero_x = compute_kronecker_whitened_nullspace_operators(torch.zeros((4, 6), dtype=torch.float64))
    assert ops_zero_x["rank_used"] == 0

    # 3. Isometric layer-to-layer parallel transport onto a rotated & curved deep layer
    q_rot, _ = torch.linalg.qr(torch.randn((d_in, d_in), generator=gen, dtype=torch.float64))
    x_deep = (x_prose @ q_rot) * 1.4
    g_deep = torch.randn((48, d_out), generator=gen, dtype=torch.float64)
    g_deep[:, :4] *= 3.5
    ops_dst = compute_kronecker_whitened_nullspace_operators(
        x_deep, grad_out_metric=g_deep, rank_keep=3, damping_a=0.05, damping_g=0.08
    )
    dw_trans, trans_meta = transport_delta_between_curved_layers(
        dw_kron, ops_src, ops_dst, max_kl_nats=0.005
    )
    assert trans_meta["isometry_rel_error"] <= 1e-9
    assert trans_meta["dst_prose_nullspace_leak"] <= 1e-10
    assert trans_meta["projected_prose_kl_dst"] <= 0.005 + 1e-10

    # Same transport without output Fisher factors
    ops_in_only_src = compute_kronecker_whitened_nullspace_operators(x_prose, rank_keep=3)
    ops_in_only_dst = compute_kronecker_whitened_nullspace_operators(x_deep, rank_keep=3)
    _, trans_in_meta = transport_delta_between_curved_layers(dw_kron, ops_in_only_src, ops_in_only_dst)
    assert trans_in_meta["dst_prose_nullspace_leak"] <= 1e-10

    # 4. Rank-preserving factorized LoRA Kronecker transport (D2b + riemannian_metric)
    lora_a = torch.randn((4, d_in), generator=gen, dtype=torch.float32)
    lora_b = torch.randn((d_out, 4), generator=gen, dtype=torch.float32)
    base_w = torch.randn((d_out, d_in), generator=gen, dtype=torch.float32)
    base_w[0, :] *= 6.0
    a_new, b_new, lora_meta = transport_factorized_lora_with_kronecker_metric(
        lora_a,
        lora_b,
        2.0,
        ops_src,
        projection_kind="riemannian_metric",
        balance_output_rows=True,
    )
    assert a_new.shape == lora_a.shape and b_new.shape == lora_b.shape
    assert lora_meta["prose_nullspace_leak"] <= 1e-10
    assert lora_meta["kronecker_prose_kl_nats"] < lora_meta["euclidean_p0_prose_kl_nats"]

    _, _, lora_oblique_meta = transport_factorized_lora_with_kronecker_metric(
        lora_a,
        lora_b,
        2.0,
        ops_src,
        base_weight=base_w,
        projection_kind="oblique_proj",
        balance_output_rows=True,
    )
    assert lora_oblique_meta["prose_nullspace_leak"] <= 1e-10
    _, _, lora_row_meta = transport_factorized_lora_with_kronecker_metric(
        lora_a,
        lora_b,
        2.0,
        ops_src,
        projection_kind="oblique_proj",
        balance_output_rows=True,
    )
    assert lora_row_meta["prose_nullspace_leak"] <= 1e-10
    _, _, lora_unbal_meta = transport_factorized_lora_with_kronecker_metric(
        lora_a, lora_b, 2.0, ops_in_only_src, balance_output_rows=False
    )
    assert lora_unbal_meta["prose_nullspace_leak"] <= 1e-10
    _, _, lora_no_g_bal_meta = transport_factorized_lora_with_kronecker_metric(
        lora_a, lora_b, 2.0, ops_in_only_src, balance_output_rows=True
    )
    assert lora_no_g_bal_meta["prose_nullspace_leak"] <= 1e-10
    with pytest.raises(ValueError, match="projection_kind"):
        transport_factorized_lora_with_kronecker_metric(
            lora_a, lora_b, 2.0, ops_src, projection_kind="bad"
        )


# ---------------------------------------------------------------------------
# S3: Geodesic KL Trust-Region Clamping & ANSE Backtracking
# ---------------------------------------------------------------------------


def test_s3_geodesic_kl_trust_region_clamping_and_backtracking() -> None:
    """(S3) Verifies exact geodesic KL trust-region clamping and halving backtracking."""
    gen = torch.Generator(device="cpu").manual_seed(20264403)
    xp = torch.randn((32, 10), generator=gen, dtype=torch.float64)
    xp[:, :2] *= 7.0
    gp = torch.randn((32, 6), generator=gen, dtype=torch.float64)
    gp[:, :2] *= 4.0
    ops = compute_kronecker_whitened_nullspace_operators(xp, grad_out_metric=gp, rank_keep=2)
    target = torch.randn((6, 10), generator=gen, dtype=torch.float64)

    for kl_cap in (1e-4, 0.005, 0.05):
        dw_g, d_g = solve_kronecker_riemannian_transport_step(
            target, ops, max_kl_nats=kl_cap, transport_mode="riemannian_gradient"
        )
        dw_w, d_w = solve_kronecker_riemannian_transport_step(
            target,
            ops,
            a_prose=ops["a_prose"],
            g_prose=ops["g_cov"],
            max_kl_nats=kl_cap,
            transport_mode="whitened_lora_transport",
        )
        assert d_g["kl_nats"] == pytest.approx(kl_cap, rel=1e-6, abs=1e-12)
        assert d_w["kl_nats"] == pytest.approx(kl_cap, rel=1e-6, abs=1e-12)
        assert d_g["kl_bound_satisfied"] is True and d_w["kl_bound_satisfied"] is True
        assert compute_kronecker_kl_nats(dw_g, ops["a_prose"], ops["g_cov"]) <= kl_cap + 1e-10
        assert compute_kronecker_kl_nats(dw_w, ops["a_prose"], ops["g_cov"]) <= kl_cap + 1e-10

    # rescale_only_if_exceeds, max_frobenius_norm, and zero target
    dw_small, d_small = solve_kronecker_riemannian_transport_step(
        1e-4 * target, ops, max_kl_nats=1.0, rescale_only_if_exceeds=True
    )
    assert d_small["trust_scale"] == pytest.approx(1.0)
    assert float(dw_small.norm().item()) > 0.0

    dw_frob, d_frob = solve_kronecker_riemannian_transport_step(
        target, ops, max_kl_nats=10.0, max_frobenius_norm=0.05
    )
    assert d_frob["frobenius_norm"] == pytest.approx(0.05, rel=1e-6)
    assert float(dw_frob.norm().item()) == pytest.approx(0.05, rel=1e-6)

    dw_zero, d_zero = solve_kronecker_riemannian_transport_step(torch.zeros((6, 10)), ops)
    assert float(dw_zero.abs().max().item()) == 0.0
    assert d_zero["kl_nats"] == 0.0

    # End-to-end trust backtracking: full step violates D0 guard; halving finds feasible commit!
    lin = nn.Linear(10, 6, bias=False)
    with torch.no_grad():
        lin.weight.zero_()

    def _backtrack_eval(mod: nn.Module) -> dict[str, float]:
        assert isinstance(mod, nn.Linear)
        w_nrm = float(mod.weight.norm().item())
        if w_nrm > 0.06:
            return {"E0_prose": 1.05, "E1_spec": 0.90}
        return {"E0_prose": 1.00 - 0.05 * w_nrm, "E1_spec": 1.20 - 1.5 * w_nrm}

    rec_bt = execute_verified_kronecker_riemannian_surgery(
        lin,
        prose_activations=xp,
        metric_grad_outs=gp,
        target_deltas={"layers.0.mlp.weight": torch.ones((6, 10), dtype=torch.float64)},
        eval_fn=_backtrack_eval,
        max_kl_nats=0.5,
        kl_budget_scope="per_unit",
        enable_trust_backtracking=True,
        domain_guard_max_ratio=1.01,
    )
    assert rec_bt["accepted"] is True
    assert rec_bt["backtrack_scale_applied"] < 1.0
    assert rec_bt["delta_e"] < 0.0
    assert verify_kronecker_riemannian_receipt_provenance(rec_bt) is True


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Atomic Rollback & Multi-Topology Support
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_rollback_and_multi_topology() -> None:
    """(S4) Enforces ANSE Maximum Pain (1e6) atomic rollback across MoE, HF, and dense topologies."""
    fix = _build_curved_8expert_moe_fixture(num_layers=1, split_mode="lora_domain_experts", seed=20264404)
    model: CrowdV1WithEngram = fix["model"]
    parent_sha = compute_state_sha256(model.state_dict())

    # 1. Infeasible general guard triggers ROLLED_BACK_GUARD_VIOLATION & atomic bitwise rollback
    rec_guard = execute_verified_kronecker_riemannian_surgery(
        model,
        fix["cal_chunks"],
        eval_chunks=fix["eval_chunks"],
        target_factors=("w3_B",),
        max_kl_nats=4.0,
        general_guard_max_ratio=1.0,
        domain_guard_max_ratio=1.01,
        enable_trust_backtracking=False,
        score_from=4,
    )
    assert rec_guard["accepted"] is False
    assert rec_guard["rolled_back"] is True
    assert rec_guard["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert rec_guard["anse_energy"] == pytest.approx(ANSE_MAXIMUM_PAIN)
    assert rec_guard["final_state_sha256"] == parent_sha
    assert compute_state_sha256(model.state_dict()) == parent_sha
    assert verify_kronecker_riemannian_receipt_provenance(rec_guard) is True

    # 2. Non-improving energy, eval failure, and fixed-point unchanged on nn.Linear
    lin = nn.Linear(8, 4, bias=False)
    lin_sha = compute_state_sha256(lin.state_dict())
    xp = torch.ones((6, 8), dtype=torch.float32)
    gm = torch.ones((6, 4), dtype=torch.float32)
    td = {"layers.0.mlp.weight": torch.ones((4, 8), dtype=torch.float32)}

    rec_noimp = execute_verified_kronecker_riemannian_surgery(
        lin,
        prose_activations=xp,
        metric_activations=xp,
        metric_grad_outs=gm,
        target_deltas=td,
        eval_fn=lambda _m: (0.8, {"E0_prose": 0.8, "E1_spec": 0.8}),
        require_energy_improvement=True,
        enable_trust_backtracking=False,
    )
    assert rec_noimp["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    assert rec_noimp["rolled_back"] is True
    assert compute_state_sha256(lin.state_dict()) == lin_sha

    calls = {"n": 0}

    def _boom_eval(_m: nn.Module) -> dict[str, float]:
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("injected evaluation failure")
        return {"E0_prose": 0.8, "E1_spec": 0.8}

    rec_boom = execute_verified_kronecker_riemannian_surgery(
        lin,
        prose_activations=xp,
        target_deltas=td,
        eval_fn=_boom_eval,
        enable_trust_backtracking=False,
    )
    assert rec_boom["status"] == "ROLLED_BACK_EVAL_FAILURE"
    assert rec_boom["anse_energy"] == pytest.approx(ANSE_MAXIMUM_PAIN)
    assert compute_state_sha256(lin.state_dict()) == lin_sha

    rec_fp = execute_verified_kronecker_riemannian_surgery(
        lin,
        prose_activations=xp,
        target_deltas={"layers.0.mlp.weight": torch.zeros((4, 8))},
        eval_fn=lambda _m: {"E0_prose": 0.8, "E1_spec": 0.8},
    )
    assert rec_fp["status"] == "FIXED_POINT_UNCHANGED"
    assert rec_fp["rolled_back"] is False

    # 3. CrowdFineGrainedMoEMLP (inplace=False, metric_source="all", attn + head factors)
    fg_fix = _build_curved_8expert_moe_fixture(num_layers=1, split_mode="fine_grained_width_split", seed=20264405)
    fg_model: CrowdV1WithEngram = fg_fix["model"]
    fg_sha = compute_state_sha256(fg_model.state_dict())
    rec_fg = execute_verified_kronecker_riemannian_surgery(
        fg_model,
        fg_fix["cal_chunks"],
        target_factors=("w3", "v", "head"),
        layer_indices=[0],
        expert_indices=[1, 2],
        metric_source="all",
        max_kl_nats=0.005,
        max_frobenius_norm=0.25,
        score_from=4,
        inplace=False,
    )
    assert compute_state_sha256(fg_model.state_dict()) == fg_sha
    assert rec_fg["accepted"] is True

    # 4. HF-style .layers (.self_attn + .mlp + .lm_head) & CrowdRuntimeMLP dense blocks
    class _HFAttn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(8, 8, bias=False)
            self.v_proj = nn.Linear(8, 8, bias=False)

    class _HFMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(8, 12, bias=False)
            self.down_proj = nn.Linear(12, 8, bias=False)

    class _HFLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = _HFAttn()
            self.mlp = _HFMLP()

    class _HFModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_HFLayer()])
            self.lm_head = nn.Linear(8, 10, bias=False)

    hf_m = _HFModel()
    rec_hf = execute_verified_kronecker_riemannian_surgery(
        hf_m,
        prose_activations=torch.randn((16, 8), generator=torch.Generator().manual_seed(7)),
        target_deltas={
            "q_proj": torch.ones((8, 8)),
            "v_proj": torch.ones((8, 8)),
            "lm_head": torch.ones((10, 8)),
        },
        eval_fn=lambda _m: {"E0_prose": 0.8, "E1_spec": 0.8},
        target_factors=("q_proj", "v_proj", "lm_head"),
        transport_mode="whitened_lora_transport",
        require_energy_improvement=False,
    )
    assert rec_hf["accepted"] is True
    assert rec_hf["units_edited"] == 3

    dense_mlp = CrowdRuntimeMLP(
        CrowdV1RuntimeConfig(
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
    )
    rec_dense = execute_verified_kronecker_riemannian_surgery(
        dense_mlp,
        prose_activations=xp,
        target_deltas={"w1": torch.ones((16, 8))},
        eval_fn=lambda _m: {"E0_prose": 0.9, "E1_spec": 0.9},
        target_factors=("w1",),
        require_energy_improvement=False,
    )
    assert rec_dense["accepted"] is True


# ---------------------------------------------------------------------------
# S5: Validation Gates & Repository Security / Zero-Stub / OwnerOrder Gates
# ---------------------------------------------------------------------------


def test_s5_validation_gates_and_security_stub_owner_compliance() -> None:
    """(S5) Assert input validation, SHA-256 provenance, zero stubs, lateral guard, and OwnerOrder."""
    # 1. Config & primitive validation errors
    with pytest.raises(ValueError, match="damping_a"):
        KroneckerRiemannianTransportConfig(damping_a=0.0)
    with pytest.raises(ValueError, match="kl_budget_scope"):
        KroneckerRiemannianTransportConfig(kl_budget_scope="bad")
    with pytest.raises(ValueError, match="transport_mode"):
        KroneckerRiemannianTransportConfig(transport_mode="bad")
    with pytest.raises(ValueError, match="metric_source"):
        KroneckerRiemannianTransportConfig(metric_source="bad")
    with pytest.raises(ValueError, match="max_frobenius_norm"):
        KroneckerRiemannianTransportConfig(max_frobenius_norm=0.0)
    with pytest.raises(ValueError, match="target_factors"):
        KroneckerRiemannianTransportConfig(target_factors=())
    with pytest.raises(ValueError, match="Unsupported target_factors"):
        KroneckerRiemannianTransportConfig(target_factors=("bad_factor",))
    with pytest.raises(ValueError, match="domain_guard_max_ratio"):
        KroneckerRiemannianTransportConfig(domain_guard_max_ratio=0.9)
    with pytest.raises(ValueError, match="general_guard_max_ratio"):
        KroneckerRiemannianTransportConfig(general_guard_max_ratio=0.95)
    with pytest.raises(ValueError, match="anse_safe_e0_ratio"):
        KroneckerRiemannianTransportConfig(anse_safe_e0_ratio=0.95)
    with pytest.raises(ValueError, match="general_prefixes"):
        KroneckerRiemannianTransportConfig(general_prefixes=())

    x_ok = torch.randn((8, 6), dtype=torch.float64)
    with pytest.raises(ValueError, match="x_metric width"):
        compute_kronecker_whitened_nullspace_operators(x_ok, x_metric=torch.randn((8, 5)))

    ops_ok = compute_kronecker_whitened_nullspace_operators(x_ok, grad_out_metric=torch.randn((8, 4)), rank_keep=2)
    with pytest.raises(ValueError, match="d_in"):
        solve_kronecker_riemannian_transport_step(torch.ones((4, 5)), ops_ok)
    with pytest.raises(ValueError, match="transport_mode"):
        solve_kronecker_riemannian_transport_step(torch.ones((4, 6)), ops_ok, transport_mode="bad")
    with pytest.raises(ValueError, match="Operator g_inv shape"):
        solve_kronecker_riemannian_transport_step(torch.ones((3, 6)), ops_ok)

    with pytest.raises(ValueError, match="input width"):
        transport_delta_between_curved_layers(torch.ones((4, 5)), ops_ok, ops_ok)
    with pytest.raises(ValueError, match="Output Fisher"):
        transport_delta_between_curved_layers(torch.ones((3, 6)), ops_ok, ops_ok)

    with pytest.raises(ValueError, match="LoRA rank mismatch"):
        transport_factorized_lora_with_kronecker_metric(torch.ones((2, 6)), torch.ones((4, 3)), 1.0, ops_ok)
    with pytest.raises(ValueError, match="operator d_in"):
        transport_factorized_lora_with_kronecker_metric(torch.ones((2, 5)), torch.ones((4, 2)), 1.0, ops_ok)
    with pytest.raises(ValueError, match="base_weight shape"):
        transport_factorized_lora_with_kronecker_metric(
            torch.ones((2, 6)), torch.ones((4, 2)), 1.0, ops_ok, base_weight=torch.ones((3, 6))
        )

    lin = nn.Linear(6, 4, bias=False)
    with pytest.raises(TypeError, match="nn.Module"):
        execute_verified_kronecker_riemannian_surgery("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable"):
        execute_verified_kronecker_riemannian_surgery(lin, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mapping"):
        execute_verified_kronecker_riemannian_surgery(lin, target_deltas="bad", eval_fn=lambda _m: {"E0_a": 1.0})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="KroneckerRiemannianTransportConfig"):
        execute_verified_kronecker_riemannian_surgery(lin, config="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="layer_indices"):
        execute_verified_kronecker_riemannian_surgery(
            lin, prose_activations=x_ok, target_deltas={"w": torch.ones((4, 6))}, eval_fn=lambda _m: {"E0_a": 1.0}, layer_indices=[3]
        )
    with pytest.raises(TypeError, match="CrowdV1WithEngram"):
        execute_verified_kronecker_riemannian_surgery(lin, [{"expert_id": "E0_a", "ids": [1, 2]}])
    with pytest.raises(ValueError, match="Either calibration_chunks"):
        execute_verified_kronecker_riemannian_surgery(lin)
    with pytest.raises(ValueError, match="prose_activations"):
        execute_verified_kronecker_riemannian_surgery(lin, eval_fn=lambda _m: {"E0_a": 1.0})
    with pytest.raises(ValueError, match="identical keys"):
        execute_verified_kronecker_riemannian_surgery(
            lin,
            prose_activations=x_ok,
            target_deltas={"w": torch.ones((4, 6))},
            eval_fn=lambda _m: {"E0_a": 1.0},
            baseline_domain_nll={"E0_other": 1.0},
        )
    with pytest.raises(ValueError, match="Missing required general-domain"):
        execute_verified_kronecker_riemannian_surgery(
            lin,
            prose_activations=x_ok,
            target_deltas={"w": torch.ones((4, 6))},
            eval_fn=lambda _m: {"E1_only": 1.0},
        )

    with pytest.raises(TypeError, match="Mapping"):
        verify_kronecker_riemannian_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_kronecker_riemannian_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert verify_kronecker_riemannian_receipt_provenance({"receipt_sha256": "0" * 64, "object": "wrong"}) is False
    assert (
        verify_kronecker_riemannian_receipt_provenance(
            {"receipt_sha256": "0" * 64, "object": KRONECKER_TRANSPORT_RECEIPT_OBJECT, "schema": "wrong"}
        )
        is False
    )
    assert (
        verify_kronecker_riemannian_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": KRONECKER_TRANSPORT_RECEIPT_OBJECT,
                "schema": KRONECKER_TRANSPORT_SCHEMA,
                "bad": float("nan"),
            }
        )
        is False
    )

    # 2. Repository AGPL-3.0, ZeroStubAudit, PRSecurityScanner, lateral movement, FORBIDDEN_ADDED & OwnerOrder
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn44", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn44", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/kronecker_riemannian_nullspace_transport.py", TARGET_MODULE_PATH),
        ("tests/test_kronecker_riemannian_nullspace_transport_scenarios.py", TEST_FILE_PATH),
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
        {"nll_mean": 1.44, "kl_nats": 0.004, "anse_energy": -0.35, "accepted": True},
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+TELEMETRY = {telemetry_json!r}"
        )
        == []
    )

    pr_title = "feat(swarm): Kronecker-factored Riemannian null-space transport surgeon for curved deep-layer prose preservation"
    pr_body = (
        "Implements NEURAL-CTRL-DYN-44 Gen-3 Direction D2 Kronecker-factored Riemannian null-space "
        "transport (K-FAC input/output Fisher covariance), geodesic KL trust-region scaling, atomic "
        "bitwise rollback, and SHA-256 receipts under ANSE Delta E < 0."
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
        pr_number=1244,
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
