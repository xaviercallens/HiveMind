# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-40: rank-preserving prose-nullspace LoRA input injector."""

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
from hivemind.security.owner_order_guardrail import (
    RequestIntentCategory,
    classify_request_intent,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import upcycle_crowd_v1_to_sparse_moe
from hivemind.swarm.rank_preserving_nullspace_input_injector import (
    ANSE_MAXIMUM_PAIN,
    INJECTOR_RECEIPT_OBJECT,
    INJECTOR_SCHEMA,
    NullspaceInputInjectionConfig,
    compare_input_nullspace_vs_full_delta_refactorization,
    execute_verified_multicampaign_nullspace_injection,
    inject_donor_lora_via_input_nullspace,
    unfold_and_recalibrate_safe_layernorm_temperature,
    verify_nullspace_injection_receipt_provenance,
)
from hivemind.swarm.self_healing_checkpoint_rescuer import (
    compute_checkpoint_sha256,
    extract_checkpoint_blocks,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:  # pragma: no cover - optional in lightweight CI runners
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
    REPO_ROOT / "src" / "hivemind" / "swarm" / "rank_preserving_nullspace_input_injector.py"
)
TEST_FILE_PATH = Path(__file__).resolve()
ROADMAP_DOC_PATH = REPO_ROOT / "docs" / "NEURO_CONTROLLER_ROADMAP.md"
PLAN_DOC_PATH = REPO_ROOT / "docs" / "NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md"
EXPECTED_AGPL_HEADER = (
    "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
)


@pytest.fixture(autouse=True, scope="module")
def _single_thread_cpu_torch() -> Any:
    """Eliminate OpenMP thread-pool spin overhead on small CPU tensors during tests."""
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


def _make_zero_mean_ortho_rows(n_rows: int, dim: int, seed: int) -> torch.Tensor:
    """Create ``n_rows`` orthonormal vectors orthogonal to all-ones in ``R^dim``."""
    assert n_rows < dim
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ones = torch.ones((dim, 1), dtype=torch.float64)
    mat = torch.cat(
        [ones, torch.randn((dim, dim - 1), generator=gen, dtype=torch.float64)],
        dim=1,
    )
    q_mat, _ = torch.linalg.qr(mat)
    return q_mat[:, 1 : 1 + n_rows].T.to(torch.float32).contiguous()


def _install_domain_lora_delta(
    exp_mod: Any,
    in_dir: torch.Tensor,
    out_dir: torch.Tensor,
    *,
    strength: float,
    inter_slice: slice,
    extra_in_dir: torch.Tensor | None = None,
    extra_strength: float = 0.0,
    gate_scale: float = 2.5,
) -> None:
    """Configure a ``CrowdLoRAExpertAdapter`` to emit ``strength * out_dir`` along ``in_dir``."""
    rank = int(exp_mod.w1_A.shape[0])
    scaling = float(exp_mod.scaling)
    b_scale = math.sqrt(abs(strength)) / max(math.sqrt(scaling), 1e-6)
    sign_b = 1.0 if strength >= 0.0 else -1.0

    exp_mod.w1_A.zero_()
    exp_mod.w1_B.zero_()
    exp_mod.w2_A.zero_()
    exp_mod.w2_B.zero_()
    exp_mod.w3_A.zero_()
    exp_mod.w3_B.zero_()

    i_start = inter_slice.start
    i_width = inter_slice.stop - inter_slice.start
    n_active = min(rank, i_width)
    exp_mod.w1_A[0] = in_dir
    exp_mod.w2_A[0] = in_dir
    exp_mod.w1_B[i_start : i_start + n_active, 0] = gate_scale
    exp_mod.w2_B[i_start : i_start + n_active, 0] = gate_scale
    exp_mod.w3_A[0, i_start : i_start + n_active] = 1.0 / float(n_active)
    exp_mod.w3_B[:, 0] = sign_b * b_scale * out_dir

    if extra_in_dir is not None and rank > 1 and extra_strength != 0.0:
        exp_mod.w1_A[1] = extra_strength * extra_in_dir
        exp_mod.w2_A[1] = extra_strength * extra_in_dir
        exp_mod.w1_B[i_start : i_start + n_active, 1] = gate_scale
        exp_mod.w2_B[i_start : i_start + n_active, 1] = gate_scale
        if rank > 2:
            exp_mod.w3_A[2, i_start : i_start + n_active] = 0.5 / float(n_active)
            exp_mod.w3_B[:, 2] = 0.25 * sign_b * b_scale * out_dir


def _build_pilot6_multicampaign_models(
    *,
    seed: int = 20261400,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    num_layers: int = 2,
    vocab_size: int = 48,
    num_experts: int = 8,
    lora_rank: int = 4,
    n_per_dom: int = 3,
) -> tuple[
    CrowdV1WithEngram,
    CrowdV1WithEngram,
    dict[str, CrowdV1WithEngram],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Build deterministic Base, co-adapted S4 Anchor, T7_code Donor, and C3_science Donor."""
    block_h = hidden_size // 4
    block_v = vocab_size // 4
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
        tie_word_embeddings=False,
    )
    base_model = CrowdV1WithEngram(cfg)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    domain_specs = (
        (0, "E0_general_fineweb", 0, 0, slice(0, block_h)),
        (1, "E1_python_systems_code", 1, block_v, slice(block_h, 2 * block_h)),
        (2, "E2_math_reasoning", 2, 2 * block_v, slice(2 * block_h, 3 * block_h)),
        (6, "E6_stellar_spectroscopy", 3, 3 * block_v, slice(3 * block_h, 4 * block_h)),
    )

    axes_in: dict[int, torch.Tensor] = {}
    axes_out: dict[int, torch.Tensor] = {}
    axes_sec: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for p in base_model.parameters():
            p.zero_()
        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        head_w = 0.04 * torch.randn((vocab_size, hidden_size), generator=gen, dtype=torch.float32)

        for e_slot, _, b_idx, v_off, h_slice in domain_specs:
            basis = _make_zero_mean_ortho_rows(4, block_h, seed=seed + 100 + b_idx * 19)
            u_in, u_out, u_sec = basis[0], basis[1], basis[2]
            axes_in[e_slot] = u_in
            axes_out[e_slot] = u_out
            axes_sec[e_slot] = u_sec
            rows = _make_zero_mean_ortho_rows(min(6, block_h - 1), block_h, seed=seed + 200 + b_idx * 31)
            for t_i in range(block_v):
                vec = 0.20 * rows[t_i % int(rows.shape[0])] + 0.55 * u_in + 0.45 * u_sec
                wte[v_off + t_i, h_slice] = vec - vec.mean()
            head_w[:, h_slice] = head_w[:, h_slice] - head_w[:, h_slice].mean(dim=1, keepdim=True)
            for tok in (v_off + 2, v_off + 3, v_off + 4):
                head_w[tok, h_slice] = 0.12 * u_in + 0.65 * u_out

        base_model.wte.weight.copy_(wte)
        base_model.head.weight.copy_(head_w)
        base_model.ln.weight.fill_(1.0)
        base_model.ln.bias.zero_()
        for blk in base_model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln2.weight.fill_(1.0)
            blk.mlp.w1.weight[:hidden_size, :hidden_size].copy_(0.35 * torch.eye(hidden_size))
            blk.mlp.w2.weight[:hidden_size, :hidden_size].copy_(0.35 * torch.eye(hidden_size))
            blk.mlp.w3.weight[:hidden_size, :hidden_size].copy_(0.15 * torch.eye(hidden_size))

    upcycle_crowd_v1_to_sparse_moe(
        base_model,
        num_experts=num_experts,
        top_k=2,
        lora_rank=lora_rank,
        routing_mode="sigma_topk",
        seed=seed + 61,
    )
    with torch.no_grad():
        for blk in base_model.blocks:
            blk.mlp.router.weight.zero_()
            for e_slot, _, _, _, h_slice in domain_specs:
                blk.mlp.router.weight[e_slot, h_slice] = 3.2 * axes_in[e_slot]

    s4_anchor = copy.deepcopy(base_model)
    t7_donor = copy.deepcopy(base_model)
    c3_donor = copy.deepcopy(base_model)

    with torch.no_grad():
        for l_idx in range(num_layers):
            i_base = block_h
            # S4 co-adapted anchor: strong router + moderate B_expert, partial initial A_anchor leaving headroom
            for e_slot, h_slice, i_slice, s4_str, in_scale, g_scale in (
                (1, slice(block_h, 2 * block_h), slice(i_base, 2 * i_base), 0.18, 0.08, 0.18),
                (2, slice(2 * block_h, 3 * block_h), slice(2 * i_base, 3 * i_base), 0.80, 0.45, 0.45),
                (6, slice(3 * block_h, 4 * block_h), slice(3 * i_base, 4 * i_base), 0.18, 0.08, 0.18),
            ):
                exp_s4 = s4_anchor.blocks[l_idx].mlp.experts[str(e_slot)]
                f_in = torch.zeros(hidden_size, dtype=torch.float32)
                f_out = torch.zeros(hidden_size, dtype=torch.float32)
                f_sec = torch.zeros(hidden_size, dtype=torch.float32)
                f_in[h_slice] = in_scale * axes_in[e_slot]
                f_out[h_slice] = axes_out[e_slot]
                f_sec[h_slice] = axes_sec[e_slot]
                _install_domain_lora_delta(
                    exp_s4,
                    f_in,
                    f_out,
                    strength=s4_str,
                    inter_slice=i_slice,
                    extra_in_dir=f_sec,
                    extra_strength=0.05,
                    gate_scale=g_scale,
                )

            # T7_code donor: much stronger E1_code input feature + prose leakage on E0 dims (filtered by P_0!)
            exp_t7_e1 = t7_donor.blocks[l_idx].mlp.experts["1"]
            f_in1 = torch.zeros(hidden_size, dtype=torch.float32)
            f_out1 = torch.zeros(hidden_size, dtype=torch.float32)
            f_sec1 = torch.zeros(hidden_size, dtype=torch.float32)
            f_in1[block_h : 2 * block_h] = 1.6 * axes_in[1]
            f_in1[:block_h] = 0.9 * axes_in[0]
            f_out1[block_h : 2 * block_h] = axes_out[1]
            f_sec1[block_h : 2 * block_h] = axes_sec[1]
            _install_domain_lora_delta(
                exp_t7_e1,
                f_in1,
                f_out1,
                strength=0.80,
                inter_slice=slice(block_h, 2 * block_h),
                extra_in_dir=f_sec1,
                extra_strength=1.4,
                gate_scale=0.35,
            )

            # C3_science donor: much stronger E6_stars input feature + prose leakage on E0 dims (filtered by P_0!)
            exp_c3_e6 = c3_donor.blocks[l_idx].mlp.experts["6"]
            f_in6 = torch.zeros(hidden_size, dtype=torch.float32)
            f_out6 = torch.zeros(hidden_size, dtype=torch.float32)
            f_sec6 = torch.zeros(hidden_size, dtype=torch.float32)
            f_in6[3 * block_h : 4 * block_h] = 1.6 * axes_in[6]
            f_in6[:block_h] = -0.95 * axes_in[0]
            f_out6[3 * block_h : 4 * block_h] = axes_out[6]
            f_sec6[3 * block_h : 4 * block_h] = axes_sec[6]
            _install_domain_lora_delta(
                exp_c3_e6,
                f_in6,
                f_out6,
                strength=0.80,
                inter_slice=slice(3 * block_h, 4 * block_h),
                extra_in_dir=f_sec6,
                extra_strength=1.4,
                gate_scale=0.35,
            )

            # T7 & C3 routers have harmful prose drift so naive cross-arm replacement would hurt E0
            t7_donor.blocks[l_idx].mlp.router.weight[1, :block_h] = 3.5 * axes_in[0]
            c3_donor.blocks[l_idx].mlp.router.weight[6, :block_h] = 3.5 * axes_in[0]

    def _make_chunks(seed_off: int, cnt: int) -> list[dict[str, Any]]:
        c_gen = torch.Generator(device="cpu").manual_seed(seed + 500 + seed_off)
        chunks: list[dict[str, Any]] = []
        seq_len = 12
        for e_slot, dom_name, _, v_off, _ in domain_specs:
            target_pool = (v_off + 2, v_off + 3, v_off + 4)
            for r_i in range(cnt):
                ids = torch.randint(v_off, v_off + block_v, (seq_len,), generator=c_gen).tolist()
                for pos in range(4, seq_len):
                    ids[pos] = target_pool[(pos + r_i + e_slot) % len(target_pool)]
                chunks.append({"expert_id": dom_name, "ids": ids})
        return chunks

    return (
        base_model,
        s4_anchor,
        {"T7_code": t7_donor, "C3_science": c3_donor},
        _make_chunks(1, n_per_dom),
        _make_chunks(2, n_per_dom),
    )


# ---------------------------------------------------------------------------
# S1: Multi-Campaign Input-Side P_0 Injection Beating Anchor with <1e-5 Prose Drift
# ---------------------------------------------------------------------------


def test_s1_multicampaign_input_nullspace_injection_beats_anchor_with_prose_invariance() -> None:
    """S1: Multi-campaign S4+T7+C3 input P_0 injection beats anchor with <1e-5 float32 drift & zero rank growth."""
    # 1. Direct tensor-level float32 invariance and multi-donor formula verification
    gen = torch.Generator(device="cpu").manual_seed(20261401)
    d_in, d_out, r, k_prose = 24, 20, 4, 6
    q_all, _ = torch.linalg.qr(torch.randn((d_in, d_in), generator=gen, dtype=torch.float64))
    u_p = q_all[:, :k_prose]
    p_0_64 = torch.eye(d_in, dtype=torch.float64) - (u_p @ u_p.T)
    p_0 = p_0_64.to(torch.float32)

    x_prose = (torch.randn((32, k_prose), generator=gen, dtype=torch.float64) @ u_p.T).to(torch.float32)
    x_spec = (torch.randn((16, d_in - k_prose), generator=gen, dtype=torch.float64) @ q_all[:, k_prose:].T).to(torch.float32)

    a_anchor = 0.25 * torch.randn((r, d_in), generator=gen, dtype=torch.float32)
    b_anchor = 0.30 * torch.randn((d_out, r), generator=gen, dtype=torch.float32)
    b_anchor_before = b_anchor.clone()
    a_t7 = a_anchor + 0.8 * torch.randn((r, d_in), generator=gen, dtype=torch.float32)
    a_c3 = a_anchor + 0.7 * torch.randn((r, d_in), generator=gen, dtype=torch.float32)

    a_new, inj_meta = inject_donor_lora_via_input_nullspace(
        a_anchor,
        b_anchor,
        [a_t7, a_c3],
        [0.15, 0.20],
        p_0,
        x_prose=x_prose,
        eps=1e-5,
    )
    assert a_new.shape == (r, d_in)
    assert torch.equal(b_anchor, b_anchor_before)
    assert inj_meta["rank_r"] == r
    assert inj_meta["rank_increase"] == 0
    assert inj_meta["svd_truncation_error_fro"] == 0.0
    assert inj_meta["b_anchor_unchanged"] is True
    assert inj_meta["prose_invariance_max_diff"] < 1e-5

    direct_prose_drift = float((b_anchor @ ((a_new - a_anchor) @ x_prose.T)).abs().max().item())
    assert direct_prose_drift < 1e-5
    spec_change = float((b_anchor @ ((a_new - a_anchor) @ x_spec.T)).abs().max().item())
    assert spec_change > 0.05

    # Single-tensor / scalar-alpha + x_prose=None + float64 branch
    a_single, meta_single = inject_donor_lora_via_input_nullspace(
        a_anchor.double(), b_anchor.double(), a_t7.double(), 0.25, p_0_64, eps=1e-10
    )
    assert a_single.dtype == torch.float64 and meta_single["prose_invariance_max_diff"] < 1e-10

    # 2. End-to-end CrowdV1WithEngram + CrowdLoRAMoEMLP multi-campaign S4 + T7_code + C3_science surgery
    base_model, s4_anchor, donors, probe_chunks, eval_chunks = _build_pilot6_multicampaign_models()
    cfg = NullspaceInputInjectionConfig(
        prose_invariance_eps=1e-5,
        inject_projections=("w1", "w2", "w3"),
        score_from=4,
        batch_size=8,
        inplace=False,
    )
    out_model, receipt = execute_verified_multicampaign_nullspace_injection(
        base_model,
        s4_anchor,
        donors,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        donor_alphas={"T7_code": 0.45, "C3_science": 0.45},
        expert_donor_map={1: "T7_code", 6: "C3_science"},
        target_experts={1, 6},
        config=cfg,
    )

    assert receipt["object"] == INJECTOR_RECEIPT_OBJECT
    assert receipt["schema"] == INJECTOR_SCHEMA
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["status"] == "MULTICAMPAIGN_NULLSPACE_INJECTION_COMMITTED"
    assert receipt["rank_preserved"] is True
    assert receipt["rank_increase"] == 0
    assert receipt["svd_truncation_error_fro"] == 0.0
    assert receipt["router_bitwise_preserved"] is True
    assert receipt["b_anchor_bitwise_preserved"] is True
    assert receipt["prose_invariance_max_diff"] < 1e-5
    assert receipt["anse_energy"] < 0.0
    assert receipt["delta_e_vs_base"] < 0.0
    assert receipt["delta_e_vs_anchor"] < 0.0
    assert receipt["specialist_gain_nats"] > 0.01
    assert receipt["final_general_drift_ratio"] <= 1.01
    assert receipt["injected_layer_expert_slots"] == 4
    assert verify_nullspace_injection_receipt_provenance(receipt) is True

    # Verify co-adapted (router, B_expert) bitwise identity against s4_anchor on returned model
    s4_blocks = extract_checkpoint_blocks(s4_anchor)
    out_blocks = extract_checkpoint_blocks(out_model)
    for bid, sub in s4_blocks.items():
        if bid.endswith(".router"):
            assert torch.equal(out_blocks[bid]["weight"], sub["weight"])
        elif ".experts." in bid:
            for sk, t in sub.items():
                if sk.endswith("_B"):
                    assert torch.equal(out_blocks[bid][sk], t)
    assert compute_checkpoint_sha256(out_blocks) == receipt["final_state_sha256"]


# ---------------------------------------------------------------------------
# S2: Exact Rank-r Preservation vs Full-Delta W @ P_0 SVD Truncation Error
# ---------------------------------------------------------------------------


def test_s2_input_nullspace_preserves_exact_rank_with_zero_svd_truncation_error() -> None:
    """S2: Input-side (A_donor - A_anchor) @ P_0 avoids rank inflation and SVD truncation error."""
    gen = torch.Generator(device="cpu").manual_seed(20261402)
    d_in, d_out, r, k_prose = 28, 24, 4, 6
    q_in, _ = torch.linalg.qr(torch.randn((d_in, d_in), generator=gen, dtype=torch.float64))
    q_out, _ = torch.linalg.qr(torch.randn((d_out, d_out), generator=gen, dtype=torch.float64))

    p_0 = torch.eye(d_in, dtype=torch.float64) - (q_in[:, :k_prose] @ q_in[:, :k_prose].T)
    x_prose = (torch.randn((24, k_prose), generator=gen, dtype=torch.float64) @ q_in[:, :k_prose].T).to(torch.float32)

    # Anchor, T7 donor, and C3 donor occupy mutually orthogonal output subspaces in R^{d_out}
    b_anchor = q_out[:, 0:r].to(torch.float32)
    b_t7 = q_out[:, r : 2 * r].to(torch.float32)
    b_c3 = q_out[:, 2 * r : 3 * r].to(torch.float32)

    a_anchor = (torch.randn((r, d_in - k_prose), generator=gen, dtype=torch.float64) @ q_in[:, k_prose:].T).to(torch.float32)
    a_t7 = (torch.randn((r, d_in - k_prose), generator=gen, dtype=torch.float64) @ q_in[:, k_prose:].T).to(torch.float32)
    a_c3 = (torch.randn((r, d_in - k_prose), generator=gen, dtype=torch.float64) @ q_in[:, k_prose:].T).to(torch.float32)

    cmp_res = compare_input_nullspace_vs_full_delta_refactorization(
        a_anchor,
        b_anchor,
        [b_t7, b_c3],
        [a_t7, a_c3],
        [0.35, 0.35],
        p_0,
        x_prose=x_prose,
        eps=1e-5,
    )

    assert cmp_res["target_rank_r"] == r
    assert cmp_res["input_injection_rank_r"] == r
    assert cmp_res["input_injection_rank_increase"] == 0
    assert cmp_res["input_injection_svd_truncation_error_fro"] == 0.0
    assert cmp_res["input_injection_b_anchor_unchanged"] is True
    assert cmp_res["input_injection_prose_invariance_max_diff"] < 1e-5

    assert cmp_res["full_delta_numerical_rank"] == 3 * r
    assert cmp_res["full_delta_rank_excess"] == 2 * r
    assert cmp_res["full_delta_svd_truncation_error_fro"] > 0.25
    assert cmp_res["full_delta_svd_truncation_rel_error"] > 0.10
    assert cmp_res["input_injection_zero_truncation_verified"] is True

    # Also cover x_prose=None branch
    cmp_no_x = compare_input_nullspace_vs_full_delta_refactorization(
        a_anchor, b_anchor, [b_t7], [a_t7], [0.35], p_0, eps=1e-5
    )
    assert cmp_no_x["full_delta_numerical_rank"] == 2 * r
    assert cmp_no_x["input_injection_svd_truncation_error_fro"] == 0.0


# ---------------------------------------------------------------------------
# S3: Un-Folding Prior Sharpening tau=0.98 & Safe Tau >= 1.0 E0 Drift Rescue
# ---------------------------------------------------------------------------


def test_s3_unfold_prior_tau_098_and_safe_tau_recalibration_rescues_e0_drift() -> None:
    """S3: Un-folding prior tau=0.98 sharpening & recalibrating over tau >= 1.0 rescues E0 >+1.0% to <+0.2%."""
    hidden_dim, vocab_size = 8, 8
    gen = torch.Generator(device="cpu").manual_seed(20261403)
    head_weight = torch.eye(vocab_size, hidden_dim, dtype=torch.float32)

    # Construct 4 general-prose chunks (noisy boundary tokens hurt by sharpening scale > 1.0)
    # and 4 specialist chunks (confident aligned tokens helped by moderate candidate LN delta)
    h_pre = torch.zeros((8, 6, hidden_dim), dtype=torch.float32)
    targets = torch.zeros((8, 6), dtype=torch.long)
    is_general = torch.tensor([True, True, True, True, False, False, False, False], dtype=torch.bool)

    for row_i in range(4):
        # General prose: target=0 has slightly lower pre-logit than distractor=1, so larger LN weight
        # or tau=0.98 sharpens distractor logit and drives general NLL > +1.0% above base
        vec_g = torch.tensor([-0.3, 0.9, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=torch.float32)
        h_pre[row_i] = vec_g + 0.02 * torch.randn((6, hidden_dim), generator=gen)
        targets[row_i] = 0

    for row_i in range(4, 8):
        # Specialist: target=2 strongly leads, so safe candidate delta lowers overall NLL
        vec_s = torch.tensor([-1.5, -1.5, 2.2, -1.5, -1.5, -1.5, -1.5, -1.5], dtype=torch.float32)
        h_pre[row_i] = vec_s + 0.02 * torch.randn((6, hidden_dim), generator=gen)
        targets[row_i] = 2

    base_ln_w = torch.ones(hidden_dim, dtype=torch.float32)
    base_ln_b = torch.zeros(hidden_dim, dtype=torch.float32)

    # Candidate un-folded weights slightly emphasize specialist dim 2 while slightly inflating dim 1
    raw_s5_w = torch.tensor([1.0, 1.45, 2.1, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    raw_s5_b = torch.tensor([0.0, 0.25, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    prior_tau = 0.98
    folded_s5_w = raw_s5_w / prior_tau
    folded_s5_b = raw_s5_b / prior_tau

    res = unfold_and_recalibrate_safe_layernorm_temperature(
        folded_s5_w,
        folded_s5_b,
        prior_tau,
        base_ln_w,
        base_ln_b,
        h_pre,
        head_weight,
        targets,
        is_general,
        tau_grid=(1.0, 1.02, 1.05),
        max_general_ratio=1.002,
    )

    assert torch.allclose(res["unfolded_ln_w"], raw_s5_w, atol=1e-6)
    assert torch.allclose(res["unfolded_ln_b"], raw_s5_b, atol=1e-6)
    assert res["general_drift_pct_before"] > 1.0
    assert res["guard_feasible"] is True
    assert res["best_tau_temp"] >= 1.0
    assert all(t >= 1.0 for t in res["tau_grid"])
    assert res["general_drift_pct_after"] < 0.2
    assert res["general_ratio_after"] <= 1.002 + 1e-9
    assert res["nll_mean"] < res["nll_before"]
    assert res["nll_drop_vs_folded_nats"] > 0.0
    assert res["matmul_speedup_factor"] > 1.0

    # Verify biasless RMSNorm-style (ln_b=None) + numpy mask + skipped batches (-100) branch
    targets_ign = targets.clone()
    targets_ign[:2] = -100
    res_nb = unfold_and_recalibrate_safe_layernorm_temperature(
        folded_s5_w,
        None,
        prior_tau,
        base_ln_w,
        None,
        h_pre,
        head_weight,
        targets_ign,
        is_general.numpy(),
        tau_grid=(1.0, 1.02, 1.05),
        max_general_ratio=1.002,
        seq_batch_size=2,
    )
    assert res_nb["recalibrated_ln_b"] is None
    assert res_nb["best_tau_temp"] >= 1.0
    assert res_nb["guard_feasible"] is True

    # End-to-end CrowdV1WithEngram rescue with prior_tau_folded=0.98
    base_model, s4_anchor, donors, probe_chunks, eval_chunks = _build_pilot6_multicampaign_models(
        seed=20261404
    )
    with torch.no_grad():
        s4_anchor.ln.weight.div_(0.98)
        s4_anchor.ln.bias.add_(0.05).div_(0.98)
    _, rec_e2e = execute_verified_multicampaign_nullspace_injection(
        base_model,
        s4_anchor,
        donors,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        donor_alphas={"T7_code": 0.45, "C3_science": 0.45},
        expert_donor_map={1: "T7_code", 6: "C3_science"},
        prior_tau_folded=0.98,
        config=NullspaceInputInjectionConfig(
            safe_tau_grid=(1.0, 1.02, 1.05),
            calibrator_max_general_ratio=1.002,
            score_from=4,
            batch_size=8,
            inplace=False,
        ),
    )
    assert rec_e2e["ln_recalibration_summary"]["executed"] is True
    assert rec_e2e["ln_recalibration_summary"]["prior_tau_unfolded"] == pytest.approx(0.98)
    assert rec_e2e["ln_recalibration_summary"]["best_tau_temp"] >= 1.0
    assert verify_nullspace_injection_receipt_provenance(rec_e2e) is True


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Atomic Bitwise Rollback on Infeasible Guard
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_bitwise_rollback_on_infeasible_guard() -> None:
    """S4: Prose guard breach (>+1.0%) or non-finite eval enforces ANSE Maximum Pain bitwise rollback."""
    base_model, s4_anchor, donors, probe_chunks, eval_chunks = _build_pilot6_multicampaign_models(
        seed=20261405
    )
    target_model = copy.deepcopy(s4_anchor)
    anchor_sha = compute_checkpoint_sha256(extract_checkpoint_blocks(s4_anchor))
    base_sha = compute_checkpoint_sha256(extract_checkpoint_blocks(base_model))

    def _guard_breach_eval(m: CrowdV1WithEngram) -> dict[str, float]:
        cur_sha = compute_checkpoint_sha256(extract_checkpoint_blocks(m))
        if cur_sha == base_sha:
            return {"E0_general_fineweb": 2.00, "E1_python_systems_code": 2.40, "E6_stellar_spectroscopy": 2.40}
        if cur_sha == anchor_sha:
            return {"E0_general_fineweb": 2.002, "E1_python_systems_code": 2.10, "E6_stellar_spectroscopy": 2.10}
        # Injected candidate breaches E0 <= +1.0% (2.05 / 2.00 = +2.5%)
        return {"E0_general_fineweb": 2.05, "E1_python_systems_code": 1.85, "E6_stellar_spectroscopy": 1.85}

    out_model, rec_rb = execute_verified_multicampaign_nullspace_injection(
        base_model,
        s4_anchor,
        donors,
        target_model=target_model,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        eval_fn=_guard_breach_eval,
        expert_donor_map={1: "T7_code", 6: "C3_science"},
        config=NullspaceInputInjectionConfig(
            score_from=4,
            batch_size=8,
            rollback_target="anchor",
            inplace=True,
        ),
    )
    assert out_model is target_model
    assert rec_rb["accepted"] is False
    assert rec_rb["rolled_back"] is True
    assert rec_rb["rollback_target"] == "anchor"
    assert rec_rb["rollback_bitwise_verified"] is True
    assert rec_rb["status"] == "ROLLED_BACK_ANSE_MAXIMUM_PAIN"
    assert rec_rb["candidate_metrics"]["anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert rec_rb["final_state_sha256"] == anchor_sha
    assert compute_checkpoint_sha256(extract_checkpoint_blocks(target_model)) == anchor_sha
    assert verify_nullspace_injection_receipt_provenance(rec_rb) is True

    # Dict-block mode testing rollback_target="base" & donor rollback, non-finite eval,
    # prose_hidden_states (Mapping/Sequence/Tensor), p_null_by_layer variants, prior_tau_folded
    gen = torch.Generator(device="cpu").manual_seed(20261406)
    base_blk = {
        "blocks.0.mlp.router": {"weight": torch.eye(4, 6)[:2]},
        "blocks.0.mlp.experts.0": {
            "w1_A": torch.ones((2, 6)),
            "w1_B": torch.zeros((4, 2)),
            "w2_A": torch.ones((2, 6)),
            "w2_B": torch.zeros((4, 2)),
            "w3_A": torch.ones((2, 4)),
            "w3_B": torch.zeros((6, 2)),
        },
        "blocks.0.mlp.experts.1": {
            "w1_A": torch.ones((2, 6)),
            "w1_B": torch.full((4, 2), 0.15),
            "w2_A": torch.ones((2, 6)),
            "w2_B": torch.full((4, 2), 0.15),
            "w3_A": torch.ones((2, 4)),
            "w3_B": torch.full((6, 2), 0.15),
        },
        "blocks.0.mlp.experts.2": {
            "w1": torch.ones((4, 6)),
            "w2": torch.ones((4, 6)),
            "w3": torch.ones((6, 4)),
        },
        "final_ln": {"weight": torch.ones(6), "bias": torch.zeros(6)},
    }
    anchor_blk = copy.deepcopy(base_blk)
    donor_blk = copy.deepcopy(base_blk)
    donor_blk["blocks.0.mlp.experts.1"]["w1_A"].add_(0.2 * torch.randn((2, 6), generator=gen))
    donor_blk["blocks.0.mlp.experts.1"]["w3_A"].add_(0.2 * torch.randn((2, 4), generator=gen))
    base_blk_sha = compute_checkpoint_sha256(base_blk)

    p_map = {"blocks.0": {"x": torch.eye(6), "act": torch.eye(4)}}
    for rb_target, exp_sha_key in (("base", "base_state_sha256"), ("T7", "donor_state_sha256")):
        out_b, rec_d = execute_verified_multicampaign_nullspace_injection(
            base_blk,
            anchor_blk,
            {"T7": donor_blk},
            p_null_by_layer=p_map,
            prior_tau_folded=0.98,
            eval_fn=lambda b: (
                {"E0_gen": 2.0, "E1_code": 2.0}
                if compute_checkpoint_sha256(b) == base_blk_sha
                else {"non_finite_outputs": True}
            ),
            config=NullspaceInputInjectionConfig(
                inject_projections=("w1", "w2", "w3"),
                reorthogonalize_projector=True,
                rollback_target=rb_target,
                inplace=False,
            ),
        )
        assert isinstance(out_b, dict)
        assert rec_d["rolled_back"] is True
        assert rec_d["rollback_bitwise_verified"] is True
        expected_sha = rec_d[exp_sha_key] if rb_target == "base" else rec_d[exp_sha_key]["T7"]
        assert rec_d["final_state_sha256"] == expected_sha
        assert verify_nullspace_injection_receipt_provenance(rec_d) is True

    # Cover prose_hidden_states Tensor/Mapping/Sequence + missing projector branch + base_checkpoint nn.Module fallback
    x_p = torch.randn((4, 6), generator=gen)
    for ph_arg in (x_p, {"blocks.0": x_p}, [x_p]):
        _, rec_ph = execute_verified_multicampaign_nullspace_injection(
            base_blk,
            anchor_blk,
            {"T7": donor_blk},
            prose_hidden_states=ph_arg,
            eval_fn=lambda _: {"E0_gen": 2.0, "E1_code": 2.0},
            config=NullspaceInputInjectionConfig(rollback_target="base"),
        )
        assert rec_ph["rolled_back"] is True

    # Base model nn.Module passed as base_checkpoint (without eval_chunks; falls back to probe_chunks)
    _, rec_bm = execute_verified_multicampaign_nullspace_injection(
        copy.deepcopy(base_model),
        extract_checkpoint_blocks(s4_anchor),
        {"T7_code": extract_checkpoint_blocks(donors["T7_code"])},
        probe_chunks=probe_chunks,
        config=NullspaceInputInjectionConfig(score_from=4, batch_size=8, inplace=False),
    )
    assert verify_nullspace_injection_receipt_provenance(rec_bm) is True


# ---------------------------------------------------------------------------
# S5: Validation Errors & Security / ZeroStub / OwnerOrderGuardrail Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_stub_owner_guardrail_gates() -> None:
    """S5: Cover config/API validation branches and AGPL/ZeroStub/PRSecurity/Lateral/OwnerGuardrail gates."""
    # 1. Config & receipt provenance validation
    with pytest.raises(ValueError, match="tau >= 1.0"):
        NullspaceInputInjectionConfig(safe_tau_grid=(0.98, 1.0))
    with pytest.raises(ValueError, match="Drift ratio bounds"):
        NullspaceInputInjectionConfig(calibrator_max_general_ratio=0.99)
    with pytest.raises(ValueError, match="inject_projections"):
        NullspaceInputInjectionConfig(inject_projections=())
    with pytest.raises(ValueError, match="inject_projections"):
        NullspaceInputInjectionConfig(inject_projections=("bad",))
    with pytest.raises(ValueError, match="safe_lambda_grid"):
        NullspaceInputInjectionConfig(safe_lambda_grid=())
    with pytest.raises(TypeError, match="safe_lambda_grid"):
        NullspaceInputInjectionConfig(safe_lambda_grid="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tau_grid"):
        NullspaceInputInjectionConfig(safe_tau_grid="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        NullspaceInputInjectionConfig(safe_tau_grid=())
    with pytest.raises(TypeError, match="bool"):
        NullspaceInputInjectionConfig(enforce_prose_invariance="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        NullspaceInputInjectionConfig(batch_size=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 1"):
        NullspaceInputInjectionConfig(batch_size=0)
    with pytest.raises(ValueError, match=">= 0"):
        NullspaceInputInjectionConfig(score_from=-1)
    with pytest.raises(TypeError, match="real number"):
        NullspaceInputInjectionConfig(default_alpha="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        NullspaceInputInjectionConfig(default_alpha=float("nan"))
    with pytest.raises(ValueError, match=">= 0.0"):
        NullspaceInputInjectionConfig(prose_invariance_eps=-0.1)
    with pytest.raises(ValueError, match="> 0.0"):
        NullspaceInputInjectionConfig(maximum_pain_energy=0.0)
    with pytest.raises(ValueError, match="<= 1.0"):
        NullspaceInputInjectionConfig(nullspace_variance_threshold=1.5)
    with pytest.raises(ValueError, match="> 0.0"):
        NullspaceInputInjectionConfig(nullspace_variance_threshold=0.0)
    with pytest.raises(ValueError, match=">= 1"):
        NullspaceInputInjectionConfig(nullspace_max_active_modes=0)
    with pytest.raises(ValueError, match="general_prefix"):
        NullspaceInputInjectionConfig(general_prefix=" ")

    assert verify_nullspace_injection_receipt_provenance({"receipt_sha256": "0" * 64, "object": "bad"}) is False
    assert verify_nullspace_injection_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert (
        verify_nullspace_injection_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": INJECTOR_RECEIPT_OBJECT,
                "schema": INJECTOR_SCHEMA,
                "bad": float("nan"),
            }
        )
        is False
    )
    with pytest.raises(TypeError, match="mapping"):
        verify_nullspace_injection_receipt_provenance("bad")  # type: ignore[arg-type]

    # 2. Component (a) & compare validation
    a_ok = torch.ones((2, 4))
    b_ok = torch.ones((3, 2))
    p_ok = torch.eye(4)
    with pytest.raises(TypeError, match="torch.Tensor"):
        inject_donor_lora_via_input_nullspace("bad", b_ok, [a_ok], [0.1], p_ok)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ndim == 2"):
        inject_donor_lora_via_input_nullspace(torch.ones(4), b_ok, [a_ok], [0.1], p_ok)
    with pytest.raises(ValueError, match="non-empty"):
        inject_donor_lora_via_input_nullspace(torch.empty((0, 4)), b_ok, [a_ok], [0.1], p_ok)
    with pytest.raises(TypeError, match="floating-point"):
        inject_donor_lora_via_input_nullspace(torch.ones((2, 4), dtype=torch.long), b_ok, [a_ok], [0.1], p_ok)
    with pytest.raises(ValueError, match="finite"):
        inject_donor_lora_via_input_nullspace(
            torch.tensor([[float("nan"), 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]), b_ok, [a_ok], [0.1], p_ok
        )
    with pytest.raises(ValueError, match="inner rank"):
        inject_donor_lora_via_input_nullspace(a_ok, torch.ones((3, 3)), [a_ok], [0.1], p_ok)
    with pytest.raises(ValueError, match="empty"):
        inject_donor_lora_via_input_nullspace(a_ok, b_ok, [], [], p_ok)
    with pytest.raises(TypeError, match="Sequence"):
        inject_donor_lora_via_input_nullspace(a_ok, b_ok, "bad", [0.1], p_ok)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        inject_donor_lora_via_input_nullspace(a_ok, b_ok, [torch.ones((2, 5))], [0.1], p_ok)
    with pytest.raises(ValueError, match="Scalar alpha"):
        inject_donor_lora_via_input_nullspace(a_ok, b_ok, [a_ok, a_ok], 0.1, p_ok)
    with pytest.raises(ValueError, match="alphas length"):
        inject_donor_lora_via_input_nullspace(a_ok, b_ok, [a_ok], [0.1, 0.2], p_ok)
    with pytest.raises(TypeError, match="alphas"):
        inject_donor_lora_via_input_nullspace(a_ok, b_ok, [a_ok], "bad", p_ok)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="square"):
        inject_donor_lora_via_input_nullspace(a_ok, b_ok, [a_ok], [0.1], torch.ones((4, 3)))
    with pytest.raises(ValueError, match="shape"):
        inject_donor_lora_via_input_nullspace(a_ok, b_ok, [a_ok], [0.1], torch.eye(5))
    with pytest.raises(ValueError, match="symmetric"):
        inject_donor_lora_via_input_nullspace(
            a_ok,
            b_ok,
            [a_ok],
            [0.1],
            torch.tensor([[1.0, 2.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        )
    with pytest.raises(ValueError, match="eigenvalues"):
        inject_donor_lora_via_input_nullspace(a_ok, b_ok, [a_ok], [0.1], 3.0 * torch.eye(4))
    with pytest.raises(ValueError, match="width"):
        inject_donor_lora_via_input_nullspace(a_ok, b_ok, [a_ok], [0.1], p_ok, x_prose=torch.ones((2, 5)))
    with pytest.raises(ValueError, match="Prose invariance violated"):
        inject_donor_lora_via_input_nullspace(
            a_ok, b_ok, [2.0 * a_ok], [0.5], p_ok, x_prose=torch.ones((2, 4)), eps=1e-6
        )

    with pytest.raises(ValueError, match="donor_b_list"):
        compare_input_nullspace_vs_full_delta_refactorization(a_ok, b_ok, [], [a_ok], [0.1], p_ok)
    with pytest.raises(ValueError, match="b_anchor shape"):
        compare_input_nullspace_vs_full_delta_refactorization(
            a_ok, b_ok, [torch.ones((4, 2))], [a_ok], [0.1], p_ok
        )

    # 3. Component (b) safe-tau validation
    h_ok = torch.ones((2, 4, 4))
    t_ok = torch.zeros((2, 4), dtype=torch.long)
    g_ok = [True, False]
    w_ok = torch.ones(4)
    b_ln_ok = torch.zeros(4)
    head_ok = torch.eye(4)
    with pytest.raises(ValueError, match="tau >= 1.0"):
        unfold_and_recalibrate_safe_layernorm_temperature(
            w_ok, b_ln_ok, 0.98, w_ok, b_ln_ok, h_ok, head_ok, t_ok, g_ok, tau_grid=(0.98, 1.02)
        )
    with pytest.raises(ValueError, match="max_general_ratio"):
        unfold_and_recalibrate_safe_layernorm_temperature(
            w_ok, b_ln_ok, 0.98, w_ok, b_ln_ok, h_ok, head_ok, t_ok, g_ok, max_general_ratio=0.95
        )
    with pytest.raises(ValueError, match="both be provided"):
        unfold_and_recalibrate_safe_layernorm_temperature(
            w_ok, b_ln_ok, 0.98, w_ok, None, h_ok, head_ok, t_ok, g_ok
        )
    with pytest.raises(ValueError, match="match hidden dim"):
        unfold_and_recalibrate_safe_layernorm_temperature(
            torch.ones(5), None, 0.98, w_ok, None, h_ok, head_ok, t_ok, g_ok
        )

    # 4. Component (c) layout/chunk/donor/eval validation
    min_blk = {
        "blocks.0.mlp.router": {"weight": torch.ones((2, 4))},
        "blocks.0.mlp.experts.0": {"w1_A": torch.ones((2, 4)), "w1_B": torch.ones((4, 2)), "w2_A": torch.ones((2, 4)), "w2_B": torch.ones((4, 2)), "w3_A": torch.ones((2, 4)), "w3_B": torch.ones((4, 2))},
    }
    ok_eval = lambda _: {"E0_gen": 1.0, "E1_code": 1.0}  # noqa: E731
    with pytest.raises(ValueError, match="donor_checkpoints"):
        execute_verified_multicampaign_nullspace_injection(min_blk, min_blk, {}, eval_fn=ok_eval)
    with pytest.raises(TypeError, match="eval_fn"):
        execute_verified_multicampaign_nullspace_injection(min_blk, min_blk, {"d": min_blk}, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="target_model"):
        execute_verified_multicampaign_nullspace_injection(min_blk, min_blk, {"d": min_blk}, target_model="bad", eval_fn=ok_eval)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="config"):
        execute_verified_multicampaign_nullspace_injection(min_blk, min_blk, {"d": min_blk}, config="bad", eval_fn=ok_eval)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rollback_target"):
        execute_verified_multicampaign_nullspace_injection(
            min_blk, min_blk, {"d": min_blk}, eval_fn=ok_eval, config=NullspaceInputInjectionConfig(rollback_target="missing")
        )
    with pytest.raises(ValueError, match="donor_alphas"):
        execute_verified_multicampaign_nullspace_injection(
            min_blk, min_blk, {"d": min_blk}, donor_alphas={"other": 0.1}, eval_fn=ok_eval
        )
    with pytest.raises(ValueError, match="unknown donor"):
        execute_verified_multicampaign_nullspace_injection(
            min_blk, min_blk, {"d": min_blk}, expert_donor_map={0: "ghost"}, eval_fn=ok_eval
        )
    with pytest.raises(TypeError, match="expert_donor_map"):
        execute_verified_multicampaign_nullspace_injection(
            min_blk, min_blk, {"d": min_blk}, expert_donor_map="bad", eval_fn=ok_eval  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="expert_donor_map"):
        execute_verified_multicampaign_nullspace_injection(
            min_blk, min_blk, {"d": min_blk}, expert_donor_map={0: []}, eval_fn=ok_eval
        )
    with pytest.raises(TypeError, match="non-negative integers"):
        execute_verified_multicampaign_nullspace_injection(
            min_blk, min_blk, {"d": min_blk}, protected_experts="bad", eval_fn=ok_eval  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="eval_fn"):
        execute_verified_multicampaign_nullspace_injection(min_blk, min_blk, {"d": min_blk})
    with pytest.raises(ValueError, match="MoE '.router'"):
        execute_verified_multicampaign_nullspace_injection({"x": {"w": torch.ones(2)}}, min_blk, {"d": min_blk}, eval_fn=ok_eval)
    with pytest.raises(ValueError, match="block layout"):
        execute_verified_multicampaign_nullspace_injection(
            min_blk, {"blocks.0.mlp.router": {"weight": torch.ones((2, 4))}}, {"d": min_blk}, eval_fn=ok_eval
        )
    with pytest.raises(ValueError, match="Sub-key mismatch"):
        execute_verified_multicampaign_nullspace_injection(
            min_blk,
            {"blocks.0.mlp.router": {"bias": torch.ones((2, 4))}, "blocks.0.mlp.experts.0": min_blk["blocks.0.mlp.experts.0"]},
            {"d": min_blk},
            eval_fn=ok_eval,
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        execute_verified_multicampaign_nullspace_injection(
            min_blk,
            {"blocks.0.mlp.router": {"weight": torch.ones((2, 5))}, "blocks.0.mlp.experts.0": min_blk["blocks.0.mlp.experts.0"]},
            {"d": min_blk},
            eval_fn=ok_eval,
        )
    for bad_blk, exc_t in (
        ("bad", TypeError),
        ({"": {"w": torch.ones(2)}}, ValueError),
        ({"b": {}}, TypeError),
        ({"b": {"w": torch.ones(2, dtype=torch.long)}}, ValueError),
        ({"blocks.0.mlp.router": {"weight": torch.tensor([[float("nan"), 0.0], [0.0, 1.0]])}, "blocks.0.mlp.experts.0": min_blk["blocks.0.mlp.experts.0"]}, ValueError),
    ):
        with pytest.raises(exc_t):
            execute_verified_multicampaign_nullspace_injection(bad_blk, min_blk, {"d": min_blk}, eval_fn=ok_eval)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="p_null_by_layer"):
        execute_verified_multicampaign_nullspace_injection(
            min_blk, min_blk, {"d": min_blk}, p_null_by_layer="bad", protected_experts=(), eval_fn=ok_eval  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="prose_hidden_states"):
        execute_verified_multicampaign_nullspace_injection(
            min_blk, min_blk, {"d": min_blk}, prose_hidden_states="bad", eval_fn=ok_eval  # type: ignore[arg-type]
        )
    for bad_ev, exc_ev in (
        ({}, ValueError),
        ({"E0_gen": "bad", "E1_code": 1.0}, TypeError),
        ({"E0_only": 1.0}, ValueError),
        ({"E0_gen": float("nan"), "E1_code": 1.0}, ValueError),
    ):
        with pytest.raises(exc_ev):
            execute_verified_multicampaign_nullspace_injection(
                min_blk, min_blk, {"d": min_blk}, eval_fn=lambda _, r=bad_ev: r
            )

    base_m_val, s4_m_val, donors_m_val, _, _ = _build_pilot6_multicampaign_models(seed=20261407, n_per_dom=1)
    for bad_chunks, exc_c in (
        ([], ValueError),
        ([{"ids": [0, 1]}], ValueError),
        ([{"expert_id": " ", "ids": [0, 1]}], ValueError),
        ([{"expert_id": "E0", "ids": "bad"}], TypeError),
        ([{"expert_id": "E0", "ids": [0]}], ValueError),
        ([{"expert_id": "E0", "ids": [-1, 1]}], ValueError),
        ([{"expert_id": "E0", "ids": [0, 1, 2]}, {"expert_id": "E1", "ids": [0, 1]}], ValueError),
        ([{"expert_id": "E0", "ids": [0, 1, 2]}, {"expert_id": "E1", "ids": [0, 1, 2]}], ValueError),
        ([{"expert_id": "E0", "ids": [0, 1, 2, 3]}, {"expert_id": "E0", "ids": [0, 1, 2, 3]}], ValueError),
    ):
        with pytest.raises(exc_c):
            execute_verified_multicampaign_nullspace_injection(
                base_m_val,
                s4_m_val,
                donors_m_val,
                probe_chunks=bad_chunks,  # type: ignore[arg-type]
                config=NullspaceInputInjectionConfig(score_from=3),
            )

    # 5. AGPL-3.0 header, ZeroStubAudit, PRSecurityScanner, lateral movement, FORBIDDEN_ADDED, OwnerOrderGuardrail
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn40", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path(
        "pr_gatekeeper_dyn40", REPO_ROOT / "scripts" / "pr_gatekeeper.py"
    )
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/rank_preserving_nullspace_input_injector.py", TARGET_MODULE_PATH),
        ("tests/test_rank_preserving_nullspace_input_injector_scenarios.py", TEST_FILE_PATH),
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
        {
            "nll_mean": 1.54,
            "prose_invariance_max_diff": 2.86e-6,
            "specialist_gain_nats": 0.5147,
            "anse_energy": -0.5147,
        },
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+TELEMETRY = {telemetry_json!r}"
        )
        == []
    )

    pr_title = "feat(swarm): rank-preserving prose-nullspace LoRA input injector and unfolded safe-tau LayerNorm recalibrator"
    pr_body = (
        "Implements NEURAL-CTRL-DYN-40 rank-preserving prose-nullspace LoRA input injector "
        "and unfolded safe-tau >= 1.0 LayerNorm recalibrator under ANSE Delta E < 0."
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
        pr_number=1240,
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
