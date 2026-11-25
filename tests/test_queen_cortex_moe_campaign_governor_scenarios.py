# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-41: Reine-Cortex (QueenCortex) MoE campaign governor."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hivemind.security.owner_order_guardrail import (
    RequestIntentCategory,
    classify_request_intent,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import upcycle_crowd_v1_to_sparse_moe
from hivemind.swarm.queen_cortex_moe_campaign_governor import (
    ANSE_MAXIMUM_PAIN,
    QUEEN_CAMPAIGN_RECEIPT_OBJECT,
    QUEEN_CORTEX_SCHEMA,
    QUEEN_PHEROMONE_OBJECT,
    QueenCortexConfig,
    QueenPheromoneDirective,
    _evaluate_model_on_chunks,
    compute_ker_g_d0_fisher_norm_step,
    distill_forager_nectar_to_royal_jelly,
    emit_queen_pheromone_campaign_directive,
    execute_queen_cortex_campaign_round,
    verify_queen_cortex_receipt_provenance,
    verify_queen_pheromone_directive_signature,
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
    REPO_ROOT / "src" / "hivemind" / "swarm" / "queen_cortex_moe_campaign_governor.py"
)
TEST_FILE_PATH = Path(__file__).resolve()
ROADMAP_DOC_PATH = REPO_ROOT / "docs" / "NEURO_CONTROLLER_ROADMAP.md"
PLAN_DOC_PATH = REPO_ROOT / "docs" / "NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md"
EXPECTED_AGPL_HEADER = (
    "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
)


@pytest.fixture(autouse=True, scope="module")
def _single_thread_cpu_torch() -> Any:
    """Eliminate OpenMP spin overhead on small CPU tensors during deterministic tests."""
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


def _install_domain_lora(
    exp_mod: Any,
    in_dir: torch.Tensor,
    out_dir: torch.Tensor,
    *,
    strength: float,
    inter_slice: slice,
    gate_scale: float = 2.0,
) -> None:
    """Write a clean rank-1 SwiGLU LoRA mapping ``in_dir -> strength * out_dir``."""
    i_start = inter_slice.start
    i_width = inter_slice.stop - inter_slice.start
    rank = int(exp_mod.w1_A.shape[0])
    n_act = min(rank, i_width)
    scaling = float(exp_mod.scaling)
    b_scale = math.sqrt(abs(strength)) / max(math.sqrt(scaling), 1e-6)
    sign_b = 1.0 if strength >= 0.0 else -1.0

    exp_mod.w1_A.zero_()
    exp_mod.w1_B.zero_()
    exp_mod.w2_A.zero_()
    exp_mod.w2_B.zero_()
    exp_mod.w3_A.zero_()
    exp_mod.w3_B.zero_()

    exp_mod.w1_A[0] = in_dir
    exp_mod.w2_A[0] = in_dir
    exp_mod.w1_B[i_start : i_start + n_act, 0] = gate_scale
    exp_mod.w2_B[i_start : i_start + n_act, 0] = gate_scale
    exp_mod.w3_A[0, i_start : i_start + n_act] = 1.0 / float(n_act)
    exp_mod.w3_B[:, 0] = sign_b * b_scale * out_dir


def _build_la_ruche_8expert_campaign_fixture(
    *,
    seed: int = 20264100,
    hidden_size: int = 48,
    intermediate_size: int = 64,
    num_layers: int = 2,
    vocab_size: int = 64,
    num_experts: int = 8,
    lora_rank: int = 4,
) -> dict[str, Any]:
    """Build a deterministic 8-expert CrowdV1 MoE hive with disjoint probe & eval splits."""
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
        max_position_embeddings=24,
        engram_buckets=0,
        tie_word_embeddings=False,
    )
    model = CrowdV1WithEngram(cfg)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    domain_specs = (
        (0, "E0_general_fineweb", 0, 0, slice(0, block_h)),
        (1, "E1_python_systems_code", 1, block_v, slice(block_h, 2 * block_h)),
        (2, "E2_math_reasoning", 2, 2 * block_v, slice(2 * block_h, 3 * block_h)),
        (7, "E7_formal_lean4", 3, 3 * block_v, slice(3 * block_h, 4 * block_h)),
    )

    axes_in: dict[int, torch.Tensor] = {}
    axes_out: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        head_w = 0.03 * torch.randn((vocab_size, hidden_size), generator=gen, dtype=torch.float32)

        for e_slot, _, b_idx, v_off, h_slice in domain_specs:
            basis = _make_zero_mean_ortho_rows(3, block_h, seed=seed + 50 + b_idx * 17)
            u_in, u_out = basis[0], basis[1]
            axes_in[e_slot] = u_in
            axes_out[e_slot] = u_out
            rows = _make_zero_mean_ortho_rows(min(6, block_h - 1), block_h, seed=seed + 180 + b_idx * 29)
            for t_i in range(block_v):
                vec = 0.25 * rows[t_i % int(rows.shape[0])] + 0.65 * u_in
                wte[v_off + t_i, h_slice] = vec - vec.mean()
            head_w[:, h_slice] = head_w[:, h_slice] - head_w[:, h_slice].mean(dim=1, keepdim=True)
            for tok in (v_off + 2, v_off + 3, v_off + 4):
                head_w[tok, h_slice] = 0.15 * u_in + 0.85 * u_out

        # Make base E0 well-calibrated while over-sharpening logit scale slightly (ECE > 0)
        model.wte.weight.copy_(wte)
        model.head.weight.copy_(head_w)
        model.ln.weight.fill_(1.45)
        model.ln.bias.zero_()
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln2.weight.fill_(1.0)
            blk.mlp.w1.weight[:hidden_size, :hidden_size].copy_(0.35 * torch.eye(hidden_size))
            blk.mlp.w2.weight[:hidden_size, :hidden_size].copy_(0.35 * torch.eye(hidden_size))
            blk.mlp.w3.weight[:hidden_size, :hidden_size].copy_(0.15 * torch.eye(hidden_size))

    upcycle_crowd_v1_to_sparse_moe(
        model,
        num_experts=num_experts,
        top_k=2,
        lora_rank=lora_rank,
        routing_mode="sigma_topk",
        seed=seed + 77,
    )
    with torch.no_grad():
        for blk in model.blocks:
            blk.mlp.router.weight.zero_()
            for e_slot, _, _, _, h_slice in domain_specs:
                blk.mlp.router.weight[e_slot, h_slice] = 3.5 * axes_in[e_slot]
            # Zero-initialize specialist LoRA adapters (standard LoRA step-0 dy=0); E0 keeps small anchor
            for exp_mod in blk.mlp.experts.values():
                for p_attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                    getattr(exp_mod, p_attr).zero_()
            f_in0 = torch.zeros(hidden_size, dtype=torch.float32)
            f_out0 = torch.zeros(hidden_size, dtype=torch.float32)
            f_in0[0:block_h] = axes_in[0]
            f_out0[0:block_h] = axes_out[0]
            _install_domain_lora(
                blk.mlp.experts["0"],
                f_in0,
                f_out0,
                strength=0.12,
                inter_slice=slice(0, block_h),
                gate_scale=0.35,
            )

    probe_chunks: list[dict[str, Any]] = []
    eval_chunks: list[dict[str, Any]] = []
    for _, dom_id, _b_idx, v_off, _ in domain_specs:
        toks_a = [v_off + 2, v_off + 3, v_off + 4, v_off + 2] * 3
        toks_b = [v_off + 3, v_off + 4, v_off + 2, v_off + 3] * 3
        for c_i in range(3):
            probe_chunks.append({"expert_id": dom_id, "ids": toks_a if c_i % 2 == 0 else toks_b})
            eval_chunks.append(
                {
                    "expert_id": dom_id,
                    "ids": ([v_off + 4, v_off + 2, v_off + 3, v_off + 4] * 3) if c_i % 2 == 0 else toks_a,
                }
            )

    return {
        "model": model,
        "probe_chunks": probe_chunks,
        "eval_chunks": eval_chunks,
        "axes_in": axes_in,
        "axes_out": axes_out,
        "domain_specs": domain_specs,
        "hidden_size": hidden_size,
        "block_h": block_h,
    }


def _make_forager_bee_round_deltas(
    base_blocks: dict[str, dict[str, torch.Tensor]],
    axes_in: dict[int, torch.Tensor],
    axes_out: dict[int, torch.Tensor],
    *,
    hidden_size: int,
    block_h: int,
    spec_step_by_expert: dict[int, float],
    prose_drift_scale: float,
    seed: int,
) -> list[dict[str, dict[str, torch.Tensor]]]:
    """Synthesize 4 volunteer Forager Bee deltas with real specialist gain + raw D0 drift + noise."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    expert_slices = {
        1: (slice(block_h, 2 * block_h), slice(block_h, 2 * block_h)),
        2: (slice(2 * block_h, 3 * block_h), slice(2 * block_h, 3 * block_h)),
        7: (slice(3 * block_h, 4 * block_h), slice(3 * block_h, 4 * block_h)),
    }
    e0_in = torch.zeros(hidden_size, dtype=torch.float32)
    e0_in[0:block_h] = axes_in[0]
    anti_e0_out = torch.zeros(hidden_size, dtype=torch.float32)
    anti_e0_out[0:block_h] = -axes_out[0]

    bees: list[dict[str, dict[str, torch.Tensor]]] = []
    for b_idx in range(4):
        bee_d: dict[str, dict[str, torch.Tensor]] = {}
        # Opposing-sign coordinate noise across bees (cleaned by TIES sign consensus)
        noise_sign = 1.0 if b_idx in (0, 1, 2) else -1.5
        for bid, sub in base_blocks.items():
            if bid.endswith(".router"):
                # Active specialist router nudge with prose leak so unguarded round routes onto drifted experts
                rw_delta = torch.zeros_like(sub["weight"])
                for e_active in spec_step_by_expert:
                    if e_active < int(rw_delta.shape[0]):
                        rw_delta[e_active, 0:block_h] = 0.35 * axes_in[0]
                bee_d[bid] = {"weight": rw_delta}
                continue
            if ".experts." not in bid:
                continue
            e_idx = int(bid.rsplit(".experts.", 1)[1])
            if e_idx not in expert_slices or e_idx not in spec_step_by_expert:
                continue
            h_sl, i_sl = expert_slices[e_idx]
            step_val = spec_step_by_expert.get(e_idx, 0.0)
            f_in = torch.zeros(hidden_size, dtype=torch.float32)
            f_in[h_sl] = axes_in[e_idx]
            f_out = torch.zeros(hidden_size, dtype=torch.float32)
            f_out[h_sl] = axes_out[e_idx]

            d_w1_a = torch.zeros_like(sub["w1_A"])
            d_w2_a = torch.zeros_like(sub["w2_A"])
            d_w1_b = torch.zeros_like(sub["w1_B"])
            d_w2_b = torch.zeros_like(sub["w2_B"])
            d_w3_a = torch.zeros_like(sub["w3_A"])
            d_w3_b = torch.zeros_like(sub["w3_B"])

            # Specialist path (rank 0) + destructive D0 prose leak on input factor w1_A/w2_A (rank 0 & 1)
            d_w1_a[0] = 0.45 * f_in + prose_drift_scale * e0_in
            d_w2_a[0] = 0.45 * f_in + prose_drift_scale * e0_in
            d_w1_b[i_sl.start : i_sl.start + 4, 0] = 1.2
            d_w2_b[i_sl.start : i_sl.start + 4, 0] = 1.2
            d_w3_a[0, i_sl.start : i_sl.start + 4] = 0.25
            d_w3_b[:, 0] = step_val * f_out

            # Extra row-1 prose corruptor driven purely by E0_in -> raw average wrecks D0 (>+2.5%), P_0 zeros it!
            d_w1_a[1] = 1.2 * prose_drift_scale * e0_in
            d_w2_a[1] = 1.2 * prose_drift_scale * e0_in
            d_w1_b[0:4, 1] = 1.8
            d_w2_b[0:4, 1] = 1.8
            d_w3_a[1, 0:4] = 0.25
            d_w3_b[:, 1] = 2.0 * prose_drift_scale * anti_e0_out

            # Small opposing-sign bee jitter
            d_w3_b[:, 2] = noise_sign * 0.01 * torch.randn(
                hidden_size, generator=gen, dtype=torch.float32
            )
            bee_d[bid] = {
                "w1_A": d_w1_a,
                "w1_B": d_w1_b,
                "w2_A": d_w2_a,
                "w2_B": d_w2_b,
                "w3_A": d_w3_a,
                "w3_B": d_w3_b,
            }
        bees.append(bee_d)
    return bees


# ---------------------------------------------------------------------------
# Scenario S1: Multi-Round La Ruche 8-Expert MoE Rescue & Monotone ANSE Descent
# ---------------------------------------------------------------------------


def test_s1_multiround_la_ruche_8expert_campaign_rescue_and_monotone_anse() -> None:
    """(S1) Rescue raw D0 > +2.5% rounds to D0 <= +0.2% with monotone ANSE descent & lower ECE."""
    fix = _build_la_ruche_8expert_campaign_fixture()
    base_pristine: CrowdV1WithEngram = fix["model"]
    base_eval = _evaluate_model_on_chunks(
        base_pristine, fix["eval_chunks"], score_from=4, general_prefixes=("E0_",)
    )
    current_model = copy.deepcopy(base_pristine)

    # Round-by-round progressive specialist nectar additions (each raw round drifts D0 > +2.5%)
    round_plans = [
        {1: 0.25},
        {2: 0.25},
        {7: 0.25},
    ]

    prev_anse = 0.0
    prev_directive: dict[str, Any] | None = None
    receipts: list[dict[str, Any]] = []
    for r_idx, spec_steps in enumerate(round_plans, start=1):
        parent_snap = copy.deepcopy(current_model)
        parent_blocks = extract_checkpoint_blocks(parent_snap)
        bee_deltas = _make_forager_bee_round_deltas(
            parent_blocks,
            fix["axes_in"],
            fix["axes_out"],
            hidden_size=fix["hidden_size"],
            block_h=fix["block_h"],
            spec_step_by_expert=spec_steps,
            prose_drift_scale=0.80,
            seed=20264100 + r_idx,
        )
        cfg = QueenCortexConfig(
            ties_density=0.25,
            anchor_gain_floor_nats=-0.50,
            score_from=4,
            inplace=False,
        )
        out_model, rec = execute_queen_cortex_campaign_round(
            base_pristine,
            bee_deltas,
            parent_checkpoint=parent_snap,
            probe_chunks=fix["probe_chunks"],
            eval_chunks=fix["eval_chunks"],
            previous_directive=prev_directive,
            domain_round_counts={
                "E1_python_systems_code": r_idx,
                "E2_math_reasoning": r_idx,
                "E7_formal_lean4": r_idx,
            },
            parent_anse_energy=prev_anse,
            campaign_id="la-ruche-8expert-s1-clean",
            round_index=r_idx,
            config=cfg,
        )
        assert verify_queen_cortex_receipt_provenance(rec) is True
        assert rec["accepted"] is True
        assert rec["rolled_back"] is False
        assert rec["rescued_from_discarded_round"] is True
        # Raw unguarded round must fail D0 > +2.5% (kept=[])
        assert rec["raw_unguarded_metrics"]["guard_ok"] is False
        assert rec["raw_unguarded_metrics"]["general_drift_ratio"] > 1.025
        # Active Reine-Cortex Royal-Jelly rescue keeps D0 <= +0.2%
        assert rec["general_drift_ratio"] <= 1.002
        assert rec["guard_ok"] is True
        # Strictly monotone ANSE descent across rounds and strictly lower ECE vs base
        assert rec["anse_energy"] < prev_anse
        assert rec["ece_value"] < base_eval["ece_value"]
        prev_anse = float(rec["anse_energy"])
        prev_directive = rec["pheromone_directive"]
        current_model = out_model
        receipts.append(rec)

    assert len(receipts) == 3
    assert receipts[2]["anse_energy"] < receipts[1]["anse_energy"] < receipts[0]["anse_energy"] < 0.0
    assert receipts[0]["distillation_summary"]["first_order_d0_orthogonality"] < 1e-4
    assert receipts[-1]["ece_value"] < 0.7 * base_eval["ece_value"]


# ---------------------------------------------------------------------------
# Scenario S2: Exact First-Order ker(G_D0) Fisher Norm Orthogonality (<1e-4)
# ---------------------------------------------------------------------------


def test_s2_exact_ker_g_d0_fisher_norm_first_order_orthogonality() -> None:
    """(S2) Closed-form ker(G_D0) Fisher norm step satisfies max |G_D0 @ delta_w| < 1e-4."""
    gen = torch.Generator(device="cpu").manual_seed(20264102)
    d_model = 48
    vocab = 36
    c_num = 10
    t_len = 8

    # Structured general vs specialist hidden directions so D0 gradient subspace is nontrivial & smaller than D
    basis = _make_zero_mean_ortho_rows(20, d_model, seed=20264103)
    d0_sub = basis[:4]  # rank-4 prose subspace
    spec_sub = basis[4:12]  # orthogonal specialist subspace

    u_chunks: list[torch.Tensor] = []
    t_chunks: list[torch.Tensor] = []
    mask_list: list[bool] = []
    for c_i in range(c_num):
        is_d0 = c_i < 4
        mask_list.append(is_d0)
        if is_d0:
            coeff = torch.randn((t_len, 4), generator=gen, dtype=torch.float32)
            u_c = coeff @ d0_sub
            tgt_c = torch.randint(0, 12, (t_len,), generator=gen, dtype=torch.long)
        else:
            coeff = torch.randn((t_len, 8), generator=gen, dtype=torch.float32)
            u_c = coeff @ spec_sub
            tgt_c = torch.randint(12, vocab, (t_len,), generator=gen, dtype=torch.long)
        u_c = u_c * torch.rsqrt(u_c.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        u_chunks.append(u_c)
        t_chunks.append(tgt_c)

    head_w = torch.zeros((vocab, d_model), dtype=torch.float32)
    head_w[:12, :] = 0.6 * torch.randn((12, 4), generator=gen) @ d0_sub
    head_w[12:, :] = 0.8 * torch.randn((vocab - 12, 8), generator=gen) @ spec_sub
    w_norm_base = torch.ones(d_model, dtype=torch.float32) * 1.35

    # P_0 orthogonal to d0_sub
    p_x = torch.eye(d_model, dtype=torch.float64) - (d0_sub.to(torch.float64).T @ d0_sub.to(torch.float64))
    p_act = torch.eye(32, dtype=torch.float64)

    base_experts = {
        0: {"w1_A": torch.zeros((4, d_model)), "w3_B": torch.zeros((d_model, 4))},
        "expert_1": {
            "w1_A": torch.zeros((4, d_model)),
            "w3_A": torch.zeros((4, 32)),
            "w3_B": torch.zeros((d_model, 4)),
            "w1_B": torch.zeros((d_model, 4)),
        },
        "expert_3": {"w1_A": torch.ones((4, d_model)), "w3_B": torch.ones((d_model, 4))},
        "expert_4": torch.zeros((16, d_model)),  # single-tensor expert path
        "expert_5": torch.zeros(d_model),  # 1D vector expert path
        "expert_6": torch.zeros((d_model, 16)),  # left-projection (d_out == p_dim) path
    }
    bee_deltas = {
        "bee_alpha": {
            "expert_1": {
                "w1_A": 0.4 * d0_sub[:4, :] + 0.5 * spec_sub[:4, :],
                "w3_A": torch.ones((4, 32)) * 0.1,
                "w3_B": torch.ones((d_model, 4)) * 0.2,
            },
            "expert_4": 0.5 * torch.ones((16, 1)) @ d0_sub[0:1, :] + 0.3 * torch.ones((16, 1)) @ spec_sub[4:5, :],
            "expert_5": 0.4 * d0_sub[0] + 0.5 * spec_sub[0],
            "expert_6": 0.3 * d0_sub[0:1, :].T @ torch.ones((1, 16)) + 0.4 * spec_sub[0:1, :].T @ torch.ones((1, 16)),
        },
        "bee_beta": {
            "expert_1": {
                "w1_A": 0.3 * d0_sub[:4, :] + 0.6 * spec_sub[:4, :],
                "w3_A": torch.ones((4, 32)) * 0.15,
                "w3_B": torch.ones((d_model, 4)) * 0.25,
            },
            "expert_4": 0.4 * torch.ones((16, 1)) @ d0_sub[0:1, :] + 0.4 * torch.ones((16, 1)) @ spec_sub[4:5, :],
            "expert_5": 0.35 * d0_sub[0] + 0.45 * spec_sub[0],
            "expert_6": 0.25 * d0_sub[0:1, :].T @ torch.ones((1, 16)) + 0.45 * spec_sub[0:1, :].T @ torch.ones((1, 16)),
        },
    }
    p_null_by_expert = {
        "expert_1": {"x": p_x, "act": p_act, "w3_B": None, "w1_B": p_x},
        "expert_4": p_x,
        "expert_5": p_x,
        "expert_6": p_x,
    }

    res = distill_forager_nectar_to_royal_jelly(
        base_experts,
        bee_deltas,
        p_null_by_expert,
        w_norm_base=w_norm_base,
        normed_unit_probe=u_chunks,
        head_weight=head_w,
        probe_targets=t_chunks,
        is_general_mask=mask_list,
        ties_density=0.25,
        ties_sign_mode="majority",
        bee_weights=[0.6, 0.4],
        sink_experts=["expert_3"],
        anchor_experts=["expert_0"],
    )

    assert res["schema"] == QUEEN_CORTEX_SCHEMA
    assert res["eta_fisher"] > 0.0
    delta_w = res["delta_w_nullspace"]
    assert float(torch.linalg.norm(delta_w).item()) > 1e-4

    # Independently verify analytical G_D0 @ delta_w orthogonality < 1e-4
    g_d0_mat = res["g_d0_matrix"].to(torch.float64)
    max_orth = float((g_d0_mat @ delta_w.to(torch.float64)).abs().max().item())
    assert max_orth < 1e-4
    assert res["first_order_d0_orthogonality"] < 1e-4
    assert res["probe_nll_after"] < res["probe_nll_before"]
    assert res["probe_d0_rel_drift"] <= 0.004

    # Verify anchor frozen, corruptor sunk, and P_0 prose projection exact on expert_1, 4, 5, 6
    assert res["expert_reports"]["0"]["action"] == "anchor_frozen"
    assert res["expert_reports"]["expert_3"]["action"] == "ghost_sunk"
    assert torch.equal(res["royal_jelly_experts"]["expert_3"]["w3_B"], torch.zeros((d_model, 4)))
    proj_leak_exp1 = float((res["royal_jelly_experts"]["expert_1"]["w1_A"] @ d0_sub.T).abs().max().item())
    proj_leak_exp4 = float((res["royal_jelly_experts"]["expert_4"] @ d0_sub.T).abs().max().item())
    proj_leak_exp5 = float((res["royal_jelly_experts"]["expert_5"] @ d0_sub.T).abs().max().item())
    proj_leak_exp6 = float((d0_sub @ res["royal_jelly_experts"]["expert_6"]).abs().max().item())
    assert proj_leak_exp1 < 1e-6
    assert proj_leak_exp4 < 1e-6
    assert proj_leak_exp5 < 1e-6
    assert proj_leak_exp6 < 1e-6

    # Also verify 2D normed_unit_probe + LayerNorm bias, ndarray mask, 2D (C, T) mask, and p_null=None
    u_2d = torch.stack(u_chunks, dim=0).reshape(-1, d_model)
    t_1d = torch.stack(t_chunks, dim=0).reshape(-1)
    m_2d = torch.tensor(mask_list, dtype=torch.bool).unsqueeze(1).expand(c_num, t_len).reshape(-1)
    b_ln = torch.zeros(d_model, dtype=torch.float32)
    w_2d, b_2d, meta_2d = compute_ker_g_d0_fisher_norm_step(
        w_norm_base, u_2d, head_w, t_1d, m_2d, b_norm_base=b_ln
    )
    assert w_2d.shape == w_norm_base.shape and b_2d is not None and b_2d.shape == b_ln.shape
    assert meta_2d["first_order_d0_orthogonality"] < 1e-4

    res_none_p0 = distill_forager_nectar_to_royal_jelly(
        {"expert_4": torch.zeros((16, d_model))},
        [{"expert_4": 0.1 * torch.ones((16, d_model))}],
        None,
        w_norm_base=w_norm_base,
        normed_unit_probe=torch.stack(u_chunks, dim=0),
        head_weight=head_w,
        probe_targets=torch.stack(t_chunks, dim=0),
        is_general_mask=np.array(mask_list, dtype=bool),
    )
    assert res_none_p0["first_order_d0_orthogonality"] < 1e-4
    _, _, meta_mask2d = compute_ker_g_d0_fisher_norm_step(
        w_norm_base,
        torch.stack(u_chunks, dim=0),
        head_w,
        torch.stack(t_chunks, dim=0),
        torch.tensor(mask_list, dtype=torch.bool).unsqueeze(1).expand(c_num, t_len),
    )
    assert meta_mask2d["first_order_d0_orthogonality"] < 1e-4


# ---------------------------------------------------------------------------
# Scenario S3: Signed QueenPheromoneDirective Verification
# ---------------------------------------------------------------------------


def test_s3_queen_pheromone_directive_planning() -> None:
    """(S3) Verify anchor freezes, corruptor sinks, router mode switch, and foraging weights."""
    gains = {
        "expert_0": -0.032,  # strong beneficial anchor (gain <= -0.01)
        "expert_1": -0.025,  # dual-active specialist (needs P_0, not frozen because harm > ceiling)
        "expert_2": -0.004,  # open specialist
        "expert_3": 0.014,   # pure corruptor
        "expert_4": -0.001,  # weak prose corruptor (harm > floor, no strong gain)
    }
    harms = {
        "expert_0": -0.002,
        "expert_1": 0.012,
        "expert_2": 0.0005,
        "expert_3": 0.024,
        "expert_4": 0.018,
    }
    domain_nlls = {
        "E0_general_fineweb": 2.10,
        "E1_python_systems_code": 2.25,
        "E2_math_reasoning": 2.55,
        "E6_formal_lean4": 3.45,
    }
    domain_eces = {
        "E0_general_fineweb": 0.020,
        "E1_python_systems_code": 0.030,
        "E2_math_reasoning": 0.045,
        "E6_formal_lean4": 0.095,
    }

    # Case A: Fragile co-adaptation KL (> ceiling 0.05) -> router_frozen
    d_fragile = emit_queen_pheromone_campaign_directive(
        gains,
        harms,
        {"block_0": 0.018, "block_1": 0.115},
        domain_nlls,
        domain_eces,
        anchor_gain_floor_nats=-0.01,
        coadapt_kl_ceiling=0.05,
        temperature=0.25,
        domain_round_counts={"E1_python_systems_code": 4, "E2_math_reasoning": 3, "E6_formal_lean4": 1},
        campaign_id="hive-c3-test",
        round_index=4,
    )
    assert isinstance(d_fragile, QueenPheromoneDirective)
    assert d_fragile.object == QUEEN_PHEROMONE_OBJECT
    assert d_fragile.anchor_experts_to_freeze == ("expert_0",)
    assert d_fragile.corruptor_experts_to_sink_or_reseed == ("expert_3", "expert_4")
    assert d_fragile.active_specialist_experts == ("expert_1", "expert_2")
    assert d_fragile.router_training_mode == "router_frozen"
    assert d_fragile.max_router_expert_kl_mismatch == pytest.approx(0.115)

    fw = d_fragile.bee_foraging_domain_weights
    assert fw["E0_general_fineweb"] == 0.0
    assert sum(fw.values()) == pytest.approx(1.0)
    assert fw["E6_formal_lean4"] > fw["E2_math_reasoning"] > fw["E1_python_systems_code"] > 0.0
    assert verify_queen_pheromone_directive_signature(d_fragile) is True

    tampered = d_fragile.to_dict()
    tampered["router_training_mode"] = "router_unfrozen_coadapted"
    assert verify_queen_pheromone_directive_signature(tampered) is False

    # Case B: Healthy co-adaptation KL (<= ceiling 0.05) -> router_unfrozen_coadapted
    d_healthy = emit_queen_pheromone_campaign_directive(
        gains,
        harms,
        0.022,
        {"E0_general": 2.0, "D0_prose": 2.3},
        {"E0_general": 0.01, "D0_prose": 0.04},
        coadapt_kl_ceiling=0.05,
    )
    assert d_healthy.router_training_mode == "router_unfrozen_coadapted"
    assert sum(d_healthy.bee_foraging_domain_weights.values()) == pytest.approx(1.0)
    assert verify_queen_pheromone_directive_signature(d_healthy.to_dict()) is True


# ---------------------------------------------------------------------------
# Scenario S4: ANSE Maximum Pain Atomic Bitwise Rollback on Byzantine Nectar
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_bitwise_rollback_on_byzantine_nectar() -> None:
    """(S4) Enforce ANSE Maximum Pain (1e6) & exact SHA-256 bitwise rollback on corrupted nectar."""
    # Part 1: Model mode with unguarded/poisoned finite delta breaching D0 > +1.0%
    fix = _build_la_ruche_8expert_campaign_fixture(seed=20264140)
    base_model: CrowdV1WithEngram = fix["model"]
    base_sha = compute_checkpoint_sha256(base_model)
    base_blocks = extract_checkpoint_blocks(base_model)

    # Pass identity P_0 and adversarial _B corrupting E0 prose so distillation cannot rescue it
    poison_bees = _make_forager_bee_round_deltas(
        base_blocks,
        fix["axes_in"],
        fix["axes_out"],
        hidden_size=fix["hidden_size"],
        block_h=fix["block_h"],
        spec_step_by_expert={1: -0.45, 2: -0.45, 7: -0.45},
        prose_drift_scale=3.5,
        seed=20264141,
    )
    identity_p_null = {
        bid: torch.eye(fix["hidden_size"], dtype=torch.float64)
        for bid in base_blocks
        if ".experts." in bid
    }
    out_model, rec_fin = execute_queen_cortex_campaign_round(
        base_model,
        poison_bees,
        p_null_by_expert=identity_p_null,
        probe_chunks=fix["probe_chunks"],
        eval_chunks=fix["eval_chunks"],
        config=QueenCortexConfig(score_from=4, inplace=True, rollback_target="base"),
        round_index=1,
    )
    assert rec_fin["accepted"] is False
    assert rec_fin["rolled_back"] is True
    assert rec_fin["rollback_target"] == "base"
    assert rec_fin["rollback_bitwise_verified"] is True
    assert rec_fin["status"] == "ROLLED_BACK_ANSE_MAXIMUM_PAIN"
    assert rec_fin["anse_energy"] == pytest.approx(ANSE_MAXIMUM_PAIN)
    assert rec_fin["final_state_sha256"] == base_sha
    assert compute_checkpoint_sha256(out_model) == base_sha
    assert verify_queen_cortex_receipt_provenance(rec_fin) is True

    # Part 2: Dict mode with non-finite (NaN) Byzantine nectar rolling back bitwise to parent
    parent_blocks = {
        "blocks.0.mlp.experts.1": {"w1_A": torch.ones((2, 8)), "w3_B": 0.2 * torch.ones((8, 2))},
        "blocks.0.mlp.router": {"weight": torch.ones((2, 8))},
        "final_ln": {"weight": torch.ones(8)},
    }
    base_d = {
        "blocks.0.mlp.experts.1": {"w1_A": torch.zeros((2, 8)), "w3_B": torch.zeros((8, 2))},
        "blocks.0.mlp.router": {"weight": torch.zeros((2, 8))},
        "final_ln": {"weight": torch.ones(8)},
    }
    nan_bee = [
        {"blocks.0.mlp.experts.1": {"w1_A": torch.full((2, 8), float("nan"))}}
    ]
    parent_sha = compute_checkpoint_sha256(parent_blocks)
    out_blocks, rec_nan = execute_queen_cortex_campaign_round(
        base_d,
        nan_bee,
        parent_checkpoint=parent_blocks,
        eval_fn=lambda _: {"E0_gen": 2.0, "E1_code": 2.2},
        w_norm_base=torch.ones(8),
        normed_unit_probe=torch.ones((4, 2, 8)),
        head_weight=torch.ones((6, 8)),
        probe_targets=torch.zeros((4, 2), dtype=torch.long),
        is_general_mask=[True, True, False, False],
        config=QueenCortexConfig(rollback_target="parent"),
        round_index=2,
    )
    assert rec_nan["accepted"] is False
    assert rec_nan["rolled_back"] is True
    assert rec_nan["rollback_target"] == "parent"
    assert rec_nan["rollback_bitwise_verified"] is True
    assert rec_nan["anse_energy"] == pytest.approx(ANSE_MAXIMUM_PAIN)
    assert rec_nan["final_state_sha256"] == parent_sha
    assert compute_checkpoint_sha256(out_blocks) == parent_sha
    assert verify_queen_cortex_receipt_provenance(rec_nan) is True

    # Part 3: Dict mode commit with dict forager_bee_deltas + explicit gains/harms + RMSNorm model + eval_fn
    def _dict_eval_fn(blks: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
        norm_val = float(blks["blocks.0.mlp.experts.1"]["w1_A"].abs().sum().item())
        return {
            "nll_mean": 2.0 - 0.05 * min(norm_val, 8.0),
            "ece_value": 0.03 - 0.002 * min(norm_val, 8.0),
            "domain_nlls": {"E0_gen": 2.0, "E1_code": 2.0 - 0.1 * min(norm_val, 8.0)},
            "domain_eces": {"E0_gen": 0.02, "E1_code": 0.04 - 0.004 * min(norm_val, 8.0)},
        }

    ok_bees = {"bee_1": {"blocks.0.mlp.experts.1": {"w1_A": 0.5 * torch.ones((2, 8))}}}
    out_ok_blk, rec_ok_blk = execute_queen_cortex_campaign_round(
        base_d,
        ok_bees,
        eval_fn=_dict_eval_fn,
        w_norm_base=torch.ones(8),
        normed_unit_probe=torch.ones((4, 2, 8)),
        head_weight=torch.ones((6, 8)),
        probe_targets=torch.zeros((4, 2), dtype=torch.long),
        is_general_mask=[True, True, False, False],
        expert_causal_gains_nats={"blocks.0.mlp.experts.1": -0.02},
        expert_prose_harm_nats={"blocks.0.mlp.experts.1": 0.0001},
    )
    assert rec_ok_blk["accepted"] is True
    assert float(out_ok_blk["blocks.0.mlp.experts.1"]["w1_A"].abs().sum().item()) > 0.0
    assert verify_queen_cortex_receipt_provenance(rec_ok_blk) is True

    # Part 4: RMSNorm (bias=None) model mode + eval_fn path
    rms_fix = _build_la_ruche_8expert_campaign_fixture(seed=20264145)
    rms_model: CrowdV1WithEngram = rms_fix["model"]
    rms_model.ln.bias = None  # bias-free RMSNorm simulation
    rms_bees = _make_forager_bee_round_deltas(
        extract_checkpoint_blocks(rms_model),
        rms_fix["axes_in"],
        rms_fix["axes_out"],
        hidden_size=rms_fix["hidden_size"],
        block_h=rms_fix["block_h"],
        spec_step_by_expert={1: 0.25},
        prose_drift_scale=0.80,
        seed=20264146,
    )
    _, rec_rms = execute_queen_cortex_campaign_round(
        rms_model,
        rms_bees,
        probe_chunks=rms_fix["probe_chunks"],
        eval_fn=lambda m: _evaluate_model_on_chunks(
            m, rms_fix["eval_chunks"], score_from=4, general_prefixes=("E0_",)
        ),
        config=QueenCortexConfig(score_from=4, inplace=False),
    )
    assert rec_rms["accepted"] is True
    assert verify_queen_cortex_receipt_provenance(rec_rms) is True


# ---------------------------------------------------------------------------
# Scenario S5: Validation Gates & Security / Zero-Stub / OwnerOrder Compliance
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_stub_owner_guardrail_compliance() -> None:
    """(S5) Assert strict input validation, zero stubs, zero secrets/lateral movement, and OwnerOrder."""
    # 1. Config validation
    with pytest.raises(ValueError, match="ties_density"):
        QueenCortexConfig(ties_density=0.0)
    with pytest.raises(ValueError, match="ties_sign_mode"):
        QueenCortexConfig(ties_sign_mode="bad")
    with pytest.raises(ValueError, match="max_peer_norm_ratio"):
        QueenCortexConfig(max_peer_norm_ratio=0.0)
    with pytest.raises(ValueError, match="rollback_target"):
        QueenCortexConfig(rollback_target="invalid")
    with pytest.raises(ValueError, match="general_prefixes"):
        QueenCortexConfig(general_prefixes=())
    with pytest.raises(TypeError, match="bool"):
        QueenCortexConfig(inplace="yes")  # type: ignore[arg-type]

    # 2. Fisher norm & Royal-Jelly distillation validation
    w_ok = torch.ones(4)
    u_ok = torch.ones((2, 2, 4))
    head_ok = torch.ones((5, 4))
    tgt_ok = torch.zeros((2, 2), dtype=torch.long)
    mask_ok = [True, False]
    base_exp_ok = {"e1": torch.ones((2, 4))}
    bee_ok = [{"e1": torch.ones((2, 4))}]

    with pytest.raises(ValueError, match="head_weight width"):
        compute_ker_g_d0_fisher_norm_step(w_ok, u_ok, torch.ones((5, 3)), tgt_ok, mask_ok)
    with pytest.raises(ValueError, match="b_norm_base dim"):
        compute_ker_g_d0_fisher_norm_step(w_ok, u_ok, head_ok, tgt_ok, mask_ok, b_norm_base=torch.ones(3))
    with pytest.raises(TypeError, match="probe_targets"):
        compute_ker_g_d0_fisher_norm_step(w_ok, u_ok, head_ok, torch.ones((2, 2)), mask_ok)
    with pytest.raises(ValueError, match="probe_targets IDs"):
        compute_ker_g_d0_fisher_norm_step(w_ok, u_ok, head_ok, torch.full((2, 2), 9), mask_ok)
    with pytest.raises(ValueError, match="at least one general-prose chunk"):
        compute_ker_g_d0_fisher_norm_step(w_ok, u_ok, head_ok, tgt_ok, [False, False])
    with pytest.raises(TypeError, match="is_general_mask"):
        compute_ker_g_d0_fisher_norm_step(w_ok, u_ok, head_ok, tgt_ok, "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="symmetric"):
        distill_forager_nectar_to_royal_jelly(
            base_exp_ok,
            bee_ok,
            {"e1": torch.tensor([[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])},
            w_norm_base=w_ok,
            normed_unit_probe=u_ok,
            head_weight=head_ok,
            probe_targets=tgt_ok,
            is_general_mask=mask_ok,
        )
    with pytest.raises(ValueError, match="eigenvalues"):
        distill_forager_nectar_to_royal_jelly(
            base_exp_ok,
            bee_ok,
            {"e1": 2.0 * torch.eye(4)},
            w_norm_base=w_ok,
            normed_unit_probe=u_ok,
            head_weight=head_ok,
            probe_targets=tgt_ok,
            is_general_mask=mask_ok,
        )
    with pytest.raises(ValueError, match="Unknown expert"):
        distill_forager_nectar_to_royal_jelly(
            base_exp_ok,
            [{"missing_exp": torch.ones((2, 4))}],
            None,
            w_norm_base=w_ok,
            normed_unit_probe=u_ok,
            head_weight=head_ok,
            probe_targets=tgt_ok,
            is_general_mask=mask_ok,
        )
    with pytest.raises(ValueError, match="Shape mismatch"):
        distill_forager_nectar_to_royal_jelly(
            base_exp_ok,
            [{"e1": torch.ones((3, 4))}],
            None,
            w_norm_base=w_ok,
            normed_unit_probe=u_ok,
            head_weight=head_ok,
            probe_targets=tgt_ok,
            is_general_mask=mask_ok,
        )
    with pytest.raises(ValueError, match="bee_weights"):
        distill_forager_nectar_to_royal_jelly(
            base_exp_ok,
            bee_ok,
            None,
            w_norm_base=w_ok,
            normed_unit_probe=u_ok,
            head_weight=head_ok,
            probe_targets=tgt_ok,
            is_general_mask=mask_ok,
            bee_weights=[0.0],
        )

    # 3. Pheromone & receipt signature verification + validation
    with pytest.raises(ValueError, match="identical keys"):
        emit_queen_pheromone_campaign_directive({"e0": -0.02}, {"e1": 0.01}, 0.01, {"E1": 1.0}, {"E1": 0.0})
    with pytest.raises(ValueError, match="identical domain keys"):
        emit_queen_pheromone_campaign_directive({"e0": -0.02}, {"e0": 0.01}, 0.01, {"E1": 1.0}, {"E2": 0.0})
    with pytest.raises(ValueError, match="temperature"):
        emit_queen_pheromone_campaign_directive({"e0": -0.02}, {"e0": 0.01}, 0.01, {"E1": 1.0}, {"E1": 0.0}, temperature=0.0)
    with pytest.raises(ValueError, match="empty"):
        emit_queen_pheromone_campaign_directive({"e0": -0.02}, {"e0": 0.01}, {}, {"E1": 1.0}, {"E1": 0.0})

    assert verify_queen_pheromone_directive_signature({"directive_sha256": "bad"}) is False
    assert verify_queen_cortex_receipt_provenance({"object": QUEEN_CAMPAIGN_RECEIPT_OBJECT, "schema": QUEEN_CORTEX_SCHEMA, "receipt_sha256": "0" * 64}) is False
    with pytest.raises(TypeError, match="directive"):
        verify_queen_pheromone_directive_signature("bad")  # type: ignore[arg-type]

    # 4. Campaign governor + exhaustive edge validation
    with pytest.raises(TypeError, match="base_checkpoint"):
        execute_queen_cortex_campaign_round("bad", bee_ok, eval_fn=lambda _: {"E0": 1.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires eval_fn"):
        execute_queen_cortex_campaign_round({"e1": torch.ones((2, 4))}, bee_ok)
    with pytest.raises(TypeError, match="config"):
        execute_queen_cortex_campaign_round({"e1": torch.ones((2, 4))}, bee_ok, config="bad", eval_fn=lambda _: {"E0": 1.0})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="eval_fn"):
        execute_queen_cortex_campaign_round({"e1": torch.ones((2, 4))}, bee_ok, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="parent_checkpoint"):
        execute_queen_cortex_campaign_round({"e1": torch.ones((2, 4))}, bee_ok, parent_checkpoint="bad", eval_fn=lambda _: {"E0": 1.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="trainable expert"):
        execute_queen_cortex_campaign_round({"final_ln": torch.ones(4)}, bee_ok, eval_fn=lambda _: {"E0": 1.0})
    with pytest.raises(TypeError, match="forager_bee_deltas"):
        execute_queen_cortex_campaign_round({"e1": torch.ones((2, 4))}, "bad", eval_fn=lambda _: {"E0": 1.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        execute_queen_cortex_campaign_round({"e1": torch.ones((2, 4))}, [], eval_fn=lambda _: {"E0": 1.0})
    with pytest.raises(TypeError, match="non-empty Mapping"):
        execute_queen_cortex_campaign_round({"e1": torch.ones((2, 4))}, [{}], eval_fn=lambda _: {"E0": 1.0})
    with pytest.raises(TypeError, match="evaluation result"):
        execute_queen_cortex_campaign_round({"e1": torch.ones((2, 4))}, bee_ok, eval_fn=lambda _: {})
    with pytest.raises(ValueError, match="domain_nlls"):
        execute_queen_cortex_campaign_round({"e1": torch.ones((2, 4))}, bee_ok, eval_fn=lambda _: {"domain_nlls": {}})
    with pytest.raises(ValueError, match="general-prose domain"):
        execute_queen_cortex_campaign_round({"e1": torch.ones((2, 4))}, bee_ok, eval_fn=lambda _: {"E1_only": 1.0})

    fix_val = _build_la_ruche_8expert_campaign_fixture(seed=20264155)
    m_val: CrowdV1WithEngram = fix_val["model"]
    with pytest.raises(ValueError, match="probe_chunks/eval_chunks"):
        execute_queen_cortex_campaign_round(m_val, [{"blocks.0.mlp.experts.1": {"w1_A": torch.zeros((4, 48))}}])
    for bad_chunks, exc_c in (
        ([], ValueError),
        (["bad"], TypeError),
        ([{"expert_id": "E0", "ids": [0]}], ValueError),
        ([{"expert_id": "E0", "ids": [0, 999]}], ValueError),
        ([{"expert_id": "E0", "ids": [0, 1, 2]}, {"expert_id": "E0", "ids": [0, 1]}], ValueError),
        ([{"expert_id": "E1", "ids": [0, 1, 2]}], ValueError),
    ):
        with pytest.raises(exc_c):
            execute_queen_cortex_campaign_round(
                m_val,
                [{"blocks.0.mlp.experts.1": {"w1_A": torch.zeros((4, 48))}}],
                probe_chunks=bad_chunks,  # type: ignore[arg-type]
            )
    del m_val.head
    with pytest.raises(ValueError, match="expose 'head'"):
        execute_queen_cortex_campaign_round(m_val, [{"blocks.0.mlp.experts.1": {"w1_A": torch.zeros((4, 48))}}])

    # Extra helper/validator error paths for complete guard coverage
    for fn_call, exc_t in (
        (lambda: QueenCortexConfig(score_from="bad"), TypeError),  # type: ignore[arg-type]
        (lambda: QueenCortexConfig(score_from=-1), ValueError),
        (lambda: QueenCortexConfig(foraging_temperature="bad"), TypeError),  # type: ignore[arg-type]
        (lambda: QueenCortexConfig(foraging_temperature=float("nan")), ValueError),
        (lambda: QueenCortexConfig(fisher_step_scales=[]), ValueError),
        (lambda: verify_queen_cortex_receipt_provenance("bad"), TypeError),  # type: ignore[arg-type]
        (lambda: verify_queen_cortex_receipt_provenance({"object": QUEEN_CAMPAIGN_RECEIPT_OBJECT, "schema": QUEEN_CORTEX_SCHEMA, "receipt_sha256": "a" * 64, "bad": float("nan")}), False),
        (lambda: compute_ker_g_d0_fisher_norm_step("bad", u_ok, head_ok, tgt_ok, mask_ok), TypeError),  # type: ignore[arg-type]
        (lambda: compute_ker_g_d0_fisher_norm_step(torch.ones((2, 2)), u_ok, head_ok, tgt_ok, mask_ok), ValueError),
        (lambda: compute_ker_g_d0_fisher_norm_step(w_ok, torch.ones(4), head_ok, tgt_ok, mask_ok), ValueError),
        (lambda: compute_ker_g_d0_fisher_norm_step(w_ok, torch.ones((0, 2, 4)), head_ok, tgt_ok, mask_ok), ValueError),
        (lambda: compute_ker_g_d0_fisher_norm_step(w_ok, torch.ones((2, 2, 4), dtype=torch.long), head_ok, tgt_ok, mask_ok), TypeError),
        (lambda: compute_ker_g_d0_fisher_norm_step(torch.tensor([1.0, float("nan"), 1.0, 1.0]), u_ok, head_ok, tgt_ok, mask_ok), ValueError),
        (lambda: compute_ker_g_d0_fisher_norm_step(w_ok, torch.ones((2, 2, 2, 4)), head_ok, tgt_ok, mask_ok), ValueError),
        (lambda: compute_ker_g_d0_fisher_norm_step(w_ok, [], head_ok, tgt_ok, mask_ok), ValueError),
        (lambda: compute_ker_g_d0_fisher_norm_step(w_ok, "bad", head_ok, tgt_ok, mask_ok), TypeError),  # type: ignore[arg-type]
        (lambda: compute_ker_g_d0_fisher_norm_step(w_ok, torch.ones((2, 2, 3)), head_ok, tgt_ok, mask_ok), ValueError),
        (lambda: compute_ker_g_d0_fisher_norm_step(w_ok, u_ok, head_ok, [], mask_ok), ValueError),
        (lambda: compute_ker_g_d0_fisher_norm_step(w_ok, u_ok, head_ok, "bad", mask_ok), TypeError),  # type: ignore[arg-type]
        (lambda: compute_ker_g_d0_fisher_norm_step(w_ok, u_ok, head_ok, torch.zeros((3, 2), dtype=torch.long), mask_ok), ValueError),
        (lambda: compute_ker_g_d0_fisher_norm_step(w_ok, u_ok, head_ok, tgt_ok, torch.ones(2)), TypeError),
        (lambda: compute_ker_g_d0_fisher_norm_step(w_ok, u_ok, head_ok, tgt_ok, np.array([1, 0])), TypeError),
        (lambda: compute_ker_g_d0_fisher_norm_step(w_ok, u_ok, head_ok, tgt_ok, [1, 0]), TypeError),  # type: ignore[list-item]
        (lambda: compute_ker_g_d0_fisher_norm_step(w_ok, u_ok, head_ok, tgt_ok, [True, True, False]), ValueError),
        (lambda: distill_forager_nectar_to_royal_jelly({}, bee_ok, None, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok), TypeError),
        (lambda: distill_forager_nectar_to_royal_jelly({True: torch.ones((2, 4))}, bee_ok, None, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok), TypeError),  # type: ignore[dict-item]
        (lambda: distill_forager_nectar_to_royal_jelly({" ": torch.ones((2, 4))}, bee_ok, None, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok), ValueError),
        (lambda: distill_forager_nectar_to_royal_jelly({1: torch.ones((2, 4)), "1": torch.ones((2, 4))}, bee_ok, None, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok), ValueError),
        (lambda: distill_forager_nectar_to_royal_jelly({"e1": "bad"}, bee_ok, None, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok), TypeError),
        (lambda: distill_forager_nectar_to_royal_jelly(base_exp_ok, {}, None, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok), ValueError),
        (lambda: distill_forager_nectar_to_royal_jelly(base_exp_ok, [], None, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok), ValueError),
        (lambda: distill_forager_nectar_to_royal_jelly(base_exp_ok, "bad", None, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok), TypeError),  # type: ignore[arg-type]
        (lambda: distill_forager_nectar_to_royal_jelly(base_exp_ok, ["bad"], None, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok), TypeError),  # type: ignore[list-item]
        (lambda: distill_forager_nectar_to_royal_jelly({"e1": {"w1_A": torch.ones((2, 4))}}, [{"e1": "bad"}], None, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok), TypeError),
        (lambda: distill_forager_nectar_to_royal_jelly({"e1": {"w1_A": torch.ones((2, 4))}}, [{"e1": {"bad_k": torch.ones((2, 4))}}], None, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok), ValueError),
        (lambda: distill_forager_nectar_to_royal_jelly({"e1": {"w1_A": torch.ones((2, 4))}}, [{"e1": {"w1_A": torch.ones((3, 4))}}], None, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok), ValueError),
        (lambda: distill_forager_nectar_to_royal_jelly(base_exp_ok, bee_ok, "bad", w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok), TypeError),  # type: ignore[arg-type]
        (lambda: distill_forager_nectar_to_royal_jelly(base_exp_ok, bee_ok, {"e1": "bad"}, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok), TypeError),
        (lambda: distill_forager_nectar_to_royal_jelly(base_exp_ok, bee_ok, {"e1": torch.ones((2, 3))}, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok), ValueError),
        (lambda: distill_forager_nectar_to_royal_jelly({"e1": torch.ones(4)}, [{"e1": torch.ones(4)}], {"e1": torch.eye(3)}, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok), ValueError),
        (lambda: distill_forager_nectar_to_royal_jelly({"e1": torch.ones((2, 2, 2))}, [{"e1": torch.ones((2, 2, 2))}], {"e1": torch.eye(2)}, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok), ValueError),
        (lambda: distill_forager_nectar_to_royal_jelly(base_exp_ok, bee_ok, {"e1": torch.eye(5)}, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok), ValueError),
        (lambda: distill_forager_nectar_to_royal_jelly(base_exp_ok, bee_ok, None, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok, ties_sign_mode="bad"), ValueError),
        (lambda: distill_forager_nectar_to_royal_jelly(base_exp_ok, bee_ok, None, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok, sink_experts="bad"), TypeError),
        (lambda: distill_forager_nectar_to_royal_jelly(base_exp_ok, bee_ok, None, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok, anchor_experts="bad"), TypeError),
        (lambda: distill_forager_nectar_to_royal_jelly(base_exp_ok, bee_ok, None, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok, bee_weights="bad"), TypeError),  # type: ignore[arg-type]
        (lambda: distill_forager_nectar_to_royal_jelly(base_exp_ok, bee_ok, None, w_norm_base=w_ok, normed_unit_probe=u_ok, head_weight=head_ok, probe_targets=tgt_ok, is_general_mask=mask_ok, bee_weights=[1.0, 2.0]), ValueError),
        (lambda: emit_queen_pheromone_campaign_directive({}, {"e0": 0.0}, 0.01, {"E1": 1.0}, {"E1": 0.0}), TypeError),
        (lambda: emit_queen_pheromone_campaign_directive({"e0": 0.0}, {}, 0.01, {"E1": 1.0}, {"E1": 0.0}), TypeError),
        (lambda: emit_queen_pheromone_campaign_directive({"e0": 0.0}, {"e0": 0.0}, 0.01, {}, {"E1": 0.0}), TypeError),
        (lambda: emit_queen_pheromone_campaign_directive({"e0": 0.0}, {"e0": 0.0}, 0.01, {"E1": 1.0}, {}), TypeError),
        (lambda: emit_queen_pheromone_campaign_directive({"e0": 0.0}, {"e0": 0.0}, 0.01, {"E1": 1.0}, {"E1": 0.0}, domain_round_counts="bad"), TypeError),  # type: ignore[arg-type]
    ):
        if exc_t is False:
            assert fn_call() is False
        else:
            with pytest.raises(exc_t):
                fn_call()

    # 5. AGPL-3.0 header, ZeroStubAudit, PRSecurityScanner, lateral movement, FORBIDDEN_ADDED, OwnerOrderGuardrail
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn41", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn41", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/queen_cortex_moe_campaign_governor.py", TARGET_MODULE_PATH),
        ("tests/test_queen_cortex_moe_campaign_governor_scenarios.py", TEST_FILE_PATH),
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
            "nll_mean": 1.48,
            "ece_value": 0.0156,
            "general_drift_ratio": 0.995,
            "anse_energy": -0.25,
            "foraging_priority_weight": 0.62,
        },
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+TELEMETRY = {telemetry_json!r}"
        )
        == []
    )

    pr_title = "feat(swarm): Reine-Cortex (QueenCortex) active MoE campaign governor, Royal-Jelly round rescuer, and bee foraging pheromone planner"
    pr_body = (
        "Implements NEURAL-CTRL-DYN-41 Reine-Cortex (QueenCortex) Royal-Jelly MoE campaign governor, "
        "ker(G_D0) Fisher norm calibrator, and signed QueenPheromoneDirective planner under ANSE Delta E < 0."
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
        pr_number=1241,
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
