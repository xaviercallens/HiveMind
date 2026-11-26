# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-43: closed-loop inter-round Royal-Jelly DiLoCo trophallaxis."""

from __future__ import annotations

import ast
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
from hivemind.swarm.inter_round_royal_jelly_trophallaxis import (
    ANSE_MAXIMUM_PAIN,
    CHAIN_SUMMARY_OBJECT,
    GENESIS_RECEIPT_SHA256,
    INTER_ROUND_TROPHALLAXIS_SCHEMA,
    ROUND_RECEIPT_OBJECT,
    InterRoundRoyalJellyTrophallaxisController,
    InterRoundTrophallaxisConfig,
    apply_inter_round_royal_jelly_momentum_step,
    compute_momentum_state_sha256,
    compute_prose_basis_from_projector,
    initialize_zero_momentum_state,
    measure_momentum_prose_leakage,
    resolve_pheromone_bee_weights,
    verify_inter_round_trophallaxis_chain,
    verify_inter_round_trophallaxis_receipt_hash,
)
from hivemind.swarm.queen_cortex_moe_campaign_governor import (
    _evaluate_model_on_chunks,
    emit_queen_pheromone_campaign_directive,
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
    REPO_ROOT / "src" / "hivemind" / "swarm" / "inter_round_royal_jelly_trophallaxis.py"
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
    """Write a deterministic rank-1 SwiGLU LoRA mapping ``in_dir -> strength * out_dir``."""
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


def _build_la_ruche_8expert_fixture(
    *,
    seed: int = 20264300,
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
    include_router_leak: bool = True,
    corruptor_expert: int | None = None,
    expert_input_scale: float = 0.45,
) -> list[dict[str, dict[str, torch.Tensor]]]:
    """Synthesize 4 volunteer Forager Bee deltas with specialist signal + D0 drift + noise."""
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
        noise_sign = 1.0 if b_idx in (0, 1, 2) else -1.5
        for bid, sub in base_blocks.items():
            if bid.endswith(".router"):
                if include_router_leak:
                    rw_delta = torch.zeros_like(sub["weight"])
                    for e_active in spec_step_by_expert:
                        if e_active < int(rw_delta.shape[0]):
                            rw_delta[e_active, 0:block_h] = 0.35 * axes_in[0]
                    bee_d[bid] = {"weight": rw_delta}
                continue
            if ".experts." not in bid:
                continue
            e_idx = int(bid.rsplit(".experts.", 1)[1])
            if corruptor_expert is not None and e_idx == corruptor_expert:
                bee_d[bid] = {
                    "w1_A": 0.9 * e0_in.unsqueeze(0).expand_as(sub["w1_A"]).clone(),
                    "w1_B": 0.7 * torch.ones_like(sub["w1_B"]),
                    "w2_A": 0.9 * e0_in.unsqueeze(0).expand_as(sub["w2_A"]).clone(),
                    "w2_B": 0.7 * torch.ones_like(sub["w2_B"]),
                    "w3_A": 0.2 * torch.ones_like(sub["w3_A"]),
                    "w3_B": 0.8 * anti_e0_out.unsqueeze(1).expand_as(sub["w3_B"]).clone(),
                }
                continue
            if e_idx not in expert_slices or e_idx not in spec_step_by_expert:
                continue
            h_sl, i_sl = expert_slices[e_idx]
            step_val = spec_step_by_expert[e_idx]
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

            d_w1_a[0] = expert_input_scale * f_in + prose_drift_scale * e0_in
            d_w2_a[0] = expert_input_scale * f_in + prose_drift_scale * e0_in
            d_w1_b[i_sl.start : i_sl.start + 4, 0] = 1.1
            d_w2_b[i_sl.start : i_sl.start + 4, 0] = 1.1
            d_w3_a[0, i_sl.start : i_sl.start + 4] = 0.25
            d_w3_b[:, 0] = step_val * f_out

            d_w1_a[1] = 1.2 * prose_drift_scale * e0_in
            d_w2_a[1] = 1.2 * prose_drift_scale * e0_in
            d_w1_b[0:4, 1] = 1.6
            d_w2_b[0:4, 1] = 1.6
            d_w3_a[1, 0:4] = 0.25
            d_w3_b[:, 1] = 1.8 * prose_drift_scale * anti_e0_out

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
# Scenario S1: Multi-Round Closed-Loop Royal-Jelly Beats Unguarded DiLoCo
# ---------------------------------------------------------------------------


def test_s1_multiround_closed_loop_royal_jelly_beats_unguarded_diloco() -> None:
    """(S1) Cumulative multi-round Royal-Jelly DiLoCo strictly beats unguarded DiLoCo at D0 <= +1.0%."""
    fix = _build_la_ruche_8expert_fixture(seed=20264301)
    base_model: CrowdV1WithEngram = fix["model"]
    base_eval = _evaluate_model_on_chunks(
        base_model, fix["eval_chunks"], score_from=4, general_prefixes=("E0_",)
    )
    base_blocks = extract_checkpoint_blocks(base_model)

    cfg = InterRoundTrophallaxisConfig(
        outer_lr=0.7,
        outer_momentum_beta=0.85,
        nesterov=True,
        anchor_gain_floor_nats=-0.50,
        score_from=4,
        inplace=False,
    )
    ctrl = InterRoundRoyalJellyTrophallaxisController(
        base_model,
        config=cfg,
        probe_chunks=fix["probe_chunks"],
        eval_chunks=fix["eval_chunks"],
        campaign_id="d1-trophallaxis-s1",
    )

    rounds = [
        {
            "forager_bee_deltas": _make_forager_bee_round_deltas(
                base_blocks,
                fix["axes_in"],
                fix["axes_out"],
                hidden_size=fix["hidden_size"],
                block_h=fix["block_h"],
                spec_step_by_expert={1: 0.22},
                prose_drift_scale=0.85,
                seed=20264311,
            ),
            "bee_domains": ["E1_python_systems_code"] * 4,
        },
        {
            "forager_bee_deltas": _make_forager_bee_round_deltas(
                base_blocks,
                fix["axes_in"],
                fix["axes_out"],
                hidden_size=fix["hidden_size"],
                block_h=fix["block_h"],
                spec_step_by_expert={2: 0.22},
                prose_drift_scale=0.85,
                seed=20264312,
            ),
            "bee_domains": ["E2_math_reasoning"] * 4,
        },
        {
            "forager_bee_deltas": _make_forager_bee_round_deltas(
                base_blocks,
                fix["axes_in"],
                fix["axes_out"],
                hidden_size=fix["hidden_size"],
                block_h=fix["block_h"],
                spec_step_by_expert={7: 0.22},
                prose_drift_scale=0.85,
                seed=20264313,
            ),
            "bee_domains": ["E7_formal_lean4"] * 4,
        },
    ]

    final_model, summary = ctrl.run_campaign(rounds)
    assert summary["object"] == CHAIN_SUMMARY_OBJECT
    assert summary["schema"] == INTER_ROUND_TROPHALLAXIS_SCHEMA
    assert summary["chain_verified"] is True
    assert summary["total_rounds"] == 3
    assert summary["accepted_rounds"] == 3
    assert summary["rolled_back_rounds"] == 0
    assert summary["rescued_rounds"] == 3
    assert summary["max_general_drift_ratio_observed"] <= 1.01
    assert summary["max_momentum_prose_leakage"] < 1e-8
    assert summary["cumulative_nll_delta"] < -0.05
    assert summary["cumulative_nll_delta_vs_unguarded_ratchet"] < -0.05

    recs = summary["round_receipts"]
    assert recs[0]["prev_receipt_sha256"] == GENESIS_RECEIPT_SHA256
    assert recs[2]["nll_mean"] < recs[1]["nll_mean"] < recs[0]["nll_mean"] < base_eval["nll_mean"]
    assert recs[2]["anse_energy"] < recs[1]["anse_energy"] < recs[0]["anse_energy"] < 0.0

    for idx, rec in enumerate(recs):
        assert rec["accepted"] is True
        assert rec["guard_ok"] is True
        assert rec["general_drift_ratio"] <= 1.01
        assert rec["worst_domain_drift_ratio"] <= 1.01
        assert rec["rescued_from_discarded_round"] is True
        assert rec["unguarded_diloco_metrics"]["would_be_rejected_by_guard"] is True
        assert rec["unguarded_diloco_metrics"]["ratchet_accepted"] is False
        assert rec["fisher_summary"]["first_order_d0_orthogonality"] < 1e-4
        assert rec["momentum_prose_leakage"]["max_abs_leakage"] < 1e-8
        assert verify_inter_round_trophallaxis_receipt_hash(rec) is True
        if idx > 0:
            assert rec["applied_directive_sha256"] == recs[idx - 1]["next_pheromone_directive"]["directive_sha256"]

    final_eval = _evaluate_model_on_chunks(
        final_model, fix["eval_chunks"], score_from=4, general_prefixes=("E0_",)
    )
    assert final_eval["nll_mean"] == pytest.approx(recs[-1]["nll_mean"])
    assert final_eval["ece_value"] < base_eval["ece_value"]
    assert ctrl.round_index == 3
    assert ctrl.best_anse_energy == pytest.approx(recs[-1]["anse_energy"])
    assert ctrl.head_receipt_sha256 == recs[-1]["receipt_sha256"]
    assert len(ctrl.receipts) == 3
    assert ctrl.active_directive is not None
    assert ctrl.domain_round_counts["E7_formal_lean4"] >= 1


# ---------------------------------------------------------------------------
# Scenario S2: Exact Lean 4 RCD_D1 Momentum Prose-Nullspace Invariance
# ---------------------------------------------------------------------------


def test_s2_lean4_rcd_d1_momentum_prose_nullspace_invariance() -> None:
    """(S2) Verify RCD_D1 ||M_t V||_F == 0 across multi-round Nesterov and heavy-ball DiLoCo."""
    d_in = 32
    d_out = 16
    basis = _make_zero_mean_ortho_rows(12, d_in, seed=20264320).to(torch.float64)
    v_prose = basis[:4].T.contiguous()  # (32, 4) orthonormal columns
    spec_dirs = basis[4:8]  # (4, 32) orthogonal specialist modes
    p_0 = torch.eye(d_in, dtype=torch.float64) - (v_prose @ v_prose.T)

    recovered_v = compute_prose_basis_from_projector(p_0)
    assert recovered_v.shape == (d_in, 4)
    gram_err = float((recovered_v.T @ recovered_v - torch.eye(4, dtype=torch.float64)).abs().max().item())
    assert gram_err < 1e-10
    empty_v = compute_prose_basis_from_projector(torch.eye(d_in, dtype=torch.float64))
    assert empty_v.shape == (d_in, 0)

    parent_blocks = {
        "blocks.0.mlp.experts.1": {
            "w1_A": torch.zeros((4, d_in), dtype=torch.float32),
            "w3_B": torch.zeros((d_out, 4), dtype=torch.float32),
        },
        "blocks.0.mlp.experts.2": {
            "w1_A": torch.zeros((4, d_in), dtype=torch.float32),
        },
        "blocks.0.mlp.router": {"weight": torch.zeros((4, d_in), dtype=torch.float32)},
        "final_ln": {"weight": torch.ones(d_in, dtype=torch.float32)},
    }
    p_null_map = {
        "blocks.0.mlp.experts.1": {"x": p_0, "w3_B": None},
        "blocks.0.mlp.experts.2": {"x": p_0},
        "blocks.0.mlp.experts.0": {"x": p_0},
    }

    for nesterov_flag, beta_val in ((True, 0.9), (False, 0.85)):
        cfg = InterRoundTrophallaxisConfig(
            outer_lr=0.7,
            outer_momentum_beta=beta_val,
            nesterov=nesterov_flag,
            project_momentum_on_step=True,
        )
        cur_blocks = parent_blocks
        m_state = initialize_zero_momentum_state(cur_blocks)
        m_ung = torch.zeros((4, d_in), dtype=torch.float64)

        for r_idx in range(1, 6):
            # Raw bee delta mixes strong prose contamination (0.6 * v_prose) with specialist signal (0.25 * spec)
            prose_leak = (0.6 + 0.05 * r_idx) * v_prose.T.to(torch.float32)
            spec_sig = 0.25 * spec_dirs.to(torch.float32)
            raw_delta_1 = prose_leak + spec_sig

            # Peers omit expert 2 on odd rounds so momentum-carry path in P_0 is also exercised
            bee_0: dict[str, Any] = {
                "blocks.0.mlp.experts.1": {"w1_A": raw_delta_1, "w3_B": 0.1 * torch.ones((d_out, 4))},
                "blocks.0.mlp.router": {"weight": prose_leak + 0.2 * spec_sig},
            }
            bee_1: dict[str, Any] = {
                "blocks.0.mlp.experts.1": {"w1_A": raw_delta_1 * 0.95, "w3_B": 0.12 * torch.ones((d_out, 4))},
                "blocks.0.mlp.router": {"weight": prose_leak * 0.9 + 0.22 * spec_sig},
            }
            if r_idx % 2 == 1:
                bee_0["blocks.0.mlp.experts.2"] = {"w1_A": raw_delta_1 * 0.5}
                bee_1["blocks.0.mlp.experts.2"] = {"w1_A": raw_delta_1 * 0.55}

            step = apply_inter_round_royal_jelly_momentum_step(
                cur_blocks,
                [bee_0, bee_1],
                m_state,
                p_null_map,
                config=cfg,
            )
            cur_blocks = step["candidate_blocks"]
            m_state = step["candidate_momentum"]

            m_ung = beta_val * m_ung + raw_delta_1.to(torch.float64)
            ung_prose_leak = float(torch.linalg.matrix_norm(m_ung @ v_prose).item())
            assert ung_prose_leak > 1.0

            rj_m1 = m_state["blocks.0.mlp.experts.1"]["w1_A"]
            rj_m2 = m_state["blocks.0.mlp.experts.2"]["w1_A"]
            rj_mr = m_state["blocks.0.mlp.router"]["weight"]
            assert float(torch.linalg.matrix_norm(rj_m1 @ v_prose).item()) < 1e-10
            assert float(torch.linalg.matrix_norm(rj_m2 @ v_prose).item()) < 1e-10
            assert float(torch.linalg.matrix_norm(rj_mr @ v_prose).item()) < 1e-10
            assert step["momentum_prose_leakage"]["max_abs_leakage"] < 1e-10
            assert float(torch.linalg.matrix_norm(rj_m1 @ spec_dirs.T).item()) > 0.2

    # Compare cumulative specialist step norm: beta=0.9 momentum > beta=0.0 momentum-free on aligned rounds
    cfg_mom = InterRoundTrophallaxisConfig(outer_lr=0.7, outer_momentum_beta=0.9, nesterov=True)
    cfg_nomom = InterRoundTrophallaxisConfig(outer_lr=0.7, outer_momentum_beta=0.0, nesterov=False)
    blk_mom, m_mom = parent_blocks, initialize_zero_momentum_state(parent_blocks)
    blk_nomom, m_nomom = parent_blocks, initialize_zero_momentum_state(parent_blocks)
    aligned_delta = 0.2 * spec_dirs.to(torch.float32) + 0.5 * v_prose.T.to(torch.float32)
    for _ in range(3):
        bees_al = [{"blocks.0.mlp.experts.1": {"w1_A": aligned_delta}}] * 2
        s_m = apply_inter_round_royal_jelly_momentum_step(blk_mom, bees_al, m_mom, p_null_map, config=cfg_mom)
        s_nm = apply_inter_round_royal_jelly_momentum_step(blk_nomom, bees_al, m_nomom, p_null_map, config=cfg_nomom)
        blk_mom, m_mom = s_m["candidate_blocks"], s_m["candidate_momentum"]
        blk_nomom, m_nomom = s_nm["candidate_blocks"], s_nm["candidate_momentum"]
    spec_gain_mom = float(torch.linalg.matrix_norm(blk_mom["blocks.0.mlp.experts.1"]["w1_A"].to(torch.float64) @ spec_dirs.T).item())
    spec_gain_nomom = float(torch.linalg.matrix_norm(blk_nomom["blocks.0.mlp.experts.1"]["w1_A"].to(torch.float64) @ spec_dirs.T).item())
    assert spec_gain_mom > 2.5 * spec_gain_nomom


# ---------------------------------------------------------------------------
# Scenario S3: Pheromone Propagation (Anchor Freeze, Corruptor Isolation, Router & UCB)
# ---------------------------------------------------------------------------


def test_s3_pheromone_propagation_anchor_freeze_corruptor_isolation_and_foraging() -> None:
    """(S3) Propagate QueenPheromoneDirective anchor freeze, sink isolation, router toggle & foraging."""
    fix = _build_la_ruche_8expert_fixture(seed=20264330)
    model: CrowdV1WithEngram = fix["model"]
    base_blocks = extract_checkpoint_blocks(model)

    cfg = InterRoundTrophallaxisConfig(
        outer_lr=0.75,
        outer_momentum_beta=0.80,
        nesterov=True,
        anchor_gain_floor_nats=-0.02,
        corruptor_prose_harm_floor_nats=0.002,
        coadapt_kl_ceiling=0.05,
        score_from=4,
    )
    ctrl = InterRoundRoyalJellyTrophallaxisController(
        model,
        config=cfg,
        probe_chunks=fix["probe_chunks"],
        eval_chunks=fix["eval_chunks"],
        campaign_id="pheromone-prop-s3",
    )

    # Round 1: train expert 1 cleanly while supplying explicit LOEO gains/harms + high KL -> freeze exp 1 & router, sink exp 3
    r1_bees = _make_forager_bee_round_deltas(
        base_blocks,
        fix["axes_in"],
        fix["axes_out"],
        hidden_size=fix["hidden_size"],
        block_h=fix["block_h"],
        spec_step_by_expert={1: 0.25},
        prose_drift_scale=0.80,
        seed=20264331,
    )
    gains_r1 = {f"expert_{e}": 0.0 for e in range(8)}
    gains_r1["expert_1"] = -0.032  # strong anchor
    gains_r1["expert_3"] = 0.015   # corruptor
    harms_r1 = {f"expert_{e}": 0.0 for e in range(8)}
    harms_r1["expert_1"] = -0.001
    harms_r1["expert_3"] = 0.024

    ckpt_r1, rec_r1 = ctrl.step_outer_round(
        r1_bees,
        expert_causal_gains_nats=gains_r1,
        expert_prose_harm_nats=harms_r1,
        router_expert_kl_mismatch={"block_0": 0.02, "block_1": 0.14},
        bee_domains=["E1_python_systems_code"] * 4,
    )
    assert rec_r1["accepted"] is True
    dir_r1 = rec_r1["next_pheromone_directive"]
    assert "expert_1" in dir_r1["anchor_experts_to_freeze"]
    assert "expert_3" in dir_r1["corruptor_experts_to_sink_or_reseed"]
    assert dir_r1["router_training_mode"] == "router_frozen"
    assert verify_queen_pheromone_directive_signature(dir_r1) is True

    r1_blocks = extract_checkpoint_blocks(ckpt_r1)
    r1_exp1_snap = {k: v.clone() for k, v in r1_blocks["blocks.0.mlp.experts.1"].items()}
    r1_router_snap = r1_blocks["blocks.0.mlp.router"]["weight"].clone()

    # Round 2: bees try to overwrite anchor expert 1, poison corruptor 3, nudge router, and advance expert 2 & 7
    r2_bees = _make_forager_bee_round_deltas(
        r1_blocks,
        fix["axes_in"],
        fix["axes_out"],
        hidden_size=fix["hidden_size"],
        block_h=fix["block_h"],
        spec_step_by_expert={1: -0.40, 2: 0.25, 7: 0.25},
        prose_drift_scale=0.80,
        seed=20264332,
        corruptor_expert=3,
    )
    r2_bee_map = {f"bee_{i}": b for i, b in enumerate(r2_bees)}
    r2_dom_map = {
        "bee_0": "E2_math_reasoning",
        "bee_1": "E7_formal_lean4",
        "bee_2": "E7_formal_lean4",
        "bee_3": "E1_python_systems_code",
    }
    ckpt_r2, rec_r2 = ctrl.step_outer_round(
        r2_bee_map,
        expert_causal_gains_nats=gains_r1,
        expert_prose_harm_nats=harms_r1,
        router_expert_kl_mismatch=0.01,  # KL heals -> Round 3 should unfreeze router
        bee_domains=r2_dom_map,
    )
    assert rec_r2["accepted"] is True
    assert rec_r2["applied_directive_sha256"] == dir_r1["directive_sha256"]
    assert rec_r2["router_action"] == "router_frozen_bitwise"
    assert rec_r2["next_pheromone_directive"]["router_training_mode"] == "router_unfrozen_coadapted"

    r2_blocks = extract_checkpoint_blocks(ckpt_r2)
    # Anchor expert 1 preserved bitwise and its momentum zeroed
    for sk, r1_t in r1_exp1_snap.items():
        assert torch.equal(r2_blocks["blocks.0.mlp.experts.1"][sk], r1_t)
        assert torch.equal(
            ctrl.momentum_state["blocks.0.mlp.experts.1"][sk],
            torch.zeros_like(ctrl.momentum_state["blocks.0.mlp.experts.1"][sk]),
        )
    # Corruptor expert 3 sunk (w_B == 0) and momentum zeroed
    for b_key in ("w1_B", "w2_B", "w3_B"):
        assert torch.equal(r2_blocks["blocks.0.mlp.experts.3"][b_key], torch.zeros_like(r2_blocks["blocks.0.mlp.experts.3"][b_key]))
    assert torch.equal(
        ctrl.momentum_state["blocks.0.mlp.experts.3"]["w1_A"],
        torch.zeros_like(ctrl.momentum_state["blocks.0.mlp.experts.3"]["w1_A"]),
    )
    # Router remained frozen bitwise during Round 2
    assert torch.equal(r2_blocks["blocks.0.mlp.router"]["weight"], r1_router_snap)

    # Round 3: router unfreezes cleanly under propagated QueenPheromoneDirective
    r3_bees = _make_forager_bee_round_deltas(
        r2_blocks,
        fix["axes_in"],
        fix["axes_out"],
        hidden_size=fix["hidden_size"],
        block_h=fix["block_h"],
        spec_step_by_expert={7: 0.22},
        prose_drift_scale=0.75,
        seed=20264333,
    )
    _, rec_r3 = ctrl.step_outer_round(
        r3_bees,
        expert_causal_gains_nats=gains_r1,
        expert_prose_harm_nats=harms_r1,
        router_expert_kl_mismatch=0.01,
        bee_domains=["E7_formal_lean4"] * 4,
    )
    assert rec_r3["accepted"] is True
    assert rec_r3["router_action"] == "router_trophallaxis_p0"
    assert verify_inter_round_trophallaxis_chain(ctrl.receipts, campaign_id="pheromone-prop-s3")["valid"] is True

    # Direct check of pheromone bee foraging weight resolution (prefix match + explicit weights + fallback)
    w_pref = resolve_pheromone_bee_weights(
        ["b0", "b1"],
        bee_domains=["E7", "E1"],
        directive=dir_r1,
        weight_floor=0.10,
    )
    assert sum(w_pref) == pytest.approx(1.0)
    assert w_pref[0] != w_pref[1]
    assert resolve_pheromone_bee_weights(["b0", "b1"], explicit_bee_weights=[3.0, 1.0]) == pytest.approx([0.75, 0.25])
    assert resolve_pheromone_bee_weights(["b0", "b1"], bee_domains=["a", "b"], directive={"bee_foraging_domain_weights": {}}) == pytest.approx([0.5, 0.5])


# ---------------------------------------------------------------------------
# Scenario S4: Atomic Bitwise Rollback (Weights + Momentum) & Chain Tamper Audits
# ---------------------------------------------------------------------------


def test_s4_atomic_bitwise_rollback_weights_and_momentum_and_chain_tamper_detection() -> None:
    """(S4) Enforce atomic bitwise rollback of weights & momentum under Maximum Pain & tamper checks."""
    fix = _build_la_ruche_8expert_fixture(seed=20264340)
    model: CrowdV1WithEngram = fix["model"]
    base_blocks = extract_checkpoint_blocks(model)

    cfg = InterRoundTrophallaxisConfig(
        outer_lr=0.7,
        outer_momentum_beta=0.85,
        anchor_gain_floor_nats=-0.50,
        score_from=4,
        inplace=True,
    )
    ctrl = InterRoundRoyalJellyTrophallaxisController(
        model,
        config=cfg,
        probe_chunks=fix["probe_chunks"],
        eval_chunks=fix["eval_chunks"],
        campaign_id="rollback-chain-s4",
    )

    # Round 1: Clean commit
    r1_bees = _make_forager_bee_round_deltas(
        base_blocks,
        fix["axes_in"],
        fix["axes_out"],
        hidden_size=fix["hidden_size"],
        block_h=fix["block_h"],
        spec_step_by_expert={1: 0.24},
        prose_drift_scale=0.80,
        seed=20264341,
    )
    ckpt_1, rec_1 = ctrl.step_outer_round(r1_bees)
    assert rec_1["accepted"] is True
    r1_state_sha = rec_1["final_state_sha256"]
    r1_mom_sha = rec_1["final_momentum_sha256"]
    assert compute_checkpoint_sha256(ckpt_1) == r1_state_sha
    assert compute_momentum_state_sha256(ctrl.momentum_state) == r1_mom_sha

    # Round 2: Byzantine NaN bee delta -> immediate ANSE Maximum Pain & bitwise rollback of weights + momentum
    nan_bees = [{"blocks.0.mlp.experts.2": {"w1_A": torch.full_like(base_blocks["blocks.0.mlp.experts.2"]["w1_A"], float("nan"))}}]
    ckpt_2, rec_2 = ctrl.step_outer_round(nan_bees)
    assert rec_2["accepted"] is False
    assert rec_2["rolled_back"] is True
    assert rec_2["rollback_bitwise_verified"] is True
    assert rec_2["anse_energy"] == pytest.approx(ANSE_MAXIMUM_PAIN)
    assert rec_2["final_state_sha256"] == r1_state_sha
    assert rec_2["final_momentum_sha256"] == r1_mom_sha
    assert compute_checkpoint_sha256(ckpt_2) == r1_state_sha
    assert compute_momentum_state_sha256(ctrl.momentum_state) == r1_mom_sha

    # Round 3: Finite adversarial delta with identity P_0 breaching D0 > +1.0% -> atomic rollback + override_directive
    id_p_null = {bid: torch.eye(fix["hidden_size"], dtype=torch.float64) for bid in base_blocks if ".experts." in bid}
    adv_bees = _make_forager_bee_round_deltas(
        base_blocks,
        fix["axes_in"],
        fix["axes_out"],
        hidden_size=fix["hidden_size"],
        block_h=fix["block_h"],
        spec_step_by_expert={2: -0.45, 7: -0.45},
        prose_drift_scale=3.2,
        seed=20264342,
    )
    override_dir = emit_queen_pheromone_campaign_directive(
        {f"expert_{e}": 0.0 for e in range(8)},
        {f"expert_{e}": 0.0 for e in range(8)},
        0.01,
        {"E0_general_fineweb": 2.0, "E2_math_reasoning": 2.5},
        {"E0_general_fineweb": 0.02, "E2_math_reasoning": 0.04},
        campaign_id="rollback-chain-s4",
        round_index=3,
    )
    ckpt_3, rec_3 = ctrl.step_outer_round(
        adv_bees,
        p_null_by_expert=id_p_null,
        override_directive=override_dir,
    )
    assert rec_3["accepted"] is False
    assert rec_3["rolled_back"] is True
    assert rec_3["rollback_bitwise_verified"] is True
    assert rec_3["directive_overridden"] is True
    assert rec_3["anse_energy"] == pytest.approx(ANSE_MAXIMUM_PAIN)
    assert rec_3["final_state_sha256"] == r1_state_sha
    assert rec_3["final_momentum_sha256"] == r1_mom_sha
    assert compute_checkpoint_sha256(ckpt_3) == r1_state_sha

    # Round 4: Clean recovery round resumes seamlessly from Round 1 state & momentum
    r4_bees = _make_forager_bee_round_deltas(
        base_blocks,
        fix["axes_in"],
        fix["axes_out"],
        hidden_size=fix["hidden_size"],
        block_h=fix["block_h"],
        spec_step_by_expert={2: 0.25},
        prose_drift_scale=0.80,
        seed=20264343,
    )
    _, rec_4 = ctrl.step_outer_round(r4_bees)
    assert rec_4["accepted"] is True
    assert rec_4["anse_energy"] < rec_1["anse_energy"] < 0.0

    chain_res = verify_inter_round_trophallaxis_chain(ctrl.receipts, campaign_id="rollback-chain-s4")
    assert chain_res["valid"] is True
    assert chain_res["entry_count"] == 4

    # Dict-mode checkpoint test (including reseed_corruptors_from_momentum=False and rollback)
    dict_base = {
        "blocks.0.mlp.experts.1": {"w1_A": torch.zeros((2, 8)), "w3_B": torch.zeros((8, 2))},
        "blocks.0.mlp.experts.3": {"w1_A": torch.ones((2, 8)), "w3_B": 0.3 * torch.ones((8, 2))},
        "blocks.0.mlp.router": {"weight": torch.zeros((2, 8))},
        "final_ln": {"weight": torch.ones(8)},
    }

    def _dict_eval(blks: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
        e1_val = float(blks["blocks.0.mlp.experts.1"]["w1_A"].abs().sum().item())
        e3_val = float(blks["blocks.0.mlp.experts.3"]["w1_A"].abs().sum().item())
        d0_drift = 0.0 if e3_val <= 8.5 else 0.05
        return {
            "nll_mean": 2.10 - 0.04 * min(e1_val, 8.0),
            "ece_value": 0.03 - 0.002 * min(e1_val, 8.0),
            "domain_nlls": {
                "E0_gen": 2.0 + d0_drift,
                "E1_code": 2.2 - 0.08 * min(e1_val, 8.0),
            },
            "domain_eces": {"E0_gen": 0.02, "E1_code": 0.04 - 0.004 * min(e1_val, 8.0)},
        }

    dict_ctrl = InterRoundRoyalJellyTrophallaxisController(
        dict_base,
        config=InterRoundTrophallaxisConfig(reseed_corruptors_from_momentum=False),
        eval_fn=_dict_eval,
        normed_unit_probe=torch.ones((4, 2, 8)),
        head_weight=torch.ones((6, 8)),
        probe_targets=torch.zeros((4, 2), dtype=torch.long),
        is_general_mask=[True, True, False, False],
        initial_directive=emit_queen_pheromone_campaign_directive(
            {"expert_1": -0.002, "expert_3": 0.02},
            {"expert_1": 0.0, "expert_3": 0.02},
            0.01,
            {"E0_gen": 2.0, "E1_code": 2.2},
            {"E0_gen": 0.02, "E1_code": 0.04},
            campaign_id="dict-mode-s4",
            round_index=1,
        ),
        initial_domain_round_counts={"E1_code": 0},
        campaign_id="dict-mode-s4",
    )
    out_dict_1, drec_1 = dict_ctrl.step_outer_round(
        [{"blocks.0.mlp.experts.1": {"w1_A": 0.6 * torch.ones((2, 8))}}],
        p_null_by_expert={"blocks.0.mlp.experts.1": torch.eye(8, dtype=torch.float64)},
    )
    assert drec_1["accepted"] is True
    assert float(out_dict_1["blocks.0.mlp.experts.1"]["w1_A"].abs().sum().item()) > 0.0
    # Second dict round pushes expert 3 past D0 drift -> rolls back
    _, drec_2 = dict_ctrl.step_outer_round(
        [{"blocks.0.mlp.experts.3": {"w1_A": 2.0 * torch.ones((2, 8))}}],
    )
    assert drec_2["rolled_back"] is True
    assert drec_2["final_state_sha256"] == drec_1["final_state_sha256"]

    # Tamper checks across every chain verification rule
    base_recs = list(ctrl.receipts)
    tampered_campaign = [dict(base_recs[0], campaign_id="other"), *base_recs[1:]]
    assert verify_inter_round_trophallaxis_chain(tampered_campaign, campaign_id="rollback-chain-s4")["valid"] is False

    tampered_prev = [base_recs[0], dict(base_recs[1], prev_receipt_sha256="f" * 64), *base_recs[2:]]
    assert verify_inter_round_trophallaxis_chain(tampered_prev)["valid"] is False

    tampered_hash = [dict(base_recs[0], nll_mean=0.123), *base_recs[1:]]
    assert verify_inter_round_trophallaxis_chain(tampered_hash)["valid"] is False

    tampered_state_link = [base_recs[0], dict(base_recs[1], pre_state_sha256="1" * 64), *base_recs[2:]]
    assert verify_inter_round_trophallaxis_chain(tampered_state_link)["valid"] is False

    tampered_mom_link = [base_recs[0], dict(base_recs[1], pre_momentum_sha256="2" * 64), *base_recs[2:]]
    assert verify_inter_round_trophallaxis_chain(tampered_mom_link)["valid"] is False

    tampered_dir_link = [base_recs[0], dict(base_recs[1], applied_directive_sha256="3" * 64), *base_recs[2:]]
    assert verify_inter_round_trophallaxis_chain(tampered_dir_link)["valid"] is False

    tampered_nonmono = [*base_recs[:3], dict(base_recs[3], anse_energy=0.0)]
    assert verify_inter_round_trophallaxis_chain(tampered_nonmono)["valid"] is False

    tampered_rb = [base_recs[0], dict(base_recs[1], final_state_sha256="4" * 64, anse_energy=0.0, rolled_back=False), *base_recs[2:]]
    assert verify_inter_round_trophallaxis_chain(tampered_rb)["valid"] is False

    with pytest.raises(ValueError, match="chain audit failed"):
        verify_inter_round_trophallaxis_chain(tampered_prev, raise_on_error=True)


# ---------------------------------------------------------------------------
# Scenario S5: Validation Gates & Security / Zero-Stub / OwnerOrder Compliance
# ---------------------------------------------------------------------------


def test_s5_validation_gates_and_security_stub_owner_guardrail_compliance() -> None:
    """(S5) Assert strict input validation, zero stubs, zero secrets/lateral movement, and OwnerOrder."""
    # 1. Config & helper validation errors
    with pytest.raises(ValueError, match="outer_lr"):
        InterRoundTrophallaxisConfig(outer_lr=0.0)
    with pytest.raises(ValueError, match="outer_momentum_beta"):
        InterRoundTrophallaxisConfig(outer_momentum_beta=1.0)
    with pytest.raises(ValueError, match="foraging_weight_floor"):
        InterRoundTrophallaxisConfig(foraging_weight_floor=1.5)
    with pytest.raises(TypeError, match="bool"):
        InterRoundTrophallaxisConfig(nesterov="yes")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="blocks must be a non-empty"):
        initialize_zero_momentum_state({})
    with pytest.raises(TypeError, match="Mapping"):
        initialize_zero_momentum_state({"e1": "bad"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="at least one expert or router"):
        initialize_zero_momentum_state({"final_ln": {"weight": torch.ones(4)}})

    with pytest.raises(ValueError, match="momentum_state must be a non-empty"):
        measure_momentum_prose_leakage({}, None)
    assert measure_momentum_prose_leakage({"e1": {"w": torch.ones(4)}}, None)["max_abs_leakage"] == 0.0

    with pytest.raises(ValueError, match="bee_ids must be non-empty"):
        resolve_pheromone_bee_weights([])
    with pytest.raises(TypeError, match="explicit_bee_weights"):
        resolve_pheromone_bee_weights(["b0"], explicit_bee_weights="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="length"):
        resolve_pheromone_bee_weights(["b0", "b1"], explicit_bee_weights=[1.0])
    with pytest.raises(ValueError, match="strictly positive sum"):
        resolve_pheromone_bee_weights(["b0"], explicit_bee_weights=[0.0])
    with pytest.raises(ValueError, match="bee_domains length"):
        resolve_pheromone_bee_weights(
            ["b0", "b1"],
            bee_domains=["E1"],
            directive={"bee_foraging_domain_weights": {"E1": 1.0}},
        )
    with pytest.raises(TypeError, match="bee_domains must be"):
        resolve_pheromone_bee_weights(
            ["b0"],
            bee_domains=123,  # type: ignore[arg-type]
            directive={"bee_foraging_domain_weights": {"E1": 1.0}},
        )

    valid_p = {"blocks.0.mlp.experts.1": {"w1_A": torch.ones((2, 4))}}
    valid_m = initialize_zero_momentum_state(valid_p)
    with pytest.raises(TypeError, match="config"):
        apply_inter_round_royal_jelly_momentum_step(valid_p, [{"blocks.0.mlp.experts.1": {"w1_A": torch.ones((2, 4))}}], valid_m, None, config="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-finite"):
        apply_inter_round_royal_jelly_momentum_step(valid_p, [{"blocks.0.mlp.experts.1": {"w1_A": torch.full((2, 4), float("nan"))}}], valid_m, None)
    with pytest.raises(ValueError, match="signature verification"):
        apply_inter_round_royal_jelly_momentum_step(valid_p, [{"blocks.0.mlp.experts.1": {"w1_A": torch.ones((2, 4))}}], valid_m, None, directive={"directive_sha256": "0" * 64})
    with pytest.raises(TypeError, match="directive"):
        apply_inter_round_royal_jelly_momentum_step(valid_p, [{"blocks.0.mlp.experts.1": {"w1_A": torch.ones((2, 4))}}], valid_m, None, directive="bad")  # type: ignore[arg-type]

    # 2. Controller & verifier validation errors
    with pytest.raises(TypeError, match="InterRoundTrophallaxisConfig"):
        InterRoundRoyalJellyTrophallaxisController(valid_p, config="bad", eval_fn=lambda _: {"E0": 1.0})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="eval_fn"):
        InterRoundRoyalJellyTrophallaxisController(valid_p, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="base_checkpoint"):
        InterRoundRoyalJellyTrophallaxisController("bad", eval_fn=lambda _: {"E0": 1.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires eval_fn"):
        InterRoundRoyalJellyTrophallaxisController(valid_p)
    with pytest.raises(TypeError, match="initial_domain_round_counts"):
        InterRoundRoyalJellyTrophallaxisController(
            valid_p,
            eval_fn=lambda _: {"E0_gen": 1.0},
            initial_domain_round_counts="bad",  # type: ignore[arg-type]
        )

    fix_v = _build_la_ruche_8expert_fixture(seed=20264355)
    m_v: CrowdV1WithEngram = fix_v["model"]
    with pytest.raises(ValueError, match="probe_chunks/eval_chunks"):
        InterRoundRoyalJellyTrophallaxisController(m_v)
    ctrl_v = InterRoundRoyalJellyTrophallaxisController(m_v, probe_chunks=fix_v["probe_chunks"])
    with pytest.raises(ValueError, match="non-empty sequence"):
        ctrl_v.run_campaign([])
    with pytest.raises(ValueError, match="forager_bee_deltas"):
        ctrl_v.run_campaign([{}])
    del m_v.head
    with pytest.raises(ValueError, match="expose 'head'"):
        InterRoundRoyalJellyTrophallaxisController(m_v, probe_chunks=fix_v["probe_chunks"])

    with pytest.raises(TypeError, match="Mapping"):
        verify_inter_round_trophallaxis_receipt_hash("bad")  # type: ignore[arg-type]
    assert verify_inter_round_trophallaxis_receipt_hash({"receipt_sha256": "bad"}) is False
    assert verify_inter_round_trophallaxis_receipt_hash({"object": ROUND_RECEIPT_OBJECT, "schema": INTER_ROUND_TROPHALLAXIS_SCHEMA, "receipt_sha256": "0" * 64}) is False
    assert verify_inter_round_trophallaxis_receipt_hash({"object": ROUND_RECEIPT_OBJECT, "schema": INTER_ROUND_TROPHALLAXIS_SCHEMA, "receipt_sha256": "a" * 64, "bad": float("nan")}) is False
    with pytest.raises(TypeError, match="Sequence"):
        verify_inter_round_trophallaxis_chain("bad")  # type: ignore[arg-type]
    assert verify_inter_round_trophallaxis_chain(["bad"])["valid"] is False  # type: ignore[list-item]

    # 3. AGPL-3.0 header, ZeroStubAudit, PRSecurityScanner, lateral movement, FORBIDDEN_ADDED, OwnerOrderGuardrail
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn43", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn43", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/inter_round_royal_jelly_trophallaxis.py", TARGET_MODULE_PATH),
        ("tests/test_inter_round_royal_jelly_trophallaxis_scenarios.py", TEST_FILE_PATH),
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

    pr_title = "feat(swarm): closed-loop inter-round Royal-Jelly DiLoCo trophallaxis and pheromone propagation controller"
    pr_body = (
        "Implements NEURAL-CTRL-DYN-43 Gen-3 Direction D1 inter-round Royal-Jelly DiLoCo "
        "trophallaxis, pre-momentum prose-nullspace invariance, QueenPheromoneDirective propagation, "
        "atomic bitwise rollback, and SHA-256 chain receipts under ANSE Delta E < 0."
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
        pr_number=1243,
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
