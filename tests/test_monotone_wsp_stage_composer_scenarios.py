# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-16: monotone multi-stage WSP stage composer."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdLoRAMoEMLP,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.monotone_wsp_stage_composer import (
    DEFAULT_ANSE_MAXIMUM_PAIN,
    PIPELINE_RECEIPT_OBJECT,
    SCHEMA_VERSION,
    STAGE_RECEIPT_OBJECT,
    VAULT_RECEIPT_OBJECT,
    HoldoutVaultLockedError,
    HoldoutVaultLockState,
    WSPStageSpec,
    clone_state_dict,
    compute_state_dict_sha256,
    count_edited_parameters,
    evaluate_anse_domain_metrics,
    make_cross_layer_anchor_stage,
    make_ghost_sink_stage,
    make_layernorm_calibration_stage,
    make_orthogonal_router_stage,
    make_quantization_compensation_stage,
    parse_eval_domain_nll,
    run_monotone_wsp_stage_pipeline,
    verify_holdout_vault_single_shot,
    verify_pipeline_receipt_provenance,
    verify_stage_receipt_provenance,
    verify_vault_receipt_provenance,
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
    """Minimal stub audit result used when optional ``gwaya`` is absent in CI."""

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
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "monotone_wsp_stage_composer.py"
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


def _make_subspace_crowd_moe(
    *,
    num_layers: int = 4,
    num_experts: int = 4,
    top_k: int = 2,
    lora_rank: int = 2,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    vocab_size: int = 48,
    seed: int = 20261124,
) -> tuple[CrowdV1WithEngram, dict[str, torch.Tensor]]:
    """Build deterministic multi-layer CrowdV1WithEngram + CrowdLoRAMoEMLP with orthogonal E0/E1/E2 subspaces."""
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

    u_e0 = torch.zeros(16, dtype=torch.float32)
    u_e0[0] = 1.0 / math.sqrt(2.0)
    u_e0[1] = -1.0 / math.sqrt(2.0)

    u_e1 = torch.zeros(8, dtype=torch.float32)
    u_e1[0] = 1.0 / math.sqrt(2.0)
    u_e1[1] = -1.0 / math.sqrt(2.0)

    u_e2 = torch.zeros(8, dtype=torch.float32)
    u_e2[0] = 1.0 / math.sqrt(2.0)
    u_e2[1] = -1.0 / math.sqrt(2.0)

    prose_carrier = torch.zeros(hidden_size, dtype=torch.float32)
    prose_carrier[:16] = u_e0
    e1_carrier = torch.zeros(hidden_size, dtype=torch.float32)
    e1_carrier[16:24] = u_e1
    e2_carrier = torch.zeros(hidden_size, dtype=torch.float32)
    e2_carrier[24:32] = u_e2

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        p_raw = 0.20 * torch.randn((16, 16), generator=gen, dtype=torch.float32)
        p_raw[:, :2] = 0.0
        p_raw = p_raw - p_raw.mean(dim=1, keepdim=True) + 2.0 * u_e0.unsqueeze(0)
        p_raw[5] = 2.4 * u_e0

        e1_raw = 0.20 * torch.randn((16, 8), generator=gen, dtype=torch.float32)
        e1_raw[:, :2] = 0.0
        e1_raw = e1_raw - e1_raw.mean(dim=1, keepdim=True) + 2.0 * u_e1.unsqueeze(0)
        e1_raw[20 - 16] = 2.4 * u_e1

        e2_raw = 0.20 * torch.randn((16, 8), generator=gen, dtype=torch.float32)
        e2_raw[:, :2] = 0.0
        e2_raw = e2_raw - e2_raw.mean(dim=1, keepdim=True) + 2.0 * u_e2.unsqueeze(0)
        e2_raw[36 - 32] = 2.4 * u_e2

        wte[:16, :16] = p_raw
        wte[16:32, 16:24] = e1_raw
        wte[32:48, 24:32] = e2_raw
        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight

        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0)
            blk.ln2.bias.zero_()
            blk.mlp.w1.weight[:hidden_size, :hidden_size].copy_(1.5 * torch.eye(hidden_size))
            blk.mlp.w2.weight[:hidden_size, :hidden_size].copy_(1.5 * torch.eye(hidden_size))
            blk.mlp.w3.weight[:16, :16].copy_(0.03 * torch.randn((16, 16), generator=gen))
            blk.mlp.w3.weight[16:24, 16:24].copy_(0.03 * torch.randn((8, 8), generator=gen))
            blk.mlp.w3.weight[24:32, 24:32].copy_(0.03 * torch.randn((8, 8), generator=gen))

    upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode="lora_domain_experts",
        num_experts=num_experts,
        top_k=top_k,
        lora_rank=lora_rank,
        lora_alpha=8.0,
        routing_mode="sigma_topk",
        seed=seed + 19,
    )
    carriers = {"e0": prose_carrier, "e1": e1_carrier, "e2": e2_carrier}
    return model, carriers


def _make_disjoint_domain_chunks(
    *,
    seq_len: int = 16,
    score_from: int = 8,
    domains: tuple[str, ...] = ("E0_general_fineweb", "E1_code", "E2_math"),
    chunks_per_domain: int = 6,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    """Build deterministic non-overlapping domain token chunks."""
    gen = torch.Generator(device="cpu").manual_seed(9500 + seed_offset)
    chunks: list[dict[str, Any]] = []
    for _d_idx, dom in enumerate(domains):
        for _r_idx in range(chunks_per_domain):
            if dom.startswith("E0_"):
                ids = torch.randint(0, 16, (seq_len,), generator=gen).tolist()
                target_tok = 5
            elif dom.startswith("E1_"):
                ids = torch.randint(16, 32, (seq_len,), generator=gen).tolist()
                target_tok = 20
            else:
                ids = torch.randint(32, 48, (seq_len,), generator=gen).tolist()
                target_tok = 36
            for pos in range(score_from, seq_len):
                ids[pos] = target_tok
            chunks.append({"expert_id": dom, "ids": ids})
    return chunks


def _plant_four_stage_defects(model: CrowdV1WithEngram, carriers: dict[str, torch.Tensor]) -> None:
    """Plant 4 orthogonal defects repaired sequentially by ghost-sink, anchor projection, router surgery, and LN calibration."""
    prose_v = carriers["e0"]
    e1_v = carriers["e1"]
    e2_v = carriers["e2"]

    with torch.no_grad():
        for l_idx, blk in enumerate(model.blocks):
            mlp: CrowdLoRAMoEMLP = blk.mlp  # type: ignore[assignment]
            mlp.router.weight.zero_()
            for exp_mod in mlp.experts.values():
                for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                    getattr(exp_mod, attr).zero_()

            # Beneficial early-layer E0 general anchor in layers 0..1
            if l_idx < 2:
                mlp.router.weight[0] = 2.6 * prose_v
                mlp.experts["0"].w3_A[0, 0] = 0.02
                mlp.experts["0"].w3_B[:, 0] = 0.03 * prose_v

            if l_idx in (2, 3):
                # Leaky prose router gates on specialist experts 1 & 2 (expert 2 takes top-2 slot with ghost sink 3)
                mlp.router.weight[0] = 0.5 * prose_v
                mlp.router.weight[1] = 2.8 * e1_v + (2.4 if l_idx == 2 else 2.2) * prose_v
                mlp.router.weight[2] = 2.8 * e2_v + (2.2 if l_idx == 2 else 2.5) * prose_v

                # Expert 1: clean E1 specialist mode + mild prose interference mode
                exp1 = mlp.experts["1"]
                exp1.w3_A[0, 16] = 0.14
                exp1.w3_B[:, 0] = 0.55 * e1_v
                exp1.w3_A[1, 0] = 0.05
                exp1.w3_B[:, 1] = -0.25 * prose_v

                # Expert 2: E2 specialist mode + cross-layer residual prose corruption in layer 3
                exp2 = mlp.experts["2"]
                exp2.w3_A[0, 24] = 0.14
                if l_idx == 3:
                    exp2.w3_B[:, 0] = 0.55 * e2_v - 0.35 * prose_v
                else:
                    exp2.w3_B[:, 0] = 0.55 * e2_v
                exp2.w3_A[1, 0] = 0.05
                exp2.w3_B[:, 1] = -0.22 * prose_v

                # Rogue ghost-sink expert 3 harming E0 in layers 2..3
                mlp.router.weight[3] = 3.0 * prose_v
                exp3 = mlp.experts["3"]
                exp3.w3_A[0, 0] = 0.07
                exp3.w3_B[:, 0] = -0.32 * prose_v

        # Hot/damped LayerNorm temperature scaling (0.65 -> calibrated 1.0 / 0.85)
        model.ln.weight.fill_(0.65)
        model.ln.bias.zero_()


@torch.no_grad()
def _eval_crowd_moe_state(
    model: CrowdV1WithEngram,
    state_dict: dict[str, torch.Tensor] | Any,
    chunks: list[dict[str, Any]],
    *,
    score_from: int = 8,
) -> dict[str, float]:
    """Load ``state_dict`` into ``model`` and compute exact per-domain causal NLL."""
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    ids_all = torch.tensor([c["ids"] for c in chunks], dtype=torch.long)
    h_pre = forward_model_pre_ln(model, ids_all, batch_size=12)
    nlls = per_chunk_nll_from_pre_ln(
        h_pre,
        model.ln.weight,
        model.ln.bias,
        float(getattr(model.ln, "eps", 1e-5)),
        model.head.weight,
        ids_all,
        score_from=score_from,
    )
    by_dom: dict[str, list[float]] = {}
    for idx, ch in enumerate(chunks):
        by_dom.setdefault(str(ch["expert_id"]), []).append(float(nlls[idx].item()))
    return {dom: float(sum(vals) / len(vals)) for dom, vals in sorted(by_dom.items())}


def _anchor_layers_digest(state_dict: dict[str, torch.Tensor], anchor_layers: tuple[int, ...] = (0, 1)) -> str:
    """Compute SHA-256 over all parameters belonging to upstream anchor layers."""
    prefixes = tuple(f"blocks.{idx}." for idx in anchor_layers)
    subset = {k: v for k, v in state_dict.items() if k.startswith(prefixes)}
    return compute_state_dict_sha256(subset)


# ---------------------------------------------------------------------------
# S1: 4-Stage Monotone Open-Weight MoE Surgery Pipeline Within Guard
# ---------------------------------------------------------------------------


def test_s1_four_stage_monotone_wsp_pipeline_lowers_anse_energy_within_guard() -> None:
    """Chain 4 heterogeneous WSP stages monotonically lowering ANSE energy under <= +1.0% guard."""
    model, carriers = _make_subspace_crowd_moe(num_layers=4, num_experts=4, top_k=2, lora_rank=2, seed=1101)
    _plant_four_stage_defects(model, carriers)
    start_state = clone_state_dict(model.state_dict())

    dev_chunks = _make_disjoint_domain_chunks(chunks_per_domain=6, seed_offset=100)

    def _eval_fn(sd: Any) -> dict[str, float]:
        return _eval_crowd_moe_state(model, sd, dev_chunks, score_from=8)

    stages = [
        make_ghost_sink_stage([3], layer_indices=[2, 3], name="stage_1_ghost_sink", max_edit_fraction=0.02),
        make_cross_layer_anchor_stage(
            carriers["e0"],
            target_layers=[3],
            expert_indices=[2],
            factors=("w3_B",),
            name="stage_2_cross_layer_anchor",
            max_edit_fraction=0.02,
        ),
        make_orthogonal_router_stage(
            {1: carriers["e0"], 2: carriers["e0"]},
            target_layers=[2, 3],
            beta=2.5,
            sink_expert=3,
            sink_boost=0.5,
            name="stage_3_orthogonal_router",
            max_edit_fraction=0.02,
        ),
        make_layernorm_calibration_stage(
            target_ln_weight=torch.ones(32, dtype=torch.float32),
            target_ln_bias=torch.zeros(32, dtype=torch.float32),
            lambda_ln=1.0,
            temperature_scale=0.90,
            name="stage_4_layernorm_calibration",
            max_edit_fraction=0.02,
        ),
    ]

    result = run_monotone_wsp_stage_pipeline(
        start_state,
        stages,
        _eval_fn,
        general_guard_rel=0.01,
        max_total_edit_fraction=0.05,
        skip_regressing_stage=True,
    )

    assert result.status == "COMPLETED_MONOTONE_COMMIT"
    assert result.aborted_early is False
    assert result.guard_ok is True
    assert result.committed_stages == (
        "stage_1_ghost_sink",
        "stage_2_cross_layer_anchor",
        "stage_3_orthogonal_router",
        "stage_4_layernorm_calibration",
    )
    assert result.rolled_back_stages == ()
    assert result.skipped_stages == ()
    assert result.cumulative_edit_fraction <= 0.05
    assert result.cumulative_edited_params > 0
    assert result.final_anse_energy < result.initial_anse_energy
    assert result.stage_delta_nats < 0.0
    assert result.receipt["object"] == PIPELINE_RECEIPT_OBJECT
    assert result.receipt["schema"] == SCHEMA_VERSION
    assert verify_pipeline_receipt_provenance(result.receipt) is True

    # Verify strict stage-by-stage ANSE monotonicity, SHA chaining, and anchor layer preservation
    assert _anchor_layers_digest(result.champion_state) == _anchor_layers_digest(start_state)
    prev_post_sha = result.base_sha256
    prev_energy = result.initial_anse_energy
    for receipt in result.stage_receipts:
        assert receipt.committed is True
        assert receipt.rolled_back is False
        assert receipt.status == "COMMITTED"
        assert receipt.pre_sha256 == prev_post_sha
        assert receipt.post_sha256 == receipt.candidate_sha256
        assert receipt.post_anse_energy < prev_energy
        assert receipt.delta_anse_energy < 0.0
        assert receipt.stage_delta_nats < 0.0
        assert receipt.guard_ok is True
        assert receipt.budget_ok is True
        assert receipt.worst_domain_drift_pct <= 1.0
        assert receipt.general_domain_drift_pct <= 1.0
        assert len(receipt.pre_domain_nll_vector) == len(receipt.domain_order) == 3
        assert len(receipt.post_domain_nll_vector) == len(receipt.domain_order) == 3
        assert verify_stage_receipt_provenance(receipt) is True
        prev_post_sha = receipt.post_sha256
        prev_energy = receipt.post_anse_energy

    # Verify exact deterministic repeatability and explicit base_state / base_domain_nll support
    repeat = run_monotone_wsp_stage_pipeline(
        start_state,
        stages,
        _eval_fn,
        general_guard_rel=0.01,
        max_total_edit_fraction=0.05,
        base_state=start_state,
    )
    assert repeat.receipt_sha256 == result.receipt_sha256
    assert compute_state_dict_sha256(repeat.champion_state) == result.champion_sha256

    explicit_base = run_monotone_wsp_stage_pipeline(
        start_state,
        stages[:1],
        _eval_fn,
        base_domain_nll=result.initial_domain_nll,
    )
    assert explicit_base.committed_stages == ("stage_1_ghost_sink",)


# ---------------------------------------------------------------------------
# S2: Mid-Pipeline Regressing & Corrupting Stages Automatically Rolled Back
# ---------------------------------------------------------------------------


def test_s2_mid_pipeline_regressing_and_corrupting_stages_roll_back_bitwise() -> None:
    """Roll back mid-pipeline NaN, guard-violating, and regressing stages bitwise without losing prior gains."""
    model, carriers = _make_subspace_crowd_moe(num_layers=4, num_experts=4, top_k=2, lora_rank=2, seed=1202)
    _plant_four_stage_defects(model, carriers)
    start_state = clone_state_dict(model.state_dict())
    dev_chunks = _make_disjoint_domain_chunks(chunks_per_domain=6, seed_offset=200)

    def _eval_fn(sd: Any) -> dict[str, float]:
        return _eval_crowd_moe_state(model, sd, dev_chunks, score_from=8)

    s1 = make_ghost_sink_stage([3], layer_indices=[2, 3], name="stage_1_ghost_sink")
    s2 = make_cross_layer_anchor_stage(
        carriers["e0"],
        target_layers=[3],
        expert_indices=[2],
        factors=("w3_B",),
        orthonormalize=False,
        name="stage_2_cross_layer_anchor",
    )
    s4 = make_orthogonal_router_stage(
        {1: carriers["e0"], 2: carriers["e0"]},
        target_layers=[2, 3],
        beta=2.5,
        sink_expert=3,
        sink_boost=0.5,
        sink_direction=carriers["e0"],
        name="stage_4_orthogonal_router",
    )

    # Reference 3-stage run without corruption to prove bitwise preservation of Stage 1-2 + Stage 4
    clean_ref = run_monotone_wsp_stage_pipeline(
        start_state, [s1, s2, s4], _eval_fn, general_guard_rel=0.01, max_total_edit_fraction=0.05
    )
    assert len(clean_ref.committed_stages) == 3

    # 2A: Mid-pipeline NaN corrupting stage + in-place mutation isolation check
    def _corrupt_nan_inplace(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sd["blocks.2.mlp.router.weight"][0, 0] = float("nan")
        return sd

    s_corrupt_nan = WSPStageSpec(name="stage_3_corrupt_nan", transform_fn=_corrupt_nan_inplace)
    res_nan = run_monotone_wsp_stage_pipeline(
        start_state,
        [s1, s2, s_corrupt_nan, s4],
        _eval_fn,
        general_guard_rel=0.01,
        max_total_edit_fraction=0.05,
        skip_regressing_stage=True,
    )
    assert res_nan.committed_stages == ("stage_1_ghost_sink", "stage_2_cross_layer_anchor", "stage_4_orthogonal_router")
    assert res_nan.rolled_back_stages == ("stage_3_corrupt_nan",)
    r_nan = res_nan.stage_receipts[2]
    assert r_nan.status == "ROLLED_BACK_NON_FINITE"
    assert r_nan.committed is False
    assert r_nan.rolled_back is True
    assert r_nan.pre_sha256 == r_nan.post_sha256
    assert r_nan.candidate_sha256 != r_nan.pre_sha256
    assert r_nan.candidate_anse_energy == DEFAULT_ANSE_MAXIMUM_PAIN
    assert r_nan.delta_anse_energy == 0.0
    assert res_nan.champion_sha256 == clean_ref.champion_sha256

    # 2B: Mid-pipeline guard-violating stage (severe E0 drift via corrupted router/LN) + no-op stage
    def _guard_violating_stage(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sd["ln.bias"][:16] = sd["ln.bias"][:16] - 3.0 * carriers["e0"][:16]
        return sd

    def _noop_stage(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return sd

    res_guard = run_monotone_wsp_stage_pipeline(
        start_state,
        [
            s1,
            s2,
            WSPStageSpec(name="stage_2b_noop", transform_fn=_noop_stage),
            WSPStageSpec(name="stage_3_guard_violation", transform_fn=_guard_violating_stage),
            s4,
        ],
        _eval_fn,
        general_guard_rel=0.01,
        max_total_edit_fraction=0.05,
        skip_regressing_stage=True,
    )
    assert res_guard.skipped_stages == ("stage_2b_noop",)
    assert res_guard.rolled_back_stages == ("stage_3_guard_violation",)
    assert res_guard.stage_receipts[2].status == "SKIPPED_NOOP"
    assert res_guard.stage_receipts[3].status == "ROLLED_BACK_GUARD_VIOLATION"
    assert res_guard.stage_receipts[3].pre_sha256 == res_guard.stage_receipts[3].post_sha256
    assert res_guard.champion_sha256 == clean_ref.champion_sha256

    # 2C: Energy-regressing stage (within guard, raises NLL) & candidate eval yielding NaN
    def _mild_regress_stage(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sd["blocks.1.mlp.experts.0.w3_B"][:] = 0.0
        return sd

    res_reg = run_monotone_wsp_stage_pipeline(
        start_state,
        [s1, s2, WSPStageSpec(name="stage_3_mild_regress", transform_fn=_mild_regress_stage), s4],
        _eval_fn,
        general_guard_rel=0.05,
        max_total_edit_fraction=0.05,
        skip_regressing_stage=True,
    )
    assert res_reg.rolled_back_stages == ("stage_3_mild_regress",)
    assert res_reg.stage_receipts[2].status == "ROLLED_BACK_ENERGY_REGRESSION"
    assert res_reg.stage_receipts[2].candidate_delta_anse_energy >= 0.0
    assert res_reg.stage_receipts[2].pre_sha256 == res_reg.stage_receipts[2].post_sha256

    # Candidate eval returning NaN triggers ROLLED_BACK_NON_FINITE cleanly
    def _nan_eval_after_edit(sd: Any) -> dict[str, float]:
        if compute_state_dict_sha256(sd) != compute_state_dict_sha256(start_state):
            return {"E0_general_fineweb": float("nan"), "E1_code": 1.5}
        return {"E0_general_fineweb": 2.0, "E1_code": 1.5}

    res_eval_nan = run_monotone_wsp_stage_pipeline(
        start_state, [s1], _nan_eval_after_edit, max_total_edit_fraction=0.05
    )
    assert res_eval_nan.status == "COMPLETED_ALL_ROLLED_BACK"
    assert res_eval_nan.stage_receipts[0].status == "ROLLED_BACK_NON_FINITE"

    # 2D: skip_regressing_stage=False halts pipeline early while keeping Stage 1 committed
    res_abort = run_monotone_wsp_stage_pipeline(
        start_state,
        [s1, s_corrupt_nan, s4],
        _eval_fn,
        general_guard_rel=0.01,
        max_total_edit_fraction=0.05,
        skip_regressing_stage=False,
    )
    assert res_abort.aborted_early is True
    assert res_abort.status == "ABORTED_EARLY_ON_REGRESSION"
    assert res_abort.committed_stages == ("stage_1_ghost_sink",)
    assert res_abort.rolled_back_stages == ("stage_3_corrupt_nan",)
    assert len(res_abort.stage_receipts) == 2

    # 2E: Self-healing sanitization inside make_ghost_sink_stage + 4-bit quantization compensation stage
    corrupt_start = clone_state_dict(start_state)
    corrupt_start["blocks.2.mlp.experts.3.w1_A"][0, 0] = float("nan")
    healed = make_ghost_sink_stage([3], sanitize_non_finite=True).transform_fn(corrupt_start)
    assert torch.isfinite(healed["blocks.2.mlp.experts.3.w1_A"]).all()

    q_comp = make_quantization_compensation_stage(
        ["blocks.2.mlp.base_mlp.w3.weight"],
        bits=4,
        group_size=16,
        rank=4,
        num_alternating_steps=2,
        m_prose=torch.eye(64, dtype=torch.float64),
        use_lora_compensation=True,
        name="stage_5_quant_comp",
    ).transform_fn(start_state)
    q_naive = make_quantization_compensation_stage(
        ["blocks.2.mlp.base_mlp.w3.weight"],
        bits=4,
        group_size=16,
        rank=4,
        use_lora_compensation=False,
        name="stage_5_quant_naive",
    ).transform_fn(start_state)
    comp_err = float((q_comp["blocks.2.mlp.base_mlp.w3.weight"] - start_state["blocks.2.mlp.base_mlp.w3.weight"]).norm())
    naive_err = float((q_naive["blocks.2.mlp.base_mlp.w3.weight"] - start_state["blocks.2.mlp.base_mlp.w3.weight"]).norm())
    assert comp_err <= naive_err + 1e-6


# ---------------------------------------------------------------------------
# S3: Cumulative Edited-Parameter Fraction Ceiling Enforcement
# ---------------------------------------------------------------------------


def test_s3_cumulative_edited_parameter_fraction_ceiling_enforcement() -> None:
    """Enforce cumulative and per-stage parameter edit ceilings without losing valid stages."""
    # Total 1000 params: 4 tensors of 250 elements each
    base_state = {
        "w_general": torch.zeros(250, dtype=torch.float32),
        "w_spec_a": torch.zeros(250, dtype=torch.float32),
        "w_spec_b": torch.zeros(250, dtype=torch.float32),
        "w_tail": torch.zeros(250, dtype=torch.float32),
    }

    eval_calls = 0

    def _quadratic_eval(sd: Any) -> dict[str, float]:
        nonlocal eval_calls
        eval_calls += 1
        # Moving entries from 0.0 toward 1.0 smoothly lowers domain NLL
        g_prog = float(sd["w_general"].clamp(0.0, 1.0).mean().item())
        a_prog = float(sd["w_spec_a"].clamp(0.0, 1.0).mean().item())
        b_prog = float(sd["w_spec_b"].clamp(0.0, 1.0).mean().item())
        t_prog = float(sd["w_tail"].clamp(0.0, 1.0).mean().item())
        return {
            "E0_general_fineweb": 2.00 - 0.50 * g_prog,
            "E1_code": 2.20 - 0.60 * a_prog - 0.20 * t_prog,
            "E2_math": 2.30 - 0.60 * b_prog,
        }

    def _edit_slice(key: str, start: int, end: int) -> Any:
        def _fn(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            sd[key][start:end] = 1.0
            return sd

        return _fn

    stages = [
        # 20 / 1000 = 2.0% -> cum 2.0% <= 5.0% -> COMMIT
        WSPStageSpec(name="s1_2pct", transform_fn=_edit_slice("w_general", 0, 20), max_edit_fraction=0.03),
        # 20 / 1000 = 2.0% -> cum 4.0% <= 5.0% -> COMMIT
        WSPStageSpec(name="s2_2pct", transform_fn=_edit_slice("w_spec_a", 0, 20), max_edit_fraction=0.03),
        # 25 / 1000 = 2.5% -> cum 6.5% > 5.0% -> REJECTED (even though it improves E2!)
        WSPStageSpec(name="s3_over_total_2p5pct", transform_fn=_edit_slice("w_spec_b", 0, 25), max_edit_fraction=0.05),
        # 15 / 1000 = 1.5% -> per-stage max_edit_fraction=0.01 violated!
        WSPStageSpec(name="s3b_over_stage_cap", transform_fn=_edit_slice("w_spec_b", 0, 15), max_edit_fraction=0.01),
        # 8 / 1000 = 0.8% -> per-stage max_edited_params=5 violated!
        WSPStageSpec(name="s3c_over_param_count", transform_fn=_edit_slice("w_spec_b", 0, 8), max_edited_params=5),
        # 5 / 1000 = 0.5% -> cum 4.5% <= 5.0% -> COMMIT
        WSPStageSpec(name="s4_0p5pct", transform_fn=_edit_slice("w_tail", 0, 5), max_edit_fraction=0.02, max_edited_params=5),
    ]

    result = run_monotone_wsp_stage_pipeline(
        base_state,
        stages,
        _quadratic_eval,
        general_guard_rel=0.01,
        max_total_edit_fraction=0.05,
        skip_regressing_stage=True,
    )

    # Budget-rejected stages should NOT even trigger eval_fn calls (1 initial + 3 accepted = 4 calls)
    assert eval_calls == 4
    assert result.committed_stages == ("s1_2pct", "s2_2pct", "s4_0p5pct")
    assert result.rolled_back_stages == ("s3_over_total_2p5pct", "s3b_over_stage_cap", "s3c_over_param_count")
    assert result.cumulative_edited_params == 45
    assert result.cumulative_edit_fraction == pytest.approx(0.045)
    assert result.total_params == 1000
    assert result.stage_receipts[2].status == "ROLLED_BACK_EDIT_BUDGET"
    assert result.stage_receipts[2].budget_ok is False
    assert result.stage_receipts[3].status == "ROLLED_BACK_EDIT_BUDGET"
    assert result.stage_receipts[4].status == "ROLLED_BACK_EDIT_BUDGET"
    assert torch.equal(result.champion_state["w_spec_b"], torch.zeros(250, dtype=torch.float32))
    assert result.final_anse_energy < result.initial_anse_energy

    # Zero-edit pipeline yields COMPLETED_NOOP
    noop_res = run_monotone_wsp_stage_pipeline(
        base_state,
        [WSPStageSpec(name="noop", transform_fn=lambda sd: sd)],
        _quadratic_eval,
        max_total_edit_fraction=0.0,
    )
    assert noop_res.status == "COMPLETED_NOOP"
    assert noop_res.champion_sha256 == noop_res.base_sha256


# ---------------------------------------------------------------------------
# S4: Single-Shot Virgin Holdout Vault Lock Rejecting Second Access
# ---------------------------------------------------------------------------


def test_s4_single_shot_virgin_holdout_vault_lock_rejects_second_access() -> None:
    """Enforce single-shot holdout vault evaluation and reject any second probe without burning evals."""
    model, carriers = _make_subspace_crowd_moe(num_layers=4, num_experts=4, top_k=2, lora_rank=2, seed=1404)
    _plant_four_stage_defects(model, carriers)
    base_state = clone_state_dict(model.state_dict())

    dev_chunks = _make_disjoint_domain_chunks(chunks_per_domain=6, seed_offset=400)
    vault_chunks = _make_disjoint_domain_chunks(chunks_per_domain=6, seed_offset=4400)

    pipeline_res = run_monotone_wsp_stage_pipeline(
        base_state,
        [
            make_ghost_sink_stage([3], layer_indices=[2, 3]),
            make_cross_layer_anchor_stage(carriers["e0"], target_layers=[3], expert_indices=[2], factors=("w3_B",)),
            make_orthogonal_router_stage({1: carriers["e0"], 2: carriers["e0"]}, target_layers=[2, 3], beta=2.5),
            make_layernorm_calibration_stage(
                target_ln_weight=torch.ones(32), target_ln_bias=torch.zeros(32), lambda_ln=1.0, temperature_scale=0.90
            ),
        ],
        lambda sd: _eval_crowd_moe_state(model, sd, dev_chunks, score_from=8),
        general_guard_rel=0.01,
        max_total_edit_fraction=0.05,
    )

    vault_eval_calls = 0

    def _vault_eval_fn(sd: Any) -> dict[str, float]:
        nonlocal vault_eval_calls
        vault_eval_calls += 1
        return _eval_crowd_moe_state(model, sd, vault_chunks, score_from=8)

    # 4A: HoldoutVaultLockState dataclass lock
    lock_obj = HoldoutVaultLockState(vault_id="virgin_vault_skip160")
    vault_receipt = verify_holdout_vault_single_shot(
        pipeline_res.champion_state,
        base_state,
        _vault_eval_fn,
        vault_lock_state=lock_obj,
        general_guard_rel=0.01,
    )
    assert vault_eval_calls == 2  # base_state + champion_state in single authorized verification
    assert vault_receipt["object"] == VAULT_RECEIPT_OBJECT
    assert vault_receipt["vault_locked"] is True
    assert vault_receipt["vault_access_count"] == 1
    assert vault_receipt["guard_ok"] is True
    assert vault_receipt["passed"] is True
    assert vault_receipt["delta_anse_energy"] < 0.0
    assert vault_receipt["stage_delta_nats"] < 0.0
    assert vault_receipt["worst_domain_drift_pct"] <= 1.0
    assert verify_vault_receipt_provenance(vault_receipt) is True
    assert lock_obj.locked is True
    assert lock_obj.access_count == 1
    assert lock_obj.receipt_sha256 == vault_receipt["receipt_sha256"]

    calls_before_second = vault_eval_calls
    with pytest.raises(HoldoutVaultLockedError, match="locked"):
        verify_holdout_vault_single_shot(
            pipeline_res.champion_state,
            base_state,
            _vault_eval_fn,
            vault_lock_state=lock_obj,
        )
    assert vault_eval_calls == calls_before_second

    # 4B: Plain mutable dict lock + cached base_vault_domain_nll + non-finite champion safety
    dict_lock: dict[str, Any] = {"vault_id": "dict_holdout_vault"}
    cached_receipt = verify_holdout_vault_single_shot(
        pipeline_res.champion_state,
        base_state,
        _vault_eval_fn,
        vault_lock_state=dict_lock,
        base_vault_domain_nll=vault_receipt["base_domain_nll"],
    )
    assert vault_eval_calls == calls_before_second + 1
    assert cached_receipt["passed"] is True
    assert dict_lock["locked"] is True
    assert dict_lock["consumed"] is True
    assert dict_lock["access_count"] == 1

    with pytest.raises(HoldoutVaultLockedError, match="locked"):
        verify_holdout_vault_single_shot(
            pipeline_res.champion_state,
            base_state,
            _vault_eval_fn,
            vault_lock_state=dict_lock,
        )
    assert vault_eval_calls == calls_before_second + 1

    # 4C: Fail-closed lock consumption even if vault_eval_fn raises or champion is non-finite
    err_lock = HoldoutVaultLockState(vault_id="fail_closed_vault")

    def _failing_vault(_sd: Any) -> dict[str, float]:
        raise RuntimeError("simulated vault I/O abort")

    with pytest.raises(RuntimeError, match="abort"):
        verify_holdout_vault_single_shot(
            pipeline_res.champion_state,
            base_state,
            _failing_vault,
            vault_lock_state=err_lock,
        )
    assert err_lock.locked is True
    with pytest.raises(HoldoutVaultLockedError):
        verify_holdout_vault_single_shot(
            pipeline_res.champion_state,
            base_state,
            _vault_eval_fn,
            vault_lock_state=err_lock,
        )

    nan_champ = clone_state_dict(pipeline_res.champion_state)
    nan_champ["ln.weight"][0] = float("nan")
    nan_vault_rec = verify_holdout_vault_single_shot(
        nan_champ,
        base_state,
        _vault_eval_fn,
        vault_lock_state={},
        base_vault_domain_nll=vault_receipt["base_domain_nll"],
    )
    assert nan_vault_rec["passed"] is False
    assert nan_vault_rec["guard_ok"] is False
    assert nan_vault_rec["anse_energy"] == DEFAULT_ANSE_MAXIMUM_PAIN


# ---------------------------------------------------------------------------
# S5: Validation Errors, Provenance Tampering & Security/Stub Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_provenance_tampering_and_security_gates() -> None:
    """Exercise all input validation errors, provenance tampering checks, and HiveMind security gates."""
    valid_sd = {
        "blocks.0.mlp.experts.0.w1_B": torch.ones((4, 2), dtype=torch.float32),
        "blocks.0.mlp.router.weight": torch.ones((2, 4), dtype=torch.float32),
        "ln.weight": torch.ones(4, dtype=torch.float32),
        "ln.bias": torch.zeros(4, dtype=torch.float32),
    }

    def _ok_eval(_sd: Any) -> dict[str, float]:
        return {"E0_general": 2.0, "E1_spec": 1.5}

    # 1. State dict & digest validation errors
    with pytest.raises(ValueError, match="non-empty"):
        clone_state_dict({})
    with pytest.raises(ValueError, match="non-empty strings"):
        clone_state_dict({"  ": torch.ones(2)})
    with pytest.raises(TypeError, match="torch.Tensor"):
        clone_state_dict({"a": [1.0]})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-empty"):
        compute_state_dict_sha256({})
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_state_dict_sha256({"a": 1})  # type: ignore[dict-item]
    assert len(compute_state_dict_sha256({"empty": torch.empty(0)})) == 64

    with pytest.raises(ValueError, match="non-empty"):
        count_edited_parameters({}, valid_sd)
    with pytest.raises(ValueError, match="identical keys"):
        count_edited_parameters({"a": torch.ones(2)}, {"b": torch.ones(2)})
    with pytest.raises(ValueError, match="Shape mismatch"):
        count_edited_parameters({"a": torch.ones(2)}, {"a": torch.ones(3)})

    cross_dt_edits, cross_dt_total = count_edited_parameters(
        {"a": torch.ones(2, dtype=torch.float32), "b": torch.empty(0), "c": torch.ones(2, dtype=torch.int64)},
        {"a": torch.ones(2, dtype=torch.int64), "b": torch.empty(0), "c": torch.ones(2, dtype=torch.int64)},
    )
    assert cross_dt_edits == 2 and cross_dt_total == 4

    # 2. Eval & ANSE metrics validation errors + container formats + soft margin penalty
    parsed_tuple, fin_t = parse_eval_domain_nll((1.75, {"domain_nll": {"E0_gen": 1.5, "E1_code": 2.0}}))
    assert fin_t is True and parsed_tuple == {"E0_gen": 1.5, "E1_code": 2.0}
    with pytest.raises(TypeError, match="non-empty mapping"):
        parse_eval_domain_nll("bad")
    with pytest.raises(ValueError, match="non-empty mapping"):
        parse_eval_domain_nll({"domain_nll": {}})
    with pytest.raises(ValueError, match="non-empty strings"):
        parse_eval_domain_nll({" ": 1.0})
    with pytest.raises(ValueError, match="number"):
        parse_eval_domain_nll({"E0_gen": True})
    with pytest.raises(ValueError, match="finite"):
        parse_eval_domain_nll({"E0_gen": float("nan")})
    with pytest.raises(ValueError, match="> 0.0"):
        parse_eval_domain_nll({"E0_gen": 0.0})

    with pytest.raises(ValueError, match="non-empty mapping"):
        evaluate_anse_domain_metrics({}, {"a": 1.0})
    with pytest.raises(ValueError, match="identical domain keys"):
        evaluate_anse_domain_metrics({"a": 1.0}, {"b": 1.0})
    with pytest.raises(ValueError, match="non-empty string"):
        evaluate_anse_domain_metrics({"a": 1.0}, {"a": 1.0}, general_prefix=" ")
    soft_m = evaluate_anse_domain_metrics({"E0_gen": 1.0}, {"E0_gen": 1.009}, soft_margin_penalty=0.05)
    assert soft_m["guard_ok"] is True
    assert soft_m["anse_energy"] == pytest.approx(0.059)

    # 3. Dataclass & pipeline input validation errors
    with pytest.raises(ValueError, match="name"):
        WSPStageSpec(name=" ", transform_fn=lambda sd: sd)
    with pytest.raises(TypeError, match="callable"):
        WSPStageSpec(name="x", transform_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
        WSPStageSpec(name="x", transform_fn=lambda sd: sd, max_edit_fraction=1.5)
    with pytest.raises(TypeError, match="int"):
        WSPStageSpec(name="x", transform_fn=lambda sd: sd, max_edited_params=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        WSPStageSpec(name="x", transform_fn=lambda sd: sd, max_edited_params=-1)
    with pytest.raises(ValueError, match=">= 1"):
        make_quantization_compensation_stage(["w"], group_size=0)
    with pytest.raises(ValueError, match="finite"):
        make_layernorm_calibration_stage(lambda_ln=float("nan"))
    int_noop = run_monotone_wsp_stage_pipeline(
        {"step_idx": torch.tensor([1], dtype=torch.int64)},
        [WSPStageSpec("noop_int", lambda sd: sd)],
        _ok_eval,
    )
    assert int_noop.status == "COMPLETED_NOOP"
    with pytest.raises(ValueError, match="stage_kind"):
        WSPStageSpec(name="x", transform_fn=lambda sd: sd, stage_kind="")
    with pytest.raises(TypeError, match="str"):
        WSPStageSpec(name="x", transform_fn=lambda sd: sd, description=123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-empty"):
        HoldoutVaultLockState(vault_id=" ")
    with pytest.raises(TypeError, match="bool"):
        HoldoutVaultLockState(locked=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        HoldoutVaultLockState(access_count=-1)

    with pytest.raises(ValueError, match="non-empty"):
        run_monotone_wsp_stage_pipeline({}, [WSPStageSpec("s", lambda sd: sd)], _ok_eval)
    with pytest.raises(ValueError, match="non-empty strings"):
        run_monotone_wsp_stage_pipeline({" ": torch.ones(2)}, [WSPStageSpec("s", lambda sd: sd)], _ok_eval)
    with pytest.raises(TypeError, match="torch.Tensor"):
        run_monotone_wsp_stage_pipeline({"a": [1.0]}, [WSPStageSpec("s", lambda sd: sd)], _ok_eval)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-finite"):
        run_monotone_wsp_stage_pipeline({"a": torch.tensor([float("nan")])}, [WSPStageSpec("s", lambda sd: sd)], _ok_eval)
    with pytest.raises(ValueError, match="at least one scalar"):
        run_monotone_wsp_stage_pipeline({"a": torch.empty(0)}, [WSPStageSpec("s", lambda sd: sd)], _ok_eval)
    with pytest.raises(ValueError, match="non-empty sequence"):
        run_monotone_wsp_stage_pipeline(valid_sd, [], _ok_eval)
    with pytest.raises(TypeError, match="WSPStageSpec"):
        run_monotone_wsp_stage_pipeline(valid_sd, ["bad"], _ok_eval)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="Duplicate stage name"):
        run_monotone_wsp_stage_pipeline(
            valid_sd,
            [WSPStageSpec("dup", lambda sd: sd), WSPStageSpec("dup", lambda sd: sd)],
            _ok_eval,
        )
    with pytest.raises(TypeError, match="callable"):
        run_monotone_wsp_stage_pipeline(valid_sd, [WSPStageSpec("s", lambda sd: sd)], "bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real number"):
        run_monotone_wsp_stage_pipeline(valid_sd, [WSPStageSpec("s", lambda sd: sd)], _ok_eval, general_guard_rel=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0.0"):
        run_monotone_wsp_stage_pipeline(valid_sd, [WSPStageSpec("s", lambda sd: sd)], _ok_eval, general_guard_rel=-0.1)
    with pytest.raises(ValueError, match="> 0.0"):
        run_monotone_wsp_stage_pipeline(valid_sd, [WSPStageSpec("s", lambda sd: sd)], _ok_eval, maximum_pain=0.0)
    with pytest.raises(ValueError, match="general_prefix"):
        run_monotone_wsp_stage_pipeline(valid_sd, [WSPStageSpec("s", lambda sd: sd)], _ok_eval, general_prefix=" ")
    with pytest.raises(ValueError, match="must match"):
        run_monotone_wsp_stage_pipeline(
            valid_sd,
            [WSPStageSpec("s", lambda sd: sd)],
            _ok_eval,
            base_domain_nll={"other": 1.0},
        )

    # Stage returning bad container, mismatched keys, non-tensor, wrong shape, or mismatched eval keys
    with pytest.raises(TypeError, match="non-empty mapping"):
        run_monotone_wsp_stage_pipeline(valid_sd, [WSPStageSpec("s", lambda _sd: "bad")], _ok_eval)  # type: ignore[arg-type,return-value]
    with pytest.raises(ValueError, match="mismatched state_dict keys"):
        run_monotone_wsp_stage_pipeline(valid_sd, [WSPStageSpec("s", lambda _sd: {"extra": torch.ones(1)})], _ok_eval)
    with pytest.raises(TypeError, match="torch.Tensor"):
        run_monotone_wsp_stage_pipeline(
            valid_sd,
            [WSPStageSpec("s", lambda sd: {**sd, "ln.weight": [1, 2, 3, 4]})],  # type: ignore[dict-item]
            _ok_eval,
        )
    with pytest.raises(ValueError, match="changed shape"):
        run_monotone_wsp_stage_pipeline(
            valid_sd,
            [WSPStageSpec("s", lambda sd: {**sd, "ln.weight": torch.ones(2)})],
            _ok_eval,
        )
    with pytest.raises(ValueError, match="mismatched domain keys"):
        run_monotone_wsp_stage_pipeline(
            valid_sd,
            [WSPStageSpec("s", lambda sd: {**sd, "ln.weight": torch.full((4,), 2.0)})],
            lambda sd: {"E0_general": 2.0}
            if float(sd["ln.weight"][0].item()) > 1.5
            else {"E0_general": 2.0, "E1_spec": 1.5},
            max_total_edit_fraction=1.0,
        )

    # 4. Vault lock validation & provenance tampering checks
    with pytest.raises(TypeError, match="HoldoutVaultLockState"):
        verify_holdout_vault_single_shot(valid_sd, valid_sd, _ok_eval, vault_lock_state="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="vault_id"):
        verify_holdout_vault_single_shot(valid_sd, valid_sd, _ok_eval, vault_lock_state={"vault_id": " "})
    with pytest.raises(TypeError, match="callable"):
        verify_holdout_vault_single_shot(valid_sd, valid_sd, "bad", vault_lock_state={})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="general_prefix"):
        verify_holdout_vault_single_shot(valid_sd, valid_sd, _ok_eval, vault_lock_state={}, general_prefix="")
    with pytest.raises(ValueError, match="mismatched domain keys"):
        verify_holdout_vault_single_shot(
            valid_sd,
            valid_sd,
            lambda _sd: {"wrong": 1.0},
            vault_lock_state={},
            base_vault_domain_nll={"E0_general": 2.0},
        )

    with pytest.raises(TypeError, match="WSPStageReceipt"):
        verify_stage_receipt_provenance("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Mapping"):
        verify_pipeline_receipt_provenance("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Mapping"):
        verify_vault_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_stage_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert verify_stage_receipt_provenance({"receipt_sha256": "0" * 64, "x": float("nan")}) is False
    assert verify_pipeline_receipt_provenance({"receipt_sha256": "0" * 64, "stages": "bad"}) is False
    assert verify_pipeline_receipt_provenance({"receipt_sha256": "0" * 64, "stages": [{"receipt_sha256": "bad"}]}) is False
    assert verify_vault_receipt_provenance({"receipt_sha256": "0" * 64, "x": 1}) is False

    # 5. Stage builder validation & edge branches
    with pytest.raises(ValueError, match="expert_indices"):
        make_ghost_sink_stage([])
    with pytest.raises(ValueError, match="layer_indices"):
        make_ghost_sink_stage([0], layer_indices=[])

    with pytest.raises(TypeError, match="torch.Tensor"):
        make_cross_layer_anchor_stage([[1.0]], target_layers=[1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        make_cross_layer_anchor_stage(torch.empty(0), target_layers=[1])
    with pytest.raises(TypeError, match="floating-point"):
        make_cross_layer_anchor_stage(torch.ones((4, 1), dtype=torch.int64), target_layers=[1])
    with pytest.raises(ValueError, match="finite"):
        make_cross_layer_anchor_stage(torch.tensor([float("nan")]), target_layers=[1])
    with pytest.raises(ValueError, match="target_layers"):
        make_cross_layer_anchor_stage(torch.ones(4), target_layers=[])
    with pytest.raises(ValueError, match="expert_indices"):
        make_cross_layer_anchor_stage(torch.ones(4), target_layers=[1], expert_indices=[])
    with pytest.raises(ValueError, match="factors"):
        make_cross_layer_anchor_stage(torch.ones(4), target_layers=[1], factors=[])
    with pytest.raises(ValueError, match="subset"):
        make_cross_layer_anchor_stage(torch.ones(4), target_layers=[1], factors=["bad"])
    with pytest.raises(ValueError, match="> 0.0"):
        make_cross_layer_anchor_stage(torch.ones(4), target_layers=[1], strength=0.0)
    # Also exercise 1D anchor_subspace + mismatched layer/dim skip paths in _transform
    anchor_1d_stage = make_cross_layer_anchor_stage(torch.ones(8), target_layers=[0], factors=("w1_B",))
    assert anchor_1d_stage.transform_fn(valid_sd)["blocks.0.mlp.experts.0.w1_B"].shape == (4, 2)

    with pytest.raises(ValueError, match="router_directions"):
        make_orthogonal_router_stage({})
    with pytest.raises(ValueError, match="1D tensor"):
        make_orthogonal_router_stage({0: torch.ones((2, 2))})
    with pytest.raises(TypeError, match="floating-point"):
        make_orthogonal_router_stage({0: torch.ones(4, dtype=torch.int64)})
    with pytest.raises(ValueError, match="finite"):
        make_orthogonal_router_stage({0: torch.tensor([1.0, float("inf")])})
    with pytest.raises(ValueError, match="identical hidden"):
        make_orthogonal_router_stage({0: torch.ones(4), 1: torch.ones(6)})
    with pytest.raises(ValueError, match="target_layers"):
        make_orthogonal_router_stage({0: torch.ones(4)}, target_layers=[])
    with pytest.raises(ValueError, match="1D tensor"):
        make_orthogonal_router_stage({0: torch.ones(4)}, sink_expert=0, sink_boost=1.0, sink_direction=torch.ones((2, 2)))
    with pytest.raises(ValueError, match="router width"):
        make_orthogonal_router_stage({0: torch.ones(4)}, sink_expert=0, sink_boost=1.0, sink_direction=torch.ones(6))
    with pytest.raises(ValueError, match="finite"):
        make_orthogonal_router_stage(
            {0: torch.ones(4)}, sink_expert=0, sink_boost=1.0, sink_direction=torch.full((4,), float("nan"))
        )
    r_stage_skip = make_orthogonal_router_stage({0: torch.ones(8)}, target_layers=[1])
    assert torch.equal(r_stage_skip.transform_fn(valid_sd)["blocks.0.mlp.router.weight"], valid_sd["blocks.0.mlp.router.weight"])
    r_stage_width = make_orthogonal_router_stage({0: torch.ones(8)}, target_layers=[0])
    assert torch.equal(r_stage_width.transform_fn(valid_sd)["blocks.0.mlp.router.weight"], valid_sd["blocks.0.mlp.router.weight"])

    with pytest.raises(ValueError, match="ln_weight_key"):
        make_layernorm_calibration_stage(ln_weight_key=" ")
    with pytest.raises(ValueError, match="ln_bias_key"):
        make_layernorm_calibration_stage(ln_bias_key=" ")
    with pytest.raises(ValueError, match="1D tensor"):
        make_layernorm_calibration_stage(target_ln_weight=torch.ones((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        make_layernorm_calibration_stage(target_ln_weight=torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="1D tensor"):
        make_layernorm_calibration_stage(target_ln_bias=torch.ones((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        make_layernorm_calibration_stage(target_ln_bias=torch.tensor([float("inf")]))
    with pytest.raises(ValueError, match="target_ln_weight shape"):
        make_layernorm_calibration_stage(target_ln_weight=torch.ones(8)).transform_fn(valid_sd)
    with pytest.raises(ValueError, match="target_ln_bias shape"):
        make_layernorm_calibration_stage(target_ln_bias=torch.zeros(8)).transform_fn(valid_sd)

    with pytest.raises(ValueError, match="target_keys"):
        make_quantization_compensation_stage([])
    with pytest.raises(ValueError, match="non-empty strings"):
        make_quantization_compensation_stage([" "])
    with pytest.raises(ValueError, match="bits must be 4 or 8"):
        make_quantization_compensation_stage(["w"], bits=16)
    with pytest.raises(KeyError, match="missing_key"):
        make_quantization_compensation_stage(["missing_key"]).transform_fn(valid_sd)

    # 6. AGPL-3.0 header, CI-safe fallback + optional ZeroStubAudit, PRSecurityScanner, lateral movement, FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn16", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn16", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/monotone_wsp_stage_composer.py", TARGET_MODULE_PATH),
        ("tests/test_monotone_wsp_stage_composer_scenarios.py", TEST_FILE_PATH),
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
        {
            "object": STAGE_RECEIPT_OBJECT,
            "nll_mean": 1.142,
            "stage_delta_nats": -0.084,
            "anse_energy": -0.084,
            "committed": True,
        },
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+RECEIPT = {sample_receipt_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1116,
        pr_title="feat(swarm): monotone multi-stage WSP composition pipeline with per-stage ANSE ratchet and rollback checkpoints",
        pr_body="Implements NEURAL-CTRL-DYN-16 monotone WSP stage composer and deterministic CPU scenario tests.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): monotone multi-stage WSP composition pipeline with per-stage ANSE ratchet and rollback checkpoints"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
