# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-05: 4-split anti-leakage ANSE neural-control harness."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from gwaya.ast_audit import ZeroStubAudit
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    extract_moe_trainable_buckets,
    load_moe_trainable_buckets,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.wsp_alphaedit_control import (
    WSPArmConfig,
    calibrate_final_layernorm_and_sink_experts,
    compute_embeddings_batch,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    per_chunk_nll_from_pre_ln,
    synthesize_wsp_arm_buckets,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = REPO_ROOT / "scripts" / "hivemind_neural_control_harness.py"
TEST_FILE_PATH = Path(__file__).resolve()
EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."

DOMAINS: tuple[str, ...] = (
    "E0_general_fineweb",
    "E1_python_systems_code",
    "E2_math_reasoning",
    "E5_exoplanet_transit",
    "E6_stellar_spectroscopy",
    "E7_defensive_cyber",
)


def _load_module_from_path(name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


harness = _load_module_from_path("hivemind_neural_control_harness", HARNESS_PATH)


def _make_disjoint_split_chunks(
    split_name: str,
    row_offset: int,
    *,
    chunks_per_domain: int = 40,
    seq_len: int = 12,
    score_from: int = 6,
    vocab_size: int = 32,
    seed: int = 20261123,
) -> list[dict[str, Any]]:
    """Build deterministic CPU token chunks tagged with unique ``(dataset_id, row)`` keys."""
    gen = torch.Generator(device="cpu").manual_seed(seed + row_offset)
    chunks: list[dict[str, Any]] = []
    for dom_idx, dom in enumerate(DOMAINS):
        is_general = dom.startswith("E0_")
        lo_tok, hi_tok = (0, vocab_size // 2) if is_general else (vocab_size // 2, vocab_size)
        target_tok = 3 if is_general else 20
        for r in range(chunks_per_domain):
            ids = torch.randint(lo_tok, hi_tok, (seq_len,), generator=gen, dtype=torch.long).tolist()
            for t in range(score_from, seq_len):
                ids[t] = target_tok
            chunks.append(
                {
                    "split": split_name,
                    "expert_id": dom,
                    "dataset_id": f"ds_{dom}",
                    "row": row_offset + r,
                    "ids": ids,
                }
            )
    return chunks


def _make_synthetic_receipt_chunks(
    split_rows: list[dict[str, Any]],
    *,
    e0_ratio: float,
    spec_ratio: float,
    base_e0: float = 4.20,
    base_spec: float = 4.30,
    noise_seed: int = 100,
    per_domain_ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Generate deterministic paired NLL receipt rows bound to exact ``split_rows`` provenance."""
    gen = torch.Generator(device="cpu").manual_seed(noise_seed)
    noise = (0.015 * torch.randn(len(split_rows), generator=gen, dtype=torch.float64)).tolist()
    chunks: list[dict[str, Any]] = []
    for idx, row in enumerate(split_rows):
        eid = str(row["expert_id"])
        is_e0 = eid.startswith("E0_")
        b_nll = (base_e0 if is_e0 else base_spec) + abs(noise[idx])
        if per_domain_ratios and eid in per_domain_ratios:
            ratio = float(per_domain_ratios[eid])
        else:
            ratio = float(e0_ratio if is_e0 else spec_ratio)
        m_nll = max(0.01, b_nll * ratio)
        chunks.append(
            {
                "expert_id": eid,
                "dataset_id": row["dataset_id"],
                "row": row["row"],
                "base_nll": b_nll,
                "moe_nll": m_nll,
            }
        )
    return {"chunks": chunks}


def _build_tiny_moe_cpu_fixture(seed: int = 20261123) -> tuple[CrowdV1WithEngram, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build a tiny deterministic CPU CrowdV1WithEngram + Sparse MoE checkpoint pair (Base & C2)."""
    vocab_size = 32
    hidden_size = 24
    inter_size = 36
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cfg = CrowdV1RuntimeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden_size // 4,
        intermediate_size=inter_size,
        max_position_embeddings=32,
        engram_buckets=0,
        tie_word_embeddings=False,
    )
    model = CrowdV1WithEngram(cfg)
    half_v = vocab_size // 2
    half_h = hidden_size // 2

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        u_p = torch.zeros(half_h, dtype=torch.float32)
        u_p[0] = 1.0 / math.sqrt(2.0)
        u_p[1] = -1.0 / math.sqrt(2.0)
        u_s = torch.zeros(hidden_size - half_h, dtype=torch.float32)
        u_s[0] = 1.0 / math.sqrt(2.0)
        u_s[1] = -1.0 / math.sqrt(2.0)

        p_raw = 0.25 * torch.randn((half_v, half_h), generator=gen, dtype=torch.float32)
        p_raw = p_raw - p_raw.mean(dim=1, keepdim=True)
        p_raw = p_raw + 2.5 * u_p.unsqueeze(0)
        s_raw = 0.25 * torch.randn((vocab_size - half_v, hidden_size - half_h), generator=gen, dtype=torch.float32)
        s_raw = s_raw - s_raw.mean(dim=1, keepdim=True)
        s_raw = s_raw + 2.5 * u_s.unsqueeze(0)
        wte[:half_v, :half_h] = p_raw
        wte[half_v:, half_h:] = s_raw
        model.wte.weight.copy_(wte)

        target_vec = torch.zeros(hidden_size, dtype=torch.float32)
        target_vec[half_h + 2] = 1.0 / math.sqrt(2.0)
        target_vec[half_h + 3] = -1.0 / math.sqrt(2.0)
        bad_e0_vec = torch.zeros(hidden_size, dtype=torch.float32)
        bad_e0_vec[:half_h] = -3.0 * u_p
        head_w = 0.25 * torch.randn((vocab_size, hidden_size), generator=gen, dtype=torch.float32)
        head_w[:, :half_h] = 0.0
        head_w[3, :half_h] = 0.55 * u_p
        head_w[20] = 2.20 * target_vec
        model.head.weight.copy_(head_w)
        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()

        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln2.weight.fill_(1.0)
            blk.mlp.w1.weight[:hidden_size, :hidden_size].copy_(1.5 * torch.eye(hidden_size))
            blk.mlp.w2.weight[:hidden_size, :hidden_size].copy_(1.5 * torch.eye(hidden_size))

    upcycle_crowd_v1_to_sparse_moe(model, num_experts=4, top_k=2, lora_rank=4)
    start_buckets = {k: v.detach().clone() for k, v in extract_moe_trainable_buckets(model, num_router_fragments=1).items()}

    with torch.no_grad():
        for blk in model.blocks:
            # Harmful ghost sink expert_0: realistic E0 drift without saturating LayerNorm
            exp0 = blk.mlp.experts["0"]
            for p in exp0.parameters():
                p.zero_()
            exp0.w3_A[0, 0] = 0.05
            exp0.w3_B[:half_h, 0].copy_(-0.35 * u_p)

            # Specialist expert_1: specialist gain along target_vec + moderate prose interference
            exp1 = blk.mlp.experts["1"]
            for p in exp1.parameters():
                p.zero_()
            exp1.w3_A[0, half_h] = 0.08
            exp1.w3_B[:, 0] = 0.25 * target_vec
            exp1.w3_A[1, 0] = 0.04
            exp1.w3_B[:half_h, 1].copy_(-0.25 * u_p)

            blk.mlp.router.weight.zero_()
            blk.mlp.router.weight[0, :half_h] = 3.0 * u_p
            blk.mlp.router.weight[0, half_h:] = -6.0 * u_s
            blk.mlp.router.weight[1, half_h:] = 2.5 * u_s
            blk.mlp.router.weight[1, :half_h] = 2.2 * u_p
        model.ln.bias[0] -= 0.15
        model.ln.bias[1] += 0.15

    c2_buckets = {k: v.detach().clone() for k, v in extract_moe_trainable_buckets(model, num_router_fragments=1).items()}
    return model, start_buckets, c2_buckets


@torch.no_grad()
def _score_model_split_nlls(
    model: CrowdV1WithEngram,
    buckets: dict[str, torch.Tensor],
    split_chunks: list[dict[str, Any]],
    *,
    score_from: int = 6,
) -> list[float]:
    load_moe_trainable_buckets(model, buckets, num_router_fragments=1)
    model.eval()
    ids_t = torch.tensor([c["ids"] for c in split_chunks], dtype=torch.long)
    xb = compute_embeddings_batch(model, ids_t)
    cos = model.cos_cached.to(ids_t.device)
    sin = model.sin_cached.to(ids_t.device)
    for blk in model.blocks:
        xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
        xb = forward_block_post_mlp(blk, xm, mi)
    nlls = per_chunk_nll_from_pre_ln(
        xb, model.ln.weight, model.ln.bias, float(model.ln.eps), model.head.weight, ids_t, score_from=score_from
    )
    return [float(x) for x in nlls.tolist()]


def _build_receipt_from_nlls(
    split_chunks: list[dict[str, Any]],
    base_nlls: list[float],
    moe_nlls: list[float],
) -> dict[str, Any]:
    return {
        "chunks": [
            {
                "expert_id": ch["expert_id"],
                "dataset_id": ch["dataset_id"],
                "row": ch["row"],
                "base_nll": b_val,
                "moe_nll": m_val,
            }
            for ch, b_val, m_val in zip(split_chunks, base_nlls, moe_nlls, strict=True)
        ]
    }


# ---------------------------------------------------------------------------
# S1: Multi-Generation Ratchet Progression (S1 -> S2 -> S3 -> finalize_vault)
# ---------------------------------------------------------------------------


def test_s1_end_to_end_cpu_moe_multi_generation_ratchet_s1_s2_s3_finalize_vault() -> None:
    """Verify full 4-split S1 -> S2 -> S3 -> S4 (no regress) -> finalize_vault ratchet on CPU."""
    probe = _make_disjoint_split_chunks("probe", 120, seed=11)
    selection = _make_disjoint_split_chunks("selection", 40, seed=22)
    benchmark = _make_disjoint_split_chunks("benchmark", 80, seed=33)
    virgin_vault = _make_disjoint_split_chunks("virgin_vault", 160, seed=44)

    model, start_buckets, c2_buckets = _build_tiny_moe_cpu_fixture(seed=20261123)
    cal = calibrate_final_layernorm_and_sink_experts(
        model, start_buckets, c2_buckets, probe, score_from=6, loeo_chunks_per_domain=4
    )

    arm_ln = WSPArmConfig(name="WSP_A1_ln_sink", use_ln_calibration=True, use_ghost_sink=True)
    arm_hyb = WSPArmConfig(
        name="WSP_A4_hybrid_balanced",
        use_ln_calibration=True,
        use_ghost_sink=True,
        router_beta_suppress=1.75,
        router_beta_sink=1.25,
        alphaedit_eta_x=2.0,
        alphaedit_eta_act=1.5,
    )
    b_a1, _ = synthesize_wsp_arm_buckets(model, start_buckets, c2_buckets, probe, arm_ln, cal, score_from=6)
    b_a4, _ = synthesize_wsp_arm_buckets(model, start_buckets, c2_buckets, probe, arm_hyb, cal, score_from=6)

    sel_base = _score_model_split_nlls(model, start_buckets, selection)
    sel_c2 = _score_model_split_nlls(model, c2_buckets, selection)
    sel_a1 = _score_model_split_nlls(model, b_a1, selection)
    sel_a4 = _score_model_split_nlls(model, b_a4, selection)

    # Verify C2 breaks E0 guard on selection while real WSP_A4_hybrid_balanced clears guard & MOSTLY_CONTENT
    eval_c2 = harness.evaluate_neural_control_receipt(_build_receipt_from_nlls(selection, sel_base, sel_c2))
    eval_a4 = harness.evaluate_neural_control_receipt(_build_receipt_from_nlls(selection, sel_base, sel_a4))
    assert eval_c2["guard_ok"] is False
    assert eval_c2["anse_energy"] >= harness.ANSE_MAXIMUM_PAIN - 10.0
    assert eval_a4["guard_ok"] is True
    assert eval_a4["corrects"] is True

    # Construct progressive S2 and S3 refinements over the real WSP_A4 baseline NLLs on selection,
    # plus a non-improving S4 candidate so ratchet monotonicity is exercised.
    a4_sel_rec = _build_receipt_from_nlls(selection, sel_base, sel_a4)
    s2_sel_chunks = []
    s3_sel_chunks = []
    s4_sel_chunks = []
    for row in a4_sel_rec["chunks"]:
        is_spec = not str(row["expert_id"]).startswith("E0_")
        s2_m = max(1e-4, row["moe_nll"] - (0.08 if is_spec else 0.002))
        s3_m = max(1e-4, row["moe_nll"] - (0.16 if is_spec else 0.004))
        s4_m = max(1e-4, row["moe_nll"] - (0.04 if is_spec else 0.0))
        s2_sel_chunks.append({**row, "moe_nll": s2_m})
        s3_sel_chunks.append({**row, "moe_nll": s3_m})
        s4_sel_chunks.append({**row, "moe_nll": s4_m})

    generations = [
        (
            "S1",
            {
                "WSP_C2_uncontrolled": _build_receipt_from_nlls(selection, sel_base, sel_c2),
                "WSP_A1_ln_sink": _build_receipt_from_nlls(selection, sel_base, sel_a1),
                "WSP_A4_hybrid_balanced": lambda rows: _build_receipt_from_nlls(list(rows), sel_base, sel_a4),
            },
        ),
        (
            "S2",
            {
                "WSP_S2_layer_localized": {"chunks": s2_sel_chunks},
                "WSP_S2_overtuned_bad": _make_synthetic_receipt_chunks(selection, e0_ratio=1.025, spec_ratio=0.80, noise_seed=201),
            },
        ),
        (
            "S3",
            {
                "WSP_S3_svd_plus_router": {"chunks": s3_sel_chunks},
            },
        ),
        (
            "S4",
            {
                "WSP_S4_weaker_multigate": {"chunks": s4_sel_chunks},
            },
        ),
    ]

    bench_calls: list[str] = []
    vault_calls: list[str] = []

    def _bench_eval(arm_name: str, split_rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
        bench_calls.append(arm_name)
        return _make_synthetic_receipt_chunks(list(split_rows), e0_ratio=1.002, spec_ratio=0.92, noise_seed=301)

    def _vault_eval(arm_name: str, split_rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
        vault_calls.append(arm_name)
        return _make_synthetic_receipt_chunks(list(split_rows), e0_ratio=1.003, spec_ratio=0.91, noise_seed=401)

    ratchet = harness.run_anse_ratchet_selection(
        probe,
        selection,
        benchmark,
        virgin_vault,
        generations,
        benchmark_evaluator=_bench_eval,
        vault_evaluator=_vault_eval,
        benchmark_baseline_diff_nats=-0.0857,
        preregistration={"experiment_id": "NEURAL-CTRL-05-TEST", "frozen": True},
    )

    assert ratchet["stopping_gate_triggered"] is False
    assert ratchet["vault_status"] == "PROMOTED"
    assert ratchet["promoted"] is True
    assert ratchet["champion_mostly_content_survived"] is True
    assert ratchet["benchmark_queries_burned"] == 1
    assert ratchet["virgin_vault_queries_burned"] == 1
    assert ratchet["single_shot_vault_invariant_ok"] is True
    assert bench_calls == ["WSP_S3_svd_plus_router"]
    assert vault_calls == ["WSP_S3_svd_plus_router"]

    hist = ratchet["generation_history"]
    assert [h["generation"] for h in hist] == ["S1", "S2", "S3", "S4"]
    assert hist[0]["ratchet_champion_after_gen"] == "WSP_A4_hybrid_balanced"
    assert hist[0]["ratchet_improved"] is True
    assert hist[1]["ratchet_champion_after_gen"] == "WSP_S2_layer_localized"
    assert hist[1]["ratchet_improved"] is True
    assert hist[1]["delta_energy_vs_prev"] < 0.0
    assert hist[2]["ratchet_champion_after_gen"] == "WSP_S3_svd_plus_router"
    assert hist[2]["ratchet_improved"] is True
    assert hist[2]["delta_energy_vs_prev"] < 0.0
    # S4 did not beat S3 -> champion remains S3
    assert hist[3]["ratchet_improved"] is False
    assert hist[3]["ratchet_champion_after_gen"] == "WSP_S3_svd_plus_router"

    assert ratchet["benchmark_result"]["beats_baseline"] is True
    assert ratchet["virgin_vault_result"]["mostly_content_survived"] is True
    assert harness.verify_receipt_provenance(ratchet) is True


# ---------------------------------------------------------------------------
# S2: 4-Split Anti-Leakage Detection & Provenance Verification
# ---------------------------------------------------------------------------


def test_s2_four_split_anti_leakage_fail_closed() -> None:
    """Assert fail-closed rejection across all 6 pairwise overlaps, duplicates, and malformed keys."""
    p = _make_disjoint_split_chunks("probe", 1000, chunks_per_domain=4, seq_len=8, score_from=4)
    s = _make_disjoint_split_chunks("selection", 2000, chunks_per_domain=4, seq_len=8, score_from=4)
    b = _make_disjoint_split_chunks("benchmark", 3000, chunks_per_domain=4, seq_len=8, score_from=4)
    v = _make_disjoint_split_chunks("virgin_vault", 4000, chunks_per_domain=4, seq_len=8, score_from=4)

    res = harness.verify_pairwise_split_disjointness(
        {"chunks": p},
        {"rows": [{**row, "row": str(row["row"])} for row in s]},
        b,
        v,
    )
    assert res["disjoint"] is True
    assert res["total_unique_rows"] == 96
    assert harness.verify_receipt_provenance(res) is True

    # Test all 6 pairwise overlaps
    pairs = (
        ("probe", "selection", p, [dict(p[0])] + s[1:], b, v),
        ("probe", "benchmark", p, s, [dict(p[0])] + b[1:], v),
        ("probe", "virgin_vault", p, s, b, [dict(p[0])] + v[1:],),
        ("selection", "benchmark", p, s, [dict(s[0])] + b[1:], v),
        ("selection", "virgin_vault", p, s, b, [dict(s[0])] + v[1:],),
        ("benchmark", "virgin_vault", p, s, b, [dict(b[0])] + v[1:],),
    )
    for left, right, sp_p, sp_s, sp_b, sp_v in pairs:
        with pytest.raises(harness.NeuralControlHarnessError, match=f"Data leakage detected between '{left}' and '{right}'"):
            harness.verify_pairwise_split_disjointness(sp_p, sp_s, sp_b, sp_v)

    # Duplicate inside a single split
    with pytest.raises(harness.NeuralControlHarnessError, match="duplicate"):
        harness.verify_pairwise_split_disjointness(p + [dict(p[0])], s, b, v)

    # Empty split fails closed by default, allowed when require_non_empty=False
    with pytest.raises(harness.NeuralControlHarnessError, match="non-empty"):
        harness.verify_pairwise_split_disjointness([], s, b, v)
    empty_ok = harness.verify_pairwise_split_disjointness([], s, b, v, require_non_empty=False)
    assert empty_ok["split_counts"]["probe"] == 0

    # Malformed split container / key branches
    for bad_split in ("not_a_list", {"other": []}, [123], [{"dataset_id": "ds"}], [{"dataset_id": "", "row": 1}], [{"dataset_id": "ds", "row": True}], [{"dataset_id": "ds", "row": 1.5}], [{"dataset_id": "ds", "row": -1}]):
        with pytest.raises(harness.NeuralControlHarnessError):
            harness.verify_pairwise_split_disjointness(bad_split, s, b, v)  # type: ignore[arg-type]

    # Receipt-level probe leakage and provenance mismatch
    probe_leaked_receipt = _make_synthetic_receipt_chunks(p[:24], e0_ratio=1.0, spec_ratio=0.9)
    probe_keys = {(str(r["dataset_id"]), int(r["row"])) for r in p}
    sel_keys = {(str(r["dataset_id"]), int(r["row"])) for r in s}
    with pytest.raises(harness.NeuralControlHarnessError, match="Data leakage detected"):
        harness.evaluate_neural_control_receipt(
            probe_leaked_receipt,
            min_chunks_per_expert=2,
            min_total_chunks=10,
            forbidden_split_keys={"probe": probe_keys},
        )
    with pytest.raises(harness.NeuralControlHarnessError, match="Split provenance mismatch"):
        harness.evaluate_neural_control_receipt(
            probe_leaked_receipt,
            min_chunks_per_expert=2,
            min_total_chunks=10,
            expected_split_name="selection",
            allowed_split_keys=sel_keys,
        )
    with pytest.raises(harness.NeuralControlHarnessError, match="Duplicate"):
        harness.evaluate_neural_control_receipt(
            {"chunks": [probe_leaked_receipt["chunks"][0], probe_leaked_receipt["chunks"][0]]},
            min_chunks_per_expert=1,
            min_total_chunks=1,
        )
    with pytest.raises(harness.NeuralControlHarnessError, match="declared split"):
        harness.evaluate_neural_control_receipt(
            {"split": "probe", "chunks": probe_leaked_receipt["chunks"]},
            expected_split_name="selection",
        )


# ---------------------------------------------------------------------------
# S3: Stopping-Gate Abort (0 Vault Runs Burned) & Single-Shot Gate Modes
# ---------------------------------------------------------------------------


def test_s3_stopping_gate_abort_zero_vault_queries_and_vault_promotion_modes() -> None:
    """Assert 0 vault/benchmark queries burned on selection failure and single-shot vault gating."""
    probe = _make_disjoint_split_chunks("probe", 120, seed=51)
    selection = _make_disjoint_split_chunks("selection", 40, seed=52)
    benchmark = _make_disjoint_split_chunks("benchmark", 80, seed=53)
    virgin_vault = _make_disjoint_split_chunks("virgin_vault", 160, seed=54)

    # Case 1: All candidates fail guard or fail CI -> stopping gate triggers, 0 vault calls!
    failing_gens = {
        "S1": {
            "guard_fail_e0": _make_synthetic_receipt_chunks(selection, e0_ratio=1.015, spec_ratio=0.85, noise_seed=1),
            "harm_both": _make_synthetic_receipt_chunks(selection, e0_ratio=1.005, spec_ratio=1.005, noise_seed=2),
        },
        "S2": {
            "guard_fail_spec": _make_synthetic_receipt_chunks(
                selection,
                e0_ratio=0.995,
                spec_ratio=0.90,
                per_domain_ratios={"E1_python_systems_code": 1.014},
                noise_seed=3,
            ),
        },
    }
    bench_calls: list[str] = []
    vault_calls: list[str] = []

    def _never_bench(arm_name: str, _rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
        bench_calls.append(arm_name)
        return {}

    def _never_vault(arm_name: str, _rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
        vault_calls.append(arm_name)
        return {}

    aborted = harness.run_anse_ratchet_selection(
        probe,
        selection,
        benchmark,
        virgin_vault,
        failing_gens,
        benchmark_evaluator=_never_bench,
        vault_evaluator=_never_vault,
        bootstrap_resamples=500,
    )
    assert aborted["stopping_gate_triggered"] is True
    assert aborted["selection_decision"]["status"] == "NO_CORRECTING_ARM"
    assert aborted["vault_status"] == "ABORTED_SELECTION_STOPPING_GATE"
    assert aborted["promoted"] is False
    assert aborted["benchmark_queries_burned"] == 0
    assert aborted["virgin_vault_queries_burned"] == 0
    assert aborted["single_shot_vault_invariant_ok"] is True
    assert bench_calls == []
    assert vault_calls == []

    # Case 2: Selection passes, defer_virgin_vault=True -> benchmark runs once, vault stays untouched (0 burned)
    passing_gens = {
        "S1": {
            "arm_alpha": _make_synthetic_receipt_chunks(selection, e0_ratio=1.001, spec_ratio=0.92, noise_seed=10),
            "arm_beta": _make_synthetic_receipt_chunks(selection, e0_ratio=1.001, spec_ratio=0.95, noise_seed=11),
        }
    }
    deferred = harness.run_anse_ratchet_selection(
        probe,
        selection,
        benchmark,
        virgin_vault,
        passing_gens,
        benchmark_evaluator={
            "arm_alpha": _make_synthetic_receipt_chunks(benchmark, e0_ratio=1.015, spec_ratio=0.92, noise_seed=12)
        },
        vault_evaluator=_never_vault,
        defer_virgin_vault=True,
        bootstrap_resamples=500,
    )
    assert deferred["vault_status"] == "DEFERRED"
    assert deferred["benchmark_queries_burned"] == 1
    assert deferred["virgin_vault_queries_burned"] == 0
    assert vault_calls == []

    # Case 3: require_benchmark_gate_before_vault=True blocks vault when benchmark fails guard
    bench_gated = harness.run_anse_ratchet_selection(
        probe,
        selection,
        benchmark,
        virgin_vault,
        passing_gens,
        benchmark_evaluator={
            "arm_alpha": _make_synthetic_receipt_chunks(benchmark, e0_ratio=1.020, spec_ratio=0.92, noise_seed=12)
        },
        vault_evaluator=_never_vault,
        require_benchmark_gate_before_vault=True,
        bootstrap_resamples=500,
    )
    assert bench_gated["vault_status"] == "ABORTED_BENCHMARK_GATE"
    assert bench_gated["virgin_vault_queries_burned"] == 0
    assert vault_calls == []

    # Case 4: Champion selected, virgin vault runs ONCE and fails guard -> REJECTED_ON_VAULT
    rejected = harness.run_anse_ratchet_selection(
        probe,
        selection,
        benchmark,
        virgin_vault,
        passing_gens,
        vault_evaluator={
            "arm_alpha": _make_synthetic_receipt_chunks(virgin_vault, e0_ratio=1.014, spec_ratio=0.92, noise_seed=13)
        },
        bootstrap_resamples=500,
    )
    assert rejected["vault_status"] == "REJECTED_ON_VAULT"
    assert rejected["promoted"] is False
    assert rejected["virgin_vault_queries_burned"] == 1

    # Case 5: Missing champion receipt in mapping / invalid evaluator type fails closed
    with pytest.raises(harness.NeuralControlHarnessError, match="Missing benchmark receipt"):
        harness.run_anse_ratchet_selection(
            probe, selection, benchmark, virgin_vault, passing_gens, benchmark_evaluator={}, bootstrap_resamples=200
        )
    with pytest.raises(harness.NeuralControlHarnessError, match="must be a callable or a mapping"):
        harness.run_anse_ratchet_selection(
            probe, selection, benchmark, virgin_vault, passing_gens, benchmark_evaluator="bad", bootstrap_resamples=200  # type: ignore[arg-type]
        )
    with pytest.raises(harness.NeuralControlHarnessError, match="non-mapping"):
        harness.run_anse_ratchet_selection(
            probe, selection, benchmark, virgin_vault, passing_gens, benchmark_evaluator=lambda _a, _r: "bad", bootstrap_resamples=200  # type: ignore[return-value,arg-type]
        )


# ---------------------------------------------------------------------------
# S4: Paired Bootstrap CI Exactness, MOSTLY_CONTENT Boundary & ANSE Energy
# ---------------------------------------------------------------------------


def test_s4_bootstrap_ci_mostly_content_anse_energy_and_validation() -> None:
    """Verify paired bootstrap 95% CIs, ANSE soft/hard penalties, MOSTLY_CONTENT threshold, and validation."""
    # 1. Exact bootstrap equivalence including >2000 batched path
    rng_data = np.random.default_rng(777)
    diffs_large = rng_data.normal(loc=-0.18, scale=0.05, size=2200)
    ci_a = harness.paired_bootstrap_ci95(diffs_large, n_resamples=10_000, seed=42)
    rng_ref = np.random.default_rng(42)
    ref_means = np.empty(10_000, dtype=np.float64)
    for s0 in range(0, 10_000, 500):
        s1 = min(10_000, s0 + 500)
        idx = rng_ref.integers(0, diffs_large.size, size=(s1 - s0, diffs_large.size))
        ref_means[s0:s1] = diffs_large[idx].mean(axis=1)
    lo_ref, hi_ref = np.percentile(ref_means, [2.5, 97.5])
    assert ci_a == pytest.approx((float(lo_ref), float(hi_ref)), abs=1e-12)

    # 2. Configurable bootstrap seed changes CI cleanly & 10,000 resamples is deterministic
    rows_200 = _make_disjoint_split_chunks("selection", 9000, chunks_per_domain=40, seq_len=8, score_from=4)[:200]
    rec_soft = _make_synthetic_receipt_chunks(rows_200, e0_ratio=1.009, spec_ratio=0.90, noise_seed=55)
    ev_seed_a = harness.evaluate_neural_control_receipt(rec_soft, bootstrap_seed=20261123)
    ev_seed_a_rep = harness.evaluate_neural_control_receipt(rec_soft, bootstrap_seed=20261123)
    ev_seed_b = harness.evaluate_neural_control_receipt(rec_soft, bootstrap_seed=20261118)
    assert ev_seed_a["receipt_sha256"] == ev_seed_a_rep["receipt_sha256"]
    assert ev_seed_a["ci95"] != ev_seed_b["ci95"]
    assert ev_seed_a["guard_ok"] is True
    # Soft penalty (+0.05) triggered because 1.0085 < e0_ratio <= 1.01
    assert ev_seed_a["anse_energy"] == pytest.approx(ev_seed_a["mean_diff_vs_base"] + 0.05, abs=1e-12)

    # 3. MOSTLY_CONTENT exact threshold (-0.1653) vs MOSTLY_FORMAT_SOME_CONTENT (-0.1652)
    chunks_mc = [
        {"expert_id": f"E{i // 40}_dom", "base_nll": 4.0, "moe_nll": 4.0 - 0.1653}
        for i in range(200)
    ]
    ev_mc = harness.evaluate_neural_control_receipt(chunks_mc, bootstrap_resamples=200)
    assert ev_mc["midrow_label"] == "MOSTLY_CONTENT"
    assert ev_mc["mostly_content"] is True
    assert ev_mc["mostly_content_survived"] is True

    chunks_fmt = [
        {"expert_id": f"E{i // 40}_dom", "base_nll": 4.0, "moe_nll": 4.0 - 0.1652}
        for i in range(200)
    ]
    ev_fmt = harness.evaluate_neural_control_receipt(chunks_fmt, bootstrap_resamples=200)
    assert ev_fmt["midrow_label"] == "MOSTLY_FORMAT_SOME_CONTENT"
    assert ev_fmt["mostly_content"] is False

    # 4. NO_GAIN, HARM, INSUFFICIENT_N, and zero qualifying experts
    chunks_nogain = [
        {"expert_id": f"E{i // 40}_dom", "base_nll": 4.0, "moe_nll": 4.0 + (0.2 if i % 2 == 0 else -0.2)}
        for i in range(200)
    ]
    ev_ng = harness.evaluate_neural_control_receipt(chunks_nogain, bootstrap_resamples=300)
    assert ev_ng["verdict_ip300"] == "NO_GAIN"
    assert ev_ng["corrects"] is False

    chunks_harm = [{"expert_id": f"E{i // 40}_dom", "base_nll": 4.0, "moe_nll": 4.3} for i in range(200)]
    ev_harm = harness.evaluate_neural_control_receipt(chunks_harm, bootstrap_resamples=200)
    assert ev_harm["verdict_ip300"] == "HARM"
    assert ev_harm["guard_ok"] is False
    assert ev_harm["anse_energy"] >= harness.ANSE_MAXIMUM_PAIN

    ev_insuff = harness.evaluate_neural_control_receipt(chunks_mc[:80], min_chunks_per_expert=20, min_total_chunks=200, bootstrap_resamples=200)
    assert ev_insuff["verdict_ip300"] == "INSUFFICIENT_N"
    assert ev_insuff["corrects"] is False

    # Zero qualifying experts when every expert has < min_chunks_per_expert
    chunks_all_lt = [{"expert_id": f"E{i}_dom", "base_nll": 4.0, "moe_nll": 3.5} for i in range(25)]
    ev_zero_qual = harness.evaluate_neural_control_receipt(chunks_all_lt, min_chunks_per_expert=20, min_total_chunks=200)
    assert ev_zero_qual["n"] == 0
    assert ev_zero_qual["verdict_ip300"] == "INSUFFICIENT_N"
    assert len(ev_zero_qual["excluded_lt_min"]) == 25

    # 5. Validation & provenance failure branches
    assert harness.verify_receipt_provenance({"receipt_sha256": "short"}) is False
    assert harness.verify_receipt_provenance({"receipt_sha256": "0" * 64, "bad": float("nan")}) is False
    with pytest.raises(harness.NeuralControlHarnessError):
        harness.verify_receipt_provenance(["not", "a", "dict"])  # type: ignore[arg-type]

    ev_named = harness.evaluate_neural_control_receipt(
        {"name": "arm_named", "split": "selection", "chunks": chunks_mc},
        bootstrap_resamples=200,
    )
    assert ev_named["arm_name"] == "arm_named"
    assert ev_named["split"] == "selection"

    for bad_receipt in (
        "not_a_receipt",
        {},
        {"chunks": "not_a_list"},
        {"chunks": []},
        {"chunks": [42]},
        {"chunks": [{"expert_id": "", "base_nll": 4.0, "moe_nll": 3.5}]},
        {"chunks": [{"expert_id": "E0", "base_nll": 4.0}]},
        {"chunks": [{"expert_id": "E0", "base_nll": True, "moe_nll": 3.5}]},
        {"chunks": [{"expert_id": "E0", "base_nll": 0.0, "moe_nll": 3.5}]},
        {"chunks": [{"expert_id": "E0", "base_nll": 4.0, "moe_nll": -0.1}]},
        {"chunks": [{"expert_id": "E0", "base_nll": float("nan"), "moe_nll": 3.5}]},
    ):
        with pytest.raises(harness.NeuralControlHarnessError):
            harness.evaluate_neural_control_receipt(bad_receipt)  # type: ignore[arg-type]

    with pytest.raises(harness.NeuralControlHarnessError):
        harness.paired_bootstrap_ci95([], n_resamples=10)
    with pytest.raises(harness.NeuralControlHarnessError):
        harness.paired_bootstrap_ci95([0.1], n_resamples=0)
    with pytest.raises(harness.NeuralControlHarnessError):
        harness.paired_bootstrap_ci95([0.1], seed=-1)
    with pytest.raises(harness.NeuralControlHarnessError):
        harness.evaluate_neural_control_receipt(chunks_mc, domain_guard_max_ratio=0.0)

    # _normalize_generations validation branches
    for bad_gens in (
        {},
        [],
        {"": {"a": {}}},
        {"S1": {}},
        [("S1",)],
        [{"generation": "", "candidates": {"a": {}}}],
        [{"generation": "S1", "candidates": {}}],
        [("S1", {"a": {}}), ("S1", {"b": {}})],
        [("S1", {"": {}})],
        [("S1", {"dup": {}}), ("S2", {"dup": {}})],
        "bad",
    ):
        with pytest.raises(harness.NeuralControlHarnessError):
            harness._normalize_generations(bad_gens)  # type: ignore[arg-type]

    with pytest.raises(harness.NeuralControlHarnessError, match="receipt mapping or callable"):
        harness.run_anse_ratchet_selection(
            rows_200[:20],
            rows_200[20:40],
            rows_200[40:60],
            rows_200[60:80],
            [("S1", {"bad_cand": "not_callable_or_mapping"})],
        )


# ---------------------------------------------------------------------------
# S5: CLI / JSON / Markdown / SHA-256 Provenance Verification & Output Safety
# ---------------------------------------------------------------------------


def test_s5_cli_json_markdown_provenance_and_output_safety(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify --bundle, --receipt, --splits-only, --verify-receipt, --json, --markdown, and --strict."""
    probe = _make_disjoint_split_chunks("probe", 120, seed=61)
    selection = _make_disjoint_split_chunks("selection", 40, seed=62)
    benchmark = _make_disjoint_split_chunks("benchmark", 80, seed=63)
    virgin_vault = _make_disjoint_split_chunks("virgin_vault", 160, seed=64)

    bundle = {
        "preregistration": {"experiment_id": "CLI-TEST-2026-11-23"},
        "splits": {
            "probe": probe,
            "selection": selection,
            "benchmark": benchmark,
            "virgin_vault": virgin_vault,
        },
        "generations": [
            {
                "generation": "S1",
                "candidates": {
                    "WSP_A1": _make_synthetic_receipt_chunks(selection, e0_ratio=1.002, spec_ratio=0.95, noise_seed=71),
                },
            },
            {
                "generation": "S2",
                "candidates": {
                    "WSP_S2": _make_synthetic_receipt_chunks(selection, e0_ratio=1.001, spec_ratio=0.90, noise_seed=72),
                },
            },
        ],
        "benchmark_receipts": {
            "WSP_S2": _make_synthetic_receipt_chunks(benchmark, e0_ratio=1.002, spec_ratio=0.90, noise_seed=73),
        },
        "virgin_vault_receipts": {
            "WSP_S2": _make_synthetic_receipt_chunks(virgin_vault, e0_ratio=1.002, spec_ratio=0.90, noise_seed=74),
        },
    }
    bundle_path = tmp_path / "ratchet_bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    out_json = tmp_path / "out" / "ratchet.json"
    out_md = tmp_path / "out" / "ratchet.md"

    rc = harness.main(
        [
            "--bundle",
            str(bundle_path),
            "--bootstrap-resamples",
            "500",
            "--benchmark-baseline-diff",
            "-0.0857",
            "--out-json",
            str(out_json),
            "--out-markdown",
            str(out_md),
            "--json",
            "--strict",
        ]
    )
    assert rc == 0
    stdout_doc = json.loads(capsys.readouterr().out)
    assert stdout_doc["promoted"] is True
    assert stdout_doc["vault_status"] == "PROMOTED"
    assert harness.verify_receipt_provenance(stdout_doc) is True

    # Determinism: second run emits exact identical receipt_sha256
    rc2 = harness.main(
        [
            "--bundle",
            str(bundle_path),
            "--bootstrap-resamples",
            "500",
            "--benchmark-baseline-diff",
            "-0.0857",
            "--markdown",
        ]
    )
    assert rc2 == 0
    md_out = capsys.readouterr().out
    assert "PROMOTED" in md_out
    assert stdout_doc["receipt_sha256"] in md_out
    assert json.loads(out_json.read_text(encoding="utf-8"))["receipt_sha256"] == stdout_doc["receipt_sha256"]

    # --verify-receipt passes on valid receipt, fails on tampered receipt
    verify_out = tmp_path / "verify.json"
    assert harness.main(["--verify-receipt", str(out_json), "--out-json", str(verify_out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True

    tampered_path = tmp_path / "tampered.json"
    tampered_doc = dict(stdout_doc)
    tampered_doc["promoted"] = False
    tampered_path.write_text(json.dumps(tampered_doc), encoding="utf-8")
    assert harness.main(["--verify-receipt", str(tampered_path)]) == 2
    assert "provenance_verified=False" in capsys.readouterr().out

    # --receipt + --splits cleanly verifies no probe leakage; rejects leaked probe row
    sel_rec_path = tmp_path / "sel_receipt.json"
    sel_rec_path.write_text(json.dumps(bundle["generations"][1]["candidates"]["WSP_S2"]), encoding="utf-8")
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(bundle["splits"]), encoding="utf-8")
    assert harness.main(["--receipt", str(sel_rec_path), "--splits", str(splits_path), "--bootstrap-resamples", "500"]) == 0
    assert "Receipt Metrics & Per-Domain Guard" in capsys.readouterr().out

    leaked_rec_path = tmp_path / "leaked_receipt.json"
    leaked_rec_path.write_text(
        json.dumps(_make_synthetic_receipt_chunks(probe, e0_ratio=1.0, spec_ratio=0.9)),
        encoding="utf-8",
    )
    assert harness.main(["--receipt", str(leaked_rec_path), "--splits", str(splits_path)]) == 2

    # --splits-only CLI & subprocess invocation for __main__ coverage
    assert harness.main(["--splits-only", str(splits_path)]) == 0
    assert "4-Split Anti-Leakage Verification Report" in capsys.readouterr().out
    sub_res = subprocess.run(
        [sys.executable, str(HARNESS_PATH), "--splits-only", str(splits_path), "--json"],
        cwd=str(REPO_ROOT),
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), CUDA_VISIBLE_DEVICES=""),
        capture_output=True,
        text=True,
        check=False,
    )
    assert sub_res.returncode == 0
    assert json.loads(sub_res.stdout)["disjoint"] is True

    # --strict exits 2 on stopping gate abort
    abort_bundle = {
        "splits": bundle["splits"],
        "generations": {
            "S1": {
                "bad": _make_synthetic_receipt_chunks(selection, e0_ratio=1.03, spec_ratio=0.90, noise_seed=81),
            }
        },
    }
    abort_path = tmp_path / "abort_bundle.json"
    abort_path.write_text(json.dumps(abort_bundle), encoding="utf-8")
    assert harness.main(["--bundle", str(abort_path), "--bootstrap-resamples", "300", "--strict"]) == 2
    assert "ABORTED_SELECTION_STOPPING_GATE" in capsys.readouterr().out

    # Error handling: missing file, malformed JSON, unsupported markdown object, output safety
    assert harness.main(["--splits-only", str(tmp_path / "missing.json")]) == 2
    bad_json_path = tmp_path / "bad.json"
    bad_json_path.write_text("{not json", encoding="utf-8")
    assert harness.main(["--bundle", str(bad_json_path)]) == 2

    bad_bundle_path = tmp_path / "missing_gens.json"
    bad_bundle_path.write_text(json.dumps({"splits": bundle["splits"]}), encoding="utf-8")
    assert harness.main(["--bundle", str(bad_bundle_path)]) == 2
    bad_bundle2 = tmp_path / "not_obj.json"
    bad_bundle2.write_text("[]", encoding="utf-8")
    assert harness.main(["--bundle", str(bad_bundle2)]) == 2
    bad_splits_file = tmp_path / "bad_splits.json"
    bad_splits_file.write_text(json.dumps({"probe": []}), encoding="utf-8")
    assert harness.main(["--splits-only", str(bad_splits_file)]) == 2
    assert harness.main(["--receipt", str(sel_rec_path), "--splits", str(bad_splits_file)]) == 2
    bad_bundle3 = tmp_path / "bad_bundle_splits.json"
    bad_bundle3.write_text(json.dumps({"splits": {"probe": []}, "generations": {}}), encoding="utf-8")
    assert harness.main(["--bundle", str(bad_bundle3)]) == 2

    with pytest.raises(harness.NeuralControlHarnessError, match="Unsupported report object"):
        harness.render_markdown_report({"object": "unknown"})

    forbidden_payload = '{"' + "sco" + 're": 9}'
    with pytest.raises(harness.NeuralControlHarnessError, match="Output safety guard"):
        harness.assert_safe_output(forbidden_payload)


# ---------------------------------------------------------------------------
# S6: AGPL-3.0, Lateral Movement, ZeroStubAudit, PRSecurityScanner & Gatekeeper
# ---------------------------------------------------------------------------


def test_s6_agpl3_lateral_movement_zero_stub_and_security_audits() -> None:
    """Verify 0 violations across lateral-movement, ZeroStubAudit, PRSecurityScanner, and gatekeeper."""
    lateral_guard = _load_module_from_path("test_no_lateral_movement_ctrl05", REPO_ROOT / "tests" / "test_no_lateral_movement.py")
    gatekeeper = _load_module_from_path("pr_gatekeeper_ctrl05", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []

    harness_src = HARNESS_PATH.read_text(encoding="utf-8")
    # Confirm zero outbound network libraries or torch.cuda references in the harness script
    harness_tree = ast.parse(harness_src, filename="scripts/hivemind_neural_control_harness.py")
    imported_roots: set[str] = set()
    for node in ast.walk(harness_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots.isdisjoint({"urllib", "requests", "httpx", "socket", "aiohttp", "subprocess"})

    for rel, path_obj in (
        ("scripts/hivemind_neural_control_harness.py", HARNESS_PATH),
        ("tests/test_hivemind_neural_control_harness.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"

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

    # Negative control: planted LAN URL fails lateral-movement scanner
    bad_host = ".".join(["192", "168", "1", "42"])
    bad_url = "ht" + f"tp://{bad_host}:8000/leak"
    poisoned = harness_src + f'\n_BAD_PEER = "{bad_url}"\n'
    assert lateral_guard.audit_source_for_lateral_movement(poisoned, "poisoned.py") != []

    pr_audit = scanner.audit_pull_request(
        pr_number=1105,
        pr_title="feat(swarm): 4-split anti-leakage ANSE neural-control scenario benchmark harness and receipt verifier",
        pr_body="Implements NEURAL-CTRL-05 4-split anti-leakage evaluation and ANSE ratchet harness.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): 4-split anti-leakage ANSE neural-control scenario benchmark harness and receipt verifier"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
