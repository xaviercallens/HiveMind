# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control hardening tests for HM-S35-01 through HM-S35-05 swarm modules."""

from __future__ import annotations

from pathlib import Path

import pytest

from hivemind.swarm.s35_01_optimal_energy_amcp_xswat_codec import (
    benchmark_a2_rope_plane_quantizer,
    pack_amcp_xswat_latent_kv_envelope,
    quantize_a2_point,
)
from hivemind.swarm.s35_02_gwaya_laya_failclosed_gate import (
    audit_candidate_gwaya_anse,
    inspect_python_ast_anti_stub,
    run_gwaya_a3_sample_and_repair,
)
from hivemind.swarm.s35_03_elenchus_leanlib_sundials_r64 import (
    audit_lean4_source_elenchus,
    strip_lean4_comments,
    verify_robertson_sundials_tier_x,
)
from hivemind.swarm.s35_04_xlayerstorm_xensorsharp_bridge import (
    compute_xlayerstorm_turboquant_schedule,
)
from hivemind.swarm.s35_05_unified_xdev_ecosystem_bridge import (
    _eval_ast_conservation_fn,
    build_xdev_portfolio_catalog,
    classify_repo_origin,
    run_unified_4layer_ecosystem_pipeline,
)


def test_s35_01_a2_rope_and_amcp_envelope_reject_malformed_inputs() -> None:
    qx, qy = quantize_a2_point(0.25, -0.5, scale=0.5)
    assert isinstance(qx, float) and isinstance(qy, float)

    for bad_coord in (True, False, float("nan"), float("inf"), float("-inf"), "0.1"):
        with pytest.raises(ValueError, match="finite real number"):
            quantize_a2_point(bad_coord, 0.0, scale=1.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="finite real number"):
            quantize_a2_point(0.0, bad_coord, scale=1.0)  # type: ignore[arg-type]

    for bad_scale in (True, 0.0, -0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            quantize_a2_point(0.1, 0.2, scale=bad_scale)  # type: ignore[arg-type]

    for bad_pairs in ([], "ab", [(0.1,)], [(True, 0.2)], [(0.1, float("nan"))]):
        with pytest.raises(ValueError):
            benchmark_a2_rope_plane_quantizer(bad_pairs)  # type: ignore[arg-type]

    valid_blocks = [[0.1 * i for i in range(8)]]
    for bad_agent in ("", "   ", "agent|inject", "agent\n1", "agent\x7f"):
        with pytest.raises(ValueError):
            pack_amcp_xswat_latent_kv_envelope(bad_agent, "amcp://recv", valid_blocks)
        with pytest.raises(ValueError):
            pack_amcp_xswat_latent_kv_envelope("amcp://send", bad_agent, valid_blocks)

    for bad_blocks in (
        [],
        "12345678",
        [[1.0] * 7],
        [[True] + [0.1] * 7],
        [[float("nan")] + [0.1] * 7],
    ):
        with pytest.raises(ValueError):
            pack_amcp_xswat_latent_kv_envelope("amcp://send", "amcp://recv", bad_blocks)  # type: ignore[arg-type]

    for bad_bpd in (True, 0.0, -1.0, 17.0, float("nan")):
        with pytest.raises(ValueError):
            pack_amcp_xswat_latent_kv_envelope(
                "amcp://send",
                "amcp://recv",
                valid_blocks,
                bits_per_dim=bad_bpd,  # type: ignore[arg-type]
            )


def test_s35_02_gwaya_gate_rejects_boolean_confidence_and_malformed_arguments() -> None:
    with pytest.raises(ValueError, match="source_code"):
        inspect_python_ast_anti_stub(123)  # type: ignore[arg-type]

    clean_src = "def f(x: int) -> int:\n    return x + 1\n"
    for bad_conf in (True, False, -0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="raw_logprob_confidence"):
            audit_candidate_gwaya_anse(
                clean_src,
                lambda _s: True,
                lambda _s: True,
                raw_logprob_confidence=bad_conf,  # type: ignore[arg-type]
            )

    for bad_lat in (True, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="latency_ms"):
            audit_candidate_gwaya_anse(
                clean_src,
                lambda _s: True,
                lambda _s: True,
                latency_ms=bad_lat,  # type: ignore[arg-type]
            )

    for bad_mem in (True, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="memory_mib"):
            audit_candidate_gwaya_anse(
                clean_src,
                lambda _s: True,
                lambda _s: True,
                memory_mib=bad_mem,  # type: ignore[arg-type]
            )

    for bad_thresh in (True, False, 0.0, -0.1, 1.01, float("nan")):
        with pytest.raises(ValueError, match="laya_threshold"):
            audit_candidate_gwaya_anse(
                clean_src,
                lambda _s: True,
                lambda _s: True,
                laya_threshold=bad_thresh,  # type: ignore[arg-type]
            )

    with pytest.raises(ValueError, match="callable"):
        audit_candidate_gwaya_anse(clean_src, "not-callable", lambda _s: True)  # type: ignore[arg-type]

    for bad_cands in ([], clean_src, [clean_src, 123]):
        with pytest.raises(ValueError):
            run_gwaya_a3_sample_and_repair(bad_cands, lambda _s: True, lambda _s: True)  # type: ignore[arg-type]


def test_s35_03_elenchus_lean_and_sundials_r64_reject_malformed_inputs() -> None:
    for bad_lean in (None, 123, "theorem t : 1 = 1 := by rfl\x00"):
        with pytest.raises(ValueError, match="lean_source"):
            strip_lean4_comments(bad_lean)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="lean_source"):
            audit_lean4_source_elenchus(bad_lean)  # type: ignore[arg-type]

    valid_args = {"problem": "robertson", "rtol": 1e-9}
    with pytest.raises(ValueError, match="mappings"):
        verify_robertson_sundials_tier_x(["bad"], valid_args)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="booleans"):
        verify_robertson_sundials_tier_x(valid_args, valid_args, ran=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="booleans"):
        verify_robertson_sundials_tier_x(valid_args, valid_args, stub="false")  # type: ignore[arg-type]

    for bad_str in ("", "  ", "bad\nstr", "bad\x00str"):
        with pytest.raises(ValueError, match="producer"):
            verify_robertson_sundials_tier_x(valid_args, valid_args, producer=bad_str)
        with pytest.raises(ValueError, match="git_rev"):
            verify_robertson_sundials_tier_x(valid_args, valid_args, git_rev=bad_str)

    for bad_triple in ((1.0, 0.0), (True, 0.0, 0.0), "100"):
        with pytest.raises(ValueError, match="y_initial"):
            verify_robertson_sundials_tier_x(valid_args, valid_args, y_initial=bad_triple)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="y_final"):
            verify_robertson_sundials_tier_x(valid_args, valid_args, y_final=bad_triple)  # type: ignore[arg-type]

    # Boolean inside received_arguments["y_final"] is ignored as invalid solver output
    bool_yf_args = {"problem": "robertson", "y_final": [True, 0.0, 0.0]}
    rec = verify_robertson_sundials_tier_x(bool_yf_args, bool_yf_args)
    assert rec.accepted_by_elenchus_bridge is False
    assert rec.evidentiary_value == "none"


def test_s35_04_xlayerstorm_schedule_rejects_invalid_numeric_parameters() -> None:
    plan = compute_xlayerstorm_turboquant_schedule()
    assert plan.within_citizen_vram_cap is True

    for bad_total in (True, False, 0, -4, 10_001, 32.0):
        with pytest.raises(ValueError, match="Invalid total_experts"):
            compute_xlayerstorm_turboquant_schedule(total_experts=bad_total)  # type: ignore[arg-type]

    for bad_topk in (True, False, 0, -1, 33, 2.0):
        with pytest.raises(ValueError, match="Invalid active_top_k_experts"):
            compute_xlayerstorm_turboquant_schedule(total_experts=32, active_top_k_experts=bad_topk)  # type: ignore[arg-type]

    for kw in ("single_expert_fp16_mib", "e8_quant_compression", "turboquant_ratio", "pcie_bandwidth_gbps"):
        for bad_val in (True, 0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match=kw):
                compute_xlayerstorm_turboquant_schedule(**{kw: bad_val})  # type: ignore[arg-type]

    for kw in ("shared_prelude_mib", "raw_kv_cache_mib", "layer_compute_ms"):
        for bad_val in (True, -0.1, float("nan"), float("inf")):
            with pytest.raises(ValueError, match=kw):
                compute_xlayerstorm_turboquant_schedule(**{kw: bad_val})  # type: ignore[arg-type]


def test_s35_05_unified_xdev_bridge_rejects_malformed_inputs(tmp_path: Path) -> None:
    for bad_repo in ("", "   ", "owner/repo", "repo\\path", "repo\nname", 123):
        with pytest.raises(ValueError):
            classify_repo_origin(bad_repo)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="xdev_root"):
        build_xdev_portfolio_catalog(xdev_root="/tmp/not-a-path-obj")  # type: ignore[arg-type]

    # Boolean literal in candidate arithmetic expression is rejected
    with pytest.raises(ValueError, match="Unsupported AST expression"):
        _eval_ast_conservation_fn(
            "def compute_conservation_residual(m0: float, m1: float) -> float:\n    return abs(m1 - m0) + True\n",
            1.0,
            2.0,
        )

    with pytest.raises(ValueError, match="xdev_root"):
        run_unified_4layer_ecosystem_pipeline(xdev_root="not-a-path")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="candidates"):
        run_unified_4layer_ecosystem_pipeline(xdev_root=tmp_path, candidates="not-a-sequence-of-strs")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="lean_source"):
        run_unified_4layer_ecosystem_pipeline(xdev_root=tmp_path, lean_source="bad\x00lean")

    with pytest.raises(ValueError, match="sent_arguments"):
        run_unified_4layer_ecosystem_pipeline(xdev_root=tmp_path, sent_arguments=["bad"])  # type: ignore[arg-type]
