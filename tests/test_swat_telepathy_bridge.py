# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit and pre-registered control tests for Season 34 SwaTTelepathyBridge (`HM-S34-01`..`HM-S34-06`)."""

from __future__ import annotations

import numpy as np

from hivemind.swarm.swat_telepathy_bridge import (
    S34_BEE_MODEL_SPECS,
    SwaTTelepathyBridge,
    UniversalHiveLatentHub,
    audit_latent_kv_safety,
    bound_kv_to_rms_stats,
    compute_cross_causal_limits,
    compute_layer_group_slices,
    compute_virtual_positions,
    cross_attention_with_causal_limits,
)


def test_identical_tokenizer_positive_control_virtual_positions_equal_integers() -> None:
    """Positive Control (E22): when T_S == T_R, character-span RoPE positions p_i == i."""
    offsets = [(0, 5), (5, 11), (11, 18), (18, 22), (22, 30), (30, 39)]
    pos = compute_virtual_positions(offsets, offsets, mode="interp")
    expected = np.arange(len(offsets), dtype=np.float64)
    np.testing.assert_allclose(pos, expected, atol=1e-12)


def test_cross_causal_limits_prevent_future_prefix_leakage() -> None:
    """Negative Control 2 (E22): mutating sender tokens with b^S_i > b^R_j causes zero change at j."""
    # Sender has finer subword splits than Receiver
    sender_offsets = [(0, 3), (3, 6), (6, 10), (10, 14), (14, 20)]
    receiver_offsets = [(0, 6), (6, 12), (12, 20)]

    limits = compute_cross_causal_limits(sender_offsets, receiver_offsets)
    # Receiver token 0 ends at 6 -> sender tokens 0 (end 3) and 1 (end 6) allowed -> limit = 2
    # Receiver token 1 ends at 12 -> sender tokens 0,1,2 (end 10) allowed, token 3 ends at 14 > 12 -> limit = 3
    # Receiver token 2 ends at 20 -> all 5 sender tokens allowed -> limit = 5
    assert limits.tolist() == [2, 3, 5]

    rng = np.random.default_rng(42)
    q = rng.standard_normal((3, 8))
    k1 = rng.standard_normal((5, 8))
    v1 = rng.standard_normal((5, 8))

    out1 = cross_attention_with_causal_limits(q, k1, v1, limits)

    # Mutate future sender tokens (indices 3 and 4, which have b^S_i > b^R_1 = 12)
    k2 = k1.copy()
    v2 = v1.copy()
    k2[3:] += 999.0
    v2[3:] -= 999.0

    out2 = cross_attention_with_causal_limits(q, k2, v2, limits)
    # Receiver positions 0 and 1 must be bitwise identical!
    assert np.array_equal(out1[0], out2[0])
    assert np.array_equal(out1[1], out2[1])
    assert not np.allclose(out1[2], out2[2])


def test_bound_kv_to_rms_stats_and_latent_safety_auditor() -> None:
    """HM-S34-06: calibrated KV caches pass (0% false rejection); tampered caches are 100% rejected."""
    rng = np.random.default_rng(7)
    raw_kv = rng.standard_normal((1, 16, 4, 2, 2, 8)) * 5.0
    target_key_rms = np.full((4, 2), 1.25, dtype=np.float64)
    target_value_rms = np.full((4, 2), 0.85, dtype=np.float64)

    calibrated = bound_kv_to_rms_stats(raw_kv, target_key_rms, target_value_rms)
    clean_audit = audit_latent_kv_safety(calibrated, target_key_rms, target_value_rms)
    assert clean_audit["accepted"] is True
    assert clean_audit["reasons"] == []

    # 1. RMS inflation attack
    inflated = calibrated * 3.0
    inf_audit = audit_latent_kv_safety(inflated, target_key_rms, target_value_rms)
    assert inf_audit["accepted"] is False
    assert any("rms_scale_inflation" in r for r in inf_audit["reasons"])

    # 2. Refusal direction suppression attack
    refusal_dir = np.ones(8, dtype=np.float64)
    suppressed = calibrated - 2.0 * refusal_dir
    suppressed_cal = bound_kv_to_rms_stats(suppressed, target_key_rms, target_value_rms)
    ref_audit = audit_latent_kv_safety(
        suppressed_cal,
        target_key_rms,
        target_value_rms,
        refusal_direction=refusal_dir,
        min_refusal_cosine=-0.20,
    )
    assert ref_audit["accepted"] is False
    assert any("refusal_direction_suppressed" in r for r in ref_audit["reasons"])


def test_universal_hive_latent_hub_2n_complexity_and_vram_guards() -> None:
    """HM-S34-01, HM-S34-03, HM-S34-04, HM-S34-05: verify 2n < n(n-1), VLM OCR gate, and 8GB WDDM bounds."""
    hub = UniversalHiveLatentHub()
    rep = hub.complexity_report()
    assert rep["num_models"] >= 5
    assert rep["hub_mappers_2n"] == 2 * rep["num_models"]
    assert rep["pairwise_mappers_n_n_minus_1"] == rep["num_models"] * (rep["num_models"] - 1)
    assert rep["strictly_fewer_mappers"] is True

    bridge = SwaTTelepathyBridge(hub=hub, n_blocks=8)
    slices_40 = compute_layer_group_slices(40, 8)
    assert len(slices_40) == 8
    assert slices_40[0] == (0, 5)
    assert slices_40[-1] == (35, 40)

    # Simulate 28K context compacted 5x (AM20 -> 5,600 tokens) from Granite-3.3-8B to SmolLM3-3B on RTX 2070
    s_offsets = [(i * 4, (i + 1) * 4) for i in range(512)]
    r_offsets = [(i * 4, (i + 1) * 4) for i in range(512)]
    plan = bridge.plan_transfer(
        "ibm-granite/granite-3.3-8b-instruct",
        "HuggingFaceTB/SmolLM3-3B",
        s_offsets,
        r_offsets,
        am_compaction_ratio=5.0,
        rtx2070_used_vram_mib=1437.0,
        t4_free_vram_mib=8200.0,
    )
    assert plan["schema"] == "hivemind.mesh.telepathy_kv.v1"
    assert plan["vram_guards"]["all_guards_passed"] is True
    assert plan["compacted_fp8_kv_mib"] < plan["raw_fp16_kv_mib"]

    # Table 21 dense-OCR guard for VLM -> text-only transfer
    vlm_gate_dense = bridge.should_fallback_to_native_vlm(
        S34_BEE_MODEL_SPECS["google/gemma-3-4b-it"],
        S34_BEE_MODEL_SPECS["HuggingFaceTB/SmolLM3-3B"],
        ocr_text_density=0.72,
    )
    assert vlm_gate_dense["use_swat_kv_transfer"] is False
    assert vlm_gate_dense["fallback_mode"] == "native_vlm_sender_decode"
