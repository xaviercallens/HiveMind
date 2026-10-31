# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit and pre-registered control tests for Season 34 SwaTTelepathyBridge (`HM-S34-01`..`HM-S34-06`)."""

from __future__ import annotations

import numpy as np

from hivemind.swarm.swat_telepathy_bridge import (
    S34_BEE_MODEL_SPECS,
    SwaTTelepathyBridge,
    UniversalHiveLatentHub,
    WakeToSleepConsolidationBridge,
    audit_latent_kv_safety,
    bound_kv_to_rms_stats,
    compact_kv_with_attention_matching,
    compute_cms_chunk_periods,
    compute_cross_causal_limits,
    compute_knowledge_seeding_loss,
    compute_layer_group_slices,
    compute_virtual_positions,
    cross_attention_with_causal_limits,
    normalized_levenshtein_similarity,
    select_rem_dreams_by_gradient_norm,
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


def test_nocturnal_sleep_cms_knowledge_seeding_and_rem_dream_selection() -> None:
    """HM-S34-08 & HM-S34-09 (`arXiv:2606.03979v2`): verify CMS chunk periods, KS loss, and 4x REM dream filtering."""
    # 1. CMS ordered chunk periods C^(l) = max_C / f_l -> [1000, 5000, 10000]
    periods = compute_cms_chunk_periods([10.0, 2.0, 1.0], max_chunk_tokens=10000)
    assert periods == [1000, 5000, 10000]

    # 2. Knowledge Seeding (GKD + LTI) positive control: identical teacher & student gives 0 GKD loss and 1.0 LTI reward
    logits = np.array([[2.0, 0.5, -1.0, 0.1], [0.2, 1.8, -0.4, 0.0]], dtype=np.float64)
    tokens_t = [0, 1, 2, 3]
    ks_exact = compute_knowledge_seeding_loss(
        logits, logits, tokens_t, tokens_t, semantic_equivalence=1.0
    )
    assert ks_exact["gkd_loss"] == 0.0
    assert ks_exact["r_abs"] == 1.0
    assert ks_exact["lti_reward"] == 1.0
    assert ks_exact["ks_total_loss"] == 0.0

    # Divergent student has strictly positive GKD and LTI loss
    ks_divergent = compute_knowledge_seeding_loss(
        -logits, logits, [3, 3, 0, 1], tokens_t, semantic_equivalence=0.2
    )
    assert ks_divergent["gkd_loss"] > 0.5
    assert ks_divergent["ks_total_loss"] > 0.5
    assert 0.0 <= normalized_levenshtein_similarity("abcd", "abxd") < 1.0

    # 3. Stage 2 REM Dreaming: m=60 dreams -> Top-12 + 3 random = 15 dreams (4.0x speedup vs SEAL)
    dream_ids = [f"dream_{i:02d}" for i in range(60)]
    grad_norms = [float(i + 1) * 0.15 for i in range(60)]
    injected_experts = [i % 16 for i in range(60)]
    rem = select_rem_dreams_by_gradient_norm(
        dream_ids, grad_norms, injected_experts, top_k=12, random_b=3, seed=42
    )
    assert rem["total_candidates_m"] == 60
    assert rem["selected_count"] == 15
    assert rem["inner_loop_speedup_vs_seal"] == 4.0
    assert rem["mean_selected_grad_norm"] > rem["mean_rejected_grad_norm"]


def test_compacted_latent_nectar_cache_and_wake_to_sleep_bridge() -> None:
    """HM-S34-02 & HM-S34-10: verify AM_U KV compaction on SwarmNectarCache and Wake->Sleep consolidation."""
    from hivemind.swarm.swarm_nectar_cache import TOPIC_TELEPATHY_KV, SwarmNectarCache

    rng = np.random.default_rng(123)
    raw_kv = rng.standard_normal((1, 100, 4, 2, 2, 16))
    compacted = compact_kv_with_attention_matching(raw_kv, target_ratio=5.0)
    assert compacted["original_seq_len"] == 100
    assert compacted["compacted_seq_len"] == 20
    assert compacted["partition_mass_sum"] == 100.0
    assert compacted["wan_2mib_budget_ok"] is True
    assert compacted["compacted_wire_bytes"] < compacted["raw_fp8_bytes"]

    cache = SwarmNectarCache()
    entry = cache.put_telepathy_kv(
        sender_model_id="ibm-granite/granite-3.3-8b-instruct",
        receiver_model_id="HuggingFaceTB/SmolLM3-3B",
        context_key="doc:k2-18b-hycean-spectrum",
        compacted_kv_manifest=compacted,
        discovered_by_bee="scout-granite-8b",
    )
    hit = cache.get_telepathy_kv(
        sender_model_id="ibm-granite/granite-3.3-8b-instruct",
        receiver_model_id="HuggingFaceTB/SmolLM3-3B",
        context_key="doc:k2-18b-hycean-spectrum",
        requesting_bee="worker-smollm3-3b",
    )
    assert hit is not None
    assert hit.nectar_id == entry.nectar_id
    assert hit.hmt_royalties_earned > 0.0

    ev = cache.build_kafka_telepathy_kv_event(hit)
    assert ev["topic"] == TOPIC_TELEPATHY_KV
    assert ev["payload_sha256"] == compacted["payload_sha256"]

    w2s = WakeToSleepConsolidationBridge()
    logits = np.array([[1.5, 0.2, -0.5, 0.0]], dtype=np.float64)
    consolidation = w2s.consolidate_wake_nectar_into_sleep(
        [hit.to_dict()],
        student_log_probs=logits,
        teacher_log_probs=logits,
        student_tokens=[0, 1, 2],
        teacher_tokens=[0, 1, 2],
        dream_ids=[f"d_{i}" for i in range(60)],
        sft_gradient_norms=[float(i + 1) for i in range(60)],
        injected_random_experts=[i % 8 for i in range(60)],
    )
    assert consolidation["wake_episodes_consolidated"] == 1
    assert consolidation["dawn_kv_eviction"]["retained_kv_tokens_at_dawn"] == 0
    assert consolidation["dawn_kv_eviction"]["swat_rms_stats_recalibrated"] is True
    assert consolidation["vram_guards"]["all_guards_passed"] is True


def test_lean4_season34_swat_theorems_and_axioms_coverage() -> None:
    """HM-S34-07: verify Season 34 Lean 4 theorems (Theorems 162-165) have 0 sorry and #print axioms coverage."""
    from pathlib import Path

    lean_path = Path(__file__).resolve().parents[1] / "lean4" / "HiveMindSpec.lean"
    text = lean_path.read_text(encoding="utf-8")
    assert "namespace Season34_SwaTTelepathyAndNocturnalSleep" in text
    expected_theorems = (
        "swat_identical_tokenizer_rope_identity",
        "swat_cross_causal_prefix_no_future_leakage",
        "swat_universal_hub_2n_strictly_below_pairwise",
        "swat_am_compacted_vram_fits_rtx2070_wddm",
    )
    for thm in expected_theorems:
        assert f"theorem {thm}" in text
        assert f"#print axioms HiveMindSpec.Season34_SwaTTelepathyAndNocturnalSleep.{thm}" in text

    s34_block = text.split("namespace Season34_SwaTTelepathyAndNocturnalSleep", 1)[1].split(
        "end Season34_SwaTTelepathyAndNocturnalSleep", 1
    )[0]
    assert "sorry" not in s34_block


def test_gateway_s34_swat_telepathy_endpoints(tmp_path) -> None:
    """HM-S34-04, HM-S34-05 & HM-S34-07: verify gateway /v1/hivemind/bees/swat-telepathy/{status,plan}."""
    from fastapi.testclient import TestClient

    from hivemind.router.gateway import RouterStore, create_router_app

    store = RouterStore(tmp_path / "router.db", b"test-secret-0123456789abcdef0123")
    app = create_router_app(store)
    client = TestClient(app)

    resp_status = client.get("/v1/hivemind/bees/swat-telepathy/status")
    assert resp_status.status_code == 200
    status_data = resp_status.json()
    assert status_data["season"] == "S34"
    assert status_data["hub_summary"]["strictly_fewer_than_pairwise"] is True
    assert len(status_data["lean4_theorems"]) == 4

    resp_plan = client.post(
        "/v1/hivemind/bees/swat-telepathy/plan",
        json={
            "sender_family": "ibm-granite/granite-3.3-8b-instruct",
            "receiver_family": "HuggingFaceTB/SmolLM3-3B",
            "task_category": "code",
            "input_tokens": 28000,
        },
    )
    assert resp_plan.status_code == 200
    plan_data = resp_plan.json()
    assert plan_data["mode"] == "swat_kv_telepathy"
    assert plan_data["compacted_tokens"] == 5600
    assert plan_data["fits_rtx2070_wddm_ceiling"] is True

    resp_ocr = client.post(
        "/v1/hivemind/bees/swat-telepathy/plan",
        json={
            "sender_family": "google/gemma-3-4b-it",
            "receiver_family": "Qwen/Qwen3.5-0.8B",
            "task_category": "docvqa",
            "input_tokens": 4096,
        },
    )
    assert resp_ocr.status_code == 200
    assert resp_ocr.json()["mode"] == "native_vlm_fallback"



