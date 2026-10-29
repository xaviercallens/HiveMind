"""Unit and Hypothesis property tests for Season 9 & Season 10 Coordinator Deliverables (`HM-S9-01..08`, `HM-S9-13..14`, `HM-S10-01..02`, `HM-S10-05`, `HM-S10-08`)."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile

from hypothesis import given, settings, strategies as st
import pytest
import torch

from hivemind.boinc.validator import (
    DeterministicSpotCheckValidator,
    ExecutionClassProfile,
    quantize_fixed_point_q16_state,
    verify_fixed_point_q16_digest,
)
from hivemind.cli.community_orchestrator import (
    evaluate_pull_request_for_merge,
    run_hourly_quality_and_experiment_audit,
    touches_scientific_lead_protected_files,
    verify_lean_print_axioms_coverage,
)
from hivemind.cli.multi_moe_hf_campaign import run_multi_moe_hf_campaign
from hivemind.compress.lattice_quantizer import (
    brute_force_decoder_check,
    is_lattice_point,
    nearest_a2,
)
from hivemind.core.btm_moe import FP8PagedKVCache
from hivemind.core.btm_upcycle import (
    LOOPED_INPUT_INJECTION_MODES,
    LearnedDepthPrior,
    LoopedExpertBackbone,
    initialize_domain_branches,
    orthogonal_input_injection,
    upcycle_branches_to_sparse_moe,
)
from hivemind.core.diloco_outer import DiLoCoOuterOptimizer
from hivemind.core.quantization import (
    attach_qlora_adapters,
    build_local_qwen2_backbone,
    extract_lora_state_dict,
)
from hivemind.swarm.decoupled_diloco import (
    ContinuousAsyncLearnerCoordinator,
    LearnerArrival,
    analyze_campaign_rda_and_delta_cosine,
)
from hivemind.swarm.dht_storage import (
    ContentAddressedSwarmStore,
    compute_lease_ttl_s,
)
from hivemind.swarm.diloco_nextgen import (
    CPUOffloadedFlatBucketOuterState,
    ComputeVisibleSparseLoCoCompressor,
    CosineGatedAdamDecayOuterOptimizer,
    FactoredGossipMix1Mix2Coordinator,
)


def _make_tiny_upcycled_moe(seed: int = 7):
    base, _, _ = build_local_qwen2_backbone(
        seed=seed,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        device=torch.device("cpu"),
        use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(
        base_model=base,
        rank_r=2,
        lora_alpha=4,
        gradient_checkpointing=False,
        seed=seed,
    )
    init_lora = extract_lora_state_dict(bundle.model)
    branches = initialize_domain_branches(init_lora, expert_domains=["Code", "Math", "Science", "Systems"])
    moe_model, report = upcycle_branches_to_sparse_moe(
        bundle.model,
        {k: v.lora_state for k, v in branches.items()},
        lora_rank_r=2,
        lora_alpha=4,
        in_place=False,
    )
    return moe_model, report


def test_hm_s9_05_periodic_outer_momentum_restart_nesterov_and_flat_bucket() -> None:
    """HM-S9-05: Periodic outer-momentum restart zeroes velocity at `outer_step % K_restart == 0`."""
    theta0 = {"w": torch.tensor([2.0, -4.0], dtype=torch.float32)}
    worker_deltas = {"w0": {"w": torch.tensor([1.0, -1.0], dtype=torch.float32)}}

    opt = DiLoCoOuterOptimizer(
        outer_lr_eta=0.5,
        outer_momentum_mu=0.9,
        outer_optimizer_mode="nesterov",
        momentum_restart_period=2,
    )
    res1 = opt.step("Code", outer_step=1, theta_prev=theta0, worker_deltas=worker_deltas)
    assert res1.momentum_restarted is False
    assert torch.allclose(res1.velocity_updated["w"], torch.tensor([1.0, -1.0]))

    # Step 2 hits 2 % 2 == 0 -> velocity buffer is restarted before accumulating current grad
    res2 = opt.step("Code", outer_step=2, theta_prev=res1.theta_updated, worker_deltas=worker_deltas)
    assert res2.momentum_restarted is True
    assert torch.allclose(res2.velocity_updated["w"], worker_deltas["w0"]["w"])

    # Also test CPUOffloadedFlatBucketOuterState.outer_nesterov_step with momentum_restart_period
    bucket = CPUOffloadedFlatBucketOuterState(theta0)
    pgrad = {"w": torch.tensor([1.0, 1.0], dtype=torch.float32)}
    _, vel0 = bucket.outer_nesterov_step(pgrad, lr=0.5, momentum=0.9, momentum_restart_period=2)
    _, vel1 = bucket.outer_nesterov_step(pgrad, lr=0.5, momentum=0.9, momentum_restart_period=2)
    assert float(vel1["w"].norm().item()) > float(vel0["w"].norm().item())
    # Step index 2 triggers velocity reset before applying pgrad
    _, vel2 = bucket.outer_nesterov_step(pgrad, lr=0.5, momentum=0.9, momentum_restart_period=2)
    assert torch.allclose(vel2["w"], pgrad["w"])


def test_hm_s9_05_cgad_first_moment_restart_preserves_second_moment() -> None:
    """HM-S9-05: CGAD periodic restart flushes `m_buffers` while keeping `v_buffers` intact."""
    params = {"lora_A": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    grads = {"lora_A": torch.tensor([0.5, -0.5], dtype=torch.float32)}
    cgad = CosineGatedAdamDecayOuterOptimizer(lr=1e-2, alpha=0.2, tau_cut=16.0, momentum_restart_period=2)

    p1, _ = cgad.step_fragments(params, grads)
    v_after_1 = cgad.v_buffers["lora_A"].clone()
    p2, _ = cgad.step_fragments(p1, grads)
    assert cgad.outer_round_count == 2
    # Next step is outer_round_count == 2 -> 2 % 2 == 0 triggers reset_first_moment_buffers()
    p3, _ = cgad.step_fragments(p2, grads)
    assert float(cgad.v_buffers["lora_A"].norm().item()) > float(v_after_1.norm().item())
    assert torch.isfinite(p3["lora_A"]).all()


def test_hm_s9_02_sigma_moe_peri_ln_and_skip_token_flops_report() -> None:
    """HM-S9-02: Sigma-MoE + Peri-LN and empirical Skip-Token FLOP report gated on forward pass."""
    moe_model, _ = _make_tiny_upcycled_moe(seed=11)
    moe_model.set_local_expert_subset([0, 1])
    moe_model.set_routing_mode("sigma_moe")
    moe_model.set_peri_ln(True)
    moe_model.reset_routing_telemetry()

    # Before forward pass, realized_from_forward_pass must be False and saved fraction must be 0.0
    pre_report = moe_model.skip_token_flops_report()
    assert pre_report["realized_from_forward_pass"] is False
    assert pre_report["measured_expert_flops_saved_fraction"] == 0.0
    assert pre_report["peri_ln_enabled"] is True

    # Run a real forward pass
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    out = moe_model(input_ids=tokens, labels=tokens)
    assert torch.isfinite(out.loss).all()

    post_report = moe_model.skip_token_flops_report()
    assert post_report["realized_from_forward_pass"] is True
    assert post_report["total_routed_assignments"] > 0
    assert 0.0 <= post_report["mean_measured_ghost_fraction"] <= 1.0
    assert abs(
        post_report["measured_expert_flops_multiplier"]
        + post_report["measured_expert_flops_saved_fraction"]
        - 1.0
    ) < 1e-6


@given(
    h_vals=st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False), min_size=8, max_size=8),
    u_vals=st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False), min_size=8, max_size=8),
)
@settings(max_examples=40, deadline=None)
def test_hm_s9_03_orthogonal_input_injection_property(h_vals: list[float], u_vals: list[float]) -> None:
    """HM-S9-03 Property: `orthogonal_input_injection` removes the component along `h_t` (Pythagorean norm)."""
    h = torch.tensor(h_vals, dtype=torch.float32).unsqueeze(0)
    u = torch.tensor(u_vals, dtype=torch.float32).unsqueeze(0)
    injected = orthogonal_input_injection(h, u)
    u_perp = injected - h
    h_norm_sq = float((h * h).sum().item())
    if h_norm_sq > 1e-4:
        dot_val = float((u_perp * h).sum().item())
        assert abs(dot_val) <= 1e-3 * max(1.0, h_norm_sq)
        lhs = float((injected * injected).sum().item())
        rhs = h_norm_sq + float((u_perp * u_perp).sum().item())
        assert abs(lhs - rhs) <= 1e-2 * max(1.0, rhs)


def test_hm_s9_03_looped_backbone_ortho_depth_prior_and_terminal_kv() -> None:
    """HM-S9-03: LoopedExpertBackbone supports `orthogonal` injection, `LearnedDepthPrior`, and terminal-KV sharing."""
    assert "orthogonal" in LOOPED_INPUT_INJECTION_MODES
    moe_model, _ = _make_tiny_upcycled_moe(seed=19)
    looped = LoopedExpertBackbone(
        moe_model,
        num_prelude_layers=1,
        num_coda_layers=1,
        loop_steps=3,
        per_step_lora_rank=2,
        input_injection="orthogonal",
        use_learned_depth_prior=True,
        depth_entropy_coef=0.02,
        depth_budget_coef=0.05,
        target_depth_budget=2.0,
    )
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    out = looped(input_ids=ids, labels=ids)
    assert out.loss is not None and bool(torch.isfinite(out.loss).item())
    assert looped.last_depth_prior_telemetry is not None
    assert 1.0 <= looped.last_depth_prior_telemetry["expected_depth_steps"] <= 3.0

    kv_rep = looped.evaluate_terminal_kv_sharing(input_ids=ids)
    assert kv_rep["loop_steps"] == 3
    assert abs(kv_rep["core_kv_cache_saved_fraction"] - (2.0 / 3.0)) < 1e-5
    assert -1.0 <= kv_rep["terminal_vs_penultimate_cosine"] <= 1.0


@given(
    logits_flat=st.lists(
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=16,
        max_size=16,
    )
)
@settings(max_examples=30, deadline=None)
def test_hm_s9_04_jensen_shannon_disagreement_properties(logits_flat: list[float]) -> None:
    """HM-S9-04 Property: Jensen-Shannon disagreement is 0 on identical distributions and non-negative."""
    dist_a = torch.tensor(logits_flat[:8], dtype=torch.float32).reshape(2, 4)
    dist_b = torch.tensor(logits_flat[8:], dtype=torch.float32).reshape(2, 4)
    js_same = FactoredGossipMix1Mix2Coordinator.jensen_shannon_router_disagreement(
        [dist_a, dist_a], inputs_are_logits=True
    )
    js_diff = FactoredGossipMix1Mix2Coordinator.jensen_shannon_router_disagreement(
        [dist_a, dist_b], inputs_are_logits=True
    )
    js_sym = FactoredGossipMix1Mix2Coordinator.jensen_shannon_router_disagreement(
        [dist_b, dist_a], inputs_are_logits=True
    )
    assert abs(js_same) < 1e-9
    assert js_diff >= 0.0
    assert abs(js_diff - js_sym) < 1e-9


def test_hm_s9_04_factored_gossip_mix1_mix2_gating() -> None:
    """HM-S9-04: `FactoredGossipMix1Mix2Coordinator` skips blocking Mix2 when JS < threshold and runs Mix2 when JS >= threshold."""
    master = {
        "backbone.layer0.mlp.up_proj": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "router.wg.weight": torch.tensor([0.5, -0.5], dtype=torch.float32),
    }
    workers = [
        {
            "backbone.layer0.mlp.up_proj": torch.tensor([0.8, 1.8], dtype=torch.float32),
            "router.wg.weight": torch.tensor([0.3, -0.3], dtype=torch.float32),
        },
        {
            "backbone.layer0.mlp.up_proj": torch.tensor([0.6, 1.6], dtype=torch.float32),
            "router.wg.weight": torch.tensor([0.1, -0.1], dtype=torch.float32),
        },
    ]
    coord = FactoredGossipMix1Mix2Coordinator(alpha_mix1=0.5, js_threshold=0.05)

    # Identical router distributions -> JS = 0 < 0.05 -> Mix2 skipped, blocking bytes saved
    p_identical = [torch.tensor([[0.7, 0.3]]), torch.tensor([[0.7, 0.3]])]
    _, rep_low = coord.step_factored_gossip(master, workers, worker_router_distributions=p_identical)
    assert rep_low["mix2_triggered"] is False
    assert rep_low["mix2_blocking_wire_bytes"] == 0
    assert rep_low["saved_blocking_wire_bytes"] > 0

    # Divergent router distributions -> JS > 0.05 -> Mix2 triggered on router.wg.weight
    p_divergent = [torch.tensor([[0.99, 0.01]]), torch.tensor([[0.01, 0.99]])]
    _, rep_high = coord.step_factored_gossip(master, workers, worker_router_distributions=p_divergent)
    assert rep_high["mix2_triggered"] is True
    assert rep_high["mix2_fragment_count"] == 1
    assert rep_high["mix1_fragment_count"] == 1


@given(
    alpha=st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    t_step=st.floats(min_value=0.05, max_value=30.0, allow_nan=False, allow_infinity=False),
    rtt=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    commit_flags=st.lists(st.booleans(), min_size=1, max_size=12),
)
@settings(max_examples=35, deadline=None)
def test_hm_s9_06_heartbeat_data_leasing_conservation_property(
    alpha: float,
    t_step: float,
    rtt: float,
    commit_flags: list[bool],
) -> None:
    """HM-S9-06 Property: `committed == granted - reclaimed` with 0 duplicate commits across arbitrary churn."""
    ttl = compute_lease_ttl_s(alpha=alpha, t_step_s=t_step, rtt_s=rtt)
    assert ttl >= t_step + rtt

    with tempfile.TemporaryDirectory() as td:
        store = ContentAddressedSwarmStore(Path(td) / "cas")
        for idx, should_commit in enumerate(commit_flags):
            cid = f"cid://shard/{idx}"
            lease = store.grant_shard_lease(
                shard_cid=cid,
                worker_id=f"worker-{idx}",
                alpha=alpha,
                t_step_s=t_step,
                rtt_s=rtt,
                now_s=10.0,
            )
            if should_commit:
                store.heartbeat_shard_lease(lease.lease_id, now_s=10.0 + 0.5 * ttl)
                store.commit_shard_lease(
                    lease.lease_id, worker_id=f"worker-{idx}", now_s=10.0 + ttl
                )
                # Attempting to re-lease an already committed shard must raise ValueError
                with pytest.raises(ValueError):
                    store.grant_shard_lease(shard_cid=cid, worker_id="worker-dup", now_s=20.0 + 2 * ttl)

        store.reclaim_expired_leases(now_s=10.0 + 2.0 * ttl + 1.0)
        summary = store.lease_ledger_summary()
        assert summary["conservation_holds"] is True
        assert summary["duplicate_commits"] == 0
        assert summary["active_count"] == 0
        assert summary["committed_count"] == summary["total_granted"] - summary["reclaimed_count"]


def test_hm_s10_01_compute_visible_compressor_with_e8_z_and_sign_codecs() -> None:
    """HM-S10-01: `ComputeVisibleSparseLoCoCompressor` supports E8, Z, and sign codecs with per-fragment rate logging."""
    torch.manual_seed(42)
    delta = torch.randn(16, 16, dtype=torch.float32) * 0.1
    ref = torch.ones(16, 16, dtype=torch.float32)

    for codec in ("E8", "Z", "sign"):
        comp = ComputeVisibleSparseLoCoCompressor(
            codec=codec,
            target_bits_per_dim=2.0,
            error_feedback=True,
            adapt_scale_per_round=True,
        )
        recon, stats = comp.compress_and_reconstruct(delta, reference_param=ref, fragment_name="layer0.lora_A")
        assert recon.shape == delta.shape
        assert stats["codec"] == codec
        assert stats["measured_rate_bits_per_dim"] > 0.0
        assert 0.0 <= stats["relative_mse"] < 1.0
        assert len(comp.fragment_rate_history["layer0.lora_A"]) == 1


def test_hm_s10_02_orchestrator_lean_axioms_gate_and_scientific_lead_pr_guard() -> None:
    """HM-S10-02 & Brief §3: Orchestrator enforces `#print axioms` coverage and blocks auto-merging `paper/` PRs."""
    complete_lean = "theorem t1 : 1 = 1 := rfl\n#print axioms t1\n"
    incomplete_lean = "theorem t1 : 1 = 1 := rfl\ntheorem t2 : 2 = 2 := rfl\n#print axioms t1\n"
    cov_ok = verify_lean_print_axioms_coverage(complete_lean)
    cov_bad = verify_lean_print_axioms_coverage(incomplete_lean)
    assert cov_ok["all_theorems_have_print_axioms"] is True
    assert cov_bad["all_theorems_have_print_axioms"] is False
    assert cov_bad["unprinted_theorems"] == ["t2"]

    # Audit the real repo's HiveMindSpec.lean (149 theorems after the Season25_Wave2 section)
    audit = run_hourly_quality_and_experiment_audit()
    assert audit["lean4_theorems_verified"] == 149
    assert audit["lean4_print_axioms_lines"] == 149
    assert audit["lean4_all_theorems_have_print_axioms"] is True
    assert audit["lean4_sorry_count"] == 0

    # PR touching paper/ or CITATION.cff must never be auto-merged by the engineering orchestrator
    paper_pr = {
        "number": 999,
        "title": "Update paper numbers",
        "body": "Updates paper table",
        "author": "contributor",
        "isDraft": False,
        "mergeable": "MERGEABLE",
        "files": [
            {
                "filename": "paper/hivemind_foundation_article.tex",
                "additions": 10,
                "deletions": 2,
                "patch": "+def valid_helper(x):\n+    return x + 1\n",
            }
        ],
    }
    decision = evaluate_pull_request_for_merge(paper_pr)
    assert decision["can_merge"] is False
    assert decision["verdict"] == "HOLD_FOR_SCIENTIFIC_LEAD_REVIEW"
    assert touches_scientific_lead_protected_files(paper_pr["files"]) == [
        "paper/hivemind_foundation_article.tex"
    ]


def test_hm_s9_01_campaign_wires_decoupled_quorum_syncer_and_stage7() -> None:
    """HM-S9-01: `run_multi_moe_hf_campaign` wires `DecoupledDiLoCoQuorumSyncer` and Stage 7 checks."""
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "campaign_s9_10.json"
        report = run_multi_moe_hf_campaign(
            contributor_handle="test-coordinator",
            preferred_device="cpu",
            steps_per_domain=4,
            domain_limit=4,
            output_path=out_path,
            min_quorum_k=3,
            grace_margin_gamma=0.5,
            quorum_merge_mode="rda",
        )
        assert report["all_checks_passed"] is True
        s7 = report["stage7_season9_10_quorum_and_lattice"]
        assert s7["hm_s9_01_quorum_reached"] is True
        assert s7["hm_s9_01_num_admitted"] >= 3
        assert s7["hm_s9_05_momentum_restarted_on_step2"] is True
        assert s7["hm_s9_06_cas_lease_ledger"]["conservation_holds"] is True
        assert s7["hm_s10_01_e8_lattice_rate_bits_per_dim"] > 0.0
        assert s7["hm_s9_13_continuous_async_learner"]["token_conservation_holds"] is True
        assert s7["hm_s9_14_campaign_rda_and_delta_cosine"]["rda_to_direct_norm_ratio_rt"] > 0.0
        assert s7["hm_s10_05_kv_cache_codec_benchmark"]["controls_all_passed"] is True
        assert s7["hm_s10_08_execution_class_spot_check"]["same_class_passed"] is True
        assert s7["hm_s10_08_execution_class_spot_check"]["cross_dtype_rejected"] is True
        assert s7["hm_s10_08_execution_class_spot_check"]["fixed_point_q16_exact"] is True


def test_hm_s10_05_a2_hexagonal_decoder_brute_force_and_kv_cache_codecs() -> None:
    """HM-S10-05: `nearest_a2` exact Conway-Sloane decoder passes brute-force check and `FP8PagedKVCache` supports 5 codecs."""
    bf = brute_force_decoder_check("A2", n_queries=1500, radius=3.5, seed=17)
    assert bf["passed"] is True
    assert bf["mismatches"] == 0

    torch.manual_seed(19)
    pts = nearest_a2((torch.randn(64, 2, dtype=torch.float32) * 2.5).numpy())
    assert bool(is_lattice_point("A2", pts).all()) is True

    k_tensor = torch.randn(2, 4, 8, 16, dtype=torch.float32) * 0.6
    v_tensor = torch.randn(2, 4, 8, 16, dtype=torch.float32) * 0.6

    for codec in FP8PagedKVCache.VALID_CODECS:
        cache = FP8PagedKVCache(page_size_tokens=16, max_pages=16, codec=codec)
        cache.write_kv_page(0, 0, k_tensor, v_tensor)
        k_hat, v_hat = cache.read_kv_page(0, 0, target_dtype=torch.float32)
        assert k_hat.shape == k_tensor.shape
        assert v_hat.shape == v_tensor.shape
        rel_mse = float(
            ((k_hat - k_tensor).pow(2).sum() + (v_hat - v_tensor).pow(2).sum()).item()
            / (k_tensor.pow(2).sum() + v_tensor.pow(2).sum()).clamp_min(1e-12).item()
        )
        assert 0.0 <= rel_mse < 0.35

    bench = FP8PagedKVCache.benchmark_kv_page_codecs(k_tensor, v_tensor, target_bits_per_dim=4.0)
    assert bench["positive_control_scalar_2bit_worse"] is True
    assert bench["negative_control_a2_8bit_better"] is True
    assert bench["controls_all_passed"] is True
    assert set(bench["codecs"].keys()) == set(FP8PagedKVCache.VALID_CODECS)


def test_hm_s10_08_execution_class_validator_and_two_statistic_spot_check() -> None:
    """HM-S10-08: Execution-class matching refuses cross-dtype pairs, two-statistic rule separates honest vs tampered, and Q16 digest verifies."""
    validator = DeterministicSpotCheckValidator(tolerance_eps=1e-4)
    prover_fp32 = ExecutionClassProfile(accelerator_class="cpu_x86_64", dtype="fp32", thread_count=4)
    val_bf16 = ExecutionClassProfile(accelerator_class="cpu_x86_64", dtype="bf16", thread_count=4)
    val_fp32_t1 = ExecutionClassProfile(accelerator_class="cpu_x86_64", dtype="fp32", thread_count=1)
    val_fp32_t4 = ExecutionClassProfile(accelerator_class="cpu_x86_64", dtype="fp32", thread_count=4)

    # Cross-dtype only must return None
    assert validator.assign_matching_validator(prover_fp32, [val_bf16]) is None
    # Exact match preferred over cross-thread match
    matched = validator.assign_matching_validator(prover_fp32, [val_bf16, val_fp32_t1, val_fp32_t4])
    assert matched is not None
    assert matched.thread_count == 4

    base_state = {f"layer.{i}.lora": torch.ones(8, 8, dtype=torch.float32) * 0.25 for i in range(10)}
    # Honest cross-thread drift: only 1 of 10 tensors (10%) exceeds 1e-4, with linf = 2.2e-4 <= 3.5e-4
    honest_cross_thread = {k: v.clone() for k, v in base_state.items()}
    honest_cross_thread["layer.0.lora"] = honest_cross_thread["layer.0.lora"] + 2.2e-4

    res_honest = validator.evaluate_two_statistic_replay(
        base_state,
        honest_cross_thread,
        worker_class=prover_fp32,
        validator_class=val_fp32_t1,
    )
    assert res_honest.passed is True
    assert res_honest.dtype_matched is True
    assert res_honest.tensor_fraction_above_eps <= 0.10

    # Tampered state: 5 of 10 tensors (50%) shifted by 2.0e-4 -> exceeds max_tensor_fraction_above_eps (10%) even though linf <= 3.5e-4
    tampered_state = {k: v.clone() for k, v in base_state.items()}
    for i in range(5):
        tampered_state[f"layer.{i}.lora"] = tampered_state[f"layer.{i}.lora"] + 2.0e-4
    res_tampered = validator.evaluate_two_statistic_replay(
        base_state,
        tampered_state,
        worker_class=prover_fp32,
        validator_class=val_fp32_t1,
    )
    assert res_tampered.passed is False
    assert "REJECTED" in res_tampered.reason

    # Cross-dtype assignment must be rejected outright
    res_cross_dtype = validator.evaluate_two_statistic_replay(
        base_state,
        base_state,
        worker_class=prover_fp32,
        validator_class=val_bf16,
    )
    assert res_cross_dtype.passed is False
    assert res_cross_dtype.dtype_matched is False
    assert "DTYPE_CLASS_MISMATCH" in res_cross_dtype.reason

    # Fixed-point Q16 prototype deterministic digest across sub-step noise (< 1/(4*65536))
    noisy_sub_lsb = {k: v + 1e-7 for k, v in base_state.items()}
    q16_check = verify_fixed_point_q16_digest(base_state, noisy_sub_lsb, frac_bits=16)
    assert q16_check["exact_digest_match"] is True
    assert q16_check["max_ulp_diff"] == 0
    q_tensors = quantize_fixed_point_q16_state(base_state, frac_bits=16)
    assert len(q16_check["digest_a"]) == 64
    assert q_tensors["layer.0.lora"].dtype == torch.int64


@given(
    token_counts=st.lists(st.integers(min_value=16, max_value=2048), min_size=4, max_size=8),
)
@settings(max_examples=25, deadline=None)
def test_hm_s9_13_continuous_async_learner_straggler_modes_and_conservation(
    token_counts: list[int],
) -> None:
    """HM-S9-13: `ContinuousAsyncLearnerCoordinator` preserves 100% of tokens under deferred and eager straggler modes."""
    num_learners = len(token_counts)
    min_k = max(2, num_learners - 2)
    total_submitted_r0 = sum(token_counts)

    for mode in ("drop", "deferred_token_weighted", "eager_one_step"):
        coord = ContinuousAsyncLearnerCoordinator(
            num_learners_m=num_learners,
            min_quorum_k=min_k,
            straggler_mode=mode,
            merge_mode="rda",
        )
        # First min_k learners arrive in round 0 before close_quorum_round
        for i in range(min_k):
            coord.submit_learner_delta(
                learner_id=f"learner-{i}",
                origin_round=0,
                delta=torch.full((4, 4), float(i + 1), dtype=torch.float32),
                tokens_since_sync=token_counts[i],
                steps_since_sync=4,
                arrival_time_s=0.01 * (i + 1),
            )
        p_init = torch.zeros((4, 4), dtype=torch.float32)
        p_after_r0, rep_r0 = coord.close_quorum_round(p_init, outer_lr=0.5, grace_window_s=0.0)
        assert rep_r0["token_conservation_holds"] is True

        # Remaining stragglers from origin_round=0 arrive during round 1
        for i in range(min_k, num_learners):
            coord.submit_learner_delta(
                learner_id=f"learner-{i}",
                origin_round=0,
                delta=torch.full((4, 4), float(i + 1), dtype=torch.float32),
                tokens_since_sync=token_counts[i],
                steps_since_sync=4,
                arrival_time_s=0.20 + 0.01 * i,
            )
        # Plus min_k fresh learners in round 1
        for i in range(min_k):
            coord.submit_learner_delta(
                learner_id=f"learner-{i}",
                origin_round=1,
                delta=torch.ones((4, 4), dtype=torch.float32),
                tokens_since_sync=100,
                steps_since_sync=4,
                arrival_time_s=0.01 * (i + 1),
            )
        _, rep_r1 = coord.close_quorum_round(p_after_r0, outer_lr=0.5, grace_window_s=1.0)
        assert rep_r1["token_conservation_holds"] is True
        if mode in ("deferred_token_weighted", "eager_one_step"):
            assert rep_r1["dropped_tokens"] == 0
            assert rep_r1["merged_tokens"] == total_submitted_r0 + min_k * 100
        else:
            assert rep_r1["dropped_tokens"] == sum(token_counts[min_k:])


def test_hm_s9_14_campaign_rda_and_delta_cosine_analysis() -> None:
    """HM-S9-14: `analyze_campaign_rda_and_delta_cosine` matches theoretical common-norm multiplier 1/sqrt(1/M + (M-1)/M * c)."""
    base_ortho = {"w": torch.zeros(4, dtype=torch.float32)}
    # Construct 4 orthogonal equal-norm learner deltas -> pairwise cosine c_bar = 0.0 -> theoretical multiplier = sqrt(4) = 2.0
    branches_ortho = {}
    for i in range(4):
        vec = torch.zeros(4, dtype=torch.float32)
        vec[i] = -2.0
        branches_ortho[f"dom_{i}"] = {"w": vec}

    rep_ortho = analyze_campaign_rda_and_delta_cosine(base_ortho, branches_ortho)
    assert abs(float(rep_ortho["mean_pairwise_delta_cosine"]) - 0.0) < 1e-5
    assert abs(float(rep_ortho["rda_to_direct_norm_ratio_rt"]) - 2.0) < 1e-4
    assert abs(float(rep_ortho["theoretical_common_norm_ratio"]) - 2.0) < 1e-4
    assert rep_ortho["low_cosine_regime_active"] is True

    # Construct 4 identical learner deltas -> pairwise cosine c_bar = 1.0 -> theoretical multiplier = 1.0
    branches_ident = {f"dom_{i}": {"w": -torch.ones(4, dtype=torch.float32)} for i in range(4)}
    rep_ident = analyze_campaign_rda_and_delta_cosine(base_ortho, branches_ident)
    assert abs(float(rep_ident["mean_pairwise_delta_cosine"]) - 1.0) < 1e-5
    assert abs(float(rep_ident["rda_to_direct_norm_ratio_rt"]) - 1.0) < 1e-4
    assert rep_ident["low_cosine_regime_active"] is False

    # Construct 4 equal-norm deltas with pairwise cosine c_bar = 0.25: v_i = a * u_0 + b * e_i (i=1..4)
    base_c25 = {"w": torch.zeros(5, dtype=torch.float32)}
    branches_c25 = {}
    for i in range(4):
        vec = torch.zeros(5, dtype=torch.float32)
        vec[0] = -0.5
        vec[i + 1] = -math.sqrt(0.75)
        branches_c25[f"dom_{i}"] = {"w": vec}
    rep_c25 = analyze_campaign_rda_and_delta_cosine(base_c25, branches_c25)
    expected_mult = 1.0 / math.sqrt(0.25 + 0.75 * 0.25)
    assert abs(float(rep_c25["mean_pairwise_delta_cosine"]) - 0.25) < 1e-5
    assert abs(float(rep_c25["rda_to_direct_norm_ratio_rt"]) - expected_mult) < 1e-4
    assert abs(float(rep_c25["theoretical_common_norm_ratio"]) - expected_mult) < 1e-4


