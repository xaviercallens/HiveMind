"""Unit tests for Season 9 & Season 10 GPU + HiveMind Swarm Contributor Campaigns."""

from __future__ import annotations

from pathlib import Path

from hivemind.cli.season9_10_gpu_swarm_campaigns import (
    GPU_CONTRIBUTOR_ROSTER,
    SEASON9_10_GPU_CAMPAIGN_CATALOG,
    format_season9_10_gpu_campaigns_markdown,
    run_season9_10_gpu_swarm_campaigns,
)


def test_season9_10_gpu_swarm_campaigns_end_to_end(tmp_path: Path) -> None:
    """Verify all 4 Season 9 & Season 10 GPU & HiveMind Swarm campaigns pass and produce signed telemetry."""
    out_file = tmp_path / "s9_10_gpu_campaigns.json"
    report = run_season9_10_gpu_swarm_campaigns(
        contributor_handle="clark719664",
        preferred_device="cpu",
        num_domains=4,
        inner_steps=2,
        seed=21,
        output_path=out_file,
    )
    assert out_file.exists()
    assert report["all_campaigns_passed"] is True
    assert len(report["campaign_suite_sha256"]) == 64
    assert report["non_monetary_charter"]["financial_payment_usd_eur"] == 0.0
    assert report["non_monetary_charter"]["no_money_involved"] is True

    c1 = report["campaign_1_s9_quorum_and_rda"]
    assert c1["quorum_reached"] is True
    assert c1["momentum_restarted_on_step2"] is True
    assert c1["straggler_modes"]["deferred_token_weighted"]["dropped_tokens"] == 0
    assert c1["straggler_modes"]["eager_one_step"]["dropped_tokens"] == 0
    assert c1["straggler_modes"]["drop"]["dropped_tokens"] > 0
    assert c1["rda_delta_cosine"]["rda_to_direct_norm_ratio_rt"] > 0.0

    c2 = report["campaign_2_s9_sigma_and_looped_moe"]
    assert c2["skip_token_report"]["realized_from_forward_pass"] is True
    assert c2["router_128_gating_reduction_x"] == 8.0

    c3 = report["campaign_3_s10_lattice_wire_and_kv"]
    assert c3["a2_decoder_brute_force_passed"] is True
    assert c3["e8_decoder_brute_force_passed"] is True
    assert c3["kv_cache_codec_benchmark"]["controls_all_passed"] is True

    c4 = report["campaign_4_s10_spotcheck_leasing_and_gossip"]
    assert c4["exact_class_replay_passed"] is True
    assert c4["cross_thread_honest_replay_passed"] is True
    assert c4["tampered_replay_rejected"] is True
    assert c4["cross_dtype_replay_rejected"] is True
    assert c4["fixed_point_q16_exact_digest"] is True
    assert c4["cas_lease_ledger"]["conservation_holds"] is True

    md = format_season9_10_gpu_campaigns_markdown(report)
    assert "CAMP-S9-E5-QUORUM-RDA-GPU" in md
    assert "CAMP-S9-E7-SIGMA-LOOPED-MOE-GPU" in md
    assert "CAMP-S10-E11-LATTICE-KV-WIRE-GPU" in md
    assert "CAMP-S10-E12-HETERO-SPOTCHECK-WAN-GPU" in md
    assert "HiveMind Reflexion Points (HMT)" in md
    assert len(GPU_CONTRIBUTOR_ROSTER) >= 4
    assert len(SEASON9_10_GPU_CAMPAIGN_CATALOG) == 4
