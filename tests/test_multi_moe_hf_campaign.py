"""Unit tests for Operation Polyglot-Singularity-128: Multi-MoE & 8-Domain Hugging Face GPU Campaign."""

from __future__ import annotations

from pathlib import Path

from hivemind.cli.multi_moe_hf_campaign import (
    format_campaign_markdown_summary,
    load_polyglot_8domain_fixture,
    run_multi_moe_hf_campaign,
)


def test_polyglot_8domain_fixture_has_8_zero_leakage_hf_domains() -> None:
    payload, digest = load_polyglot_8domain_fixture()
    assert len(digest) == 64
    domains = payload["domains"]
    assert len(domains) == 8
    super_clusters = {spec["super_cluster_id"] for spec in domains.values()}
    assert super_clusters == {0, 1, 2, 3}


def test_run_multi_moe_hf_campaign_proves_holdout_convergence_and_byzantine_ties(
    tmp_path: Path,
) -> None:
    out_json = tmp_path / "campaign_report.json"
    report = run_multi_moe_hf_campaign(
        contributor_handle="test-campaign-runner",
        preferred_device="cpu",
        steps_per_domain=4,
        domain_limit=4,
        output_path=out_json,
    )
    assert out_json.exists()
    assert report["all_checks_passed"] is True
    assert report["symbolic_compute_reciprocity_points"]["monetary_payment_eur_usd"] == 0.0
    assert len(report["campaign_sha256"]) == 64

    s23 = report["stage2_and_3_multi_domain_lora_and_ef21_cas"]
    assert s23["all_domains_improved_train"] is True
    assert s23["all_domains_improved_holdout"] is True
    assert s23["all_cas_zero_step_transfers_exact"] is True
    assert s23["all_ef21_residuals_conserved"] is True
    assert s23["mean_holdout_nll_reduction_pct"] > 0.0

    s4 = report["stage4_byzantine_ties_merge_vs_naive"]
    assert s4["ties_beats_poisoned_naive"] is True
    assert s4["ties_beats_frozen_baseline"] is True

    md = format_campaign_markdown_summary(report)
    assert "Operation Polyglot-Singularity-128" in md
    assert report["campaign_sha256"] in md
    assert "HiveMind Reflexion Points (HMT)" in md
