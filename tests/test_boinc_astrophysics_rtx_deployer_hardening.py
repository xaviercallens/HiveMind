# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control regression tests for BOINC astrophysics campaigns and RTX BOINC/Steam deployers."""

from __future__ import annotations

from pathlib import Path

import pytest

from hivemind.boinc.astrophysics_24h_campaign import (
    _load_6domain_astrophysics_cosmology_examples,
    annotate_campaign_report_provenance,
    annotate_discovery_catalog_provenance,
    create_signed_6domain_parquet_shard,
    run_24h_astrophysics_gpu_campaign,
)
from hivemind.boinc.local_astrophysics_boinc import (
    _load_astrophysics_examples,
    create_signed_astrophysics_parquet_shard,
    run_local_boinc_astrophysics_campaign,
)
from hivemind.boinc.rtx_boinc_deployer import (
    RTXBoincServer,
    RTXWorkunitLeaseRequest,
    RTXWorkunitSubmitRequest,
    detect_rtx_boinc_profile,
)
from hivemind.game.rtx_steam_tool_deployer import (
    deploy_free_steam_rtx_tool_v1,
    run_intense_rtx_stress_suite,
)


def test_local_astrophysics_boinc_rejects_invalid_parameters(tmp_path: Path) -> None:
    samples = _load_astrophysics_examples(
        "Kepler_Exoplanet_Transit_Classifier",
        "mzrk36/exoplanets-unified-dataset",
        num_samples=4,
    )
    assert len(samples) == 4

    with pytest.raises(ValueError, match="branch_name"):
        _load_astrophysics_examples("   ", "mzrk36/exoplanets-unified-dataset", num_samples=4)
    with pytest.raises(ValueError, match="dataset_id"):
        _load_astrophysics_examples("Kepler_Exoplanet_Transit_Classifier", "", num_samples=4)
    with pytest.raises(ValueError, match="num_samples"):
        _load_astrophysics_examples("Kepler_Exoplanet_Transit_Classifier", "ds", num_samples=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_samples"):
        _load_astrophysics_examples("Kepler_Exoplanet_Transit_Classifier", "ds", num_samples=0)

    with pytest.raises(ValueError, match="rng_seed"):
        create_signed_astrophysics_parquet_shard(tmp_path / "s.parquet", "b", "d", rng_seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_sequences"):
        create_signed_astrophysics_parquet_shard(tmp_path / "s.parquet", "b", "d", rng_seed=1, num_sequences=0)
    with pytest.raises(ValueError, match="seq_len"):
        create_signed_astrophysics_parquet_shard(tmp_path / "s.parquet", "b", "d", rng_seed=1, seq_len=2)
    with pytest.raises(ValueError, match="vocab_size"):
        create_signed_astrophysics_parquet_shard(tmp_path / "s.parquet", "b", "d", rng_seed=1, vocab_size=32)
    with pytest.raises(ValueError, match="secret_key"):
        create_signed_astrophysics_parquet_shard(tmp_path / "s.parquet", "b", "d", rng_seed=1, secret_key=b"short")

    with pytest.raises(ValueError, match="inner_steps_H"):
        run_local_boinc_astrophysics_campaign(inner_steps_H=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="spot_check_interval_s"):
        run_local_boinc_astrophysics_campaign(inner_steps_H=8, spot_check_interval_s=12)


def test_rtx_boinc_deployer_rejects_nan_losses_and_invalid_bounds(tmp_path: Path) -> None:
    prof = detect_rtx_boinc_profile(override_vram_total_mib=8192, override_vram_free_mib=6800)
    assert prof.plan_class == "cuda_hivemind"

    with pytest.raises(ValueError, match="override_vram_total_mib"):
        detect_rtx_boinc_profile(override_vram_total_mib=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="override_vram_free_mib"):
        detect_rtx_boinc_profile(override_vram_total_mib=8192, override_vram_free_mib=-10)
    with pytest.raises(ValueError, match="cannot exceed"):
        detect_rtx_boinc_profile(override_vram_total_mib=8192, override_vram_free_mib=12000)

    server = RTXBoincServer(db_path=tmp_path / "rtx.db", boinc_ledger_dir=tmp_path / "ledger")
    with pytest.raises(ValueError, match="wu_id"):
        server.sign_workunit("   ", "code", 42)
    with pytest.raises(ValueError, match="rng_seed"):
        server.sign_workunit("wu-1", "code", True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="limit"):
        server.status_summary(limit=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="limit"):
        server.status_summary(limit=0)

    leased = server.lease_workunit(RTXWorkunitLeaseRequest(handle="xavier", node_id="rtx-01"))
    nan_sub = RTXWorkunitSubmitRequest(
        wu_id=leased["wu_id"],
        handle="xavier",
        node_id="rtx-01",
        gpu_name="NVIDIA GeForce RTX 2070",
        cuda_verified=True,
        eval_loss_before=2.5,
        eval_loss_after=float("nan"),
        state_sha256=["a" * 64],
        hmac_signature=leased["hmac_signature"],
    )
    verdict = server.submit_workunit(nan_sub)
    assert verdict["valid"] is False
    assert "must be finite" in verdict["reason"]


def test_astrophysics_24h_and_rtx_steam_deployer_reject_invalid_inputs(tmp_path: Path) -> None:
    examples = _load_6domain_astrophysics_cosmology_examples(
        "SDSS_DESI_Cosmology_Redshift_BAO_Analyzer",
        "sdss/cosmology_quasar_lya_bao_dr16",
        num_samples=3,
    )
    assert len(examples) == 3

    with pytest.raises(ValueError, match="num_samples"):
        _load_6domain_astrophysics_cosmology_examples("b", "d", num_samples=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rng_seed"):
        create_signed_6domain_parquet_shard(tmp_path / "s6.parquet", "b", "d", rng_seed=-1)
    with pytest.raises(ValueError, match="catalog"):
        annotate_discovery_catalog_provenance(["not-a-dict"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="report"):
        annotate_campaign_report_provenance(["not-a-dict"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_cycles"):
        run_24h_astrophysics_gpu_campaign(num_cycles=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="spot_check_interval_s"):
        run_24h_astrophysics_gpu_campaign(inner_steps_per_cycle=6, spot_check_interval_s=10)

    with pytest.raises(ValueError, match="vram_cycles"):
        run_intense_rtx_stress_suite(vram_cycles=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_hotplug_experts"):
        run_intense_rtx_stress_suite(num_hotplug_experts=1)
    with pytest.raises(ValueError, match="stress_report"):
        deploy_free_steam_rtx_tool_v1(stress_report={"incomplete": True})
