# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The public owner-fleet-status payload must not present configured constants as live telemetry."""

from __future__ import annotations

from hivemind.swarm.rusty_sundials_scientific_bee import OwnerFleetCoServingOrchestrator


def test_foreground_flag_cannot_change_reported_state() -> None:
    quiet = OwnerFleetCoServingOrchestrator.public_status(foreground_gaming=False)
    gaming = OwnerFleetCoServingOrchestrator.public_status(foreground_gaming=True)
    assert quiet == gaming
    assert quiet["measured"] is False


def test_no_node_claims_models_or_memory_in_use() -> None:
    payload = OwnerFleetCoServingOrchestrator.public_status()
    for node in ("rtx_2070_local_8gb", "gcp_tesla_t4_16gb"):
        assert payload[node]["serving_state"] == "NOT_MEASURED"
        assert not {"live_models", "serving_active", "allocated_vram_mib", "free_vram_mib"} & set(payload[node])


def test_unrun_experiments_are_reported_as_not_run() -> None:
    payload = OwnerFleetCoServingOrchestrator.public_status()
    assert payload["latest_nocturnal_distillation"]["status"] == "NOT_RUN"
    assert payload["s1b_cross_cpu_reproducibility"]["status"] == "NOT_RUN_CROSS_HOST"
    assert payload["distillation_runs"] == 0
