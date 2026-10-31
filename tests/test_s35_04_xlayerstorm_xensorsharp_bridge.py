# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for HM-S35-04: xLayerStoRm PCIe MoE Streaming x xensorSharp Native Worker x TurboQuant."""

from __future__ import annotations

from hivemind.swarm.s35_04_xlayerstorm_xensorsharp_bridge import (
    CITIZEN_VRAM_CAP_MIB,
    build_native_worker_fleet_manifests,
    compute_xlayerstorm_turboquant_schedule,
)


def test_xlayerstorm_turboquant_schedule_respects_citizen_vram_ceiling() -> None:
    plan = compute_xlayerstorm_turboquant_schedule()
    assert plan.within_citizen_vram_cap is True
    assert plan.peak_active_vram_mib <= CITIZEN_VRAM_CAP_MIB
    assert plan.vram_headroom_mib > 200.0
    assert plan.overlap_efficiency_pct >= 90.0


def test_native_worker_fleet_manifests_require_zero_wsl_or_python() -> None:
    manifests = build_native_worker_fleet_manifests()
    assert len(manifests) == 4
    for m in manifests:
        assert m.requires_python_or_wsl is False
        assert m.active_vram_cap_mib <= CITIZEN_VRAM_CAP_MIB
        assert m.steam_preemption_ms < 1500.0
