# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for HM-S32-01 Symplectic Velocity-Verlet KEPLER-HM-442b & Google Antigravity SDK Co-Pilot."""

from __future__ import annotations

import pytest

from hivemind.swarm.s32_01_verlet_kepler442b_copilot import (
    HIVEMIND_API_KEY_ENV,
    DEFAULT_HIVEMIND_BASE_URL,
    AntigravityHiveMindCoPilot,
    SymplecticVelocityVerletIntegrator,
    ensure_antigravity_sdk_lightweight_compat,
)

TEST_API_KEY = "hm-" + "t" * 40  # a test string, not a key


def test_velocity_verlet_conserves_energy_and_angular_momentum_on_kepler442b() -> None:
    integrator = SymplecticVelocityVerletIntegrator()
    report = integrator.benchmark_kepler442b_orbit(steps=2000, dt_yr=0.00015)

    assert report.target_body == "KEPLER-HM-442b"
    assert report.symplectic_gate_passed is True
    assert report.max_relative_energy_drift_verlet < 1e-6
    assert report.angular_momentum_relative_drift < 1e-12
    assert report.verlet_over_euler_improvement_ratio > 500.0
    assert report.periapsis_au == pytest.approx(0.409 * (1.0 - 0.04), rel=1e-3)


def test_antigravity_sdk_lightweight_compat_and_config_builder() -> None:
    assert ensure_antigravity_sdk_lightweight_compat() is True
    copilot = AntigravityHiveMindCoPilot(
        api_key=TEST_API_KEY,
        base_url=DEFAULT_HIVEMIND_BASE_URL,
        model="hivemind/bee-stars",
    )
    cfg = copilot.build_antigravity_config()
    assert cfg.model == "hivemind/bee-stars"
    assert cfg.base_url == DEFAULT_HIVEMIND_BASE_URL
    assert copilot.authenticated_base_url.endswith(f"/k/{TEST_API_KEY}/v1")


def test_copilot_has_no_built_in_key_and_requires_the_callers_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import re
    from pathlib import Path

    monkeypatch.delenv(HIVEMIND_API_KEY_ENV, raising=False)
    with pytest.raises(ValueError):
        AntigravityHiveMindCoPilot()
    monkeypatch.setenv(HIVEMIND_API_KEY_ENV, TEST_API_KEY)
    assert AntigravityHiveMindCoPilot().api_key == TEST_API_KEY
    swarm_dir = Path(__file__).resolve().parents[1] / "src" / "hivemind" / "swarm"
    hits = [str(f.relative_to(swarm_dir)) for f in swarm_dir.glob("*.py") if re.search(r"hm-[A-Za-z0-9]{40,}", f.read_text(encoding="utf-8"))]
    assert hits == []
