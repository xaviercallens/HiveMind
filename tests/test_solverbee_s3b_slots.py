# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for S-3b slot-filling presets, extended time/period units, and Wave 3 scientific slot families."""

from __future__ import annotations

import math

import pytest

from hivemind.solverbee import families as fam
from hivemind.solverbee import science_bee as sb


class _DummyPipeline:
    def run(self, spec_bytes: bytes) -> dict:
        return {
            "verdict": "AUDITED",
            "result": {"status": "ok", "t": [1.0, 2.0], "y": [[0.1, 0.2], [0.05, 0.1]]},
            "certificate": {"schema": "hivemind-solver-cert/1", "verdict": "AUDITED", "checks": [], "reasons": []},
            "wall_ms": 1,
        }


def test_wave3_yukawa_crystal_slot_compilation() -> None:
    slot = {
        "schema": fam.SLOT_SCHEMA,
        "id": "qf-yukawa-1",
        "slot_family": "yukawa_crystal",
        "slots": {"kappa": 2.0, "damping": 0.1, "x0": 0.08, "v0": 0.0, "t_out": [0.5, 1.0, 2.0]},
    }
    compiled, trace = fam.compile_slot_spec(slot, return_trace=True)
    validated = sb.validate_spec(compiled)
    assert validated["problem"]["kind"] == "named"
    assert validated["problem"]["family"] == "harmonic"
    assert len(validated["y0"]) == 2
    assert validated["y0"][0] == pytest.approx(0.08)
    expected_omega = math.sqrt(1.0 + 2.0 * (2.0 ** 2) * math.exp(-2.0))
    assert validated["problem"]["params"]["omega"] == pytest.approx(expected_omega)
    assert any("omega_eff" in d for d in trace["derived_parameters"])


def test_wave3_gpe_vortex_slot_compilation() -> None:
    slot = {
        "schema": fam.SLOT_SCHEMA,
        "id": "qf-gpe-1",
        "slot_family": "gpe_vortex",
        "slots": {"omega_trap": 1.5, "g_int": 2.0, "zeta": 0.02, "x0": 0.8, "v0": 0.6, "t_out": [0.2, 0.4]},
    }
    compiled, trace = fam.compile_slot_spec(slot, return_trace=True)
    validated = sb.validate_spec(compiled)
    assert validated["problem"]["kind"] == "named"
    assert validated["problem"]["family"] == "harmonic"
    assert validated["y0"] == [0.8, 0.6]
    # omega_bog = 1.5 * sqrt(2 + 2.0) = 3.0
    assert validated["problem"]["params"]["omega"] == pytest.approx(3.0)
    assert any("omega_B" in d for d in trace["derived_parameters"])


def test_wave3_art_exo_genetics_slot_compilation() -> None:
    slot = {
        "schema": fam.SLOT_SCHEMA,
        "id": "gen-art-exo-1",
        "slot_family": "art_exo_genetics",
        "slots": {"k_pol": 1.2, "k_exo": 0.9, "k_ext": 0.4, "T0": 1.0, "M0": 0.2, "P0": 0.0, "t_out": [1.0, 5.0]},
    }
    compiled, trace = fam.compile_slot_spec(slot, return_trace=True)
    validated = sb.validate_spec(compiled)
    assert validated["problem"]["kind"] == "mass_action"
    assert validated["problem"]["species"] == ["T", "M", "P"]
    assert validated["y0"] == [1.0, 0.2, 0.0]
    assert any("ART-Exo" in d for d in trace["derived_parameters"])


def test_wave3_bao_cosmology_slot_compilation() -> None:
    slot = {
        "schema": fam.SLOT_SCHEMA,
        "id": "astro-bao-1",
        "slot_family": "bao_cosmology",
        "slots": {"m_eff": 0.5, "hubble": 0.7, "phi0": 1.0, "dphi0": -0.1, "t_out": [1.0, 2.0, 4.0]},
    }
    compiled, trace = fam.compile_slot_spec(slot, return_trace=True)
    validated = sb.validate_spec(compiled)
    assert validated["problem"]["kind"] == "linear"
    assert validated["problem"]["A"][1][0] == pytest.approx(-0.25)
    assert validated["problem"]["A"][1][1] == pytest.approx(-2.1)
    assert any("3*H=2.1" in d for d in trace["derived_parameters"])


def test_extended_time_units_and_pendulum_period() -> None:
    slot = {
        "schema": fam.SLOT_SCHEMA,
        "slot_family": "pendulum",
        "slots": {"theta0_deg": 10.0, "period_s": 2.0, "t_out_h": [0.001, 0.002]},
    }
    compiled, trace = fam.compile_slot_spec(slot, return_trace=True)
    validated = sb.validate_spec(compiled)
    expected_g_over_l = (2.0 * math.pi / 2.0) ** 2
    assert validated["problem"]["params"]["g_over_l"] == pytest.approx(expected_g_over_l)
    assert validated["t_out"] == pytest.approx([3.6, 7.2])
    assert any("h -> s" in c for c in trace["conversions"])


def test_slot_formulate_wave3_and_pendulum_period_questions() -> None:
    q_yukawa = "Simulate a Yukawa crystal with kappa = 2.0, damping = 0.05, x0 = 0.05, v0 = 0 at times 0.5, 1.0, 2.0."
    s_yukawa = sb.slot_formulate(q_yukawa)
    assert s_yukawa is not None
    assert s_yukawa["slot_family"] == "yukawa_crystal"

    q_gpe = "Compute a Gross-Pitaevskii condensate mode with omega_trap = 1.2, g_int = 0.3, zeta = 0.01, x0 = 1.0, v0 = 0.0 at t = 0.5, 1.0."
    s_gpe = sb.slot_formulate(q_gpe)
    assert s_gpe is not None
    assert s_gpe["slot_family"] == "gpe_vortex"

    q_art = "Model ART-Exo exonuclease proofreading with k_pol = 1.5, k_exo = 0.8, k_ext = 0.3, T0 = 1.0, M0 = 0.1, P0 = 0.0 at times 1, 2, 5."
    s_art = sb.slot_formulate(q_art)
    assert s_art is not None
    assert s_art["slot_family"] == "art_exo_genetics"

    q_bao = "Solve a quintessence Klein-Gordon scalar field with m_eff = 0.4, hubble = 0.68, phi0 = 1.2, dphi0 = 0.0 at times 1, 2, 3."
    s_bao = sb.slot_formulate(q_bao)
    assert s_bao is not None
    assert s_bao["slot_family"] == "bao_cosmology"

    q_pend = "A pendulum has period = 2 s and initial angle 15 deg. Give its state at times 0.5, 1.0, 1.5."
    s_pend = sb.slot_formulate(q_pend)
    assert s_pend is not None
    assert s_pend["slots"]["period_s"] == pytest.approx(2.0)


def test_prefer_slots_prompt_and_science_bee_echo() -> None:
    prompt_default = sb.build_system_prompt(prefer_slots=False)
    prompt_slots = sb.build_system_prompt(prefer_slots=True)
    assert "hivemind-solver-slots/1" not in prompt_default
    assert "hivemind-solver-slots/1" in prompt_slots
    assert "yukawa_crystal" in prompt_slots

    bee = sb.ScienceBee(sb.SlotBackend(), _DummyPipeline(), prefer_slots=True)
    ans = bee.ask("Simulate a Yukawa crystal with kappa = 2.0, damping = 0.05, x0 = 0.05, v0 = 0 at times 0.5, 1.0.")
    assert ans["formulated"] is True
    assert ans["verdict"] == "AUDITED"
    assert ans["interpreted_problem_echo"]["schema"] == "hivemind-problem-echo/1"
    assert len(ans["interpreted_problem_echo"]["derived_parameters"]) >= 1
