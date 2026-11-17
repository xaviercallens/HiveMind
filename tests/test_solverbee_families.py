# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for the Science Bee family catalogue and spec builders (stdlib only, deterministic)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hivemind.solverbee import families as fam  # noqa: E402
from hivemind.solverbee import science_bee as sb  # noqa: E402

T = [1.0, 2.0, 5.0]


def test_catalogue_covers_new_and_old_families():
    for name in ("pendulum", "projectile_drag", "kepler", "logistic", "robertson", "van_der_pol", "brusselator", "lotka_volterra", "harmonic", "stiff_chain"):
        assert name in fam.FAMILIES
    assert fam.FAMILIES["pendulum"]["params"]["g_over_l"][0] == 9.81
    assert fam.FAMILIES["pendulum"]["params"]["damping"][0] == 0.0
    assert fam.FAMILIES["projectile_drag"]["params"]["g"][0] == 9.81 and fam.FAMILIES["projectile_drag"]["params"]["k"][0] == 0.0
    assert fam.FAMILIES["kepler"]["params"]["mu"][0] == 1.0
    assert fam.FAMILIES["logistic"]["params"]["r"][0] == 1.0 and fam.FAMILIES["logistic"]["params"]["K"][0] == 1.0
    assert fam.FAMILIES["kepler"]["invariants"] == ["energy", "angular momentum"]
    for name, e in fam.FAMILIES.items():
        assert e["summary"] and "params" in e
        if e["dim"] is not None:
            assert len(e["state"]) == e["dim"], name
    assert "pendulum" in fam.catalogue_text() and "sir" in fam.presets_text()


def test_every_builder_output_passes_the_strict_validator():
    specs = [
        fam.pendulum_spec(0.5, 0.0, T, damping=0.1), fam.projectile_drag_spec(0, 0, 10, 10, T, k=0.01), fam.kepler_spec(1, 0, 0, 1, T),
        fam.logistic_spec(0.1, T, r=0.5, K=2.0), fam.sir_spec(0.5, 0.2, 0.99, 0.01, T), fam.seir_spec(0.5, 0.3, 0.2, 0.98, 0.01, 0.01, T),
        fam.decay_chain_spec([1.0, 0.5], [1, 0, 0], T), fam.series_rlc_spec(10, 0.5, 0.01, 5, 0, 0, T), fam.spring_mass_damper_spec(1, 0.4, 4, 0, 1, 0, T),
        fam.named_spec("harmonic", {"omega": 2.0}, [1, 0], T), fam.named_spec("stiff_chain", {"n": 5, "k": 10}, [1, 0, 0, 0, 0], T),
    ]
    for s in specs:
        sb.validate_spec(json.loads(json.dumps(s)))


def test_sir_and_seir_are_mass_action_presets_as_specified():
    s = fam.sir_spec(0.5, 0.2, 0.99, 0.01, T)
    assert s["problem"]["species"] == ["S", "I", "R"]
    assert s["problem"]["reactions"][0] == {"reactants": {"S": 1, "I": 1}, "products": {"I": 2}, "k": 0.5}
    assert s["problem"]["reactions"][1] == {"reactants": {"I": 1}, "products": {"R": 1}, "k": 0.2}
    assert s["y0"][:2] == [0.99, 0.01] and abs(s["y0"][2]) < 1e-12
    e = fam.seir_spec(0.5, 0.3, 0.2, 0.98, 0.01, 0.01, T)
    assert e["problem"]["species"] == ["S", "E", "I", "R"]
    assert e["problem"]["reactions"][0] == {"reactants": {"S": 1, "I": 1}, "products": {"E": 1, "I": 1}, "k": 0.5}
    assert e["problem"]["reactions"][1]["reactants"] == {"E": 1} and e["problem"]["reactions"][1]["products"] == {"I": 1}


def test_linear_presets_fix_state_order_and_matrices():
    rlc = fam.series_rlc_spec(10.0, 0.5, 0.01, 5.0, 0.0, 0.0, T)
    assert rlc["problem"]["A"] == [[0.0, 1.0], [-1.0 / (0.5 * 0.01), -10.0 / 0.5]] and rlc["problem"]["b"] == [0.0, 10.0]
    smd = fam.spring_mass_damper_spec(2.0, 0.4, 8.0, 1.0, 1.0, 0.0, T)
    assert smd["problem"]["A"] == [[0.0, 1.0], [-4.0, -0.2]] and smd["problem"]["b"] == [0.0, 0.5]


@pytest.mark.parametrize("call", [
    lambda: fam.pendulum_spec(0.5, 0.0, [2.0, 1.0]),                    # not increasing
    lambda: fam.pendulum_spec(0.5, 0.0, [0.0]),                         # not > t0
    lambda: fam.pendulum_spec(float("nan"), 0.0, T),
    lambda: fam.pendulum_spec(True, 0.0, T),                            # bool is not a number
    lambda: fam.pendulum_spec(0.5, 0.0, T, damping=-1.0),
    lambda: fam.pendulum_spec(0.5, 0.0, T, g_over_l=0.0),
    lambda: fam.named_spec("pendulum", {"bogus": 1.0}, [0, 0], T),
    lambda: fam.named_spec("nope", {}, [0], T),
    lambda: fam.named_spec("pendulum", {}, [0.0], T),                   # wrong y0 length
    lambda: fam.named_spec("stiff_chain", {"n": 2.5}, [1, 0], T),
    lambda: fam.projectile_drag_spec(0, 0, 1, 1, T, k=-0.1),
    lambda: fam.logistic_spec(-0.1, T),
    lambda: fam.sir_spec(0.5, 0.2, 0.9, 0.3, T),                        # fractions do not sum to 1
    lambda: fam.sir_spec(-0.5, 0.2, 0.9, 0.1, T),
    lambda: fam.seir_spec(0.5, 0.3, 0.2, 0.5, 0.2, 0.2, T, r0=0.5),
    lambda: fam.series_rlc_spec(10, 0.0, 0.01, 5, 0, 0, T),             # L = 0
    lambda: fam.spring_mass_damper_spec(0.0, 0.4, 4, 0, 1, 0, T),       # m = 0
    lambda: fam.decay_chain_spec([], [1], T),
    lambda: fam.decay_chain_spec([1.0], [1.0], T),                      # wrong y0 length
    lambda: fam.pendulum_spec(0.5, 0.0, T, spec_id="bad id!"),
    lambda: fam.pendulum_spec(0.5, 0.0, T, rtol=0.0),
    lambda: fam.pendulum_spec(0.5, 0.0, T, method="rk4"),
    lambda: fam.pendulum_spec(0.5, 0.0, T, g_over_l=1e300),
])
def test_strict_argument_validation(call):
    with pytest.raises(fam.FamilyError):
        call()


def test_defaults_are_filled_for_comparison_only():
    a = fam.pendulum_spec(0.5, 0.0, T)
    b = fam.named_spec("pendulum", {"g_over_l": 9.81, "damping": 0.0}, [0.5, 0.0], T)
    assert sb.specs_match(a, b)
    assert not sb.specs_match(a, fam.pendulum_spec(0.5, 0.0, T, damping=0.2))
    assert fam.family_dim("stiff_chain", {"n": 7}) == 7 and math.isclose(fam.family_dim("pendulum"), 2)
