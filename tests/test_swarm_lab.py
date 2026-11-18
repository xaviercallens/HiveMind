"""Tests for the HiveMind Swarm Lab toy (CPU only, stdlib only; qualitative analogy, not E13)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from hivemind.lab import swarm_lab as lab
from hivemind.lab.swarm_lab import (
    PRESETS,
    Rng,
    Scenario,
    carried_weights,
    compare_policies,
    contract_digest,
    main,
    merge_deltas,
    retention,
    simulate,
    sparkline,
)

REPO = Path(__file__).resolve().parent.parent


def test_prng_pinned_values_for_js_port() -> None:
    r = Rng(12345)
    got = [r.uniform() for _ in range(5)]
    want = [0.9797282677609473, 0.3067522644996643, 0.484205421525985, 0.817934412509203, 0.5094283693470061]
    assert got == want
    g = Rng(12345)
    assert [g.gauss() for _ in range(3)] == pytest.approx([-0.07064807449259204, 0.49860719149527216, -0.6676634049638013], abs=1e-15)
    assert Rng(2**32 + 12345).state == Rng(12345).state  # seeds are taken mod 2^32


def test_gauss_guards_zero_uniform(monkeypatch: pytest.MonkeyPatch) -> None:
    r = Rng(1)
    vals = iter([0.0, 0.25])
    monkeypatch.setattr(r, "uniform", lambda: next(vals))
    assert math.isfinite(r.gauss())


def test_determinism_and_seed_sensitivity() -> None:
    a, b = simulate({"seed": 3, "straggle": "rotating"}), simulate({"seed": 3, "straggle": "rotating"})
    assert a == b
    assert simulate({"seed": 4, "straggle": "rotating"})["loss_curve"] != a["loss_curve"]
    assert contract_digest({"seed": 3, "straggle": "rotating"}) == a["digest"]
    assert len(a["digest"]) == 16


def test_sync_reference_has_retention_one_and_json_roundtrip() -> None:
    res = simulate(Scenario(K=4, straggle="none", policy="discard", seed=5))
    assert res["retention"] == pytest.approx(1.0, abs=0)
    assert res["final_loss"] == res["sync_final_loss"]
    assert json.loads(json.dumps(res)) == res
    assert all(r["late"] == [] and len(r["admitted"]) == 4 for r in res["rounds"])


@pytest.mark.parametrize("policy", ["discard", "fair", "deferred"])
@pytest.mark.parametrize("straggle", ["none", "persistent", "rotating"])
def test_accounting_invariants(policy: str, straggle: str) -> None:
    res = simulate(Scenario(K=2, policy=policy, straggle=straggle, churn_start=3, churn_end=5, seed=2))
    for r in res["rounds"]:
        assert sorted(r["admitted"] + r["late"]) == r["alive"]
        assert set(r["required"]) <= set(r["admitted"])
        assert r["round_time"] > 0
        if 3 <= r["round"] <= 5:
            assert 1 not in r["alive"]
        if policy != "deferred":
            assert r["carried"] == 0
        if policy != "fair":
            assert r["required"] == []
    assert res["total_time"] == pytest.approx(sum(r["round_time"] for r in res["rounds"]))
    if policy == "deferred":
        assert res["rounds"][0]["carried"] == 0
        assert [r["carried"] for r in res["rounds"][1:]] == [len(r["late"]) for r in res["rounds"][:-1]]


def test_fair_requires_previously_excluded_worker() -> None:
    res = simulate(Scenario(K=3, policy="fair", straggle="persistent"))
    r1, r2 = res["rounds"][0], res["rounds"][1]
    assert r1["late"] == [3] and r1["required"] == []
    assert r2["required"] == [3] and 3 in r2["admitted"]  # the round waited for it
    assert r2["round_time"] > 2.0 > r1["round_time"]


def test_fair_dead_worker_never_blocks() -> None:
    # K=1, grace=0: only the fastest worker is admitted, so worker 1 is excluded often; find a seed where it is late in round 2, then it leaves in round 3
    seed = next(s for s in range(200) if 1 in simulate(Scenario(M=3, K=1, grace=0.0, policy="fair", seed=s))["rounds"][1]["late"])
    res = simulate(Scenario(M=3, K=1, grace=0.0, policy="fair", seed=seed, churn_start=3, churn_end=4))
    for r in res["rounds"][2:4]:
        assert 1 not in r["alive"] and 1 not in r["required"] and 1 not in r["admitted"] + r["late"]
    assert max(res["rounds"][2]["round_time"], res["rounds"][3]["round_time"]) < 1.3  # nobody waited for the dead worker


def test_fair_total_time_at_least_discard() -> None:
    for seed in range(1, 6):
        for st in ("persistent", "rotating"):
            fair = simulate(Scenario(policy="fair", straggle=st, seed=seed))["total_time"]
            disc = simulate(Scenario(policy="discard", straggle=st, seed=seed))["total_time"]
            assert fair >= disc - 1e-12
    assert simulate(Scenario(policy="fair", straggle="persistent"))["total_time"] > simulate(Scenario(policy="discard", straggle="persistent"))["total_time"]


def test_qualitative_ordering_over_seeds() -> None:
    def mean(**kw: object) -> float:
        vals = [simulate(Scenario(seed=s, **kw))["retention"] for s in range(1, 9)]  # type: ignore[arg-type]
        return sum(vals) / len(vals)

    persistent = mean(straggle="persistent")
    assert mean(straggle="persistent", policy="fair") > persistent + 0.1
    assert mean(straggle="persistent", policy="deferred") > persistent + 0.1
    assert mean(straggle="rotating") > persistent


def test_straggler_schedule() -> None:
    assert [lab._penalised(Scenario(straggle="persistent", straggle_every=2), r) for r in (1, 2, 3, 4, 5)] == [3, -1, 3, -1, 3]
    rot = [lab._penalised(Scenario(straggle="rotating"), r) for r in range(1, 8)]
    assert rot == [3, 2, 1, 3, 2, 1, 3]  # cycles M-1 .. 1, never worker 0
    assert lab._penalised(Scenario(), 1) == -1


def test_merge_and_rda_definition() -> None:
    items = [([1.0, 0.0], 1.0), ([0.0, 1.0], 1.0)]
    assert merge_deltas(items, "direct") == [0.5, 0.5]
    rda = merge_deltas(items, "rda")
    r = 2.0 / math.sqrt(2.0)  # sum |D| / |sum D|
    assert rda == pytest.approx([0.5 * r, 0.5 * r])
    cancel = [([1.0, 0.0], 1.0), ([-1.0, 0.0], 1.0)]
    assert merge_deltas(cancel, "rda") == [0.0, 0.0]  # zero-norm guard: r = 1
    weighted = merge_deltas([([2.0], 1.0), ([4.0], 0.5)], "direct")
    assert weighted == pytest.approx([(2.0 + 2.0) / 1.5])


def test_carried_weights_and_staleness_drop() -> None:
    q = [{"delta": [0.0], "base": 5, "worker": 2}, {"delta": [0.0], "base": 2, "worker": 1}, {"delta": [0.0], "base": 4, "worker": 0}]
    keep, dropped = carried_weights(q, 5)
    assert dropped == 1  # staleness 3 > MAX_STALENESS
    assert [(e["base"], e["w"]) for e in keep] == [(4, 0.5), (5, 1.0)]
    keep2, _ = carried_weights([{"delta": [0.0], "base": 3, "worker": 1}], 5)
    assert keep2[0]["w"] == pytest.approx(1.0 / 3.0)


def test_retention_formula_edge_cases() -> None:
    assert retention(10.0, 4.0, 2.0) == pytest.approx(0.75)
    assert retention(5.0, 5.0, 5.0) is None
    assert retention(5.0, 4.0, 5.0 + 1e-13) is None
    res = simulate(Scenario(spread=0.0, offset=0.0, sigma=0.0))  # L0 == 0 == L_sync: nothing to learn
    assert res["retention"] is None


def test_iid_preset_learns_and_presets_exist() -> None:
    assert set(PRESETS) == {"iid-sync", "persistent-straggler", "rotating-straggler", "churn"}
    res = simulate(PRESETS["iid-sync"])
    assert res["final_loss"] < 0.01 * res["initial_loss"] and res["retention"] == pytest.approx(1.0)
    for p in PRESETS.values():
        assert simulate(p)["rounds"]


def test_compare_policies_table() -> None:
    rows = compare_policies(None)
    assert len(rows) == 9 and {(r["policy"], r["straggle"]) for r in rows} == {(p, s) for p in lab.POLICIES for s in lab.STRAGGLES}
    assert all({"retention", "total_time", "final_loss", "late_total", "carried_total"} <= set(r) for r in rows)
    assert compare_policies(Scenario(M=3, K=3), policies=("discard",), straggles=("none",))[0]["retention"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "bad",
    [
        {"M": 0}, {"M": 7}, {"K": 0}, {"K": 5}, {"rounds": 0}, {"dim": 1}, {"eta": 0.0}, {"sigma": -1.0}, {"outer_lr": 0.0}, {"outer_momentum": 1.0},
        {"merge": "x"}, {"policy": "x"}, {"straggle": "x"}, {"M": 1, "K": 1, "straggle": "persistent"}, {"churn_start": 5, "churn_end": 2}, {"churn_start": 1, "churn_end": 99},
        {"M": 1, "K": 1, "churn_start": 1, "churn_end": 2}, {"offset": -1.0},
    ],
)
def test_validation_errors(bad: dict) -> None:
    with pytest.raises(ValueError):
        simulate(bad)


def test_sparkline() -> None:
    assert sparkline([1.0, 1.0, 1.0]) == "   "
    s = sparkline([float(i) for i in range(200)], width=40)
    assert len(s) <= 40 and s[0] == " " and s[-1] == "@"


def test_cli_simulate_text_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["simulate", "--preset", "persistent-straggler", "--seed", "3"]) == 0
    out = capsys.readouterr().out
    assert "retention" in out and "qualitative toy" in out and "loss  [" in out
    assert main(["simulate", "--preset", "churn", "--json", "--rounds", "5", "--churn-end", "5"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["scenario"]["rounds"] == 5 and data["scenario"]["policy"] == "fair" and len(data["loss_curve"]) == 6


def test_cli_compare_presets_play(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compare"]) == 0
    assert "persistent" in capsys.readouterr().out
    assert main(["compare", "--json", "--K", "2"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 9
    assert main(["presets"]) == 0
    assert "iid-sync" in capsys.readouterr().out
    assert main(["presets", "--json"]) == 0
    assert set(json.loads(capsys.readouterr().out)) == set(PRESETS)
    assert main(["play", "--auto"]) == 0
    out = capsys.readouterr().out
    assert "learning score" in out and "non-monetary" in out and "Level 3" in out


def test_cli_play_interactive_and_eof(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    answers = iter(["9", "3", "nonsense", "fair", "4", "deferred"])  # then EOF -> scripted choices

    def fake_input(prompt: str) -> str:
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)
    assert main(["play"]) == 0
    out = capsys.readouterr().out
    assert "please answer one of" in out and out.count("learning score") >= 4


def test_cli_error_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["simulate", "--M", "9"]) == 2
    assert "error" in capsys.readouterr().err


def test_cli_flags_cover_every_scenario_field() -> None:
    help_text = lab.build_parser()._subparsers._group_actions[0].choices["simulate"].format_help()  # type: ignore[union-attr]
    for f in dataclasses.fields(Scenario):
        assert "--" + f.name.replace("_", "-") in help_text


def test_module_docstring_and_notice_wording() -> None:
    doc = lab.__doc__ or ""
    assert "QUALITATIVE" in doc and "NOT" in doc and "E13" in doc
    assert "non-monetary" in lab.NONMONETARY_NOTICE and "$HIVE" in lab.NONMONETARY_NOTICE


def test_runner_script_and_hivemind_subcommand() -> None:
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "swarm_lab.py"), "presets", "--json"], capture_output=True, text=True, check=True,
                       env={"CUDA_VISIBLE_DEVICES": "", "PATH": ""})
    assert set(json.loads(r.stdout)) == set(PRESETS)
    cli_main = pytest.importorskip("hivemind.cli.main")
    assert cli_main.main(["swarm-lab", "presets"]) == 0


@pytest.mark.parametrize("knob", ["eta", "sigma", "spread", "offset", "penalty", "grace", "outer_lr", "outer_momentum"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_float_knobs_are_rejected(knob: str, value: float) -> None:
    # NaN fails every comparison, so the range checks alone let it through and the run reported final_loss nan.
    with pytest.raises(ValueError, match="finite"):
        simulate({knob: value})
    assert math.isfinite(simulate({})["final_loss"])
