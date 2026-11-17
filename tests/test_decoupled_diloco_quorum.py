"""Tests for the Decoupled DiLoCo syncer primitives (`SI-12`, `arXiv:2604.21428`)."""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest
import torch

from hivemind.swarm.decoupled_diloco import (
    AdaptiveGraceWindowController,
    DecoupledDiLoCoQuorumSyncer,
    LearnerArrival,
    compute_cosine_damped_outer_hyperparams,
    direct_weighted_average,
    radial_directional_average,
    select_quorum,
    token_weighted_quality_weight,
)
from hivemind.swarm.diloco_nextgen import CPUOffloadedFlatBucketOuterState


# --------------------------------------------------------------------------------------
# Weight(c_tokens, c_steps) = c_tokens * (c_tokens / c_steps)
# --------------------------------------------------------------------------------------


def test_token_weight_matches_paper_formula_and_zero_steps_gives_zero() -> None:
    assert token_weighted_quality_weight(4096, 64) == pytest.approx(4096.0 * 4096.0 / 64.0)
    assert token_weighted_quality_weight(4096, 0) == 0.0
    # Same tokens amortised over fewer steps = higher "quality" = larger weight.
    assert token_weighted_quality_weight(4096, 32) > token_weighted_quality_weight(4096, 64)
    with pytest.raises(ValueError):
        token_weighted_quality_weight(-1, 10)


@given(
    tokens_a=st.integers(min_value=0, max_value=10**7),
    tokens_b=st.integers(min_value=0, max_value=10**7),
    steps=st.integers(min_value=1, max_value=10**5),
)
def test_token_weight_is_monotone_in_tokens_and_antitone_in_steps(tokens_a: int, tokens_b: int, steps: int) -> None:
    lo, hi = sorted((tokens_a, tokens_b))
    assert token_weighted_quality_weight(lo, steps) <= token_weighted_quality_weight(hi, steps)
    assert token_weighted_quality_weight(hi, steps) >= token_weighted_quality_weight(hi, steps + 1)
    assert token_weighted_quality_weight(hi, steps) >= 0.0


# --------------------------------------------------------------------------------------
# Quorum selection
# --------------------------------------------------------------------------------------


def _arrivals(times: dict[str, float]) -> list[LearnerArrival]:
    return [
        LearnerArrival(learner_id=k, arrival_time_s=t, steps_since_sync=10, tokens_since_sync=1000)
        for k, t in times.items()
    ]


def test_select_quorum_admits_first_k_then_only_inside_grace_window() -> None:
    arr = _arrivals({"a": 0.0, "b": 1.0, "c": 2.5, "d": 9.0})
    no_grace = select_quorum(arr, min_quorum_k=2, grace_window_s=0.0)
    assert no_grace.reached
    assert no_grace.admitted_learner_ids == ("a", "b")
    assert no_grace.deferred_learner_ids == ("c", "d")
    assert no_grace.quorum_reached_at_s == 1.0

    # Grace window of 1.5 s after the quorum at t=1.0 closes at 2.5 s: "c" is admitted, "d" is not.
    partial = select_quorum(arr, min_quorum_k=2, grace_window_s=1.5)
    assert partial.admitted_learner_ids == ("a", "b", "c")
    assert partial.deferred_learner_ids == ("d",)
    assert partial.window_close_s == pytest.approx(2.5)

    everyone = select_quorum(arr, min_quorum_k=2, grace_window_s=100.0)
    assert everyone.admitted_learner_ids == ("a", "b", "c", "d")


def test_select_quorum_not_reached_admits_nobody() -> None:
    arr = _arrivals({"a": 0.0, "b": 1.0})
    sel = select_quorum(arr, min_quorum_k=3, grace_window_s=5.0)
    assert not sel.reached
    assert sel.admitted_learner_ids == ()
    assert sel.deferred_learner_ids == ("a", "b")
    assert sel.quorum_reached_at_s is None


def test_select_quorum_rejects_duplicates_and_bad_k() -> None:
    arr = _arrivals({"a": 0.0}) + _arrivals({"a": 1.0})
    with pytest.raises(ValueError):
        select_quorum(arr, min_quorum_k=1, grace_window_s=0.0)
    with pytest.raises(ValueError):
        select_quorum(_arrivals({"a": 0.0}), min_quorum_k=0, grace_window_s=0.0)


@settings(max_examples=200)
@given(
    times=st.lists(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False), min_size=1, max_size=12),
    k=st.integers(min_value=1, max_value=12),
    grace=st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
)
def test_select_quorum_invariants(times: list[float], k: int, grace: float) -> None:
    arr = [
        LearnerArrival(learner_id=f"L{i:02d}", arrival_time_s=t, steps_since_sync=1, tokens_since_sync=1)
        for i, t in enumerate(times)
    ]
    sel = select_quorum(arr, min_quorum_k=k, grace_window_s=grace)
    m = len(arr)
    assert set(sel.admitted_learner_ids) | set(sel.deferred_learner_ids) == {a.learner_id for a in arr}
    assert not (set(sel.admitted_learner_ids) & set(sel.deferred_learner_ids))
    if m < k:
        assert not sel.reached and len(sel.admitted_learner_ids) == 0
        return
    assert sel.reached
    assert k <= len(sel.admitted_learner_ids) <= m
    ordered = sorted(arr, key=lambda a: (a.arrival_time_s, a.learner_id))
    first_k = {a.learner_id for a in ordered[:k]}
    assert first_k <= set(sel.admitted_learner_ids)
    by_id = {a.learner_id: a for a in arr}
    assert sel.window_close_s is not None
    for lid in sel.admitted_learner_ids:
        if lid not in first_k:
            assert by_id[lid].arrival_time_s <= sel.window_close_s
    for lid in sel.deferred_learner_ids:
        assert by_id[lid].arrival_time_s > sel.window_close_s


# --------------------------------------------------------------------------------------
# Adaptive grace window
# --------------------------------------------------------------------------------------


def test_grace_controller_tracks_slack_and_stays_below_gamma_times_slack() -> None:
    ctrl = AdaptiveGraceWindowController(overlap_steps_tau=2, safety_margin_gamma=0.5, ema_beta=0.0, sync_duration_s=1.0)
    assert ctrl.grace_window_seconds() == 0.0  # nothing observed yet
    ctrl.observe_step_duration(10.0)
    ctrl.observe_quorum_duration(4.0)
    # slack = 2 * 10 - (4 + 1) = 15 ; grace = 0.5 * 15 = 7.5 < 15
    assert ctrl.slack_seconds() == pytest.approx(15.0)
    assert ctrl.grace_window_seconds() == pytest.approx(7.5)
    assert ctrl.grace_window_seconds() < ctrl.slack_seconds()
    # Quorum gathering slower than the compute budget: negative slack -> no grace at all.
    ctrl.observe_quorum_duration(40.0)
    assert ctrl.slack_seconds() < 0.0
    assert ctrl.grace_window_seconds() == 0.0


def test_grace_controller_ema_and_validation() -> None:
    ctrl = AdaptiveGraceWindowController(ema_beta=0.5)
    assert ctrl.observe_step_duration(2.0) == pytest.approx(2.0)
    assert ctrl.observe_step_duration(4.0) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        AdaptiveGraceWindowController(safety_margin_gamma=1.0)
    with pytest.raises(ValueError):
        ctrl.observe_step_duration(-1.0)


@given(
    step=st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_subnormal=False),
    quorum=st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_subnormal=False),
    sync=st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_subnormal=False),
    gamma=st.floats(min_value=0.0, max_value=0.999, allow_nan=False, allow_subnormal=False),
    tau=st.integers(min_value=1, max_value=8),
)
def test_grace_window_never_exceeds_gamma_times_slack(step: float, quorum: float, sync: float, gamma: float, tau: int) -> None:
    ctrl = AdaptiveGraceWindowController(overlap_steps_tau=tau, safety_margin_gamma=gamma, ema_beta=0.0, sync_duration_s=sync)
    ctrl.observe_step_duration(step)
    ctrl.observe_quorum_duration(quorum)
    slack = tau * step - (quorum + sync)
    grace = ctrl.grace_window_seconds()
    assert grace >= 0.0
    assert grace <= gamma * max(0.0, slack) + 1e-9
    if slack > 1e-300:
        assert grace < slack


# --------------------------------------------------------------------------------------
# Radial-Directional Averaging
# --------------------------------------------------------------------------------------


def test_rda_keeps_norm_r_for_orthogonal_equal_norm_vectors_while_direct_average_shrinks() -> None:
    m, d, r = 8, 64, 3.0
    basis = torch.eye(d)[:m] * r  # pairwise orthogonal, all of norm r
    direct = direct_weighted_average(list(basis))
    rda = radial_directional_average(list(basis))
    assert float(direct.norm()) == pytest.approx(r / math.sqrt(m), rel=1e-5)
    assert float(rda.norm()) == pytest.approx(r, rel=1e-5)
    # RDA keeps the average direction of the plain mean.
    cos = torch.dot(direct / direct.norm(), rda / rda.norm())
    assert float(cos) == pytest.approx(1.0, abs=1e-6)


def test_rda_identical_vectors_is_identity_and_cancelling_directions_give_zero() -> None:
    v = torch.tensor([1.0, -2.0, 0.5])
    assert torch.allclose(radial_directional_average([v, v, v]), v, atol=1e-6)
    assert torch.allclose(direct_weighted_average([v, v, v]), v, atol=1e-6)
    out = radial_directional_average([v, -v])
    assert torch.allclose(out, torch.zeros_like(v))
    with pytest.raises(ValueError):
        radial_directional_average([v, torch.zeros(2)])
    with pytest.raises(ValueError):
        radial_directional_average([v, v], weights=[0.0, 0.0])


@settings(max_examples=100, deadline=None)
@given(
    m=st.integers(min_value=1, max_value=6),
    r=st.floats(min_value=0.1, max_value=50.0, allow_nan=False),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_rda_norm_equals_common_norm_r(m: int, r: float, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    vecs = []
    for _ in range(m):
        v = torch.randn(16, generator=g)
        v = v / v.norm() * r
        vecs.append(v)
    weights = torch.rand(m, generator=g).add(0.1).tolist()
    out = radial_directional_average(vecs, weights=weights)
    # Random directions in 16-D virtually never cancel exactly; if they did, RDA returns 0.
    if float(out.norm()) > 0.0:
        assert float(out.norm()) == pytest.approx(r, rel=1e-4)
    assert float(direct_weighted_average(vecs, weights=weights).norm()) <= r + 1e-4


# --------------------------------------------------------------------------------------
# Full syncer round
# --------------------------------------------------------------------------------------


def _learner_round(global_prev: torch.Tensor, deltas: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {lid: global_prev - d for lid, d in deltas.items()}


def test_syncer_merges_quorum_excludes_late_straggler_and_feeds_flat_bucket_outer_step() -> None:
    torch.manual_seed(7)
    d = 32
    global_prev = torch.zeros(d)
    e = torch.eye(d)
    deltas = {"fast0": e[0] * 2.0, "fast1": e[1] * 2.0, "fast2": e[2] * 2.0, "slow": e[3] * 500.0}
    learner_frags = _learner_round(global_prev, deltas)
    arrivals = [
        LearnerArrival("fast0", 0.0, steps_since_sync=50, tokens_since_sync=5000),
        LearnerArrival("fast1", 0.2, steps_since_sync=50, tokens_since_sync=5000),
        LearnerArrival("fast2", 0.4, steps_since_sync=50, tokens_since_sync=5000),
        LearnerArrival("slow", 30.0, steps_since_sync=50, tokens_since_sync=5000),
    ]
    syncer = DecoupledDiLoCoQuorumSyncer(num_learners_m=4, min_quorum_k=2, merge_mode="rda")
    delta, sel, tel = syncer.merge_fragment("w", global_prev, learner_frags, arrivals, grace_window_s=1.0)
    assert delta is not None
    assert sel.admitted_learner_ids == ("fast0", "fast1", "fast2")
    assert tel.num_deferred == 1 and tel.deferred_tokens == 5000 and tel.admitted_tokens == 15000
    # The straggler's huge delta must not leak into the merge.
    assert float(delta[3]) == 0.0
    # RDA restores the common norm 2.0, the direct average would be 2/sqrt(3).
    assert tel.merged_delta_norm == pytest.approx(2.0, rel=1e-5)
    assert tel.direct_average_norm == pytest.approx(2.0 / math.sqrt(3), rel=1e-5)

    # Delta_p is consumed by the existing CPU flat-bucket Nesterov outer step unchanged.
    state = CPUOffloadedFlatBucketOuterState({"w": global_prev.clone()})
    updated, _ = state.outer_nesterov_step({"w": delta}, lr=0.7, momentum=0.9)
    expected = global_prev - 0.7 * delta  # velocity starts at 0 -> step = lr * (0.9*delta + delta)? no: v=delta, step=mu*v+delta
    expected = global_prev - 0.7 * (0.9 * delta + delta)
    assert torch.allclose(updated["w"], expected, atol=1e-6)


def test_syncer_waits_when_quorum_not_reached_and_direct_mode_matches_weighted_mean() -> None:
    d = 8
    global_prev = torch.ones(d)
    deltas = {"a": torch.full((d,), 1.0), "b": torch.full((d,), 3.0)}
    frags = _learner_round(global_prev, deltas)
    syncer = DecoupledDiLoCoQuorumSyncer(num_learners_m=3, min_quorum_k=3, merge_mode="direct")
    arrivals = [LearnerArrival("a", 0.0, 10, 100), LearnerArrival("b", 1.0, 10, 300)]
    delta, sel, tel = syncer.merge_fragment("w", global_prev, frags, arrivals, grace_window_s=0.0)
    assert delta is None and not sel.reached and tel.num_admitted == 0

    syncer2 = DecoupledDiLoCoQuorumSyncer(num_learners_m=3, min_quorum_k=2, merge_mode="direct")
    delta2, _, tel2 = syncer2.merge_fragment("w", global_prev, frags, arrivals, grace_window_s=0.0)
    assert delta2 is not None
    w_a = token_weighted_quality_weight(100, 10)
    w_b = token_weighted_quality_weight(300, 10)
    expected = (w_a * 1.0 + w_b * 3.0) / (w_a + w_b)
    assert torch.allclose(delta2, torch.full((d,), expected), atol=1e-5)
    assert tel2.learner_weights == {"a": w_a, "b": w_b}


def test_syncer_rejects_bad_configuration_and_zero_step_learners() -> None:
    with pytest.raises(ValueError):
        DecoupledDiLoCoQuorumSyncer(num_learners_m=2, min_quorum_k=3)
    with pytest.raises(ValueError):
        DecoupledDiLoCoQuorumSyncer(num_learners_m=2, min_quorum_k=1, merge_mode="slerp")
    syncer = DecoupledDiLoCoQuorumSyncer(num_learners_m=2, min_quorum_k=1)
    gp = torch.zeros(4)
    frags = {"a": torch.ones(4)}
    with pytest.raises(ValueError):
        syncer.merge_fragment("w", gp, frags, [LearnerArrival("a", 0.0, steps_since_sync=0, tokens_since_sync=0)])
    with pytest.raises(KeyError):
        syncer.merge_fragment("w", gp, {}, [LearnerArrival("a", 0.0, 1, 1)])
    with pytest.raises(ValueError):
        syncer.merge_fragment("w", gp, {"a": torch.ones(5)}, [LearnerArrival("a", 0.0, 1, 1)])


def test_diloco_outer_optimizer_with_decoupled_quorum_rda() -> None:
    """HM-S9-01: Verify DiLoCoOuterOptimizer.step integrates DecoupledDiLoCoQuorumSyncer end-to-end."""
    from hivemind.core.diloco_outer import DiLoCoOuterOptimizer

    theta_prev = {"w": torch.zeros(16)}
    e = torch.eye(16)
    opt = DiLoCoOuterOptimizer(outer_lr_eta=0.5, outer_momentum_mu=0.0, max_byzantine_f=0, cosine_sigma=10.0)
    # 3 fast learners with orthogonal updates of norm 2.0, plus 1 late straggler with norm 2.0 on index 3
    worker_deltas = {
        "fast0": {"w": e[0] * 2.0},
        "fast1": {"w": e[1] * 2.0},
        "fast2": {"w": e[2] * 2.0},
        "slow0": {"w": e[3] * 2.0},
    }
    arrivals = [
        LearnerArrival("fast0", 0.0, steps_since_sync=50, tokens_since_sync=5000),
        LearnerArrival("fast1", 0.2, steps_since_sync=50, tokens_since_sync=5000),
        LearnerArrival("fast2", 0.4, steps_since_sync=50, tokens_since_sync=5000),
        LearnerArrival("slow0", 25.0, steps_since_sync=50, tokens_since_sync=5000),
    ]
    res = opt.step(
        branch_name="Code",
        outer_step=1,
        theta_prev=theta_prev,
        worker_deltas=worker_deltas,
        learner_arrivals=arrivals,
        min_quorum_k=2,
        grace_window_s=1.0,
        quorum_merge_mode="rda",
    )
    assert "w" in res.quorum_telemetry
    tel = res.quorum_telemetry["w"]
    assert tel.num_admitted == 3
    assert tel.num_deferred == 1
    assert tel.merged_delta_norm == pytest.approx(2.0, rel=1e-5)
    # With outer_lr=0.5 and momentum=0.0, theta_updated = theta_prev - 0.5 * delta -> norm = 1.0
    assert float(res.theta_updated["w"].norm()) == pytest.approx(1.0, rel=1e-5)
    # Straggler slow0 (coordinate 3) was deferred outside the grace window
    assert float(res.theta_updated["w"][3]) == 0.0


def test_e18b_cosine_damped_outer_hyperparams_iid_vs_non_iid() -> None:
    """Issue #308 / E18b: Orthogonal non-IID deltas keep Nesterov (mu=0.9, lr=0.7); aligned IID deltas damp to FedAvg (mu=0, lr=1.0)."""
    e = torch.eye(8)
    # Orthogonal non-IID deltas: cosine = 0 -> mu_eff = 0.9, lr_eff = 0.7
    non_iid = compute_cosine_damped_outer_hyperparams([e[0], e[1], e[2]], base_lr=0.7, base_momentum=0.9)
    assert non_iid["mean_pairwise_cosine"] == pytest.approx(0.0, abs=1e-6)
    assert non_iid["effective_momentum"] == pytest.approx(0.9, abs=1e-6)
    assert non_iid["effective_lr"] == pytest.approx(0.7, abs=1e-6)
    assert non_iid["regime"] == "non_iid_nesterov"

    # Perfectly aligned IID deltas: cosine = 1.0 -> mu_eff = 0.0, lr_eff = 1.0 (FedAvg)
    v = torch.tensor([1.0, 2.0, -1.0, 0.5])
    iid = compute_cosine_damped_outer_hyperparams([v, v * 2.0, v * 0.5], base_lr=0.7, base_momentum=0.9)
    assert iid["mean_pairwise_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert iid["effective_momentum"] == pytest.approx(0.0, abs=1e-6)
    assert iid["effective_lr"] == pytest.approx(1.0, abs=1e-6)
    assert iid["regime"] == "iid_fedavg_damped"

    with pytest.raises(ValueError):
        compute_cosine_damped_outer_hyperparams([v])


def test_e18b_cosine_damped_momentum_report_artifact() -> None:
    """Issue #308 / E18b: Verify pre-registered E18b T4 artifact passes P18b_1..3 and respects T4 VRAM floor."""
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    report_path = repo_root / "artifacts" / "experiments" / "E18b_cosine_damped_momentum_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["experiment_id"] == "E18b"
    assert report["all_preregistered_hypotheses_passed"] is True
    assert report["hardware"]["t4_policy_4300mib_floor_respected"] is True
    assert report["hypothesis_verdicts"]["P18b_1_iid_overshoot_eliminated"]["passed"] is True
    assert report["hypothesis_verdicts"]["P18b_2_non_iid_gain_preserved"]["passed"] is True
    assert report["hypothesis_verdicts"]["P18b_3_uniform_dominance_over_inverted_control"]["passed"] is True


