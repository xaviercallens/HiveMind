"""Regression coverage for continuous learner quorum boundaries and token accounting."""

import pytest
import torch

from hivemind.swarm.decoupled_diloco import ContinuousAsyncLearnerCoordinator


@pytest.mark.parametrize("straggler_mode", ContinuousAsyncLearnerCoordinator.STRAGGLER_MODES)
@pytest.mark.parametrize("merge_mode", ("rda", "direct"))
def test_underfilled_round_waits_without_losing_fresh_work(straggler_mode, merge_mode):
    coord = ContinuousAsyncLearnerCoordinator(3, 2, straggler_mode, merge_mode)
    initial = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    coord.submit_learner_delta("a", 0, torch.tensor([2.0, 0.0]), 100, 1)

    for _ in range(2):
        waiting, report = coord.close_quorum_round(initial, outer_lr=0.5)
        torch.testing.assert_close(waiting, initial)
        assert coord.current_round == 0
        assert report["quorum_reached"] is False
        assert report["round_closed"] is None
        assert report["admitted_learners"] == []
        assert report["pending_tokens"] == 100
        assert report["merged_tokens"] == report["dropped_tokens"] == 0
        assert report["token_conservation_holds"] is True

    coord.submit_learner_delta("b", 0, torch.tensor([2.0, 0.0]), 100, 1)
    merged, report = coord.close_quorum_round(initial, outer_lr=0.5)

    torch.testing.assert_close(merged, torch.tensor([-1.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(initial, torch.zeros(2, dtype=torch.float64))
    assert merged.requires_grad is False
    assert coord.current_round == 1
    assert report["quorum_reached"] is True
    assert report["round_closed"] == 0
    assert report["admitted_learners"] == ["a", "b"]
    assert report["pending_tokens"] == report["dropped_tokens"] == 0
    assert report["merged_tokens"] == report["total_submitted_tokens"] == 200
    assert report["token_conservation_holds"] is True


@pytest.mark.parametrize("straggler_mode", ContinuousAsyncLearnerCoordinator.STRAGGLER_MODES)
def test_empty_round_does_not_advance(straggler_mode):
    coord = ContinuousAsyncLearnerCoordinator(2, 2, straggler_mode)
    initial = torch.tensor([3.0, -2.0])

    result, report = coord.close_quorum_round(initial)

    torch.testing.assert_close(result, initial)
    assert coord.current_round == 0
    assert report["quorum_reached"] is False
    assert report["round_closed"] is None
    assert report["pending_tokens"] == report["merged_tokens"] == 0
    assert report["token_conservation_holds"] is True


def test_deferred_work_survives_waiting_and_merges_once_quorum_arrives():
    coord = ContinuousAsyncLearnerCoordinator(3, 2, merge_mode="direct")
    initial = torch.zeros(2)
    for learner in ("a", "b"):
        coord.submit_learner_delta(learner, 0, torch.zeros(2), 100, 1)
    initial, _ = coord.close_quorum_round(initial)
    coord.submit_learner_delta("c", 0, torch.tensor([2.0, 0.0]), 300, 1)

    waiting, report = coord.close_quorum_round(initial, outer_lr=0.5)

    torch.testing.assert_close(waiting, initial)
    assert coord.current_round == 1
    assert report["quorum_reached"] is False
    assert report["pending_tokens"] == 300
    assert report["merged_tokens"] == 200
    assert report["token_conservation_holds"] is True

    coord.submit_learner_delta("a", 1, torch.tensor([2.0, 0.0]), 100, 1)
    merged, report = coord.close_quorum_round(waiting, outer_lr=0.5)

    torch.testing.assert_close(merged, torch.tensor([-1.0, 0.0]))
    assert coord.current_round == 2
    assert report["round_closed"] == 1
    assert report["quorum_reached"] is True
    assert set(report["admitted_learners"]) == {"a", "c"}
    assert report["merged_tokens"] == report["total_submitted_tokens"] == 600
    assert report["pending_tokens"] == report["dropped_tokens"] == 0
    assert report["token_conservation_holds"] is True

    unchanged, report = coord.close_quorum_round(merged, outer_lr=0.5)
    torch.testing.assert_close(unchanged, merged)
    assert coord.current_round == 2
    assert report["merged_tokens"] == 600


def test_eager_stale_step_drains_once_while_fresh_quorum_waits():
    coord = ContinuousAsyncLearnerCoordinator(3, 2, "eager_one_step", "direct", 0.5)
    initial = torch.zeros(2)
    for learner in ("a", "b"):
        coord.submit_learner_delta(learner, 0, torch.zeros(2), 100, 1)
    initial, _ = coord.close_quorum_round(initial)
    coord.submit_learner_delta("c", 0, torch.tensor([4.0, 0.0]), 150, 1)
    coord.submit_learner_delta("a", 1, torch.tensor([2.0, 0.0]), 100, 1)

    eager, report = coord.close_quorum_round(initial, outer_lr=0.5)

    torch.testing.assert_close(eager, torch.tensor([-1.0, 0.0]))
    assert coord.current_round == 1
    assert report["quorum_reached"] is False
    assert report["round_closed"] is None
    assert report["eager_applied_learners"] == ["c"]
    assert coord.eager_applied_steps == 1
    assert report["merged_tokens"] == 350
    assert report["pending_tokens"] == 100
    assert report["token_conservation_holds"] is True

    unchanged, report = coord.close_quorum_round(eager, outer_lr=0.5)
    torch.testing.assert_close(unchanged, eager)
    assert report["eager_applied_learners"] == []
    assert coord.eager_applied_steps == 1
    assert report["merged_tokens"] == 350

    coord.submit_learner_delta("b", 1, torch.tensor([2.0, 0.0]), 100, 1)
    merged, report = coord.close_quorum_round(unchanged, outer_lr=0.5)
    torch.testing.assert_close(merged, torch.tensor([-2.0, 0.0]))
    assert report["quorum_reached"] is True
    assert report["merged_tokens"] == report["total_submitted_tokens"] == 550
    assert report["pending_tokens"] == report["dropped_tokens"] == 0
    assert report["token_conservation_holds"] is True


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_learners_m": 0}, "num_learners_m must be >= 1"),
        ({"min_quorum_k": 0}, "min_quorum_k must satisfy"),
        ({"min_quorum_k": 4}, "min_quorum_k must satisfy"),
        ({"straggler_mode": "unknown"}, "straggler_mode must be one of"),
        ({"merge_mode": "unknown"}, "merge_mode must be one of"),
        ({"staleness_decay": 0.0}, "staleness_decay must be in"),
        ({"staleness_decay": 1.1}, "staleness_decay must be in"),
    ],
)
def test_invalid_coordinator_configuration_is_rejected(kwargs, message):
    options = {"num_learners_m": 3, "min_quorum_k": 2, **kwargs}
    with pytest.raises(ValueError, match=message):
        ContinuousAsyncLearnerCoordinator(**options)


@pytest.mark.parametrize("tokens, steps", [(-1, 1), (100, -1)])
def test_negative_work_counters_do_not_change_accounting(tokens, steps):
    coord = ContinuousAsyncLearnerCoordinator(2, 2)
    with pytest.raises(ValueError, match="must be >= 0"):
        coord.submit_learner_delta("a", 0, torch.ones(2), tokens, steps)
    assert coord.total_submitted_tokens == 0
    assert coord.current_round == 0
