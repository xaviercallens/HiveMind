"""Rejected submissions must not lose accepted work or alter accounting."""
import pytest
import torch

from hivemind.swarm.decoupled_diloco import ContinuousAsyncLearnerCoordinator


@pytest.mark.parametrize("mode", ContinuousAsyncLearnerCoordinator.STRAGGLER_MODES)
def test_duplicate_fresh_submission_preserves_first_update(mode):
    coordinator = ContinuousAsyncLearnerCoordinator(2, 2, mode, "direct")
    coordinator.submit_learner_delta("a", 0, torch.tensor([2.0]), 100, 1)
    with pytest.raises(ValueError, match="already.*pending"):
        coordinator.submit_learner_delta("a", 0, torch.tensor([20.0]), 900, 1)
    coordinator.submit_learner_delta("b", 0, torch.tensor([4.0]), 100, 1)
    result, report = coordinator.close_quorum_round(torch.tensor([10.0]), outer_lr=1.0)
    torch.testing.assert_close(result, torch.tensor([7.0]))
    assert report["total_submitted_tokens"] == report["merged_tokens"] == 200
    assert report["token_conservation_holds"]


@pytest.mark.parametrize("mode", ContinuousAsyncLearnerCoordinator.STRAGGLER_MODES)
def test_future_round_submission_is_rejected_without_mutation(mode):
    coordinator = ContinuousAsyncLearnerCoordinator(2, 2, mode, "direct")
    with pytest.raises(ValueError, match="future"):
        coordinator.submit_learner_delta("a", 1, torch.tensor([20.0]), 900, 1)
    result, report = coordinator.close_quorum_round(torch.tensor([10.0]), outer_lr=1.0)
    torch.testing.assert_close(result, torch.tensor([10.0]))
    assert coordinator.current_round == 0
    assert report["total_submitted_tokens"] == report["pending_tokens"] == 0
    assert report["merged_tokens"] == report["dropped_tokens"] == 0
    assert report["token_conservation_holds"]
