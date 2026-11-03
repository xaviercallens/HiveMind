"""IP-238: non-finite tensors must fail the validator regardless of parameter key order."""

from __future__ import annotations

import math

import pytest
import torch

from hivemind.boinc import validator

NAN_TENSOR = torch.tensor([0.0, float("nan")])
ZERO_TENSOR = torch.tensor([0.0, 0.0])


@pytest.mark.parametrize("nan_key,finite_key", [("a", "b"), ("b", "a"), ("a", "z")])
def test_two_statistic_replay_rejects_nan_in_any_key_position(nan_key: str, finite_key: str) -> None:
    worker = {finite_key: ZERO_TENSOR.clone(), nan_key: NAN_TENSOR.clone()}
    reference = {finite_key: ZERO_TENSOR.clone(), nan_key: ZERO_TENSOR.clone()}
    result = validator.DeterministicSpotCheckValidator().evaluate_two_statistic_replay(worker, reference)
    assert result.passed is False
    assert result.reason.startswith(validator.NON_FINITE_TENSOR)
    assert result.linf_divergence == math.inf


def test_two_statistic_replay_rejects_nan_in_validator_state() -> None:
    worker = {"a": ZERO_TENSOR.clone(), "b": ZERO_TENSOR.clone()}
    reference = {"a": NAN_TENSOR.clone(), "b": ZERO_TENSOR.clone()}
    result = validator.DeterministicSpotCheckValidator().evaluate_two_statistic_replay(worker, reference)
    assert result.passed is False
    assert result.reason.startswith(validator.NON_FINITE_TENSOR)


def test_two_statistic_replay_still_passes_identical_finite_states() -> None:
    state = {"a": ZERO_TENSOR.clone(), "b": ZERO_TENSOR.clone()}
    result = validator.DeterministicSpotCheckValidator().evaluate_two_statistic_replay(
        state, {k: v.clone() for k, v in state.items()},
    )
    assert result.passed is True
    assert result.reason.startswith("PASSED")


@pytest.mark.parametrize("nan_key", ["a", "b"])
def test_fixed_point_digest_rejects_nan_in_any_key_position(nan_key: str) -> None:
    state = {"a": ZERO_TENSOR.clone(), "b": ZERO_TENSOR.clone()}
    state[nan_key] = NAN_TENSOR.clone()
    result = validator.verify_fixed_point_q16_digest(state, {k: v.clone() for k, v in state.items()})
    assert result["passed"] is False
    assert result["reason"] == validator.NON_FINITE_TENSOR


def test_fixed_point_digest_rejects_nan_against_finite_state() -> None:
    result = validator.verify_fixed_point_q16_digest(
        {"w": NAN_TENSOR.clone()}, {"w": ZERO_TENSOR[:1].clone().repeat(2)},
    )
    assert result["passed"] is False
    assert result["reason"] == validator.NON_FINITE_TENSOR


def test_fixed_point_quantizer_raises_on_infinity() -> None:
    with pytest.raises(ValueError, match=validator.NON_FINITE_TENSOR):
        validator.quantize_fixed_point_q16_state({"w": torch.tensor([math.inf])})
