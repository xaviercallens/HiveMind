"""Regression tests for workunit ownership (FL-08) and consume-after-verdict ordering (FL-05).

Before this fix ``verify_workunit_submission_envelope`` never checked that the submitting
``(worker_id, steam_id64)`` was the pair the workunit was issued to, and it marked the workunit
consumed as soon as the envelope passed, before the spot-check validator produced a verdict.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hivemind.boinc.coordinator_api import BOINCCoordinatorService
from hivemind.client.worker_node import HiveMindWorkerNode
from hivemind.config import HiveMindConfig
from hivemind.core.quantization import (
    attach_qlora_adapters,
    build_local_qwen2_backbone,
    extract_lora_state_dict,
)
from hivemind.security.federated_learning_hardening import FederatedLearningSecurityGuard
from hivemind.security.swarm_runtime_hardening import SwarmSecurityViolationError

OWNER_SID = "76561198000000001"
OWNER_WID = "w-owner"
OTHER_SID = "76561198000000002"
OTHER_WID = "w-other"


def _envelope_kwargs(**overrides):
    base = dict(
        wu_id="wu-own-1",
        expected_branch="Astro",
        submitted_branch="Astro",
        coordinator_outer_step=3,
        submitted_outer_step=3,
        steam_id64=OWNER_SID,
        worker_id=OWNER_WID,
        steps_completed=10,
        expected_inner_steps=10,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Guard-level unit tests (no model needed)
# ---------------------------------------------------------------------------


def test_fl08_submission_from_non_assignee_is_rejected_and_does_not_consume() -> None:
    guard = FederatedLearningSecurityGuard()
    guard.register_workunit_assignment("wu-own-1", worker_id=OWNER_WID, steam_id64=OWNER_SID)

    with pytest.raises(SwarmSecurityViolationError) as exc:
        guard.verify_workunit_submission_envelope(**_envelope_kwargs(steam_id64=OTHER_SID, worker_id=OTHER_WID))
    assert exc.value.rule_id == "FL-08"
    assert "WORKUNIT_ASSIGNEE_MISMATCH" in str(exc.value)
    assert "wu-own-1" not in guard.consumed_workunit_ids

    # The rightful assignee can still submit afterwards.
    guard.verify_workunit_submission_envelope(**_envelope_kwargs())
    assert "wu-own-1" in guard.consumed_workunit_ids


@pytest.mark.parametrize(
    "overrides",
    [
        {"steam_id64": OTHER_SID},  # same worker_id, different Steam account
        {"worker_id": OTHER_WID},  # same Steam account, different worker process
    ],
)
def test_fl08_assignee_match_requires_both_worker_id_and_steam_id(overrides: dict) -> None:
    guard = FederatedLearningSecurityGuard()
    guard.register_workunit_assignment("wu-own-1", worker_id=OWNER_WID, steam_id64=OWNER_SID)
    with pytest.raises(SwarmSecurityViolationError) as exc:
        guard.verify_workunit_submission_envelope(**_envelope_kwargs(**overrides))
    assert exc.value.rule_id == "FL-08"


def test_check_envelope_does_not_consume_until_record_is_called() -> None:
    guard = FederatedLearningSecurityGuard()
    guard.register_workunit_assignment("wu-own-1", worker_id=OWNER_WID, steam_id64=OWNER_SID)

    guard.check_workunit_submission_envelope(**_envelope_kwargs())
    assert "wu-own-1" not in guard.consumed_workunit_ids
    # A second pure check is still allowed: nothing has been committed yet.
    guard.check_workunit_submission_envelope(**_envelope_kwargs())

    guard.record_workunit_submission(
        wu_id="wu-own-1",
        expected_branch="Astro",
        coordinator_outer_step=3,
        steam_id64=OWNER_SID,
        worker_id=OWNER_WID,
    )
    assert "wu-own-1" in guard.consumed_workunit_ids
    with pytest.raises(SwarmSecurityViolationError) as exc:
        guard.check_workunit_submission_envelope(**_envelope_kwargs())
    assert exc.value.rule_id == "FL-05"
    assert "WORKUNIT_REPLAY_DETECTED" in str(exc.value)


def test_fl08_rejection_happens_before_fl06_vote_is_recorded() -> None:
    guard = FederatedLearningSecurityGuard()
    guard.register_workunit_assignment("wu-own-1", worker_id=OWNER_WID, steam_id64=OWNER_SID)
    guard.register_workunit_assignment("wu-own-2", worker_id=OTHER_WID, steam_id64=OTHER_SID)

    with pytest.raises(SwarmSecurityViolationError):
        guard.verify_workunit_submission_envelope(**_envelope_kwargs(steam_id64=OTHER_SID, worker_id=OTHER_WID))
    # The rejected attempt must not have burned OTHER_SID's one FL-06 vote for (Astro, 3).
    guard.verify_workunit_submission_envelope(
        **_envelope_kwargs(wu_id="wu-own-2", steam_id64=OTHER_SID, worker_id=OTHER_WID)
    )


# ---------------------------------------------------------------------------
# Coordinator-level tests (tiny CPU Qwen2 + QLoRA, same shape as test_cov_game_and_client)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_bundle():
    base_model, base_sha, _ = build_local_qwen2_backbone(
        seed=42,
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        device=torch.device("cpu"),
        use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(
        base_model=base_model, rank_r=2, lora_alpha=4, gradient_checkpointing=False, seed=42
    )
    return bundle, extract_lora_state_dict(bundle.model), base_sha


def _coordinator(tiny_bundle, tmp_path: Path) -> BOINCCoordinatorService:
    _, init_lora, base_sha = tiny_bundle
    return BOINCCoordinatorService(
        config=HiveMindConfig.from_yaml(),
        initial_lora_state=init_lora,
        base_model_sha256=base_sha,
        shards_dir=tmp_path / "coord_shards",
        vocab_size=128,
    )


def _assign(coordinator: BOINCCoordinatorService, worker_id: str, steam_id64: str):
    return coordinator.assign_workunit(
        worker_id=worker_id,
        steam_id64=steam_id64,
        branch_name="Code",
        inner_steps_H=2,
        spot_check_interval_s=1,
        num_sequences=4,
        seq_len=8,
    )


def _worker(bundle, worker_id: str, steam_id64: str) -> HiveMindWorkerNode:
    return HiveMindWorkerNode(worker_id=worker_id, steam_id64=steam_id64, persona_name=worker_id, model=bundle.model)


def test_coordinator_rejects_other_worker_submitting_assigned_workunit(tiny_bundle, tmp_path: Path) -> None:
    bundle, _, _ = tiny_bundle
    coordinator = _coordinator(tiny_bundle, tmp_path)
    wu, state = _assign(coordinator, OWNER_WID, OWNER_SID)

    # An attacker who obtained the workunit payload runs it honestly and submits under its own identity.
    sub_other = _worker(bundle, OTHER_WID, OTHER_SID).execute_workunit(wu, state)
    with pytest.raises(SwarmSecurityViolationError, match="WORKUNIT_ASSIGNEE_MISMATCH") as exc:
        coordinator.validate_and_record_submission(
            validator_model=bundle.model, workunit=wu, submission=sub_other, steam_id64=OTHER_SID
        )
    assert exc.value.rule_id == "FL-08"
    assert wu.wu_id not in coordinator.fl_security_guard.consumed_workunit_ids
    assert OTHER_SID not in coordinator.reputation_ledger

    # The honest assignee's own submission still goes through.
    sub_owner = _worker(bundle, OWNER_WID, OWNER_SID).execute_workunit(wu, state)
    verdict = coordinator.validate_and_record_submission(
        validator_model=bundle.model, workunit=wu, submission=sub_owner, steam_id64=OWNER_SID
    )
    assert verdict.passed is True
    assert wu.wu_id in coordinator.fl_security_guard.consumed_workunit_ids


def test_coordinator_does_not_consume_workunit_when_validator_raises(tiny_bundle, tmp_path: Path, monkeypatch) -> None:
    bundle, _, _ = tiny_bundle
    coordinator = _coordinator(tiny_bundle, tmp_path)
    wu, state = _assign(coordinator, OWNER_WID, OWNER_SID)
    sub = _worker(bundle, OWNER_WID, OWNER_SID).execute_workunit(wu, state)

    real_verify = coordinator.validator.verify_submission

    def _boom(*args, **kwargs):
        raise RuntimeError("validator crashed before producing a verdict")

    monkeypatch.setattr(coordinator.validator, "verify_submission", _boom)
    with pytest.raises(RuntimeError, match="validator crashed"):
        coordinator.validate_and_record_submission(
            validator_model=bundle.model, workunit=wu, submission=sub, steam_id64=OWNER_SID
        )
    # No verdict was produced, so the workunit must still be open for the assignee.
    assert wu.wu_id not in coordinator.fl_security_guard.consumed_workunit_ids

    monkeypatch.setattr(coordinator.validator, "verify_submission", real_verify)
    verdict = coordinator.validate_and_record_submission(
        validator_model=bundle.model, workunit=wu, submission=sub, steam_id64=OWNER_SID
    )
    assert verdict.passed is True


def test_coordinator_failed_spot_check_still_consumes_workunit(tiny_bundle, tmp_path: Path) -> None:
    """A verdict (pass or fail) is final for a wu_id: a failed submitter gets no second attempt
    at the same deterministic shard, which would otherwise be a free oracle on the validator."""
    bundle, _, _ = tiny_bundle
    coordinator = _coordinator(tiny_bundle, tmp_path)
    wu, state = _assign(coordinator, OWNER_WID, OWNER_SID)
    sub_poisoned = _worker(bundle, OWNER_WID, OWNER_SID).execute_workunit(wu, state, byzantine_poison_scale=10.0)

    verdict = coordinator.validate_and_record_submission(
        validator_model=bundle.model, workunit=wu, submission=sub_poisoned, steam_id64=OWNER_SID
    )
    assert verdict.passed is False
    assert wu.wu_id in coordinator.fl_security_guard.consumed_workunit_ids

    sub_clean = _worker(bundle, OWNER_WID, OWNER_SID).execute_workunit(wu, state)
    with pytest.raises(SwarmSecurityViolationError, match="WORKUNIT_REPLAY_DETECTED"):
        coordinator.validate_and_record_submission(
            validator_model=bundle.model, workunit=wu, submission=sub_clean, steam_id64=OWNER_SID
        )


def test_coordinator_reuses_parquet_shards_and_caps_active_workunits_per_node(
    tiny_bundle, tmp_path: Path
) -> None:
    """IP-249: 1,000 workunit requests from one worker must reuse shards per (branch, outer_step)
    and cap active_workunits per node rather than writing 1,000 Parquet files."""
    coordinator = _coordinator(tiny_bundle, tmp_path)
    for _ in range(1000):
        coordinator.assign_workunit(
            worker_id=OWNER_WID,
            steam_id64=OWNER_SID,
            branch_name=None,
            inner_steps_H=2,
            spot_check_interval_s=1,
            num_sequences=4,
            seq_len=8,
        )
    parquet_files = list(coordinator.shards_dir.glob("*.parquet"))
    assert len(parquet_files) <= len(coordinator.branches)
    assert len(coordinator.active_workunits) <= coordinator.max_active_workunits_per_node
