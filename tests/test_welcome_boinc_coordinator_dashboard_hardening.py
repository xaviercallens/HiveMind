# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Input-validation hardening tests for welcome_email, coordinator_api, hivemind_at_home_dashboard, and workunit."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import pytest
import torch

from hivemind.boinc.coordinator_api import BOINCCoordinatorService
from hivemind.boinc.hivemind_at_home_dashboard import (
    _hash_password,
    portal_csp,
    sanitize_prompt_text,
)
from hivemind.boinc.workunit import (
    create_signed_parquet_shard,
    verify_and_load_parquet_shard,
)
from hivemind.config import HiveMindConfig
from hivemind.router.welcome_email import _mask_email, build_welcome_email_content
from hivemind.security.swarm_runtime_hardening import SwarmSecurityViolationError


def _make_coordinator(tmp_path: Path) -> BOINCCoordinatorService:
    cfg = HiveMindConfig()
    initial_lora = {"lora_A": torch.zeros(2, 4), "lora_B": torch.zeros(4, 2)}
    return BOINCCoordinatorService(
        config=cfg,
        initial_lora_state=initial_lora,
        base_model_sha256="a" * 64,
        shards_dir=tmp_path / "shards",
        vocab_size=64,
    )


def test_welcome_email_validation() -> None:
    content = build_welcome_email_content(
        account_id="acc_123",
        handle="alice",
        role="prosumer",
        balance_hmt=100.0,
        recipient_email="alice@example.org",
    )
    assert content["recipient_email_masked"] == "al***@example.org"
    assert _mask_email("a@example.org") == "a***@example.org"

    with pytest.raises(ValueError, match="recipient_email"):
        build_welcome_email_content(
            account_id="acc_123",
            handle="alice",
            role="prosumer",
            balance_hmt=100.0,
            recipient_email=12345,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="email"):
        _mask_email(None)  # type: ignore[arg-type]


def test_coordinator_api_validation(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)
    rec = coord.register_or_get_node("76561198000000001", "worker-01", is_non_limited=True)
    assert rec.steam_id64 == "76561198000000001"

    with pytest.raises(ValueError, match="is_non_limited"):
        coord.register_or_get_node("76561198000000001", "worker-01", is_non_limited=1)  # type: ignore[arg-type]

    with pytest.raises(SwarmSecurityViolationError):
        coord.register_or_get_node(True, "worker-01")  # type: ignore[arg-type]
    with pytest.raises(SwarmSecurityViolationError):
        coord.register_or_get_node("76561198000000001", 12345)  # type: ignore[arg-type]

    wu, _ = coord.assign_workunit(
        worker_id="worker-01",
        steam_id64="76561198000000001",
        num_sequences=4,
        seq_len=8,
    )
    dummy_model = MagicMock()
    dummy_sub = MagicMock()

    for bad_sid in ("", "   ", None, 123):
        with pytest.raises(ValueError, match="steam_id64"):
            coord.validate_and_record_submission(
                dummy_model, wu, dummy_sub, steam_id64=bad_sid  # type: ignore[arg-type]
            )

    for bad_lr in (True, 0.0, -1e-4, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="lr"):
            coord.validate_and_record_submission(
                dummy_model, wu, dummy_sub, steam_id64="76561198000000001", lr=bad_lr  # type: ignore[arg-type]
            )

    for bad_wd in (True, -0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="weight_decay"):
            coord.validate_and_record_submission(
                dummy_model, wu, dummy_sub, steam_id64="76561198000000001", weight_decay=bad_wd  # type: ignore[arg-type]
            )

    for bad_mbs in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="micro_batch_size"):
            coord.validate_and_record_submission(
                dummy_model, wu, dummy_sub, steam_id64="76561198000000001", micro_batch_size=bad_mbs  # type: ignore[arg-type]
            )


def test_dashboard_helpers_validation() -> None:
    salt_hex, digest_hex = _hash_password("CorrectHorseBatteryStaple", salt=b"12345678")
    assert len(salt_hex) == 16 and len(digest_hex) == 64

    for bad_pw in ("", None, 12345):
        with pytest.raises(ValueError, match="password"):
            _hash_password(bad_pw)  # type: ignore[arg-type]

    for bad_salt in (b"short", "12345678", 12345):
        with pytest.raises(ValueError, match="salt"):
            _hash_password("ValidPass", salt=bad_salt)  # type: ignore[arg-type]

    assert sanitize_prompt_text("hello world", max_chars=5) == "hello"
    for bad_max in (True, 0, -5, 10.5):
        with pytest.raises(ValueError, match="max_chars"):
            sanitize_prompt_text("hello", max_chars=bad_max)  # type: ignore[arg-type]

    assert "frame-ancestors 'none'" in portal_csp(False)
    assert "frame-ancestors 'self'" in portal_csp(True)
    for bad_flag in (1, "true", None):
        with pytest.raises(ValueError, match="allow_sameorigin_frame"):
            portal_csp(bad_flag)  # type: ignore[arg-type]


def test_workunit_parquet_shard_validation(tmp_path: Path) -> None:
    shard_file = tmp_path / "shard.parquet"
    path, sha, sig = create_signed_parquet_shard(
        shard_path=shard_file,
        branch_name="code",
        rng_seed=42,
        num_sequences=4,
        seq_len=8,
        vocab_size=64,
    )
    tensor = verify_and_load_parquet_shard(path, sha, sig, rng_seed=42)
    assert tensor.shape == (4, 8)

    for bad_path in ("", "   ", None, 123):
        with pytest.raises(ValueError, match="shard_path"):
            create_signed_parquet_shard(bad_path, "code", 42)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="shard_path"):
            verify_and_load_parquet_shard(bad_path, sha, sig, 42)  # type: ignore[arg-type]

    for bad_sha in ("", "   ", None, 123):
        with pytest.raises(ValueError, match="expected_sha256"):
            verify_and_load_parquet_shard(path, bad_sha, sig, 42)  # type: ignore[arg-type]

    for bad_sig in ("", "   ", None, 123):
        with pytest.raises(ValueError, match="expected_hmac_signature"):
            verify_and_load_parquet_shard(path, sha, bad_sig, 42)  # type: ignore[arg-type]
