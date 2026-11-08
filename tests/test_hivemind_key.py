# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for scripts/hivemind_key.py (create, check, and rotate gateway API keys)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hivemind_key.py"
spec = importlib.util.spec_from_file_location("hivemind_key", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
hm_key = importlib.util.module_from_spec(spec)
sys.modules["hivemind_key"] = hm_key
spec.loader.exec_module(hm_key)


def test_summary_and_check_subcommand(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def _call(path: str, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        assert path == "/v1/hivemind/me"
        if kwargs.get("key") == "valid-token":
            return 200, {"handle": "alice", "role": "trainer", "balance": 12.5}
        return 401, {"error": {"message": "invalid key"}}

    monkeypatch.setattr(hm_key, "call", _call)
    assert hm_key.summary("valid-token", hm_key.GATEWAY) == "valid: handle=alice role=trainer balance=12.5"
    assert hm_key.summary("bad-token", hm_key.GATEWAY) == "key rejected (HTTP 401)"

    monkeypatch.setenv("HIVEMIND_API_KEY", "valid-token")
    assert hm_key.main(["check"]) == 0
    assert "handle=alice" in capsys.readouterr().out


def test_new_and_rotate_subcommands_enforce_admin_and_handle_refusals(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def _call(path: str, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        calls.append((path, kwargs))
        if path == "/v1/hivemind/register":
            return 200, {"account_id": "acc-1", "api_key": "issued-token"}
        if path == "/v1/hivemind/admin/accounts/rotate-key":
            return 200, {"account_id": "acc-1", "api_key": "rotated-token"}
        return 200, {"handle": "alice", "role": "consumer", "balance": 0.0}

    monkeypatch.setattr(hm_key, "call", _call)
    assert hm_key.main(["new", "--handle", "alice", "--role", "consumer"]) == 0
    out = capsys.readouterr().out
    assert "account: acc-1" in out and "valid: handle=alice" in out

    # Negative control: rotate without HIVEMIND_ADMIN_KEY fails with exit code 2.
    monkeypatch.delenv("HIVEMIND_ADMIN_KEY", raising=False)
    assert hm_key.main(["rotate", "--account-id", "acc-1"]) == 2

    # Positive control: rotate with HIVEMIND_ADMIN_KEY and --account-id succeeds.
    monkeypatch.setenv("HIVEMIND_ADMIN_KEY", "admin-token-for-test")
    assert hm_key.main(["rotate", "--account-id", "acc-1"]) == 0
    assert any(p == "/v1/hivemind/admin/accounts/rotate-key" for p, _ in calls)
