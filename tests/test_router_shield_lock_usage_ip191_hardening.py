# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Iteration 191 negative-control and positive-control hardening tests for router modules."""

from __future__ import annotations

import pytest

from hivemind.router import forwarded_client, password_reset_email
from hivemind.router.hivemind_producer_shield import _error, create_shield_app
from hivemind.router.litestream_lock import (
    InMemoryGenerationStore,
    LitestreamWriterLock,
    LockGenerationRecord,
)
from hivemind.router.usage_capture import (
    ClientAttribution,
    PromptIntent,
    compute_nocturnal_moe_workunit_weights,
)


def test_forwarded_client_ip_validation_and_negative_now_hardening() -> None:
    secret = bytearray(b"0123456789abcdef0123456789abcdef")
    header = forwarded_client.sign(secret, "2001:db8::1", 1000.0)
    assert forwarded_client.verify(secret, header, 1005.0) == "2001:db8::1"
    assert forwarded_client.verify(secret, header, -10.0) is None

    for bad_ip in ("203.0.113.42 with space", "203.0.113.42\n", "203.0.113.42\x00", "203.0.113.42\u00e9"):
        with pytest.raises(ValueError, match="without '\\|' or whitespace/control characters"):
            forwarded_client.sign(secret, bad_ip, 1000.0)


def test_producer_shield_error_and_config_hardening() -> None:
    resp = _error(401, "Missing token.", "unauthorized")
    assert resp.status_code == 401

    for bad_status in (True, 200, 399, 600, "401"):
        with pytest.raises(ValueError, match="status must be an HTTP error status integer"):
            _error(bad_status, "msg", "code")  # type: ignore[arg-type]
    for bad_msg, bad_code in (("", "code"), ("   ", "code"), ("msg", ""), ("msg", "   ")):
        with pytest.raises(ValueError, match="message and code must be non-empty strings"):
            _error(400, bad_msg, bad_code)

    tok = "secret-endpoint-token-123456"
    with pytest.raises(ValueError, match="without control characters"):
        create_shield_app("http://127.0.0.1:11434\n", tok)
    with pytest.raises(ValueError, match="without control characters"):
        create_shield_app("http://127.0.0.1:11434", tok + "\n")
    for bad_allowed in ("qwen2.5:7b", [""], ["   "], ["qwen\n7b"]):
        with pytest.raises(ValueError, match="allowed_models must be a sequence"):
            create_shield_app("http://127.0.0.1:11434", tok, allowed_models=bad_allowed)  # type: ignore[arg-type]
    for bad_routes in (
        ["not-mapping"],
        {"": "http://127.0.0.1:8000"},
        {"m1\n": "http://127.0.0.1:8000"},
        {"m1": "ftp://127.0.0.1:8000"},
        {"m1": "http://127.0.0.1:8000\n"},
    ):
        with pytest.raises(ValueError, match="routes must map non-empty model strings"):
            create_shield_app("http://127.0.0.1:11434", tok, routes=bad_routes)  # type: ignore[arg-type]
    for bad_ctrl in ("", "   ", "/tmp/state\x00.json", "/tmp/state\n.json"):
        with pytest.raises(ValueError, match="control_state_file must be a non-empty string"):
            create_shield_app("http://127.0.0.1:11434", tok, control_state_file=bad_ctrl)


def test_litestream_lock_control_chars_and_negative_now_utc_hardening() -> None:
    rec = LockGenerationRecord("gs://bucket/replica", "owner-1", 1, 100.0, 400.0)
    assert rec.generation == 1

    with pytest.raises(ValueError, match="without control characters"):
        LockGenerationRecord("gs://bucket/replica\n", "owner-1", 1, 100.0, 400.0)
    with pytest.raises(ValueError, match="without control characters"):
        LockGenerationRecord("gs://bucket/replica", "owner-1\x00", 1, 100.0, 400.0)

    mem = InMemoryGenerationStore()
    with pytest.raises(ValueError, match="without control characters"):
        LitestreamWriterLock("gs://bucket/replica\r", "owner-1", mem)
    with pytest.raises(ValueError, match="without control characters"):
        LitestreamWriterLock("gs://bucket/replica", "owner-1\n", mem)

    lock = LitestreamWriterLock("gs://bucket/replica", "owner-1", mem)
    with pytest.raises(ValueError, match="now_utc must be a finite number >= 0"):
        lock.acquire(now_utc=-1.0)


def test_password_reset_email_and_usage_capture_hardening() -> None:
    env = {
        "HIVEMIND_SMTP_HOST": "smtp.example.org",
        "HIVEMIND_SMTP_FROM": "noreply@example.org",
        "HIVEMIND_PUBLIC_WEB_URL": "https://hivemind.example.org",
    }
    for bad_token in ("tok with space", "tok#injected", "tok&extra=1", "tok\ninjected", "tok\x00"):
        with pytest.raises(ValueError, match="URL-safe string"):
            password_reset_email.reset_link(bad_token, env)

    valid_link = password_reset_email.reset_link("tok-safe-123", env)
    with pytest.raises(ValueError, match="without control characters"):
        password_reset_email.build_reset_message("user@example.org\r\nBcc:evil@example.org", valid_link, "noreply@example.org")
    with pytest.raises(ValueError, match="without control characters"):
        password_reset_email.build_reset_message("user@example.org", valid_link + "\nInjected", "noreply@example.org")
    with pytest.raises(ValueError, match="without control characters"):
        password_reset_email.build_reset_message("user@example.org", valid_link, "noreply@example.org\n")

    intent = PromptIntent("coding_python", "en", "coder")
    assert intent.task_domain == "coding_python"
    with pytest.raises(ValueError, match="task_domain must be one of"):
        PromptIntent("unknown_domain", "en", None)
    with pytest.raises(ValueError, match="language must be one of"):
        PromptIntent("coding_python", "de", None)
    with pytest.raises(ValueError, match="without control characters"):
        PromptIntent("coding_python", "en", "coder\n")

    attr = ClientAttribution("cursor", "Cursor IDE")
    assert attr.client_sdk == "cursor"
    with pytest.raises(ValueError, match="client_sdk must be one of"):
        ClientAttribution("unknown_sdk", "Title")
    with pytest.raises(ValueError, match="without control characters"):
        ClientAttribution("cursor", "Cursor\nIDE")

    for bad_totals in ({"": 10}, {"   ": 10}, {"coding_python\n": 10}):
        with pytest.raises(ValueError, match="domain_token_totals keys must be non-empty strings"):
            compute_nocturnal_moe_workunit_weights(bad_totals)
