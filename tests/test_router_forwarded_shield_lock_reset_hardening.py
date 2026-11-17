"""Negative-control tests for forwarded_client, hivemind_producer_shield, litestream_lock, and password_reset_email."""

from __future__ import annotations

from pathlib import Path

import pytest

from hivemind.router import forwarded_client, password_reset_email
from hivemind.router.hivemind_producer_shield import create_shield_app
from hivemind.router.litestream_lock import (
    FileGenerationStore,
    InMemoryGenerationStore,
    LitestreamWriterLock,
)


def test_forwarded_client_sign_and_verify_reject_invalid_inputs() -> None:
    secret = b"0123456789abcdef0123456789abcdef"
    header = forwarded_client.sign(secret, "203.0.113.42", 1_700_000_000.0)
    assert forwarded_client.verify(secret, header, 1_700_000_010.0) == "203.0.113.42"

    with pytest.raises(ValueError, match="secret"):
        forwarded_client.sign(b"", "203.0.113.42", 100.0)
    with pytest.raises(ValueError, match="secret"):
        forwarded_client.sign("str-secret", "203.0.113.42", 100.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ip"):
        forwarded_client.sign(secret, "   ", 100.0)
    with pytest.raises(ValueError, match="ip"):
        forwarded_client.sign(secret, "203.0.113.42|injected", 100.0)
    with pytest.raises(ValueError, match="now"):
        forwarded_client.sign(secret, "203.0.113.42", True)
    with pytest.raises(ValueError, match="now"):
        forwarded_client.sign(secret, "203.0.113.42", -1.0)
    with pytest.raises(ValueError, match="now"):
        forwarded_client.sign(secret, "203.0.113.42", float("nan"))

    assert forwarded_client.verify(b"", header, 1_700_000_010.0) is None
    assert forwarded_client.verify("bad", header, 1_700_000_010.0) is None  # type: ignore[arg-type]
    assert forwarded_client.verify(secret, 12345, 1_700_000_010.0) is None  # type: ignore[arg-type]
    assert forwarded_client.verify(secret, header, True) is None
    assert forwarded_client.verify(secret, header, float("nan")) is None


def test_create_shield_app_rejects_invalid_upstream_token_and_bool_limits() -> None:
    valid_token = "secret-endpoint-token-123456"
    app = create_shield_app("http://127.0.0.1:11434", valid_token, max_concurrent=2, max_body_bytes=4096)
    assert app is not None

    with pytest.raises(ValueError, match="upstream"):
        create_shield_app("   ", valid_token)
    with pytest.raises(ValueError, match="upstream"):
        create_shield_app("ftp://127.0.0.1:11434", valid_token)
    with pytest.raises(ValueError, match="token"):
        create_shield_app("http://127.0.0.1:11434", 12345678901234567890)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_concurrent"):
        create_shield_app("http://127.0.0.1:11434", valid_token, max_concurrent=True)
    with pytest.raises(ValueError, match="max_body_bytes"):
        create_shield_app("http://127.0.0.1:11434", valid_token, max_body_bytes=True)


def test_litestream_lock_and_stores_reject_invalid_keys_and_options(tmp_path: Path) -> None:
    mem = InMemoryGenerationStore()
    fs = FileGenerationStore(tmp_path / "locks")

    assert mem.get_record("valid_key") is None
    assert fs.get_record("valid_key") is None

    for store in (mem, fs):
        with pytest.raises(ValueError, match="key"):
            store.get_record("   ")
        with pytest.raises(ValueError, match="key"):
            store.get_record(None)  # type: ignore[arg-type]

    lock = LitestreamWriterLock("gs://bucket/replica", "owner-1", mem, fail_closed=True)
    assert lock.acquire(now_utc=1000.0) is True

    with pytest.raises(ValueError, match="replica_url"):
        LitestreamWriterLock(None, "owner-1", mem)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fail_closed"):
        LitestreamWriterLock("gs://bucket/replica", "owner-1", mem, fail_closed="yes")  # type: ignore[arg-type]


def test_password_reset_email_helpers_reject_invalid_inputs() -> None:
    env = {
        "HIVEMIND_SMTP_HOST": "smtp.example.org",
        "HIVEMIND_SMTP_FROM": "noreply@example.org",
        "HIVEMIND_PUBLIC_WEB_URL": "https://hivemind.example.org",
    }
    link = password_reset_email.reset_link("tok-abc-123", env)
    assert link == "https://hivemind.example.org/auth/reset#token=tok-abc-123"
    msg = password_reset_email.build_reset_message("user@example.org", link, "noreply@example.org")
    assert msg["To"] == "user@example.org"

    with pytest.raises(ValueError, match="token"):
        password_reset_email.reset_link("   ", env)
    with pytest.raises(ValueError, match="token"):
        password_reset_email.reset_link(None, env)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="recipient"):
        password_reset_email.build_reset_message("invalid-recipient", link, "noreply@example.org")
    with pytest.raises(ValueError, match="link"):
        password_reset_email.build_reset_message("user@example.org", "   ", "noreply@example.org")
    with pytest.raises(ValueError, match="sender"):
        password_reset_email.build_reset_message("user@example.org", link, "invalid-sender")

    assert password_reset_email.send_reset_email("user@example.org", "   ", env) is False
    assert password_reset_email.send_reset_email(None, "tok-abc-123", env) is False  # type: ignore[arg-type]
