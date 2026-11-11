# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""A seeded dashboard account (owner, MasterMinders) is opened only by its token, its own password, or a password the
gateway verifies for the SAME handle. A stranger can neither choose its password on first sign-in nor borrow a gateway
account registered under one of its (unverified) e-mail addresses."""
from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app

SEEDED = ["xavier_callens", "callensxavier@gmail.com", "vxtzq", "clark719664"]


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_hivemind_at_home_app(vault_dir=tmp_path / "vault"))


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_gateway(monkeypatch: pytest.MonkeyPatch, answer_handle: str | None, seen: list) -> None:
    """Stand-in for the gateway's /api/v1/auth/login: says ok for any password, as the account ``answer_handle``."""
    monkeypatch.setenv("HIVEMIND_OPENROUTER_URL", "http://gateway.invalid")

    def urlopen(req, timeout=None):
        body = json.loads(req.data.decode("utf-8"))
        seen.append(body.get("handle_or_email"))
        if answer_handle is None:
            raise OSError("gateway down")
        return _Resp(json.dumps({"ok": True, "handle": answer_handle, "balance": 1.0}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)


@pytest.mark.parametrize("ident", SEEDED)
def test_a_stranger_cannot_choose_a_seeded_accounts_password(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ident: str) -> None:
    _fake_gateway(monkeypatch, None, [])  # gateway unreachable: no one vouches for the password
    client = _client(tmp_path)
    r = client.post("/api/v1/auth/login", json={"handle_or_email": ident, "password": "attacker-chosen-pw"})
    assert r.status_code == 401 and "token" not in r.json()
    # and the attempt did not bind the password: the same password still fails
    r2 = client.post("/api/v1/auth/login", json={"handle_or_email": ident, "password": "attacker-chosen-pw"})
    assert r2.status_code in (401, 429) and "token" not in r2.json()


@pytest.mark.parametrize("ident", ["vxtzq", "xavier_callens"])
def test_a_gateway_account_with_another_handle_does_not_open_a_seeded_account(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ident: str) -> None:
    seen: list = []
    _fake_gateway(monkeypatch, "attacker_account", seen)  # the gateway vouches, but for a different account
    r = _client(tmp_path).post("/api/v1/auth/login", json={"handle_or_email": ident, "password": "attacker-chosen-pw"})
    assert r.status_code == 401 and "token" not in r.json()
    assert all("@" not in str(s) for s in seen), f"the gateway was asked by e-mail: {seen}"


def test_the_gateway_password_of_the_same_handle_still_opens_the_account(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list = []
    _fake_gateway(monkeypatch, "vxtzq", seen)
    client = _client(tmp_path)
    r = client.post("/api/v1/auth/login", json={"handle_or_email": "vxtzq", "password": "the-real-gateway-pw"})
    assert r.status_code == 200 and r.json()["token"].startswith("hm_live_")
    assert seen == ["vxtzq"]
