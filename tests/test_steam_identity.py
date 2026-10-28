# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Steam identity: proof of a SteamID64, the Sybil facts from Steam, tokens, the coordinator gate and the bridge's secret. No network: HTTP is injected."""

from __future__ import annotations

import hashlib
import hmac
import time
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from hivemind.boinc.coordinator_api import NodeReputationRecord, create_coordinator_fastapi_app
from hivemind.client import steamworks
from hivemind.client.steam_identity import (
    OPENID_ENDPOINT,
    IdentityError,
    IdentityTokens,
    SteamIdentityGate,
    SteamOpenIDVerifier,
    fetch_account_facts,
    is_valid_steamid64,
    login_url,
)

RETURN_TO = "https://coord.example/identity/steam/callback"
SID = "76561197960287930"
OTHER = "76561198000000001"
NOW = 1_800_000_000.0


def nonce(at: float = NOW, salt: str = "Xk3") -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(at)) + salt


def assertion(sid: str = SID, **override: str) -> dict[str, str]:
    base = {
        "openid.ns": "http://specs.openid.net/auth/2.0", "openid.mode": "id_res", "openid.op_endpoint": OPENID_ENDPOINT,
        "openid.claimed_id": f"https://steamcommunity.com/openid/id/{sid}", "openid.identity": f"https://steamcommunity.com/openid/id/{sid}",
        "openid.return_to": RETURN_TO, "openid.response_nonce": nonce(), "openid.assoc_handle": "1234567890",
        "openid.signed": "signed,op_endpoint,claimed_id,identity,return_to,response_nonce,assoc_handle", "openid.sig": "c2ln",
    }
    base.update(override)
    return base


def profile_xml(sid: str = SID, limited: str = "0", vac: str = "0") -> str:
    return f"<profile><steamID64>{sid}</steamID64><vacBanned>{vac}</vacBanned><isLimitedAccount>{limited}</isLimitedAccount></profile>"


class FakeSteam:
    """Stands in for steamcommunity.com: confirms assertions (or not) and serves profile XML."""

    def __init__(self, *, valid: bool = True, limited: str = "0", vac: str = "0", down: bool = False, status: int = 200) -> None:
        self.valid, self.limited, self.vac, self.down, self.status = valid, limited, vac, down, status
        self.calls: list[tuple[str, str, dict[str, str] | None]] = []

    def __call__(self, method: str, url: str, form=None):  # noqa: ANN001, ANN204
        self.calls.append((method, url, dict(form) if form else None))
        if self.down:
            raise IdentityError("Steam could not be reached")
        if method == "POST":
            return self.status, f"ns:http://specs.openid.net/auth/2.0\nis_valid:{'true' if self.valid else 'false'}\n"
        sid = url.split("/profiles/")[1].split("?")[0]
        return self.status, profile_xml(sid, self.limited, self.vac)


def verifier(steam: FakeSteam | None = None) -> tuple[SteamOpenIDVerifier, FakeSteam]:
    steam = steam or FakeSteam()
    return SteamOpenIDVerifier(RETURN_TO, http=steam, clock=lambda: NOW), steam


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value,ok", [(SID, True), (OTHER, True), ("x", False), ("", False), ("7656119796028793", False), ("765611979602879300", False),
                                      ("76561197960287930\n", False), ("../76561197960287", False), (76561197960287930, False), (None, False)])
def test_steamid64_must_be_a_17_digit_individual_account(value: object, ok: bool) -> None:
    assert is_valid_steamid64(value) is ok


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# OpenID
# ---------------------------------------------------------------------------------------------------------------------------------------------------


def test_an_assertion_steam_confirms_yields_the_steamid_and_is_handed_back_to_steam_unchanged_except_the_mode() -> None:
    v, steam = verifier()
    params = assertion()
    assert v.verify_assertion(params) == SID
    method, url, form = steam.calls[0]
    assert (method, url) == ("POST", OPENID_ENDPOINT)
    assert form["openid.mode"] == "check_authentication" and form["openid.sig"] == "c2ln" and form["openid.claimed_id"] == params["openid.claimed_id"]


@pytest.mark.parametrize("override,why", [
    ({"openid.mode": "cancel"}, "not an OpenID 2.0 positive assertion"),
    ({"openid.ns": "http://example.com/other"}, "not an OpenID 2.0 positive assertion"),
    ({"openid.op_endpoint": "https://evil.example/openid/login"}, "not issued by Steam"),
    ({"openid.return_to": "https://other.example/cb"}, "different return URL"),
    ({"openid.signed": "signed,op_endpoint,claimed_id"}, "does not sign"),
    ({"openid.claimed_id": "https://evil.example/openid/id/" + SID}, "Steam individual account"),
    ({"openid.claimed_id": "https://steamcommunity.com/openid/id/12345"}, "Steam individual account"),
    ({"openid.identity": f"https://steamcommunity.com/openid/id/{OTHER}"}, "disagree"),
    ({"openid.response_nonce": "garbage"}, "malformed response nonce"),
    ({"openid.response_nonce": nonce(NOW - 301)}, "stale"),
    ({"openid.response_nonce": nonce(NOW + 120)}, "future"),
])
def test_assertions_that_are_not_bound_to_this_login_are_refused_before_steam_is_asked(override: dict[str, str], why: str) -> None:
    v, steam = verifier()
    with pytest.raises(IdentityError, match=why):
        v.verify_assertion(assertion(**override))
    assert steam.calls == []


def test_an_assertion_cannot_be_replayed_even_if_it_was_valid() -> None:
    v, _ = verifier()
    params = assertion()
    assert v.verify_assertion(params) == SID
    with pytest.raises(IdentityError, match="already used"):
        v.verify_assertion(params)


@pytest.mark.parametrize("steam", [FakeSteam(valid=False), FakeSteam(down=True), FakeSteam(status=500)])
def test_steam_denying_or_being_unreachable_never_proves_an_identity(steam: FakeSteam) -> None:
    v, _ = verifier(steam)
    with pytest.raises(IdentityError):
        v.verify_assertion(assertion())


def test_return_to_must_be_https_or_loopback() -> None:
    with pytest.raises(ValueError):
        SteamOpenIDVerifier("http://coord.example/cb")
    SteamOpenIDVerifier("http://127.0.0.1:8080/cb")


def test_login_url_points_at_steam_with_our_return_url() -> None:
    url = login_url(RETURN_TO, "https://coord.example")
    assert url.startswith(OPENID_ENDPOINT + "?") and "return_to=https%3A%2F%2Fcoord.example%2Fidentity%2Fsteam%2Fcallback" in url


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# facts from Steam
# ---------------------------------------------------------------------------------------------------------------------------------------------------


def test_limited_and_vac_flags_come_from_the_profile() -> None:
    facts = fetch_account_facts(SID, FakeSteam(limited="1", vac="1"))
    assert (facts.limited, facts.vac_banned) == (True, True)
    facts = fetch_account_facts(SID, FakeSteam())
    assert (facts.limited, facts.vac_banned) == (False, False)


@pytest.mark.parametrize("http", [
    lambda m, u, f=None: (404, ""),
    lambda m, u, f=None: (200, "<profile><steamID64>" + SID + "</steamID64></profile>"),
    lambda m, u, f=None: (200, profile_xml(OTHER)),
    lambda m, u, f=None: (200, profile_xml(limited="maybe")),
    lambda m, u, f=None: (200, ""),
])
def test_an_incomplete_or_foreign_profile_answer_fails_closed(http) -> None:  # noqa: ANN001
    with pytest.raises(IdentityError):
        fetch_account_facts(SID, http)


def test_facts_are_not_fetched_for_a_malformed_id() -> None:
    steam = FakeSteam()
    with pytest.raises(IdentityError):
        fetch_account_facts("../etc/passwd", steam)
    assert steam.calls == []


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# tokens and the gate
# ---------------------------------------------------------------------------------------------------------------------------------------------------


def test_tokens_roundtrip_expire_and_reject_tampering() -> None:
    clock = [NOW]
    tokens = IdentityTokens(secret=b"s" * 32, ttl_s=60, clock=lambda: clock[0])
    token = tokens.issue(SID)
    assert tokens.verify(token) == SID
    forged = token.replace(SID, OTHER)
    for bad in (forged, token[:-1] + ("0" if token[-1] != "0" else "1"), "", "v1.x", "v2." + token[3:], None):
        with pytest.raises(IdentityError):
            tokens.verify(bad)  # type: ignore[arg-type]
    with pytest.raises(IdentityError):
        IdentityTokens(secret=b"t" * 32, clock=lambda: clock[0]).verify(token)  # another secret
    clock[0] += 61
    with pytest.raises(IdentityError, match="expired"):
        tokens.verify(token)
    with pytest.raises(ValueError):
        IdentityTokens(secret=b"short")


def test_default_token_secrets_differ_between_instances() -> None:
    assert IdentityTokens().issue(SID) != IdentityTokens().issue(SID)


def gate(steam: FakeSteam | None = None, **kw) -> SteamIdentityGate:  # noqa: ANN003
    steam = steam or FakeSteam()
    return SteamIdentityGate(RETURN_TO, http=steam, verifier=SteamOpenIDVerifier(RETURN_TO, http=steam, clock=lambda: NOW), **kw)


def test_gate_issues_a_token_only_for_an_account_steam_vouches_for() -> None:
    g = gate()
    sid, token = g.complete_login(assertion())
    assert sid == SID and g.authorize(token, SID) == SID
    with pytest.raises(IdentityError, match="different Steam account"):
        g.authorize(token, OTHER)
    with pytest.raises(IdentityError, match="required"):
        g.authorize(None, SID)


def test_limited_and_vac_banned_accounts_get_no_token_unless_limited_is_explicitly_allowed() -> None:
    with pytest.raises(IdentityError, match="Limited"):
        gate(FakeSteam(limited="1")).complete_login(assertion())
    with pytest.raises(IdentityError, match="VAC"):
        gate(FakeSteam(vac="1")).complete_login(assertion(**{"openid.response_nonce": nonce(salt="Y")}))
    assert gate(FakeSteam(limited="1"), allow_limited=True).complete_login(assertion())[0] == SID


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# the coordinator app
# ---------------------------------------------------------------------------------------------------------------------------------------------------


def service() -> Mock:
    svc = Mock()
    svc.assign_workunit.return_value = (Mock(to_dict=lambda: {"wu_id": "wu-1"}), {})
    svc.reputation_ledger = {SID: NodeReputationRecord(steam_id64=SID, worker_id="rig-7")}
    return svc


REQUEST = {"worker_id": "rig-7", "steam_id64": SID}


def test_development_mode_is_unchanged_a_body_claimed_id_is_still_accepted() -> None:
    svc = service()
    client = TestClient(create_coordinator_fastapi_app(svc))
    assert client.post("/workunit/request", json={"worker_id": "w", "steam_id64": "sim-node-1"}).status_code == 200
    assert SID in client.get("/reputation").json()
    assert client.get("/identity/steam/login", follow_redirects=False).status_code == 404  # the sign-in routes exist only with a gate


def test_with_a_gate_a_workunit_needs_a_token_for_the_same_steamid() -> None:
    svc, g = service(), gate()
    client = TestClient(create_coordinator_fastapi_app(svc, identity=g))
    assert client.post("/workunit/request", json=REQUEST).status_code == 401
    token = g.tokens.issue(SID)
    wrong = client.post("/workunit/request", json={**REQUEST, "steam_id64": OTHER}, headers={"Authorization": f"Bearer {token}"})
    assert wrong.status_code == 403
    assert client.post("/workunit/request", json=REQUEST, headers={"Authorization": "Bearer v1.garbage"}).status_code == 403
    svc.assign_workunit.assert_not_called()
    ok = client.post("/workunit/request", json=REQUEST, headers={"Authorization": f"Bearer {token}"})
    assert ok.status_code == 200 and ok.json()["workunit"] == {"wu_id": "wu-1"}
    svc.assign_workunit.assert_called_once()


def test_the_sign_in_flow_end_to_end_with_steam_replaced_by_a_fake() -> None:
    svc, g = service(), gate()
    client = TestClient(create_coordinator_fastapi_app(svc, identity=g))
    redirect = client.get("/identity/steam/login", follow_redirects=False)
    assert redirect.status_code == 302 and redirect.headers["location"].startswith(OPENID_ENDPOINT)
    cb = client.get("/identity/steam/callback", params=assertion())
    assert cb.status_code == 200 and cb.json()["steam_id64"] == SID
    token = cb.json()["identity_token"]
    assert client.post("/workunit/request", json=REQUEST, headers={"Authorization": f"Bearer {token}"}).status_code == 200
    replay = client.get("/identity/steam/callback", params=assertion())
    assert replay.status_code == 403 and "already used" in replay.json()["detail"]


def test_a_limited_account_cannot_sign_in_so_it_never_gets_a_workunit() -> None:
    client = TestClient(create_coordinator_fastapi_app(service(), identity=gate(FakeSteam(limited="1"))))
    response = client.get("/identity/steam/callback", params=assertion())
    assert response.status_code == 403 and "Limited" in response.json()["detail"]


CHAT = {"messages": [{"role": "user", "content": "hello swarm, please answer"}]}


def test_the_simulated_chat_endpoint_needs_a_signed_in_node_when_gated_and_never_runs_the_probe_without_one(monkeypatch) -> None:  # noqa: ANN001
    probe = Mock(side_effect=AssertionError("an unauthenticated request reached the probe"))
    monkeypatch.setattr("hivemind.game.hf_hack_miner.execute_interactive_bounty_probe", probe)
    g = gate()
    client = TestClient(create_coordinator_fastapi_app(service(), identity=g))
    assert client.post("/v1/chat/completions", json=CHAT).status_code == 401
    assert client.post("/v1/chat/completions", json=CHAT, headers={"Authorization": "Bearer nonsense"}).status_code == 403
    expired = IdentityTokens(secret=b"z" * 32, clock=lambda: NOW).issue(SID)  # signed with another secret
    assert client.post("/v1/chat/completions", json=CHAT, headers={"Authorization": f"Bearer {expired}"}).status_code == 403
    probe.assert_not_called()

    probe2 = Mock(return_value={"generated_solution": "ok", "top2_active_experts": ["a", "b"], "expert_routing_weights": [0.6, 0.4]})
    monkeypatch.setattr("hivemind.game.hf_hack_miner.execute_interactive_bounty_probe", probe2)
    response = client.post("/v1/chat/completions", json=CHAT, headers={"Authorization": f"Bearer {g.tokens.issue(SID)}"})
    assert response.status_code == 200 and response.json()["choices"][0]["message"]["content"] == "ok"
    probe2.assert_called_once()


def test_the_simulated_chat_endpoint_stays_open_in_development_mode(monkeypatch) -> None:  # noqa: ANN001
    probe = Mock(return_value={"generated_solution": "ok", "top2_active_experts": ["a", "b"], "expert_routing_weights": [0.6, 0.4]})
    monkeypatch.setattr("hivemind.game.hf_hack_miner.execute_interactive_bounty_probe", probe)
    assert TestClient(create_coordinator_fastapi_app(service())).post("/v1/chat/completions", json=CHAT).status_code == 200


def test_the_public_reputation_listing_does_not_expose_steam_ids_or_worker_names_when_gated() -> None:
    client = TestClient(create_coordinator_fastapi_app(service(), identity=gate()))
    text = client.get("/reputation").text
    assert SID not in text and "rig-7" not in text and "reputation_score" in text


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# the Steamworks bridge's simulated tickets
# ---------------------------------------------------------------------------------------------------------------------------------------------------


def forge(secret: bytes, sid: str = SID, spend: float = 999.0) -> steamworks.SteamAuthSessionTicket:
    payload = f"{sid}:{steamworks.STEAMWORKS_APP_ID}:{spend:.2f}:0"
    return steamworks.SteamAuthSessionTicket(sid, steamworks.STEAMWORKS_APP_ID, spend, False, hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest())


def test_a_ticket_forged_with_the_old_published_secret_is_rejected(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("HIVEMIND_STEAMWORKS_SECRET", raising=False)
    ok, reason = steamworks.SteamworksSDKBridge().verify_sybil_resistance(forge(b"HiveMind-Steamworks-Auth-Key"))
    assert not ok and "signature" in reason


def test_bridges_in_one_process_still_interoperate_and_an_operator_secret_is_honoured(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("HIVEMIND_STEAMWORKS_SECRET", raising=False)
    ticket = steamworks.SteamworksSDKBridge().get_auth_session_ticket(SID)
    assert steamworks.SteamworksSDKBridge().verify_sybil_resistance(ticket)[0]
    monkeypatch.setenv("HIVEMIND_STEAMWORKS_SECRET", "ab" * 32)
    assert steamworks.SteamworksSDKBridge().signing_secret == bytes.fromhex("ab" * 32)
    assert steamworks.SteamworksSDKBridge(signing_secret=b"explicit-secret-for-this-test-0123456789").signing_secret.startswith(b"explicit")


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------------------------------------


def test_serve_coordinator_refuses_a_public_bind_without_identity_verification(capsys, monkeypatch) -> None:  # noqa: ANN001
    from hivemind.cli import main as cli_main

    run = Mock()
    monkeypatch.setattr("uvicorn.run", run)
    assert cli_main.main(["serve-coordinator", "--host", "0.0.0.0", "--port", "18081"]) == 2
    assert "--public-url" in capsys.readouterr().err
    run.assert_not_called()
