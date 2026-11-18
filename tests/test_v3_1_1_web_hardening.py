"""The v3.1.1 fixes, each as the attack or false claim that v3.1.0 allowed."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app
from hivemind.boinc.stitch_portal import render_stitch_enhanced_portal  # noqa: F401
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app

ROUTER_ACCOUNT_WRITES = (
    "/api/v1/amcp/kafka/publish",
    "/api/v1/training/calendar/propose",
    "/api/v1/training/calendar/pledge-gpu",
)
ADMIN_KEY = "admin-key-for-this-test"


@pytest.fixture()
def router() -> tuple[TestClient, dict[str, str]]:
    store = RouterStore(":memory:", b"k" * 32)
    app = create_router_app(store, admin_key_hash=store.key_digest(ADMIN_KEY))
    client = TestClient(app, raise_server_exceptions=False)
    made = client.post("/v1/hivemind/register", json={"handle": "probe_user", "role": "prosumer"})
    assert made.status_code in (200, 201), made.text
    return client, {"Authorization": f"Bearer {made.json()['api_key']}"}


@pytest.fixture()
def dashboard() -> TestClient:
    return TestClient(create_hivemind_at_home_app(), raise_server_exceptions=False)


def _register(client: TestClient, handle: str, password: str = "Correct#Horse9!") -> str:
    r = client.post(
        "/api/v1/auth/register",
        json={"handle": handle, "email": f"{handle}@example.invalid", "password": password, "hardware_profile": "rtx_4090", "target_profile": "developer", "role": "HiveMinder Contributor"},
    )
    assert r.status_code == 200, r.text
    return r.json()["token"]


@pytest.mark.parametrize("path", ROUTER_ACCOUNT_WRITES)
def test_router_writes_refuse_anonymous(router, path: str) -> None:
    client, _ = router
    assert client.post(path, json={}).status_code == 401


def test_router_t4_schedule_is_owner_only(router) -> None:
    client, user = router
    before = client.get("/api/v1/training/t4-daily-schedule").json()
    assert client.post("/api/v1/training/t4-daily-schedule", json={"enabled": False, "window_start": "00:00"}).status_code == 403
    assert client.post("/api/v1/training/t4-daily-schedule", headers=user, json={"enabled": False}).status_code == 403
    assert client.get("/api/v1/training/t4-daily-schedule").json() == before
    done = client.post("/api/v1/training/t4-daily-schedule", headers={"x-admin-key": ADMIN_KEY}, json={"enabled": True})
    assert done.status_code == 200
    assert done.json()["status"] == "SCHEDULE_SAVED_NOT_APPLIED"
    assert done.json()["applied_to_gcp"] is False


def test_router_publish_cannot_speak_as_someone_else_and_says_it_is_simulated(router) -> None:
    client, user = router
    r = client.post("/api/v1/amcp/kafka/publish", headers=user, json={"actor_handle": "@xaviercallens", "message": "owner says stop"})
    assert r.status_code == 200
    body = r.json()
    assert body["published_event"]["actor_handle"] == "@probe_user"
    assert body["status"] == "RECORDED_IN_MEMORY_NOT_KAFKA"
    assert body["simulated"] is True
    assert "hm-kafka" not in r.text.split('"space"')[0]


@pytest.mark.parametrize("raw", [b"{not json", b"[1, 2]"])
def test_router_malformed_body_is_400(router, raw: bytes) -> None:
    client, user = router
    r = client.post("/api/v1/amcp/kafka/publish", headers={**user, "content-type": "application/json"}, content=raw)
    assert r.status_code == 400


def test_router_wrong_field_type_is_400(router) -> None:
    client, user = router
    assert client.post("/api/v1/amcp/kafka/publish", headers=user, json={"topic": {"a": 1}}).status_code == 400
    assert client.post("/api/v1/dashboard/nash-equilibrium/simulate", json={"active_producers": "many"}).status_code == 400


def test_router_page_sends_csp_and_hsts(router) -> None:
    client, _ = router
    r = client.get("/v3.1")
    assert r.status_code == 200
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert "max-age=" in r.headers["strict-transport-security"]


def test_router_red_team_route_claims_nothing(router) -> None:
    client, _ = router
    body = client.get("/api/v1/security/red-team-v3-audit").json()
    assert "VERIFIED" not in body["status"].replace("NOT_INDEPENDENTLY_VERIFIED", "")
    assert "all_6_findings_verified" not in body


@pytest.mark.parametrize(
    ("path", "code"),
    [
        ("/api/v1/amcp/kafka/publish", 401),
        ("/api/v1/training/calendar/propose", 401),
        ("/api/v1/training/calendar/pledge-gpu", 401),
        ("/api/v1/maintenance/acknowledge", 401),
        ("/api/v1/maintenance/toggle", 403),
        ("/api/v1/gcp/portal-readiness/review", 403),
        ("/api/v1/training/personal-schedule", 403),
    ],
)
def test_dashboard_writes_refuse_anonymous(dashboard: TestClient, path: str, code: int) -> None:
    assert dashboard.post(path, json={}).status_code == code


def test_dashboard_reads_stay_public(dashboard: TestClient) -> None:
    for path in ("/api/v1/training/personal-schedule", "/api/v1/maintenance/status", "/api/v1/amcp/kafka/space"):
        assert dashboard.get(path).status_code == 200


def test_dashboard_ordinary_user_cannot_toggle_maintenance(dashboard: TestClient) -> None:
    tok = _register(dashboard, "plain_user")
    assert dashboard.post("/api/v1/maintenance/toggle", headers={"Authorization": f"Bearer {tok}"}, json={"active": False}).status_code == 403


def test_dashboard_handle_comes_from_the_token(dashboard: TestClient) -> None:
    tok = _register(dashboard, "pledger")
    hdr = {"Authorization": f"Bearer {tok}"}
    pub = dashboard.post("/api/v1/amcp/kafka/publish", headers=hdr, json={"actor_handle": "@xavier_callens", "message": "x"})
    assert pub.json()["published_event"]["actor_handle"] == "@pledger"
    ack = dashboard.post("/api/v1/maintenance/acknowledge", headers=hdr, json={"hiveminer_handle": "someone_else"}).json()
    assert ack["hiveminer_handle"] == "pledger"
    assert ack["loyalty_hmt_awarded"] == 0


def test_login_does_not_create_accounts(dashboard: TestClient) -> None:
    assert dashboard.post("/api/v1/auth/login", json={"handle_or_email": "never_registered"}).status_code == 404
    assert dashboard.post("/api/v1/auth/login", json={"handle_or_email": "never_registered", "strict_existing": False}).status_code == 404


def test_failed_passwords_do_not_lock_the_owner_out_from_elsewhere(dashboard: TestClient) -> None:
    _register(dashboard, "victim_user")
    attacker = {"x-forwarded-for": "203.0.113.9"}
    codes = [dashboard.post("/api/v1/auth/login", headers=attacker, json={"handle_or_email": "victim_user", "password": f"bad{i}"}).status_code for i in range(12)]
    assert codes[0] == 401
    assert codes[-1] == 429
    owner = dashboard.post("/api/v1/auth/login", headers={"x-forwarded-for": "198.51.100.7"}, json={"handle_or_email": "victim_user", "password": "Correct#Horse9!"})
    assert owner.status_code == 200


def test_failed_passwords_have_a_ceiling_across_clients(dashboard: TestClient) -> None:
    _register(dashboard, "spread_user")
    for i in range(200):
        dashboard.post("/api/v1/auth/login", headers={"x-forwarded-for": f"203.0.113.{i % 250}, 10.0.0.{i % 250}"}, json={"handle_or_email": "spread_user", "password": "bad"})
    late = dashboard.post("/api/v1/auth/login", headers={"x-forwarded-for": "198.51.100.8"}, json={"handle_or_email": "spread_user", "password": "bad"})
    assert late.status_code == 429


def test_page_has_no_invented_credentials_and_escapes_feed_fields(dashboard: TestClient) -> None:
    page = dashboard.get("/v3").text
    assert "HiveMind#MVP2026" not in page
    assert "hm_live_demo_guest" not in page
    assert "Math.random().toString(36)" not in page
    assert not re.findall(r"hm_live_[A-Za-z0-9]{8,}", page)
    assert "function esc(value)" in page
    for raw in ("${ev.message}", "${ev.actor_handle}", "${ev.target_bee_or_agent}", "${s.title}", "${data.message}"):
        assert raw not in page, raw
    assert "${esc(ev.message)}" in page
