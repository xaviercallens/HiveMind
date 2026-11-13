# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit & router integration tests for the HiveMind v3.4.0 5-Phase Communication Plan & Raid Hub."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from hivemind.router.accounts import RouterStore
from hivemind.router.community_outreach_v34 import (
    COMMUNICATION_PLAN_PHASES,
    OWNER_SENDER_EMAIL,
    PIONEER_COMMUNITY_HANDLES,
    VERIFIED_PIONEER_EMAILS,
    build_v34_communication_plan_status,
    build_v34_discord_outreach,
    build_v34_github_outreach,
    build_v34_pioneer_email,
    build_v34_reddit_outreach,
    publish_v34_amcp_pioneer_outreach,
    raid_hub_content_security_policy,
    render_v34_raid_hub_page,
    send_discord_webhook,
)
from hivemind.router.gateway import create_router_app


def test_pioneer_email_enforces_owner_sender_and_bcc_privacy() -> None:
    pkg = build_v34_pioneer_email()
    assert pkg["from_email"] == "callensxavier@gmail.com"
    assert pkg["to_email"] == "callensxavier@gmail.com"
    assert set(pkg["bcc_emails"]) == set(VERIFIED_PIONEER_EMAILS)
    assert OWNER_SENDER_EMAIL not in pkg["bcc_emails"]
    assert len(pkg["gmail_web_compose_url"]) < 2000

    md = pkg["markdown_body"]
    html_body = pkg["html_body"]
    for body in (md, html_body):
        assert "callensxavier@gmail.com" in body
        assert "thank" in body.lower()
        assert "Web UI" in body
        assert "CrowdGPT" in body
        assert "14" in body and "15" in body and "2026" in body
        assert "validate" in body.lower() or "valider" in body.lower()

    assert "<table style=" in html_body
    assert "border-collapse:collapse" in html_body
    assert "authuser=callensxavier%40gmail.com" in pkg["gmail_web_compose_url"]

    with pytest.raises(ValueError, match="callensxavier@gmail.com"):
        build_v34_pioneer_email(sender_email="someone_else@example.com")

    with pytest.raises(ValueError, match="invalid email"):
        build_v34_pioneer_email(bcc_recipients=["bad-email\r\ninjection"])


def test_github_discord_reddit_and_5phase_plan_builders() -> None:
    assert len(COMMUNICATION_PLAN_PHASES) == 5
    status = build_v34_communication_plan_status()
    assert status["organizer"]["email"] == "callensxavier@gmail.com"
    assert status["dual_concept_validation"]["cuda_preflight_loss_trajectory"] == [5.3042, 5.1742, 0.9668]
    assert "@jblam-O" in status["pioneer_honor_roll"]["github_contributors"]
    assert "@Vxtzq" in status["pioneer_honor_roll"]["github_contributors"]

    gh = build_v34_github_outreach()
    for handle in ("@clark719664", "@pedromineo", "@radialmonster", "@jblam-O", "@Vxtzq"):
        assert handle in gh["discussion_body"]
    assert len(gh["discussion_new_url"]) < 2000
    assert len(gh["issue_new_url"]) < 2000
    assert len(gh["release_v340_new_url"]) < 2000

    disc = build_v34_discord_outreach()
    assert "@everyone" in disc["markdown_post"]
    assert len(disc["webhook_payload"]["content"]) <= 2000

    reddit = build_v34_reddit_outreach()
    assert [r["subreddit"] for r in reddit] == ["r/LocalLLaMA", "r/BOINC", "r/MachineLearning", "r/homelab"]
    for post in reddit:
        assert len(post["submit_url"]) < 2000


def test_raid_hub_page_csp_and_gateway_routes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nonce"):
        raid_hub_content_security_policy("bad nonce!")

    html_fr = render_v34_raid_hub_page(lang="fr", nonce="ValidNonce123456")
    assert 'lang="fr"' in html_fr
    assert "Merci aux Premiers Pionniers HiveMind" in html_fr
    assert 'nonce="ValidNonce123456"' in html_fr

    store = RouterStore(tmp_path / "router_v34_comm.sqlite3", b"0" * 32)
    app = create_router_app(store=store)
    client = TestClient(app)

    # 1. Root landing page displays the #weekend-raid banner while keeping v3.1.1 engine header intact
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert 'id="weekend-raid"' in r_root.text
    assert "v3.1.1 (v3.0.0 Core Engine)" in r_root.text

    # 2. Dedicated /raid & /crowdgpt-raid bilingual hub with strict CSP & security headers
    r_raid_en = client.get("/raid?lang=en")
    assert r_raid_en.status_code == 200
    assert "Thank You HiveMind Pioneers!" in r_raid_en.text
    assert "default-src 'none'" in r_raid_en.headers["Content-Security-Policy"]
    assert r_raid_en.headers["X-Frame-Options"] == "DENY"
    assert r_raid_en.headers["X-Content-Type-Options"] == "nosniff"
    assert client.head("/raid").status_code == 200

    r_raid_fr = client.get("/crowdgpt-raid?lang=fr")
    assert r_raid_fr.status_code == 200
    assert "Merci aux Premiers Pionniers HiveMind" in r_raid_fr.text

    # 3. Public 6-wave .ics calendar download route
    r_ics = client.get("/calendar/crowdgpt-weekend-raid.ics")
    assert r_ics.status_code == 200
    assert "BEGIN:VCALENDAR" in r_ics.text
    assert r_ics.text.count("BEGIN:VEVENT") == 6

    # 4. Public JSON 5-phase communication plan endpoint
    r_json = client.get("/v1/hivemind/community/v34-outreach")
    assert r_json.status_code == 200
    payload = r_json.json()
    assert payload["object"] == "hivemind.community.v34_outreach_plan"
    assert len(payload["phases"]) == 5


class _DummyResp:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._raw = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_DummyResp":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_publish_amcp_and_discord_webhook_dispatch() -> None:
    captured_msgs: list[dict[str, Any]] = []

    def _fake_urlopen(req: Any, timeout: float = 20.0) -> _DummyResp:
        body = json.loads(req.data.decode("utf-8"))
        captured_msgs.append(body)
        return _DummyResp({"event_id": f"amcp-test-{len(captured_msgs)}", "topic": "hivemind.amcp.human_agent.v1"})

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        receipts = publish_v34_amcp_pioneer_outreach(
            gateway_url="https://hivemind-openrouter-1003063861791.us-central1.run.app",
            api_key="hm-testkey1234567890",
            handles=PIONEER_COMMUNITY_HANDLES,
            include_bee_report=True,
        )

    assert len(receipts) == 2 + len(PIONEER_COMMUNITY_HANDLES)
    assert captured_msgs[0]["event_type"] == "io.hivemind.amcp.dev.proposal"
    assert captured_msgs[1]["event_type"] == "io.hivemind.amcp.bee.report"
    assert captured_msgs[2]["recipient_id"] == "vxtzq"
    assert all(r["status"] == "ok" for r in receipts)

    with pytest.raises(ValueError, match="discord.com/api/webhooks"):
        send_discord_webhook("https://example.com/not-discord")



def test_no_contributor_email_address_is_committed_or_served() -> None:
    """Contributors' addresses are personal data: only the owner's own address may appear in source or public output."""
    import json
    import re
    from pathlib import Path

    from hivemind.router import community_outreach_v34 as mod

    addr = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    allowed = {mod.OWNER_SENDER_EMAIL.lower()}
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert {a.lower() for a in addr.findall(src)} <= allowed | {"noreply@github.com"}
    public = json.dumps(mod.build_v34_communication_plan_status())
    assert {a.lower() for a in addr.findall(public)} <= allowed
    assert "gmail_web_compose_url" not in public
