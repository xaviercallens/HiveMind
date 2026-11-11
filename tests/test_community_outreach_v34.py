# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit & integration tests for HiveMind v3.4.0 Multi-Channel Pioneer Outreach & Weekend Raid dispatcher."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from hivemind.router.community_outreach_v34 import (
    OWNER_SENDER_EMAIL,
    PIONEER_COMMUNITY_HANDLES,
    VERIFIED_PIONEER_EMAILS,
    build_v34_discord_outreach,
    build_v34_github_outreach,
    build_v34_pioneer_email,
    build_v34_reddit_outreach,
    publish_v34_amcp_pioneer_outreach,
    send_discord_webhook,
)


def test_pioneer_email_enforces_owner_sender_and_bcc_privacy() -> None:
    pkg = build_v34_pioneer_email()
    assert pkg["from_email"] == "callensxavier@gmail.com"
    assert pkg["to_email"] == "callensxavier@gmail.com"
    assert set(pkg["bcc_emails"]) == set(VERIFIED_PIONEER_EMAILS)
    assert OWNER_SENDER_EMAIL not in pkg["bcc_emails"]

    # Must cover all 5 core user-requested themes
    md = pkg["markdown_body"]
    html_body = pkg["html_body"]
    for body in (md, html_body):
        assert "callensxavier@gmail.com" in body
        assert "thank" in body.lower()
        assert "Web UI" in body
        assert "CrowdGPT" in body
        assert "14" in body and "15" in body and "2026" in body
        assert "validate" in body.lower() or "valider" in body.lower()

    # Inline CSS table styling required by Gmail CLI rules
    assert "<table style=" in html_body
    assert "border-collapse:collapse" in html_body
    assert "authuser=callensxavier%40gmail.com" in pkg["gmail_web_compose_url"]

    with pytest.raises(ValueError, match="callensxavier@gmail.com"):
        build_v34_pioneer_email(sender_email="someone_else@example.com")

    with pytest.raises(ValueError, match="invalid email"):
        build_v34_pioneer_email(bcc_recipients=["bad-email\r\ninjection"])


def test_github_discord_and_reddit_outreach_builders() -> None:
    gh = build_v34_github_outreach()
    assert "@clark719664" in gh["discussion_body"]
    assert "@pedromineo" in gh["discussion_body"]
    assert "@radialmonster" in gh["discussion_body"]
    assert "@Vxtzq" in gh["discussion_body"]
    assert "discussions/new?" in gh["discussion_new_url"]
    assert "issues/new?" in gh["issue_new_url"]
    assert "releases/new?" in gh["release_v340_new_url"]

    disc = build_v34_discord_outreach()
    assert "@everyone" in disc["markdown_post"]
    assert "14–15" in disc["markdown_post"] or "14-15" in disc["markdown_post"]
    assert len(disc["webhook_payload"]["content"]) <= 2000
    assert disc["webhook_payload"]["embeds"][0]["title"].startswith("HiveMind v3.4.0")

    reddit = build_v34_reddit_outreach()
    subs = [r["subreddit"] for r in reddit]
    assert subs == ["r/LocalLLaMA", "r/BOINC", "r/MachineLearning", "r/homelab"]
    for post in reddit:
        assert "CrowdGPT" in post["title"] or "CrowdGPT" in post["body"]
        assert "reddit.com/r/" in post["submit_url"]


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
        )

    assert len(receipts) == 1 + len(PIONEER_COMMUNITY_HANDLES)
    assert captured_msgs[0]["recipient_id"] == "broadcast"
    assert captured_msgs[1]["recipient_id"] == "vxtzq"
    assert all(r["status"] == "ok" for r in receipts)

    with pytest.raises(ValueError, match="discord.com/api/webhooks"):
        send_discord_webhook("https://example.com/not-discord")
