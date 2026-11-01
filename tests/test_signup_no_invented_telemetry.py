# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The signup page shows live readings or "n/a"; it never fills a missing reading with an invented number."""

from __future__ import annotations

from hivemind.router.signup_page import render_signup_page


def test_missing_telemetry_renders_na_not_constants() -> None:
    html = render_signup_page(None, "nonce-1")
    assert "69 Registered Accounts" not in html
    assert "10,519" not in html and "10519" not in html
    assert "n/a Registered Accounts" in html
    assert "T4 Free VRAM n/a MiB" in html
    assert "live telemetry partly unavailable" in html


def test_injected_telemetry_is_rendered_verbatim() -> None:
    live = {
        "accounts": {"total_registered_accounts": 58, "by_role": {"prosumer": 7, "producer": 2, "consumer": 49, "trainer": 0}},
        "models": {"live_models_count": 12},
        "t4_boinc": {"vram_free_mib": 4609},
        "opentrainer": {"total_registered_nodes": 1, "accepted_rounds_count": 13},
    }
    html = render_signup_page(None, "nonce-2", live_telemetry=live)
    assert "Live GCP Swarm: 58 Registered Accounts (7 prosumers, 2 producers, 49 consumers, 0 trainers)" in html
    assert "12 Live Models" in html
    assert "T4 Free VRAM 4,609 MiB" in html
    assert "1 Trainer GPU(s) / 13 Accepted Round(s)" in html
    assert "n/a" not in html.split('id="signupLiveTelemetry"')[1].split("</div>")[0]
