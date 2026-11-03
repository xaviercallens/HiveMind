# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The game pages must not fabricate accounts, keys, balances or GPU contributions in the visitor's browser (audit 2026-11-03).

Before: the WebGPU page invented an ``hm-webgpu-*`` key and a 500 HMT balance when the gateway was unreachable, claimed "+250 HMT" for work
the server credits with 0 until it can be verified, and the Stitch page registered an account and posted a fabricated RTX 2070 contribution
(5.8% GPU use, 12 s idle) to the dashboard for every visitor on page load."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src/hivemind/game"
WEBGPU = (ROOT / "webgpu_galaxy_cockpit.py").read_text(encoding="utf-8")
STITCH = (ROOT / "stitch_v46_cockpit_template.py").read_text(encoding="utf-8")


def test_webgpu_page_invents_no_key_balance_or_credit() -> None:
    assert "hm-webgpu-" not in WEBGPU  # no key minted in the browser
    assert "balance_hmt: 500" not in WEBGPU and "balanceHmt + 250" not in WEBGPU and "500 HMT" not in WEBGPU
    assert "+250 HMT" not in WEBGPU and "+45 Cobblestones" not in WEBGPU
    assert "api/v1/community/boinc-sync" not in WEBGPU  # nothing mirrored to the dashboard from the browser
    assert "aucun compte n'a été créé" in WEBGPU  # unreachable gateway is an honest error
    assert "data.hmt_awarded" in WEBGPU  # what the server reports is what the page shows


def test_stitch_page_posts_nothing_and_does_not_run_on_load() -> None:
    body = STITCH[STITCH.index("window.simulateSteamRegisterLoginAndContribute"):]
    body = body[: body.index("};") + 2]
    for forbidden in ("fetch(", "register-login", "contribute-boinc", "boinc-sync", "gpu_utilization_pct", "hivemind_api_key"):
        assert forbidden not in body, forbidden
    assert "SIMULATION LOCALE" in body
    assert "\n  simulateSteamRegisterLoginAndContribute();\n" not in STITCH  # no automatic call for every visitor
    assert "+250 HMT" not in STITCH and "+500 HMT" not in STITCH
