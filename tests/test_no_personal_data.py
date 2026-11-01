# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""No real person's contact data, router account ids or live tunnel hostnames in the outreach script or its doc."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hivemind_gpu_outreach_campaign.py"
FILES = (SCRIPT, ROOT / "docs" / "PRODUCER_AUTOSTART_AND_OUTREACH_CAMPAIGN.md")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_ALLOWED = ("callensxavier@gmail.com", "noreply@anthropic.com")


def test_no_foreign_email_addresses() -> None:
    found = [m for f in FILES for m in _EMAIL.findall(f.read_text(encoding="utf-8")) if m not in _ALLOWED]
    assert found == [], f"{len(found)} non-owner email address(es) in tracked outreach files"


def test_no_account_ids_or_tunnel_hosts() -> None:
    for f in FILES:
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"\b[0-9a-f]{32}\b", text), f"32-hex account id in {f.name}"
        assert "trycloudflare.com" not in text, f"tunnel hostname in {f.name}"


def test_default_cohort_is_empty() -> None:
    spec = importlib.util.spec_from_file_location("outreach_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["outreach_under_test"] = mod
    spec.loader.exec_module(mod)
    assert mod.VERIFIED_COHORT == ()
    assert len(mod.build_campaign_report()["invitations"]) == 0
