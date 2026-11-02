# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The foundation-paper portal payload labels its hard-coded baseline as a dated snapshot, not as live or verified data."""

from __future__ import annotations

from hivemind.economy.international_launch_model import build_foundation_paper_v42_portal_payload


def test_wan_baseline_block_is_labelled_snapshot_and_claims_no_verification() -> None:
    block = build_foundation_paper_v42_portal_payload()["live_experimentation_v42"]["live_47_node_wan_baseline"]
    assert block["data_source"].startswith("SNAPSHOT_") and "NOT_LIVE" in block["data_source"]
    assert block["snapshot_date_utc"] == "2026-10-31"
    assert "/boinc/status" in block["live_readings_at"]
    offending = [k for k in block if "verified" in k or "attested" in k]
    assert offending == []
    assert block["boinc_cobblestones_snapshot"] == 5632.11
