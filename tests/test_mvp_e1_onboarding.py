# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for MVP Criterion E1 onboarding, yield check, and live T4 telemetry push helper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import mvp_e1_check as e1  # noqa: E402
import push_t4_telemetry as tel  # noqa: E402


def test_mvp_e1_onboarding_check_passes(tmp_path: Path) -> None:
    out = tmp_path / "E1_onboarding.json"
    assert e1.main(["--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["yield_elapsed_s"] <= 1.5
    assert all(payload["checks"].values())


def test_push_t4_telemetry_builds_valid_snapshot_and_fails_without_key(tmp_path: Path, monkeypatch) -> None:
    hold_file = tmp_path / "HOLD"
    rounds_file = tmp_path / "rounds.jsonl"
    rounds_file.write_text(
        json.dumps({"eval_loss_before": 5.272, "eval_loss_after": 5.271, "round_index": 49}) + "\n",
        encoding="utf-8",
    )

    def _runner(cmd, timeout=10.0):
        if cmd[0] == "nvidia-smi":
            return 0, "7353, 15360, 7559, 75, 69.99, 33"
        if cmd[0] == "systemctl":
            return 3, "inactive"
        return 1, ""

    payload = tel.build_telemetry_payload(
        hold_file=hold_file,
        rounds_file=rounds_file,
        sole_active_gpu=False,
        runner=_runner,
    )
    assert payload["gpu"]["used_mib"] == 7353
    assert payload["gpu"]["free_mib"] == 7559
    assert payload["governor"]["hold_file_present"] is False
    assert payload["qlora_progress"]["held_out_eval_loss_after"] == 5.271

    # Also test validate_t4_snapshot compatibility via t4_visibility
    from hivemind.router.t4_visibility import update_t4_telemetry_snapshot

    snap = update_t4_telemetry_snapshot(payload)
    assert snap["telemetry_source"] == "LIVE_UPDATE"
    assert snap["gpu"]["used_mib"] == 7353

    # Missing admin key fails closed
    monkeypatch.delenv("HIVEMIND_ADMIN_KEY", raising=False)
    monkeypatch.setattr(
        tel,
        "read_gpu_metrics",
        lambda runner=None: {"used_mib": 1000, "total_mib": 15360, "free_mib": 14000, "temp_c": 50.0, "power_w": 30.0, "util_pct": 0.0},
    )
    import pytest

    with pytest.raises(SystemExit, match="HIVEMIND_ADMIN_KEY"):
        tel.main(["--hold-file", str(hold_file), "--rounds-file", str(rounds_file)])
