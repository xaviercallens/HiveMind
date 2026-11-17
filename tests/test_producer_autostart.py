# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for the cross-platform Producer Auto-Start Service Generator and End-User GPU Outreach Campaign."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hivemind.router.accounts import RouterStore

REPO_ROOT = Path(__file__).resolve().parents[1]
AUTOSTART_SCRIPT = REPO_ROOT / "scripts" / "hivemind_producer_autostart.py"
OUTREACH_SCRIPT = REPO_ROOT / "scripts" / "hivemind_gpu_outreach_campaign.py"


def test_producer_autostart_generates_linux_windows_macos_bundles(tmp_path: Path) -> None:
    for platform, expected_files in (
        ("linux", ("hivemind-producer-runner.sh", "hivemind-producer.service")),
        ("windows", ("run-producer-service.ps1", "install-producer-service.ps1")),
        ("macos", ("hivemind-producer-runner.sh", "org.hivemind.producer.plist")),
    ):
        dest = tmp_path / platform
        proc = subprocess.run(
            [
                sys.executable,
                str(AUTOSTART_SCRIPT),
                platform,
                "--model",
                "coder=qwen2.5-coder:7b",
                "--apply",
                str(dest),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        for fname in expected_files:
            fpath = dest / fname
            assert fpath.exists(), f"missing {fpath}"
            text = fpath.read_text(encoding="utf-8")
            if fname in ("hivemind-producer-runner.sh", "run-producer-service.ps1"):
                assert "hivemind_producer_shield.py" in text
                assert "hivemind_producer_agent.py" in text
                assert "qwen2.5-coder:7b" in text
            elif fname == "install-producer-service.ps1":
                assert "Register-ScheduledTask" in text
                assert "HiveMindProducerAgent" in text


def test_producer_autostart_rejects_bad_model_spec() -> None:
    proc = subprocess.run(
        [sys.executable, str(AUTOSTART_SCRIPT), "linux", "--model", "bad..spec"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "refused:" in proc.stderr


def test_gpu_outreach_campaign_reads_db_and_renders_bilingual_templates(tmp_path: Path) -> None:
    db_path = tmp_path / "router.db"
    store = RouterStore(db_path, secret=b"outreach-campaign-test-secret-32b!")
    store.register_with_credentials("tibrib", "StrongPass123!", role="prosumer")
    store.register_with_credentials("vxtzq", "StrongPass123!", role="prosumer")
    store.register_with_credentials("someone@example.com", "StrongPass123!", role="prosumer", email="someone@example.com")
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(
        '[{"account_id": "x", "handle": "tibrib", "email": null, "role": "prosumer", "os_platform": "windows", '
        '"known_gpu": "NVIDIA GeForce RTX 4070 (12,282 MiB VRAM)", "recommended_model_spec": "coder=qwen2.5-coder:7b", '
        '"previous_tunnel_note": "test"}]',
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(OUTREACH_SCRIPT), "--db", str(db_path), "--cohort-file", str(cohort_path), "--format", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    import json

    data = json.loads(proc.stdout)
    assert data["recipients_count"] >= 3
    handles = {item["recipient"]["handle"] for item in data["invitations"]}
    assert "tibrib" in handles
    assert "vxtzq" in handles
    tibrib_inv = next(item for item in data["invitations"] if item["recipient"]["handle"] == "tibrib")
    assert "RTX 4070" in tibrib_inv["subject_fr"]
    assert "install-producer-service.ps1" in tibrib_inv["body_en"]
    assert "22:00" in tibrib_inv["body_en"]
