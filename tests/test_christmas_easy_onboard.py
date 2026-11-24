# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for the Holiday Quick-Connect CLI, Gamer-Whisper preemption, `/noel` & `/quick` pages, and usability feedback API."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import hivemind_quickstart as hc  # noqa: E402
from hivemind.router.accounts import RouterStore  # noqa: E402
from hivemind.router.gateway import create_router_app  # noqa: E402


def _make_mock_http(registered_key: str = "hm-test-holiday-key-123456"):
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def _http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        calls.append((method, url, payload))
        if url.endswith("/v1/hivemind/register"):
            assert payload is not None
            return 200, {
                "account_id": "acc_holiday_1",
                "api_key": registered_key,
                "role": payload["role"],
            }
        if url.endswith("/v1/hivemind/me"):
            assert headers and headers.get("Authorization") == f"Bearer {registered_key}"
            return 200, {
                "account_id": "acc_holiday_1",
                "handle": "sophia-dev",
                "role": "prosumer",
                "balance": 0.0,
            }
        if url.endswith("/v1/models"):
            return 200, {"data": [{"id": "hivemind/coder"}, {"id": "hivemind/bee-stars"}]}
        if url.endswith("/v1/hivemind/onboarding/feedback"):
            assert payload is not None
            return 200, {
                "object": "hivemind.onboarding_feedback_summary",
                "total_responses": 1,
                "mean_ease_rating": float(payload["ease_rating"]),
                "zero_game_lag_ratio": 1.0 if payload["zero_game_lag"] else 0.0,
            }
        return 404, {"error": "not_found"}

    return calls, _http


def test_quick_connect_registers_saves_reuses_and_uninstalls(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HIVEMIND_API_KEY", raising=False)
    cred_file = tmp_path / ".hivemind" / "credentials.json"
    cont_file = tmp_path / ".continue" / "hivemind_provider.yaml"
    calls, mock_http = _make_mock_http()

    rc = hc.main(
        [
            "--gateway",
            "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
            "--handle",
            "sophia-dev",
            "--credentials-file",
            str(cred_file),
            "--continue-config-path",
            str(cont_file),
            "--write-continue-config",
            "--json",
        ],
        http_fn=mock_http,
    )
    assert rc == 0
    assert cred_file.is_file()
    assert cont_file.is_file()
    if os.name != "nt":
        assert (cred_file.stat().st_mode & 0o777) == 0o600
    saved = json.loads(cred_file.read_text(encoding="utf-8"))
    assert saved["api_key"] == "hm-test-holiday-key-123456"
    assert "hivemind/coder" in cont_file.read_text(encoding="utf-8")
    assert sum(1 for _, u, _ in calls if u.endswith("/v1/hivemind/register")) == 1

    # Second call reuses saved credentials without calling /register again
    rc2 = hc.main(
        [
            "--gateway",
            "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
            "--credentials-file",
            str(cred_file),
            "--continue-config-path",
            str(cont_file),
            "--json",
        ],
        http_fn=mock_http,
    )
    assert rc2 == 0
    assert sum(1 for _, u, _ in calls if u.endswith("/v1/hivemind/register")) == 1

    # Clean 1-command uninstall removes both files
    rc_un = hc.main(
        [
            "--uninstall",
            "--credentials-file",
            str(cred_file),
            "--continue-config-path",
            str(cont_file),
        ],
        http_fn=mock_http,
    )
    assert rc_un == 0
    assert not cred_file.exists()
    assert not cont_file.exists()


def test_quick_connect_refuses_untrusted_gateway_and_bad_feedback(tmp_path: Path) -> None:
    _, mock_http = _make_mock_http()
    assert hc.main(["--gateway", "https://evil.example.com"], http_fn=mock_http) == 2
    assert (
        hc.main(
            [
                "--gateway",
                "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
                "--feedback",
                "--ease-rating",
                "9",
            ],
            http_fn=mock_http,
        )
        == 2
    )
    assert (
        hc.main(
            [
                "--gateway",
                "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
                "--feedback",
                "--profile",
                "gamer",
                "--ease-rating",
                "5",
            ],
            http_fn=mock_http,
        )
        == 0
    )


def test_gamer_whisper_guard_and_sub_second_game_preemption() -> None:
    cool_gpu = hc.GpuSnapshot(
        present=True,
        name="NVIDIA GeForce RTX 4070",
        total_mib=12288,
        free_mib=10240,
        temp_c=52.0,
        util_pct=3.0,
    )
    ok = hc.evaluate_gamer_whisper_guard(cool_gpu, running_procs={"explorer.exe", "discord.exe"})
    assert ok["allow"] is True

    # Game active -> refused immediately
    game_blocked = hc.evaluate_gamer_whisper_guard(cool_gpu, running_procs={"steam.exe", "cs2.exe"})
    assert game_blocked["allow"] is False
    assert game_blocked["active_game"] == "cs2.exe"

    # Temperature >= 70 C -> refused in whisper mode
    hot_gpu = hc.GpuSnapshot(
        present=True, name="RTX 4070", total_mib=12288, free_mib=10240, temp_c=71.0, util_pct=5.0
    )
    assert hc.evaluate_gamer_whisper_guard(hot_gpu, running_procs=set())["allow"] is False

    # Live worker preemption test: game starts on 2nd poll -> child process killed in < 1.5s
    poll_count = {"n": 0}

    def _procs() -> set[str]:
        poll_count["n"] += 1
        return set() if poll_count["n"] < 2 else {"eldenring.exe"}

    def _fake_worker(_cmd: list[str], **kwargs: Any) -> subprocess.Popen[Any]:
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)

    res = hc.run_gamer_whisper_session(
        gateway="https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        api_key="hm-test-key",
        extra_games=[],
        reserve_mib=2048,
        hot_c=70.0,
        cool_c=64.0,
        once=True,
        gpu_probe_fn=lambda: cool_gpu,
        proc_list_fn=_procs,
        popen_fn=_fake_worker,
        poll_interval_s=0.05,
    )
    assert res["status"] == "yielded_to_owner"
    assert res["active_game"] == "eldenring.exe"
    assert float(res["yield_elapsed_s"]) < 1.5


def test_gateway_noel_quick_pages_and_feedback_endpoints() -> None:
    app = create_router_app(RouterStore(":memory:", b"test-secret-0123456789abcdef0123456789abcdef"))
    client = TestClient(app)

    r_fr = client.get("/noel")
    assert r_fr.status_code == 200
    assert "Accès Simple Vacances de Noël" in r_fr.text
    assert "0,00 EUR" in r_fr.text
    assert "join_hivemind.ps1" in r_fr.text

    r_en = client.get("/quick")
    assert r_en.status_code == 200
    assert "Holiday Quick Start" in r_en.text
    assert "personal home machines" in r_en.text

    # Trailing punctuation typo on chat apps redirects cleanly to /noel
    r_typo = client.get("/noel)", follow_redirects=False)
    assert r_typo.status_code in (301, 302, 307, 308)
    assert r_typo.headers["location"] == "/noel"

    # Usability & comfort feedback loop
    s0 = client.get("/v1/hivemind/onboarding/feedback/summary").json()
    assert s0["total_responses"] == 0

    fb1 = client.post(
        "/v1/hivemind/onboarding/feedback",
        json={"profile": "gamer", "ease_rating": 5, "zero_game_lag": True, "platform": "windows"},
    )
    assert fb1.status_code == 200
    fb2 = client.post(
        "/v1/hivemind/onboarding/feedback",
        json={"profile": "dev", "ease_rating": 4, "zero_game_lag": True, "platform": "linux"},
    )
    assert fb2.status_code == 200
    summary = fb2.json()
    assert summary["total_responses"] == 2
    assert summary["mean_ease_rating"] == 4.5
    assert summary["zero_game_lag_ratio"] == 1.0
    assert summary["by_profile"] == {"dev": 1, "gamer": 1}

    # Negative controls: out-of-range rating and unknown fields rejected
    bad_rating = client.post(
        "/v1/hivemind/onboarding/feedback",
        json={"profile": "dev", "ease_rating": 0, "zero_game_lag": True},
    )
    assert bad_rating.status_code == 400
    bad_extra = client.post(
        "/v1/hivemind/onboarding/feedback",
        json={"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "email": "secret@example.com"},
    )
    assert bad_extra.status_code == 400

    # Zero-clone script download endpoints (quickstart.py + join_hivemind.ps1) and 404 guard
    dl_py = client.get("/scripts/hivemind_quickstart.py")
    assert dl_py.status_code == 200
    assert "def ensure_connected_account" in dl_py.text

    dl_ps1 = client.get("/scripts/join_hivemind.ps1")
    assert dl_ps1.status_code == 200
    assert "GamerWhisper" in dl_ps1.text

    dl_bad = client.get("/scripts/unknown_evil.py")
    assert dl_bad.status_code == 404


def test_powershell_and_bash_launchers_exist_and_have_valid_headers() -> None:
    ps1 = REPO_ROOT / "scripts" / "join_hivemind.ps1"
    sh = REPO_ROOT / "scripts" / "join_hivemind_campaign.sh"
    assert ps1.is_file()
    ps1_text = ps1.read_text(encoding="utf-8")
    assert "AGPL-3.0" in ps1_text
    assert "GamerWhisper" in ps1_text
    assert "0.00 EUR" in ps1_text

    proc_syntax = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True, timeout=10, check=False)
    assert proc_syntax.returncode == 0
    proc_help = subprocess.run(["bash", str(sh), "--help"], capture_output=True, text=True, timeout=10, check=False)
    assert proc_help.returncode == 0
    assert "--gamer-whisper" in proc_help.stdout
    assert "--connect" in proc_help.stdout
