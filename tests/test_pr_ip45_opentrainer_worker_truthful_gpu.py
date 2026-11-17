# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for truthful GPU detection in ``hivemind_opentrainer_agent.py`` and turnkey ``hivemind_opentrainer_worker.py``."""

from __future__ import annotations

from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = REPO_ROOT / "scripts" / "hivemind_opentrainer_agent.py"
WORKER_PATH = REPO_ROOT / "scripts" / "hivemind_opentrainer_worker.py"


def _load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _StubOpenTrainerServer:
    def __init__(self, *, zero_private_prompts: bool = True) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.zero_private_prompts = zero_private_prompts
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {
                        "version": "1.0.0-s36",
                        "amcp_topic": "hivemind.amcp.federation.v1",
                        "portal_category": {
                            "signup_url_path": "/signup?role=trainer",
                            "zero_private_user_prompts_on_trainer": parent.zero_private_prompts,
                        },
                        "firewall_traversal_modes": [
                            {"mode_id": "outbound-https-relay", "inbound_ports_required": 0}
                        ],
                        "active_task_offer": {"round_id": 7, "topic": "hivemind.amcp.federation.v1"},
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                parent.posts.append((self.path, json.loads(raw or "{}")))
                body = json.dumps(
                    {
                        "node": "trainer-01",
                        "aglet_lifecycle_phase": "arrival",
                        "active_task_offer": {"round_id": 7},
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def stub_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[_StubOpenTrainerServer]:
    monkeypatch.setenv("HIVEMIND_API_KEY", "test-trainer-token-01")
    srv = _StubOpenTrainerServer()
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def test_detect_local_gpu_never_invents_rtx2070_when_nvidia_smi_fails(
    monkeypatch: pytest.MonkeyPatch,
    stub_server: _StubOpenTrainerServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Negative control: when nvidia-smi fails, detect_local_gpu returns ('', 0, 0) and main sends zero POSTs."""
    agent = _load_script("hivemind_opentrainer_agent", AGENT_PATH)

    def _fail_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["nvidia-smi"], returncode=1, stdout="", stderr="not found")

    monkeypatch.setattr(subprocess, "run", _fail_run)
    gpu_name, total_mib, free_mib = agent.detect_local_gpu()
    assert (gpu_name, total_mib, free_mib) == ("", 0, 0)
    assert "RTX 2070" not in gpu_name

    rc = agent.main(["--server-url", stub_server.url, "--once"])
    assert rc == 2
    assert stub_server.posts == []
    assert "refusing to send fabricated GPU telemetry" in capsys.readouterr().err


def test_opentrainer_worker_caps_vram_to_5_5_gb_and_verifies_privacy(
    monkeypatch: pytest.MonkeyPatch,
    stub_server: _StubOpenTrainerServer,
) -> None:
    """Positive control: when nvidia-smi reports an 8 GB GPU with 7200 MiB free, --max-vram-gb 5.5 caps free VRAM to 5632 MiB."""
    worker = _load_script("hivemind_opentrainer_worker", WORKER_PATH)

    def _ok_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="NVIDIA GeForce RTX 3070, 8192, 7200\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _ok_run)
    rc = worker.main(["--server-url", stub_server.url, "--max-vram-gb", "5.5", "--verify-privacy", "--once"])
    assert rc == 0
    assert len(stub_server.posts) == 1
    path, payload = stub_server.posts[0]
    assert path == "/v1/hivemind/opentrainer/heartbeat"
    assert payload["gpu_name"] == "NVIDIA GeForce RTX 3070"
    assert payload["vram_total_mib"] == 8192
    assert payload["vram_free_mib"] == 5632


def test_opentrainer_worker_rejects_unverified_privacy_and_nonfinite_knobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative controls: --verify-privacy fails closed when zero_private_user_prompts_on_trainer is False, and NaN/Inf CLI knobs exit 2."""
    worker = _load_script("hivemind_opentrainer_worker", WORKER_PATH)
    agent = _load_script("hivemind_opentrainer_agent", AGENT_PATH)
    monkeypatch.setenv("HIVEMIND_API_KEY", "test-trainer-token-02")

    bad_srv = _StubOpenTrainerServer(zero_private_prompts=False)
    bad_srv.start()
    try:
        rc_priv = worker.main(
            [
                "--server-url",
                bad_srv.url,
                "--vram-total-mib",
                "8192",
                "--vram-free-mib",
                "6400",
                "--verify-privacy",
                "--once",
            ]
        )
        assert rc_priv == 1
        assert bad_srv.posts == []

        for bad_args in (
            ["--server-url", bad_srv.url, "--max-vram-gb", "nan", "--once"],
            ["--server-url", bad_srv.url, "--max-vram-gb", "-1", "--once"],
            ["--server-url", bad_srv.url, "--interval-s", "inf", "--once"],
            ["--server-url", bad_srv.url, "--interval-s", "0", "--once"],
        ):
            assert agent.main(bad_args) == 2
            assert bad_srv.posts == []
    finally:
        bad_srv.stop()


def test_callensxavier_email_login_and_register_restores_without_password_overwrite(
    tmp_path: Path,
) -> None:
    """RouterStore resolves callensxavier@gmail.com / handle callensxavier to xavier_callens and never overwrites a bound password_hash on wrong password."""
    import sqlite3
    from hivemind.router.accounts import RouterStore

    db_path = tmp_path / "router.db"
    store = RouterStore(db_path, b"0123456789abcdef0123456789abcdef")
    store.register("xavier_callens", "prosumer")
    real_pwd = "MyRealLegacyPassword!2026"
    real_hash = store._hash_password("xavier_callens", real_pwd)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE accounts SET email = ?, password_hash = ? WHERE handle = 'xavier_callens'",
            ("callensxavier@gmail.com", real_hash),
        )
        conn.commit()

    # Negative control: wrong password must raise ValueError and leave password_hash untouched
    with pytest.raises(ValueError, match="Invalid"):
        store.login_with_credentials("callensxavier@gmail.com", "WrongPassword123!")
    with pytest.raises(ValueError, match="already registered"):
        store.register_with_credentials("callensxavier", "WrongPassword123!", email="callensxavier@gmail.com")

    # Positive control: both login_with_credentials and register_with_credentials succeed with real_pwd
    login_acct, login_tok, _ = store.login_with_credentials("callensxavier@gmail.com", real_pwd)
    assert login_acct.handle == "xavier_callens"
    assert login_tok.startswith("hm-")

    reg_acc_id, reg_tok, reg_created = store.register_with_credentials("callensxavier", real_pwd, email="callensxavier@gmail.com")
    assert reg_acc_id == login_acct.account_id
    assert reg_created is False
    assert reg_tok.startswith("hm-")

