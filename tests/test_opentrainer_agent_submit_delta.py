# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Contract tests for ``scripts/hivemind_opentrainer_agent.py --submit-delta`` (IP-221).

A stub HTTP gateway records every POST. The delta must only be sent from a measured round report,
and no POST of any kind may happen without one.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = REPO_ROOT / "scripts" / "hivemind_opentrainer_agent.py"


def _load_agent() -> Any:
    spec = importlib.util.spec_from_file_location("hivemind_opentrainer_agent", AGENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["hivemind_opentrainer_agent"] = module
    spec.loader.exec_module(module)
    return module


agent = _load_agent()


class _StubGateway:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server API
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                recorder.posts.append((self.path, json.loads(raw or "{}")))
                body = json.dumps({"node": "stub", "active_task_offer": {"round": 3}}).encode("utf-8")
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

    def submit_delta_posts(self) -> list[dict[str, Any]]:
        return [body for path, body in self.posts if path.endswith("/submit-delta")]


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> Iterator[_StubGateway]:
    monkeypatch.setenv("HIVEMIND_API_KEY", "test-trainer-key")
    stub = _StubGateway()
    stub.start()
    try:
        yield stub
    finally:
        stub.stop()


def _run(gateway: _StubGateway, extra: list[str]) -> int:
    argv = [
        "--gateway",
        gateway.url,
        "--once",
        "--vram-total-mib",
        "8192",
        "--vram-free-mib",
        "6400",
        "--submit-delta",
        *extra,
    ]
    return int(agent.main(argv))


def test_no_post_without_round_report(gateway: _StubGateway, capsys: pytest.CaptureFixture[str]) -> None:
    code = _run(gateway, [])
    assert code == 2
    assert gateway.posts == [], f"expected zero POSTs without a round report, got {gateway.posts}"
    assert "--round-report" in capsys.readouterr().err


def test_no_post_when_report_file_missing(gateway: _StubGateway, tmp_path: Path) -> None:
    code = _run(gateway, ["--round-report", str(tmp_path / "absent.json")])
    assert code == 2
    assert gateway.posts == [], f"expected zero POSTs for a missing report, got {gateway.posts}"


def test_no_post_when_report_lacks_measured_fields(gateway: _StubGateway, tmp_path: Path) -> None:
    report = tmp_path / "round.json"
    report.write_text(json.dumps({"loss_before": 4.0, "loss_after": "n/a"}), encoding="utf-8")
    code = _run(gateway, ["--round-report", str(report)])
    assert code == 2
    assert gateway.posts == [], f"expected zero POSTs for a malformed report, got {gateway.posts}"


def test_delta_sent_only_with_measured_values_and_real_bytes(gateway: _StubGateway, tmp_path: Path) -> None:
    delta_bytes = b"\x00\x01measured-delta-payload\xff" * 64
    (tmp_path / "delta.bin").write_bytes(delta_bytes)
    report = tmp_path / "round.json"
    report.write_text(
        json.dumps(
            {
                "loss_before": 2.3456,
                "loss_after": 1.9876,
                "delta_norm": 0.0731,
                "delta_path": "delta.bin",
            }
        ),
        encoding="utf-8",
    )
    code = _run(gateway, ["--round-report", str(report)])
    assert code == 0
    deltas = gateway.submit_delta_posts()
    assert len(deltas) == 1, f"expected exactly one submit-delta POST, got {len(deltas)}"
    body = deltas[0]
    assert body["loss_before"] == 2.3456
    assert body["loss_after"] == 1.9876
    assert body["delta_norm"] == 0.0731
    assert body["delta_sha256"] == hashlib.sha256(delta_bytes).hexdigest()
    assert body["delta_sha256"] != hashlib.sha256(b"4.1721").hexdigest()
    assert 4.1721 not in [body["loss_before"], body["loss_after"]]


def test_loader_rejects_nonfinite_and_bool_numbers(tmp_path: Path) -> None:
    (tmp_path / "delta.bin").write_bytes(b"x")
    report = tmp_path / "round.json"
    report.write_text(
        json.dumps({"loss_before": True, "loss_after": 1.0, "delta_norm": 0.1, "delta_path": "delta.bin"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="loss_before"):
        agent.load_round_report(str(report))
    report.write_text(
        json.dumps({"loss_before": 1.0, "loss_after": float("nan"), "delta_norm": 0.1, "delta_path": "delta.bin"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="loss_after"):
        agent.load_round_report(str(report))


def test_loader_rejects_missing_delta_file(tmp_path: Path) -> None:
    report = tmp_path / "round.json"
    report.write_text(
        json.dumps({"loss_before": 1.0, "loss_after": 0.5, "delta_norm": 0.1, "delta_path": "nope.bin"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="delta file unreadable"):
        agent.load_round_report(str(report))
