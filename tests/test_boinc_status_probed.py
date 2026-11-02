# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""IP-177: /boinc/status reports the Ruche's probed state, never a literal ONLINE."""

from __future__ import annotations

import http.server
import threading

from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app


class _Ok(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args: object) -> None:
        return


def _clear_probe_cache() -> None:
    # imported lazily so the module still collects on a main that lacks remote_probe (the negative control)
    try:
        from hivemind.router import remote_probe
    except ImportError:
        return
    remote_probe._CACHE.clear()


def test_boinc_status_is_unreachable_when_the_ruche_cannot_be_reached(monkeypatch) -> None:
    monkeypatch.setenv("HIVEMIND_RUCHE_MASTER_URL", "http://127.0.0.1:9/")
    _clear_probe_cache()
    client = TestClient(create_router_app(RouterStore(":memory:", b"k" * 32)))
    body = client.get("/boinc/status").json()
    assert body["status"] == "UNREACHABLE" and body["status"] != "ONLINE"
    assert body["probe"]["probe_error"] and body["probe"]["probe_url"] == "http://127.0.0.1:9/"
    assert body["status_basis"].startswith("GET http://127.0.0.1:9/")


def test_boinc_status_is_online_only_when_the_ruche_answers(monkeypatch) -> None:
    server = http.server.HTTPServer(("127.0.0.1", 0), _Ok)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/"
        monkeypatch.setenv("HIVEMIND_RUCHE_MASTER_URL", url)
        _clear_probe_cache()
        client = TestClient(create_router_app(RouterStore(":memory:", b"k" * 32)))
        first = client.get("/boinc/status").json()
        assert first["status"] == "ONLINE" and first["probe"]["http_status"] == 200
        assert 0.0 <= first["probe"]["probe_latency_ms"] < 2500.0 and first["probe"]["probe_cached"] is False
        second = client.get("/boinc/status").json()
        assert second["probe"]["probe_cached"] is True and second["status"] == "ONLINE"
    finally:
        server.shutdown()
        server.server_close()
