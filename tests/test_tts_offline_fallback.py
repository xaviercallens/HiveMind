# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""With no network the text-to-speech helper must answer ok=False and the route must answer its documented 503 fallback, not a 500."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

import hivemind.game.exoplanet_hunter_cockpit as cockpit_mod
import hivemind.game.exoplanet_hunter_engine as engine_mod


def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(req: Any, timeout: float | None = None) -> Any:
        raise urllib.error.URLError("no network in tests")

    monkeypatch.setattr(engine_mod, "_load_gemini_api_key_from_env", lambda: None)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def test_the_helper_reports_failure_instead_of_raising_when_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _offline(monkeypatch)
    res = engine_mod.synthesize_gemini_or_gcp_free_tts(text="hello", lang="en", voice_name="Puck", cache_dir=tmp_path)
    assert res["ok"] is False and not res.get("audio_bytes")
    assert list(tmp_path.iterdir()) == []  # nothing cached from a failed call


def test_the_route_answers_the_503_fallback_when_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _offline(monkeypatch)
    monkeypatch.setattr(engine_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cockpit_mod, "synthesize_gemini_or_gcp_free_tts", lambda **kw: engine_mod.synthesize_gemini_or_gcp_free_tts(cache_dir=tmp_path, **kw))
    client = TestClient(cockpit_mod.create_exoplanet_hunter_fastapi_app(), raise_server_exceptions=False)
    r = client.get("/api/astro/gemini-tts", params={"text": "hello"})
    assert r.status_code == 503 and r.json()["engine"] == "fallback-web-speech"
