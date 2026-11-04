# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The public text-to-speech endpoint must refuse an oversized text before any outbound call or cache write."""

from __future__ import annotations

from typing import Any

import pytest
from starlette.testclient import TestClient

import hivemind.game.exoplanet_hunter_cockpit as cockpit_mod


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def recorder(text: str, lang: str = "en", voice_name: str = "Puck", **_: Any) -> dict[str, Any]:
        seen.append({"chars": len(text), "lang": lang})
        return {"ok": True, "audio_bytes": b"RIFF", "mime_type": "audio/wav", "engine": "recorded", "voice_name": voice_name}

    monkeypatch.setattr(cockpit_mod, "synthesize_gemini_or_gcp_free_tts", recorder)
    return seen


def test_an_oversized_text_is_refused_without_an_outbound_call(calls: list[dict[str, Any]]) -> None:
    client = TestClient(cockpit_mod.create_exoplanet_hunter_fastapi_app())
    r = client.get("/api/astro/gemini-tts", params={"text": "x" * 50_000, "lang": "en"})
    assert r.status_code == 413
    assert calls == []


def test_a_normal_text_is_still_spoken_and_the_limit_is_inclusive(calls: list[dict[str, Any]]) -> None:
    client = TestClient(cockpit_mod.create_exoplanet_hunter_fastapi_app())
    ok = client.get("/api/astro/gemini-tts", params={"text": "y" * cockpit_mod.MAX_TTS_TEXT_CHARS})
    assert ok.status_code == 200 and ok.content == b"RIFF"
    over = client.get("/api/astro/gemini-tts", params={"text": "y" * (cockpit_mod.MAX_TTS_TEXT_CHARS + 1)})
    assert over.status_code == 413
    assert [c["chars"] for c in calls] == [cockpit_mod.MAX_TTS_TEXT_CHARS]
