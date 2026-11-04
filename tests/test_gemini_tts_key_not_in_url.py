# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The Gemini API key goes in a request header, never in the URL, where proxies and logs would record it."""

from __future__ import annotations

import contextlib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

import hivemind.game.exoplanet_hunter_engine as engine_mod

SENTINEL_KEY = "test-sentinel-key-0000"  # a test string, not a credential


def test_the_gemini_request_carries_the_key_in_a_header_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[urllib.request.Request] = []

    def refuse(req: Any, timeout: float | None = None) -> Any:
        seen.append(req)
        raise urllib.error.URLError("no network in tests")

    monkeypatch.setattr(engine_mod, "_load_gemini_api_key_from_env", lambda: SENTINEL_KEY)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    # With no network the second tier raises; that is a separate matter, here only the requests that were made are inspected.
    with contextlib.suppress(urllib.error.URLError):
        engine_mod.synthesize_gemini_or_gcp_free_tts(text="hello", lang="en", voice_name="Puck", cache_dir=tmp_path)
    gemini = [r for r in seen if "generativelanguage.googleapis.com" in r.full_url]
    assert len(gemini) == 1
    assert all(SENTINEL_KEY not in r.full_url for r in seen), "the key appeared in a request URL"
    assert gemini[0].get_header("X-goog-api-key") == SENTINEL_KEY
