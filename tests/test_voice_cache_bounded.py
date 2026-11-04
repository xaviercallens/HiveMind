# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Every distinct text used to be cached on disk forever; the voice cache is now bounded by file count and by bytes."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import pytest

import hivemind.game.exoplanet_hunter_engine as engine_mod

MP3 = b"\xff\xfb" + b"a" * 400  # longer than the 200-byte minimum the helper accepts as audio


class _Resp:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _speak_many(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    monkeypatch.setattr(engine_mod, "_load_gemini_api_key_from_env", lambda: None)
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp(MP3))
    for i in range(n):
        res = engine_mod.synthesize_gemini_or_gcp_free_tts(text=f"distinct sentence number {i}", lang="en", cache_dir=tmp_path)
        assert res["ok"] is True


def test_the_cache_never_holds_more_files_than_the_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_mod, "VOICE_CACHE_MAX_FILES", 5, raising=False)
    _speak_many(tmp_path, monkeypatch, 12)
    assert len(list(tmp_path.iterdir())) <= 5


def test_the_cache_never_holds_more_bytes_than_the_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_mod, "VOICE_CACHE_MAX_BYTES", 3 * len(MP3), raising=False)
    _speak_many(tmp_path, monkeypatch, 10)
    assert sum(f.stat().st_size for f in tmp_path.iterdir()) <= 3 * len(MP3)


def test_the_newest_entry_survives_and_is_served_from_the_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_mod, "VOICE_CACHE_MAX_FILES", 3, raising=False)
    _speak_many(tmp_path, monkeypatch, 6)
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: (_ for _ in ()).throw(AssertionError("went to the network")))
    again = engine_mod.synthesize_gemini_or_gcp_free_tts(text="distinct sentence number 5", lang="en", cache_dir=tmp_path)
    assert again["ok"] is True and again["cached"] is True
