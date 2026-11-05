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

    def __exit__(self, *exc: Any) -> bool:
        return bool(exc and False)


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


def test_gemini_wav_cache_is_bounded_by_files_and_bytes_and_rejects_bad_voice_before_fs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IP-269 (#583): gemini_*.wav cache entries are pruned to VOICE_CACHE_MAX_FILES/BYTES, and over-long/unlisted voice fails before touching cache_dir."""
    import base64
    import json

    uncreated_dir = tmp_path / "never_created"
    with pytest.raises(ValueError, match="voice_name"):
        engine_mod.synthesize_gemini_or_gcp_free_tts("hi", "en", "P" * 300, cache_dir=uncreated_dir)
    with pytest.raises(ValueError, match="voice_name"):
        engine_mod.synthesize_gemini_or_gcp_free_tts("hi", "en", "NotARealVoice", cache_dir=uncreated_dir)
    with pytest.raises(ValueError, match="lang"):
        engine_mod.synthesize_gemini_or_gcp_free_tts("hi", "en" * 10, "Puck", cache_dir=uncreated_dir)
    assert not uncreated_dir.exists()

    pcm_payload = base64.b64encode(b"\x00\x01" * 256).decode("ascii")
    gemini_json = json.dumps(
        {"candidates": [{"content": {"parts": [{"inlineData": {"data": pcm_payload}}]}}]}
    ).encode("utf-8")
    monkeypatch.setattr(engine_mod, "_load_gemini_api_key_from_env", lambda: "test-key-not-real")
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp(gemini_json))
    monkeypatch.setattr(engine_mod, "VOICE_CACHE_MAX_FILES", 4, raising=False)

    wav_dir = tmp_path / "wav_cache"
    for i in range(9):
        res = engine_mod.synthesize_gemini_or_gcp_free_tts(
            text=f"gemini wav sentence {i}", lang="en", voice_name="Puck", cache_dir=wav_dir
        )
        assert res["ok"] is True and res["mime_type"] == "audio/wav"

    wav_files = list(wav_dir.glob("gemini_*.wav"))
    assert len(wav_files) <= 4
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: (_ for _ in ()).throw(AssertionError("went to the network")),
    )
    cached_wav = engine_mod.synthesize_gemini_or_gcp_free_tts(
        text="gemini wav sentence 8", lang="en", voice_name="Puck", cache_dir=wav_dir
    )
    assert cached_wav["ok"] is True and cached_wav["cached"] is True and cached_wav["mime_type"] == "audio/wav"
