# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The C# fat client the builder emits must send the Gemini key in a header, not in the request URL."""

from __future__ import annotations

import re

import hivemind.game.fat_steam_astro_builder as builder_mod


def test_no_request_url_in_the_csharp_source_carries_a_key_parameter() -> None:
    src = builder_mod.NATIVE_CSHARP_FAT_CLIENT_SOURCE
    assert "generativelanguage.googleapis.com" in src  # the call this test is about still exists
    assert re.search(r"[?&]key=", src) is None, "a request URL in the C# client carries the key"


def test_the_csharp_client_sets_the_key_header_on_the_gemini_request() -> None:
    src = builder_mod.NATIVE_CSHARP_FAT_CLIENT_SOURCE
    assert 'req.Headers["x-goog-api-key"] = apiKey;' in src
    # the header is set on the request that targets the Gemini endpoint, before the body is written
    assert src.index('req.Headers["x-goog-api-key"]') < src.index("req.GetRequestStream()", src.index("generativelanguage.googleapis.com"))
