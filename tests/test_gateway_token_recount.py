# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Billing caps a producer's completion count at RECOUNT_SLACK x the gateway's own tokenizer count (MVP E2)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tokenizer_file(path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {"[UNK]": 0, "hello": 1, "world": 2, "regex": 3}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.save(str(path))


def test_meter_usage_caps_an_inflated_report_at_the_recount() -> None:
    from hivemind.router.gateway import RECOUNT_SLACK, meter_usage

    assert meter_usage(0, 10**9, 0, 4000, 10**6) == (0, 4000, 4000)  # bytes cap only: about 4x over 1,000 honest tokens
    p, c, b = meter_usage(0, 10**9, 0, 4000, 10**6, counted_completion=1000)
    assert c == b == int(RECOUNT_SLACK * 1000) == 1250
    assert meter_usage(0, 900, 0, 4000, 10**6, counted_completion=1000)[1] == 900  # an honest under-report stays
    assert meter_usage(0, 10, 0, 4000, 10**6, counted_completion=None)[1] == 10


def test_token_counter_uses_the_model_file_and_falls_back_to_none(tmp_path) -> None:
    from hivemind.router.gateway import TokenCounter

    _tokenizer_file(tmp_path / "hivemind__crowd-v1.json")
    counter = TokenCounter(tmp_path)
    assert counter.count("hivemind/crowd-v1", "hello world regex") == 3
    assert counter.count("hivemind/coder", "hello") is None  # no file for this model
    assert TokenCounter("").count("hivemind/crowd-v1", "hello") is None  # not configured
    (tmp_path / "hivemind__broken.json").write_text("not a tokenizer")
    assert counter.count("hivemind/broken", "hello") is None  # a broken file never breaks billing


def test_e2_billing_check_passes_only_when_the_recount_is_deployed() -> None:
    spec = importlib.util.spec_from_file_location("e2_security_eval", ROOT / "scripts" / "e2_security_eval.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    met = mod.metering_eval()
    assert met["bytes_cap_only"]["max_inflation_factor"] == 4.0
    assert met["tokenizer_recount"]["max_inflation_factor"] == 1.25
    expected = 1.25 if met["tokenizer_recount_deployed"] else 4.0
    assert met["max_inflation_factor"] == expected
