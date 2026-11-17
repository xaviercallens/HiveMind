# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""A2 harness logic: no model, no network. A split that silently stayed local, or any token mismatch, must not pass."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a2_split_moe_rpc as a2  # noqa: E402


def _r(tokens: list[int], text: str = "x") -> dict:
    return {"tokens": tokens, "text": text, "predicted_n": len(tokens), "predicted_ms": 100.0}


def test_identical_runs_pass_the_comparison() -> None:
    runs = [_r([i, i + 1]) for i in range(20)]
    c = a2.compare_runs(runs, [dict(r) for r in runs])
    assert c["all_identical"] is True and c["identical"] == 20 and c["first_mismatch"] is None


def test_one_mismatch_fails_and_is_located() -> None:
    a = [_r([1, 2, 3], "abc") for _ in range(20)]
    b = [dict(r) for r in a]
    b[7] = _r([1, 9, 3], "azc")
    c = a2.compare_runs(a, b)
    assert c["all_identical"] is False and c["identical"] == 19
    assert c["first_mismatch"]["prompt_index"] == 7 and c["first_mismatch"]["token_position"] == 1


def test_a_missing_split_run_or_empty_tokens_is_not_met() -> None:
    a = [_r([1])] * 20
    assert a2.compare_runs(a, None)["all_identical"] is False
    assert a2.compare_runs(a, a[:5])["all_identical"] is False
    empty = [_r([])] * 20
    assert a2.compare_runs(empty, empty)["all_identical"] is False  # a server that returns no token ids proves nothing


def test_the_log_parser_counts_only_layers_on_the_rpc_device() -> None:
    log = "load_tensors: layer 0 assigned to device CPU\nload_tensors: layer 1 assigned to device RPC0\nload_tensors: layer 2 assigned to device RPC0\nload_tensors: layer 2 assigned to device RPC0\n"
    assert a2.rpc_layers_from_log(log) == 2
    assert a2.rpc_layers_from_log("load_tensors: layer 0 assigned to device CPU\n") == 0


def test_decide_requires_a_real_split_identity_and_speeds() -> None:
    good = {"compared": 20, "identical": 20, "all_identical": True}
    assert a2.decide(good, 12, 50.0, 20.0) == (True, "")
    assert a2.decide(good, 0, 50.0, 20.0)[0] is False  # silently stayed on one host
    assert a2.decide({"compared": 20, "identical": 19, "all_identical": False}, 12, 50.0, 20.0)[0] is False
    assert a2.decide({"compared": 5, "identical": 5, "all_identical": True}, 12, 50.0, 20.0)[0] is False  # too few prompts
    assert a2.decide(good, 12, 50.0, 0.0)[0] is False


def test_token_agreement_counts_tokens_before_the_first_divergence() -> None:
    assert a2.token_agreement([_r([1, 2, 3, 4])], [_r([1, 2, 9, 4])]) == 0.5
    assert a2.token_agreement([_r([1, 2])], [_r([1, 2])]) == 1.0 and a2.token_agreement([_r([1])], []) == 0.0


def test_there_are_twenty_distinct_prompts() -> None:
    assert len(a2.PROMPTS) == 20 and len(set(a2.PROMPTS)) == 20
    assert a2.tokens_per_second([_r([1, 2]), _r([3, 4])]) == 20.0 and a2.tokens_per_second([]) == 0.0
