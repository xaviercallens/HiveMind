# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""A1 harness logic, no network: it must not pass unless both GPUs really served and the kill cost no request."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a1_multi_gpu_run as a1  # noqa: E402

P = {"t4": "86cd", "rtx": "ef1f"}


def _rows(n: int, pid: str, status: int = 200, charged: float = 0.01) -> list[dict]:
    return [{"status": status, "seconds": 1.0, "producer_id": pid, "charged": charged} for _ in range(n)]


def test_a_clean_failover_passes() -> None:
    before, after = a1.summarize(_rows(12, "86cd1") + _rows(12, "ef1f1"), P), a1.summarize(_rows(24, "ef1f1"), P)
    assert a1.decide(before, after, "t4", "rtx", 0.48) == (True, [])


def test_one_gpu_serving_everything_before_the_kill_is_not_two_gpus() -> None:
    before, after = a1.summarize(_rows(24, "86cd1"), P), a1.summarize(_rows(24, "ef1f1"), P)
    ok, problems = a1.decide(before, after, "t4", "rtx", 0.48)
    assert ok is False and "both GPUs" in problems[0]


def test_a_failed_request_after_the_kill_fails() -> None:
    before = a1.summarize(_rows(12, "86cd1") + _rows(12, "ef1f1"), P)
    after = a1.summarize(_rows(23, "ef1f1") + _rows(1, "", status=502, charged=0.0), P)
    ok, problems = a1.decide(before, after, "t4", "rtx", 0.47)
    assert ok is False and "failed requests" in problems[0]


def test_the_killed_gpu_answering_afterwards_or_a_billing_gap_fails() -> None:
    before = a1.summarize(_rows(12, "86cd1") + _rows(12, "ef1f1"), P)
    assert a1.decide(before, a1.summarize(_rows(24, "86cd1"), P), "t4", "rtx", 0.48)[0] is False
    ok, problems = a1.decide(before, a1.summarize(_rows(24, "ef1f1"), P), "t4", "rtx", 0.60)
    assert ok is False and "balance changed" in problems[0]


def test_too_few_requests_fail_and_labels_work() -> None:
    assert a1.decide(a1.summarize(_rows(2, "86cd1") + _rows(2, "ef1f1"), P), a1.summarize(_rows(4, "ef1f1"), P), "t4", "rtx", 0.08)[0] is False
    assert a1.label("zzz123456789", P) == "other:zzz12345" and a1.label(None, P) == "other:none"
