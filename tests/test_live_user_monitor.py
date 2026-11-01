# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Checkers of the live monitor: each must accept the right answer and reject near-misses, or the probes prove nothing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

_SPEC = importlib.util.spec_from_file_location("live_user_monitor", Path(__file__).resolve().parents[1] / "scripts" / "live_user_monitor.py")
assert _SPEC is not None and _SPEC.loader is not None
monitor = importlib.util.module_from_spec(_SPEC)
sys.modules["live_user_monitor"] = monitor
_SPEC.loader.exec_module(monitor)


def test_checker_controls_all_hold_for_the_probe_item() -> None:
    controls = monitor.checker_controls("Class: M")
    assert set(controls) >= {"bee_rejects_wrong", "coder_rejects_minus", "arith_rejects_381"}
    assert all(controls.values()), controls


def test_coder_check_reads_the_syntax_tree_not_keywords() -> None:
    assert monitor.check_coder("```python\ndef add(a, b):\n    return b + a\n```")
    assert not monitor.check_coder("```python\ndef add(a, b):\n    return a * b\n```")
    assert not monitor.check_coder("```python\ndef add(x, y):\n    return x + y\n```")  # the prompt names a and b
    assert not monitor.check_coder("```python\ndef add(a, b):\n    return a +\n```")  # syntax error
    assert not monitor.check_coder("def add(a, b): return a + b  # and print(1)\nprint(")


def test_bee_check_is_exact() -> None:
    assert monitor.check_bee(" Class: M\n", "Class: M")
    assert not monitor.check_bee("Class: M because it is red", "Class: M")
    assert not monitor.check_bee("Class: K", "Class: M")


@given(st.lists(st.floats(0.0, 1000.0, allow_nan=False), min_size=1, max_size=50))
def test_percentiles_are_ordered_and_inside_the_sample(values: list[float]) -> None:
    p50 = monitor._percentile(values, 0.5)
    p95 = monitor._percentile(values, 0.95)
    assert min(values) <= p50 <= p95 <= max(values)
    assert p50 in values and p95 in values


def test_percentile_of_nothing_is_none() -> None:
    assert monitor._percentile([], 0.5) is None
    assert monitor._percentile([3.0], 0.95) == 3.0


def test_t4_reader_summarises_governor_telemetry(tmp_path: Path) -> None:
    """The VM-side reader counts only events inside the window and reports yields, states, inactive units and the hold file."""
    import json
    import subprocess

    (tmp_path / "latest.json").write_text(json.dumps({
        "t": 2000.0, "governor": {"state": "training", "boinc_gpu_mode": "always"}, "qlora_progress": {"assimilated": 3},
        "gpu": {"used_mib": 13800, "total_mib": 15360, "util_pct": 97},
        "units": {"a": {"active": "active", "enabled": "enabled"}, "b": {"active": "inactive", "enabled": "enabled"},
                  "c": {"active": "inactive", "enabled": "disabled"}}}))
    rows = [{"event": "snapshot", "state": "serving", "t": 900.0},       # before the window: ignored
            {"event": "snapshot", "state": "training", "t": 1100.0},
            {"event": "yield", "in_flight": 1, "set_never_ok": True, "gpu_freed_ms": 840, "yields": 1, "t": 1200.0},
            {"event": "yield", "in_flight": 1, "set_never_ok": False, "gpu_freed_ms": None, "yields": 2, "t": 1300.0},
            {"event": "yield", "trigger": "nginx", "had_boinc_gpu_task": True, "gpu_freed_ms": 7129, "ended_by_governor": True, "yields": 3, "t": 1250.0},
            {"event": "yield", "trigger": "gpu", "had_boinc_gpu_task": False, "gpu_freed_ms": 60, "yields": 4, "t": 1260.0},
            {"event": "snapshot", "state": "serving", "t": 1201.0}]
    (tmp_path / "governor.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    (tmp_path / "HOLD").write_text("")
    out = subprocess.run([sys.executable, "-c", monitor._T4_READER % (tmp_path.as_posix(), 1000.0)], capture_output=True, text=True, check=True)
    summary = json.loads(out.stdout)
    assert summary["snapshot_states"] == {"training": 1, "serving": 1}
    assert summary["yield_ms"] == [840, 7129, 60]
    assert summary["real_yield_ms"] == [7129]
    assert summary["real_yield_ended_by_governor"] == [True]
    assert summary["units_inactive"] == ["b"]
    assert summary["hold_file"] is True
    assert summary["qlora_progress"] == {"assimilated": 3}


def test_monitor_reads_real_and_test_users_from_the_router_admin_endpoint() -> None:
    """Against the real router app: the monitor reports a flagged account under 'test' and refuses without the admin key."""
    import hashlib
    import hmac

    from fastapi.testclient import TestClient

    from hivemind.router import RouterStore, create_router_app

    secret, admin_key = b"m" * 32, "monitor-admin-key"
    app = create_router_app(RouterStore(":memory:", secret), admin_key_hash=hmac.new(secret, admin_key.encode(), hashlib.sha256).hexdigest())
    client = TestClient(app)
    client.post("/v1/hivemind/register", json={"handle": "real-one", "role": "consumer"})
    client.post("/v1/hivemind/register", headers={"X-Admin-Key": admin_key}, json={"handle": "probe-one", "role": "consumer", "test": True})

    counts = monitor.user_counts(client, "http://testserver", admin_key, 3600.0)
    refused = monitor.user_counts(client, "http://testserver", "wrong-key", 3600.0)

    assert counts["real"]["total"] == 1 and counts["test"]["total"] == 1
    assert counts["real"]["created"] == 1 and counts["test"]["created"] == 1
    assert refused == {"error": "HTTP 403"}


def _chat_row(latency: str, size: int, ua: str = "python-httpx/0.27", status: int = 200) -> dict[str, object]:
    return {"timestamp": "2026-11-01T01:00:00Z",
            "httpRequest": {"latency": latency, "responseSize": str(size), "userAgent": ua, "status": status, "requestUrl": "https://x/v1/chat/completions"}}


def test_breakdown_separates_long_generations_from_slow_small_responses() -> None:
    """Positive control: slow requests with big responses and slow requests with tiny ones land in different buckets."""
    rows = [_chat_row("30.0s", 90_000), _chat_row("28.0s", 80_000), _chat_row("1.0s", 500), _chat_row("0.8s", 400),
            _chat_row("25.0s", 600, ua="Mozilla/5.0")]
    out = monitor.chat_latency_breakdown(rows)
    assert out["by_response_size"]["ge_20kb"]["n"] == 2
    assert out["by_response_size"]["ge_20kb"]["p50_s"] == 28.0  # _percentile rounds the middle index of two values down
    assert out["by_response_size"]["lt_2kb"]["n"] == 3
    assert out["by_response_size"]["lt_2kb"]["p95_s"] == 25.0  # the slow small response is visible, not averaged away
    assert out["by_client"]["browser"]["n"] == 1 and out["by_client"]["scripted"]["n"] == 4
    assert [s["latency_s"] for s in out["slowest"]] == [30.0, 28.0, 25.0, 1.0, 0.8]
    assert out["slowest"][2]["client"] == "browser" and out["slowest"][2]["response_bytes"] == 600


def test_breakdown_negative_control_uniform_traffic_has_no_spread_and_empty_input_is_empty() -> None:
    """Negative control: identical rows give one bucket with equal p50 and p95; no rows give no buckets and no slowest."""
    same = monitor.chat_latency_breakdown([_chat_row("2.0s", 1000) for _ in range(6)])
    assert list(same["by_response_size"]) == ["lt_2kb"] and list(same["by_client"]) == ["scripted"]
    assert same["by_response_size"]["lt_2kb"]["p50_s"] == same["by_response_size"]["lt_2kb"]["p95_s"] == 2.0
    empty = monitor.chat_latency_breakdown([])
    assert empty == {"by_client": {}, "by_response_size": {}, "slowest": []}


def test_breakdown_keeps_no_url_address_or_identity() -> None:
    row = _chat_row("3.0s", 100)
    row["httpRequest"]["remoteIp"] = "203.0.113.9"  # type: ignore[index]
    text = str(monitor.chat_latency_breakdown([row]))
    assert "203.0.113.9" not in text and "v1/chat" not in text and "python-httpx" not in text
