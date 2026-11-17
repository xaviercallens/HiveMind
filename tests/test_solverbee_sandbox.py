# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests of the Solver Bee binary sandbox runner (fake solver; the real Rust binary is not needed)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hivemind.solverbee import sandbox
from hivemind.solverbee.sandbox import Limits, run_binary
from solverbee_fakes import write_fake_solver

NS = sandbox.namespace_available()
needs_ns = pytest.mark.skipif(not NS, reason="user/mount/net/pid namespaces unavailable here (unshare failed): " + sandbox.availability_reason())
FAST = Limits(timeout_s=3.0)


@pytest.fixture()
def fake(tmp_path: Path) -> str:
    return write_fake_solver(tmp_path)


def run(fake: str, payload: dict, limits: Limits = FAST, args=("run",)):
    return run_binary([fake, *args], json.dumps(payload).encode(), limits=limits)


@needs_ns
def test_well_behaved_solver(fake: str) -> None:
    o = run(fake, {"id": "a", "t_out": [1.0, 2.0], "y0": [1.0, 2.0]})
    assert o.ok and o.rc == 0 and o.limit is None and o.sandbox == "namespace", o
    res = json.loads(o.stdout)
    assert res["status"] == "ok" and res["t"] == [1.0, 2.0] and res["y"] == [[1.0, 2.0]] * 2
    assert o.wall_s < 3


@needs_ns
def test_infinite_loop_killed_by_timeout(fake: str) -> None:
    o = run(fake, {"mode": "loop"}, Limits(timeout_s=1.0))
    assert o.ok and o.limit == "timeout" and o.rc == -9 and o.wall_s < 6


@needs_ns
def test_sleeping_process_killed_by_timeout(fake: str) -> None:
    o = run(fake, {"mode": "sleep"}, Limits(timeout_s=1.0))
    assert o.ok and o.limit == "timeout"


@needs_ns
def test_memory_hog_hits_address_space_limit(fake: str) -> None:
    o = run(fake, {"mode": "hog"}, Limits(timeout_s=5.0, mem_mb=512))
    assert o.ok and o.limit == "memory" and o.rc != 0, o
    assert b"allocated" not in o.stdout


@needs_ns
def test_output_flood_is_capped(fake: str) -> None:
    o = run(fake, {"mode": "flood"}, Limits(timeout_s=10.0, stdout_limit=1024 * 1024))
    assert o.ok and o.limit == "stdout" and len(o.stdout) <= 1024 * 1024 and o.rc == -9


@needs_ns
def test_default_stdout_cap_is_8mib() -> None:
    assert Limits().stdout_limit == 8 * 1024 * 1024 and Limits().stdin_limit == 64 * 1024 and Limits().timeout_s == 30.0


@needs_ns
def test_fork_bomb_is_contained_by_nproc(fake: str) -> None:
    o = run(fake, {"mode": "forkbomb"}, Limits(timeout_s=8.0, nproc=40))
    assert o.ok and o.limit is None, o
    forked = json.loads(o.stdout)["forked"]
    assert forked < 40, forked  # the bound held; with a busy host uid it may be 0, which is also contained
    # the pid namespace and the process-group kill leave nothing behind
    assert o.wall_s < 8


@needs_ns
def test_network_is_unreachable(fake: str) -> None:
    o = run(fake, {"mode": "net"})
    assert o.ok and json.loads(o.stdout) == {"ext": False, "loop": False}


@needs_ns
def test_home_directory_not_visible(fake: str, tmp_path: Path) -> None:
    canary_dir = Path(os.path.expanduser("~"))
    canary = canary_dir / ".solverbee_canary_does_not_matter"
    paths = [str(canary_dir), str(Path(__file__).resolve()), str(tmp_path), os.path.expanduser("~/.claude.json")]
    o = run(fake, {"mode": "home", "paths": paths})
    assert o.ok, o
    r = json.loads(o.stdout)
    assert r["home_list"] == [] and r["tmp_list"] == []
    # the home directory itself exists as an empty tmpfs mountpoint, but the repository test file and the pytest tmp dir are gone
    assert r[str(Path(__file__).resolve())] is False and r[str(tmp_path)] is False and r[os.path.expanduser("~/.claude.json")] is False
    assert not canary.exists()


@needs_ns
def test_environment_is_scrubbed(fake: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOLVERBEE_TEST_SECRET", "hunter2")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-real")
    o = run(fake, {"mode": "env"})
    assert o.ok
    d = json.loads(o.stdout)
    assert set(d["env"]) <= {"PATH", "LC_CTYPE"} and d["env"]["PATH"] == "/usr/bin:/bin"
    assert b"hunter2" not in o.stdout and b"sk-not-real" not in o.stdout
    assert d["cwd"] == "/tmp/w"


@needs_ns
def test_capabilities_dropped_and_no_new_privs(fake: str, tmp_path: Path) -> None:
    probe = tmp_path / "probe"
    probe.write_text("#!/usr/bin/python3\nprint(''.join(l for l in open('/proc/self/status') if l.startswith(('CapEff','NoNewPrivs'))).replace('\\n',' '))\n")
    probe.chmod(0o755)
    o = run_binary([str(probe)], b"", limits=FAST)
    assert o.ok and "CapEff:\t0000000000000000" in o.stdout.decode() and "NoNewPrivs:\t1" in o.stdout.decode()


@needs_ns
def test_stderr_tail_and_exit_code(fake: str) -> None:
    o = run(fake, {"mode": "crash"})
    assert o.ok and o.rc == 3 and "fake crash" in o.stderr_tail and o.limit is None


def test_stdin_over_limit_refused(fake: str) -> None:
    o = run_binary([fake, "run"], b"x" * (64 * 1024 + 1), limits=FAST)
    assert not o.ok and o.refused and "stdin" in o.error


def test_missing_or_non_executable_binary_refused(tmp_path: Path) -> None:
    assert run_binary([str(tmp_path / "nope"), "run"], b"{}", limits=FAST).refused
    f = tmp_path / "plain"
    f.write_text("x")
    assert run_binary([str(f)], b"{}", limits=FAST).refused


def test_expected_sha_mismatch_refused(fake: str) -> None:
    o = run_binary([fake, "run"], b"{}", limits=FAST, expected_sha256="0" * 64)
    assert o.refused and "sha256" in o.error


def test_refuses_without_namespaces_by_default(fake: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_NS_CACHE", False)
    monkeypatch.setattr(sandbox, "_NS_WHY", "simulated")
    o = run(fake, {"id": "a"})
    assert o.refused and not o.ok and o.sandbox == "none" and "allow_unsandboxed" in o.error


def test_unsandboxed_fallback_is_flagged_and_still_limited(fake: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_NS_CACHE", False)
    monkeypatch.setattr(sandbox, "_NS_WHY", "simulated")
    lim = Limits(timeout_s=2.0, allow_unsandboxed=True)
    o = run(fake, {"id": "a", "t_out": [1.0], "y0": [1.0]}, lim)
    assert o.ok and o.sandbox == "none" and json.loads(o.stdout)["status"] == "ok"
    monkeypatch.setenv("SOLVERBEE_TEST_SECRET", "hunter2")
    e = json.loads(run(fake, {"mode": "env"}, lim).stdout)
    assert set(e["env"]) <= {"PATH", "LC_CTYPE"}  # environment scrubbed even without namespaces
    t = run(fake, {"mode": "loop"}, Limits(timeout_s=1.0, allow_unsandboxed=True))
    assert t.ok and t.limit == "timeout" and t.sandbox == "none"
