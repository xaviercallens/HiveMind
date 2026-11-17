# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Test doubles for the Solver Bee sandbox and service tests: a fake solver executable and a fake verifier. No real solver or verifier is needed."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# A python script used as the "binary". It reads the spec bytes from stdin and prints a contract-conformant result. The behaviour is chosen by a "mode" key
# (sandbox tests) or by the spec id prefix before the first dot (service tests). Python without -I because a shebang carries one argument only.
FAKE_SOLVER_SRC = r'''#!/usr/bin/python3
import hashlib, json, os, socket, sys, time

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "version":
        print(json.dumps({"crate_version": "0.0.0-fake", "solver_rev": "0" * 40}))
        return
    raw = sys.stdin.buffer.read()
    try:
        d = json.loads(raw.decode("utf-8"))
    except ValueError:
        d = {}
    mode = d.get("mode") or str(d.get("id", "")).split(".")[0] or "ok"
    if mode == "loop":
        while True:
            pass
    if mode == "sleep":
        time.sleep(600)
    if mode == "slow":
        time.sleep(1.5)
    if mode == "hog":
        try:
            x = bytearray(6 * 1024 * 1024 * 1024)
            print(json.dumps({"hog": "allocated", "n": len(x)}))
        except MemoryError:
            sys.stderr.write("MemoryError: cannot allocate\n")
            sys.exit(7)
        return
    if mode == "flood":
        chunk = b"x" * 65536
        while True:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
    if mode == "forkbomb":
        n = 0
        pids = []
        for _ in range(2000):
            try:
                pid = os.fork()
            except OSError:
                break
            if pid == 0:
                time.sleep(30)
                os._exit(0)
            pids.append(pid)
            n += 1
        print(json.dumps({"forked": n}))
        sys.stdout.flush()
        for p in pids:
            try:
                os.kill(p, 9)
            except OSError:
                pass
        return
    if mode == "net":
        r = {}
        for name, addr in (("ext", ("93.184.216.34", 80)), ("loop", ("127.0.0.1", 9))):
            s = socket.socket()
            s.settimeout(1.0)
            try:
                s.connect(addr)
                r[name] = True
            except OSError:
                r[name] = False
        print(json.dumps(r))
        return
    if mode == "home":
        r = {"home_list": os.listdir("/home") if os.path.isdir("/home") else [], "tmp_list": [n for n in os.listdir("/tmp") if n != "w"]}
        for path in d.get("paths", []):
            r[path] = os.path.exists(path)
        print(json.dumps(r))
        return
    if mode == "env":
        print(json.dumps({"env": dict(os.environ), "cwd": os.getcwd()}))
        return
    if mode == "crash":
        sys.stderr.write("fake crash: segfault-like failure\n")
        sys.exit(3)
    if mode == "garbage":
        print("this is not json")
        return
    if mode == "solvererr":
        print(json.dumps({"schema": "hivemind-solver-result/1", "id": d.get("id"), "status": "error", "error": "step size too small",
                          "input_sha256": hashlib.sha256(raw).hexdigest()}))
        return
    if mode == "echo":
        sys.stdout.buffer.write(raw)
        return
    t_out = d.get("t_out", [1.0])
    y0 = d.get("y0", [1.0])
    res = {"schema": "hivemind-solver-result/1", "id": d.get("id"), "status": "ok", "t": t_out, "y": [list(y0) for _ in t_out],
           "stats": {"steps": 3, "rhs_evals": 9, "jac_evals": None, "last_order": 1},
           "solver": {"name": "rusty-sundials/cvode", "rev": "0" * 40, "method": d.get("method", "bdf"), "crate_version": "0.0.0-fake"},
           "input_sha256": hashlib.sha256(raw).hexdigest()}
    if mode == "wronghash":
        res["input_sha256"] = "0" * 64
    sys.stdout.write(json.dumps(res))

main()
'''


def write_fake_solver(directory: Path, name: str = "fake-solver") -> str:
    path = Path(directory) / name
    path.write_text(FAKE_SOLVER_SRC, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


def result_sha(result: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


class FakeVerifier:
    """Mimics verify(); records its calls. ``behaviour``: pass | fail | raise | audited | reference_required."""

    def __init__(self, behaviour: str = "pass", call_rerun: bool = True) -> None:
        self.behaviour, self.call_rerun = behaviour, call_rerun
        self.calls: list = []

    def verify(self, spec_bytes: bytes, result: dict, *, problem: Optional[dict] = None, rerun: Optional[Callable[[bytes], dict]] = None,
               binary_sha256: Optional[str] = None) -> dict:
        rerun_result = None
        if self.call_rerun and rerun is not None:
            spec = json.loads(spec_bytes)
            tight = dict(spec, rtol=spec.get("rtol", 1e-8) / 100.0)
            rerun_result = rerun(json.dumps(tight).encode("utf-8"))
        self.calls.append({"problem": problem, "binary_sha256": binary_sha256, "rerun_result": rerun_result, "spec_bytes": spec_bytes})
        if self.behaviour == "raise":
            raise RuntimeError("verifier exploded")
        verdict = {"pass": "VERIFIED" if problem is not None else "AUDITED", "fail": "FAILED", "audited": "AUDITED"}.get(self.behaviour, "AUDITED")
        return {"schema": "hivemind-solver-cert/1", "id": result.get("id"), "verdict": verdict, "checks": [{"name": "shape", "status": "pass", "value": None, "bound": None,
                "detail": "fake"}], "reasons": [] if verdict != "FAILED" else ["fake failure"], "input_sha256": hashlib.sha256(spec_bytes).hexdigest(),
                "result_sha256": result_sha(result), "binary_sha256": binary_sha256, "solver_rev": (result.get("solver") or {}).get("rev"), "verifier_version": "1",
                "caveat": "VERIFIED means the listed checks passed. In the E16 measurement 5-7% of VERIFIED code answers still failed hidden tests; it never means 'correct'."}


def have_user_namespaces() -> bool:
    from hivemind.solverbee import sandbox
    return sandbox.namespace_available()


def secret_env_name() -> str:
    return "SOLVERBEE_TEST_SECRET_" + str(os.getpid())
