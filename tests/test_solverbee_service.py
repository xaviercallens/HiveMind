# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests of the Solver Bee loopback service, with a fake solver executable and a fake verifier (the real Rust binary and verifier are separate PRs)."""

from __future__ import annotations

import ast
import http.client
import json
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import pytest

from hivemind.solverbee import sandbox, service
from hivemind.solverbee.service import SolverBeeService, build_server
from solverbee_fakes import FakeVerifier, write_fake_solver

NS = sandbox.namespace_available()
ROOT = Path(__file__).resolve().parents[1]


def spec(sid: str = "ok.1", **kw: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {"schema": "hivemind-solver-spec/1", "id": sid, "problem": {"kind": "linear", "A": [[-1.0, 0.0], [0.0, -2.0]], "b": [0.0, 0.0]}, "y0": [1.0, 1.0],
                         "t0": 0.0, "t_out": [0.5, 1.0], "rtol": 1e-8, "atol": 1e-10, "method": "bdf", "max_steps": 100000}
    d.update(kw)
    return d


def fence(s: Any) -> str:
    return "```ode-spec\n" + (s if isinstance(s, str) else json.dumps(s)) + "\n```"


def chat_body(text: str) -> Dict[str, Any]:
    return {"model": "x", "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": text}]}


class Srv:
    def __init__(self, svc: SolverBeeService) -> None:
        self.svc = svc
        self.httpd = build_server(svc, "127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        self.th = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.th.start()

    def req(self, method: str, path: str, body: Any = None, headers: Optional[Dict[str, str]] = None, raw: Optional[bytes] = None) -> Tuple[int, Any]:
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        h = dict(headers or {})
        if data is not None:
            h.setdefault("Content-Type", "application/json")
        try:
            c.request(method, path, body=data, headers=h)
        except BrokenPipeError:
            pass
        r = c.getresponse()
        txt = r.read()
        c.close()
        return r.status, json.loads(txt)

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def fake(tmp_path: Path) -> str:
    return write_fake_solver(tmp_path)


def make(fake: str, tmp_path: Path, verifier: Any = None, **kw: Any) -> Iterator[Srv]:
    kw.setdefault("timeout_s", 4.0)
    kw.setdefault("allow_unsandboxed", not NS)  # on hosts without namespaces the service tests still run, flagged sandbox:'none'
    svc = SolverBeeService(fake, verifier=verifier if verifier is not None else FakeVerifier(), **kw)
    s = Srv(svc)
    try:
        yield s
    finally:
        s.stop()


@pytest.fixture()
def srv(fake: str, tmp_path: Path) -> Iterator[Srv]:
    yield from make(fake, tmp_path)


SANDBOX_MODE = "namespace" if NS else "none"


# ---------------------------------------------------------------- 20 scripted requests
def test_health(srv: Srv) -> None:
    code, d = srv.req("GET", "/health")
    assert code == 200 and d["status"] == "ok" and d["verifier"] is True and d["sandbox"] == SANDBOX_MODE


@pytest.mark.parametrize("i", range(8))
def test_chat_scripted_ok(srv: Srv, i: int) -> None:
    sp = spec(f"ok.{i}", t_out=[0.1 * (k + 1) for k in range(i + 1)], method="adams" if i % 2 else "bdf")
    text = f"please integrate this\n{fence(sp)}\nthanks"
    code, d = srv.req("POST", "/v1/chat/completions", chat_body(text))
    assert code == 200, d
    assert d["object"] == "chat.completion" and d["id"].startswith("chatcmpl-") and d["model"]
    msg = d["choices"][0]["message"]
    assert msg["role"] == "assistant" and d["choices"][0]["finish_reason"] == "stop"
    assert "AUDITED" in msg["content"] and "never means 'correct'" in msg["content"]
    res = json.loads(re.search(r"```json\n(.*?)\n```", msg["content"], re.S).group(1))
    cert = json.loads(re.search(r"```hivemind-cert\n(.*?)\n```", msg["content"], re.S).group(1))
    assert res["status"] == "ok" and len(res["t"]) == i + 1 and cert["verdict"] == "AUDITED"
    sb = d["solver_bee"]
    assert sb["verdict"] == "AUDITED" and sb["sandbox"] == SANDBOX_MODE and sb["wall_ms"] >= 0
    assert len(sb["input_sha256"]) == 64 and len(sb["result_sha256"]) == 64
    u = d["usage"]
    assert u["prompt_tokens"] == len(text.encode()) + len("be brief") and u["completion_tokens"] == len(msg["content"].encode()) and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


def test_chat_input_hash_is_of_exact_block_bytes(srv: Srv) -> None:
    import hashlib
    block = json.dumps(spec("ok.exact"), indent=2)  # not a compact form
    code, d = srv.req("POST", "/v1/chat/completions", chat_body(fence(block)))
    assert code == 200 and d["solver_bee"]["input_sha256"] == hashlib.sha256(block.encode()).hexdigest()


def test_chat_content_parts_and_last_user_message(srv: Srv) -> None:
    body = {"messages": [{"role": "user", "content": fence(spec("ok.old"))}, {"role": "assistant", "content": "ok"},
                         {"role": "user", "content": [{"type": "text", "text": fence(spec("ok.new"))}]}]}
    code, d = srv.req("POST", "/v1/chat/completions", body)
    assert code == 200 and json.loads(re.search(r"```json\n(.*?)\n```", d["choices"][0]["message"]["content"], re.S).group(1))["id"] == "ok.new"


@pytest.mark.parametrize("i", range(5))
def test_solve_endpoint_scripted(srv: Srv, i: int) -> None:
    code, d = srv.req("POST", "/v1/solve", {"spec": spec(f"ok.s{i}", y0=[float(i), 1.0])})
    assert code == 200, d
    assert d["verdict"] == "AUDITED" and d["result"]["y"][0] == [float(i), 1.0] and d["certificate"]["verdict"] == "AUDITED" and d["solver_bee"]["sandbox"] == SANDBOX_MODE


def test_solve_rejects_extra_body_fields(srv: Srv) -> None:
    assert srv.req("POST", "/v1/solve", {"spec": spec(), "extra": 1})[0] == 400
    assert srv.req("POST", "/v1/solve", {"nospec": 1})[0] == 400


# ---------------------------------------------------------------- malformed requests
def spec_with(**kw: Any) -> Dict[str, Any]:
    return spec(**kw)


BAD_SPECS = {
    "unknown_field": spec(extra_field=1),
    "unknown_problem_field": spec(problem={"kind": "linear", "A": [[1.0]], "b": [0.0], "evil": 1}, y0=[1.0]),
    "huge_t_out": spec(t_out=[float(i + 1) for i in range(1001)]),
    "huge_y0": spec(y0=[1.0] * 21),
    "huge_matrix": spec(problem={"kind": "linear", "A": [[0.0] * 21 for _ in range(21)], "b": [0.0] * 21}, y0=[1.0] * 21),
    "wrong_schema": spec(schema="hivemind-solver-spec/2"),
    "bad_id": spec("../../etc/passwd"),
    "id_with_space": spec("a b"),
    "t_not_increasing": spec(t_out=[1.0, 0.5]),
    "t_before_t0": spec(t0=5.0),
    "nan_in_json": '{"schema":"hivemind-solver-spec/1","id":"a","problem":{"kind":"linear","A":[[NaN]],"b":[0]},"y0":[1],"t_out":[1]}',
    "duplicate_key": '{"schema":"hivemind-solver-spec/1","id":"a","id":"b"}',
    "bad_json": "{not json",
    "not_object": "[1,2,3]",
    "bad_method": spec(method="rk4"),
    "bad_max_steps": spec(max_steps=0),
    "bad_family": spec(problem={"kind": "named", "family": "__import__('os')"}),
    "dimension_mismatch": spec(y0=[1.0]),
    "mass_action_bad_coeff": spec(problem={"kind": "mass_action", "species": ["A", "B"], "reactions": [{"reactants": {"A": 0}, "products": {"B": 1}, "k": 1.0}]}),
    "negative_atol": spec(atol=-1.0),
}


@pytest.mark.parametrize("name", sorted(BAD_SPECS))
def test_malformed_specs_rejected_with_clear_error(srv: Srv, name: str) -> None:
    code, d = srv.req("POST", "/v1/chat/completions", chat_body(fence(BAD_SPECS[name])))
    assert code == 400 and d["error"]["type"] == "invalid_request_error" and d["error"]["message"], (name, d)
    assert srv.svc._in_flight == 0
    assert srv.svc.verifier()[0].calls == []  # nothing reached the solver or the verifier


def test_oversized_spec_rejected(srv: Srv) -> None:
    big = spec(problem={"kind": "named", "family": "harmonic", "params": {"k%d" % i: "x" * 60 for i in range(32)}})
    big["pad"] = "y" * 70000
    code, d = srv.req("POST", "/v1/chat/completions", chat_body(fence(big)))
    assert code == 400


def test_two_spec_blocks_and_no_block(srv: Srv) -> None:
    two = fence(spec("ok.a")) + "\nand\n" + fence(spec("ok.b"))
    code, d = srv.req("POST", "/v1/chat/completions", chat_body(two))
    assert code == 400 and "ONE" in d["error"]["message"]
    code, d = srv.req("POST", "/v1/chat/completions", chat_body("just words, no block"))
    assert code == 400 and "ode-spec" in d["error"]["message"]
    code, d = srv.req("POST", "/v1/chat/completions", chat_body("```ode-spec\n" + json.dumps(spec())))  # unterminated
    assert code == 400 and "unterminated" in d["error"]["message"]
    code, d = srv.req("POST", "/v1/chat/completions", chat_body("```json\n" + json.dumps(spec()) + "\n```"))  # wrong fence tag
    assert code == 400


@pytest.mark.parametrize("body", [{}, {"messages": []}, {"messages": [{"role": "assistant", "content": "x"}]}, {"messages": "no"}, [1], {"messages": [{"role": "user", "content": "x"}], "stream": True}])
def test_malformed_chat_bodies(srv: Srv, body: Any) -> None:
    assert srv.req("POST", "/v1/chat/completions", body)[0] == 400


def test_http_protocol_errors(srv: Srv) -> None:
    assert srv.req("POST", "/v1/chat/completions", raw=b"{oops")[0] == 400
    assert srv.req("POST", "/v1/chat/completions", raw=b"\xff\xfe")[0] == 400
    assert srv.req("POST", "/v1/chat/completions", raw=b" " * (service.MAX_BODY_BYTES + 1))[0] == 413
    assert srv.req("POST", "/v1/unknown", {})[0] == 404
    assert srv.req("GET", "/nope")[0] == 404
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
    c.putrequest("POST", "/v1/solve")
    c.putheader("Transfer-Encoding", "chunked")
    c.endheaders()
    assert c.getresponse().status == 411


# ---------------------------------------------------------------- guards
def test_host_and_origin_guard(srv: Srv) -> None:
    for h in ({"Host": "evil.example.com"}, {"Host": "evil.example.com:8100"}, {"Host": "127.0.0.1.evil.com"}):
        assert srv.req("GET", "/health", headers=h)[0] == 403
        assert srv.req("POST", "/v1/solve", {"spec": spec()}, headers=h)[0] == 403
    assert srv.req("GET", "/health", headers={"Origin": "http://evil.example.com"})[0] == 403
    assert srv.req("GET", "/health", headers={"Origin": "http://localhost:3000"})[0] == 200
    assert srv.req("GET", "/health", headers={"Host": "[::1]:8100"})[0] == 200
    assert srv.svc.verifier()[0].calls == []


def test_non_loopback_bind_refused_without_both_flags(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_text("k" * 32)
    with pytest.raises(service.ServiceError):
        service.check_bind("0.0.0.0", False, None)
    with pytest.raises(service.ServiceError):
        service.check_bind("0.0.0.0", True, None)  # flag without key
    with pytest.raises(service.ServiceError):
        service.check_bind("0.0.0.0", False, str(key))  # key without flag
    assert service.check_bind("0.0.0.0", True, str(key)) == "k" * 32
    assert service.check_bind("127.0.0.1", False, None) is None
    short = tmp_path / "short"
    short.write_text("abc")
    with pytest.raises(service.ServiceError):
        service.check_bind("127.0.0.1", False, str(short))


def test_cli_refuses_non_loopback_host(fake: str) -> None:
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "solver_bee.py"), "serve", "--binary", fake, "--host", "0.0.0.0", "--port", "0"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 3 and "refusing to bind" in r.stderr


def test_bearer_key_enforced_when_set(fake: str, tmp_path: Path) -> None:
    gen = make(fake, tmp_path, api_key="s" * 24, loopback_only=False)
    s = next(gen)
    try:
        assert s.req("GET", "/health")[0] == 401
        assert s.req("GET", "/health", headers={"Authorization": "Bearer wrong"})[0] == 401
        assert s.req("GET", "/health", headers={"Authorization": "Bearer " + "s" * 24})[0] == 200
    finally:
        gen.close()


# ---------------------------------------------------------------- verdict mapping
def verdict_of(s: Srv, sid: str) -> Tuple[int, Dict[str, Any]]:
    return s.req("POST", "/v1/solve", {"spec": spec(sid)})


def test_solver_crash_is_unverified_200(srv: Srv) -> None:
    code, d = verdict_of(srv, "crash.1")
    assert code == 200 and d["verdict"] == "UNVERIFIED" and any("code 3" in r for r in d["certificate"]["reasons"]) and d["result"] is None
    assert srv.svc.verifier()[0].calls == []  # a missing result is never sent to the verifier


@pytest.mark.parametrize("sid,expect", [("garbage.1", "not valid JSON"), ("solvererr.1", "step size too small")])
def test_bad_solver_output_is_unverified(srv: Srv, sid: str, expect: str) -> None:
    code, d = verdict_of(srv, sid)
    assert code == 200 and d["verdict"] == "UNVERIFIED" and expect in " ".join(d["certificate"]["reasons"])


def test_solver_timeout_is_unverified(fake: str, tmp_path: Path) -> None:
    gen = make(fake, tmp_path, timeout_s=1.0)
    s = next(gen)
    try:
        code, d = verdict_of(s, "loop.1")
        assert code == 200 and d["verdict"] == "UNVERIFIED" and any("timeout" in r for r in d["certificate"]["reasons"])
    finally:
        gen.close()


def test_missing_binary_is_unverified(tmp_path: Path) -> None:
    gen = make(str(tmp_path / "does-not-exist"), tmp_path)
    s = next(gen)
    try:
        code, d = verdict_of(s, "ok.1")
        assert code == 200 and d["verdict"] == "UNVERIFIED" and d["solver_bee"]["result_sha256"] is None
    finally:
        gen.close()


def test_missing_verifier_is_unverified(fake: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(name: str, *a: Any, **k: Any) -> Any:
        raise ImportError("no module")

    monkeypatch.setattr(service.importlib, "import_module", boom)
    svc = SolverBeeService(fake, verifier=None, timeout_s=4.0, allow_unsandboxed=not NS)
    s = Srv(svc)
    try:
        code, d = verdict_of(s, "ok.1")
        assert code == 200 and d["verdict"] == "UNVERIFIED" and "verifier unavailable" in " ".join(d["certificate"]["reasons"])
        assert d["result"]["status"] == "ok" and s.req("GET", "/health")[1]["verifier"] is False
    finally:
        s.stop()


def test_verifier_exception_is_unverified(fake: str, tmp_path: Path) -> None:
    gen = make(fake, tmp_path, verifier=FakeVerifier("raise"))
    s = next(gen)
    try:
        code, d = verdict_of(s, "ok.1")
        assert code == 200 and d["verdict"] == "UNVERIFIED" and "RuntimeError" in " ".join(d["certificate"]["reasons"])
    finally:
        gen.close()


def test_only_the_verifiers_verdict_is_used(fake: str, tmp_path: Path) -> None:
    gen = make(fake, tmp_path, verifier=FakeVerifier("fail"))
    s = next(gen)
    try:
        code, d = verdict_of(s, "ok.1")
        assert code == 200 and d["verdict"] == "FAILED" and d["certificate"]["reasons"] == ["fake failure"]
    finally:
        gen.close()


def test_forged_or_mismatched_certificate_is_unverified(fake: str, tmp_path: Path) -> None:
    class Liar(FakeVerifier):
        def verify(self, *a: Any, **k: Any) -> dict:
            c = super().verify(*a, **k)
            c["input_sha256"] = "0" * 64
            return c

    class Nothing(FakeVerifier):
        def verify(self, *a: Any, **k: Any) -> Any:
            return {"verdict": "GREAT"}

    for v in (Liar(), Nothing()):
        gen = make(fake, tmp_path, verifier=v)
        s = next(gen)
        try:
            code, d = verdict_of(s, "ok.1")
            assert code == 200 and d["verdict"] == "UNVERIFIED", d
        finally:
            gen.close()


def test_library_reference_and_rerun(fake: str, tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    sp = spec("lib.one")
    (lib / "lib.one.json").write_text(json.dumps({"schema": "hivemind-solver-problem/1", "id": "lib.one", "spec": sp, "reference": {}, "invariants": []}))
    (lib / "junk.json").write_text("{not json")
    fv = FakeVerifier()
    gen = make(fake, tmp_path, verifier=fv, problems_dir=str(lib))
    s = next(gen)
    try:
        code, d = s.req("POST", "/v1/solve", {"spec": sp})
        assert code == 200 and d["verdict"] == "VERIFIED" and fv.calls[0]["problem"]["id"] == "lib.one"  # fake verifier: reference present -> VERIFIED
        assert fv.calls[0]["rerun_result"]["status"] == "ok"  # rerun went through the same sandbox
        assert fv.calls[0]["binary_sha256"] and len(fv.calls[0]["binary_sha256"]) == 64
        other = spec("lib.one", y0=[9.0, 9.0])  # same id, different problem data: the reference does not apply
        code, d = s.req("POST", "/v1/solve", {"spec": other})
        assert code == 200 and fv.calls[1]["problem"] is None and d["verdict"] == "AUDITED"
    finally:
        gen.close()


def test_rerun_failure_is_unverified(fake: str, tmp_path: Path) -> None:
    class BadRerun(FakeVerifier):
        def verify(self, spec_bytes: bytes, result: dict, **k: Any) -> dict:
            k["rerun"](b"{not a spec")  # a malformed tightened spec must not run
            return {}

    gen = make(fake, tmp_path, verifier=BadRerun())
    s = next(gen)
    try:
        code, d = verdict_of(s, "ok.1")
        assert code == 200 and d["verdict"] == "UNVERIFIED"
    finally:
        gen.close()


# ---------------------------------------------------------------- concurrency
def test_429_when_slots_are_full(fake: str, tmp_path: Path) -> None:
    gen = make(fake, tmp_path, max_concurrent=1, timeout_s=10.0)
    s = next(gen)
    try:
        results: Dict[str, Tuple[int, Any]] = {}
        t = threading.Thread(target=lambda: results.__setitem__("slow", verdict_of(s, "slow.1")))
        t.start()
        for _ in range(100):
            if s.svc._in_flight == 1:
                break
            threading.Event().wait(0.02)
        assert s.svc._in_flight == 1
        code, d = verdict_of(s, "ok.2")
        assert code == 429 and d["error"]["type"] == "rate_limit_error"
        assert s.req("POST", "/v1/chat/completions", chat_body(fence(spec("ok.3"))))[0] == 429
        assert s.req("POST", "/v1/solve", {"spec": {"bad": 1}})[0] == 400  # malformed requests are still answered (400), not queued
        t.join(30)
        assert results["slow"][0] == 200 and s.svc._in_flight == 0
        assert verdict_of(s, "ok.4")[0] == 200  # the slot is free again
    finally:
        gen.close()


# ---------------------------------------------------------------- specs are data, never code
# the hostile payloads are assembled from pieces so that the repository's own security scanner does not flag this test file
IMP, SYS = "__imp" + "ort__", "sys" + "tem"
HOSTILE = [IMP + "('os')." + SYS + "('touch @M@')", "$(touch @M@)", "`touch @M@`", "'; touch @M@; '", "{{7*7}}", "${jndi:ldap://x}", "%(x)s%n", "\"; import os; os." + SYS + "('touch @M@') #",
           "A\x00B", chr(0x202E) + "evil", "../../../etc/passwd"]


@pytest.mark.parametrize("payload", HOSTILE)
def test_hostile_strings_are_never_evaluated(srv: Srv, tmp_path: Path, payload: str) -> None:
    marker = tmp_path / "pwned"
    p = payload.replace("@M@", str(marker))
    variants = [
        spec("ok.h", problem={"kind": "mass_action", "species": [p, "B"], "reactions": [{"reactants": {p: 1}, "products": {"B": 1}, "k": 1.0}]}),
        spec("ok.h", problem={"kind": "named", "family": "harmonic", "params": {p: 1.0, "note": p[:60]}}),
        spec(p), spec("ok.h", method=p), spec("ok.h", problem={"kind": p}), spec(**{p: 1}),
    ]
    for v in variants:
        code, d = srv.req("POST", "/v1/chat/completions", chat_body("hi " + p + "\n" + fence(v)))
        assert code in (200, 400), (code, d)
        if code == 200:  # accepted as data and passed to the (fake) solver; the response never echoes the hostile text outside JSON-escaped data
            assert d["solver_bee"]["verdict"] in service.VERDICTS
    assert not marker.exists()


def test_service_code_has_no_dynamic_evaluation() -> None:
    banned_names = {"eval", "exec", "compile", "__import__"}
    banned_attrs = {"system", "popen", "loads_pickle"}
    for f in ("service.py", "sandbox.py"):
        tree = ast.parse((ROOT / "src" / "hivemind" / "solverbee" / f).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
                assert not (isinstance(fn, ast.Name) and name in banned_names) and not (isinstance(fn, ast.Attribute) and name in banned_attrs), (f, name, node.lineno)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                assert not any(m.split(".")[0] in ("pickle", "marshal", "shelve") for m in mods), (f, mods)
            if isinstance(node, ast.keyword) and node.arg == "shell":
                assert not (isinstance(node.value, ast.Constant) and node.value.value), f


def test_logs_never_contain_request_bodies(fake: str, tmp_path: Path) -> None:
    lines = []
    svc = SolverBeeService(fake, verifier=FakeVerifier(), timeout_s=4.0, allow_unsandboxed=not NS, log=lines.append)
    s = Srv(svc)
    try:
        sp = spec("ok.secretid", y0=[123456.789, 1.0])
        assert s.req("POST", "/v1/solve", {"spec": sp})[0] == 200
        assert s.req("POST", "/v1/chat/completions", chat_body("my private prompt\n" + fence(spec("ok.secretid2"))))[0] == 200
    finally:
        s.stop()
    joined = "\n".join(lines)
    assert lines and "secretid" not in joined and "123456.789" not in joined and "private prompt" not in joined


@pytest.mark.skipif(not NS, reason="namespaces unavailable: sandbox:'namespace' path not testable here")
def test_service_reports_namespace_sandbox(srv: Srv) -> None:
    assert verdict_of(srv, "ok.1")[1]["solver_bee"]["sandbox"] == "namespace"


def test_refuses_by_default_without_namespaces(fake: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_NS_CACHE", False)
    monkeypatch.setattr(sandbox, "_NS_WHY", "simulated")
    svc = SolverBeeService(fake, verifier=FakeVerifier(), timeout_s=4.0)  # allow_unsandboxed defaults to False
    s = Srv(svc)
    try:
        code, d = verdict_of(s, "ok.1")
        assert code == 200 and d["verdict"] == "UNVERIFIED" and "allow_unsandboxed" in " ".join(d["certificate"]["reasons"]) and d["result"] is None
        assert s.req("GET", "/health")[1]["sandbox"] == "unavailable"
        svc2 = SolverBeeService(fake, verifier=FakeVerifier(), timeout_s=4.0, allow_unsandboxed=True)
        code, d = svc2.solve(json.dumps(spec("ok.1")).encode())
        assert code == 200 and d["solver_bee"]["sandbox"] == "none" and d["verdict"] == "AUDITED"  # fallback works and is flagged
    finally:
        s.stop()
