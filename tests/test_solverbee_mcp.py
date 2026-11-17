# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Solver Bee MCP server: scripted JSON-RPC sessions in-process, one session against a subprocess of the real script, and one real solve.

The solver is a fake executable (a Python script that prints contract-shaped results and a family list); the verifier is the injected
FakeVerifier or the real verifier module. Real-binary tests are skipped when the release binary is missing.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from hivemind.solverbee import mcp, sandbox as sbx, service as svc_mod
from solverbee_fakes import FakeVerifier

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "solvers" / "problems"
SCRIPT = ROOT / "scripts" / "solver_bee_mcp.py"
REAL_BINARY = os.environ.get("SOLVERBEE_BINARY", "/mnt/disks/disk-socrateai-local-1/solver-build/target-a/release/hivemind-solver")
NS = sbx.namespace_available()

FAMILIES_DATA = {
    "families": [
        {"family": "robertson", "dim": "3", "equations": "y1'=-k1 y1+k3 y2 y3; ...", "params": [{"name": "k1", "default": 0.04}, {"name": "k2", "default": 30000000.0}, {"name": "k3", "default": 10000.0}]},
        {"family": "harmonic", "dim": "2", "equations": "x'=v; v'=-omega^2 x-2 zeta omega v", "params": [{"name": "omega", "default": 1.0}, {"name": "zeta", "default": 0.0}]},
    ],
    "kinds": ["named", "linear", "mass_action"],
    "limits": {"max_dim": 20, "max_input_bytes": 65536},
    "spec_schema": "hivemind-solver-spec/1",
}

# A fake hivemind-solver: "families" prints the list, "run" reads a spec on stdin. Ids prefixed "solvererr." or "crash." misbehave on purpose.
FAKE_SOLVER_SRC = r'''#!/usr/bin/python3
import hashlib, json, sys

FAMILIES = json.loads(r"""@FAMILIES@""")

def main():
    argv = sys.argv[1:]
    if argv == ["families"]:
        sys.stdout.write(json.dumps(FAMILIES))
        return
    if argv != ["run"]:
        sys.exit(2)
    raw = sys.stdin.buffer.read()
    d = json.loads(raw.decode("utf-8"))
    sid = str(d.get("id", ""))
    if sid.startswith("crash."):
        sys.stderr.write("fake crash\n")
        sys.exit(3)
    if sid.startswith("solvererr."):
        out = {"schema": "hivemind-solver-result/1", "id": sid, "status": "error", "error": "step size too small", "input_sha256": hashlib.sha256(raw).hexdigest()}
        sys.stdout.write(json.dumps(out))
        return
    t_out, y0 = d.get("t_out", [1.0]), d.get("y0", [1.0])
    res = {"schema": "hivemind-solver-result/1", "id": sid, "status": "ok", "t": t_out, "y": [list(y0) for _ in t_out],
           "stats": {"steps": 3, "rhs_evals": 9, "jac_evals": None, "last_order": 1},
           "solver": {"name": "rusty-sundials/cvode", "rev": "0" * 40, "method": d.get("method", "bdf"), "crate_version": "0.0.0-fake"},
           "input_sha256": hashlib.sha256(raw).hexdigest()}
    sys.stdout.write(json.dumps(res))

main()
'''.replace("@FAMILIES@", json.dumps(FAMILIES_DATA))


@pytest.fixture()
def fake_bin(tmp_path: Path) -> str:
    p = tmp_path / "fake-hivemind-solver"
    p.write_text(FAKE_SOLVER_SRC, encoding="utf-8")
    p.chmod(0o755)
    return str(p)


def library_spec(problem_id: str) -> Dict[str, Any]:
    return json.loads((PROBLEMS / f"{problem_id}.json").read_text(encoding="utf-8"))["spec"]


def adhoc(spec_id: str = "adhoc.lin", **over: Any) -> Dict[str, Any]:
    spec = {"schema": "hivemind-solver-spec/1", "id": spec_id, "problem": {"kind": "linear", "A": [[-1.0, 0.0], [0.0, -2.0]], "b": [0.0, 0.0]},
            "y0": [1.0, 1.0], "t0": 0.0, "t_out": [0.5, 1.0]}
    spec.update(over)
    return spec


class Session:
    """Drives one in-process server like a client would: initialize first, then requests; keeps every response."""

    def __init__(self, server: mcp.SolverBeeMcp) -> None:
        self.server = server
        self.next_id = 100

    def raw(self, line: bytes) -> Optional[Dict[str, Any]]:
        return self.server.handle_line(line)

    def call(self, method: str, params: Any = None) -> Dict[str, Any]:
        self.next_id += 1
        msg: Dict[str, Any] = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        if params is not None:
            msg["params"] = params
        return self.server.handle_line(json.dumps(msg).encode("utf-8"))  # type: ignore[return-value]

    def tool(self, name: str, arguments: Any = None) -> Dict[str, Any]:
        resp = self.call("tools/call", {"name": name, "arguments": {} if arguments is None else arguments})
        assert "result" in resp, resp
        return resp["result"]

    def verdict(self, name: str, arguments: Any) -> Dict[str, Any]:
        r = self.tool(name, arguments)
        assert r["isError"] is False, r
        return r["structuredContent"]


def session(server: mcp.SolverBeeMcp) -> Session:
    s = Session(server)
    init = s.call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}})
    assert "result" in init
    assert s.server.handle_line(b'{"jsonrpc":"2.0","method":"notifications/initialized"}') is None
    return s


def make(fake_bin: str, *, verifier: Any = None, problems: Optional[str] = str(PROBLEMS), flagged: bool = True) -> mcp.SolverBeeMcp:
    return mcp.build_server(fake_bin, problems_dir=problems, timeout_s=20, allow_unsandboxed=flagged, verifier=verifier)


# ---------------------------------------------------------------- protocol
def test_initialize_negotiates_protocol_version(fake_bin: str) -> None:
    srv = make(fake_bin)
    for asked, expect in (("2025-03-26", "2025-03-26"), ("2024-11-05", "2024-11-05"), ("1999-01-01", mcp.PROTOCOL_VERSIONS[0])):
        r = Session(srv).call("initialize", {"protocolVersion": asked, "capabilities": {}})
        assert r["result"]["protocolVersion"] == expect
        assert r["result"]["capabilities"] == {"tools": {}}
        assert r["result"]["serverInfo"]["name"] == "hivemind-solver-bee"
        assert "Sandbox: none" in r["result"]["instructions"]


def test_initialize_requires_protocol_version(fake_bin: str) -> None:
    r = Session(make(fake_bin)).call("initialize", {"capabilities": {}})
    assert r["error"]["code"] == mcp.E_INVALID_PARAMS


def test_notifications_never_get_a_response(fake_bin: str) -> None:
    s = session(make(fake_bin))
    assert s.raw(b'{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":3}}') is None
    assert s.raw(b'{"jsonrpc":"2.0","method":"notifications/unknown-thing"}') is None


def test_ping_and_unknown_method(fake_bin: str) -> None:
    s = session(make(fake_bin))
    assert s.call("ping") == {"jsonrpc": "2.0", "id": s.next_id, "result": {}}
    r = s.call("resources/list")
    assert r["error"]["code"] == mcp.E_METHOD_NOT_FOUND


def test_requests_before_initialize_are_refused(fake_bin: str) -> None:
    s = Session(make(fake_bin))
    r = s.call("tools/list")
    assert r["error"]["code"] == mcp.E_INVALID_REQUEST


def test_parse_and_invalid_request_errors(fake_bin: str) -> None:
    s = session(make(fake_bin))
    cases = [
        (b"{not json", mcp.E_PARSE, None),
        (b'{"jsonrpc":"2.0","id":1,"method":"ping","x":NaN}', mcp.E_PARSE, None),
        (b'{"jsonrpc":"2.0","id":1,"method":"ping","id":2}', mcp.E_PARSE, None),
        (b"\xff\xfe", mcp.E_PARSE, None),
        (b"[1,2]", mcp.E_INVALID_REQUEST, None),
        (b'{"id":1,"method":"ping"}', mcp.E_INVALID_REQUEST, 1),  # a detectable id is echoed even when jsonrpc is missing
        (b'{"jsonrpc":"2.0","id":1}', mcp.E_INVALID_REQUEST, 1),
        (b'{"jsonrpc":"2.0","id":{"a":1},"method":"ping"}', mcp.E_INVALID_REQUEST, None),
        (b'{"jsonrpc":"2.0","id":1,"method":7}', mcp.E_INVALID_REQUEST, 1),
        (b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":"x"}', mcp.E_INVALID_PARAMS, 1),
    ]
    for raw, code, rid in cases:
        r = s.raw(raw)
        assert r is not None and r["error"]["code"] == code, (raw, r)
        assert r["id"] == rid, (raw, r)


def test_response_echoes_string_and_numeric_ids(fake_bin: str) -> None:
    s = session(make(fake_bin))
    assert s.raw(b'{"jsonrpc":"2.0","id":"abc-1","method":"ping"}')["id"] == "abc-1"
    assert s.raw(b'{"jsonrpc":"2.0","id":42,"method":"ping"}')["id"] == 42


def test_oversize_message_is_refused_and_drained() -> None:
    srv = mcp.build_server("/nonexistent", allow_unsandboxed=True)
    r = srv.handle_line(b"x" * (mcp.MAX_LINE_BYTES + 1))
    assert r["error"]["code"] == mcp.E_INVALID_REQUEST
    stream = io.BytesIO(b"a" * 50 + b"\n" + b"ok\n")
    first = mcp.read_frame(stream, limit=10)
    assert first is not None and len(first) > 10  # truncated, so the server refuses it
    assert mcp.read_frame(stream, limit=10) == b"ok"
    assert mcp.read_frame(stream, limit=10) is None


def test_encode_is_one_ascii_line() -> None:
    data = mcp.encode({"jsonrpc": "2.0", "id": 1, "result": {"s": "a b\u0085c\nd\re"}})
    assert data.endswith(b"\n") and data.count(b"\n") == 1
    data.decode("ascii")  # no raw U+2028, U+0085 or newline inside the message
    assert json.loads(data)["result"]["s"] == "a b\u0085c\nd\re"


# ---------------------------------------------------------------- tools/list and schemas
def test_tools_list_names_and_schemas(fake_bin: str) -> None:
    s = session(make(fake_bin))
    tools = s.call("tools/list")["result"]["tools"]
    assert {t["name"] for t in tools} == {"solve_ode", "list_problems", "list_families", "verify_result"}
    for t in tools:
        assert t["description"] and t["inputSchema"]["type"] == "object"
        assert t["inputSchema"]["additionalProperties"] is False
    by = {t["name"]: t for t in tools}
    assert by["solve_ode"]["inputSchema"]["required"] == ["spec"]
    assert by["verify_result"]["inputSchema"]["required"] == ["spec", "result"]
    assert by["list_problems"]["inputSchema"]["properties"] == {}
    spec_desc = by["solve_ode"]["inputSchema"]["properties"]["spec"]["description"]
    assert "UNITS ARE YOUR RESPONSIBILITY" in spec_desc and "never executed" in spec_desc
    solve_desc = by["solve_ode"]["description"]
    assert "does NOT mean the physical model is right" in solve_desc and "VERIFIED" in solve_desc


# ---------------------------------------------------------------- malformed params
@pytest.mark.parametrize("params", [
    None,
    ["solve_ode"],
    {"arguments": {}},
    {"name": 5, "arguments": {}},
    {"name": "no_such_tool", "arguments": {}},
    {"name": "solve_ode", "arguments": []},
    {"name": "solve_ode", "arguments": {}},
    {"name": "solve_ode", "arguments": {"spec": "a string"}},
    {"name": "solve_ode", "arguments": {"spec": {}, "extra": 1}},
    {"name": "list_problems", "arguments": {"unexpected": True}},
    {"name": "verify_result", "arguments": {"spec": {}}},
    {"name": "verify_result", "arguments": {"spec": {}, "result": "text"}},
])
def test_invalid_params_are_jsonrpc_errors(fake_bin: str, params: Any) -> None:
    s = session(make(fake_bin))
    r = s.call("tools/call", params)
    assert r["error"]["code"] == mcp.E_INVALID_PARAMS, r


# ---------------------------------------------------------------- solve_ode
def test_solve_ode_library_problem_with_fake_verifier(fake_bin: str) -> None:
    fv = FakeVerifier("pass")
    s = session(make(fake_bin, verifier=fv))
    r = s.tool("solve_ode", {"spec": library_spec("linear_2x2")})
    assert r["isError"] is False
    assert [b["type"] for b in r["content"]] == ["text", "text"]
    assert "VERIFIED" in r["content"][0]["text"] and "never means 'correct'" in r["content"][0]["text"]
    data = r["structuredContent"]
    assert data["verdict"] == "VERIFIED" and data["sandbox"] == "none"
    assert data["certificate"]["schema"] == "hivemind-solver-cert/1"
    assert data["result"]["schema"] == "hivemind-solver-result/1" and data["result"]["status"] == "ok"
    assert json.loads(r["content"][1]["text"]) == data  # the JSON block carries the same result and certificate
    assert fv.calls and fv.calls[0]["problem"] is not None  # the library reference reached the verifier
    assert data["solver_bee"]["sandbox"] == "none"


def test_solve_ode_adhoc_spec_uses_the_real_verifier(fake_bin: str) -> None:
    # The fake solver returns y = y0 for every t, which the verifier cannot tell from a real answer: no reference exists, so AUDITED.
    s = session(make(fake_bin))
    data = s.verdict("solve_ode", {"spec": adhoc()})
    assert data["verdict"] == "AUDITED"
    names = {c["name"]: c["status"] for c in data["certificate"]["checks"]}
    assert names["hashes"] == "pass" and names["tightening"] == "pass" and names["reference"] == "skip"


@pytest.mark.parametrize("spec_id, reason", [("solvererr.1", "solver reported status=error"), ("crash.1", "solver exited with code 3")])
def test_solver_failures_are_unverified(fake_bin: str, spec_id: str, reason: str) -> None:
    s = session(make(fake_bin))
    data = s.verdict("solve_ode", {"spec": adhoc(spec_id)})
    assert data["verdict"] == "UNVERIFIED"
    assert any(reason in x for x in data["certificate"]["reasons"]), data["certificate"]["reasons"]


def test_verifier_exception_fails_closed(fake_bin: str) -> None:
    s = session(make(fake_bin, verifier=FakeVerifier("raise")))
    data = s.verdict("solve_ode", {"spec": adhoc()})
    assert data["verdict"] == "UNVERIFIED"


def test_missing_verifier_module_fails_closed(fake_bin: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "hivemind.solverbee.verifier", None)  # import raises ImportError
    s = session(make(fake_bin))
    data = s.verdict("solve_ode", {"spec": adhoc()})
    assert data["verdict"] == "UNVERIFIED" and "verifier unavailable" in data["certificate"]["reasons"][0]


def test_contract_rejection_is_a_tool_error_not_a_crash(fake_bin: str) -> None:
    s = session(make(fake_bin))
    r = s.tool("solve_ode", {"spec": adhoc(evil="x")})
    assert r["isError"] is True and "unknown spec field" in r["structuredContent"]["error"]
    assert r["structuredContent"]["sandbox"] == "none"


def test_oversize_spec_is_a_tool_error(fake_bin: str) -> None:
    s = session(make(fake_bin))
    big = adhoc(problem={"kind": "named", "family": "harmonic", "params": {"note": "x" * 70000}})
    r = s.tool("solve_ode", {"spec": big})
    assert r["isError"] is True and "larger than" in r["structuredContent"]["error"]


def test_hostile_strings_are_data_never_code(fake_bin: str, tmp_path: Path) -> None:
    # assembled from pieces so that the repository's own security scanner does not flag this file
    marker = tmp_path / "pwned"
    evil = "__imp" + "ort__('os')." + "sys" + f"tem('touch {marker}')"
    s = session(make(fake_bin))
    r = s.tool("solve_ode", {"spec": adhoc("bad id;" + evil)})
    assert r["isError"] is True
    spec = adhoc("ok.hostile", problem={"kind": "named", "family": "harmonic", "params": {"note": evil[:60]}})
    data = s.verdict("solve_ode", {"spec": spec})
    assert data["verdict"] in ("AUDITED", "UNVERIFIED", "FAILED")
    for bad in ("$(touch x)", "`touch x`", "'; touch x; '"):
        assert s.tool("solve_ode", {"spec": adhoc(bad)})["isError"] is True
    assert not marker.exists()


def test_busy_server_refuses_a_second_solve(fake_bin: str) -> None:
    srv = make(fake_bin)
    s = session(srv)
    assert srv._busy.acquire(blocking=False)
    try:
        r = s.tool("solve_ode", {"spec": adhoc()})
        assert r["isError"] is True and "busy" in r["structuredContent"]["error"]
        r2 = s.tool("verify_result", {"spec": adhoc(), "result": {}})
        assert r2["isError"] is True and "busy" in r2["structuredContent"]["error"]
    finally:
        srv._busy.release()


# ---------------------------------------------------------------- list tools
def test_list_problems_reads_the_library(fake_bin: str) -> None:
    s = session(make(fake_bin))
    data = s.verdict("list_problems", {})
    ids = {p["id"] for p in data["problems"]}
    assert data["count"] == len(list(PROBLEMS.glob("*.json"))) == len(ids) >= 12
    assert {"linear_2x2", "harmonic_undamped", "robertson"} <= ids
    assert all(p["description"] for p in data["problems"])
    assert data["sandbox"] == "none"


def test_list_problems_without_directory(fake_bin: str) -> None:
    s = session(make(fake_bin, problems=None))
    r = s.tool("list_problems", {})
    assert r["structuredContent"]["count"] == 0 and "No problems directory" in r["content"][0]["text"]


def test_list_families_from_the_binary(fake_bin: str) -> None:
    s = session(make(fake_bin))
    data = s.verdict("list_families", {})
    fams = {f["family"]: f for f in data["families"]}
    assert set(fams) == {"robertson", "harmonic"}
    assert [p["name"] for p in fams["robertson"]["params"]] == ["k1", "k2", "k3"]
    assert fams["robertson"]["params"][0]["default"] == 0.04
    assert data["sandbox"] == "none" and "robertson" in s.tool("list_families", {})["content"][0]["text"]


def test_list_families_reports_a_broken_binary(tmp_path: Path) -> None:
    bad = tmp_path / "broken"
    bad.write_text("#!/usr/bin/python3\nimport sys\nsys.stdout.write('not json')\n", encoding="utf-8")
    bad.chmod(0o755)
    s = session(make(str(bad)))
    r = s.tool("list_families", {})
    assert r["isError"] is True and "not the expected JSON" in r["structuredContent"]["error"]


# ---------------------------------------------------------------- verify_result
def test_verify_result_round_trip_and_tamper(fake_bin: str) -> None:
    s = session(make(fake_bin))
    spec = adhoc("rt.1")
    result = s.verdict("solve_ode", {"spec": spec})["result"]
    ok = s.verdict("verify_result", {"spec": spec, "result": result})
    assert ok["verdict"] == "AUDITED"
    tampered = json.loads(json.dumps(result))
    tampered["y"][0][0] = tampered["y"][0][0] + 1.0
    bad = s.verdict("verify_result", {"spec": spec, "result": tampered})
    assert bad["verdict"] == "FAILED"


def test_verify_result_library_reference_reaches_the_verifier(fake_bin: str) -> None:
    fv = FakeVerifier("pass")
    s = session(make(fake_bin, verifier=fv))
    spec = library_spec("linear_2x2")
    result = s.verdict("solve_ode", {"spec": spec})["result"]
    calls_before = len(fv.calls)
    data = s.verdict("verify_result", {"spec": spec, "result": result})
    assert data["verdict"] == "VERIFIED"
    assert len(fv.calls) == calls_before + 1
    assert fv.calls[-1]["problem"] is not None  # the library entry (the reference) was passed to the verifier
    assert fv.calls[-1]["binary_sha256"] is None  # the result was not produced by this server's binary, so none is claimed


def test_verify_result_wrong_hash_is_failed(fake_bin: str) -> None:
    s = session(make(fake_bin))
    result = s.verdict("solve_ode", {"spec": adhoc("rt.2")})["result"]
    other = adhoc("rt.2", y0=[2.0, 2.0])  # same id, different initial values: the result's input hash does not match
    assert s.verdict("verify_result", {"spec": other, "result": result})["verdict"] == "FAILED"


def test_verify_result_error_result_is_unverified(fake_bin: str) -> None:
    s = session(make(fake_bin))
    res = {"schema": "hivemind-solver-result/1", "id": "solvererr.9", "status": "error", "error": "nope", "input_sha256": "0" * 64}
    assert s.verdict("verify_result", {"spec": adhoc("solvererr.9"), "result": res})["verdict"] == "UNVERIFIED"


def test_verify_result_reference_needs_the_exact_library_spec(fake_bin: str) -> None:
    s = session(make(fake_bin, verifier=None))  # real verifier, library problems loaded
    changed = library_spec("linear_2x2")
    changed["y0"] = [0.5] + changed["y0"][1:]
    result = s.verdict("solve_ode", {"spec": changed})["result"]
    data = s.verdict("verify_result", {"spec": changed, "result": result})
    assert data["verdict"] == "AUDITED"  # same id as a library entry, but not the same problem: no reference applies


def test_verify_result_missing_verifier_is_unverified(fake_bin: str, monkeypatch: pytest.MonkeyPatch) -> None:
    s = session(make(fake_bin))
    result = s.verdict("solve_ode", {"spec": adhoc("rt.3")})["result"]
    monkeypatch.setitem(sys.modules, "hivemind.solverbee.verifier", None)
    assert s.verdict("verify_result", {"spec": adhoc("rt.3"), "result": result})["verdict"] == "UNVERIFIED"


# ---------------------------------------------------------------- sandbox flags
def test_flagged_mode_says_none_in_every_tool_result(fake_bin: str) -> None:
    s = session(make(fake_bin, flagged=True))
    assert s.server.sandbox_mode == "none"
    results = [
        s.tool("solve_ode", {"spec": adhoc()}), s.tool("list_problems", {}), s.tool("list_families", {}),
        s.tool("verify_result", {"spec": adhoc(), "result": {}}), s.tool("solve_ode", {"spec": {"bad": 1}}),
    ]
    for r in results:
        assert r["structuredContent"]["sandbox"] == "none"
    assert results[0]["structuredContent"]["solver_bee"]["sandbox"] == "none"  # the solver really ran unsandboxed


def test_unflagged_without_namespaces_refuses_to_solve(fake_bin: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sbx, "_NS_CACHE", False)
    srv = make(fake_bin, flagged=False)
    assert srv.sandbox_mode == "unavailable"
    s = session(srv)
    data = s.verdict("solve_ode", {"spec": adhoc()})
    assert data["verdict"] == "UNVERIFIED" and data["sandbox"] == "unavailable"


@pytest.mark.skipif(not NS, reason="user namespaces are unavailable here")
def test_namespace_mode_solve_reports_namespace(fake_bin: str) -> None:
    s = session(make(fake_bin, flagged=False))
    data = s.verdict("solve_ode", {"spec": adhoc()})
    assert data["sandbox"] == "namespace" and data["solver_bee"]["sandbox"] == "namespace"
    assert data["verdict"] == "AUDITED"


def test_script_refuses_to_start_without_namespaces_or_flag(fake_bin: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(sbx, "_NS_CACHE", False)
    spec = importlib.util.spec_from_file_location("solver_bee_mcp_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    rc = mod.main(["--binary", fake_bin, "--problems", str(PROBLEMS)], stdin=io.BytesIO(b""), stdout=io.BytesIO())
    assert rc == 3
    assert "--allow-unsandboxed" in capsys.readouterr().err


def test_script_rejects_missing_binary_and_bad_timeout(fake_bin: str, capsys: pytest.CaptureFixture) -> None:
    spec = importlib.util.spec_from_file_location("solver_bee_mcp_script2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    assert mod.main(["--binary", "/nonexistent/solver", "--allow-unsandboxed"], stdin=io.BytesIO(b""), stdout=io.BytesIO()) == 3
    assert mod.main(["--binary", fake_bin, "--timeout", "0", "--allow-unsandboxed"], stdin=io.BytesIO(b""), stdout=io.BytesIO()) == 3


# ---------------------------------------------------------------- static: no dynamic evaluation
def test_mcp_code_has_no_dynamic_evaluation() -> None:
    banned_names = {"eval", "exec", "compile", "__import__", "pickle"}
    banned_attrs = {"system", "popen"}
    for path in (ROOT / "src" / "hivemind" / "solverbee" / "mcp.py", SCRIPT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id not in banned_names, (path.name, node.id)
            if isinstance(node, ast.Attribute):
                assert node.attr not in banned_attrs, (path.name, node.attr)
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    assert not (kw.arg == "shell" and getattr(kw.value, "value", None) is True), path.name
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                assert not any(n in ("pickle", "marshal", "shelve") for n in names), path.name


# ---------------------------------------------------------------- one session against the real script (subprocess)
def _run_script(args: List[str], lines: List[bytes]) -> subprocess.CompletedProcess:
    stdin = b"".join(line + b"\n" for line in lines)
    return subprocess.run([sys.executable, str(SCRIPT)] + args, input=stdin, capture_output=True, timeout=240)


def _assert_pure_stdout(out: bytes) -> List[Dict[str, Any]]:
    msgs = []
    for line in out.splitlines():
        msg = json.loads(line.decode("ascii"))  # every stdout line is one JSON-RPC message
        assert msg.get("jsonrpc") == "2.0" and ("result" in msg or "error" in msg), msg
        msgs.append(msg)
    return msgs


def test_subprocess_session_stdout_is_pure_jsonrpc(fake_bin: str) -> None:
    lines = [
        b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{}}}',
        b'{"jsonrpc":"2.0","method":"notifications/initialized"}',
        b'{"jsonrpc":"2.0","id":2,"method":"tools/list"}',
        b'{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"list_problems","arguments":{}}}',
        json.dumps({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "solve_ode", "arguments": {"spec": adhoc("sub.1")}}}).encode(),
        b'{"jsonrpc":"2.0","id":5,"method":"ping"}',
        b'{"jsonrpc":"2.0","id":6,"method":"bogus"}',
        b"this is not json",
    ]
    p = _run_script(["--binary", fake_bin, "--problems", str(PROBLEMS), "--allow-unsandboxed"], lines)
    assert p.returncode == 0, p.stderr[-800:]
    msgs = _assert_pure_stdout(p.stdout)
    assert [m["id"] for m in msgs] == [1, 2, 3, 4, 5, 6, None]
    assert msgs[1]["result"]["tools"] and msgs[2]["result"]["structuredContent"]["count"] >= 12
    assert msgs[3]["result"]["structuredContent"]["verdict"] == "AUDITED"
    assert msgs[5]["error"]["code"] == mcp.E_METHOD_NOT_FOUND and msgs[6]["error"]["code"] == mcp.E_PARSE
    assert b"solver_bee_mcp" in p.stderr  # logs went to stderr


@pytest.mark.skipif(not Path(REAL_BINARY).is_file(), reason="the release hivemind-solver binary is not built here")
def test_real_binary_harmonic_undamped_session() -> None:
    harmonic = library_spec("harmonic_undamped")
    lines = [
        b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{}}}',
        b'{"jsonrpc":"2.0","method":"notifications/initialized"}',
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "solve_ode", "arguments": {"spec": harmonic}}}).encode(),
    ]
    args = ["--binary", REAL_BINARY, "--problems", str(PROBLEMS)] + ([] if NS else ["--allow-unsandboxed"])
    p = _run_script(args, lines)
    assert p.returncode == 0, p.stderr[-800:]
    msgs = _assert_pure_stdout(p.stdout)
    data = msgs[1]["result"]["structuredContent"]
    assert data["verdict"] == "VERIFIED", data["certificate"]["reasons"]
    assert data["sandbox"] == ("namespace" if NS else "none")
