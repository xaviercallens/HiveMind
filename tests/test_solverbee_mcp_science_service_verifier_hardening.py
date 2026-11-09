# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-functional tests for SolverBee MCP, ScienceBee, Service, and Verifier hardening (Iteration 138)."""

from __future__ import annotations

import math
import pytest

from hivemind.solverbee import families as fam
from hivemind.solverbee import mcp as mcp_mod
from hivemind.solverbee import science_bee as sb
from hivemind.solverbee import service as svc_mod
from hivemind.solverbee import verifier as ver


# =====================================================================
# 1. mcp.py boundary validation tests
# =====================================================================


def test_mcp_error_and_rpc_error_validate_code_and_message() -> None:
    for bad_code in (True, False, "100", 1.5, None):
        with pytest.raises(ValueError, match="code must be an integer"):
            mcp_mod.McpError(bad_code, "bad request")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="code must be an integer"):
            mcp_mod._rpc_error(1, bad_code, "bad request")  # type: ignore[arg-type]

    for bad_msg in ("", "   ", None, 123):
        with pytest.raises(ValueError, match="message must be a non-empty string"):
            mcp_mod.McpError(-32600, bad_msg)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="message must be a non-empty string"):
            mcp_mod._rpc_error(1, -32600, bad_msg)  # type: ignore[arg-type]

    err = mcp_mod.McpError(-32602, "invalid params")
    assert err.code == -32602
    assert err.message == "invalid params"
    rpc = mcp_mod._rpc_error(7, -32602, "invalid params")
    assert rpc == {"jsonrpc": "2.0", "id": 7, "error": {"code": -32602, "message": "invalid params"}}


def test_mcp_spec_bytes_of_and_tool_result_validate_inputs() -> None:
    for bad_spec in (None, "not-a-dict", [1, 2], 42):
        with pytest.raises(TypeError, match="spec must be a dict"):
            mcp_mod.spec_bytes_of(bad_spec)

    spec = fam.pendulum_spec(0.2, 0.0, [1.0, 2.0])
    raw = mcp_mod.spec_bytes_of(spec)
    assert isinstance(raw, bytes) and b'"family":"pendulum"' in raw

    with pytest.raises(ValueError, match="text must be a string"):
        mcp_mod._tool_result(None, {"ok": True})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="payload must be a dict"):
        mcp_mod._tool_result("ok", ["not-a-dict"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="is_error must be a boolean"):
        mcp_mod._tool_result("ok", {"ok": True}, is_error="no")  # type: ignore[arg-type]

    tr = mcp_mod._tool_result("solved", {"verdict": "VERIFIED"}, is_error=False)
    assert tr["isError"] is False
    assert tr["structuredContent"]["verdict"] == "VERIFIED"


def test_mcp_check_args_and_solverbee_mcp_init_and_tool_error() -> None:
    for bad_schema, bad_args in ((None, {}), ({}, None), ("bad", {}), ({}, [])):
        with pytest.raises(mcp_mod.McpError, match="schema and args must be JSON objects"):
            mcp_mod._check_args(bad_schema, bad_args)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="service must be a SolverBeeService"):
        mcp_mod.SolverBeeMcp(None, sandbox_mode="none", problems_dir=None)  # type: ignore[arg-type]

    svc = svc_mod.SolverBeeService("/bin/true", allow_unsandboxed=True)
    with pytest.raises(ValueError, match="problems_dir must be a string or None"):
        mcp_mod.SolverBeeMcp(svc, sandbox_mode="none", problems_dir=123)  # type: ignore[arg-type]

    server = mcp_mod.SolverBeeMcp(svc, sandbox_mode="none", problems_dir=None)
    for bad_msg in ("", "   ", None, 42):
        with pytest.raises(ValueError, match="message must be a non-empty string"):
            server._tool_error(bad_msg)  # type: ignore[arg-type]

    err_res = server._tool_error("solver busy")
    assert err_res["isError"] is True
    assert err_res["structuredContent"]["error"] == "solver busy"


# =====================================================================
# 2. science_bee.py boundary validation tests
# =====================================================================


def test_science_bee_balanced_objects_and_fail_validate_inputs() -> None:
    for bad_text in (None, 123, ["{a:1}"]):
        with pytest.raises(sb.SpecInvalid, match="text must be a string"):
            sb._balanced_objects(bad_text)  # type: ignore[arg-type]

    objs = sb._balanced_objects('prefix {"a": 1, "b": {"c": "}"}} suffix {"d": 2}')
    assert len(objs) == 2

    with pytest.raises(ValueError, match="errors and raws must be lists"):
        sb._fail("not-a-list", [], 1, "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="errors and raws must be lists"):
        sb._fail([], "not-a-list", 1, "bad")  # type: ignore[arg-type]
    for bad_attempts in (True, False, -1, 1.5, "1"):
        with pytest.raises(ValueError, match="attempts must be an integer >= 0"):
            sb._fail([], [], bad_attempts, "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reason must be a string"):
        sb._fail([], [], 1, None)  # type: ignore[arg-type]

    failed = sb._fail(["err1"], ["raw1"], 2, "failed validation")
    assert failed["ok"] is False
    assert failed["retries"] == 1


def test_science_bee_describe_setup_state_names_canon_problem_and_find_helpers() -> None:
    spec = fam.pendulum_spec(0.2, 0.0, [1.0, 2.0])
    for bad_spec in (None, {}, {"problem": {}}, {"y0": [1.0]}):
        with pytest.raises(ValueError, match="spec must be a validated spec dict"):
            sb._describe_setup(bad_spec, "en")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="spec must be a validated spec dict"):
            sb._state_names(bad_spec)  # type: ignore[arg-type]

    for bad_lang in ("de", "", None, 123):
        with pytest.raises(ValueError, match="lang must be 'en' or 'fr'"):
            sb._describe_setup(spec, bad_lang)  # type: ignore[arg-type]

    assert sb._state_names(spec) == ["theta", "omega"]
    assert any("pendulum" in line for line in sb._describe_setup(spec, "en"))

    for bad_canon in (None, {}, {"problem": "not-dict"}):
        with pytest.raises(ValueError, match="spec must be a dict with a problem object"):
            sb._canon_problem(bad_canon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="text must be a string"):
        sb._fr_decimals(None)  # type: ignore[arg-type]
    assert sb._fr_decimals("3,14 et 2,5") == "3.14 et 2.5"

    with pytest.raises(ValueError, match="text must be a string"):
        sb._find(None, ["beta"])  # type: ignore[arg-type]
    for bad_syns in ([], "beta", [""], [123]):
        with pytest.raises(ValueError, match="synonyms must be a non-empty sequence of strings"):
            sb._find("beta = 0.5", bad_syns)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="synonyms must be a non-empty sequence of strings"):
            sb._find_with_unit("L = 10 mH", bad_syns, "mH")  # type: ignore[arg-type]

    for bad_unit in ("", "   ", None, 123):
        with pytest.raises(ValueError, match="text and unit_re must be strings"):
            sb._find_with_unit("L = 10 mH", ["L"], bad_unit)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="text and unit_re must be strings"):
            sb._find_before_unit("10 mH", bad_unit)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="noun_re must be a non-empty string or None"):
        sb._find_before_unit("10 kg mass", "kg", noun_re="   ")

    assert sb._find("beta = 0.5", ["beta"]) == 0.5
    assert sb._find_with_unit("L = 10 mH", ["L"], "mH") == 10.0
    assert sb._find_before_unit("10 kg mass", "kg", noun_re="mass") == 10.0


# =====================================================================
# 3. service.py boundary validation tests
# =====================================================================


def test_service_clean_loopback_hostname_and_internal_helpers() -> None:
    for bad_limit in (True, False, 0, -5, "10"):
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            svc_mod._clean("hello", limit=bad_limit)  # type: ignore[arg-type]
    assert svc_mod._clean("hello\nworld", limit=5) == "hello"

    for bad_host in (None, 123, ["localhost"]):
        with pytest.raises(ValueError, match="h must be a string"):
            svc_mod._is_loopback_host(bad_host)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="host_header must be a string"):
            svc_mod._hostname_of(bad_host)  # type: ignore[arg-type]

    assert svc_mod._is_loopback_host("127.0.0.1") is True
    assert svc_mod._hostname_of("[::1]:8080") == "::1"

    svc = svc_mod.SolverBeeService("/bin/true", allow_unsandboxed=True)
    with pytest.raises(ValueError, match="spec_id must be a non-empty string"):
        svc._unverified("", "abcd", ["err"], None, None)
    with pytest.raises(ValueError, match="input_sha must be a non-empty string"):
        svc._unverified("id-1", "   ", ["err"], None, None)
    with pytest.raises(ValueError, match="reasons must be a list"):
        svc._unverified("id-1", "abcd", "not-a-list", None, None)  # type: ignore[arg-type]

    for bad_spec in (None, {}, {"id": 123}):
        with pytest.raises(ValueError, match="spec must be a dict with a string id"):
            svc._library_problem(bad_spec)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="spec_id must be a non-empty string"):
        svc._summary("", {"verdict": "VERIFIED"}, "none", 10)
    with pytest.raises(ValueError, match="cert must be a dict with verdict"):
        svc._summary("id-1", {}, "none", 10)
    with pytest.raises(ValueError, match="sandbox must be a non-empty string"):
        svc._summary("id-1", {"verdict": "VERIFIED"}, "  ", 10)
    for bad_ms in (True, False, -1, 1.5, "10"):
        with pytest.raises(ValueError, match="wall_ms must be a non-negative integer"):
            svc._summary("id-1", {"verdict": "VERIFIED"}, "none", bad_ms)  # type: ignore[arg-type]

    s = svc._summary("id-1", {"verdict": "AUDITED", "checks": [{"name": "shape", "status": "pass"}]}, "none", 12)
    assert "AUDITED" in s and "12 ms" in s


# =====================================================================
# 4. verifier.py boundary validation tests
# =====================================================================


def test_verifier_safe_dimension_check_hashes_worst_ratio_and_quad() -> None:
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        ver._safe("", lambda: {})
    with pytest.raises(ValueError, match="fn must be callable"):
        ver._safe("shape", "not-callable")  # type: ignore[arg-type]

    for bad_spec in (None, {}, {"y0": "not-list"}):
        with pytest.raises(ver._CheckError, match="spec must contain a y0 list"):
            ver._dimension(bad_spec)  # type: ignore[arg-type]

    for bad_bytes, bad_res in (("str", {}), (b"{}", None), (123, {})):
        with pytest.raises(ver._CheckError, match="spec_bytes must be bytes and result must be a dict"):
            ver._check_hashes(bad_bytes, bad_res)  # type: ignore[arg-type]

    with pytest.raises(ver._CheckError, match="y and y_ref must be lists of equal length"):
        ver._worst_ratio([[1.0]], [[1.0], [2.0]], 1e-8, [1e-10], 100.0)
    with pytest.raises(ver._CheckError, match="rtol and factor must be positive finite numbers"):
        ver._worst_ratio([[1.0]], [[1.0]], 0.0, [1e-10], 100.0)
    with pytest.raises(ver._CheckError, match="rtol and factor must be positive finite numbers"):
        ver._worst_ratio([[1.0]], [[1.0]], 1e-8, [1e-10], math.nan)
    with pytest.raises(ver._CheckError, match="atols must be a non-empty list of positive finite numbers"):
        ver._worst_ratio([[1.0]], [[1.0]], 1e-8, [0.0], 100.0)

    assert ver._worst_ratio([[1.0]], [[1.0]], 1e-8, [1e-10], 100.0) == 0.0

    for bad_m, bad_x in ((None, [1.0]), ([], []), ([[1.0]], [1.0, 2.0])):
        with pytest.raises(ver._CheckError, match="m and x must be non-empty lists of matching dimension"):
            ver._quad(bad_m, bad_x)
    assert ver._quad([[2.0, 0.0], [0.0, 3.0]], [2.0, 1.0]) == 11.0
