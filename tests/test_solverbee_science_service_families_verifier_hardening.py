# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-functional tests for SolverBee hardening (Iteration 134).

Covers boundary validation across:
- hivemind.solverbee.science_bee (FakeBackend, OllamaBackend.payload, canonical_bytes,
  with_house_tolerances, spec_sha256, formulate, detect_language, units_mentioned,
  echo_interpreted_problem, explain, specs_match, baseline_formulate, slot_formulate)
- hivemind.solverbee.service (canonical_sha256, load_problem_library, SolverBeeService,
  authorised, make_handler, build_server)
- hivemind.solverbee.families (_state, family_dim, named_spec, describe_family)
- hivemind.solverbee.verifier (sha256_hex, _chk, decide, _check_tightening)
"""

from __future__ import annotations

import json
import math
import pytest

from hivemind.solverbee import families as fam
from hivemind.solverbee import science_bee as sb
from hivemind.solverbee import service as svc
from hivemind.solverbee import verifier as ver


# =====================================================================
# 1. science_bee.py hardening tests
# =====================================================================


def test_fake_backend_rejects_invalid_script() -> None:
    for bad in (None, 123, "not-a-list", {"a": 1}):
        with pytest.raises(ValueError, match="script must be a callable or a list/tuple"):
            sb.FakeBackend(bad)

    fb_list = sb.FakeBackend(['{"cannot_formulate": "test"}'])
    assert fb_list.complete([{"role": "user", "content": "hi"}]) == '{"cannot_formulate": "test"}'

    fb_fn = sb.FakeBackend(lambda msgs: f"got:{len(msgs)}")
    assert fb_fn.complete([{"role": "user", "content": "hi"}]) == "got:1"


def test_ollama_backend_payload_validates_messages() -> None:
    backend = sb.OllamaBackend("qwen3:1.7b", think=False)
    for bad in (None, "bad", [123], [{"role": "user"}], [{"role": 1, "content": "x"}]):
        with pytest.raises(sb.BackendError, match="messages must be a list of role/content dicts"):
            backend.payload(bad)  # type: ignore[arg-type]

    p = backend.payload([{"role": "user", "content": "solve pendulum"}])
    assert p["model"] == "qwen3:1.7b"
    assert p["stream"] is False
    assert p["think"] is False


def test_canonical_bytes_with_house_tolerances_and_spec_sha256_validation() -> None:
    spec = fam.pendulum_spec(0.1, 0.0, [1.0, 2.0])
    for fn in (sb.canonical_bytes, sb.with_house_tolerances, sb.spec_sha256):
        for bad in (None, [], "not-a-dict", 42):
            with pytest.raises(ValueError, match="spec must be a dict"):
                fn(bad)  # type: ignore[arg-type]

    with_tol = sb.with_house_tolerances(spec)
    assert with_tol["rtol"] == sb.HOUSE_RTOL
    assert with_tol["atol"] == sb.HOUSE_ATOL
    assert len(sb.spec_sha256(with_tol)) == 64


def test_formulate_validates_messages_and_backend() -> None:
    fb = sb.FakeBackend([json.dumps(fam.pendulum_spec(0.1, 0.0, [1.0]))])
    for bad_msgs in (None, "msgs", ["not-dict"]):
        with pytest.raises(ValueError, match="messages must be a list of dicts"):
            sb.formulate(bad_msgs, fb)  # type: ignore[arg-type]

    for bad_backend in (None, 123, "backend", object()):
        with pytest.raises(ValueError, match="backend must be a Backend"):
            sb.formulate([{"role": "user", "content": "q"}], bad_backend)  # type: ignore[arg-type]

    res = sb.formulate([{"role": "user", "content": "q"}], fb)
    assert res["ok"] is True
    assert res["attempts"] == 1


def test_detect_language_and_units_mentioned_validate_inputs() -> None:
    for bad in (None, 123, ["fr"]):
        with pytest.raises(ValueError, match="text must be a string"):
            sb.detect_language(bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="question must be a string"):
            sb.units_mentioned(bad)  # type: ignore[arg-type]

    assert sb.detect_language("Quel est le mouvement du pendule après 2 secondes ?") == "fr"
    assert sb.detect_language("What is the pendulum angle after 2 seconds?") == "en"
    um = sb.units_mentioned("A pendulum of 50 cm over 2 seconds at 15 degrees")
    assert um["length"] == ["centimetres"]
    assert um["time"] == ["seconds"]
    assert um["angle"] == ["degrees"]


def test_echo_interpreted_problem_and_explain_validate_inputs() -> None:
    spec = fam.pendulum_spec(0.1, 0.0, [1.0, 2.0])
    for bad_spec in (None, {}, {"problem": "bad"}, {"problem": {}, "y0": "bad", "t_out": [1.0]}):
        with pytest.raises(ValueError, match="spec must be a validated spec dict"):
            sb.echo_interpreted_problem(bad_spec)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="language must be a string"):
        sb.echo_interpreted_problem(spec, language=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="slot_trace must be a dict or None"):
        sb.echo_interpreted_problem(spec, slot_trace="bad")  # type: ignore[arg-type]

    echo = sb.echo_interpreted_problem(spec, language="en")
    assert echo["schema"] == "hivemind-problem-echo/1"
    assert echo["state_variables"] == ["theta", "omega"]

    with pytest.raises(ValueError, match="question must be a string"):
        sb.explain(123, spec, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="spec must be a dict or None"):
        sb.explain("question", "bad-spec", None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="report must be a dict or None"):
        sb.explain("question", spec, "bad-report")  # type: ignore[arg-type]


def test_specs_match_baseline_and_slot_formulate_validation() -> None:
    s1 = fam.pendulum_spec(0.2, 0.0, [1.0, 2.0])
    s2 = fam.pendulum_spec(0.2, 0.0, [1.0, 2.0], spec_id="other-id")
    assert sb.specs_match(s1, s2) is True

    with pytest.raises(ValueError, match="a and b must be dicts"):
        sb.specs_match(None, s2)  # type: ignore[arg-type]
    for bad_rel in (True, -1e-6, math.nan, math.inf, "bad"):
        with pytest.raises(ValueError, match="rel must be a finite non-negative number"):
            sb.specs_match(s1, s2, rel=bad_rel)  # type: ignore[arg-type]

    for bad_q in (None, 123, ["q"]):
        with pytest.raises(ValueError, match="question must be a string"):
            sb.baseline_formulate(bad_q)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="question must be a string"):
            sb.slot_formulate(bad_q)  # type: ignore[arg-type]


# =====================================================================
# 2. service.py hardening tests
# =====================================================================


def test_service_canonical_sha256_and_load_problem_library_validation() -> None:
    with pytest.raises(ValueError):
        svc.canonical_sha256({"x": math.nan})
    with pytest.raises(ValueError):
        svc.canonical_sha256({"x": math.inf})
    assert len(svc.canonical_sha256({"a": 1, "b": [2, 3]})) == 64

    for bad_dir in (123, ["dir"], {"path": "x"}):
        with pytest.raises(ValueError, match="directory must be a path or None"):
            svc.load_problem_library(bad_dir)  # type: ignore[arg-type]
    assert svc.load_problem_library(None) == {}
    assert svc.load_problem_library("") == {}


def test_solver_bee_service_init_authorised_and_server_builders() -> None:
    for bad_key in ("", "   ", 123):
        with pytest.raises(ValueError, match="api_key must be a non-empty string or None"):
            svc.SolverBeeService("/bin/echo", api_key=bad_key)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="log must be callable"):
        svc.SolverBeeService("/bin/echo", log="not-callable")  # type: ignore[arg-type]

    service = svc.SolverBeeService("/bin/echo", api_key="secret-token-123456")
    with pytest.raises(ValueError, match="header must be a string or None"):
        service.authorised(123)  # type: ignore[arg-type]
    assert service.authorised("Bearer secret-token-123456") is True
    assert service.authorised("Bearer wrong-token") is False

    with pytest.raises(ValueError, match="svc must be a SolverBeeService"):
        svc.make_handler(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="svc must be a SolverBeeService"):
        svc.build_server(None, "127.0.0.1", 8080)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="host must be a non-empty string"):
        svc.build_server(service, "", 8080)
    for bad_port in (True, -1, 70000, "8080"):
        with pytest.raises(ValueError, match="port must be an integer in 0..65535"):
            svc.build_server(service, "127.0.0.1", bad_port)  # type: ignore[arg-type]


# =====================================================================
# 3. families.py hardening tests
# =====================================================================


def test_families_state_dim_named_spec_and_describe_validation() -> None:
    with pytest.raises(ValueError, match="name, meaning, and unit must be non-empty strings"):
        fam._state("", "angle", "rad")
    with pytest.raises(ValueError, match="name, meaning, and unit must be non-empty strings"):
        fam._state("theta", "  ", "rad")

    with pytest.raises(fam.FamilyError, match="unknown family"):
        fam.family_dim(123)  # type: ignore[arg-type]
    with pytest.raises(fam.FamilyError, match="params must be an object or None"):
        fam.family_dim("stiff_chain", params=[("n", 5)])  # type: ignore[arg-type]
    assert fam.family_dim("stiff_chain", params={"n": 6}) == 6

    with pytest.raises(fam.FamilyError, match="params must be an object or None"):
        fam.named_spec("pendulum", [("g_over_l", 9.81)], [0.1, 0.0], [1.0])  # type: ignore[arg-type]

    with pytest.raises(fam.FamilyError, match="unknown family"):
        fam.describe_family(None)  # type: ignore[arg-type]
    assert "pendulum" in fam.describe_family("pendulum")


# =====================================================================
# 4. verifier.py hardening tests
# =====================================================================


def test_verifier_sha256_hex_chk_decide_and_tightening_validation() -> None:
    for bad_bytes in (None, "str", 123, [1, 2]):
        with pytest.raises(ValueError, match="data must be bytes"):
            ver.sha256_hex(bad_bytes)  # type: ignore[arg-type]
    assert len(ver.sha256_hex(b"hivemind")) == 64

    with pytest.raises(ValueError, match="name must be a non-empty string"):
        ver._chk("", "pass", "ok")
    with pytest.raises(ValueError, match="status must be 'pass', 'fail', or 'skip'"):
        ver._chk("shape", "unknown", "ok")
    with pytest.raises(ValueError, match="detail must be a string"):
        ver._chk("shape", "pass", 123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="checks must be a dict"):
        ver.decide(None, usable=True, has_problem=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="checks must contain valid status dicts"):
        ver.decide({"shape": {"status": "pass"}}, usable=True, has_problem=False)

    spec = fam.pendulum_spec(0.1, 0.0, [1.0])
    spec_bytes = sb.canonical_bytes(spec)
    result = {
        "schema": ver.RESULT_SCHEMA,
        "id": spec["id"],
        "status": "ok",
        "t": [1.0],
        "y": [[0.05, -0.1]],
        "input_sha256": ver.sha256_hex(spec_bytes),
    }
    cert = ver.verify(spec_bytes, result, rerun="not-a-callable")  # type: ignore[arg-type]
    assert cert["verdict"] == "FAILED"
    assert any("rerun must be callable" in r for r in cert["reasons"])
