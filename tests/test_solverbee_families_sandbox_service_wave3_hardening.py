"""Negative-control and positive regression tests for SolverBee hardening (IP-113)."""

from __future__ import annotations

import pytest

from hivemind.solverbee.families import (
    FamilyError,
    catalogue_text,
    compile_slot_spec,
    linear_spec,
    mass_action_spec,
)
from hivemind.solverbee.sandbox import Limits, run_binary
from hivemind.solverbee.service import (
    ServiceError,
    SolverBeeService,
    SpecError,
    check_bind,
    extract_spec_block,
)
from hivemind.solverbee.wave3_verifiers import verify_numeric_solver_trajectory


def test_mass_action_spec_rejects_non_list_tuple_inputs() -> None:
    rx = [{"reactants": {"A": 1}, "products": {"B": 1}, "k": 0.5}]
    valid = mass_action_spec(["A", "B"], rx, [1.0, 0.0], [1.0, 2.0])
    assert valid["problem"]["kind"] == "mass_action"

    with pytest.raises(FamilyError, match="species, reactions and y0 must be lists or tuples"):
        mass_action_spec("AB", rx, [1.0, 0.0], [1.0, 2.0])  # type: ignore[arg-type]

    with pytest.raises(FamilyError, match="species, reactions and y0 must be lists or tuples"):
        mass_action_spec(["A", "B"], "not-a-list", [1.0, 0.0], [1.0, 2.0])  # type: ignore[arg-type]

    with pytest.raises(FamilyError, match="species, reactions and y0 must be lists or tuples"):
        mass_action_spec(["A", "B"], rx, "10", [1.0, 2.0])  # type: ignore[arg-type]


def test_linear_spec_rejects_non_list_tuple_inputs() -> None:
    valid = linear_spec([[-1.0, 0.0], [0.0, -2.0]], [0.0, 0.0], [1.0, 2.0], [1.0, 2.0])
    assert valid["problem"]["kind"] == "linear"

    with pytest.raises(FamilyError, match="A, b and y0 must be lists or tuples"):
        linear_spec("not-a-matrix", [0.0], [1.0], [1.0])  # type: ignore[arg-type]

    with pytest.raises(FamilyError, match="A, b and y0 must be lists or tuples"):
        linear_spec([[-1.0]], "0", [1.0], [1.0])  # type: ignore[arg-type]

    with pytest.raises(FamilyError, match="A must be n x n"):
        linear_spec(["ab", "cd"], [0.0, 0.0], [1.0, 2.0], [1.0])  # type: ignore[list-item]


def test_catalogue_text_and_compile_slot_spec_require_booleans() -> None:
    text = catalogue_text(only_new=True)
    assert "pendulum" in text

    with pytest.raises(FamilyError, match="only_new must be a boolean"):
        catalogue_text(only_new=1)  # type: ignore[arg-type]

    spec = compile_slot_spec(
        {"slot_family": "harmonic", "omega": 2.0, "zeta": 0.1, "x0": 1.0, "v0": 0.0, "t_out": [1.0]},
        return_trace=False,
    )
    assert spec["problem"]["family"] == "harmonic"

    with pytest.raises(FamilyError, match="return_trace must be a boolean"):
        compile_slot_spec(
            {"slot_family": "harmonic", "omega": 2.0, "zeta": 0.1, "x0": 1.0, "v0": 0.0, "t_out": [1.0]},
            return_trace="yes",  # type: ignore[arg-type]
        )


def test_run_binary_validates_limits_argv_and_stdin_types() -> None:
    with pytest.raises(ValueError, match="limits must be a Limits instance"):
        run_binary(["/bin/true"], b"", limits="bad")  # type: ignore[arg-type]

    out_str_argv = run_binary("/bin/true", b"")  # type: ignore[arg-type]
    assert out_str_argv.ok is False
    assert out_str_argv.refused is True
    assert out_str_argv.error == "empty command"

    out_bad_stdin = run_binary(["/bin/true"], "not-bytes")  # type: ignore[arg-type]
    assert out_bad_stdin.ok is False
    assert out_bad_stdin.refused is True
    assert out_bad_stdin.error == "stdin_bytes must be bytes"

    assert isinstance(Limits(), Limits)


def test_extract_spec_block_rejects_non_string() -> None:
    raw = extract_spec_block("Here is the spec:\n```ode-spec\n{\"schema\": \"hivemind-solver-spec/1\"}\n```\n")
    assert raw == b'{"schema": "hivemind-solver-spec/1"}'

    with pytest.raises(SpecError, match="text must be a string"):
        extract_spec_block(None)  # type: ignore[arg-type]

    with pytest.raises(SpecError, match="text must be a string"):
        extract_spec_block(b"```ode-spec\n{}\n```")  # type: ignore[arg-type]


def test_check_bind_and_solverbee_service_validate_inputs() -> None:
    assert check_bind("127.0.0.1", False, None) is None

    with pytest.raises(ServiceError, match="host must be a non-empty string"):
        check_bind("", False, None)

    with pytest.raises(ServiceError, match="allow_nonlocal must be a boolean"):
        check_bind("127.0.0.1", 1, None)  # type: ignore[arg-type]

    svc = SolverBeeService("/bin/true", allow_unsandboxed=False, loopback_only=True)
    assert svc.loopback_only is True

    with pytest.raises(ValueError, match="allow_unsandboxed and loopback_only must be booleans"):
        SolverBeeService("/bin/true", allow_unsandboxed=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="allow_unsandboxed and loopback_only must be booleans"):
        SolverBeeService("/bin/true", loopback_only="yes")  # type: ignore[arg-type]


def test_verify_numeric_solver_trajectory_rejects_non_dict_spec() -> None:
    cert = verify_numeric_solver_trajectory([("id", "bad")], [[1.0, 0.0]], [[1.0, 0.0]])  # type: ignore[arg-type]
    assert cert["verdict"] == "FAILED"
    assert any("spec must be a dict" in r for r in cert["reasons"])
