# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Integration: the service with the REAL verifier (default, no injected fake) and the REAL solver binary, in-process.

Skipped when the release binary, the problem library or user namespaces are missing. Set SOLVERBEE_BINARY to point at another build of hivemind-solver.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hivemind.solverbee import sandbox, verifier
from hivemind.solverbee.service import SolverBeeService

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "solvers" / "problems"
BINARY = os.environ.get("SOLVERBEE_BINARY", "/mnt/disks/disk-socrateai-local-1/solver-build/target-a/release/hivemind-solver")

pytestmark = pytest.mark.skipif(
    not (Path(BINARY).is_file() and PROBLEMS.is_dir() and sandbox.namespace_available()),
    reason="needs the built hivemind-solver binary, solvers/problems and user namespaces",
)


def _chat_for(problem_id: str):
    prob = json.loads((PROBLEMS / f"{problem_id}.json").read_text(encoding="utf-8"))
    block = json.dumps(prob["spec"], separators=(",", ":"))
    return {"model": "x", "messages": [{"role": "user", "content": "Solve this:\n```ode-spec\n" + block + "\n```"}]}


def test_default_verifier_is_the_real_module() -> None:
    svc = SolverBeeService(BINARY, problems_dir=str(PROBLEMS))
    v, why = svc.verifier()
    assert v is verifier, why
    assert svc.health()["verifier"] is True


def test_chat_with_library_problem_is_verified_by_real_verifier() -> None:
    svc = SolverBeeService(BINARY, problems_dir=str(PROBLEMS), timeout_s=60)
    code, rep = svc.chat(_chat_for("linear_2x2"))
    assert code == 200
    assert rep["solver_bee"]["verdict"] == "VERIFIED"
    assert rep["solver_bee"]["sandbox"] == "namespace"
    content = rep["choices"][0]["message"]["content"]
    cert = json.loads(content.split("```hivemind-cert\n", 1)[1].split("```", 1)[0])
    assert cert["schema"] == "hivemind-solver-cert/1"
    assert cert["verdict"] == "VERIFIED"
    assert {c["name"]: c["status"] for c in cert["checks"]} == {
        "shape": "pass", "hashes": "pass", "reference": "pass", "invariants": "skip", "tightening": "pass"}
    assert cert["binary_sha256"] == sandbox.file_sha256(os.path.realpath(BINARY))


def test_ad_hoc_spec_is_audited_not_verified() -> None:
    prob = json.loads((PROBLEMS / "linear_2x2.json").read_text(encoding="utf-8"))
    spec = dict(prob["spec"], id="adhoc.linear")
    svc = SolverBeeService(BINARY, problems_dir=str(PROBLEMS), timeout_s=60)
    code, rep = svc.solve(json.dumps(spec, separators=(",", ":")).encode("utf-8"))
    assert code == 200
    assert rep["verdict"] == "AUDITED"
