# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Hardening tests for node_federation_plan, validator_assignment, ef_compression_benchmark, and federation_agents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hivemind.boinc.node_federation_plan import (
    FederationPlan,
    PlanStep,
    build_plan,
)
from hivemind.boinc.validator_assignment import (
    EpsilonTable,
    _validate_class_key,
)
from hivemind.swarm.ef_compression_benchmark import (
    SyntheticSwarmProblem,
    build_problem,
)
from hivemind.swarm.federation_agents import (
    FederationError,
    RoundBook,
)


def test_node_federation_plan_step_and_plan_validation() -> None:
    step = PlanStep("preflight", "READY", "python3 check.py", "checks hardware")
    assert step.status == "READY"

    with pytest.raises(ValueError, match="name must be a non-empty string"):
        PlanStep("   ", "READY", "cmd", "note")
    with pytest.raises(ValueError, match="status must be"):
        PlanStep("preflight", "INVALID", "cmd", "note")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="command must be a string"):
        PlanStep("preflight", "READY", 123, "note")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="note must be a non-empty string"):
        PlanStep("preflight", "READY", "cmd", "   ")

    plan = build_plan("RTX 2070", 8192.0, coordinator="https://example.org", handle="node1")
    assert plan.plan_class == "cuda_hivemind" and len(plan.steps) >= 4

    with pytest.raises(ValueError, match="gpu_name must be None or a non-empty string"):
        build_plan("   ", 8192.0, coordinator="https://example.org", handle="node1")

    with pytest.raises(ValueError, match="gpu_name must be a non-empty string"):
        FederationPlan("", 8192.0, 6656, "cuda_hivemind", "hivemind/bee-stars", "fits", (step,))
    with pytest.raises(ValueError, match="total_mib must be a finite non-negative"):
        FederationPlan("RTX", -1.0, 6656, "cuda_hivemind", "hivemind/bee-stars", "fits", (step,))
    with pytest.raises(ValueError, match="usable_mib must be a non-negative integer"):
        FederationPlan("RTX", 8192.0, -1, "cuda_hivemind", "hivemind/bee-stars", "fits", (step,))
    with pytest.raises(ValueError, match="plan_class must be"):
        FederationPlan("RTX", 8192.0, 6656, "bad_class", "hivemind/bee-stars", "fits", (step,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bee must be None or a non-empty string"):
        FederationPlan("RTX", 8192.0, 6656, "cuda_hivemind", "   ", "fits", (step,))
    with pytest.raises(ValueError, match="bee_note must be a non-empty string"):
        FederationPlan("RTX", 8192.0, 6656, "cuda_hivemind", None, "", (step,))
    with pytest.raises(ValueError, match="steps must be a non-empty tuple"):
        FederationPlan("RTX", 8192.0, 6656, "cuda_hivemind", None, "note", ())


def test_validator_assignment_class_key_and_epsilon_table_validation(tmp_path: Path) -> None:
    k_cuda = ("cuda", "7.5", "2.4", "float32")
    k_cpu = ("cpu", None, "2.4", "float32")
    assert _validate_class_key(k_cuda) == k_cuda

    for bad_key in (
        ("cuda", "7.5", "2.4"),
        ("bad_dev", "7.5", "2.4", "float32"),
        ("cuda", "", "2.4", "float32"),
        ("cuda", "7.5", "", "float32"),
        ("cuda", "7.5", "2.4", "   "),
    ):
        with pytest.raises(ValueError):
            _validate_class_key(bad_key)

    tbl = EpsilonTable()
    tbl.set_eps(k_cuda, k_cpu, 1e-3)
    assert tbl.eps_for(k_cuda, k_cpu) == 1e-3

    with pytest.raises(ValueError, match="must be a 4-tuple"):
        tbl.set_eps(("cuda",), k_cpu, 1e-3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a 4-tuple"):
        tbl.eps_for(k_cuda, ("cpu",))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="path must be a non-empty"):
        EpsilonTable.from_e12_report("   ")
    bad_json = tmp_path / "bad_e12.json"
    bad_json.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="E12 report must be a JSON object"):
        EpsilonTable.from_e12_report(bad_json)


def test_ef_compression_benchmark_synthetic_problem_validation() -> None:
    prob = build_problem(seed=1, num_workers=2, samples_per_worker=8, d_in=4, d_out=2)
    assert prob.w_true.shape == (2, 4)

    with pytest.raises(ValueError, match="w_true must be a non-empty 2D finite"):
        SyntheticSwarmProblem(torch.ones(4), prob.worker_x, prob.worker_y, prob.holdout_x, prob.holdout_y)
    with pytest.raises(ValueError, match="worker_x and worker_y must be non-empty lists"):
        SyntheticSwarmProblem(prob.w_true, [], [], prob.holdout_x, prob.holdout_y)
    with pytest.raises(ValueError, match="worker_x and worker_y tensors must be 2D finite"):
        SyntheticSwarmProblem(
            prob.w_true,
            [torch.ones(8, 3)],
            [prob.worker_y[0]],
            prob.holdout_x,
            prob.holdout_y,
        )
    with pytest.raises(ValueError, match="holdout_x and holdout_y must be non-empty 2D finite"):
        SyntheticSwarmProblem(
            prob.w_true,
            prob.worker_x,
            prob.worker_y,
            torch.ones(4, 3),
            prob.holdout_y,
        )


def test_federation_agents_round_book_validation() -> None:
    tmpl = {"w": torch.zeros(2, 2)}
    rb = RoundBook(
        run_id="run-01",
        rnd=1,
        allowed=["learner-a", "learner-b"],
        quorum=1,
        deadline_unix=1000.0,
        global_sha256="a" * 64,
        pin="b" * 64,
        max_delta_l2=5.0,
        template=tmpl,
    )
    assert rb.status(now=500.0) == "open"

    with pytest.raises(FederationError, match="allowed must be a non-empty sequence"):
        RoundBook("run-01", 1, "learner-a", 1, 1000.0, "a" * 64, "b" * 64, 5.0, tmpl)
    with pytest.raises(FederationError, match="allowed must be a non-empty sequence"):
        RoundBook("run-01", 1, ["   "], 1, 1000.0, "a" * 64, "b" * 64, 5.0, tmpl)
    with pytest.raises(FederationError, match="global_sha256 must be a non-empty string"):
        RoundBook("run-01", 1, ["l1"], 1, 1000.0, "   ", "b" * 64, 5.0, tmpl)
    with pytest.raises(FederationError, match="pin must be a non-empty string"):
        RoundBook("run-01", 1, ["l1"], 1, 1000.0, "a" * 64, "", 5.0, tmpl)
    with pytest.raises(FederationError, match="template must be a non-empty mapping"):
        RoundBook("run-01", 1, ["l1"], 1, 1000.0, "a" * 64, "b" * 64, 5.0, ["not_a_dict"])  # type: ignore[arg-type]

    with pytest.raises(FederationError, match="payload must be a Mapping"):
        rb.claim("not_a_mapping")  # type: ignore[arg-type]
    with pytest.raises(FederationError, match="payload must be a Mapping"):
        rb.offer_result("not_a_mapping")  # type: ignore[arg-type]
    for bad_now in (True, float("nan"), float("inf")):
        with pytest.raises(FederationError, match="now must be a finite timestamp"):
            rb.status(now=bad_now)  # type: ignore[arg-type]
