# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The federation plan must follow the measured card and never turn an unmeasured or unwired step into READY."""

from __future__ import annotations

from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from hivemind.boinc.node_federation_plan import (
    AMCP_LEARNER_SCRIPT,
    MIN_FREE_VRAM_MIB,
    NIGHT_WINDOW_PARIS,
    build_plan,
    classify_plan_class,
    pick_bee,
)

COORD = "https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _plan(name: str | None, mib: float | None, overlay: str | None = None):
    return build_plan(name, mib, coordinator=COORD, handle="node-x", overlay_ip=overlay)


def _step(plan, name):
    return next(s for s in plan.steps if s.name == name)


def test_rtx2070_gets_cuda_class_and_a_bee_inside_the_wddm_cap() -> None:
    plan = _plan("NVIDIA GeForce RTX 2070", 8192.0)
    assert plan.plan_class == "cuda_hivemind"
    assert plan.usable_mib == 6656
    assert plan.bee == "hivemind/bee-rusty-sundials"
    assert _step(plan, "idle_gate_dry_run").status == "READY"


def test_t4_gets_cuda_class_not_big() -> None:
    assert classify_plan_class(15360.0) == "cuda_hivemind_big"
    assert classify_plan_class(11999.0) == "cuda_hivemind"


def test_unread_gpu_is_blocked_with_no_bee() -> None:
    plan = _plan(None, None)
    assert plan.plan_class == "cpu_hivemind_solver" and plan.bee is None
    assert _step(plan, "idle_gate_dry_run").status == "BLOCKED"
    assert _step(plan, "cross_host_pilot").status == "MANUAL"


def test_training_is_always_blocked_and_never_claims_points() -> None:
    for mib in (None, 4096.0, 8192.0, 24576.0):
        step = _step(_plan("gpu", mib), "qlora_training")
        assert step.status == "BLOCKED" and step.command == ""


def test_overlay_enables_the_pilot_command_only_with_a_gpu() -> None:
    manual_with_gpu = _step(_plan("NVIDIA GeForce RTX 2070", 8192.0, None), "cross_host_pilot")
    assert manual_with_gpu.status == "MANUAL"
    assert manual_with_gpu.command == ""
    assert NIGHT_WINDOW_PARIS in manual_with_gpu.note
    assert "4,300 MiB" in manual_with_gpu.note and MIN_FREE_VRAM_MIB == 4300

    ready_step = _step(_plan("gpu", 8192.0, "10.77.0.5"), "cross_host_pilot")
    assert ready_step.status == "READY"
    assert ready_step.command == (
        "python3 scripts/e8_amcp_round.py learner --run-id night-003 "
        "--learner-id node-x --index 1 --quant nf4 --device cuda --night-only"
    )
    assert "10.77.0.5" in ready_step.note
    assert NIGHT_WINDOW_PARIS in ready_step.note and "4,300 MiB" in ready_step.note
    assert (REPO_ROOT / AMCP_LEARNER_SCRIPT).is_file()

    assert _step(_plan(None, None, "10.77.0.5"), "cross_host_pilot").status == "BLOCKED"


@given(st.integers(min_value=0, max_value=6000))
def test_bee_never_exceeds_usable_memory(usable_mib: int) -> None:
    bee = pick_bee(usable_mib)
    assert bee is None or bee.min_vram_gb * 1024 <= usable_mib
