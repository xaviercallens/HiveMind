# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Federation plan for one trainer node: which workunit class, which bee, and which steps are runnable now.

The plan is a pure function of a measured GPU reading. It never contacts anything and never reports a step as done:
each step is READY (a command the user can run), MANUAL (needs an operator action elsewhere) or BLOCKED (with the
reason). The "bee approach": a node trains one small specialist expert (a bee, a rank-4 LoRA) that fits its card, and the
router recruits experts per task cluster; only the expert's outer delta crosses the network (DiLoCo-style).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final, Literal

from hivemind.router.catalog import ModelTier, tiers_for_vram

StepStatus = Literal["READY", "MANUAL", "BLOCKED"]
PlanClass = Literal["cuda_hivemind_big", "cuda_hivemind", "cpu_hivemind_solver"]

# Plan-class floors published by the coordinator (/boinc/status).
CUDA_MIN_VRAM_MIB: Final[int] = 3500
CUDA_BIG_MIN_VRAM_MIB: Final[int] = 12000
# Usable fraction of total VRAM for a desktop (WDDM) card: 6656 MiB of 8192 MiB, the owner-fleet cap for the RTX 2070.
DESKTOP_USABLE_FRACTION: Final[float] = 6656 / 8192
OVERLAY_SYNCER: Final[str] = "http://10.77.0.4:8765"


@dataclass(frozen=True)
class PlanStep:
    name: str
    status: StepStatus
    command: str
    note: str


@dataclass(frozen=True)
class FederationPlan:
    gpu_name: str
    total_mib: float
    usable_mib: int
    plan_class: PlanClass
    bee: str | None
    bee_note: str
    steps: tuple[PlanStep, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_plan_class(total_mib: float | None) -> PlanClass:
    """Coordinator plan class for a card; no GPU reading falls to the CPU solver class."""
    if total_mib is None or total_mib < CUDA_MIN_VRAM_MIB:
        return "cpu_hivemind_solver"
    return "cuda_hivemind_big" if total_mib >= CUDA_BIG_MIN_VRAM_MIB else "cuda_hivemind"


def pick_bee(usable_mib: int) -> ModelTier | None:
    """Largest specialist bee (`bee-*`, dense) whose recommended memory fits the usable VRAM, else None."""
    fitting = [t for t in tiers_for_vram(usable_mib / 1024.0) if t.tier_id.startswith("bee-") and not t.moe]
    return max(fitting, key=lambda t: t.min_vram_gb) if fitting else None


def build_plan(
    gpu_name: str | None,
    total_mib: float | None,
    *,
    coordinator: str,
    handle: str,
    overlay_ip: str | None = None,
) -> FederationPlan:
    """Build the plan from a measured reading (`None` for both when no GPU could be read)."""
    plan_class = classify_plan_class(total_mib)
    measured = total_mib is not None
    usable = int(total_mib * DESKTOP_USABLE_FRACTION) if measured and total_mib is not None else 0
    bee = pick_bee(usable) if plan_class != "cpu_hivemind_solver" else None
    if bee is not None:
        bee_note = f"{bee.public_id}: needs {bee.min_vram_gb} GB, fits the {usable} MiB budget; rank-4 LoRA, outer delta only"
    elif plan_class == "cpu_hivemind_solver":
        bee_note = "no CUDA card read: CPU solver work only, no bee is trained"
    else:
        bee_note = f"no specialist bee fits {usable} MiB"
    steps: list[PlanStep] = [
        PlanStep("preflight", "READY",
                 f"python3 scripts/hivemind_volunteer_preflight.py --project-url {coordinator}",
                 "read-only hardware and coordinator check"),
        PlanStep("register", "READY",
                 f"curl -s {coordinator.replace('hivemind-at-home-ruche', 'hivemind-openrouter')}/v1/hivemind/register "
                 f"-H 'Content-Type: application/json' -d '{{\"handle\":\"{handle}\",\"role\":\"trainer\"}}'",
                 "the API key is shown once; the handle is public"),
        PlanStep("idle_gate_dry_run", "READY" if measured else "BLOCKED",
                 "python3 scripts/hivemind_boinc_idle_worker.py --dry-run",
                 "checks the gate and a CUDA self-check on this GPU" if measured else "no GPU could be read"),
        PlanStep("qlora_training", "BLOCKED", "", "no trainer is wired into the idle worker (reason no_trainer_wired): "
                 "no workunit is leased and no points are earned by the worker today"),
    ]
    if overlay_ip is None:
        steps.append(PlanStep("cross_host_pilot", "MANUAL", "",
                              "needs the operator to open UDP 51820 on the T4 and add this node's WireGuard key; "
                              "see docs/E8_TWO_HOST_REAL_TEST.md"))
    else:
        steps.append(PlanStep("cross_host_pilot", "READY" if measured else "BLOCKED",
                              f"python3 scripts/run_experiment_E8_pilot.py learner --syncer {OVERLAY_SYNCER} "
                              f"--learner-id 1 --host-label {handle}",
                              f"overlay address {overlay_ip}; the T4 syncer must be running"))
    return FederationPlan(gpu_name or "none", float(total_mib or 0.0), usable, plan_class,
                          bee.public_id if bee else None, bee_note, tuple(steps))
