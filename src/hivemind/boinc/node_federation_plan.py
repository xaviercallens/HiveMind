# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Federation plan for one trainer node: which workunit class, which bee, and which steps are runnable now.

The plan is a pure function of a measured GPU reading. It never contacts anything and never reports a step as done:
each step is READY (a command the user can run), MANUAL (needs an operator action elsewhere) or BLOCKED (with the
reason). The "bee approach": a node trains one small specialist expert (a bee, a rank-4 LoRA) that fits its card, and the
router recruits experts per task cluster; only the expert's outer delta crosses the network (DiLoCo-style).
"""

from __future__ import annotations

import math
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
AMCP_LEARNER_SCRIPT: Final[str] = "scripts/e8_amcp_round.py"
NIGHT_WINDOW_PARIS: Final[str] = "22:00-07:00 Paris"
MIN_FREE_VRAM_MIB: Final[int] = 4300


@dataclass(frozen=True)
class PlanStep:
    name: str
    status: StepStatus
    command: str
    note: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if self.status not in ("READY", "MANUAL", "BLOCKED"):
            raise ValueError("status must be 'READY', 'MANUAL', or 'BLOCKED'")
        if not isinstance(self.command, str):
            raise ValueError("command must be a string")
        if not isinstance(self.note, str) or not self.note.strip():
            raise ValueError("note must be a non-empty string")


@dataclass(frozen=True)
class FederationPlan:
    gpu_name: str
    total_mib: float
    usable_mib: int
    plan_class: PlanClass
    bee: str | None
    bee_note: str
    steps: tuple[PlanStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.gpu_name, str) or not self.gpu_name.strip():
            raise ValueError("gpu_name must be a non-empty string")
        if isinstance(self.total_mib, bool) or not isinstance(self.total_mib, (int, float)) or not math.isfinite(float(self.total_mib)) or float(self.total_mib) < 0.0:
            raise ValueError("total_mib must be a finite non-negative number")
        if isinstance(self.usable_mib, bool) or not isinstance(self.usable_mib, int) or self.usable_mib < 0:
            raise ValueError("usable_mib must be a non-negative integer")
        if self.plan_class not in ("cuda_hivemind_big", "cuda_hivemind", "cpu_hivemind_solver"):
            raise ValueError("plan_class must be 'cuda_hivemind_big', 'cuda_hivemind', or 'cpu_hivemind_solver'")
        if self.bee is not None and (not isinstance(self.bee, str) or not self.bee.strip()):
            raise ValueError("bee must be None or a non-empty string")
        if not isinstance(self.bee_note, str) or not self.bee_note.strip():
            raise ValueError("bee_note must be a non-empty string")
        if not isinstance(self.steps, tuple) or len(self.steps) == 0 or not all(isinstance(s, PlanStep) for s in self.steps):
            raise ValueError("steps must be a non-empty tuple of PlanStep instances")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_plan_class(total_mib: float | None) -> PlanClass:
    """Coordinator plan class for a card; no GPU reading falls to the CPU solver class."""
    if total_mib is not None and (isinstance(total_mib, bool) or not isinstance(total_mib, (int, float)) or not math.isfinite(float(total_mib)) or float(total_mib) < 0.0):
        raise ValueError("total_mib must be None or a finite non-negative number")
    if total_mib is None or total_mib < CUDA_MIN_VRAM_MIB:
        return "cpu_hivemind_solver"
    return "cuda_hivemind_big" if total_mib >= CUDA_BIG_MIN_VRAM_MIB else "cuda_hivemind"


def pick_bee(usable_mib: int) -> ModelTier | None:
    """Largest specialist bee (`bee-*`, dense) whose recommended memory fits the usable VRAM, else None."""
    if isinstance(usable_mib, bool) or not isinstance(usable_mib, int) or usable_mib < 0:
        raise ValueError("usable_mib must be a non-negative integer")
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
    if gpu_name is not None and (not isinstance(gpu_name, str) or not gpu_name.strip()):
        raise ValueError("gpu_name must be None or a non-empty string")
    if not isinstance(coordinator, str) or not coordinator.strip().startswith(("http://", "https://")):
        raise ValueError("coordinator must be a non-empty http/https URL")
    if not isinstance(handle, str) or not handle.strip():
        raise ValueError("handle must be a non-empty string")
    if overlay_ip is not None and (not isinstance(overlay_ip, str) or not overlay_ip.strip()):
        raise ValueError("overlay_ip must be None or a non-empty string")
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
                              "AMCP learners claim only 22:00-07:00 Paris and never below 4,300 MiB free VRAM; "
                              "see docs/FEDERATION_V2_AMCP_AGLETS.md"))
    else:
        steps.append(PlanStep("cross_host_pilot", "READY" if measured else "BLOCKED",
                              f"python3 {AMCP_LEARNER_SCRIPT} learner --run-id night-003 "
                              f"--learner-id {handle} --index 1 --quant nf4 --device cuda --night-only",
                              f"overlay address {overlay_ip}; AMCP learner claims only {NIGHT_WINDOW_PARIS} "
                              f"and never below 4,300 MiB free VRAM"))
    return FederationPlan(gpu_name or "none", float(total_mib or 0.0), usable, plan_class,
                          bee.public_id if bee else None, bee_note, tuple(steps))
