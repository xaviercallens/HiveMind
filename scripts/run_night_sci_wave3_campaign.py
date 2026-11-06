#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tonight Intense Training Orchestrator (`NIGHT-SCI-WAVE3-2026`, Phases 0-7) for the 5 Wave 3 Scientific MoE Bees.

Enforces `AGENTS.md` hardware & scientific governance:
  - Never modifies `/opt/hivemind/t4/`.
  - Requires `>= 4,300 MiB` free VRAM on the local NVIDIA Tesla T4 GPU before every GPU phase.
  - Verifies frozen SHA-256 dataset digests against `NIGHT_SCI_WAVE3_prereg.json`.
  - Runs S-3b slot-filling baseline (`prefer_slots=True`) before any GPU LoRA step.
  - Evaluates the 5x5 cross-domain matrix + exact two-sided McNemar test + Gates G1-G3 before promoting
    any bee from `candidate -> shadow` in `ExpertRegistry`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from hivemind.core.expert_registry import (
    SHADOW,
    WAVE3_PAIRED_MOE_BASE_MODEL_ID,
    WAVE3_PRIMARY_BASE_MODEL_ID,
    WAVE3_SCIENTIFIC_EXPERT_SPECS,
    ExpertRegistry,
    ServingNode,
    evaluate_wave3_promotion_gates,
    plan_adapter_residency,
    validate_residency_plan,
)
from hivemind.solverbee.families import compile_slot_spec
from hivemind.solverbee.science_bee import build_system_prompt

PROTECTED_T4_DIR = Path("/opt/hivemind/t4")
MIN_FREE_VRAM_MIB = 4300


class VRAMHeadroomError(RuntimeError):
    """Raised when free GPU VRAM on the local Tesla T4 falls below the 4,300 MiB floor."""

    error_code: str = "VRAM_HEADROOM_BELOW_FLOOR"


def query_free_vram_mib() -> int:
    """Query free GPU memory (MiB) via `nvidia-smi`."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    ).strip()
    first_line = out.splitlines()[0].strip()
    return int(first_line)


def check_vram_headroom(
    min_free_mib: int = MIN_FREE_VRAM_MIB,
    *,
    vram_probe: Optional[Callable[[], int]] = None,
) -> int:
    """Verify `>= min_free_mib` (default 4,300 MiB) free VRAM before starting a GPU phase."""
    probe = vram_probe or query_free_vram_mib
    free_mib = int(probe())
    if free_mib < int(min_free_mib):
        raise VRAMHeadroomError(
            f"VRAM headroom guard triggered: free VRAM is {free_mib} MiB, below required {min_free_mib} MiB floor"
        )
    return free_mib


def verify_preregistration_integrity(prereg_path: Path) -> Dict[str, Any]:
    """Verify that `NIGHT_SCI_WAVE3_prereg.json` exists and every split file matches its frozen SHA-256."""
    prereg = json.loads(Path(prereg_path).read_text(encoding="utf-8"))
    if prereg.get("schema") != "hivemind-night-sci-wave3-prereg/1":
        raise ValueError(f"Invalid pre-registration schema: {prereg.get('schema')!r}")
    if prereg.get("hardware_governance", {}).get("vram_free_floor_mib", 0) < MIN_FREE_VRAM_MIB:
        raise ValueError("Pre-registration vram_free_floor_mib must be >= 4300 MiB")

    bees = prereg.get("bees", {})
    if set(bees) != set(WAVE3_SCIENTIFIC_EXPERT_SPECS):
        raise ValueError(f"Pre-registration bee set mismatch: {sorted(bees)}")

    for bee_id, info in sorted(bees.items()):
        for split_name, split_meta in info["splits"].items():
            fpath = Path(split_meta["path"])
            if not fpath.exists():
                raise FileNotFoundError(f"Missing frozen split file for {bee_id}/{split_name}: {fpath}")
            actual_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
            if actual_sha != split_meta["sha256"]:
                raise ValueError(
                    f"SHA-256 mismatch for {bee_id}/{split_name}: expected {split_meta['sha256']}, got {actual_sha}"
                )
    return prereg


def exact_mcnemar_p_value(b_expert_only: int, c_baseline_only: int) -> float:
    """Two-sided exact binomial McNemar p-value for discordant pairs `(b, c)`."""
    b = int(b_expert_only)
    c = int(c_baseline_only)
    n = b + c
    if n <= 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * float(tail))


def evaluate_s3b_slot_baseline(prereg: Mapping[str, Any]) -> Dict[str, Any]:
    """Phase 1.5: Verify S-3b slot-filling (`prefer_slots=True`) across all Wave 3 slot-backed held-out splits."""
    prompt = build_system_prompt(prefer_slots=True)
    if "hivemind-solver-slots/1" not in prompt:
        raise RuntimeError("build_system_prompt(prefer_slots=True) missing hivemind-solver-slots/1 schema")

    slot_results: Dict[str, Any] = {}
    for bee_id, info in sorted(prereg["bees"].items()):
        test_path = Path(info["splits"]["held_out_test"]["path"])
        rows = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        compiled_ok = 0
        total_with_slots = 0
        for row in rows:
            if "slots" in row:
                total_with_slots += 1
                spec = compile_slot_spec(row["slots"])
                if spec.get("schema") == "hivemind-solver-spec/1":
                    compiled_ok += 1
        slot_results[bee_id] = {
            "total_held_out": len(rows),
            "slot_examples": total_with_slots,
            "slot_compiled_ok": compiled_ok,
            "slot_compile_rate": (compiled_ok / total_with_slots) if total_with_slots > 0 else 1.0,
        }
    return slot_results


def evaluate_5x5_cross_domain_matrix(
    metrics_by_bee: Mapping[str, Mapping[str, Any]],
    *,
    base_model_id: str = WAVE3_PRIMARY_BASE_MODEL_ID,
    adapter_hashes: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Phase 7: Evaluate 5x5 specialization matrix, check Gates G1-G3, and promote passing bees to `shadow`.

    Discordant McNemar counts (`discordant_expert_wins`, `discordant_base_wins`) are mandatory per bee;
    missing counts refuse promotion (IP-245). Only bees with a real content `adapter_sha256` (via
    `adapter_hashes` or `metrics_by_bee[bee_id]["adapter_sha256"]`) are registered and promoted with a
    `gate_report_id` recorded in their transition history.
    """
    reg = ExpertRegistry()
    explicit_hashes: Dict[str, str] = dict(adapter_hashes or {})
    for bee_id in sorted(WAVE3_SCIENTIFIC_EXPERT_SPECS):
        m = metrics_by_bee.get(bee_id) or {}
        if bee_id not in explicit_hashes and isinstance(m.get("adapter_sha256"), str):
            explicit_hashes[bee_id] = str(m["adapter_sha256"])

    if len(explicit_hashes) == len(WAVE3_SCIENTIFIC_EXPERT_SPECS):
        reg.register_wave3_scientific_experts(
            base_model_id=base_model_id,
            adapter_hashes=explicit_hashes,
        )

    gate_reports: Dict[str, Any] = {}
    promoted: List[str] = []
    for bee_id in sorted(WAVE3_SCIENTIFIC_EXPERT_SPECS):
        m = metrics_by_bee.get(bee_id)
        if not isinstance(m, Mapping):
            gate_reports[bee_id] = {
                "passed": False,
                "gate_report_id": f"gate-{bee_id}-missing-metrics",
                "reasons": [f"missing evaluation metrics for {bee_id}"],
            }
            continue
        if "discordant_expert_wins" not in m or "discordant_base_wins" not in m:
            gate_reports[bee_id] = {
                "passed": False,
                "gate_report_id": f"gate-{bee_id}-missing-mcnemar-counts",
                "reasons": [
                    "missing McNemar discordant counts (discordant_expert_wins and discordant_base_wins are required)"
                ],
            }
            continue
        if any(k not in m for k in ("in_domain_acc", "base_acc", "max_off_diagonal_acc")):
            gate_reports[bee_id] = {
                "passed": False,
                "gate_report_id": f"gate-{bee_id}-missing-accuracy-fields",
                "reasons": ["missing required accuracy fields (in_domain_acc, base_acc, max_off_diagonal_acc)"],
            }
            continue
        b_disc = int(m["discordant_expert_wins"])
        c_disc = int(m["discordant_base_wins"])
        if b_disc < 0 or c_disc < 0:
            gate_reports[bee_id] = {
                "passed": False,
                "gate_report_id": f"gate-{bee_id}-negative-mcnemar-counts",
                "reasons": ["discordant McNemar counts must be >= 0"],
            }
            continue
        p_val = float(m.get("mcnemar_p_value", exact_mcnemar_p_value(b_disc, c_disc)))
        report = evaluate_wave3_promotion_gates(
            in_domain_acc=float(m["in_domain_acc"]),
            base_acc=float(m["base_acc"]),
            cross_domain_acc=float(m["max_off_diagonal_acc"]),
            mcnemar_p_value=p_val,
            wrong_verified_count=int(m.get("wrong_verified_count", 0)),
        )
        sha_tag = explicit_hashes.get(bee_id, "unregistered")[:12]
        gate_report_id = str(m.get("gate_report_id") or f"gate-{bee_id}-{sha_tag}")
        report["gate_report_id"] = gate_report_id
        gate_reports[bee_id] = report
        if report["passed"] and bee_id in reg:
            reg.transition(
                bee_id,
                SHADOW,
                (
                    f"NIGHT-SCI-WAVE3 gate_report_id={gate_report_id}: Gates G1-G3 passed "
                    f"(lift=+{report['in_domain_margin']:.2%}, p={p_val:.4g})"
                ),
            )
            promoted.append(bee_id)

    node = ServingNode(
        node_id="socreateai-agora-hermes-node1",
        vram_mib=15360 - MIN_FREE_VRAM_MIB,
        base_model_id=base_model_id,
        base_resident_mib=3716,
        max_lora_slots=8,
    )
    footprints = reg.residency_inputs(states=(SHADOW,))
    demand = {eid: 10.0 for eid in promoted}
    residency = plan_adapter_residency([node], footprints, demand)
    violations = validate_residency_plan(residency, [node], footprints)

    return {
        "base_model_id": base_model_id,
        "promoted_to_shadow": promoted,
        "gate_reports": gate_reports,
        "residency_plan": residency.to_dict(),
        "residency_violations": violations,
        "registry": reg.to_dict(),
    }


def run_wave3_campaign(
    prereg_path: Path,
    *,
    base_model_id: str = WAVE3_PRIMARY_BASE_MODEL_ID,
    vram_probe: Optional[Callable[[], int]] = None,
    metrics_by_bee: Optional[Mapping[str, Mapping[str, Any]]] = None,
    adapter_hashes: Optional[Mapping[str, str]] = None,
    eval_report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the Wave 3 pre-flight & campaign verification pipeline and write the campaign report.

    Does not fabricate simulation metrics (IP-220): if neither ``metrics_by_bee`` nor an existing
    ``eval_report_path`` is supplied, the campaign status is ``BLOCKED`` and zero bees are promoted.
    """
    prereg = verify_preregistration_integrity(prereg_path)
    free_vram_mib = check_vram_headroom(MIN_FREE_VRAM_MIB, vram_probe=vram_probe)
    s3b_baseline = evaluate_s3b_slot_baseline(prereg)
    out_dir = Path(prereg_path).parent
    registry_path = out_dir / "wave3_expert_registry.json"

    resolved_metrics = metrics_by_bee
    resolved_hashes = adapter_hashes
    if resolved_metrics is None and eval_report_path is not None and Path(eval_report_path).exists():
        loaded_eval = json.loads(Path(eval_report_path).read_text(encoding="utf-8"))
        if isinstance(loaded_eval.get("metrics_by_bee"), Mapping):
            resolved_metrics = loaded_eval["metrics_by_bee"]
        if resolved_hashes is None and isinstance(loaded_eval.get("adapter_hashes"), Mapping):
            resolved_hashes = loaded_eval["adapter_hashes"]

    if resolved_metrics is None:
        empty_reg = ExpertRegistry()
        empty_reg.save(registry_path)
        eval_summary: Dict[str, Any] = {
            "status": "BLOCKED",
            "blocked_reason": "No measured 5x5 evaluation metrics supplied (pass metrics_by_bee or eval_report_path).",
            "base_model_id": base_model_id,
            "promoted_to_shadow": [],
            "gate_reports": {},
            "residency_plan": {"hot": {}, "cold": []},
            "residency_violations": [],
            "registry": empty_reg.to_dict(),
        }
        status = "BLOCKED"
    else:
        eval_summary = evaluate_5x5_cross_domain_matrix(
            resolved_metrics,
            base_model_id=base_model_id,
            adapter_hashes=resolved_hashes,
        )
        eval_summary["status"] = "EVALUATED"
        ExpertRegistry.from_dict(eval_summary["registry"]).save(registry_path)
        status = "COMPLETED" if eval_summary["promoted_to_shadow"] else "GATES_BLOCKED"

    report = {
        "campaign_id": prereg["campaign_id"],
        "status": status,
        "primary_base_model_id": WAVE3_PRIMARY_BASE_MODEL_ID,
        "paired_moe_base_model_id": WAVE3_PAIRED_MOE_BASE_MODEL_ID,
        "active_base_model_id": base_model_id,
        "free_vram_mib": free_vram_mib,
        "vram_headroom_floor_mib": MIN_FREE_VRAM_MIB,
        "s3b_slot_baseline": s3b_baseline,
        "evaluation": eval_summary,
        "registry_path": str(registry_path),
    }
    report_path = out_dir / "NIGHT_SCI_WAVE3_campaign_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or verify Tonight's Wave 3 Scientific MoE Bee Campaign.")
    parser.add_argument(
        "--prereg",
        type=Path,
        default=Path("artifacts/night_sci_wave3/NIGHT_SCI_WAVE3_prereg.json"),
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=WAVE3_PRIMARY_BASE_MODEL_ID,
        choices=[WAVE3_PRIMARY_BASE_MODEL_ID, WAVE3_PAIRED_MOE_BASE_MODEL_ID],
    )
    parser.add_argument("--dry-run", action="store_true", help="Verify pre-reg, VRAM guard, S-3b baseline, and S21 gates.")
    args = parser.parse_args()

    report = run_wave3_campaign(args.prereg, base_model_id=args.base_model)
    print(
        f"[NIGHT-SCI-WAVE3] Campaign check complete: free_vram={report['free_vram_mib']} MiB "
        f"(floor={report['vram_headroom_floor_mib']} MiB), promoted={report['evaluation']['promoted_to_shadow']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
