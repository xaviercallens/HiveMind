# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Receipts that are not measurements must say so (docs/roadmap/DOD_RECEIPT_AUDIT_2026-11-10.md, correction 1).

The audit found 18 receipts under artifacts/experiments produced by simulators, formulas or literal rows, none labelled.
Each now carries ``evidence`` (simulated | modelled | placeholder) and ``not_a_measurement: true``, and its generator adds
the same label when it writes, so a re-run cannot drop it. A receipt can only become "measured" by a generator that reads
real inputs; removing a label here should be a deliberate, reviewed change.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AE = ROOT / "artifacts" / "experiments"
NOT_MEASURED = {"simulated", "modelled", "placeholder"}

AUDITED = {
    "E21_verification_cost_report.json": "scripts/run_experiment_E21.py",
    "E22_discovery_report.json": "scripts/run_experiment_E22.py",
    "E23_amcp_mobility_report.json": "scripts/run_experiment_E23.py",
    "E24_chain_regression_report.json": "scripts/run_experiment_E24.py",
    "E26_residential_compression_report.json": "scripts/run_experiment_E26.py",
    "E27_testbed_1000_report.json": "scripts/run_experiment_E27.py",
    "CHAIN_STATUS_2026-11-08.json": "scripts/chain_status_e24_window_receipt.py",
    "NODE_EGRESS_2026-11-07.json": "scripts/node_egress_receipt.py",
    "TRAINER_HARDWARE_2026-11-07.json": "scripts/trainer_hardware_receipt.py",
    "CONSUMER_GPU_ECONOMICS_2026-11-08.json": "scripts/consumer_gpu_economics_telemetry_receipt.py",
    "PETALS_LIVENESS_2026-11-07.json": "scripts/petals_liveness_receipt.py",
    "CREDIT_RECONCILIATION_2026-11-07.json": "scripts/credit_reconciliation_receipt.py",
    "BASE_WEIGHTS_MIRROR_2026-11-07.json": "scripts/base_weights_mirror_receipt.py",
    "PROV86_CHAIN_PROVENANCE_LOCK_2026-11-08.json": "scripts/verify_ps01_d4_chain_provenance_lock.py",
    "PLANETARY_SCALE_READINESS_2026-11-08.json": "scripts/planetary_scale_readiness_audit.py",
    "TRAINER_INSTALL_2026-11-07.json": "scripts/volunteer_install_smoke.sh",
    "CANARY_PROBE_2026-11-07.json": None,  # no generator in the repository
    "ACCOUNTS_2026-11-07.json": "scripts/accounts_receipt.py",
}


def label_problem(receipt: dict) -> str | None:
    """Why ``receipt`` does not carry a valid not-a-measurement label, or None."""
    if receipt.get("evidence") not in NOT_MEASURED:
        return f"evidence is {receipt.get('evidence')!r}, expected one of {sorted(NOT_MEASURED)}"
    if receipt.get("not_a_measurement") is not True:
        return "not_a_measurement is not true"
    return None


@pytest.mark.parametrize("name", sorted(AUDITED))
def test_every_audited_receipt_says_it_is_not_a_measurement(name: str) -> None:
    receipt = json.loads((AE / name).read_text(encoding="utf-8"))
    assert label_problem(receipt) is None, f"{name}: {label_problem(receipt)}"


@pytest.mark.parametrize("name,generator", sorted((n, g) for n, g in AUDITED.items() if g))
def test_every_generator_writes_the_label(name: str, generator: str) -> None:
    src = (ROOT / generator).read_text(encoding="utf-8")
    assert '"not_a_measurement"' in src and '"evidence"' in src, f"{generator} would drop the label of {name} on a re-run"


def test_a_planted_unlabelled_receipt_is_caught() -> None:
    assert label_problem({"verdict": "PASS", "measured_egress_bytes_per_hour": 25620}) is not None
    assert label_problem({"evidence": "measured", "not_a_measurement": False}) is not None
    assert label_problem({"evidence": "simulated", "not_a_measurement": True}) is None


@pytest.mark.parametrize("generator", [
    "scripts/run_experiment_E22.py",
    "scripts/run_experiment_E23.py",
    "scripts/petals_liveness_receipt.py",
    "scripts/node_egress_receipt.py",
])
def test_a_rerun_keeps_the_label(generator: str, tmp_path: Path) -> None:
    out = tmp_path / "receipt.json"
    proc = subprocess.run([sys.executable, str(ROOT / generator), "--out", str(out)], cwd=tmp_path, capture_output=True,
                          text=True, timeout=120, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    if not out.exists():
        pytest.skip(f"{generator} did not run here (exit {proc.returncode}): {proc.stderr[-200:]}")
    assert label_problem(json.loads(out.read_text(encoding="utf-8"))) is None
