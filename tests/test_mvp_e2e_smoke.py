# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Real-model MVP smoke test (scripts/mvp_e2e_smoke.py). Runs only when HIVEMIND_MVP_GGUF points at a GGUF file and a
llama-server binary is available (HIVEMIND_LLAMA_SERVER or on PATH); otherwise it is skipped, so CI without a model is unaffected."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mvp_e2e_smoke as smoke  # noqa: E402

GGUF = os.environ.get("HIVEMIND_MVP_GGUF", "")
LLAMA = os.environ.get("HIVEMIND_LLAMA_SERVER") or shutil.which("llama-server") or ""


def test_the_smoke_script_helpers_are_importable_without_a_model() -> None:
    assert smoke.free_port() > 0
    assert smoke.wait_port(smoke.free_port(), 0.2) is False  # nothing listens there


@pytest.mark.skipif(not (GGUF and LLAMA and Path(GGUF).exists()), reason="set HIVEMIND_MVP_GGUF (and HIVEMIND_LLAMA_SERVER) to run the real-model smoke test")
def test_real_model_end_to_end_through_shield_gateway_and_ledger(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    code = smoke.main(["--gguf", GGUF, "--llama-server", LLAMA, "--report", str(report)])
    assert code == 0, report.read_text(encoding="utf-8") if report.exists() else "no report"


def test_pytorch_tiny_lm_end_to_end_through_shield_gateway_and_ledger_ci(tmp_path: Path) -> None:
    """Audit C11: run all 15 E2E smoke checks in CI using a CPU PyTorch causal Transformer LM."""
    import json

    report = tmp_path / "ci_smoke_report.json"
    code = smoke.main(["--pytorch-tiny-lm", "--report", str(report)])
    assert code == 0, report.read_text(encoding="utf-8") if report.exists() else "no report"
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert len(data["checks"]) == 15
    assert all(data["checks"].values()), data["checks"]
    assert data["ledger"]["consumer_spent_hmt"] > data["ledger"]["producer_earned_hmt"] > 0.0
