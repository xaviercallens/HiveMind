# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""PS-02-D1: the T4 trainers refuse to run with a free-VRAM floor below the AGENTS.md 4,300 MiB rule.

The probe queue already refuses (scripts/night_probe_queue.py, PR #761). The governor (which starts BOINC training on the
T4) and the E8 learner accepted any --min-free-mib; the live governor once ran with 3,800 (deploy/t4_node/live/README.md).
Volunteer GPUs keep the floor the producer kit sets for them: the rule names the Tesla T4.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import night_probe_queue  # noqa: E402
import t4_governor  # noqa: E402


def test_all_t4_guards_agree_on_the_agents_md_floor() -> None:
    assert t4_governor.T4_FREE_FLOOR_MIB == night_probe_queue.VRAM_FLOOR_MIB == 4300
    e8_src = (SCRIPTS / "e8_qlora_moe_round.py").read_text(encoding="utf-8")
    assert "T4_FREE_FLOOR_MIB = 4300" in e8_src
    unit = (Path(__file__).resolve().parents[1] / "deploy/t4_node/node/hivemind-t4-governor.service").read_text(encoding="utf-8")
    assert "--min-free-mib 4300" in unit


@pytest.mark.parametrize("floor,name,refused", [
    (4300, "Tesla T4", False),
    (5000, "Tesla T4", False),
    (3800, "Tesla T4", True),        # the 2026 incident value
    (3800, None, True),              # unidentified GPU: fail closed
    (3800, "NVIDIA GeForce RTX 2070", False),  # a volunteer card keeps its own floor
    (2048, "NVIDIA GeForce RTX 3060", False),
    (0, "tesla t4", True),
])
def test_t4_floor_refusal(floor: float, name: str | None, refused: bool) -> None:
    assert (t4_governor.t4_floor_refusal(floor, name) is not None) is refused


def test_a_name_that_merely_contains_t4_letters_is_not_a_t4() -> None:
    assert t4_governor.t4_floor_refusal(3000, "NVIDIA RTX A4000") is None
    assert t4_governor.t4_floor_refusal(3000, "GT40") is None


def test_governor_refuses_to_start_on_a_t4_with_a_lowered_floor(tmp_path, monkeypatch) -> None:
    modes: list[str] = []
    monkeypatch.setattr(t4_governor, "gpu_name", lambda: "Tesla T4")
    monkeypatch.setattr(t4_governor.Boinc, "set_gpu", lambda self, mode: modes.append(mode) or True)
    telemetry = tmp_path / "gov.jsonl"
    monkeypatch.setattr(sys, "argv", ["t4_governor.py", "--min-free-mib", "3800", "--telemetry", str(telemetry), "--dry-run",
                                      "--hold-file", str(tmp_path / "HOLD")])
    assert t4_governor.main() == 2
    assert modes == ["never"], "training must be switched off before the governor exits"
    ev = json.loads(telemetry.read_text().splitlines()[-1])
    assert ev["event"] == "refused" and ev["min_free_mib"] == 3800


def test_governor_does_not_refuse_the_agents_md_default(monkeypatch) -> None:
    assert t4_governor.t4_floor_refusal(4300, "Tesla T4") is None


# ---- E8 learner ---------------------------------------------------------------------------------------------------------

def _e8():
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    import e8_qlora_moe_round as q

    return q


def test_e8_learner_refuses_a_lowered_floor_on_a_t4() -> None:
    q = _e8()
    with pytest.raises(q.GuardError, match="4300"):
        q.vram_guard(3800, 600, "cuda", lambda: 15_000.0, lambda: "Tesla T4")
    with pytest.raises(q.GuardError):
        q.vram_guard(3800, 600, "cuda", lambda: 15_000.0, lambda: None)


def test_e8_learner_keeps_volunteer_floors_and_cpu_runs() -> None:
    q = _e8()
    assert q.vram_guard(3000, 600, "cuda", lambda: 8_000.0, lambda: "NVIDIA GeForce RTX 2070") == 8_000.0
    assert q.vram_guard(0, 600, "cpu", lambda: None, lambda: None) is None
    assert q.vram_guard(4300, 600, "cuda", lambda: 10_000.0, lambda: "Tesla T4") == 10_000.0
