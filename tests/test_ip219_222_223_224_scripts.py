# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression tests for PR-IP-30 script honesty and dry-run/opt-in safety (IP-219, IP-222, IP-223, IP-224)."""

from __future__ import annotations

import datetime as dt
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_ip219_monitor_nocturnal_night_run_empty_artifact_at_noon_never_fabricates_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IP-219 negative & positive controls:
    - With an empty/missing CODING_REPORT_PATH and now=12:00, no stage is COMPLETED,
      no stage has fast_ratchet_verified=True, and t4_vram_guard_passed is not True.
    - With a verifier-attested artifact and a live VRAM probe, stages and VRAM guard
      reflect the artifact and probe values.
    """
    import monitor_nocturnal_night_run as mon

    empty_coding = tmp_path / "empty_coding_report.json"
    empty_coding.write_text("", encoding="utf-8")
    out_status = tmp_path / "eta_status.json"

    monkeypatch.setattr(mon, "CODING_REPORT_PATH", empty_coding)
    monkeypatch.setattr(mon, "ETA_STATUS_OUTPUT_PATH", out_status)
    monkeypatch.setattr(mon, "probe_t4_free_vram_mib", lambda: getattr(mon, "_UNSET_PROBE", None))

    noon_paris = dt.datetime(2026, 11, 3, 12, 0, 0, tzinfo=ZoneInfo("Europe/Paris"))
    payload = mon.compute_night_run_eta_status(noon_paris)

    assert payload["overall_night_run_status"] != "COMPLETED"
    assert payload["t4_free_vram_mib"] is None
    assert payload["t4_vram_guard_passed"] is not True
    for stage in payload["stages"]:
        assert stage["status"] != "COMPLETED"
        assert stage.get("fast_ratchet_verified") is not True
        for key, val in stage.items():
            if key.startswith("verified_"):
                assert val is None

    # Positive control: verifier-attested report + explicit VRAM probe
    verified_coding = tmp_path / "verified_coding_report.json"
    verified_coding.write_text(
        json.dumps(
            {
                "verifier_attested": True,
                "inner_diloco_steps": 500,
                "big_gpu_nodes_count": 8,
                "specialist_bees_count": 16,
                "stage5_upcycled_moe_loss": 0.008,
                "verified_upcycled_loss": 0.008,
                "verified_routing_acc_pct": 99.5,
                "overall_gate_passed": True,
                "quality_loop_iterations": [
                    {
                        "loop_index": 1,
                        "inner_steps_executed": 500,
                        "fast_ratchet_verified": True,
                        "verified_hard_eval_loss": 1.85,
                        "verified_hard_top1_acc_pct": 81.2,
                        "verified_top2_expert_acc_pct": 99.0,
                    },
                    {
                        "loop_index": 2,
                        "inner_steps_executed": 250,
                        "fast_ratchet_verified": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    pos = mon.compute_night_run_eta_status(
        noon_paris,
        coding_report_path=verified_coding,
        vision_report_path=tmp_path / "missing_vision.json",
        vram_probe=9500.0,
        output_path=out_status,
    )
    assert pos["t4_free_vram_mib"] == 9500.0
    assert pos["t4_vram_guard_passed"] is True
    stages_by_id = {s["stage_id"]: s for s in pos["stages"]}
    assert stages_by_id["tome_1_foundation_btm_diloco"]["status"] == "COMPLETED"
    assert stages_by_id["tome_1_foundation_btm_diloco"]["fast_ratchet_verified"] is True
    assert stages_by_id["tome_1_foundation_btm_diloco"]["verified_hard_eval_loss"] == 1.85
    assert stages_by_id["tome_2_hard_mining_nesterov_muon"]["status"] == "IN_PROGRESS"
    assert stages_by_id["tome_2_hard_mining_nesterov_muon"]["progress_pct"] == 50.0
    assert stages_by_id["tome_2_hard_mining_nesterov_muon"]["fast_ratchet_verified"] is False
    assert stages_by_id["tome_3_verifier_distillation_router"]["status"] == "NO_ARTIFACT"


def test_ip222_stress_benchmark_soundness_fields_computed_from_runs() -> None:
    """IP-222 negative & positive controls:
    - When one domain has gate_passed=False or negative VRAM headroom,
      mcnemar_gate_success_rate < 1.0 and vram_envelope_breaches > 0.
    - When all domains pass and idempotent replay grants 0 HMT, soundness is 1.0 / True / 0.
    """
    from run_nocturnal_20_moe_stress_benchmarks import compute_security_and_soundness_summary

    runs = [
        SimpleNamespace(gate_passed=True, peak_vram_mb=5200.0),
        SimpleNamespace(gate_passed=True, peak_vram_mb=5300.0),
        SimpleNamespace(gate_passed=False, peak_vram_mb=9000.0),
        SimpleNamespace(gate_passed=True, peak_vram_mb=5100.0),
    ]
    domain_telemetry: dict[str, dict[str, Any]] = {
        "domain_a": {"gate_passed_rate": 1.0, "vram_headroom_mb": 2048.0},
        "domain_b": {"gate_passed_rate": 0.5, "vram_headroom_mb": -512.0},
    }
    bad_replay = SimpleNamespace(total_hmt_granted=10.0, already_credited_count=2)

    summary_fail = compute_security_and_soundness_summary(
        runs,
        domain_telemetry,
        ledger_intact=True,
        idempotent_report=bad_replay,
    )
    assert summary_fail["mcnemar_gate_success_rate"] < 1.0
    assert summary_fail["mcnemar_gate_success_rate"] == 0.75
    assert summary_fail["vram_envelope_breaches"] == 1
    assert summary_fail["idempotency_audit_passed"] is False
    assert summary_fail["byzantine_replay_blocked"] is False

    all_pass_runs = [SimpleNamespace(gate_passed=True, peak_vram_mb=5000.0) for _ in range(4)]
    all_pass_domains = {
        "domain_a": {"gate_passed_rate": 1.0, "vram_headroom_mb": 2048.0},
        "domain_b": {"gate_passed_rate": 1.0, "vram_headroom_mb": 1024.0},
    }
    good_replay = SimpleNamespace(total_hmt_granted=0.0, already_credited_count=4)
    summary_ok = compute_security_and_soundness_summary(
        all_pass_runs,
        all_pass_domains,
        ledger_intact=True,
        idempotent_report=good_replay,
    )
    assert summary_ok["mcnemar_gate_success_rate"] == 1.0
    assert summary_ok["vram_envelope_breaches"] == 0
    assert summary_ok["idempotency_audit_passed"] is True
    assert summary_ok["byzantine_replay_blocked"] is True
    assert summary_ok["ledger_hash_chain_verified"] is True


def test_ip223_amcp_kafka_real_proof_dry_run_prints_plan_and_calls_no_gcloud(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """IP-223 negative control: --dry-run succeeds, prints the execution plan, and invokes no gcloud."""
    import amcp_kafka_real_proof

    gcloud_calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        amcp_kafka_real_proof,
        "gcloud",
        lambda *args, **kwargs: gcloud_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        amcp_kafka_real_proof. subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess.run must not be called in --dry-run")),
    )

    rc = amcp_kafka_real_proof.main(
        [
            "--project",
            "hm-test-proj",
            "--voters",
            "hm-kafka-1:us-east4-a",
            "hm-kafka-2:us-east4-b",
            "hm-kafka-3:us-east4-c",
            "--count",
            "7",
            "--dry-run",
        ]
    )
    assert rc == 0
    assert gcloud_calls == []
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True
    assert out["project"] == "hm-test-proj"
    assert out["count"] == 7
    assert len(out["voters"]) == 3
    assert len(out["phases"]) == 5


def test_ip224_t4_monitor_does_not_restart_inactive_unit_unless_restart_flag_passed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IP-224 negative & positive controls:
    - Stubbed inactive unit with --once issues no `systemctl restart` unless `--restart` is passed.
    - Passing `--restart` (when euid==0 and not `--dry-run`) restarts the enabled inactive unit.
    - Passing `--restart --dry-run` records `unit_restart_dry_run` without issuing `systemctl restart`.
    """
    import t4_monitor

    commands_issued: list[list[str]] = []

    def recorded_sh(cmd: list[str], timeout: int = 10) -> str:
        commands_issued.append(list(cmd))
        if cmd[:2] == ["systemctl", "list-unit-files"]:
            return "nginx.service enabled\nollama.service enabled\n"
        if cmd[:2] == ["systemctl", "is-active"]:
            unit = cmd[2]
            return "inactive" if unit == "nginx" else "active"
        if cmd[:2] == ["systemctl", "is-enabled"]:
            return "enabled"
        if cmd[:2] == ["systemctl", "show"]:
            return "0"
        if cmd[0] == "nvidia-smi":
            if "--query-gpu=" in cmd[1]:
                return "1200, 15360, 48, 5, 28.5"
            return ""
        return ""

    monkeypatch.setattr(t4_monitor, "sh", recorded_sh)
    monkeypatch.setattr(t4_monitor, "http", lambda url, headers=None, timeout=4: (int(200), json.dumps({"data": []})))
    monkeypatch.setattr(t4_monitor, "metadata", lambda path: str("NONE").strip())
    monkeypatch.setattr(t4_monitor.os, "geteuid", lambda: int(0))

    out_default = tmp_path / "tele_default"
    rc1 = t4_monitor.main(["--out-dir", str(out_default), "--once"])
    assert rc1 == 0
    restart_cmds_default = [c for c in commands_issued if c[:2] == ["systemctl", "restart"]]
    assert restart_cmds_default == []

    # With --restart --dry-run: still no systemctl restart, but unit_restart_dry_run event logged
    commands_issued.clear()
    out_dry = tmp_path / "tele_dry"
    rc_dry = t4_monitor.main(["--out-dir", str(out_dry), "--restart", "--dry-run", "--once"])
    assert rc_dry == 0
    assert [c for c in commands_issued if c[:2] == ["systemctl", "restart"]] == []
    events_dry = [json.loads(line) for line in (out_dry / "events.jsonl").read_text().splitlines() if line.strip()]
    assert any(e.get("event") == "unit_restart_dry_run" and e.get("unit") == "nginx" for e in events_dry)

    # Positive control: with explicit --restart, nginx is restarted
    commands_issued.clear()
    out_optin = tmp_path / "tele_optin"
    rc2 = t4_monitor.main(["--out-dir", str(out_optin), "--restart", "--once"])
    assert rc2 == 0
    restart_cmds_optin = [c for c in commands_issued if c[:2] == ["systemctl", "restart"]]
    assert restart_cmds_optin == [["systemctl", "restart", "nginx"]]
