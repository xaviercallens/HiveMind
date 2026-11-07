from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_cla  # noqa: E402
import generate_live_wan_swarm_report  # noqa: E402
import run_international_launch_and_vmoe_monitor  # noqa: E402
from tests.test_readme_claims import SPEC, _strip_lean_comments  # noqa: E402


def test_ip120_strip_lean_comments_detects_sorry_outside_comments() -> None:
    """IP-120: _strip_lean_comments must not eat code between two line comments."""
    cleaned_spec = _strip_lean_comments(SPEC)
    assert "sorry" not in cleaned_spec

    injected = SPEC + "\ntheorem t : False := by sorry\n"
    cleaned_injected = _strip_lean_comments(injected)
    assert "sorry" in cleaned_injected

    between_line_comments = (
        "-- first line comment\n"
        "theorem broken_claim : False := by sorry\n"
        "-- second line comment\n"
    )
    assert "sorry" in _strip_lean_comments(between_line_comments)


def test_ip121_live_wan_swarm_report_computes_catch_rate_from_verdicts(tmp_path: Path) -> None:
    """IP-121: adversarial_catch_rate_pct must be computed from actual E2E verdicts."""
    real_artifacts = REPO_ROOT / "artifacts"
    tmp_artifacts = tmp_path / "artifacts"
    tmp_artifacts.mkdir(parents=True, exist_ok=True)

    for subdir in ("boinc_prod_server", "boinc_core_node", "boinc_real"):
        dst_dir = tmp_artifacts / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(real_artifacts / subdir / "e2e_report.json", dst_dir / "e2e_report.json")

    for filename in (
        "rtx_intense_stress_and_steam_deploy_report.json",
        "nocturnal_moe_auto_improvement_report.json",
    ):
        shutil.copy2(real_artifacts / filename, tmp_artifacts / filename)

    baseline = generate_live_wan_swarm_report.build_live_wan_swarm_report(
        tmp_artifacts, write_output=False
    )
    audit_base = baseline["byzantine_and_volunteer_tax_audit"]
    assert audit_base["adversarial_catch_rate_pct"] == 100.0
    assert audit_base["adversarial_faults_injected"] == 6
    assert audit_base["adversarial_faults_rejected"] == 6

    boinc_real_path = tmp_artifacts / "boinc_real" / "e2e_report.json"
    boinc_real_data = json.loads(boinc_real_path.read_text(encoding="utf-8"))
    boinc_real_data["verdicts"]["hm_fault_001"]["valid"] = True
    boinc_real_data["verdicts"]["hm_fault_001"]["grant"]["capped_by"] = "verified"
    boinc_real_path.write_text(json.dumps(boinc_real_data, indent=2), encoding="utf-8")

    mutated = generate_live_wan_swarm_report.build_live_wan_swarm_report(
        tmp_artifacts, write_output=False
    )
    audit_mut = mutated["byzantine_and_volunteer_tax_audit"]
    assert audit_mut["adversarial_faults_injected"] == 6
    assert audit_mut["adversarial_faults_rejected"] == 5
    assert audit_mut["adversarial_catch_rate_pct"] < 100.0
    assert audit_mut["adversarial_catch_rate_pct"] == pytest.approx(83.3, abs=0.1)


def test_ip122_e20_benchmark_suite_derives_status_from_predictions_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IP-122: status must become FAILED when any prediction check fails."""
    passing_payload = run_international_launch_and_vmoe_monitor.run_e20_benchmark_suite()
    assert all(passing_payload["predictions_check"].values())
    assert passing_payload["status"] == "VERIFIED_PASS"

    orig_select = run_international_launch_and_vmoe_monitor.select_adaptive_multisplit_cut

    def _violating_select(*args: object, **kwargs: object):
        plan = orig_select(*args, **kwargs)
        return replace(
            plan,
            client_prelude_vram_mib=2048.0,
            within_1500mb_background_ceiling=False,
        )

    monkeypatch.setattr(
        run_international_launch_and_vmoe_monitor,
        "select_adaptive_multisplit_cut",
        _violating_select,
    )
    failing_payload = run_international_launch_and_vmoe_monitor.run_e20_benchmark_suite()
    assert failing_payload["predictions_check"]["P20a_all_prelude_vram_le_1536mb"] is False
    assert failing_payload["status"] == "FAILED"


def test_ip123_check_cla_rejects_unchecked_pr_template_checkbox() -> None:
    """IP-123: An unticked PR template checkbox must not satisfy CLA verification."""
    template_body = (REPO_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")
    ok_unticked, msg_unticked = check_cla.verify_cla_signature(
        "unregistered-contributor-xyz", pr_body=template_body
    )
    assert ok_unticked is False
    assert "has not signed CLA.md" in msg_unticked

    ticked_body = template_body.replace(
        "- [ ] **Contributor License Agreement & Dual-License IP Protection",
        "- [x] **Contributor License Agreement & Dual-License IP Protection",
    )
    ok_ticked, msg_ticked = check_cla.verify_cla_signature(
        "unregistered-contributor-xyz", pr_body=ticked_body
    )
    assert ok_ticked is True
    assert "signed the CLA in the PR body" in msg_ticked
