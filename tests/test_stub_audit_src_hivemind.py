# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression & negative-control tests for Issue #415 (IP-24): zero-stub audit across src/hivemind.

Verifies:
1. Every Python file under ``src/hivemind/`` passes ``gwaya.ast_audit.ZeroStubAudit`` with zero violations.
2. ``HostDirectory.lookup`` in ``contributor_policy.py`` fails closed with ``(None, 0)`` rather than returning ``None``.
3. ``Backend.complete`` in ``science_bee.py`` raises ``BackendError`` so ``formulate()`` fails closed instead of crashing with ``NotImplementedError``.
4. ``_NoRedirect.redirect_request`` in ``science_bee.py`` records the refused redirect URL and returns ``None`` cleanly.
5. ``NocturnalAutoImprovementScheduler.verify_workunit_execution`` in ``auto_improvement.py`` replaces the forbidden ``simulate_`` identifier and verifies observed worker telemetry deterministically.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = PROJECT_ROOT / "third_party"
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))

from gwaya.ast_audit import ZeroStubAudit  # noqa: E402
from hivemind.boinc.real_client.contributor_policy import (  # noqa: E402
    HostDirectory,
    replay_all_for_result,
)
from hivemind.core.auto_improvement import (  # noqa: E402
    NocturnalAutoImprovementScheduler,
)
from hivemind.solverbee import science_bee as sb  # noqa: E402


@pytest.mark.parametrize(
    "rel_path",
    [
        "src/hivemind/boinc/real_client/contributor_policy.py",
        "src/hivemind/core/auto_improvement.py",
        "src/hivemind/solverbee/science_bee.py",
    ],
)
def test_target_src_files_pass_gwaya_zero_stub_audit(rel_path: str) -> None:
    """Negative control on main: these three src/hivemind files failed ZeroStubAudit before #415."""
    source = (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")
    res = ZeroStubAudit.audit_python_code(source)
    assert res.is_clean is True, f"{rel_path} failed ZeroStubAudit: {res.violations}"
    assert res.violations == []
    assert res.penalty_energy == 0.0


def test_entire_src_hivemind_tree_passes_gwaya_zero_stub_audit() -> None:
    """Every Python file under src/hivemind/ must pass ZeroStubAudit with 0 violations."""
    src_files = sorted((PROJECT_ROOT / "src" / "hivemind").rglob("*.py"))
    assert len(src_files) >= 100
    failures: dict[str, list[str]] = {}
    for f in src_files:
        res = ZeroStubAudit.audit_python_code(f.read_text(encoding="utf-8"))
        if not res.is_clean:
            failures[str(f.relative_to(PROJECT_ROOT))] = list(res.violations)
    assert failures == {}, f"src/hivemind files failed ZeroStubAudit: {failures}"


def test_host_directory_protocol_default_lookup_fails_closed() -> None:
    """Negative control: calling HostDirectory.lookup directly must return (None, 0) so replay_all_for_result fails closed."""
    class _UnconfiguredDirectory(HostDirectory):
        pass

    directory = _UnconfiguredDirectory()
    user_id, valid_count = directory.lookup(42)
    assert user_id is None
    assert valid_count == 0
    assert replay_all_for_result(42, directory, env={"HIVEMIND_COMMUNITY": "1"}) is True


def test_science_bee_base_backend_fails_closed_in_formulate() -> None:
    """Negative control: formulating with an unconfigured base Backend must fail closed via BackendError, not crash with NotImplementedError."""
    base_backend = sb.Backend()
    with pytest.raises(sb.BackendError, match="no language model backend configured"):
        base_backend.complete([{"role": "user", "content": "test"}])

    out = sb.formulate([{"role": "user", "content": "test"}], base_backend, max_retries=1)
    assert out["ok"] is False
    assert out["reason"] == "the language model could not be reached"
    assert any("backend error" in err for err in out["errors"])


def test_science_bee_no_redirect_records_refused_target() -> None:
    handler = sb._NoRedirect()
    res = handler.redirect_request(None, None, 302, "Moved", {}, "http://169.254.169.254/latest/meta-data/")
    assert res is None
    assert handler.last_refused_url == "http://169.254.169.254/latest/meta-data/"
    assert handler.last_refused_code == 302


def test_auto_improvement_verify_workunit_execution_and_observed_telemetry() -> None:
    scheduler = NocturnalAutoImprovementScheduler()
    wu = scheduler.create_domain_workunit(
        domain="maths_formal",
        num_samples=20,
        inner_steps=40,
        seed=20261101,
    )
    res = scheduler.verify_workunit_execution(
        workunit=wu,
        worker_id="worker-rtx2070",
        account_id="acct-rtx2070",
        execution_class="rtx2070-8gb-turing",
        observed_telemetry={
            "base_correct": 100,
            "adapted_correct": 152,
            "base_only_correct": 6,
            "adapted_only_correct": 58,
            "total_eval_samples": 200,
            "loss_initial": 2.40,
            "loss_final": 1.75,
            "peak_vram_mb": 4890.0,
            "duration_seconds": 84.5,
        },
    )
    assert res.wu_id == wu.wu_id
    assert res.accuracy_base == pytest.approx(0.50)
    assert res.accuracy_adapted == pytest.approx(0.76)
    assert res.gate_passed is True
    assert res.mcnemar_p_value < 0.001
    assert res.peak_vram_mb == pytest.approx(4890.0)
