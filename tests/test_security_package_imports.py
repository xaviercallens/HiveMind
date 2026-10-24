"""Import-order regressions for hivemind.security, checked in fresh interpreters."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import hivemind.security as security

SRC = Path(__file__).resolve().parents[1] / "src"


def _run(code: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize(
    "first_import",
    ["hivemind.core.diloco_outer", "hivemind.boinc.coordinator_api", "hivemind.security.federated_learning_hardening"],
)
def test_no_circular_import_from_any_entry_point(first_import: str) -> None:
    result = _run(f"import {first_import}")
    assert result.returncode == 0, result.stderr[-2000:]


def test_pr_scanner_imports_without_torch() -> None:
    result = _run(
        "import sys; sys.modules['torch'] = None\n"
        "from hivemind.security.pr_security_scanner import PRSecurityScanner\n"
        "result = PRSecurityScanner().audit_pull_request(1, 'Update helper', '', 'contributor', "
        "[{'filename': 'helper.py', 'patch': '+value = 1'}], ['Update helper'], "
        "author_email='contributor@example.org')\n"
        "assert result.scanned_files_count == 1\n"
        "assert 'hivemind.security.swarm_runtime_hardening' not in sys.modules"
    )
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.parametrize(
    "statement",
    [
        "from hivemind.security.owner_order_guardrail import OwnerOrderGuardrail",
        "from hivemind.security import SwarmSecurityViolationError",
    ],
)
def test_lightweight_security_exports_without_torch(statement: str) -> None:
    result = _run(
        "import sys; sys.modules['torch'] = None\n"
        + statement
        + "\nassert 'hivemind.security.swarm_runtime_hardening' not in sys.modules"
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_security_exception_exports_preserve_type_and_fields() -> None:
    from hivemind.security._exceptions import SwarmSecurityViolationError
    from hivemind.security.owner_order_guardrail import SwarmSecurityViolationError as owner_error
    from hivemind.security.swarm_runtime_hardening import SwarmSecurityViolationError as runtime_error

    assert security.SwarmSecurityViolationError is owner_error is runtime_error is SwarmSecurityViolationError
    error = SwarmSecurityViolationError("TEST-01", "fixture", "example")
    assert isinstance(error, RuntimeError)
    assert (error.rule_id, error.category, error.message) == ("TEST-01", "fixture", "example")
    assert str(error) == "[TEST-01:fixture] example"


@pytest.mark.parametrize(
    "first_import", ["hivemind.boinc.validator", "hivemind.boinc.coordinator_api", "hivemind.boinc.workunit"]
)
def test_boinc_submodules_import_without_pillow(first_import: str) -> None:
    result = _run(
        "import sys; sys.modules['PIL'] = None\n"
        f"import {first_import}\n"
        "assert 'hivemind.boinc.astrophysics_24h_campaign' not in sys.modules"
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_campaign_render_names_pillow_when_missing() -> None:
    result = _run(
        "import sys; sys.modules['PIL'] = None\n"
        "import hivemind.boinc as b\n"
        "from hivemind.boinc.astrophysics_24h_campaign import _require_pil\n"
        "assert sorted(b.__all__) == sorted(b._EXPORTS)\n"
        "try:\n"
        "    _require_pil()\n"
        "except ImportError as e:\n"
        "    assert 'pillow' in str(e).lower(); print('ok')\n"
        "else:\n"
        "    raise SystemExit('expected ImportError')"
    )
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-2000:]


def test_lazy_exports_resolve_every_public_name() -> None:
    assert sorted(security._LAZY_EXPORTS) == sorted(security.__all__)
    for name in security.__all__:
        assert getattr(security, name) is not None
    with pytest.raises(AttributeError):
        security.not_a_real_export  # noqa: B018
