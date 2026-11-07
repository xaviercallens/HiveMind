"""Negative-control tests for Iteration 40: Cloud Run cryptography>=50.0.0 pin and Improvement Plan status headers."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cloud_run_requirements_pin_cryptography_50_or_newer() -> None:
    """Both Cloud Run requirements.txt files must pin cryptography>=50.0.0 (Dependabot CVE remediation)."""
    for rel in (
        "deploy/gcp_cloud_run/requirements.txt",
        "deploy/gcp_openrouter/requirements.txt",
    ):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        m = re.search(r"^cryptography==(\d+)\.(\d+)\.(\d+)\s*$", text, flags=re.MULTILINE)
        assert m is not None, f"{rel} must pin cryptography==X.Y.Z"
        major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
        assert (major, minor, patch) >= (50, 0, 0), f"{rel} pins vulnerable cryptography=={major}.{minor}.{patch}"


def test_all_six_improvement_plans_document_status_on_main() -> None:
    """All six IMPROVEMENT_PLAN_2026-*.md files must carry an explicit 'Status on `main` (2026-11-07):' header."""
    for date_suffix in (
        "2026-10-31",
        "2026-11-01",
        "2026-11-02",
        "2026-11-03",
        "2026-11-04",
        "2026-11-05",
    ):
        doc_path = REPO_ROOT / "docs" / f"IMPROVEMENT_PLAN_{date_suffix}.md"
        assert doc_path.is_file(), f"Missing {doc_path}"
        content = doc_path.read_text(encoding="utf-8")
        assert "Status on `main` (2026-11-07):" in content, f"{doc_path.name} missing status header"
