# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Verification tests for AGPL-3.0 Dual-License IP protection, copyright headers, and CLA gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import check_cla  # noqa: E402

EXPECTED_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
EXPECTED_README_NOTE = (
    "Project HiveMind is open-source under the AGPL-3.0 license. "
    "For enterprise use cases requiring a proprietary license exception, please contact: callensxavier@gmail.com"
)


def test_root_agpl3_license_exists() -> None:
    license_path = REPO_ROOT / "LICENSE"
    assert license_path.exists(), "Root LICENSE file must exist"
    text = license_path.read_text(encoding="utf-8")
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in text
    assert "Version 3, 19 November 2007" in text
    assert "Copyright (c) 2024-2026 Xavier Callens / Project HiveMind" in text
    assert EXPECTED_README_NOTE in text


def test_all_hivemind_source_files_have_agpl3_copyright_header() -> None:
    py_files = sorted((REPO_ROOT / "src" / "hivemind").rglob("*.py"))
    assert len(py_files) >= 100
    missing = []
    for p in py_files:
        head = p.read_text(encoding="utf-8")[:300]
        if EXPECTED_HEADER not in head:
            missing.append(str(p.relative_to(REPO_ROOT)))
    assert not missing, f"Source files missing AGPL-3.0 copyright header: {missing}"


def test_cla_document_and_verifier_work() -> None:
    root_cla_doc = REPO_ROOT / "CLA.md"
    cla_doc = REPO_ROOT / ".github" / "CLA.md"
    cla_sig = REPO_ROOT / ".github" / "cla-signatures.json"
    assert root_cla_doc.exists(), "Root CLA.md must exist"
    assert cla_doc.exists(), ".github/CLA.md must exist"
    assert cla_sig.exists()
    root_text = root_cla_doc.read_text(encoding="utf-8")
    doc_text = cla_doc.read_text(encoding="utf-8")
    for text in (root_text, doc_text):
        assert "Project Hive Mind — Individual & Entity Contributor License Agreement (CLA)" in text
        assert "Version 1.0 — October 2026" in text
        assert "I have read the CLA Document and I hereby sign the CLA" in text
        assert "Article L. 121-7 of the French Intellectual Property Code" in text
        assert "Nice / Marseille, France" in text
        assert "AGPL-3.0" in text
        assert "callensxavier@gmail.com" in text
    ledger = json.loads(cla_sig.read_text(encoding="utf-8"))
    assert ledger["required_phrase"] == "I have read the CLA Document and I hereby sign the CLA"

    ok_owner, _ = check_cla.verify_cla_signature("xaviercallens")
    assert ok_owner is True

    ok_unsigned, reason = check_cla.verify_cla_signature("new-contributor")
    assert ok_unsigned is False
    assert "I have read the CLA Document and I hereby sign the CLA" in reason

    ok_signed_comment, _ = check_cla.verify_cla_signature(
        "new-contributor",
        comments=[{"user": "new-contributor", "body": "I have read the CLA Document and I hereby sign the CLA"}],
    )
    assert ok_signed_comment is True

    ok_signed_full_statement, _ = check_cla.verify_cla_signature(
        "new-contributor",
        pr_body="I have read and agree to the Project Hive Mind Contributor License Agreement (CLA.md v1.0) and confirm that I have the legal authority to grant the rights therein.",
    )
    assert ok_signed_full_statement is True


def test_readme_states_agpl3_dual_license_intentions() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert EXPECTED_README_NOTE in readme
