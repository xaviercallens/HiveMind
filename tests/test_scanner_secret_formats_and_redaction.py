# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The PR scanner catches the common credential formats and never copies the credential into its own report."""

from __future__ import annotations

import json

import pytest

from hivemind.security.pr_security_scanner import PRSecurityScanner

# Synthetic strings built at run time: none of them is a real credential.
SYNTHETIC = {
    "anthropic": "sk-ant-" + "api03-" + "A" * 40,
    "openrouter": "sk-or-" + "v1-" + "b" * 40,
    "openai-project": "sk-" + "proj-" + "C" * 40,
    "google": "AIza" + "Sy" + "d" * 33,
    "github-oauth": "gho" + "_" + "e" * 36,
    "github-server": "ghs" + "_" + "f" * 36,
    "stripe-live": "sk" + "_live_" + "g" * 24,
    "jwt": "eyJ" + "h" * 12 + ".eyJ" + "i" * 12 + "." + "j" * 12,
    "pem-encrypted": "-----BEGIN ENCRYPTED " + "PRIVATE KEY-----",
    "pem-pgp": "-----BEGIN PGP " + "PRIVATE KEY BLOCK-----",
}


def _scan(line: str) -> list:
    return PRSecurityScanner().scan_text_for_secrets_and_exfiltration(line, "a.py")


@pytest.mark.parametrize("name", sorted(SYNTHETIC))
def test_each_common_credential_format_is_detected(name: str) -> None:
    assert len(_scan(f'KEY = "{SYNTHETIC[name]}"')) >= 1


def test_controls_ordinary_text_and_test_fixtures_stay_clean() -> None:
    assert _scan('NAME = "hm-test-fixture-do-not-print"') == []
    assert _scan("risk-assessment-of-the-task-queue-for-sklearn-users") == []
    assert _scan("# sk-learn and gh_pages are not keys") == []


def test_the_report_never_contains_the_secret_it_found() -> None:
    token = "ghp" + "_" + "k" * 36
    findings = _scan(f'TOKEN = "{token}"')
    assert findings
    dumped = json.dumps([f.to_dict() for f in findings])
    assert token not in dumped and token[:12] not in dumped
    assert "redacted" in dumped
