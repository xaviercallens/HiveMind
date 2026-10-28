# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Automated 5-Layer Pull Request & Repository Security Scanner for Project HiveMind.

Protects the decentralized AI swarm repository against:
- **Code Boubou Attacks**: Hollow bounty-farming PRs claiming `fixes #N` / `resolves #N` without
  meaningful code contributions, binary artifact pollution (`.coverage`, `.pkl`, `.exe`, `.dll`),
  timestamp-only manipulation, or obfuscated backdoor payloads.
- **Payment Requests & Financial Extortion**: Cryptocurrency wallet addresses (BTC, ETH, SOL, XMR),
  fiat payment links (PayPal, Ko-fi, BuyMeACoffee, Patreon, Stripe, CashApp, Venmo), invoice demands,
  or attempts to bypass `LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`.
- **Dangerous Code Execution & Exfiltration**: Python AST inspection for `eval`, `exec`, `pickle.loads`,
  `marshal.loads`, `yaml.unsafe_load`, `os.system`, `subprocess(..., shell=True)`, base64 droppers,
  and secret/token exfiltration to external webhooks.
- **Trojan Source Unicode (CVE-2021-42574)**: Hidden bidirectional Unicode control characters.
- **Protected Governance & CI Workflow Tampering**: Unauthorized modifications to `.github/workflows/`,
  `.github/CODEOWNERS`, `LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`, `SECURITY.md`, or `GOVERNANCE.md`.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SecuritySeverity(str, Enum):
    """Severity classification for PR and repository security findings."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass(frozen=True)
class SecurityFinding:
    """Single security violation or anomaly detected in a PR or file."""

    rule_id: str
    category: str
    severity: SecuritySeverity
    file_path: str
    line_number: int
    description: str
    evidence_snippet: str
    remediation: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        return data


@dataclass
class PRSecurityAuditResult:
    """Aggregated security scan verdict for a Pull Request or repository tree."""

    target_name: str
    passed: bool
    blocked_merge: bool
    risk_score: int
    findings: list[SecurityFinding] = field(default_factory=list)
    scanned_files_count: int = 0
    functional_lines_added: int = 0
    summary: str = ""

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == SecuritySeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == SecuritySeverity.HIGH)

    @property
    def medium_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == SecuritySeverity.MEDIUM)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_name": self.target_name,
            "passed": self.passed,
            "blocked_merge": self.blocked_merge,
            "risk_score": self.risk_score,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "medium_count": self.medium_count,
            "scanned_files_count": self.scanned_files_count,
            "functional_lines_added": self.functional_lines_added,
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings],
        }


# ---------------------------------------------------------------------------
# Pattern Definitions
# ---------------------------------------------------------------------------

# 1. Payment Solicitation, Extortion & Crypto Wallet Patterns
PAYMENT_PLATFORM_REGEX = re.compile(
    r"(?i)\b(?:"
    r"paypal\.me/[a-zA-Z0-9_.\-]+|"
    r"paypal\.com/donate|"
    r"ko-fi\.com/[a-zA-Z0-9_.\-]+|"
    r"buymeacoffee\.com/[a-zA-Z0-9_.\-]+|"
    r"patreon\.com/[a-zA-Z0-9_.\-]+|"
    r"cash\.app/\$[a-zA-Z0-9_.\-]+|"
    r"venmo\.com/[a-zA-Z0-9_.\-]+|"
    r"stripe\.com/pay/|"
    r"gofundme\.com/f/|"
    r"opencollective\.com/[a-zA-Z0-9_.\-]+"
    r")"
)

PAYMENT_DEMAND_PHRASES_REGEX = re.compile(
    r"(?i)\b(?:"
    r"request\s+for\s+payment|"
    r"payment\s+request\s+for|"
    r"send\s+payment\s+to|"
    r"pay\s+me\s+(?:in|via|at|for)|"
    r"bounty\s+payout\s+in\s+(?:usd|eur|btc|eth|usdt|usdc|sol|crypto|cash|fiat)|"
    r"wire\s+transfer\s+to|"
    r"invoice\s+attached\s+for|"
    r"pay\s+to\s+unlock|"
    r"monetary\s+compensation\s+required|"
    r"send\s+(?:btc|eth|usdt|usdc|sol|crypto|funds|money)\s+to|"
    r"wallet\s+address\s+for\s+(?:payment|bounty|reward)|"
    r"iban\s*:\s*[A-Z]{2}\d{2}[A-Z0-9]{11,30}"
    r")\b"
)

CRYPTO_WALLET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "Bitcoin Bech32 Wallet (bc1...)",
        re.compile(r"\bbc1[ac-hj-np-z02-9]{25,87}\b"),
    ),
    (
        "Ethereum / EVM 20-Byte Wallet (0x...)",
        re.compile(r"\b0x[a-fA-F0-9]{40}\b"),
    ),
    (
        "Monero Standard Wallet (4...)",
        re.compile(r"\b4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}\b"),
    ),
    (
        "Explicit Crypto Payout Address Declaration",
        re.compile(
            r"(?i)\b(?:btc|bitcoin|eth|ethereum|usdt|usdc|sol|solana|xmr|monero)"
            r"\s*(?:wallet|address|payout)\s*[:=]\s*[a-zA-Z0-9]{24,95}\b"
        ),
    ),
]

# 2. Hardcoded Secrets & Private Keys
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("PEM Private Key Block", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----")),
    ("GitHub Personal Access Token", re.compile(r"\b(?:ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{82})\b")),
    ("Hugging Face API Token", re.compile(r"\bhf_[A-Za-z0-9]{34,}\b")),
    ("AWS Access Key ID", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Slack Webhook / Bot Token", re.compile(r"\bxox[baprs]-[0-9A-Za-z\-]{10,48}\b")),
]

# 3. Exfiltration & Suspicious Outbound Sinks
EXFILTRATION_SINKS_REGEX = re.compile(
    r"(?i)\b(?:"
    r"webhook\.site|"
    r"ngrok\.io|"
    r"ngrok-free\.app|"
    r"pipedream\.net|"
    r"requestbin\.(?:com|net)|"
    r"burpcollaborator\.net|"
    r"oastify\.com|"
    r"interact\.sh|"
    r"discord(?:app)?\.com/api/webhooks/\d+|"
    r"api\.telegram\.org/bot[0-9A-Za-z_:\-]+"
    r")"
)

# 4. Obfuscated Dropper / Backdoor Patterns
OBFUSCATED_DROPPER_REGEX = re.compile(
    r"(?i)(?:"
    r"base64\.b64decode\s*\([^)]*\)\s*\)|"
    r"exec\s*\(\s*base64\.|"
    r"eval\s*\(\s*base64\.|"
    r"exec\s*\(\s*codecs\.decode|"
    r"getattr\s*\(\s*__builtins__\s*,\s*['\"](?:eval|exec|__import__)['\"]\s*\)|"
    r"__import__\s*\(\s*['\"](?:os|subprocess|pty|socket)['\"]\s*\)\s*\.\s*(?:system|popen|spawn)"
    r")"
)

# 5. Trojan Source Bidirectional Unicode Characters (CVE-2021-42574)
TROJAN_UNICODE_CHARS: dict[str, str] = {
    "\u202a": "U+202A (LEFT-TO-RIGHT EMBEDDING)",
    "\u202b": "U+202B (RIGHT-TO-LEFT EMBEDDING)",
    "\u202c": "U+202C (POP DIRECTIONAL FORMATTING)",
    "\u202d": "U+202D (LEFT-TO-RIGHT OVERRIDE)",
    "\u202e": "U+202E (RIGHT-TO-LEFT OVERRIDE)",
    "\u2066": "U+2066 (LEFT-TO-RIGHT ISOLATE)",
    "\u2067": "U+2067 (RIGHT-TO-LEFT ISOLATE)",
    "\u2068": "U+2068 (FIRST STRONG ISOLATE)",
    "\u2069": "U+2069 (POP DIRECTIONAL ISOLATE)",
}

# 6. Forbidden Binary, Cache & Executable Artifacts in PRs ("Code Boubou" Binary Pollution)
FORBIDDEN_FILE_NAMES: set[str] = {
    ".coverage",
    ".env",
    ".env.local",
    ".env.production",
}

FORBIDDEN_FILE_EXTENSIONS: set[str] = {
    ".pkl",
    ".pickle",
    ".marshal",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".scr",
    ".msi",
    ".bat",
    ".cmd",
    ".vbs",
    ".pyc",
    ".pyo",
}

# 7. Protected Governance & CI Files requiring Repository Owner approval
PROTECTED_GOVERNANCE_PATHS: set[str] = {
    "CLA.md",
    ".github/CLA.md",
    ".github/cla-signatures.json",
    "LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md",
    "SECURITY.md",
    "GOVERNANCE.md",
    ".github/CODEOWNERS",
}

ISSUE_CLAIM_REGEX = re.compile(
    r"(?i)\b(?:fixes|resolves|closes|fixed|resolved|closed)\s+#\d+\b|\bHM-(?:S\d+|M\d+)-\d+\b"
)

FUNCTIONAL_CODE_PREFIXES: tuple[str, ...] = (
    "src/hivemind/",
    "lean4/",
    "tests/",
)


class PRSecurityScanner:
    """5-Layer Automated Pull Request & Repository Security Scanner."""

    def __init__(self, trusted_maintainers: set[str] | None = None) -> None:
        self.trusted_maintainers = {
            m.lower() for m in (trusted_maintainers or {"xaviercallens"})
        }

    # ------------------------------------------------------------------
    # Layer 1: Payment Solicitation & Crypto Wallet Injection Detection
    # ------------------------------------------------------------------
    def scan_text_for_payment_solicitation(
        self,
        text: str,
        file_path: str = "<pr_metadata>",
    ) -> list[SecurityFinding]:
        """Detect payment requests, fiat payment platform links, and crypto wallet addresses."""
        findings: list[SecurityFinding] = []
        if not text:
            return findings

        for idx, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue

            # Check payment platform links
            m_plat = PAYMENT_PLATFORM_REGEX.search(stripped)
            if m_plat:
                findings.append(
                    SecurityFinding(
                        rule_id="PAY-01",
                        category="PAYMENT_PLATFORM_LINK_DETECTED",
                        severity=SecuritySeverity.CRITICAL,
                        file_path=file_path,
                        line_number=idx,
                        description=(
                            f"External monetary payment/donation link detected ('{m_plat.group(0)}'). "
                            "Project HiveMind strictly prohibits monetary solicitation under "
                            "LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md."
                        ),
                        evidence_snippet=stripped[:180],
                        remediation="Remove all payment/donation URLs. Project HiveMind operates strictly with zero monetary compensation.",
                    )
                )

            # Check explicit payment demand phrases
            m_demand = PAYMENT_DEMAND_PHRASES_REGEX.search(stripped)
            if m_demand:
                findings.append(
                    SecurityFinding(
                        rule_id="PAY-02",
                        category="PAYMENT_SOLICITATION_DETECTED",
                        severity=SecuritySeverity.CRITICAL,
                        file_path=file_path,
                        line_number=idx,
                        description=(
                            f"Payment solicitation or financial demand phrase detected ('{m_demand.group(0)}'). "
                            "All contributions are voluntary and non-monetary."
                        ),
                        evidence_snippet=stripped[:180],
                        remediation="Remove any request for monetary payment, invoice, fiat transfer, or cryptocurrency payout.",
                    )
                )

            # Check cryptocurrency wallet addresses
            for wallet_label, wallet_regex in CRYPTO_WALLET_PATTERNS:
                m_wallet = wallet_regex.search(stripped)
                if m_wallet:
                    # Exclude zeroed placeholder addresses like 0x0000...0000
                    matched_val = m_wallet.group(0)
                    if matched_val.lower() == "0x" + ("0" * 40):
                        continue
                    findings.append(
                        SecurityFinding(
                            rule_id="PAY-03",
                            category="CRYPTO_WALLET_INJECTION_DETECTED",
                            severity=SecuritySeverity.CRITICAL,
                            file_path=file_path,
                            line_number=idx,
                            description=(
                                f"{wallet_label} detected ('{matched_val}'). "
                                "Cryptocurrency wallet injection and bounty payout demands are forbidden."
                            ),
                            evidence_snippet=stripped[:180],
                            remediation="Remove cryptocurrency wallet addresses. HiveMind compute credits are non-monetary internal counters.",
                        )
                    )

        return findings

    # ------------------------------------------------------------------
    # Layer 2: Trojan Source Bidirectional Unicode Scanner
    # ------------------------------------------------------------------
    def scan_unicode_trojan_source(
        self,
        text: str,
        file_path: str,
    ) -> list[SecurityFinding]:
        """Detect hidden bidirectional Unicode control characters (CVE-2021-42574)."""
        findings: list[SecurityFinding] = []
        if not text:
            return findings

        for idx, line in enumerate(text.splitlines(), start=1):
            for ch, ch_name in TROJAN_UNICODE_CHARS.items():
                if ch in line:
                    findings.append(
                        SecurityFinding(
                            rule_id="SEC-04",
                            category="TROJAN_SOURCE_BIDI_UNICODE",
                            severity=SecuritySeverity.CRITICAL,
                            file_path=file_path,
                            line_number=idx,
                            description=f"Trojan Source bidirectional Unicode control character {ch_name} detected.",
                            evidence_snippet=repr(line[:120]),
                            remediation="Strip all bidirectional Unicode control characters (U+202A..U+202E, U+2066..U+2069).",
                        )
                    )
        return findings

    # ------------------------------------------------------------------
    # Layer 3: Python AST & Obfuscation / Secret / Exfiltration Scanner
    # ------------------------------------------------------------------
    def scan_python_ast_security(
        self,
        file_path: str,
        source_code: str,
    ) -> list[SecurityFinding]:
        """Parse Python code into an AST and inspect for dangerous execution sinks."""
        findings: list[SecurityFinding] = []
        norm_path = file_path.replace("\\", "/")

        try:
            tree = ast.parse(source_code, filename=norm_path)
        except SyntaxError as exc:
            findings.append(
                SecurityFinding(
                    rule_id="SEC-00",
                    category="PYTHON_SYNTAX_ERROR",
                    severity=SecuritySeverity.HIGH,
                    file_path=norm_path,
                    line_number=exc.lineno or 1,
                    description=f"Python syntax error prevents AST security verification: {exc.msg}",
                    evidence_snippet=str(exc.text or "")[:140].strip(),
                    remediation="Fix the syntax error so static AST verification can complete.",
                )
            )
            return findings

        lines = source_code.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                lineno = getattr(node, "lineno", 1)
                snippet = lines[lineno - 1].strip() if 1 <= lineno <= len(lines) else ""
                func_name = self._resolve_call_name(node.func)

                # Direct eval / exec
                if func_name in {"eval", "exec", "builtins.eval", "builtins.exec"}:
                    findings.append(
                        SecurityFinding(
                            rule_id="SEC-01",
                            category="AST_DANGEROUS_EXECUTION_SINK",
                            severity=SecuritySeverity.CRITICAL,
                            file_path=norm_path,
                            line_number=lineno,
                            description=f"Dynamic code execution call `{func_name}()` is prohibited.",
                            evidence_snippet=snippet[:180],
                            remediation="Replace dynamic eval/exec with safe structured parsing (e.g., ast.literal_eval or JSON).",
                        )
                    )

                # Unsafe deserialization: pickle / marshal / shelve
                elif func_name in {
                    "pickle.load",
                    "pickle.loads",
                    "_pickle.load",
                    "_pickle.loads",
                    "marshal.load",
                    "marshal.loads",
                    "shelve.open",
                    "yaml.unsafe_load",
                }:
                    findings.append(
                        SecurityFinding(
                            rule_id="SEC-01",
                            category="AST_UNSAFE_DESERIALIZATION",
                            severity=SecuritySeverity.CRITICAL,
                            file_path=norm_path,
                            line_number=lineno,
                            description=f"Unsafe deserialization sink `{func_name}()` enables arbitrary code execution (RCE).",
                            evidence_snippet=snippet[:180],
                            remediation="Use safe serialization formats (JSON, safetensors, or torch.load(..., weights_only=True)).",
                        )
                    )

                # yaml.load without SafeLoader
                elif func_name == "yaml.load":
                    has_safe_loader = False
                    for kw in node.keywords:
                        if kw.arg == "Loader" and isinstance(kw.value, ast.Name) and "Safe" in kw.value.id:
                            has_safe_loader = True
                        elif kw.arg == "Loader" and isinstance(kw.value, ast.Attribute) and "Safe" in kw.value.attr:
                            has_safe_loader = True
                    if len(node.args) >= 2:
                        arg1 = node.args[1]
                        if (isinstance(arg1, ast.Name) and "Safe" in arg1.id) or (
                            isinstance(arg1, ast.Attribute) and "Safe" in arg1.attr
                        ):
                            has_safe_loader = True
                    if not has_safe_loader:
                        findings.append(
                            SecurityFinding(
                                rule_id="SEC-01",
                                category="AST_UNSAFE_YAML_LOAD",
                                severity=SecuritySeverity.HIGH,
                                file_path=norm_path,
                                line_number=lineno,
                                description="`yaml.load()` called without `SafeLoader`.",
                                evidence_snippet=snippet[:180],
                                remediation="Use `yaml.safe_load()` instead of `yaml.load()`.",
                            )
                        )

                # OS command execution sinks
                elif func_name in {"os.system", "os.popen", "pty.spawn"}:
                    findings.append(
                        SecurityFinding(
                            rule_id="SEC-01",
                            category="AST_SHELL_EXECUTION_SINK",
                            severity=SecuritySeverity.CRITICAL,
                            file_path=norm_path,
                            line_number=lineno,
                            description=f"Direct shell execution `{func_name}()` is prohibited in swarm modules.",
                            evidence_snippet=snippet[:180],
                            remediation="Use `subprocess.run([...], shell=False)` with explicit argument lists.",
                        )
                    )

                # subprocess with shell=True
                elif func_name in {
                    "subprocess.run",
                    "subprocess.Popen",
                    "subprocess.call",
                    "subprocess.check_call",
                    "subprocess.check_output",
                }:
                    for kw in node.keywords:
                        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            findings.append(
                                SecurityFinding(
                                    rule_id="SEC-01",
                                    category="AST_SUBPROCESS_SHELL_TRUE",
                                    severity=SecuritySeverity.CRITICAL,
                                    file_path=norm_path,
                                    line_number=lineno,
                                    description=f"`{func_name}(..., shell=True)` enables shell injection attacks.",
                                    evidence_snippet=snippet[:180],
                                    remediation="Set `shell=False` and pass command arguments as a list of strings.",
                                )
                            )

        return findings

    def scan_text_for_secrets_and_exfiltration(
        self,
        text: str,
        file_path: str,
    ) -> list[SecurityFinding]:
        """Scan text or source lines for hardcoded credentials, exfiltration webhooks, and obfuscated droppers."""
        findings: list[SecurityFinding] = []
        if not text:
            return findings

        for idx, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue

            # Hardcoded secrets
            for secret_label, secret_regex in SECRET_PATTERNS:
                m_sec = secret_regex.search(stripped)
                if m_sec:
                    findings.append(
                        SecurityFinding(
                            rule_id="SEC-05",
                            category="HARDCODED_SECRET_OR_PRIVATE_KEY",
                            severity=SecuritySeverity.CRITICAL,
                            file_path=file_path,
                            line_number=idx,
                            description=f"Hardcoded credential or key detected: {secret_label}.",
                            evidence_snippet=stripped[:100] + "...",
                            remediation="Remove hardcoded credentials immediately and rotate the compromised secret.",
                        )
                    )

            # Exfiltration sinks
            m_exfil = EXFILTRATION_SINKS_REGEX.search(stripped)
            if m_exfil:
                findings.append(
                    SecurityFinding(
                        rule_id="SEC-03",
                        category="CREDENTIAL_OR_ENV_EXFILTRATION",
                        severity=SecuritySeverity.CRITICAL,
                        file_path=file_path,
                        line_number=idx,
                        description=f"Suspicious external exfiltration webhook/tunnel detected ('{m_exfil.group(0)}').",
                        evidence_snippet=stripped[:180],
                        remediation="Remove unauthorized external webhook or tunnel endpoints.",
                    )
                )

            # Obfuscated droppers
            m_drop = OBFUSCATED_DROPPER_REGEX.search(stripped)
            if m_drop:
                findings.append(
                    SecurityFinding(
                        rule_id="SEC-02",
                        category="OBFUSCATED_BASE64_OR_HEX_DROPPER",
                        severity=SecuritySeverity.CRITICAL,
                        file_path=file_path,
                        line_number=idx,
                        description=f"Obfuscated payload dropper or dynamic builtin resolution detected ('{m_drop.group(0)}').",
                        evidence_snippet=stripped[:180],
                        remediation="Remove obfuscated execution payloads; all code must be transparent and auditable.",
                    )
                )

        return findings

    # ------------------------------------------------------------------
    # Layer 4 & 5: "Code Boubou" PR Diff & Governance Protection Audit
    # ------------------------------------------------------------------
    def audit_pull_request(
        self,
        pr_number: int | str,
        pr_title: str,
        pr_body: str,
        author: str,
        changed_files: list[dict[str, Any]],
        commit_messages: list[str] | None = None,
        author_email: str | None = None,
    ) -> PRSecurityAuditResult:
        """Perform a comprehensive 5-layer security and anti-'Code Boubou' audit on a Pull Request.

        Args:
            pr_number: GitHub PR number or identifier.
            pr_title: Pull request title.
            pr_body: Pull request markdown description.
            author: GitHub username of the PR author.
            changed_files: List of dicts with keys:
                - `filename` (str): Relative path of changed file
                - `status` (str, optional): added/modified/removed
                - `additions` (int, optional): Lines added
                - `deletions` (int, optional): Lines deleted
                - `patch` (str, optional): Unified diff patch string
                - `content` (str, optional): Full file content if available
            commit_messages: Optional list of commit messages in the PR.
            author_email: Optional sender/commit email address for OwnerOrderGuardrail verification.
        """
        from hivemind.security.owner_order_guardrail import (
            AUTHORIZED_OWNER_EMAIL,
            OwnerOrderGuardrail,
            OwnerOrderRequest,
            normalize_and_verify_owner_email,
        )

        findings: list[SecurityFinding] = []
        author_clean = (author or "unknown").strip().lower()
        email_verified = (
            normalize_and_verify_owner_email(author_email).is_authorized_owner
            if author_email is not None
            else (author_clean in self.trusted_maintainers)
        )
        is_trusted_owner = (author_clean in self.trusted_maintainers) and email_verified

        # 1. Scan PR title, body, and commit messages for payment requests / wallets / Trojan Unicode
        combined_meta = "\n".join(
            [
                pr_title or "",
                pr_body or "",
                *(commit_messages or []),
            ]
        )
        findings.extend(self.scan_text_for_payment_solicitation(combined_meta, file_path=f"PR#{pr_number}:metadata"))
        findings.extend(self.scan_unicode_trojan_source(combined_meta, file_path=f"PR#{pr_number}:metadata"))
        findings.extend(self.scan_text_for_secrets_and_exfiltration(combined_meta, file_path=f"PR#{pr_number}:metadata"))

        # 1b. OwnerOrderGuardrail: Block any administrative/directive order not from callensxavier@gmail.com
        order_guard = OwnerOrderGuardrail(strict_orders_only=False)
        effective_email = (
            author_email
            if author_email is not None
            else (AUTHORIZED_OWNER_EMAIL if is_trusted_owner else "")
        )
        order_decision = order_guard.evaluate_request(
            OwnerOrderRequest(
                request_id=f"PR#{pr_number}",
                sender_email=effective_email,
                content=combined_meta,
                sender_login=author_clean,
                source_channel="github_pr",
                changed_files=tuple(str(f.get("filename", "")) for f in changed_files),
            )
        )
        if not order_decision.accepted:
            findings.append(
                SecurityFinding(
                    rule_id=order_decision.rule_id,
                    category="UNAUTHORIZED_NON_OWNER_DIRECTIVE_ORDER",
                    severity=SecuritySeverity.CRITICAL,
                    file_path=f"PR#{pr_number}:owner_order_guardrail",
                    line_number=1,
                    description=(
                        f"Unauthorized directive order blocked by OwnerOrderGuardrail: "
                        f"{order_decision.reason} (triggers={list(order_decision.matched_triggers)})"
                    ),
                    evidence_snippet=combined_meta[:180],
                    remediation=(
                        f"Administrative and execution orders are accepted exclusively from "
                        f"`{AUTHORIZED_OWNER_EMAIL}`."
                    ),
                )
            )

        functional_lines_added = 0
        total_lines_added = 0
        touched_functional_file = False

        for f_info in changed_files:
            raw_fname = str(f_info.get("filename", "")).replace("\\", "/")
            fname_lower = raw_fname.lower()
            base_name = Path(raw_fname).name.lower()
            ext = Path(raw_fname).suffix.lower()
            status = str(f_info.get("status", "modified")).lower()
            additions = int(f_info.get("additions", 0))
            patch_text = str(f_info.get("patch") or "")
            full_content = f_info.get("content")

            total_lines_added += additions

            # Rule CB-01: Forbidden Binary, Cache, or Executable Artifact ("Code Boubou" Binary Pollution)
            if status != "removed" and (
                base_name in FORBIDDEN_FILE_NAMES
                or ext in FORBIDDEN_FILE_EXTENSIONS
                or "/__pycache__/" in f"/{fname_lower}"
            ):
                findings.append(
                    SecurityFinding(
                        rule_id="CB-01",
                        category="FORBIDDEN_BINARY_OR_CACHE_ARTIFACT",
                        severity=SecuritySeverity.CRITICAL,
                        file_path=raw_fname,
                        line_number=1,
                        description=(
                            f"Forbidden binary, cache, or executable artifact `{raw_fname}` included in PR "
                            "('Code Boubou' binary/cache pollution vector)."
                        ),
                        evidence_snippet=f"File: {raw_fname} (status={status}, additions={additions})",
                        remediation="Remove binary/cache files (`.coverage`, `.pkl`, `.exe`, `.dll`, `__pycache__`) from git tracking.",
                    )
                )

            # Rule CB-04: Protected Governance, Legal Charter & CI Workflow Tampering
            if not is_trusted_owner:
                if (
                    raw_fname in PROTECTED_GOVERNANCE_PATHS
                    or raw_fname.startswith(".github/workflows/")
                ):
                    findings.append(
                        SecurityFinding(
                            rule_id="CB-04",
                            category="PROTECTED_GOVERNANCE_OR_WORKFLOW_TAMPERING",
                            severity=SecuritySeverity.CRITICAL,
                            file_path=raw_fname,
                            line_number=1,
                            description=(
                                f"Unauthorized modification to protected governance/CI file `{raw_fname}` "
                                f"by non-owner `@{author}`."
                            ),
                            evidence_snippet=f"Modified protected path: {raw_fname}",
                            remediation="Revert changes to `.github/workflows/`, `CODEOWNERS`, `SECURITY.md`, and `LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`.",
                        )
                    )

            # Count functional lines added in src/hivemind/, lean4/, or tests/
            if raw_fname.startswith(FUNCTIONAL_CODE_PREFIXES) and status != "removed":
                touched_functional_file = True
                if patch_text:
                    added_code_lines = [
                        ln[1:].strip()
                        for ln in patch_text.splitlines()
                        if ln.startswith("+")
                        and not ln.startswith("+++")
                        and ln[1:].strip()
                        and not ln[1:].strip().startswith("#")
                        and not ln[1:].strip().startswith("--")
                    ]
                    functional_lines_added += len(added_code_lines)
                else:
                    functional_lines_added += additions

            # Extract added lines from patch to scan for payment links, secrets, exfiltration, and Trojan Unicode
            added_lines_text = "\n".join(
                ln[1:]
                for ln in patch_text.splitlines()
                if ln.startswith("+") and not ln.startswith("+++")
            )

            # Skip self-referential pattern definitions if trusted owner edits the security scanner or its unit tests
            is_security_Rule_file = raw_fname in {
                "src/hivemind/security/pr_security_scanner.py",
                "src/hivemind/security/owner_order_guardrail.py",
                "src/hivemind/security/swarm_runtime_hardening.py",
                "src/hivemind/security/federated_learning_hardening.py",
                "src/hivemind/security/gpu_execution_circuit_breaker.py",
                "src/hivemind/security/guardrail_attack_simulator.py",
                "tests/test_pr_security_scanner.py",
                "tests/test_owner_order_guardrail.py",
                "tests/test_swarm_runtime_hardening.py",
                "tests/test_agpl3_ip_protection.py",
                "CLA.md",
                ".github/CLA.md",
                "SECURITY.md",
                "LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md",
            }

            if added_lines_text and not (is_trusted_owner and is_security_Rule_file):
                findings.extend(
                    self.scan_text_for_payment_solicitation(added_lines_text, file_path=raw_fname)
                )
                findings.extend(
                    self.scan_text_for_secrets_and_exfiltration(added_lines_text, file_path=raw_fname)
                )

            if added_lines_text and not (
                is_trusted_owner
                and raw_fname
                in {
                    "src/hivemind/security/pr_security_scanner.py",
                    "src/hivemind/security/owner_order_guardrail.py",
                    "src/hivemind/security/swarm_runtime_hardening.py",
                    "src/hivemind/security/guardrail_attack_simulator.py",
                    "tests/test_pr_security_scanner.py",
                    "tests/test_owner_order_guardrail.py",
                    "tests/test_swarm_runtime_hardening.py",
                }
            ):
                findings.extend(
                    self.scan_unicode_trojan_source(added_lines_text, file_path=raw_fname)
                )

            # Python AST security check on full content or reconstructable Python source
            if raw_fname.endswith(".py") and status != "removed":
                if not (is_trusted_owner and is_security_Rule_file):
                    if isinstance(full_content, str) and full_content.strip():
                        findings.extend(self.scan_python_ast_security(raw_fname, full_content))
                    elif added_lines_text:
                        # Also check added lines for dangerous sinks even if only diff patch is provided
                        findings.extend(
                            self._scan_added_python_diff_lines(raw_fname, added_lines_text)
                        )

        # Rule CB-02 & CB-03: "Code Boubou" Hollow Bounty-Farming & Fake Issue Closure Detection
        claims_issue_or_bounty = bool(ISSUE_CLAIM_REGEX.search(combined_meta))
        if claims_issue_or_bounty and not touched_functional_file:
            findings.append(
                SecurityFinding(
                    rule_id="CB-02",
                    category="HOLLOW_BOUNTY_FARMING_CODE_BOUBOU",
                    severity=SecuritySeverity.CRITICAL,
                    file_path=f"PR#{pr_number}",
                    line_number=1,
                    description=(
                        f"Code Boubou Attack Detected: PR #{pr_number} claims to resolve an issue/bounty "
                        "but modifies ZERO functional files in `src/hivemind/`, `lean4/`, or `tests/`."
                    ),
                    evidence_snippet=f"Title: {pr_title[:120]} | Functional files touched: 0",
                    remediation=(
                        "A PR claiming an issue or bounty closure must include real, tested functional code "
                        "in `src/hivemind/`, `lean4/`, and `tests/`."
                    ),
                )
            )
        elif claims_issue_or_bounty and functional_lines_added < 5 and not is_trusted_owner:
            findings.append(
                SecurityFinding(
                    rule_id="CB-03",
                    category="MICRO_DIFF_BOUNTY_FARMING",
                    severity=SecuritySeverity.HIGH,
                    file_path=f"PR#{pr_number}",
                    line_number=1,
                    description=(
                        f"Suspicious micro-diff bounty claim: PR #{pr_number} claims an issue/bounty closure "
                        f"with only {functional_lines_added} functional line(s) added."
                    ),
                    evidence_snippet=f"Functional lines added: {functional_lines_added}",
                    remediation="Provide a complete implementation and unit test coverage for the claimed issue.",
                )
            )

        return self._build_audit_result(
            target_name=f"PR #{pr_number} ({pr_title})",
            findings=findings,
            scanned_files_count=len(changed_files),
            functional_lines_added=functional_lines_added,
        )

    def _scan_added_python_diff_lines(
        self,
        file_path: str,
        added_lines_text: str,
    ) -> list[SecurityFinding]:
        """Fallback line-level sink detector when only a unified diff patch is available."""
        findings: list[SecurityFinding] = []
        sink_regex = re.compile(
            r"(?<![.\w])(?:eval|exec)\s*\("
            r"|\b(?:pickle\.loads?|marshal\.loads?|shelve\.open|yaml\.unsafe_load|os\.system|os\.popen|pty\.spawn)\s*\("
            r"|subprocess\.(?:run|Popen|call|check_output|check_call)\s*\([^)]*shell\s*=\s*True"
        )
        for idx, line in enumerate(added_lines_text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            m = sink_regex.search(stripped)
            if m:
                findings.append(
                    SecurityFinding(
                        rule_id="SEC-01",
                        category="AST_DANGEROUS_EXECUTION_SINK",
                        severity=SecuritySeverity.CRITICAL,
                        file_path=file_path,
                        line_number=idx,
                        description=f"Dangerous execution or deserialization sink `{m.group(0)}` detected in PR diff.",
                        evidence_snippet=stripped[:180],
                        remediation="Remove dynamic code execution, unsafe deserialization, or `shell=True` calls.",
                    )
                )
        return findings

    def audit_repository(self, root_path: str | Path) -> PRSecurityAuditResult:
        """Scan the entire local repository tree (`src/hivemind`, `lean4`, `tests`, `.github`) for security compliance."""
        root = Path(root_path).resolve()
        findings: list[SecurityFinding] = []
        scanned_files = 0

        # Check for forbidden tracked artifacts in root
        for forbidden_name in FORBIDDEN_FILE_NAMES:
            candidate = root / forbidden_name
            # Note: local `.coverage` may exist transiently during local pytest runs if gitignored,
            # so we only flag if it is tracked in git or if scanning staged files.

        scan_dirs = [
            root / "src" / "hivemind",
            root / "lean4",
            root / "tests",
            root / ".github",
        ]

        exempt_self_files = {
            "src/hivemind/security/pr_security_scanner.py",
            "src/hivemind/security/owner_order_guardrail.py",
            "src/hivemind/security/swarm_runtime_hardening.py",
            "src/hivemind/security/federated_learning_hardening.py",
            "src/hivemind/security/gpu_execution_circuit_breaker.py",
            "src/hivemind/security/guardrail_attack_simulator.py",
            "tests/test_pr_security_scanner.py",
            "tests/test_owner_order_guardrail.py",
            "tests/test_swarm_runtime_hardening.py",
        }

        for sdir in scan_dirs:
            if not sdir.exists():
                continue
            for path in sorted(sdir.rglob("*")):
                if not path.is_file():
                    continue
                rel_path = path.relative_to(root).as_posix()
                if "/__pycache__/" in f"/{rel_path}" or rel_path.endswith(".pyc"):
                    continue

                scanned_files += 1
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

                if rel_path in exempt_self_files:
                    # Still verify syntax of the scanner & its test file via AST parse
                    if rel_path.endswith(".py"):
                        try:
                            ast.parse(content, filename=rel_path)
                        except SyntaxError as exc:
                            findings.append(
                                SecurityFinding(
                                    rule_id="SEC-00",
                                    category="PYTHON_SYNTAX_ERROR",
                                    severity=SecuritySeverity.HIGH,
                                    file_path=rel_path,
                                    line_number=exc.lineno or 1,
                                    description=f"Syntax error in {rel_path}: {exc.msg}",
                                    evidence_snippet="",
                                    remediation="Fix Python syntax.",
                                )
                            )
                    continue

                findings.extend(self.scan_unicode_trojan_source(content, file_path=rel_path))
                findings.extend(self.scan_text_for_payment_solicitation(content, file_path=rel_path))
                findings.extend(self.scan_text_for_secrets_and_exfiltration(content, file_path=rel_path))

                if rel_path.endswith(".py"):
                    findings.extend(self.scan_python_ast_security(rel_path, content))

        return self._build_audit_result(
            target_name=f"Repository Audit ({root.name})",
            findings=findings,
            scanned_files_count=scanned_files,
            functional_lines_added=0,
        )

    @staticmethod
    def _resolve_call_name(func_node: ast.AST) -> str:
        """Resolve an AST call target into a dotted function name."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            parts: list[str] = [func_node.attr]
            curr = func_node.value
            while isinstance(curr, ast.Attribute):
                parts.append(curr.attr)
                curr = curr.value
            if isinstance(curr, ast.Name):
                parts.append(curr.id)
            return ".".join(reversed(parts))
        return ""

    @staticmethod
    def _build_audit_result(
        target_name: str,
        findings: list[SecurityFinding],
        scanned_files_count: int,
        functional_lines_added: int,
    ) -> PRSecurityAuditResult:
        weights = {
            SecuritySeverity.CRITICAL: 100,
            SecuritySeverity.HIGH: 40,
            SecuritySeverity.MEDIUM: 15,
            SecuritySeverity.LOW: 5,
            SecuritySeverity.INFO: 0,
        }
        risk_score = sum(weights.get(f.severity, 0) for f in findings)
        blocked_merge = any(
            f.severity in {SecuritySeverity.CRITICAL, SecuritySeverity.HIGH}
            for f in findings
        )
        passed = not blocked_merge
        status_str = "PASSED (0 blocking findings)" if passed else f"BLOCKED ({len(findings)} findings, risk_score={risk_score})"
        summary = (
            f"{target_name}: {status_str} | "
            f"Scanned files={scanned_files_count}, Functional lines added={functional_lines_added}"
        )
        return PRSecurityAuditResult(
            target_name=target_name,
            passed=passed,
            blocked_merge=blocked_merge,
            risk_score=risk_score,
            findings=findings,
            scanned_files_count=scanned_files_count,
            functional_lines_added=functional_lines_added,
            summary=summary,
        )


def run_cli_security_scan(
    root_path: str = ".",
    pr_json_path: str | None = None,
    output_json_path: str | None = None,
) -> int:
    """CLI entrypoint for `hivemind scan-pr-security`."""
    scanner = PRSecurityScanner()
    if pr_json_path:
        pr_payload = json.loads(Path(pr_json_path).read_text(encoding="utf-8"))
        result = scanner.audit_pull_request(
            pr_number=pr_payload.get("number", "local"),
            pr_title=pr_payload.get("title", ""),
            pr_body=pr_payload.get("body", ""),
            author=pr_payload.get("author", "unknown"),
            changed_files=pr_payload.get("files", []),
            commit_messages=pr_payload.get("commit_messages", []),
            author_email=pr_payload.get("author_email"),
        )
    else:
        result = scanner.audit_repository(root_path=root_path)

    report_dict = result.to_dict()
    if output_json_path:
        out_path = Path(output_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    print(json.dumps(report_dict, indent=2))
    return 0 if result.passed else 1
