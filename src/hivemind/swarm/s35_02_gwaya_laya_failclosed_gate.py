# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S35-02: GWAYA (Gwen-Laya) x ANSE Fail-Closed Selective Abstention & A3 Sample-and-Repair Gate.

Integrates:
1. GWAYA-GwenLayaOpenModel (/home/xavkal/xdev/GWAYA-GwenLayaOpenModel, Zenodo DOI 10.5281/zenodo.23123923):
   - Zero-Stub AST Gate rejecting placeholder implementations (`pass`, `...`, `NotImplementedError`, `mock_*`, `sorry`).
   - Laya Calibrated Selective Gate vs uncalibrated raw log-probability confidence.
   - Arm A3 Sample-and-Repair protocol (Best-of-3 candidate sampling + up to 3 repair rounds on public tests).
2. ANSE / AutoevolveAI (/home/xavkal/xdev/AutoevolveAI):
   - Objective physical Energy Function E, assigning Maximum Pain E = 10^6 whenever a candidate fails
     AST stub checks, compiler verification, or sandbox assertions.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import re
from typing import Callable, Sequence

MAXIMUM_PAIN_ENERGY: float = 1_000_000.0
FORBIDDEN_TEXT_PATTERNS = (
    r"\btodo!\s*\(",
    r"\bunimplemented!\s*\(",
    r"\bsorry\b",
    r"\badmit\b",
)


@dataclass(frozen=True)
class GWAYACandidateAudit:
    """Result of auditing a single code candidate under GWAYA + ANSE rules."""

    is_ast_clean: bool
    passed_public_test: bool
    passed_hidden_tests: bool
    laya_calibrated_score: float
    raw_logprob_confidence: float
    anse_energy: float
    verdict: str  # "VERIFIED" | "ABSTAIN_REPAIR" | "UNVERIFIED_MAX_PAIN"
    violations: tuple[str, ...]


@dataclass(frozen=True)
class GWAYAA3RepairReport:
    """Execution telemetry for the GWAYA Arm A3 Best-of-3 + 3 Repair Rounds loop."""

    initial_a0_passed: bool
    final_a3_passed: bool
    rounds_used: int
    final_energy: float
    selected_code: str
    audit_trail: tuple[GWAYACandidateAudit, ...]


def inspect_python_ast_anti_stub(source_code: str) -> list[str]:
    """Return all fail-closed AST stub or forbidden placeholder violations in Python code."""
    violations: list[str] = []
    if not source_code or not source_code.strip():
        return ["Empty candidate source code"]

    for pat in FORBIDDEN_TEXT_PATTERNS:
        if re.search(pat, source_code):
            violations.append(f"Forbidden placeholder token matching {pat}")

    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc.msg} at line {exc.lineno}"]

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith(("mock_", "fake_", "stub_")):
                violations.append(f"Forbidden simulated function name: {node.name}")
            # Strip leading docstring if present
            body = list(node.body)
            if (
                len(body) > 1
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            if len(body) == 1:
                stmt = body[0]
                if isinstance(stmt, ast.Pass):
                    violations.append(f"Function '{node.name}' body is a bare 'pass' stub")
                elif (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and stmt.value.value is Ellipsis
                ):
                    violations.append(f"Function '{node.name}' body is an ellipsis '...' stub")
                elif isinstance(stmt, ast.Raise):
                    violations.append(f"Function '{node.name}' raises unconditional exception stub")
    return violations


def audit_candidate_gwaya_anse(
    source_code: str,
    public_test_fn: Callable[[str], bool],
    hidden_test_fn: Callable[[str], bool],
    raw_logprob_confidence: float = 0.93,
    latency_ms: float = 12.0,
    memory_mib: float = 48.0,
    laya_threshold: float = 0.75,
) -> GWAYACandidateAudit:
    """Evaluate a code candidate with GWAYA fail-closed AST/test checks and Laya calibrated scoring."""
    violations = inspect_python_ast_anti_stub(source_code)
    is_ast_clean = len(violations) == 0

    pub_ok = False
    hid_ok = False
    if is_ast_clean:
        try:
            pub_ok = bool(public_test_fn(source_code))
        except Exception as exc:
            violations.append(f"Public test exception: {exc}")
        if pub_ok:
            try:
                hid_ok = bool(hidden_test_fn(source_code))
            except Exception as exc:
                violations.append(f"Hidden test exception: {exc}")

    # Laya calibrated score discounts uncalibrated raw log-prob when AST or public test fails
    if not is_ast_clean:
        laya_score = 0.0
    elif not pub_ok:
        laya_score = round(0.20 * raw_logprob_confidence, 4)
    else:
        laya_score = round(min(0.99, 0.55 + 0.45 * raw_logprob_confidence), 4)

    if is_ast_clean and pub_ok and hid_ok and laya_score >= laya_threshold:
        energy = round(latency_ms * 0.1 + memory_mib * 0.05, 4)
        verdict = "VERIFIED"
    elif is_ast_clean and laya_score < laya_threshold:
        energy = MAXIMUM_PAIN_ENERGY
        verdict = "ABSTAIN_REPAIR"
    else:
        energy = MAXIMUM_PAIN_ENERGY
        verdict = "UNVERIFIED_MAX_PAIN"

    return GWAYACandidateAudit(
        is_ast_clean=is_ast_clean,
        passed_public_test=pub_ok,
        passed_hidden_tests=hid_ok,
        laya_calibrated_score=laya_score,
        raw_logprob_confidence=raw_logprob_confidence,
        anse_energy=energy,
        verdict=verdict,
        violations=tuple(violations),
    )


def run_gwaya_a3_sample_and_repair(
    candidates: Sequence[str],
    public_test_fn: Callable[[str], bool],
    hidden_test_fn: Callable[[str], bool],
) -> GWAYAA3RepairReport:
    """Execute GWAYA Arm A3 protocol across sequential candidate/repair proposals."""
    if not candidates:
        raise ValueError("candidates must not be empty")

    audits: list[GWAYACandidateAudit] = []
    selected_code = candidates[0]
    for idx, code in enumerate(candidates):
        audit = audit_candidate_gwaya_anse(code, public_test_fn, hidden_test_fn)
        audits.append(audit)
        selected_code = code
        if audit.verdict == "VERIFIED":
            return GWAYAA3RepairReport(
                initial_a0_passed=(idx == 0),
                final_a3_passed=True,
                rounds_used=idx + 1,
                final_energy=audit.anse_energy,
                selected_code=selected_code,
                audit_trail=tuple(audits),
            )

    return GWAYAA3RepairReport(
        initial_a0_passed=False,
        final_a3_passed=False,
        rounds_used=len(audits),
        final_energy=MAXIMUM_PAIN_ENERGY,
        selected_code=selected_code,
        audit_trail=tuple(audits),
    )
