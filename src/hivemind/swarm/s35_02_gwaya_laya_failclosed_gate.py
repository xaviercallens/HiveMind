# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S35-02: GWAYA (Gwen-Laya) x ANSE Fail-Closed Selective Abstention & A3 Sample-and-Repair Gate.

Integrates:
1. GWAYA-GwenLayaOpenModel (xaviercallens/GWAYA-GwenLayaOpenModel, Zenodo DOI 10.5281/zenodo.23123923):
   - Zero-Stub AST Gate rejecting placeholder implementations (`pass`, `...`, `NotImplementedError`, `mock_*`, `sorry`).
   - Laya Calibrated Selective Gate vs uncalibrated raw log-probability confidence.
   - Arm A3 Sample-and-Repair protocol (Best-of-3 candidate sampling + up to 3 repair rounds on public tests).
2. ANSE / AutoevolveAI (xaviercallens/AutoevolveAI):
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


def _param_names(args: ast.arguments) -> set[str]:
    names: set[str] = set()
    for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        if arg.arg not in ("self", "cls"):
            names.add(arg.arg)
    if args.vararg and args.vararg.arg not in ("self", "cls"):
        names.add(args.vararg.arg)
    if args.kwarg and args.kwarg.arg not in ("self", "cls"):
        names.add(args.kwarg.arg)
    return names


def _referenced_names(nodes: Sequence[ast.AST]) -> set[str]:
    used: set[str] = set()
    for root in nodes:
        for sub in ast.walk(root):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                used.add(sub.id)
    return used


def _is_not_implemented_raise(stmt: ast.Raise) -> bool:
    exc = stmt.exc
    if exc is None:
        return False
    if isinstance(exc, ast.Name) and exc.id in ("NotImplementedError", "NotImplemented"):
        return True
    if isinstance(exc, ast.Call):
        func = exc.func
        if isinstance(func, ast.Name) and func.id in ("NotImplementedError", "NotImplemented"):
            return True
        if isinstance(func, ast.Attribute) and func.attr in ("NotImplementedError", "NotImplemented"):
            return True
    return False


def _strip_leading_docstring(body: Sequence[ast.stmt]) -> tuple[list[ast.stmt], bool]:
    stmts = list(body)
    if (
        stmts
        and isinstance(stmts[0], ast.Expr)
        and isinstance(stmts[0].value, ast.Constant)
        and isinstance(stmts[0].value.value, str)
    ):
        return stmts[1:], True
    return stmts, False


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
            if node.name.startswith(("mock_", "fake_", "stub_", "simulate_")):
                violations.append(f"Forbidden simulated function name: {node.name}")
            body, has_leading_docstring = _strip_leading_docstring(node.body)
            if has_leading_docstring and not body:
                # A body that is only a docstring returns None and does nothing: it is a stub like 'pass'.
                violations.append(f"Function '{node.name}' body is only a docstring stub")
                continue
            if not body:
                violations.append(f"Function '{node.name}' has an empty body")
                continue
            for stmt in body:
                if isinstance(stmt, ast.Raise) and _is_not_implemented_raise(stmt):
                    violations.append(f"Function '{node.name}' raises NotImplementedError stub")
                    break
            last = body[-1]
            if isinstance(last, ast.Pass):
                violations.append(f"Function '{node.name}' ends with a bare 'pass' stub")
            elif (
                isinstance(last, ast.Expr)
                and isinstance(last.value, ast.Constant)
                and last.value.value is Ellipsis
            ):
                violations.append(f"Function '{node.name}' ends with an ellipsis '...' stub")
            elif isinstance(last, ast.Raise) and len(body) == 1:
                violations.append(f"Function '{node.name}' raises unconditional exception stub")
            elif isinstance(last, ast.Return):
                val = last.value
                if isinstance(val, ast.Constant) and val.value is Ellipsis:
                    violations.append(f"Function '{node.name}' returns ellipsis '...' stub")
                elif isinstance(val, ast.Name) and val.id == "NotImplemented":
                    violations.append(f"Function '{node.name}' returns NotImplemented stub")
                elif val is None or (isinstance(val, ast.Constant) and val.value is None):
                    params = _param_names(node.args)
                    used = _referenced_names(body)
                    if len(body) == 1 or (params and not (params & used)):
                        violations.append(
                            f"Function '{node.name}' returns None stub without computing on its inputs"
                        )
        elif isinstance(node, ast.ClassDef):
            body, _ = _strip_leading_docstring(node.body)
            if not body or all(
                isinstance(s, ast.Pass)
                or (
                    isinstance(s, ast.Expr)
                    and isinstance(s.value, ast.Constant)
                    and s.value.value is Ellipsis
                )
                for s in body
            ):
                violations.append(f"Class '{node.name}' body is an empty/pass/ellipsis stub")
        elif isinstance(node, ast.Lambda):
            if isinstance(node.body, ast.Constant) or (
                isinstance(node.body, ast.Name) and node.body.id == "NotImplemented"
            ):
                violations.append("Lambda body is a constant/placeholder stub")
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
    # NaN passed `min(0.99, 0.55 + 0.45 * nan)` and scored VERIFIED; a confidence outside [0, 1] is not a probability.
    if not (isinstance(raw_logprob_confidence, (int, float)) and 0.0 <= raw_logprob_confidence <= 1.0):
        raise ValueError(f"raw_logprob_confidence must be a number in [0, 1], got {raw_logprob_confidence!r}")
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
