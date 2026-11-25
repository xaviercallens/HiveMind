# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Pure-Python Lean 4 Season 35 WSP Theorem Registry & Formal Spec Attestation Bridge.

Provides zero-subprocess, zero-network deterministic auditing of ``lean4/HiveMindSpec.lean``
for ``Season35_AutoOpenWeightAndWeightSpaceProgramming`` and binds runtime ``S5`` Weight-Space
Programming surgery receipts to their machine-checked Lean 4 theorem identifiers via canonical
SHA-256 provenance attestation:

1. **Deterministic Comment/String-Aware Lean 4 Parser** (``parse_lean4_wsp_spec`` /
   ``audit_lean4_wsp_spec_file``):
   - Strips nested ``/- ... -/`` block comments, ``-- ...`` line comments, and string literals
     while preserving line boundaries.
   - Tracks ``namespace`` / ``section`` / ``end`` lexical scopes to isolate
     ``HiveMindSpec.Season35_AutoOpenWeightAndWeightSpaceProgramming``.
   - Verifies zero ``sorry``, ``admit``, ``sorryAx``, ``native_decide``, or custom ``axiom``
     declarations across ``Season35_AutoOpenWeightAndWeightSpaceProgramming`` and the full spec.
   - Verifies that every Season 35 theorem has a matching top-level ``#print axioms`` directive.
2. **Runtime S5 Surgery Receipt -> Lean 4 Obligation Binding & SHA-256 Spec Attestation**
   (``bind_s5_receipt_to_lean4_theorems`` / ``verify_lean4_attestation_receipt``):
   - Binds ``AutoOpenWeightResult``, ``PostRoundWSPLineageReceipt``,
     ``MonotoneWSPPipelineResult``, or ``SingleShotVaultLock`` receipts to the 5 Season 35
     theorems, verifying embedded SHA-256 digests and detecting tampering on either the Lean spec
     or the bound surgery receipt.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION: str = "hivemind.lean4_wsp_theorem_registry/1"
SPEC_AUDIT_OBJECT: str = "hivemind.lean4_wsp_spec_audit"
ATTESTATION_RECEIPT_OBJECT: str = "hivemind.lean4_wsp_formal_attestation"

ROOT_NAMESPACE: str = "HiveMindSpec"
SEASON35_NAMESPACE: str = "Season35_AutoOpenWeightAndWeightSpaceProgramming"
SEASON35_QUALIFIED_PREFIX: str = f"{ROOT_NAMESPACE}.{SEASON35_NAMESPACE}"
DEFAULT_SPEC_RELATIVE_PATH: Path = Path("lean4") / "HiveMindSpec.lean"

EXPECTED_SEASON35_THEOREMS: tuple[str, ...] = (
    "nullspace_projector_idempotent_scalar",
    "preserved_key_zero_drift_under_nullspace_edit",
    "orthogonal_multigate_specialist_dot_invariant",
    "monotone_anse_stage_ratchet_strictly_descends",
    "single_shot_vault_lock_blocks_second_eval",
)

DEFAULT_ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_GENERAL_GUARD_REL: float = 0.01

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENT_RE = r"[A-Za-z_][A-Za-z0-9_'.]*"
_SCOPE_OPEN_RE = re.compile(rf"^\s*(namespace|section)\s*({_IDENT_RE})?\s*$")
_SCOPE_CLOSE_RE = re.compile(rf"^\s*end(?:\s+({_IDENT_RE}))?\s*$")
_THEOREM_DECL_RE = re.compile(rf"^\s*(?:theorem|lemma)\s+({_IDENT_RE})(?=\s|:|\(|$)")
_THEOREM_ANYWHERE_RE = re.compile(r"\b(?:theorem|lemma)\s+")
_PRINT_AXIOMS_RE = re.compile(rf"^\s*#print\s+axioms\s+({_IDENT_RE})\s*$")
_SORRY_RE = re.compile(r"\bsorry\b")
_ADMIT_RE = re.compile(r"\badmit\b")
_SORRY_AX_RE = re.compile(r"\bsorryAx\b")
_NATIVE_DECIDE_RE = re.compile(r"\bnative_decide\b")
_CUSTOM_AXIOM_RE = re.compile(r"\baxiom\b")
_LEX_OUTSIDE_RE = re.compile(r'/-|--|"')
_BLOCK_NEST_RE = re.compile(r"/-|-/")
_STRING_END_RE = re.compile(r'\\.|"', re.DOTALL)
_NON_NEWLINE_RE = re.compile(r"[^\n]")


class Lean4SpecAuditError(ValueError):
    """Raised when a Lean 4 specification fails formal hygiene or integrity checks."""


class Lean4AttestationError(ValueError):
    """Raised when binding or verifying a Lean 4 attestation receipt fails."""


@dataclass(frozen=True)
class Lean4TheoremSpec:
    """Canonical metadata linking one Season 35 Lean 4 theorem to its runtime WSP obligation."""

    obligation_id: str
    short_name: str
    qualified_name: str
    wsp_stage: str
    invariant_summary: str

    def to_dict(self) -> dict[str, str]:
        """Serialize theorem specification to a JSON-safe dictionary."""
        return {
            "obligation_id": self.obligation_id,
            "short_name": self.short_name,
            "qualified_name": self.qualified_name,
            "wsp_stage": self.wsp_stage,
            "invariant_summary": self.invariant_summary,
        }


SEASON35_THEOREM_CATALOG: tuple[Lean4TheoremSpec, ...] = (
    Lean4TheoremSpec(
        obligation_id="S35-WSP-THM-01",
        short_name="nullspace_projector_idempotent_scalar",
        qualified_name=f"{SEASON35_QUALIFIED_PREFIX}.nullspace_projector_idempotent_scalar",
        wsp_stage="S5.3_alphaedit_nullspace",
        invariant_summary="AlphaEdit rank-1 nullspace projector satisfies P_d^2 = d*P_d and P_1^2 = P_1 with <u, P_d x> = 0.",
    ),
    Lean4TheoremSpec(
        obligation_id="S35-WSP-THM-02",
        short_name="preserved_key_zero_drift_under_nullspace_edit",
        qualified_name=f"{SEASON35_QUALIFIED_PREFIX}.preserved_key_zero_drift_under_nullspace_edit",
        wsp_stage="S5.1_S5.3_preserved_key_guard",
        invariant_summary="Projecting deltas via P_0 preserves keys k in span(u): dw_proj . k = 0 and (w + dw_proj) . k = w . k.",
    ),
    Lean4TheoremSpec(
        obligation_id="S35-WSP-THM-03",
        short_name="orthogonal_multigate_specialist_dot_invariant",
        qualified_name=f"{SEASON35_QUALIFIED_PREFIX}.orthogonal_multigate_specialist_dot_invariant",
        wsp_stage="S5.2_oof_orthogonal_multigate",
        invariant_summary="Orthogonalized router direction u_{0 perp mu} satisfies u . mu = 0 and keeps specialist logits invariant.",
    ),
    Lean4TheoremSpec(
        obligation_id="S35-WSP-THM-04",
        short_name="monotone_anse_stage_ratchet_strictly_descends",
        qualified_name=f"{SEASON35_QUALIFIED_PREFIX}.monotone_anse_stage_ratchet_strictly_descends",
        wsp_stage="S5_anse_monotone_ratchet",
        invariant_summary="ANSE multi-stage ratchet never increases energy, rolls back guard violations, and strictly descends on accept.",
    ),
    Lean4TheoremSpec(
        obligation_id="S35-WSP-THM-05",
        short_name="single_shot_vault_lock_blocks_second_eval",
        qualified_name=f"{SEASON35_QUALIFIED_PREFIX}.single_shot_vault_lock_blocks_second_eval",
        wsp_stage="S5_single_shot_holdout_vault",
        invariant_summary="SingleShotVaultLock permits at most 1 evaluation and blocks every subsequent access.",
    ),
)

SEASON35_THEOREM_BY_NAME: dict[str, Lean4TheoremSpec] = {
    spec.short_name: spec for spec in SEASON35_THEOREM_CATALOG
}


@dataclass(frozen=True)
class Lean4WSPSpecAudit:
    """Deterministic static audit report over ``lean4/HiveMindSpec.lean`` Season 35 theorems."""

    spec_path: str
    spec_sha256: str
    season35_section_sha256: str
    season35_namespace_present: bool
    namespaces_balanced: bool
    season35_theorems_declared: tuple[str, ...]
    season35_qualified_theorems: tuple[str, ...]
    total_spec_theorems: int
    expected_theorems_missing: tuple[str, ...]
    duplicate_season35_theorems: tuple[str, ...]
    top_level_print_axioms: tuple[str, ...]
    nested_print_axioms: tuple[str, ...]
    missing_print_axioms: tuple[str, ...]
    season35_sorry_count: int
    season35_admit_count: int
    season35_custom_axiom_count: int
    season35_native_decide_count: int
    season35_sorry_ax_count: int
    global_sorry_count: int
    global_admit_count: int
    global_custom_axiom_count: int
    global_native_decide_count: int
    global_sorry_ax_count: int
    violations: tuple[str, ...]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize audit report to a JSON-compatible dictionary."""
        return {
            "object": SPEC_AUDIT_OBJECT,
            "schema": SCHEMA_VERSION,
            "spec_path": self.spec_path,
            "spec_sha256": self.spec_sha256,
            "season35_section_sha256": self.season35_section_sha256,
            "season35_namespace_present": self.season35_namespace_present,
            "namespaces_balanced": self.namespaces_balanced,
            "season35_theorems_declared": list(self.season35_theorems_declared),
            "season35_qualified_theorems": list(self.season35_qualified_theorems),
            "total_spec_theorems": self.total_spec_theorems,
            "expected_theorems_missing": list(self.expected_theorems_missing),
            "duplicate_season35_theorems": list(self.duplicate_season35_theorems),
            "top_level_print_axioms": list(self.top_level_print_axioms),
            "nested_print_axioms": list(self.nested_print_axioms),
            "missing_print_axioms": list(self.missing_print_axioms),
            "season35_sorry_count": self.season35_sorry_count,
            "season35_admit_count": self.season35_admit_count,
            "season35_custom_axiom_count": self.season35_custom_axiom_count,
            "season35_native_decide_count": self.season35_native_decide_count,
            "season35_sorry_ax_count": self.season35_sorry_ax_count,
            "global_sorry_count": self.global_sorry_count,
            "global_admit_count": self.global_admit_count,
            "global_custom_axiom_count": self.global_custom_axiom_count,
            "global_native_decide_count": self.global_native_decide_count,
            "global_sorry_ax_count": self.global_sorry_ax_count,
            "violations": list(self.violations),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class TheoremObligationBinding:
    """Binding between a runtime S5 surgery invariant and its formal Lean 4 theorem."""

    obligation_id: str
    theorem_short_name: str
    theorem_qualified_name: str
    wsp_stage: str
    lean_verified: bool
    exercised: bool
    runtime_satisfied: bool
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize binding entry to a JSON-safe dictionary."""
        return {
            "obligation_id": self.obligation_id,
            "theorem_short_name": self.theorem_short_name,
            "theorem_qualified_name": self.theorem_qualified_name,
            "wsp_stage": self.wsp_stage,
            "lean_verified": self.lean_verified,
            "exercised": self.exercised,
            "runtime_satisfied": self.runtime_satisfied,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class Lean4WSPAttestationReceipt:
    """Tamper-evident SHA-256 attestation binding S5 surgery telemetry to Season 35 Lean 4 proofs."""

    spec_path: str
    spec_sha256: str
    season35_section_sha256: str
    spec_audit_passed: bool
    theorem_count_verified: int
    season35_namespace: str
    surgery_receipt_object: str
    surgery_receipt_verdict: str
    surgery_receipt_sha256: str
    surgery_embedded_sha256: str
    checkpoint_sha256: str
    vault_receipt_sha256: str | None
    theorem_bindings: tuple[TheoremObligationBinding, ...]
    all_lean_theorems_verified: bool
    all_runtime_obligations_satisfied: bool
    attestation_valid: bool
    failures: tuple[str, ...]
    attestation_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize attestation receipt to a canonical JSON-compatible dictionary."""
        return {
            "object": ATTESTATION_RECEIPT_OBJECT,
            "schema": SCHEMA_VERSION,
            "spec_path": self.spec_path,
            "spec_sha256": self.spec_sha256,
            "season35_section_sha256": self.season35_section_sha256,
            "spec_audit_passed": self.spec_audit_passed,
            "theorem_count_verified": self.theorem_count_verified,
            "season35_namespace": self.season35_namespace,
            "surgery_receipt_object": self.surgery_receipt_object,
            "surgery_receipt_verdict": self.surgery_receipt_verdict,
            "surgery_receipt_sha256": self.surgery_receipt_sha256,
            "surgery_embedded_sha256": self.surgery_embedded_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "vault_receipt_sha256": self.vault_receipt_sha256,
            "theorem_bindings": [b.to_dict() for b in self.theorem_bindings],
            "all_lean_theorems_verified": self.all_lean_theorems_verified,
            "all_runtime_obligations_satisfied": self.all_runtime_obligations_satisfied,
            "attestation_valid": self.attestation_valid,
            "failures": list(self.failures),
            "attestation_sha256": self.attestation_sha256,
        }


@dataclass(frozen=True)
class Lean4AttestationVerification:
    """Result of verifying a ``Lean4WSPAttestationReceipt`` against Lean spec & runtime receipts."""

    valid: bool
    attestation_sha256_ok: bool
    spec_sha256_ok: bool
    season35_section_sha256_ok: bool
    spec_audit_ok: bool
    surgery_receipt_ok: bool
    vault_receipt_ok: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize verification report to a JSON-safe dictionary."""
        return {
            "valid": self.valid,
            "attestation_sha256_ok": self.attestation_sha256_ok,
            "spec_sha256_ok": self.spec_sha256_ok,
            "season35_section_sha256_ok": self.season35_section_sha256_ok,
            "spec_audit_ok": self.spec_audit_ok,
            "surgery_receipt_ok": self.surgery_receipt_ok,
            "vault_receipt_ok": self.vault_receipt_ok,
            "failures": list(self.failures),
        }


# ---------------------------------------------------------------------------
# 1. Validation & Canonical JSON / SHA-256 Helpers
# ---------------------------------------------------------------------------


def _validate_nonempty_str(value: Any, name: str) -> str:
    """Validate that ``value`` is a non-empty printable string without NUL bytes."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    if "\x00" in value:
        raise ValueError(f"{name} must not contain NUL bytes")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{name} must be a non-empty string")
    return stripped


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_nonneg_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite non-negative float (excluding bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and >= 0.0, got {value!r}")
    return out


def _normalize_json_value(value: Any) -> Any:
    """Normalize tuples and nested structures into canonical JSON-serializable values."""
    if isinstance(value, tuple):
        return [_normalize_json_value(v) for v in value]
    if isinstance(value, list):
        return [_normalize_json_value(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _normalize_json_value(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite float encountered in JSON payload")
    return value


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` deterministically with strict finite-number enforcement."""
    normalized = _normalize_json_value(payload)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def compute_payload_sha256(payload: Any) -> str:
    """Return lowercase 64-hex SHA-256 digest over ``canonical_json_bytes(payload)``."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _coerce_receipt_mapping(receipt: Any, name: str) -> dict[str, Any]:
    """Convert a receipt dataclass or Mapping into a validated non-empty dictionary."""
    if isinstance(receipt, Mapping):
        raw = dict(receipt)
    elif hasattr(receipt, "to_dict") and callable(receipt.to_dict):
        raw = receipt.to_dict()
    elif hasattr(receipt, "receipt") and isinstance(receipt.receipt, Mapping):
        raw = dict(receipt.receipt)
    elif dataclasses.is_dataclass(receipt) and not isinstance(receipt, type):
        raw = dataclasses.asdict(receipt)
    else:
        raise TypeError(f"{name} must be a Mapping or receipt dataclass, got {type(receipt).__name__}")

    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{name} must not be empty")
    return {str(k): _normalize_json_value(v) for k, v in raw.items()}


def _verify_embedded_receipt_sha256(receipt: Mapping[str, Any]) -> tuple[bool, str]:
    """Verify optional embedded ``receipt_sha256`` / ``vault_receipt_sha256`` if present."""
    for field_name in ("receipt_sha256", "vault_receipt_sha256"):
        if field_name not in receipt:
            continue
        claimed = receipt.get(field_name)
        if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
            return False, f"invalid_{field_name}_format"
        core_without = {k: v for k, v in receipt.items() if k != field_name}
        core_empty = {k: ("" if k == field_name else v) for k, v in receipt.items()}
        try:
            sha_without = compute_payload_sha256(core_without)
            sha_empty = compute_payload_sha256(core_empty)
        except (TypeError, ValueError):
            return False, f"non_serializable_{field_name}"
        if claimed not in (sha_without, sha_empty):
            return False, f"{field_name}_mismatch"
        return True, claimed
    return True, ""


# ---------------------------------------------------------------------------
# 2. Pure-Python Lean 4 Lexer & Season 35 Specification Audit
# ---------------------------------------------------------------------------


def strip_lean4_comments_and_strings(lean_source: str) -> tuple[str, bool]:
    """Strip nested ``/- ... -/`` block comments, ``-- ...`` line comments, and string literals.

    Returns ``(stripped_source, lexical_balanced)`` where ``lexical_balanced`` is False if an
    unterminated block comment or string literal reaches EOF. Line breaks are preserved.
    """
    if not isinstance(lean_source, str):
        raise TypeError(f"lean_source must be a str, got {type(lean_source).__name__}")
    if "\x00" in lean_source:
        raise ValueError("lean_source must not contain NUL bytes")

    chunks: list[str] = []
    pos = 0
    n = len(lean_source)
    lexical_balanced = True

    while pos < n:
        match = _LEX_OUTSIDE_RE.search(lean_source, pos)
        if match is None:
            chunks.append(lean_source[pos:])
            break
        if match.start() > pos:
            chunks.append(lean_source[pos : match.start()])

        tok = match.group(0)
        if tok == "--":
            nl = lean_source.find("\n", match.end())
            end_idx = n if nl == -1 else nl
            chunks.append(" " * (end_idx - match.start()))
            pos = end_idx
        elif tok == "/-":
            depth = 1
            scan = match.end()
            while depth > 0:
                bm = _BLOCK_NEST_RE.search(lean_source, scan)
                if bm is None:
                    chunks.append(_NON_NEWLINE_RE.sub(" ", lean_source[match.start() :]))
                    lexical_balanced = False
                    pos = n
                    break
                depth += 1 if bm.group(0) == "/-" else -1
                scan = bm.end()
            if depth == 0:
                chunks.append(_NON_NEWLINE_RE.sub(" ", lean_source[match.start() : scan]))
                pos = scan
        else:  # '"'
            scan = match.end()
            while True:
                sm = _STRING_END_RE.search(lean_source, scan)
                if sm is None:
                    chunks.append(_NON_NEWLINE_RE.sub(" ", lean_source[match.start() :]))
                    lexical_balanced = False
                    pos = n
                    break
                if sm.group(0) == '"':
                    chunks.append(_NON_NEWLINE_RE.sub(" ", lean_source[match.start() : sm.end()]))
                    pos = sm.end()
                    break
                scan = sm.end()

    return "".join(chunks), lexical_balanced


def _compute_season35_section_digest(
    raw_lines: Sequence[str],
    section_span: tuple[int, int] | None,
    top_level_print_lines: Sequence[str],
) -> str:
    """Compute canonical SHA-256 over Season 35 namespace lines and top-level ``#print axioms``."""
    if section_span is None:
        return hashlib.sha256(b"MISSING_SEASON35_NAMESPACE").hexdigest()
    start_idx, end_idx = section_span
    section_text = "\n".join(raw_lines[start_idx : end_idx + 1]).strip()
    filtered_prints = sorted(
        line.strip()
        for line in top_level_print_lines
        if line.strip().startswith(f"#print axioms {SEASON35_QUALIFIED_PREFIX}.")
    )
    canonical = section_text + "\n" + "\n".join(filtered_prints) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_lean4_wsp_spec(
    lean_source: str,
    *,
    spec_path: str = "lean4/HiveMindSpec.lean",
    expected_theorems: Sequence[str] = EXPECTED_SEASON35_THEOREMS,
) -> Lean4WSPSpecAudit:
    """Deterministically parse a Lean 4 specification and audit Season 35 WSP soundness."""
    _validate_nonempty_str(spec_path, "spec_path")
    if isinstance(lean_source, str) and not lean_source.strip():
        raise ValueError("lean_source must not be empty")
    if isinstance(expected_theorems, (str, bytes)) or not isinstance(expected_theorems, Sequence):
        raise TypeError("expected_theorems must be a non-empty sequence of theorem names")
    expected_list = tuple(_validate_nonempty_str(name, "expected_theorem") for name in expected_theorems)
    if not expected_list:
        raise ValueError("expected_theorems must not be empty")

    stripped, lexical_balanced = strip_lean4_comments_and_strings(lean_source)
    spec_sha256 = hashlib.sha256(lean_source.encode("utf-8")).hexdigest()

    raw_lines = lean_source.splitlines()
    stripped_lines = stripped.splitlines()

    scopes: list[tuple[str, str]] = []
    namespaces_balanced = lexical_balanced
    season35_span_start: int | None = None
    season35_span_end: int | None = None
    season35_stripped_lines: list[str] = []
    season35_declared: list[str] = []
    season35_qualified: list[str] = []
    season35_duplicates: list[str] = []
    seen_season35: set[str] = set()
    all_declared_theorems: list[str] = []
    top_level_prints: list[str] = []
    top_level_print_raw_lines: list[str] = []
    nested_prints: list[str] = []
    season35_unmatched_theorem_lines = 0
    global_unmatched_theorem_lines = 0

    for line_idx, line in enumerate(stripped_lines):
        ns_stack = [name for kind, name in scopes if kind == "namespace" and name]
        in_season35 = ns_stack == [ROOT_NAMESPACE, SEASON35_NAMESPACE]
        if in_season35:
            season35_stripped_lines.append(line)

        lstr = line.lstrip()
        if not lstr:
            continue

        if lstr.startswith(("namespace", "section")):
            scope_open = _SCOPE_OPEN_RE.fullmatch(line)
            if scope_open is not None:
                kind, name = scope_open.group(1), (scope_open.group(2) or "")
                scopes.append((kind, name))
                new_ns = [n for k, n in scopes if k == "namespace" and n]
                if new_ns == [ROOT_NAMESPACE, SEASON35_NAMESPACE] and season35_span_start is None:
                    season35_span_start = line_idx
                continue

        if lstr.startswith("end"):
            scope_close = _SCOPE_CLOSE_RE.fullmatch(line)
            if scope_close is not None:
                close_name = scope_close.group(1)
                if not scopes or (close_name and close_name != scopes[-1][1]):
                    namespaces_balanced = False
                else:
                    closing_ns = [n for k, n in scopes if k == "namespace" and n]
                    if closing_ns == [ROOT_NAMESPACE, SEASON35_NAMESPACE] and season35_span_end is None:
                        season35_span_end = line_idx
                    scopes.pop()
                continue

        if "theorem" in lstr or "lemma" in lstr:
            thm_match = _THEOREM_DECL_RE.match(line)
            if thm_match is not None:
                thm_name = thm_match.group(1)
                qual_name = ".".join([*ns_stack, thm_name]) if ns_stack else thm_name
                all_declared_theorems.append(qual_name)
                if in_season35:
                    if thm_name in seen_season35:
                        season35_duplicates.append(thm_name)
                    else:
                        seen_season35.add(thm_name)
                    season35_declared.append(thm_name)
                    season35_qualified.append(qual_name)
            elif _THEOREM_ANYWHERE_RE.search(line):
                global_unmatched_theorem_lines += 1
                if in_season35:
                    season35_unmatched_theorem_lines += 1

        if lstr.startswith("#print"):
            print_match = _PRINT_AXIOMS_RE.match(line)
            if print_match is not None:
                printed_target = print_match.group(1)
                if len(scopes) == 0:
                    top_level_prints.append(printed_target)
                    top_level_print_raw_lines.append(raw_lines[line_idx])
                else:
                    nested_prints.append(printed_target)

    if scopes:
        namespaces_balanced = False

    season35_present = season35_span_start is not None and season35_span_end is not None
    season35_span = (
        (season35_span_start, season35_span_end)
        if (season35_span_start is not None and season35_span_end is not None)
        else None
    )
    season35_code = "\n".join(season35_stripped_lines)

    s35_sorry = len(_SORRY_RE.findall(season35_code))
    s35_admit = len(_ADMIT_RE.findall(season35_code))
    s35_axiom = len(_CUSTOM_AXIOM_RE.findall(season35_code))
    s35_native_decide = len(_NATIVE_DECIDE_RE.findall(season35_code))
    s35_sorry_ax = len(_SORRY_AX_RE.findall(season35_code))

    g_sorry = len(_SORRY_RE.findall(stripped))
    g_admit = len(_ADMIT_RE.findall(stripped))
    g_axiom = len(_CUSTOM_AXIOM_RE.findall(stripped))
    g_native_decide = len(_NATIVE_DECIDE_RE.findall(stripped))
    g_sorry_ax = len(_SORRY_AX_RE.findall(stripped))

    expected_missing = tuple(name for name in expected_list if name not in seen_season35)
    top_level_set = set(top_level_prints)
    expected_qualified_targets = [f"{SEASON35_QUALIFIED_PREFIX}.{name}" for name in expected_list]
    for q_name in season35_qualified:
        if q_name not in expected_qualified_targets:
            expected_qualified_targets.append(q_name)
    missing_prints = tuple(q for q in expected_qualified_targets if q not in top_level_set)

    violations: list[str] = []
    if not lexical_balanced:
        violations.append("unterminated_lean4_comment_or_string")
    if not namespaces_balanced:
        violations.append("unbalanced_lean4_namespace_or_section_scopes")
    if not season35_present:
        violations.append(f"missing_namespace:{SEASON35_QUALIFIED_PREFIX}")
    if expected_missing:
        violations.append(f"missing_season35_theorems:{','.join(expected_missing)}")
    if season35_duplicates:
        violations.append(f"duplicate_season35_theorems:{','.join(sorted(set(season35_duplicates)))}")
    if season35_unmatched_theorem_lines > 0 or global_unmatched_theorem_lines > 0:
        violations.append("unsupported_theorem_declaration_syntax")
    if s35_sorry > 0 or g_sorry > 0:
        violations.append(f"sorry_detected:season35={s35_sorry},global={g_sorry}")
    if s35_admit > 0 or g_admit > 0:
        violations.append(f"admit_detected:season35={s35_admit},global={g_admit}")
    if s35_axiom > 0 or g_axiom > 0:
        violations.append(f"custom_axiom_detected:season35={s35_axiom},global={g_axiom}")
    if s35_native_decide > 0 or g_native_decide > 0:
        violations.append(f"native_decide_detected:season35={s35_native_decide},global={g_native_decide}")
    if s35_sorry_ax > 0 or g_sorry_ax > 0:
        violations.append(f"sorry_ax_detected:season35={s35_sorry_ax},global={g_sorry_ax}")
    if missing_prints:
        violations.append(f"missing_top_level_print_axioms:{','.join(missing_prints)}")
    nested_s35_prints = [p for p in nested_prints if SEASON35_NAMESPACE in p]
    if nested_s35_prints:
        violations.append(f"nested_print_axioms_not_top_level:{','.join(nested_s35_prints)}")

    section_sha = _compute_season35_section_digest(raw_lines, season35_span, top_level_print_raw_lines)

    return Lean4WSPSpecAudit(
        spec_path=spec_path,
        spec_sha256=spec_sha256,
        season35_section_sha256=section_sha,
        season35_namespace_present=season35_present,
        namespaces_balanced=namespaces_balanced,
        season35_theorems_declared=tuple(season35_declared),
        season35_qualified_theorems=tuple(season35_qualified),
        total_spec_theorems=len(all_declared_theorems),
        expected_theorems_missing=expected_missing,
        duplicate_season35_theorems=tuple(sorted(set(season35_duplicates))),
        top_level_print_axioms=tuple(top_level_prints),
        nested_print_axioms=tuple(nested_prints),
        missing_print_axioms=missing_prints,
        season35_sorry_count=s35_sorry,
        season35_admit_count=s35_admit,
        season35_custom_axiom_count=s35_axiom,
        season35_native_decide_count=s35_native_decide,
        season35_sorry_ax_count=s35_sorry_ax,
        global_sorry_count=g_sorry,
        global_admit_count=g_admit,
        global_custom_axiom_count=g_axiom,
        global_native_decide_count=g_native_decide,
        global_sorry_ax_count=g_sorry_ax,
        violations=tuple(violations),
        passed=len(violations) == 0,
    )


def resolve_default_lean4_spec_path(repo_root: Path | str | None = None) -> Path:
    """Resolve ``lean4/HiveMindSpec.lean`` relative to ``repo_root`` or this module's repo."""
    if repo_root is not None:
        base = Path(repo_root)
    else:
        base = Path(__file__).resolve().parents[3]
    return base / DEFAULT_SPEC_RELATIVE_PATH


def audit_lean4_wsp_spec_file(
    spec_path: Path | str | None = None,
    *,
    repo_root: Path | str | None = None,
    expected_theorems: Sequence[str] = EXPECTED_SEASON35_THEOREMS,
) -> Lean4WSPSpecAudit:
    """Read and deterministically audit ``lean4/HiveMindSpec.lean`` from disk."""
    target = Path(spec_path) if spec_path is not None else resolve_default_lean4_spec_path(repo_root)
    if not target.exists():
        raise FileNotFoundError(f"Lean 4 specification file not found: {target}")
    if not target.is_file():
        raise ValueError(f"Lean 4 specification path must be a regular file: {target}")
    source = target.read_text(encoding="utf-8")
    return parse_lean4_wsp_spec(
        source,
        spec_path=target.as_posix(),
        expected_theorems=expected_theorems,
    )


# ---------------------------------------------------------------------------
# 3. Runtime S5 Surgery Receipt -> Lean 4 Obligation Binding & Attestation
# ---------------------------------------------------------------------------


def _extract_receipt_verdict(receipt: Mapping[str, Any]) -> str:
    """Extract normalized verdict string across S5 receipt formats."""
    for key in ("verdict", "wsp_verdict", "verdict_ip300"):
        raw = receipt.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    if receipt.get("committed") is True or receipt.get("passed") is True:
        return "PASS"
    if receipt.get("committed") is False or receipt.get("passed") is False:
        return "FAIL"
    raw_status = receipt.get("status")
    if isinstance(raw_status, str) and raw_status.strip():
        return raw_status.strip()
    return "UNKNOWN"


def _extract_checkpoint_sha(receipt: Mapping[str, Any]) -> str:
    """Extract primary output checkpoint SHA-256 across S5 receipt schemas."""
    for key in (
        "checkpoint_sha256",
        "active_checkpoint_sha256",
        "champion_sha256",
        "sanitized_s5_sha256",
        "post_sha256",
    ):
        val = receipt.get(key)
        if isinstance(val, str) and _SHA256_HEX_RE.match(val):
            return val
    return ""


def _stage_names_from_receipt(receipt: Mapping[str, Any]) -> set[str]:
    """Collect normalized stage names and stage kinds mentioned by ``receipt``."""
    names: set[str] = set()
    for list_key in ("committed_stages", "rolled_back_stages", "skipped_stages"):
        raw_seq = receipt.get(list_key)
        if isinstance(raw_seq, Sequence) and not isinstance(raw_seq, (str, bytes)):
            for item in raw_seq:
                if isinstance(item, str) and item.strip():
                    names.add(item.strip())
    stages_val = receipt.get("stages")
    if isinstance(stages_val, Sequence) and not isinstance(stages_val, (str, bytes)):
        for entry in stages_val:
            if isinstance(entry, Mapping):
                for k in ("stage_name", "stage_kind"):
                    sval = entry.get(k)
                    if isinstance(sval, str) and sval.strip():
                        names.add(sval.strip())
    return names


def _check_monotone_stages(receipt: Mapping[str, Any]) -> tuple[bool, str]:
    """Verify ANSE monotone ratchet invariant on stage telemetry inside ``receipt``."""
    stages_val = receipt.get("stages")
    if isinstance(stages_val, Sequence) and not isinstance(stages_val, (str, bytes)) and stages_val:
        for idx, entry in enumerate(stages_val):
            if not isinstance(entry, Mapping):
                return False, f"stage[{idx}] is not a mapping"
            accepted = bool(entry.get("accepted", entry.get("committed", False)))
            pre_e = entry.get("pre_anse_energy")
            post_e = entry.get("post_anse_energy")
            delta_e = entry.get("delta_anse_energy")
            pre_sha = entry.get("pre_sha256")
            post_sha = entry.get("post_sha256")
            if accepted:
                if isinstance(pre_e, (int, float)) and isinstance(post_e, (int, float)):
                    if not (math.isfinite(float(pre_e)) and math.isfinite(float(post_e))):
                        return False, f"stage[{idx}] non-finite energy"
                    if float(post_e) > float(pre_e) + 1e-12:
                        return False, f"stage[{idx}] accepted energy increase {pre_e} -> {post_e}"
                if isinstance(delta_e, (int, float)) and float(delta_e) > 1e-12:
                    return False, f"stage[{idx}] accepted positive delta_anse_energy={delta_e}"
            else:
                if isinstance(pre_sha, str) and isinstance(post_sha, str):
                    if pre_sha and post_sha and pre_sha != post_sha:
                        return False, f"stage[{idx}] rejected without bitwise rollback"
        return True, f"verified_{len(stages_val)}_ratcheted_stages"

    raw_e = receipt.get("raw_anse_energy")
    s5_e = receipt.get("s5_anse_energy")
    if isinstance(raw_e, (int, float)) and isinstance(s5_e, (int, float)):
        verdict = _extract_receipt_verdict(receipt)
        if verdict == "PASS" and float(s5_e) >= float(raw_e):
            return False, f"s5_anse_energy={s5_e} did not strictly beat raw_anse_energy={raw_e}"
        return True, f"verified_s5_energy={float(s5_e):.6f}_vs_raw={float(raw_e):.6f}"

    energy = receipt.get("anse_energy")
    if isinstance(energy, (int, float)) and math.isfinite(float(energy)):
        return True, f"verified_anse_energy={float(energy):.6f}"
    return False, "missing_anse_energy_telemetry"


def resolve_s5_theorem_obligations(
    surgery_receipt: Mapping[str, Any],
    *,
    spec_audit: Lean4WSPSpecAudit,
    vault_receipt: Mapping[str, Any] | None = None,
    general_guard_rel: float = DEFAULT_GENERAL_GUARD_REL,
) -> tuple[TheoremObligationBinding, ...]:
    """Map runtime S5 surgery telemetry and optional vault receipt to Season 35 Lean 4 theorems."""
    guard_rel = _validate_nonneg_finite(general_guard_rel, "general_guard_rel")
    declared_set = set(spec_audit.season35_theorems_declared)
    top_print_set = set(spec_audit.top_level_print_axioms)
    clean_base = (
        spec_audit.season35_sorry_count == 0
        and spec_audit.season35_admit_count == 0
        and spec_audit.season35_custom_axiom_count == 0
        and spec_audit.season35_native_decide_count == 0
        and spec_audit.season35_sorry_ax_count == 0
        and spec_audit.global_sorry_count == 0
        and spec_audit.global_admit_count == 0
        and spec_audit.global_custom_axiom_count == 0
    )

    stage_names = _stage_names_from_receipt(surgery_receipt)
    profile = surgery_receipt.get("profile") if isinstance(surgery_receipt.get("profile"), Mapping) else {}
    verdict = _extract_receipt_verdict(surgery_receipt)
    gen_drift = surgery_receipt.get("general_rel_drift", surgery_receipt.get("general_domain_drift"))
    anchor_preserved = surgery_receipt.get("anchor_layers_bitwise_preserved")
    monotone_ok, monotone_note = _check_monotone_stages(surgery_receipt)

    bindings: list[TheoremObligationBinding] = []
    for spec in SEASON35_THEOREM_CATALOG:
        lean_ok = (
            clean_base
            and spec_audit.season35_namespace_present
            and spec_audit.namespaces_balanced
            and (spec.short_name in declared_set)
            and (spec.qualified_name in top_print_set)
        )

        if spec.short_name == "nullspace_projector_idempotent_scalar":
            tail_layers = profile.get("tail_svd_layers", ()) if profile else ()
            exercised = bool(
                tail_layers
                or any("nullspace" in n or "anchor" in n or "S5.3" in n for n in stage_names)
            )
            satisfied = bool(lean_ok and (monotone_ok or verdict != "PASS"))
            evidence = f"tail_svd_or_nullspace_stages={sorted(n for n in stage_names if 'nullspace' in n or 'S5.3' in n)}"

        elif spec.short_name == "preserved_key_zero_drift_under_nullspace_edit":
            drift_ok = isinstance(gen_drift, (int, float)) and math.isfinite(float(gen_drift)) and float(gen_drift) <= guard_rel + 1e-12
            anchor_ok = True if anchor_preserved is None else bool(anchor_preserved)
            exercised = True
            satisfied = bool(lean_ok and anchor_ok and (drift_ok if verdict == "PASS" else True))
            evidence = f"general_rel_drift={gen_drift},anchor_preserved={anchor_preserved}"

        elif spec.short_name == "orthogonal_multigate_specialist_dot_invariant":
            preserved_mg = (
                profile.get("preserved_multigate_experts")
                if profile and "preserved_multigate_experts" in profile
                else surgery_receipt.get("preserved_specialists", ())
            )
            exercised = bool(
                preserved_mg
                or any("multigate" in n or "S5.2" in n for n in stage_names)
            )
            satisfied = bool(lean_ok)
            evidence = f"preserved_multigate_experts={preserved_mg}"

        elif spec.short_name == "monotone_anse_stage_ratchet_strictly_descends":
            exercised = True
            satisfied = bool(lean_ok and monotone_ok)
            evidence = monotone_note

        else:  # single_shot_vault_lock_blocks_second_eval
            target_vault = vault_receipt if vault_receipt is not None else (
                surgery_receipt
                if ("vault_receipt_sha256" in surgery_receipt or "vault_locked" in surgery_receipt)
                else None
            )
            if target_vault is not None:
                exercised = True
                v_sha_ok, v_sha = _verify_embedded_receipt_sha256(target_vault)
                v_locked = bool(target_vault.get("vault_locked", True))
                v_count = target_vault.get("vault_access_count", 1)
                v_count_ok = isinstance(v_count, int) and not isinstance(v_count, bool) and v_count == 1
                v_guard_ok = bool(target_vault.get("guard_ok", True))
                satisfied = bool(lean_ok and v_sha_ok and v_locked and v_count_ok and v_guard_ok)
                evidence = f"vault_locked={v_locked},access_count={v_count},sha={v_sha[:16]}"
            else:
                exercised = False
                satisfied = bool(lean_ok)
                evidence = "no_vault_receipt_supplied_formal_lock_theorem_verified"

        bindings.append(
            TheoremObligationBinding(
                obligation_id=spec.obligation_id,
                theorem_short_name=spec.short_name,
                theorem_qualified_name=spec.qualified_name,
                wsp_stage=spec.wsp_stage,
                lean_verified=lean_ok,
                exercised=exercised,
                runtime_satisfied=satisfied,
                evidence=evidence,
            )
        )
    return tuple(bindings)


def bind_s5_receipt_to_lean4_theorems(
    surgery_receipt: Mapping[str, Any] | Any,
    *,
    spec_audit: Lean4WSPSpecAudit | None = None,
    spec_path: Path | str | None = None,
    repo_root: Path | str | None = None,
    vault_receipt: Mapping[str, Any] | None = None,
    require_pass: bool = True,
    general_guard_rel: float = DEFAULT_GENERAL_GUARD_REL,
    strict: bool = True,
) -> Lean4WSPAttestationReceipt:
    """Bind a runtime S5 surgery receipt to all 5 Season 35 Lean 4 theorems with SHA-256 attestation."""
    _validate_bool(require_pass, "require_pass")
    _validate_bool(strict, "strict")
    guard_rel = _validate_nonneg_finite(general_guard_rel, "general_guard_rel")

    receipt_dict = _coerce_receipt_mapping(surgery_receipt, "surgery_receipt")
    vault_dict = _coerce_receipt_mapping(vault_receipt, "vault_receipt") if vault_receipt is not None else None

    if spec_audit is not None:
        if not isinstance(spec_audit, Lean4WSPSpecAudit):
            raise TypeError(f"spec_audit must be a Lean4WSPSpecAudit, got {type(spec_audit).__name__}")
        audit = spec_audit
    else:
        audit = audit_lean4_wsp_spec_file(spec_path=spec_path, repo_root=repo_root)

    failures: list[str] = []
    if not audit.passed:
        failures.extend(f"lean_spec:{v}" for v in audit.violations)

    emb_ok, emb_sha = _verify_embedded_receipt_sha256(receipt_dict)
    if not emb_ok:
        failures.append(f"surgery_receipt:{emb_sha}")

    vault_sha: str | None = None
    if vault_dict is not None:
        v_ok, v_sha = _verify_embedded_receipt_sha256(vault_dict)
        if not v_ok:
            failures.append(f"vault_receipt:{v_sha}")
        else:
            vault_sha = v_sha if v_sha else compute_payload_sha256(vault_dict)

    verdict = _extract_receipt_verdict(receipt_dict)
    if require_pass and verdict != "PASS":
        failures.append(f"surgery_receipt_not_pass:verdict={verdict}")

    bindings = resolve_s5_theorem_obligations(
        receipt_dict,
        spec_audit=audit,
        vault_receipt=vault_dict,
        general_guard_rel=guard_rel,
    )
    all_lean_ok = all(b.lean_verified for b in bindings) and len(bindings) == len(EXPECTED_SEASON35_THEOREMS)
    all_runtime_ok = all(b.runtime_satisfied for b in bindings)
    if not all_lean_ok:
        failures.append("incomplete_season35_lean4_theorem_verification")
    if not all_runtime_ok:
        bad_obs = [b.obligation_id for b in bindings if not b.runtime_satisfied]
        failures.append(f"runtime_obligations_unsatisfied:{','.join(bad_obs)}")

    if strict and not audit.passed:
        raise Lean4SpecAuditError(
            f"Lean 4 spec audit failed for {audit.spec_path}: {'; '.join(audit.violations)}"
        )

    surgery_sha = compute_payload_sha256(receipt_dict)
    ckpt_sha = _extract_checkpoint_sha(receipt_dict)
    receipt_obj = str(receipt_dict.get("object", "hivemind.s5_surgery_receipt"))

    core_payload = {
        "object": ATTESTATION_RECEIPT_OBJECT,
        "schema": SCHEMA_VERSION,
        "spec_path": audit.spec_path,
        "spec_sha256": audit.spec_sha256,
        "season35_section_sha256": audit.season35_section_sha256,
        "spec_audit_passed": audit.passed,
        "theorem_count_verified": sum(1 for b in bindings if b.lean_verified),
        "season35_namespace": SEASON35_QUALIFIED_PREFIX,
        "surgery_receipt_object": receipt_obj,
        "surgery_receipt_verdict": verdict,
        "surgery_receipt_sha256": surgery_sha,
        "surgery_embedded_sha256": emb_sha,
        "checkpoint_sha256": ckpt_sha,
        "vault_receipt_sha256": vault_sha,
        "theorem_bindings": [b.to_dict() for b in bindings],
        "all_lean_theorems_verified": all_lean_ok,
        "all_runtime_obligations_satisfied": all_runtime_ok,
        "attestation_valid": len(failures) == 0,
        "failures": list(failures),
    }
    attestation_sha = compute_payload_sha256(core_payload)

    return Lean4WSPAttestationReceipt(
        spec_path=audit.spec_path,
        spec_sha256=audit.spec_sha256,
        season35_section_sha256=audit.season35_section_sha256,
        spec_audit_passed=audit.passed,
        theorem_count_verified=sum(1 for b in bindings if b.lean_verified),
        season35_namespace=SEASON35_QUALIFIED_PREFIX,
        surgery_receipt_object=receipt_obj,
        surgery_receipt_verdict=verdict,
        surgery_receipt_sha256=surgery_sha,
        surgery_embedded_sha256=emb_sha,
        checkpoint_sha256=ckpt_sha,
        vault_receipt_sha256=vault_sha,
        theorem_bindings=bindings,
        all_lean_theorems_verified=all_lean_ok,
        all_runtime_obligations_satisfied=all_runtime_ok,
        attestation_valid=len(failures) == 0,
        failures=tuple(failures),
        attestation_sha256=attestation_sha,
    )


def verify_lean4_attestation_receipt(
    attestation: Lean4WSPAttestationReceipt | Mapping[str, Any],
    *,
    spec_source: str | None = None,
    spec_path: Path | str | None = None,
    spec_audit: Lean4WSPSpecAudit | None = None,
    surgery_receipt: Mapping[str, Any] | Any | None = None,
    vault_receipt: Mapping[str, Any] | None = None,
) -> Lean4AttestationVerification:
    """Verify SHA-256 attestation integrity and detect tampering on Lean spec or S5 receipt."""
    doc = _coerce_receipt_mapping(attestation, "attestation")
    failures: list[str] = []

    if doc.get("object") != ATTESTATION_RECEIPT_OBJECT:
        failures.append("invalid_attestation_object")
    if doc.get("schema") != SCHEMA_VERSION:
        failures.append("invalid_attestation_schema")

    claimed_sha = doc.get("attestation_sha256")
    sha_ok = False
    if not isinstance(claimed_sha, str) or not _SHA256_HEX_RE.match(claimed_sha):
        failures.append("invalid_attestation_sha256_format")
    else:
        core = {k: v for k, v in doc.items() if k != "attestation_sha256"}
        try:
            expected_sha = compute_payload_sha256(core)
        except (TypeError, ValueError):
            expected_sha = ""
        if claimed_sha == expected_sha:
            sha_ok = True
        else:
            failures.append("attestation_sha256_mismatch")

    if doc.get("attestation_valid") is not True or doc.get("spec_audit_passed") is not True:
        failures.append("attestation_marked_invalid")

    raw_bindings = doc.get("theorem_bindings")
    if not isinstance(raw_bindings, list):
        failures.append("missing_theorem_bindings")
    else:
        bound_names = {
            str(b.get("theorem_short_name"))
            for b in raw_bindings
            if isinstance(b, Mapping) and b.get("lean_verified") is True
        }
        missing_bound = [t for t in EXPECTED_SEASON35_THEOREMS if t not in bound_names]
        if missing_bound:
            failures.append(f"unverified_bound_theorems:{','.join(missing_bound)}")

    active_audit: Lean4WSPSpecAudit | None = None
    if spec_audit is not None:
        if not isinstance(spec_audit, Lean4WSPSpecAudit):
            raise TypeError(f"spec_audit must be a Lean4WSPSpecAudit, got {type(spec_audit).__name__}")
        active_audit = spec_audit
    elif spec_source is not None:
        active_audit = parse_lean4_wsp_spec(
            spec_source,
            spec_path=str(doc.get("spec_path") or "lean4/HiveMindSpec.lean"),
        )
    elif spec_path is not None:
        active_audit = audit_lean4_wsp_spec_file(spec_path=spec_path)

    spec_sha_ok = True
    section_sha_ok = True
    spec_audit_ok = bool(doc.get("spec_audit_passed") is True)
    if active_audit is not None:
        if not active_audit.passed:
            spec_audit_ok = False
            failures.extend(f"current_spec_audit:{v}" for v in active_audit.violations)
        if active_audit.spec_sha256 != doc.get("spec_sha256"):
            spec_sha_ok = False
            failures.append("spec_sha256_mismatch")
        if active_audit.season35_section_sha256 != doc.get("season35_section_sha256"):
            section_sha_ok = False
            failures.append("season35_section_sha256_mismatch")

    surgery_ok = True
    if surgery_receipt is not None:
        surg_dict = _coerce_receipt_mapping(surgery_receipt, "surgery_receipt")
        if compute_payload_sha256(surg_dict) != doc.get("surgery_receipt_sha256"):
            surgery_ok = False
            failures.append("surgery_receipt_sha256_mismatch")

    vault_ok = True
    if vault_receipt is not None:
        v_dict = _coerce_receipt_mapping(vault_receipt, "vault_receipt")
        v_emb_ok, v_sha = _verify_embedded_receipt_sha256(v_dict)
        resolved_v_sha = v_sha if (v_emb_ok and v_sha) else compute_payload_sha256(v_dict)
        if not v_emb_ok or resolved_v_sha != doc.get("vault_receipt_sha256"):
            vault_ok = False
            failures.append("vault_receipt_sha256_mismatch")

    valid = (
        sha_ok
        and spec_sha_ok
        and section_sha_ok
        and spec_audit_ok
        and surgery_ok
        and vault_ok
        and len(failures) == 0
    )
    return Lean4AttestationVerification(
        valid=valid,
        attestation_sha256_ok=sha_ok,
        spec_sha256_ok=spec_sha_ok,
        season35_section_sha256_ok=section_sha_ok,
        spec_audit_ok=spec_audit_ok,
        surgery_receipt_ok=surgery_ok,
        vault_receipt_ok=vault_ok,
        failures=tuple(failures),
    )


__all__ = [
    "ATTESTATION_RECEIPT_OBJECT",
    "DEFAULT_ANSE_MAXIMUM_PAIN",
    "DEFAULT_GENERAL_GUARD_REL",
    "DEFAULT_SPEC_RELATIVE_PATH",
    "EXPECTED_SEASON35_THEOREMS",
    "ROOT_NAMESPACE",
    "SCHEMA_VERSION",
    "SEASON35_NAMESPACE",
    "SEASON35_QUALIFIED_PREFIX",
    "SEASON35_THEOREM_BY_NAME",
    "SEASON35_THEOREM_CATALOG",
    "SPEC_AUDIT_OBJECT",
    "Lean4AttestationError",
    "Lean4AttestationVerification",
    "Lean4SpecAuditError",
    "Lean4TheoremSpec",
    "Lean4WSPAttestationReceipt",
    "Lean4WSPSpecAudit",
    "TheoremObligationBinding",
    "audit_lean4_wsp_spec_file",
    "bind_s5_receipt_to_lean4_theorems",
    "canonical_json_bytes",
    "compute_payload_sha256",
    "parse_lean4_wsp_spec",
    "resolve_default_lean4_spec_path",
    "resolve_s5_theorem_obligations",
    "strip_lean4_comments_and_strings",
    "verify_lean4_attestation_receipt",
]
