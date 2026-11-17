# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S35-03: Elenchus x SocrateAI-Lean-Lib x rusty-SUNDIALS Tier-X Provenance (R-64) Verifier.

Integrates:
1. elenchus-solvers-mcp (xaviercallens/elenchus-solvers-mcp) & SocrateAI-Scientific-Elenchus:
   - Tier-X instrument contract (`tier: "X"`, `claim_licence: false`, `ran: bool`, `stub: bool`).
   - Finding #4 from elenchus-solvers-mcp README: raw grep counts docstrings like
     `**Kernel Certified:** 0 sorry, 0 admit.` — Elenchus strips comments (`-- ...` and `/- ... -/`)
     before scanning for `sorry`, `admit`, and `native_decide`.
   - Rule R-64 Provenance: binds producer, version, tool, exact arguments as received, toolchain,
     and git revision into a deterministic SHA-256 receipt; refuses any `stub: true` or `ran: false`.
2. SocrateAI-Lean-Lib (xaviercallens/SocrateAI-Lean-Lib):
   - 14-domain Lean 4 formal library with explicit axiom quarantine (`PROVEN_INVARIANT`,
     `QUARANTINED_AXIOM`, `UNVERIFIED_SORRY`).
3. rusty-SUNDIALS (xaviercallens/rusty-SUNDIALS):
   - Robertson chemical kinetics stiff ODE conservation check (y1 + y2 + y3 = 1, |Delta m| < 1e-12).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class LeanCommentStrippedAudit:
    """Result of Elenchus comment-stripped Lean 4 source inspection."""

    raw_sorry_substring_hits: int
    comment_stripped_sorry_count: int
    comment_stripped_admit_count: int
    comment_stripped_native_decide_count: int
    quarantined_axiom_count: int
    theorem_count: int
    classification: str  # "PROVEN_INVARIANT" | "QUARANTINED_AXIOM" | "UNVERIFIED_SORRY"
    is_kernel_clean: bool

    def __post_init__(self) -> None:
        for i_field in (
            "raw_sorry_substring_hits",
            "comment_stripped_sorry_count",
            "comment_stripped_admit_count",
            "comment_stripped_native_decide_count",
            "quarantined_axiom_count",
            "theorem_count",
        ):
            val = getattr(self, i_field)
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                raise ValueError(f"{i_field} must be a non-negative integer")
        if self.classification not in ("PROVEN_INVARIANT", "QUARANTINED_AXIOM", "UNVERIFIED_SORRY"):
            raise ValueError("classification must be 'PROVEN_INVARIANT', 'QUARANTINED_AXIOM', or 'UNVERIFIED_SORRY'")
        if not isinstance(self.is_kernel_clean, bool):
            raise ValueError("is_kernel_clean must be a boolean")


@dataclass(frozen=True)
class TierXR64InstrumentReceipt:
    """R-64 provenance-bound Tier-X instrument receipt from elenchus-solvers-mcp."""

    tier: str
    claim_licence: bool
    ran: bool
    stub: bool
    evidentiary_value: str  # "tier_x_measured" | "none"
    accepted_by_elenchus_bridge: bool
    mass_conservation_drift: float
    provenance_sha256: str
    simulated: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.tier, str) or not self.tier.strip():
            raise ValueError("tier must be a non-empty string")
        for b_field in ("claim_licence", "ran", "stub", "accepted_by_elenchus_bridge", "simulated"):
            if not isinstance(getattr(self, b_field), bool):
                raise ValueError(f"{b_field} must be a boolean")
        if self.evidentiary_value not in ("tier_x_measured", "none"):
            raise ValueError("evidentiary_value must be 'tier_x_measured' or 'none'")
        if (
            isinstance(self.mass_conservation_drift, bool)
            or not isinstance(self.mass_conservation_drift, (int, float))
            or math.isnan(float(self.mass_conservation_drift))
            or float(self.mass_conservation_drift) < 0.0
        ):
            raise ValueError("mass_conservation_drift must be a non-negative number")
        if not isinstance(self.provenance_sha256, str) or len(self.provenance_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in self.provenance_sha256):
            raise ValueError("provenance_sha256 must be a 64-character hex string")


def strip_lean4_comments(lean_source: str) -> str:
    """Strip nested/block `/- ... -/` comments and single-line `-- ...` comments from Lean 4 code."""
    if not isinstance(lean_source, str) or "\x00" in lean_source:
        raise ValueError("lean_source must be a valid string without NUL bytes")
    # Iteratively remove innermost /- ... -/ blocks to handle nested block comments
    cleaned = lean_source
    pattern_block = re.compile(r"/-(?:(?!/\-|\-/).)*?-/", re.DOTALL)
    while True:
        next_cleaned, n = pattern_block.subn(" ", cleaned)
        cleaned = next_cleaned
        if n == 0:
            break
    # Remove single-line comments -- ...
    cleaned = re.sub(r"--[^\n]*", " ", cleaned)
    return cleaned


def audit_lean4_source_elenchus(lean_source: str) -> LeanCommentStrippedAudit:
    """Audit Lean 4 source after stripping comments (avoiding false positives/negatives in docstrings)."""
    if not isinstance(lean_source, str) or "\x00" in lean_source:
        raise ValueError("lean_source must be a valid string without NUL bytes")
    raw_hits = len(re.findall(r"\bsorry\b", lean_source))
    stripped = strip_lean4_comments(lean_source)

    sorry_cnt = len(re.findall(r"\bsorry\b", stripped))
    admit_cnt = len(re.findall(r"\badmit\b", stripped))
    native_decide_cnt = len(re.findall(r"\bnative_decide\b", stripped))
    axiom_cnt = len(re.findall(r"\baxiom\s+\w+", stripped))
    thm_cnt = len(re.findall(r"\b(?:theorem|lemma)\s+\w+", stripped))

    has_sorry = (sorry_cnt > 0) or (admit_cnt > 0) or (native_decide_cnt > 0)
    is_clean = (not has_sorry) and (axiom_cnt == 0)
    if has_sorry:
        classification = "UNVERIFIED_SORRY"
    elif axiom_cnt > 0:
        classification = "QUARANTINED_AXIOM"
    else:
        classification = "PROVEN_INVARIANT"

    return LeanCommentStrippedAudit(
        raw_sorry_substring_hits=raw_hits,
        comment_stripped_sorry_count=sorry_cnt,
        comment_stripped_admit_count=admit_cnt,
        comment_stripped_native_decide_count=native_decide_cnt,
        quarantined_axiom_count=axiom_cnt,
        theorem_count=thm_cnt,
        classification=classification,
        is_kernel_clean=is_clean,
    )


def _validate_triple(vec: object, name: str) -> tuple[float, float, float]:
    if not isinstance(vec, (list, tuple)) or len(vec) != 3:
        raise ValueError(f"{name} must be a 3-element tuple or list of numbers")
    out: list[float] = []
    for idx, v in enumerate(vec):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{name}[{idx}] must be a real number, not bool or {type(v).__name__}")
        out.append(float(v))
    return (out[0], out[1], out[2])


def verify_robertson_sundials_tier_x(
    sent_arguments: Mapping[str, Any],
    received_arguments: Mapping[str, Any],
    y_initial: tuple[float, float, float] = (1.0, 0.0, 0.0),
    y_final: tuple[float, float, float] | None = None,
    ran: bool = True,
    stub: bool = False,
    producer: str = "elenchus-solvers-mcp/rusty-SUNDIALS",
    git_rev: str = "bsd3-cvode-r64-verified",
) -> TierXR64InstrumentReceipt:
    """Verify a rusty-SUNDIALS Robertson stiff ODE reading under Elenchus Tier-X and R-64 rules."""
    if not isinstance(sent_arguments, Mapping) or not isinstance(received_arguments, Mapping):
        raise ValueError("sent_arguments and received_arguments must be mappings")
    if not isinstance(ran, bool) or not isinstance(stub, bool):
        raise ValueError("ran and stub must be booleans")
    if not isinstance(producer, str) or not producer.strip() or any(ord(c) < 32 or ord(c) == 127 for c in producer):
        raise ValueError("producer must be a non-empty string without control characters")
    if not isinstance(git_rev, str) or not git_rev.strip() or any(ord(c) < 32 or ord(c) == 127 for c in git_rev):
        raise ValueError("git_rev must be a non-empty string without control characters")

    clean_y_init = _validate_triple(y_initial, "y_initial")
    args_match = dict(sent_arguments) == dict(received_arguments)
    effective_y_final: tuple[float, float, float] | None = (
        _validate_triple(y_final, "y_final") if y_final is not None else None
    )
    if effective_y_final is None and "y_final" in received_arguments:
        raw_yf = received_arguments["y_final"]
        if (
            isinstance(raw_yf, (list, tuple))
            and len(raw_yf) == 3
            and all(not isinstance(x, bool) and isinstance(x, (int, float)) for x in raw_yf)
        ):
            effective_y_final = (float(raw_yf[0]), float(raw_yf[1]), float(raw_yf[2]))

    has_valid_output = (
        effective_y_final is not None
        and len(effective_y_final) == 3
        and all(math.isfinite(v) for v in effective_y_final)
        and all(math.isfinite(v) for v in clean_y_init)
    )
    if has_valid_output and effective_y_final is not None:
        mass_init = sum(clean_y_init)
        mass_final = sum(effective_y_final)
        mass_drift = abs(mass_final - mass_init)
    else:
        mass_drift = float("inf")

    payload = json.dumps(
        {
            "producer": producer,
            "git_rev": git_rev,
            "received_arguments": dict(received_arguments),
            "ran": ran,
            "stub": stub,
            "has_solver_output": has_valid_output,
            "mass_drift": round(mass_drift, 16) if math.isfinite(mass_drift) else None,
        },
        sort_keys=True,
    )
    prov_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    if stub or not ran or not args_match or not has_valid_output or mass_drift > 1e-12:
        evidentiary = "none"
        accepted = False
    else:
        evidentiary = "tier_x_measured"
        accepted = True

    return TierXR64InstrumentReceipt(
        tier="X",
        claim_licence=False,
        ran=ran,
        stub=stub,
        evidentiary_value=evidentiary,
        accepted_by_elenchus_bridge=accepted,
        mass_conservation_drift=mass_drift,
        provenance_sha256=prov_hash,
    )
