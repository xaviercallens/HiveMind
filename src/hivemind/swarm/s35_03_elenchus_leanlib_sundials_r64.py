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


def strip_lean4_comments(lean_source: str) -> str:
    """Strip nested/block `/- ... -/` comments and single-line `-- ...` comments from Lean 4 code."""
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
    args_match = dict(sent_arguments) == dict(received_arguments)
    effective_y_final = y_final
    if effective_y_final is None and "y_final" in received_arguments:
        raw_yf = received_arguments["y_final"]
        if isinstance(raw_yf, (list, tuple)) and len(raw_yf) == 3:
            effective_y_final = (float(raw_yf[0]), float(raw_yf[1]), float(raw_yf[2]))

    has_valid_output = (
        effective_y_final is not None
        and len(effective_y_final) == 3
        and all(math.isfinite(float(v)) for v in effective_y_final)
        and all(math.isfinite(float(v)) for v in y_initial)
    )
    if has_valid_output and effective_y_final is not None:
        mass_init = sum(float(v) for v in y_initial)
        mass_final = sum(float(v) for v in effective_y_final)
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
