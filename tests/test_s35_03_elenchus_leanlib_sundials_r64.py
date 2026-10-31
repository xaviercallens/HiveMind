# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for HM-S35-03: Elenchus x SocrateAI-Lean-Lib x rusty-SUNDIALS Tier-X (R-64) Verifier."""

from __future__ import annotations

from hivemind.swarm.s35_03_elenchus_leanlib_sundials_r64 import (
    audit_lean4_source_elenchus,
    verify_robertson_sundials_tier_x,
)


def test_elenchus_comment_stripped_lean_audit_ignores_docstring_vacuity() -> None:
    clean_lean_with_docstring = (
        "/- **Kernel Certified:** 0 sorry, 0 admit. -/\n"
        "-- Do not use sorry\n"
        "theorem chi_k3 : 24 = 24 := by rfl\n"
    )
    audit_clean = audit_lean4_source_elenchus(clean_lean_with_docstring)
    assert audit_clean.raw_sorry_substring_hits == 2
    assert audit_clean.comment_stripped_sorry_count == 0
    assert audit_clean.is_kernel_clean is True
    assert audit_clean.classification == "PROVEN_INVARIANT"

    dirty_lean = "theorem broken : 1 = 2 := by sorry\n"
    audit_dirty = audit_lean4_source_elenchus(dirty_lean)
    assert audit_dirty.comment_stripped_sorry_count == 1
    assert audit_dirty.is_kernel_clean is False
    assert audit_dirty.classification == "UNVERIFIED_SORRY"


def test_robertson_sundials_tier_x_r64_provenance_and_stub_refusal() -> None:
    args = {"problem": "robertson", "rtol": 1e-9, "atol": 1e-12}
    ok_receipt = verify_robertson_sundials_tier_x(sent_arguments=args, received_arguments=args)
    assert ok_receipt.accepted_by_elenchus_bridge is True
    assert ok_receipt.evidentiary_value == "tier_x_measured"
    assert ok_receipt.mass_conservation_drift < 1e-12

    stub_receipt = verify_robertson_sundials_tier_x(
        sent_arguments=args,
        received_arguments=args,
        stub=True,
    )
    assert stub_receipt.accepted_by_elenchus_bridge is False
    assert stub_receipt.evidentiary_value == "none"
