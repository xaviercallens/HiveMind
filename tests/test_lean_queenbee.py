"""The QueenBee Lean section (`HiveMindSpec.ALIX_QueenBee`, Theorems 129-139) stays declared, printed and whitelisted."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "lean4" / "HiveMindSpec.lean"
RECEIPT = ROOT / "artifacts" / "lean_queenbee_check.json"
THEOREMS = (
    "faded_record_recruits_nobody",
    "recruit_weight_le_cap",
    "sybil_influence_le_count_times_cap",
    "score_monotone_in_successes",
    "score_antitone_in_trials",
    "unverified_outcomes_never_move_the_count",
    "verified_rows_keep_mix_admissible",
    "dilution_cap_ratio",
    "lump_sum_bonus_preserves_best_response",
    "lump_sum_bonus_preserves_nash",
    "knowledge_bonus_payouts_never_exceed_pool",
)
HELPERS = ("add_div_le_div_add", "payouts_le_pool_times_weights_over_W")
ALLOWED = {"propext", "Quot.sound", "Classical.choice"}


def test_theorems_are_declared_once_and_printed() -> None:
    source = SPEC.read_text(encoding="utf-8")
    assert "namespace ALIX_QueenBee" in source and "end ALIX_QueenBee" in source
    assert "import Mathlib" not in source
    for name in THEOREMS:
        assert len(re.findall(rf"^theorem {name}\b", source, flags=re.M)) == 1, name
        assert source.count(f"#print axioms HiveMindSpec.ALIX_QueenBee.{name}") == 1, name
    section = source[source.index("namespace ALIX_QueenBee"): source.index("end ALIX_QueenBee")]
    assert "sorry" not in section and "axiom " not in section


def test_receipt_records_every_theorem_with_whitelisted_axioms() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["queenbee_namespace"] == "HiveMindSpec.ALIX_QueenBee"
    assert set(receipt["queenbee_theorems"]) == set(THEOREMS) | set(HELPERS)
    for name, axioms in receipt["queenbee_theorems"].items():
        assert set(axioms) <= ALLOWED, (name, axioms)
    assert receipt["theorems_failing_whitelist"] == []
    # a receipt for an older spec is stale: rerun anse.formal.lean_runner and regenerate it
    assert receipt["spec_sha256"] == hashlib.sha256(SPEC.read_bytes()).hexdigest()


@pytest.mark.skipif(not os.environ.get("LEAN_BIN"), reason="LEAN_BIN not set; CI compiles the spec separately")
def test_theorems_are_machine_checked_with_standard_axioms_only() -> None:
    from paper.autoevolve_peer_review_pipeline import require_machine_checked, verify_lean4_specification

    result = verify_lean4_specification()
    require_machine_checked(result)
    for name in THEOREMS:
        axioms = result["axioms_by_theorem"][f"HiveMindSpec.ALIX_QueenBee.{name}"]
        assert set(axioms) <= ALLOWED, (name, axioms)
