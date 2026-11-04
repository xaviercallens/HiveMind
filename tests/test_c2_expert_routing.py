# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""MVP criterion C2: an adapter reaches LIVE only through a passing train-to-serve gate for exactly its bytes."""

from __future__ import annotations

import hashlib

import pytest

from hivemind.core.expert_registry import (
    LIVE,
    SHADOW,
    ExpertRecord,
    ExpertRegistry,
    GateNotPassedError,
)

BASE = "ibm-granite/granite-3.1-1b-a400m-base"
ADAPTER_BYTES = b"gguf-lora-bytes-for-test"
SHA = hashlib.sha256(ADAPTER_BYTES).hexdigest()


def _registry(strict: bool = True) -> ExpertRegistry:
    reg = ExpertRegistry(strict=strict)
    reg.register(ExpertRecord("boinc-granite", "lora_domain", "general", BASE, adapter_sha256=SHA, adapter_size_mib=3, keywords=("wikipedia",)))
    reg.transition("boinc-granite", SHADOW, "candidate trained by the BOINC chain, shadow evaluation")
    return reg


def _gate(servable: bool = True, sha: str = SHA, base: str = BASE) -> dict:
    return {"servable": servable, "base": base, "adapter": {"lora_gguf_sha256": sha},
            "pooled": {"mean_nll_gain_nats": 0.02, "bootstrap95": [0.01, 0.03]}, "worst_domain_regression": 0.002}


def test_a_passing_gate_for_the_same_bytes_promotes_to_live_and_records_why() -> None:
    reg = _registry()
    rec = reg.promote_with_gate("boinc-granite", _gate())
    assert rec.state == LIVE and "train-to-serve gate passed" in rec.history[-1][2]
    assert reg.ids_in_state(LIVE) == ["boinc-granite"] and "boinc-granite" in reg.residency_inputs()


def test_a_failed_gate_leaves_the_adapter_in_shadow() -> None:
    reg = _registry()
    with pytest.raises(GateNotPassedError):
        reg.promote_with_gate("boinc-granite", _gate(servable=False))
    assert reg.get("boinc-granite").state == SHADOW and reg.ids_in_state(LIVE) == []


def test_a_gate_for_different_bytes_is_refused() -> None:
    reg = _registry()
    with pytest.raises(GateNotPassedError, match="tested adapter"):
        reg.promote_with_gate("boinc-granite", _gate(sha=hashlib.sha256(b"other").hexdigest()))
    with pytest.raises(GateNotPassedError):
        reg.promote_with_gate("boinc-granite", {"servable": True, "base": BASE, "adapter": {}})  # a legacy artifact without the file hash
    assert reg.get("boinc-granite").state == SHADOW


def test_a_gate_for_another_base_model_is_refused() -> None:
    reg = _registry()
    with pytest.raises(GateNotPassedError, match="base"):
        reg.promote_with_gate("boinc-granite", _gate(base="some/other-base"))


def test_a_strict_registry_refuses_the_plain_transition_to_live() -> None:
    reg = _registry(strict=True)
    with pytest.raises(GateNotPassedError):
        reg.transition("boinc-granite", LIVE, "looks good to me")
    assert reg.get("boinc-granite").state == SHADOW


def test_the_default_registry_keeps_the_old_behaviour() -> None:
    reg = _registry(strict=False)
    assert reg.transition("boinc-granite", LIVE, "legacy callers promote directly").state == LIVE


def test_the_adapter_bytes_check_ties_the_record_to_the_file() -> None:
    reg = _registry()
    assert reg.verify_adapter_bytes("boinc-granite", ADAPTER_BYTES) and not reg.verify_adapter_bytes("boinc-granite", ADAPTER_BYTES + b"x")


def test_the_real_gate_artifact_for_hmq_00147_cannot_promote_anything() -> None:
    import json
    from pathlib import Path

    gate = json.loads((Path(__file__).resolve().parents[1] / "artifacts/experiments/GRANITE_ADAPTER_hmq_00147_gate_f16.json").read_text(encoding="utf-8"))
    assert gate["servable"] is False
    with pytest.raises(GateNotPassedError):
        _registry().promote_with_gate("boinc-granite", gate)
