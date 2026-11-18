# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Verification suite for PS-07-D1: Ed25519 + W3C Verifiable Credential identity bridge
for KwaaiNet/Petals swarm interoperability, including shared test vectors and negative controls."""

from __future__ import annotations

import copy

import pytest

from hivemind.swarm.identity_bridge import (
    SHARED_ED25519_VC_TEST_VECTORS,
    did_key_from_ed25519_public_bytes,
    ed25519_public_bytes_from_did_key,
    hash_hmt_key_commitment,
    issue_producer_verifiable_credential,
    verify_producer_verifiable_credential,
)


def test_ps07_d1_did_key_roundtrip_and_multicodec_prefix() -> None:
    """PS-07-D1: 32-byte Ed25519 public key encodes to did:key:z6Mk... and round-trips losslessly."""
    raw_pub = bytes(range(32))
    did = did_key_from_ed25519_public_bytes(raw_pub)
    assert did.startswith("did:key:z6Mk")
    decoded = ed25519_public_bytes_from_did_key(did)
    assert decoded == raw_pub

    with pytest.raises(ValueError, match="32 bytes"):
        did_key_from_ed25519_public_bytes(b"short")

    with pytest.raises(ValueError, match="Unsupported did:key"):
        ed25519_public_bytes_from_did_key("did:web:example.com")


def test_ps07_d1_all_shared_ed25519_vc_test_vectors_verify() -> None:
    """PS-07-D1: All shared Ed25519/W3C VC test vectors verify cleanly with HMT commitment and notice."""
    assert len(SHARED_ED25519_VC_TEST_VECTORS) >= 3
    for vec in SHARED_ED25519_VC_TEST_VECTORS:
        vc = vec["credential"]
        ok, reason = verify_producer_verifiable_credential(
            vc,
            expected_node_id=vec["node_id"],
            expected_hmt_key=vec["hmt_key"],
            now_utc="2026-11-07T12:00:00Z",
        )
        assert ok is True, f"Vector {vec['vector_id']} failed: {reason}"
        assert reason == "ok"
        # Raw HMT key must never appear anywhere in the serialized credential
        assert vec["hmt_key"] not in str(vc)
        assert vc["credentialSubject"]["hmt_key_sha256"] == hash_hmt_key_commitment(vec["hmt_key"])
        assert vc["credentialSubject"]["producer_sees_prompts"] is True


def test_ps07_d1_negative_controls_tampered_vectors_rejected() -> None:
    """PS-07-D1 negative controls: tampered subject, signature, HMT key, expiry, or notice are rejected."""
    base_vec = SHARED_ED25519_VC_TEST_VECTORS[0]
    base_vc = base_vec["credential"]

    # 1. Tampered node_id in credentialSubject invalidates Ed25519 signature
    tampered_node = copy.deepcopy(base_vc)
    tampered_node["credentialSubject"]["node_id"] = "node-attacker-spoofed"
    ok1, reason1 = verify_producer_verifiable_credential(tampered_node)
    assert ok1 is False
    assert reason1 == "invalid_ed25519_signature"

    # 2. Tampered tiers list in credentialSubject invalidates Ed25519 signature
    tampered_tiers = copy.deepcopy(base_vc)
    tampered_tiers["credentialSubject"]["tiers"] = ["hivemind/moe-coder"]
    ok2, reason2 = verify_producer_verifiable_credential(tampered_tiers)
    assert ok2 is False
    assert reason2 == "invalid_ed25519_signature"

    # 3. Tampered proofValue signature bytes are rejected
    tampered_sig = copy.deepcopy(base_vc)
    pv = tampered_sig["proof"]["proofValue"]
    flipped_char = "A" if pv[-1] != "A" else "B"
    tampered_sig["proof"]["proofValue"] = pv[:-1] + flipped_char
    ok3, reason3 = verify_producer_verifiable_credential(tampered_sig)
    assert ok3 is False
    assert reason3 in {"invalid_ed25519_signature", "invalid_proof_base64", "invalid_signature_length"}

    # 4. Wrong HMT key presented alongside valid VC is rejected by commitment check
    ok4, reason4 = verify_producer_verifiable_credential(
        base_vc,
        expected_hmt_key="hmt_wrong_key_for_this_producer",
    )
    assert ok4 is False
    assert reason4 == "hmt_key_commitment_mismatch"

    # 5. Expired credential rejected when evaluated after expirationDate
    ok5, reason5 = verify_producer_verifiable_credential(
        base_vc,
        now_utc="2028-01-01T00:00:00Z",
    )
    assert ok5 is False
    assert reason5 == "credential_expired_or_not_yet_valid"

    # 6. Missing producer_sees_prompts notice rejected even before signature check
    missing_notice = copy.deepcopy(base_vc)
    missing_notice["credentialSubject"]["producer_sees_prompts_notice"] = ""
    ok6, reason6 = verify_producer_verifiable_credential(missing_notice)
    assert ok6 is False
    assert reason6 == "missing_producer_sees_prompts_notice"

    # 7. Credential signed by key A but claiming issuer did:key of key B is rejected
    vc_b = issue_producer_verifiable_credential(
        private_key_seed=bytes.fromhex("44" * 32),
        node_id="node-b",
        hmt_key="hmt_b",
        kwaainet_peer_id="12D3KooWNodeB",
        tiers=["hivemind/coder"],
    )
    cross_signed = copy.deepcopy(base_vc)
    cross_signed["proof"]["proofValue"] = vc_b["proof"]["proofValue"]
    ok7, reason7 = verify_producer_verifiable_credential(cross_signed)
    assert ok7 is False
    assert reason7 == "invalid_ed25519_signature"
