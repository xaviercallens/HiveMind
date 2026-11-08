# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Ed25519 + W3C Verifiable Credential identity bridge for KwaaiNet/Petals swarm interoperability (PS-07-D1).

Allows a HiveMind producer or trainer node to present a portable Ed25519 ``did:key``
and a signed W3C Verifiable Credential alongside its HMT account key (bound via
``SHA-256(hmt_key)`` so the raw bearer key is never exposed in the credential),
enabling peer-to-peer verification when the Cloud Run gateway is offline.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)

W3C_VC_CONTEXT = "https://www.w3.org/2018/credentials/v1"
VC_TYPE_BASE = "VerifiableCredential"
VC_TYPE_BRIDGE = "HiveMindProducerIdentityBridgeCredential"
PROOF_TYPE = "Ed25519Signature2020"
ED25519_MULTICODEC_PREFIX = b"\xed\x01"
PRODUCER_SEES_PROMPTS_NOTICE = (
    "Prompts routed to a community producer are visible in cleartext to that producer host; "
    "do not send confidential or personal secrets."
)

_BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_BASE58_INDEX = {ch: idx for idx, ch in enumerate(_BASE58_ALPHABET)}


def _b58encode(raw: bytes) -> str:
    if not isinstance(raw, (bytes, bytearray)) or len(raw) == 0:
        raise ValueError("raw must be non-empty bytes")
    num = int.from_bytes(raw, "big")
    chars: list[str] = []
    while num > 0:
        num, rem = divmod(num, 58)
        chars.append(_BASE58_ALPHABET[rem])
    leading_zeros = len(raw) - len(raw.lstrip(b"\x00"))
    return ("1" * leading_zeros) + "".join(reversed(chars))


def _b58decode(encoded: str) -> bytes:
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("Empty base58 string")
    num = 0
    for ch in encoded:
        if ch not in _BASE58_INDEX:
            raise ValueError(f"Invalid base58 character {ch!r}")
        num = num * 58 + _BASE58_INDEX[ch]
    leading_ones = len(encoded) - len(encoded.lstrip("1"))
    body = num.to_bytes((num.bit_length() + 7) // 8, "big") if num > 0 else b""
    return (b"\x00" * leading_ones) + body


def did_key_from_ed25519_public_bytes(pub_bytes: bytes) -> str:
    """Encode a 32-byte Ed25519 public key as a standard W3C ``did:key:z6Mk...`` identifier."""
    if not isinstance(pub_bytes, (bytes, bytearray)) or len(pub_bytes) != 32:
        raise ValueError("Ed25519 public key must be exactly 32 bytes")
    multicodec = ED25519_MULTICODEC_PREFIX + bytes(pub_bytes)
    return f"did:key:z{_b58encode(multicodec)}"


def ed25519_public_bytes_from_did_key(did: str) -> bytes:
    """Decode a ``did:key:z6Mk...`` identifier into raw 32-byte Ed25519 public key bytes."""
    if not isinstance(did, str) or not did.startswith("did:key:z"):
        raise ValueError(f"Unsupported did:key format: {did!r}")
    decoded = _b58decode(did[len("did:key:z") :])
    if len(decoded) != 34 or not decoded.startswith(ED25519_MULTICODEC_PREFIX):
        raise ValueError("did:key payload does not have a valid 32-byte Ed25519 multicodec prefix")
    return decoded[2:]


def hash_hmt_key_commitment(hmt_key: str) -> str:
    """Compute a domain-separated SHA-256 commitment to an HMT key without revealing the secret key."""
    if not isinstance(hmt_key, str) or not hmt_key.strip():
        raise ValueError("hmt_key must be non-empty")
    cleaned = hmt_key.strip()
    return hashlib.sha256(f"hivemind.identity_bridge.hmt.v1:{cleaned}".encode("utf-8")).hexdigest()


def _canonical_vc_bytes_without_proof(vc: dict[str, Any]) -> bytes:
    unsigned = {k: v for k, v in vc.items() if k != "proof"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _parse_iso_utc(ts: str) -> datetime:
    if not isinstance(ts, str) or not ts.strip():
        raise ValueError("timestamp must be a non-empty ISO-8601 string")
    cleaned = ts.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def issue_producer_verifiable_credential(
    *,
    private_key_seed: bytes,
    node_id: str,
    hmt_key: str,
    kwaainet_peer_id: str,
    tiers: list[str],
    issuance_date: str = "2026-11-07T00:00:00Z",
    expiration_date: str = "2027-11-07T00:00:00Z",
) -> dict[str, Any]:
    """Issue a signed W3C Verifiable Credential binding an Ed25519 ``did:key``, ``node_id``, and ``hmt_key``."""
    if not isinstance(private_key_seed, (bytes, bytearray)) or len(private_key_seed) != 32:
        raise ValueError("private_key_seed must be exactly 32 bytes")
    if not isinstance(node_id, str) or not isinstance(kwaainet_peer_id, str):
        raise ValueError("node_id and kwaainet_peer_id must be strings")
    node_id_clean = node_id.strip()
    peer_id_clean = kwaainet_peer_id.strip()
    if not node_id_clean or not peer_id_clean:
        raise ValueError("node_id and kwaainet_peer_id must be non-empty")
    if not isinstance(tiers, list) or not tiers or not all(isinstance(t, str) and t.strip() for t in tiers):
        raise ValueError("tiers must be a non-empty list of tier IDs")
    issued_dt = _parse_iso_utc(issuance_date)
    expires_dt = _parse_iso_utc(expiration_date)
    if expires_dt <= issued_dt:
        raise ValueError("expiration_date must be strictly after issuance_date")

    priv = Ed25519PrivateKey.from_private_bytes(bytes(private_key_seed))
    pub_bytes = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    did = did_key_from_ed25519_public_bytes(pub_bytes)
    hmt_commitment = hash_hmt_key_commitment(hmt_key)

    vc: dict[str, Any] = {
        "@context": [W3C_VC_CONTEXT],
        "id": f"urn:hivemind:vc:producer:{node_id_clean}:{hmt_commitment[:16]}",
        "type": [VC_TYPE_BASE, VC_TYPE_BRIDGE],
        "issuer": did,
        "issuanceDate": issuance_date,
        "expirationDate": expiration_date,
        "credentialSubject": {
            "id": did,
            "node_id": node_id_clean,
            "hmt_key_sha256": hmt_commitment,
            "kwaainet_peer_id": peer_id_clean,
            "tiers": [t.strip() for t in tiers],
            "producer_sees_prompts": True,
            "producer_sees_prompts_notice": PRODUCER_SEES_PROMPTS_NOTICE,
        },
    }
    msg = _canonical_vc_bytes_without_proof(vc)
    signature = priv.sign(msg)
    proof_value = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    vc["proof"] = {
        "type": PROOF_TYPE,
        "created": issuance_date,
        "verificationMethod": f"{did}#key-1",
        "proofPurpose": "assertionMethod",
        "proofValue": proof_value,
    }
    return vc


def verify_producer_verifiable_credential(
    vc: dict[str, Any],
    *,
    expected_node_id: str | None = None,
    expected_hmt_key: str | None = None,
    now_utc: str | None = None,
) -> tuple[bool, str]:
    """Verify a producer W3C Verifiable Credential, its Ed25519 proof, and its HMT commitment."""
    if not isinstance(vc, dict):
        return False, "credential_not_a_dict"
    ctx = vc.get("@context")
    if not isinstance(ctx, list) or W3C_VC_CONTEXT not in ctx:
        return False, "invalid_context"
    vc_types = vc.get("type")
    if not isinstance(vc_types, list) or VC_TYPE_BASE not in vc_types or VC_TYPE_BRIDGE not in vc_types:
        return False, "invalid_type"

    issuer = str(vc.get("issuer") or "").strip()
    try:
        pub_bytes = ed25519_public_bytes_from_did_key(issuer)
    except ValueError as exc:
        return False, f"invalid_issuer_did:{exc}"

    subject = vc.get("credentialSubject")
    if not isinstance(subject, dict):
        return False, "missing_credential_subject"
    if subject.get("id") != issuer:
        return False, "subject_id_mismatch_with_issuer"
    if subject.get("producer_sees_prompts") is not True:
        return False, "missing_producer_sees_prompts_flag"
    notice = str(subject.get("producer_sees_prompts_notice") or "").strip()
    if "Prompts routed to a community producer are visible" not in notice:
        return False, "missing_producer_sees_prompts_notice"

    node_id = str(subject.get("node_id") or "").strip()
    if not node_id:
        return False, "missing_node_id"
    if expected_node_id is not None and node_id != expected_node_id:
        return False, "node_id_mismatch"

    hmt_sha = str(subject.get("hmt_key_sha256") or "").strip()
    if len(hmt_sha) != 64:
        return False, "invalid_hmt_key_sha256"
    if expected_hmt_key is not None and hmt_sha != hash_hmt_key_commitment(expected_hmt_key):
        return False, "hmt_key_commitment_mismatch"

    try:
        issued_dt = _parse_iso_utc(str(vc.get("issuanceDate") or ""))
        expires_dt = _parse_iso_utc(str(vc.get("expirationDate") or ""))
    except Exception:
        return False, "invalid_dates"
    if expires_dt <= issued_dt:
        return False, "expiration_not_after_issuance"
    if now_utc is not None:
        try:
            now_dt = _parse_iso_utc(now_utc)
        except Exception:
            return False, "invalid_now_utc"
        if now_dt < issued_dt or now_dt >= expires_dt:
            return False, "credential_expired_or_not_yet_valid"

    proof = vc.get("proof")
    if not isinstance(proof, dict):
        return False, "missing_proof"
    if proof.get("type") != PROOF_TYPE or proof.get("proofPurpose") != "assertionMethod":
        return False, "invalid_proof_metadata"
    if proof.get("verificationMethod") != f"{issuer}#key-1":
        return False, "verification_method_mismatch"

    proof_val = str(proof.get("proofValue") or "").strip()
    if not proof_val:
        return False, "missing_proof_value"
    padding = "=" * (-len(proof_val) % 4)
    try:
        sig_bytes = base64.urlsafe_b64decode(proof_val + padding)
    except Exception:
        return False, "invalid_proof_base64"
    if len(sig_bytes) != 64:
        return False, "invalid_signature_length"

    msg = _canonical_vc_bytes_without_proof(vc)
    pub_key = Ed25519PublicKey.from_public_bytes(pub_bytes)
    try:
        pub_key.verify(sig_bytes, msg)
    except InvalidSignature:
        return False, "invalid_ed25519_signature"

    return True, "ok"


def build_shared_test_vectors() -> list[dict[str, Any]]:
    """Return the 3 deterministic Ed25519 + W3C VC test vectors shared across HiveMind and KwaaiNet/Petals."""
    specs = [
        {
            "vector_id": "hm-kwaainet-vec-01-coder-7b",
            "seed_hex": "11" * 32,
            "node_id": "node-paris-rtx2070-01",
            "hmt_key": "hmt_live_vector01_paris_rtx2070",
            "kwaainet_peer_id": "12D3KooWParisNode01KwaaiNetPetalsBridge",
            "tiers": ["hivemind/coder", "hivemind/granite-moe"],
            "issuance_date": "2026-11-07T00:00:00Z",
            "expiration_date": "2027-11-07T00:00:00Z",
        },
        {
            "vector_id": "hm-kwaainet-vec-02-granite-cpu",
            "seed_hex": "22" * 32,
            "node_id": "node-lyon-cpu-02",
            "hmt_key": "hmt_live_vector02_lyon_cpu",
            "kwaainet_peer_id": "12D3KooWLyonNode02KwaaiNetPetalsBridge",
            "tiers": ["hivemind/granite-1b-base"],
            "issuance_date": "2026-11-07T00:00:00Z",
            "expiration_date": "2027-11-07T00:00:00Z",
        },
        {
            "vector_id": "hm-kwaainet-vec-03-bee-stars",
            "seed_hex": "33" * 32,
            "node_id": "node-brussels-rtx4070-03",
            "hmt_key": "hmt_live_vector03_brussels_rtx4070",
            "kwaainet_peer_id": "12D3KooWBrusselsNode03KwaaiNetPetalsBridge",
            "tiers": ["hivemind/bee-stars", "hivemind/coder-plus"],
            "issuance_date": "2026-11-07T00:00:00Z",
            "expiration_date": "2027-11-07T00:00:00Z",
        },
    ]
    vectors: list[dict[str, Any]] = []
    for item in specs:
        seed = bytes.fromhex(item["seed_hex"])
        vc = issue_producer_verifiable_credential(
            private_key_seed=seed,
            node_id=item["node_id"],
            hmt_key=item["hmt_key"],
            kwaainet_peer_id=item["kwaainet_peer_id"],
            tiers=list(item["tiers"]),
            issuance_date=item["issuance_date"],
            expiration_date=item["expiration_date"],
        )
        vectors.append(
            {
                "vector_id": item["vector_id"],
                "seed_hex": item["seed_hex"],
                "node_id": item["node_id"],
                "hmt_key": item["hmt_key"],
                "did_key": vc["issuer"],
                "credential": vc,
            }
        )
    return vectors


SHARED_ED25519_VC_TEST_VECTORS: list[dict[str, Any]] = build_shared_test_vectors()
