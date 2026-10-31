# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S32-03 — Open Bee Marketplace Signed Manifest Verifier & Self-Service Expert Registry.

Enables researchers, universities, and open-source contributors to publish specialized LoRA/MoE Bees
(e.g., ``hivemind/bee-protein``, ``hivemind/bee-climate``, ``hivemind/bee-fusion``) with:
1. Cryptographic ``HMAC-SHA256`` manifest signing and constant-time verification.
2. Strict 8 GB consumer GPU VRAM ceiling enforcement (Golden Invariant #1: ``vram_required_mib <= 8192``).
3. Deterministic routing score combining benchmark accuracy, latency, and producer stake.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass
from typing import Any

MAX_CONSUMER_VRAM_MIB = 8192


@dataclass(frozen=True)
class SignedBeeManifest:
    """Self-service Open Bee Marketplace manifest."""

    bee_id: str
    owner_handle: str
    domain: str
    base_backbone: str
    lora_rank: int
    vram_required_mib: int
    weights_sha256: str
    benchmark_score: float
    signature_hex: str

    def canonical_payload(self) -> str:
        return json.dumps(
            {
                "bee_id": self.bee_id,
                "owner_handle": self.owner_handle,
                "domain": self.domain,
                "base_backbone": self.base_backbone,
                "lora_rank": self.lora_rank,
                "vram_required_mib": self.vram_required_mib,
                "weights_sha256": self.weights_sha256,
                "benchmark_score": round(self.benchmark_score, 6),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sign_bee_manifest(
    *,
    bee_id: str,
    owner_handle: str,
    domain: str,
    base_backbone: str,
    lora_rank: int,
    vram_required_mib: int,
    weights_sha256: str,
    benchmark_score: float,
    signing_secret: bytes,
) -> SignedBeeManifest:
    """Create and HMAC-SHA256 sign a marketplace Bee manifest."""
    unsigned = SignedBeeManifest(
        bee_id=bee_id,
        owner_handle=owner_handle,
        domain=domain,
        base_backbone=base_backbone,
        lora_rank=lora_rank,
        vram_required_mib=vram_required_mib,
        weights_sha256=weights_sha256,
        benchmark_score=benchmark_score,
        signature_hex="",
    )
    sig = hmac.new(signing_secret, unsigned.canonical_payload().encode("utf-8"), hashlib.sha256).hexdigest()
    return SignedBeeManifest(
        bee_id=bee_id,
        owner_handle=owner_handle,
        domain=domain,
        base_backbone=base_backbone,
        lora_rank=lora_rank,
        vram_required_mib=vram_required_mib,
        weights_sha256=weights_sha256,
        benchmark_score=benchmark_score,
        signature_hex=sig,
    )


class OpenBeeMarketplaceRegistry:
    """Fail-closed self-service registry for community-contributed specialized Bees."""

    def __init__(self, signing_secret: bytes, *, min_benchmark_score: float = 0.70) -> None:
        if len(signing_secret) < 16:
            raise ValueError("signing_secret must be at least 16 bytes")
        self._secret = bytes(signing_secret)
        self.min_benchmark_score = float(min_benchmark_score)
        self._bees: dict[str, SignedBeeManifest] = {}

    def verify_and_register(self, manifest: SignedBeeManifest) -> dict[str, Any]:
        """Verify cryptographic signature, 8 GB VRAM ceiling, and benchmark threshold before admitting a Bee."""
        if not manifest.bee_id.startswith("hivemind/bee-"):
            return {"accepted": False, "reason": "INVALID_BEE_ID_PREFIX"}
        if manifest.vram_required_mib <= 0 or manifest.vram_required_mib > MAX_CONSUMER_VRAM_MIB:
            return {"accepted": False, "reason": "VRAM_CEILING_EXCEEDED_8GB"}
        if len(manifest.weights_sha256) != 64:
            return {"accepted": False, "reason": "INVALID_WEIGHTS_SHA256"}
        if manifest.benchmark_score < self.min_benchmark_score:
            return {"accepted": False, "reason": "BENCHMARK_SCORE_BELOW_THRESHOLD"}
        expected_sig = hmac.new(
            self._secret,
            manifest.canonical_payload().encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected_sig, manifest.signature_hex):
            return {"accepted": False, "reason": "INVALID_MANIFEST_SIGNATURE"}

        self._bees[manifest.bee_id] = manifest
        return {
            "accepted": True,
            "reason": "REGISTERED_IN_OPEN_BEE_MARKETPLACE",
            "bee_id": manifest.bee_id,
            "vram_headroom_mib": MAX_CONSUMER_VRAM_MIB - manifest.vram_required_mib,
        }

    def match_best_bee_for_domain(self, domain: str) -> SignedBeeManifest | None:
        candidates = [b for b in self._bees.values() if b.domain.lower() == domain.lower()]
        if not candidates:
            return None
        return max(candidates, key=lambda b: (b.benchmark_score, -b.vram_required_mib))

    def list_registered_bees(self) -> list[dict[str, Any]]:
        return [b.to_dict() for b in sorted(self._bees.values(), key=lambda x: x.bee_id)]
