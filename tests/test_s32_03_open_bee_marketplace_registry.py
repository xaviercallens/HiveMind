# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for HM-S32-03 Open Bee Marketplace Signed Manifest Verifier & Expert Registry."""

from __future__ import annotations

import hashlib

from hivemind.swarm.s32_03_open_bee_marketplace_registry import (
    OpenBeeMarketplaceRegistry,
    SignedBeeManifest,
    sign_bee_manifest,
)

SECRET = b"open-bee-marketplace-secret-key-2026"


def test_open_bee_marketplace_accepts_valid_signed_manifest_under_8gb_vram() -> None:
    registry = OpenBeeMarketplaceRegistry(SECRET)
    manifest = sign_bee_manifest(
        bee_id="hivemind/bee-protein-fold",
        owner_handle="dr_elena_astro",
        domain="structural_biology",
        base_backbone="gemma-4-9b-it-q4_k_m",
        lora_rank=16,
        vram_required_mib=6144,
        weights_sha256=hashlib.sha256(b"protein-lora-v1").hexdigest(),
        benchmark_score=0.91,
        signing_secret=SECRET,
    )
    res = registry.verify_and_register(manifest)
    assert res["accepted"] is True
    matched = registry.match_best_bee_for_domain("structural_biology")
    assert matched is not None
    assert matched.bee_id == "hivemind/bee-protein-fold"


def test_open_bee_marketplace_rejects_vram_overflow_and_tampered_signature() -> None:
    registry = OpenBeeMarketplaceRegistry(SECRET)
    over_vram = sign_bee_manifest(
        bee_id="hivemind/bee-giant",
        owner_handle="xavier_callens",
        domain="physics",
        base_backbone="qwen3.5-32b",
        lora_rank=32,
        vram_required_mib=12288,
        weights_sha256=hashlib.sha256(b"giant").hexdigest(),
        benchmark_score=0.95,
        signing_secret=SECRET,
    )
    assert registry.verify_and_register(over_vram)["reason"] == "VRAM_CEILING_EXCEEDED_8GB"

    tampered = SignedBeeManifest(
        bee_id="hivemind/bee-climate",
        owner_handle="xavier_callens",
        domain="climate",
        base_backbone="gemma-4-9b",
        lora_rank=16,
        vram_required_mib=5120,
        weights_sha256=hashlib.sha256(b"climate").hexdigest(),
        benchmark_score=0.88,
        signature_hex="0" * 64,
    )
    assert registry.verify_and_register(tampered)["reason"] == "INVALID_MANIFEST_SIGNATURE"
