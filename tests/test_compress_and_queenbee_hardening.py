# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and regression tests for KV/Delta Lattice Quantizers and QueenBee Dance/Knowledge Market."""

from __future__ import annotations

import hashlib
import pytest
import torch

from hivemind.compress.kv_lattice import (
    factorised_rate_t,
    from_rows,
    lattice_quantize,
    to_rows,
)
from hivemind.compress.lattice_quantizer import (
    LatticeDeltaQuantizer,
    SignDeltaQuantizer,
    calibrate_scale_for_rate,
)
from hivemind.queenbee.dance import (
    DanceRecord,
    derive_signer_secret,
    wilson_interval,
)
from hivemind.queenbee.knowledge_market import (
    KnowledgeSettlement,
    KnowledgeShare,
    admit_share,
    certify_knowledge_bonus,
    settle_knowledge_bonus,
)
from hivemind.router.market import ConsumerBid, FeePolicy, ProducerOffer, clear_market


def test_kv_lattice_rejects_invalid_kind_scale_and_nonfinite_inputs() -> None:
    x = torch.randn(2, 16)
    with pytest.raises(KeyError):
        to_rows("BAD", x)

    with pytest.raises(KeyError):
        from_rows("BAD", x, x.shape)

    with pytest.raises(KeyError):
        lattice_quantize("BAD", x, 0.5)

    with pytest.raises(ValueError, match="scale"):
        lattice_quantize("E8", x, 0.0)

    with pytest.raises(ValueError, match="scale"):
        lattice_quantize("E8", x, float("nan"))

    x_nan = x.clone()
    x_nan[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or inf"):
        lattice_quantize("E8", x_nan, 0.5)

    with pytest.raises(ValueError, match="codewords"):
        factorised_rate_t("E8", torch.zeros(0, 8), torch.zeros(0, dtype=torch.int64))

    with pytest.raises(ValueError, match="context"):
        factorised_rate_t("E8", torch.zeros(4, 8), torch.zeros(3, dtype=torch.int64))


def test_lattice_and_sign_delta_quantizers_reject_invalid_blocks_empty_and_carry_length_mismatch() -> None:
    with pytest.raises(ValueError, match="block"):
        LatticeDeltaQuantizer("Z", scale=0.5, block=0)

    with pytest.raises(ValueError, match="requires block == 8"):
        LatticeDeltaQuantizer("E8", scale=0.5, block=4)

    with pytest.raises(ValueError, match="requires block == 2"):
        LatticeDeltaQuantizer("A2", scale=0.5, block=4)

    q_lat = LatticeDeltaQuantizer("E8", scale=0.5, error_feedback=True)
    with pytest.raises(ValueError, match="non-empty"):
        q_lat.quantize(torch.zeros(0))

    q_lat.quantize(torch.randn(16))
    with pytest.raises(ValueError, match="carry length"):
        q_lat.quantize(torch.randn(8))

    q_sgn = SignDeltaQuantizer(error_feedback=True)
    with pytest.raises(ValueError, match="non-empty"):
        q_sgn.quantize(torch.zeros(0))

    q_sgn.quantize(torch.randn(16))
    with pytest.raises(ValueError, match="carry length"):
        q_sgn.quantize(torch.randn(8))

    with pytest.raises(KeyError):
        calibrate_scale_for_rate("BAD", torch.randn(64), 2.0)

    with pytest.raises(ValueError, match="outside the reachable range"):
        calibrate_scale_for_rate("E8", torch.randn(64), -0.5)

    with pytest.raises(ValueError, match="tol"):
        calibrate_scale_for_rate("E8", torch.randn(64), 2.0, tol=-0.01)

    with pytest.raises(ValueError, match="iters"):
        calibrate_scale_for_rate("E8", torch.randn(64), 2.0, iters=-1)


def test_queenbee_dance_rejects_whitespace_bool_counts_and_invalid_wilson_z() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        DanceRecord("e1", "   ", "c1", 10, 5, 0.2, 1, "unit_tests", "s1")

    with pytest.raises(ValueError, match="not bool"):
        DanceRecord("e1", "e1", "c1", True, 1, 0.2, 1, "unit_tests", "s1")

    with pytest.raises(ValueError, match="z must be"):
        wilson_interval(5, 10, z=0.0)

    with pytest.raises(ValueError, match="z must be"):
        wilson_interval(5, 10, z=float("nan"))

    with pytest.raises(ValueError, match="master_secret"):
        derive_signer_secret(b"", "s1")

    with pytest.raises(ValueError, match="signer_id"):
        derive_signer_secret(b"secret", "   ")


def test_queenbee_knowledge_market_rejects_nonhex_hash_negative_payouts_and_bad_settlement() -> None:
    with pytest.raises(ValueError, match="64-char hex"):
        KnowledgeShare("u1", "verified_code", "CC0-1.0", True, 1, "z" * 64)

    with pytest.raises(ValueError, match="non-empty"):
        KnowledgeShare("   ", "verified_code", "CC0-1.0", True, 1, "a" * 64)

    clean = "def f(x):\n    return x + 1\n"
    sh = KnowledgeShare(
        "u1",
        "verified_code",
        "CC0-1.0",
        True,
        1,
        hashlib.sha256(clean.encode()).hexdigest(),
        verifier_receipt_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="prior_points_this_epoch"):
        admit_share(sh, clean, prior_points_this_epoch=-1)

    with pytest.raises(ValueError, match="epoch"):
        settle_knowledge_bonus(-1, 10.0, 0.5, {"u1": 1})

    with pytest.raises(ValueError, match="points"):
        settle_knowledge_bonus(1, 10.0, 0.5, {"u1": True})  # type: ignore[dict-item]

    bids = [ConsumerBid("c1", 4.0), ConsumerBid("c2", 3.0)]
    offers = [ProducerOffer("p1", 0.5, 100.0)]
    fee = FeePolicy()
    eq = clear_market("hivemind/coder", bids, offers, fee)
    payment = eq.price_per_1k * eq.supply_1k

    # Negative-payout cancellation attack: {"u1": -50.0, "u2": 50.0} sums to 0.0 <= pool_hmt,
    # but must fail certification because individual payouts cannot be negative.
    neg_settlement = KnowledgeSettlement(
        epoch=1,
        validator_fee_hmt=fee.validator_share * payment,
        bonus_share=0.5,
        pool_hmt=0.5 * fee.validator_share * payment,
        payouts_hmt={"u1": -50.0, "u2": 50.0},
        points={"u1": 1, "u2": 1},
    )
    cert = certify_knowledge_bonus(eq, eq, neg_settlement, fee, payment, True, True)
    assert cert.payouts_within_pool is False
    assert cert.ok is False

    with pytest.raises(ValueError, match="epoch_payment_hmt"):
        certify_knowledge_bonus(eq, eq, neg_settlement, fee, -1.0, True, True)
