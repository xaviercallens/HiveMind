"""settle_swarm_inference must credit providers exactly what it debits the consumer (no value minted by rounding)."""
from __future__ import annotations

from hypothesis import given, settings, strategies as st
import pytest

from hivemind.security.swarm_runtime_hardening import SwarmSecurityViolationError
from hivemind.swarm.ledger import HiveSwarmLedger, StorageProofReceipt, WorkReceipt


def _micro(x: float) -> int:
    return int(round(x * 1_000_000))


def test_fee_of_five_micro_hive_over_three_providers_conserves_total() -> None:
    ledger = HiveSwarmLedger()
    ledger.credit_reward("c", 1.0)
    before = _micro(sum(ledger.balances.values()))
    receipt = ledger.settle_swarm_inference("c", ["p1", "p2", "p3"], "Astro", verified_tokens=1, rtt_speedup=1.0, rate_per_token=5e-06)
    assert receipt.fee_paid_hive == 5e-06
    assert _micro(sum(ledger.balances.values())) == before
    assert sorted(_micro(ledger.balances[p]) for p in ("p1", "p2", "p3")) == [1, 2, 2]


@settings(max_examples=60, deadline=None)
@given(tokens=st.integers(1, 5000), micro_rate=st.integers(1, 20_000), n=st.integers(1, 9))
def test_total_balances_unchanged_by_any_settlement(tokens: int, micro_rate: int, n: int) -> None:
    ledger = HiveSwarmLedger()
    ledger.credit_reward("c", 1000.0)
    before = _micro(sum(ledger.balances.values()))
    providers = [f"p{i}" for i in range(n)]
    ledger.settle_swarm_inference("c", providers, "Astro", verified_tokens=tokens, rtt_speedup=1.0, rate_per_token=micro_rate / 1_000_000)
    assert _micro(sum(ledger.balances.values())) == before


def test_empty_receipt_blocks_mint_zero_to_validators_ip270() -> None:
    ledger = HiveSwarmLedger()
    for i in range(3):
        blk = ledger.mine_pouw_block(i, "cid://x", [], [], ["sybil"])
        assert blk.total_block_reward == 0.0
        assert blk.reward_allocations == {}
    assert ledger.get_balance("sybil") == 0.0
    assert ledger.total_circulating_supply() == 0.0
    assert ledger.verify_chain_integrity() is True

    # When a verified WorkReceipt binds an explicit validator_node_id, only that validator earns
    work = WorkReceipt(
        node_id="worker-1",
        steam_id64="76561198000000001",
        branch_name="astro-branch",
        flops_contributed=1e12,
        cosine_alignment=0.95,
        spot_check_passed=True,
        multikrum_accepted=True,
        delta_sha256="a" * 64,
        validator_node_id="honest-val",
    )
    blk2 = ledger.mine_pouw_block(3, "cid://x", [work], [], ["honest-val", "sybil"])
    assert ledger.get_balance("sybil") == 0.0
    assert ledger.get_balance("honest-val") == 5.0
    assert ledger.get_balance("worker-1") == 37.5
    assert blk2.total_block_reward == 50.0


def test_supply_cap_tracks_cumulative_minted_across_slashing_ip271() -> None:
    ledger = HiveSwarmLedger()
    ledger.balances["n"] = ledger.MAX_SUPPLY_HIVE
    ledger.balances["n"] /= 2
    assert ledger.total_circulating_supply() == ledger.MAX_SUPPLY_HIVE / 2
    assert ledger.total_minted_supply() == ledger.MAX_SUPPLY_HIVE
    assert ledger.block_reward_at_height(5) == 0.0
    with pytest.raises(SwarmSecurityViolationError) as exc_info:
        ledger.credit_reward("other", 1.0)
    assert exc_info.value.rule_id == "LEDGER-04"

    # Also verify slashing inside mine_pouw_block does not reopen mint capacity
    ledger2 = HiveSwarmLedger()
    ledger2.credit_reward("byzantine-worker", ledger2.MAX_SUPPLY_HIVE)
    assert ledger2.block_reward_at_height(1) == 0.0
    bad_work = WorkReceipt(
        node_id="byzantine-worker",
        steam_id64="76561198000000002",
        branch_name="astro-branch",
        flops_contributed=1e12,
        cosine_alignment=0.1,
        spot_check_passed=False,
        multikrum_accepted=False,
        delta_sha256="b" * 64,
    )
    ledger2.mine_pouw_block(1, "cid://x", [bad_work], [], ["val-1"])
    assert ledger2.get_balance("byzantine-worker") == ledger2.MAX_SUPPLY_HIVE / 2
    assert ledger2.total_minted_supply() == ledger2.MAX_SUPPLY_HIVE
    assert ledger2.block_reward_at_height(2) == 0.0

