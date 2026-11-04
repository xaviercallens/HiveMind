"""settle_swarm_inference must credit providers exactly what it debits the consumer (no value minted by rounding)."""
from __future__ import annotations

from hypothesis import given, settings, strategies as st

from hivemind.swarm.ledger import HiveSwarmLedger


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
