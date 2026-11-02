# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control tests for #424 (IP-109), #423 (IP-110), and #382 (IP-08 / IP-134)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pytest

from hivemind.security.swarm_runtime_hardening import SwarmSecurityViolationError
from hivemind.swarm import nat_traversal as nt
from hivemind.swarm.ledger import HiveSwarmLedger
from hivemind.swarm.swarm_fabric import HiveLayer2StateChannel


def test_l2_state_channel_rejects_double_settlement_post_refund_and_tampered_voucher(
    tmp_path: Path,
) -> None:
    """#424 / IP-109: HiveLayer2StateChannel must enforce single settlement, block post-refund settlement, and verify HMAC."""
    ledger = HiveSwarmLedger(ledger_path=tmp_path / "l2_guard_ledger.json")
    ledger.balances["consumer-node-1"] = 10.0

    # 1. Tampered voucher signature or amount must be rejected on settle_to_l1_ledger
    ch_tampered = HiveLayer2StateChannel(
        consumer_id="consumer-node-1",
        provider_ids=["provider-a", "provider-b"],
        escrow_hive=5.0,
    )
    v1 = ch_tampered.sign_micro_voucher(20)
    ch_tampered.latest_voucher = replace(v1, cumulative_tokens=1000, cumulative_hive_spent=5.0)
    with pytest.raises(ValueError, match="signature|voucher|tampered"):
        ch_tampered.settle_to_l1_ledger(ledger)

    # 2. Settlement after refund_consumer_escrow() must be rejected
    ch_refunded = HiveLayer2StateChannel(
        consumer_id="consumer-node-1",
        provider_ids=["provider-a"],
        escrow_hive=5.0,
    )
    ch_refunded.sign_micro_voucher(20)
    refund = ch_refunded.refund_consumer_escrow(reason="OPOI_FRAUD_PROVEN")
    assert refund == pytest.approx(0.1)
    with pytest.raises(RuntimeError, match="refunded|disputed|closed"):
        ch_refunded.settle_to_l1_ledger(ledger)

    # 3. Double settlement of the same voucher/channel must be rejected
    ch_honest = HiveLayer2StateChannel(
        consumer_id="consumer-node-1",
        provider_ids=["provider-a", "provider-b"],
        escrow_hive=5.0,
    )
    ch_honest.sign_micro_voucher(40)
    first = ch_honest.settle_to_l1_ledger(ledger)
    assert first["settled_hive"] == pytest.approx(0.2)
    with pytest.raises(RuntimeError, match="already settled|nonce"):
        ch_honest.settle_to_l1_ledger(ledger)


def test_turn_relay_blocks_ssrf_private_peers_and_enforces_user_allocation_quota() -> None:
    """#423 / IP-110: TurnRelayServer must block SSRF/private/multicast peer IPs (403) and enforce per-user allocation quota (486)."""
    with nt.TurnRelayServer(
        {"alice": "s3cret"},
        realm="hivemind.local",
        max_allocations_per_user=2,
    ) as turn:
        c1 = nt.TurnClient(turn.address, "alice", "s3cret", timeout=0.6)
        c2 = nt.TurnClient(turn.address, "alice", "s3cret", timeout=0.6)
        c3 = nt.TurnClient(turn.address, "alice", "s3cret", timeout=0.6)
        try:
            c1.allocate()
            c2.allocate()
            # 3rd concurrent allocation for 'alice' exceeds max_allocations_per_user=2 -> 486
            with pytest.raises(nt.StunError) as quota_err:
                c3.allocate()
            assert quota_err.value.code == 486

            # SSRF / link-local metadata / RFC1918 / multicast / unspecified peers -> 403 Forbidden
            for forbidden_ip in (
                "169.254.169.254",
                "10.150.0.2",
                "192.168.1.1",
                "172.16.0.1",
                "224.0.0.1",
                "0.0.0.0",
                "255.255.255.255",
            ):
                with pytest.raises(nt.StunError) as perm_err:
                    c1.create_permission(forbidden_ip)
                assert perm_err.value.code == 403, f"expected 403 for {forbidden_ip}"

                with pytest.raises(nt.StunError) as bind_err:
                    c1.bind_channel((forbidden_ip, 5000))
                assert bind_err.value.code == 403, f"expected 403 for {forbidden_ip}"
        finally:
            c1.close()
            c2.close()
            c3.close()


def test_hive_swarm_ledger_reloads_and_verifies_persisted_chain_on_init(
    tmp_path: Path,
) -> None:
    """#382 / IP-08 & IP-134: HiveSwarmLedger must reload existing ledger_path on init and reject tampered chain files."""
    ledger_file = tmp_path / "persisted_ledger.json"
    ledger1 = HiveSwarmLedger(ledger_path=ledger_file)
    ledger1.credit_reward("node-alpha", 12.5)
    ledger1.mine_pouw_block(
        outer_step=1,
        expert_cid="cid://expert-wave3",
        work_receipts=[],
        storage_receipts=[],
        validator_node_ids=["node-validator"],
    )
    expected_bal = ledger1.get_balance("node-alpha")
    assert len(ledger1.chain) == 2

    # Reload from existing file
    ledger2 = HiveSwarmLedger(ledger_path=ledger_file)
    assert len(ledger2.chain) == 2
    assert ledger2.get_balance("node-alpha") == pytest.approx(expected_bal)
    assert ledger2.verify_chain_integrity() is True

    # Tamper with persisted block hash on disk -> reload must raise SwarmSecurityViolationError
    raw_text = ledger_file.read_text(encoding="utf-8")
    tampered_text = raw_text.replace("cid://expert-wave3", "cid://forged-expert")
    ledger_file.write_text(tampered_text, encoding="utf-8")
    with pytest.raises(SwarmSecurityViolationError):
        HiveSwarmLedger(ledger_path=ledger_file)
