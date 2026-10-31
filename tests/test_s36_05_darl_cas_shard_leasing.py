"""Unit tests for HM-S36-05 DARL Heartbeat-Backed CAS Shard Leasing."""

from __future__ import annotations

from hivemind.swarm.s36_05_darl_cas_shard_leasing import DARLShardLeaseCoordinator


def test_darl_shard_leasing_conserves_committed_equals_granted_minus_reclaimed() -> None:
    coord = DARLShardLeaseCoordinator(["s1", "s2", "s3", "s4"], alpha=3.0)
    l1 = coord.grant_lease("worker-a", now_s=0.0, t_step_s=2.0, rtt_s=0.1)
    l2 = coord.grant_lease("worker-b", now_s=0.0, t_step_s=2.0, rtt_s=0.1)
    assert l1 is not None and l2 is not None

    # worker-a renews heartbeat and commits s1
    assert coord.renew_heartbeat(l1.lease_id, "worker-a", now_s=4.0) is True
    assert coord.commit_lease(l1.lease_id, "worker-a", now_s=5.0) is True

    # worker-b stalls beyond TTL (6.1s); lease is reclaimed at t=10.0
    reclaimed = coord.reclaim_expired(now_s=10.0)
    assert reclaimed == 1
    # late commit by worker-b is rejected
    assert coord.commit_lease(l2.lease_id, "worker-b", now_s=11.0) is False

    # Drain remaining shards (including reclaimed s2)
    for idx in range(3):
        lease = coord.grant_lease(f"worker-c-{idx}", now_s=12.0 + idx)
        assert lease is not None
        assert coord.commit_lease(lease.lease_id, f"worker-c-{idx}", now_s=13.0 + idx) is True

    snap = coord.snapshot()
    assert snap.conservation_holds is True
    assert snap.active == 0
    assert snap.committed == 4
    assert snap.committed == snap.granted - snap.reclaimed
    assert snap.duplicate_commits_blocked >= 1
