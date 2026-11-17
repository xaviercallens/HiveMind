"""Unit tests for HM-S36-05 DARL Heartbeat-Backed CAS Shard Leasing."""

from __future__ import annotations

import math

import pytest

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


def test_non_finite_times_are_rejected_and_expiry_still_works() -> None:
    import math

    import pytest

    from hivemind.swarm.s36_05_darl_cas_shard_leasing import DARLShardLeaseCoordinator

    coord = DARLShardLeaseCoordinator(["s1", "s2", "s3", "s4"])
    lease = coord.grant_lease("w1", now_s=0.0)
    assert lease is not None
    # a NaN clock would make every "now > expires" comparison False: the lease could never expire or be reclaimed
    for bad in (float("nan"), math.inf):
        with pytest.raises(ValueError):
            coord.commit_lease(lease.lease_id, "w1", now_s=bad)
        with pytest.raises(ValueError):
            coord.grant_lease("w2", now_s=bad)
        with pytest.raises(ValueError):
            coord.reclaim_expired(now_s=bad)
    with pytest.raises(ValueError):
        coord.grant_lease("w2", now_s=1.0, t_step_s=float("nan"))
    # positive control: after the TTL a finite clock still expires the lease and refuses the late commit
    assert coord.commit_lease(lease.lease_id, "w1", now_s=1e6) is False
    assert coord.commit_lease(coord.grant_lease("w3", now_s=1e6).lease_id, "w3", now_s=1e6 + 0.5) is True


@pytest.mark.parametrize("alpha", [float("inf"), float("-inf"), float("nan"), 0.0, -5.0])
def test_alpha_must_be_finite_and_positive(alpha: float) -> None:
    # alpha=inf gave expires_at_s=inf (a lease never reclaimed); NaN or a negative alpha silently clamped the TTL to 1.0.
    with pytest.raises(ValueError, match="alpha"):
        DARLShardLeaseCoordinator(["s1", "s2"], alpha=alpha)
    rec = DARLShardLeaseCoordinator(["s1"], alpha=4.0).grant_lease("w", now_s=0.0)
    assert rec is not None and math.isfinite(rec.expires_at_s)
