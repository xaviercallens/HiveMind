# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S36-05 (HM-S9-06) — DARL Heartbeat-Backed CAS Shard Leasing Under Worker Churn.

Implements Dynamic Asynchronous Lease-Backed Data Sharding (``arXiv:2610.03457``):
1. Assigns CAS data shards with dynamic heartbeat TTL ``TTL = alpha * t_step + rtt_s``.
2. Automatically reclaims expired leases when volunteer workers churn or disconnect.
3. Guarantees that no shard is committed twice and enforces the exact conservation identity
   ``committed == granted - reclaimed - active`` at every point in time, reaching
   ``committed == granted - reclaimed`` when all active leases resolve.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ShardLeaseRecord:
    """Lease state for a single CAS data shard."""

    lease_id: int
    shard_id: str
    worker_id: str
    granted_at_s: float
    expires_at_s: float
    status: str = "ACTIVE"  # ACTIVE | COMMITTED | RECLAIMED


@dataclass(frozen=True)
class DARLLedgerSnapshot:
    """Conservation snapshot of the DARL shard lease coordinator."""

    total_shards: int
    granted: int
    reclaimed: int
    committed: int
    active: int
    unique_committed_shards: int
    duplicate_commits_blocked: int
    conservation_holds: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DARLShardLeaseCoordinator:
    """Heartbeat-backed CAS shard lease manager guaranteeing zero duplicate training under churn."""

    def __init__(self, shard_ids: list[str], *, alpha: float = 4.0) -> None:
        if not shard_ids or len(set(shard_ids)) != len(shard_ids):
            raise ValueError("shard_ids must be a non-empty list of unique shard identifiers")
        self.alpha = float(alpha)
        self._pending_shards: list[str] = list(shard_ids)
        self._total_shards = len(shard_ids)
        self._leases: dict[int, ShardLeaseRecord] = {}
        self._active_by_shard: dict[str, int] = {}
        self._committed_shards: set[str] = set()
        self._next_lease_id = 1
        self.granted_count = 0
        self.reclaimed_count = 0
        self.committed_count = 0
        self.duplicate_commits_blocked = 0

    def compute_ttl_s(self, *, t_step_s: float, rtt_s: float) -> float:
        return max(1.0, self.alpha * float(t_step_s) + float(rtt_s))

    def grant_lease(
        self,
        worker_id: str,
        *,
        now_s: float,
        t_step_s: float = 2.0,
        rtt_s: float = 0.08,
    ) -> ShardLeaseRecord | None:
        """Grant the next available unassigned shard to ``worker_id``."""
        self.reclaim_expired(now_s=now_s)
        while self._pending_shards:
            shard_id = self._pending_shards.pop(0)
            if shard_id in self._committed_shards or shard_id in self._active_by_shard:
                continue
            ttl = self.compute_ttl_s(t_step_s=t_step_s, rtt_s=rtt_s)
            lease_id = self._next_lease_id
            self._next_lease_id += 1
            rec = ShardLeaseRecord(
                lease_id=lease_id,
                shard_id=shard_id,
                worker_id=worker_id,
                granted_at_s=now_s,
                expires_at_s=now_s + ttl,
                status="ACTIVE",
            )
            self._leases[lease_id] = rec
            self._active_by_shard[shard_id] = lease_id
            self.granted_count += 1
            return rec
        return None

    def renew_heartbeat(
        self,
        lease_id: int,
        worker_id: str,
        *,
        now_s: float,
        t_step_s: float = 2.0,
        rtt_s: float = 0.08,
    ) -> bool:
        rec = self._leases.get(lease_id)
        if rec is None or rec.status != "ACTIVE" or rec.worker_id != worker_id:
            return False
        if now_s > rec.expires_at_s:
            self.reclaim_expired(now_s=now_s)
            return False
        rec.expires_at_s = now_s + self.compute_ttl_s(t_step_s=t_step_s, rtt_s=rtt_s)
        return True

    def reclaim_expired(self, *, now_s: float) -> int:
        """Reclaim all active leases whose TTL has expired at ``now_s``."""
        reclaimed_now = 0
        for rec in list(self._leases.values()):
            if rec.status == "ACTIVE" and now_s > rec.expires_at_s:
                rec.status = "RECLAIMED"
                self._active_by_shard.pop(rec.shard_id, None)
                if rec.shard_id not in self._committed_shards:
                    self._pending_shards.append(rec.shard_id)
                self.reclaimed_count += 1
                reclaimed_now += 1
        return reclaimed_now

    def commit_lease(self, lease_id: int, worker_id: str, *, now_s: float) -> bool:
        """Commit a completed shard lease; rejects expired or stale/duplicate commits."""
        self.reclaim_expired(now_s=now_s)
        rec = self._leases.get(lease_id)
        if rec is None or rec.worker_id != worker_id or rec.status != "ACTIVE":
            self.duplicate_commits_blocked += 1
            return False
        if rec.shard_id in self._committed_shards:
            rec.status = "RECLAIMED"
            self._active_by_shard.pop(rec.shard_id, None)
            self.reclaimed_count += 1
            self.duplicate_commits_blocked += 1
            return False
        rec.status = "COMMITTED"
        self._active_by_shard.pop(rec.shard_id, None)
        self._committed_shards.add(rec.shard_id)
        self.committed_count += 1
        return True

    def snapshot(self) -> DARLLedgerSnapshot:
        active = len(self._active_by_shard)
        conservation = (self.committed_count + active) == (self.granted_count - self.reclaimed_count)
        return DARLLedgerSnapshot(
            total_shards=self._total_shards,
            granted=self.granted_count,
            reclaimed=self.reclaimed_count,
            committed=self.committed_count,
            active=active,
            unique_committed_shards=len(self._committed_shards),
            duplicate_commits_blocked=self.duplicate_commits_blocked,
            conservation_holds=conservation and (self.committed_count == len(self._committed_shards)),
        )
