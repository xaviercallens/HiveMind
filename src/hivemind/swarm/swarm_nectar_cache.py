# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Swarm Nectar Cache: P2P Knowledge & Food Source Sharing over Kafka (HM-S23-NECTAR-CACHE).

Implements Milestone 3 of the Honeybee Agent Mesh Roadmap:
- Topic: `hivemind.mesh.food_sources.v1`
- Cross-bee caching of verified tool outputs (web search snippets, solver proofs, structured extractions).
- Canonical SHA-256 deduplication and TTL-based cache eviction.
- Zero-redundancy policy: eliminates duplicate external API queries and GPU forward passes.
- PoUW HMT Royalties: awards micro-credits to the discoverer Bee whenever another Bee reuses its nectar.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

TOPIC_FOOD_SOURCES = "hivemind.mesh.food_sources.v1"
DEFAULT_NECTAR_TTL_S = 86400.0  # 24 hours
MICRO_ROYALTY_HMT = 0.005  # 0.005 HMT awarded to discoverer per peer hit


def canonical_query_hash(tool_name: str, query: str | Mapping[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash for a given tool and query."""
    normalized_tool = tool_name.strip().lower()
    if isinstance(query, Mapping):
        canonical_content = json.dumps(query, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    else:
        canonical_content = str(query).strip().lower()

    raw = f"{normalized_tool}:{canonical_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class NectarEntry:
    """One verified, reusable piece of information (food source / tool result)."""

    nectar_id: str
    tool_name: str
    query_hash: str
    content: dict[str, Any]
    nectar_richness: float  # Quality / fitness score in [0.0, 1.0]
    discovered_by_bee: str
    access_count: int = 0
    hmt_royalties_earned: float = 0.0
    created_utc: float = field(default_factory=time.time)
    ttl_s: float = DEFAULT_NECTAR_TTL_S

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_utc) > self.ttl_s

    def record_access(self) -> float:
        """Increment access count and distribute micro-royalty."""
        self.access_count += 1
        self.hmt_royalties_earned = round(self.hmt_royalties_earned + MICRO_ROYALTY_HMT, 6)
        return self.hmt_royalties_earned

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["created_iso"] = datetime.fromtimestamp(self.created_utc, tz=timezone.utc).isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NectarEntry:
        """Deserialize a NectarEntry from a persisted dictionary."""
        return cls(
            nectar_id=str(data["nectar_id"]),
            tool_name=str(data["tool_name"]),
            query_hash=str(data["query_hash"]),
            content=dict(data["content"]),
            nectar_richness=float(data["nectar_richness"]),
            discovered_by_bee=str(data["discovered_by_bee"]),
            access_count=int(data.get("access_count", 0)),
            hmt_royalties_earned=float(data.get("hmt_royalties_earned", 0.0)),
            created_utc=float(data.get("created_utc", time.time())),
            ttl_s=float(data.get("ttl_s", DEFAULT_NECTAR_TTL_S)),
        )


class FileNectarPersistence:
    """Atomic JSON file persistence backend for cross-process SwarmNectarCache sharing (HM-S23)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, entries: Mapping[str, NectarEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        active = {k: v.to_dict() for k, v in entries.items() if not v.is_expired}
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(active, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)

    def load(self) -> dict[str, NectarEntry]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        loaded: dict[str, NectarEntry] = {}
        for k, val in raw.items():
            entry = NectarEntry.from_dict(val)
            if not entry.is_expired:
                loaded[str(k)] = entry
        return loaded


class InMemoryClusterNectarHub:
    """Redis/Memcached-compatible in-memory multi-node KV bus for `< 50 ms` cross-bee sync (`#319` Task 2)."""

    def __init__(self) -> None:
        self._kv: dict[str, tuple[str, float]] = {}

    def get(self, key: str) -> str | None:
        item = self._kv.get(key)
        if item is None:
            return None
        payload, expires_at = item
        if time.time() > expires_at:
            self._kv.pop(key, None)
            return None
        return payload

    def setex(self, key: str, ttl_seconds: int | float, value: str) -> None:
        self._kv[key] = (str(value), time.time() + float(ttl_seconds))

    def delete(self, key: str) -> None:
        self._kv.pop(key, None)


class MultiNodeNectarSyncBackend:
    """Distributed KV connector (Redis / Memcached / Cluster Hub) for `SwarmNectarCache` (`HM-S23` / `#319` Task 2)."""

    def __init__(
        self,
        kv_client: Any | None = None,
        *,
        namespace: str = "hivemind:nectar:v1",
    ) -> None:
        self.kv_client = kv_client if kv_client is not None else InMemoryClusterNectarHub()
        self.namespace = namespace.strip(":")

    def _key(self, query_hash: str) -> str:
        return f"{self.namespace}:{query_hash}"

    def put_entry(self, entry: NectarEntry) -> None:
        remaining_ttl = max(1.0, entry.ttl_s - (time.time() - entry.created_utc))
        payload = json.dumps(entry.to_dict(), ensure_ascii=False, sort_keys=True)
        key = self._key(entry.query_hash)
        if hasattr(self.kv_client, "setex"):
            self.kv_client.setex(key, remaining_ttl, payload)
        else:
            self.kv_client.set(key, payload)

    def get_entry(self, query_hash: str) -> NectarEntry | None:
        raw = self.kv_client.get(self._key(query_hash))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        entry = NectarEntry.from_dict(json.loads(raw))
        if entry.is_expired:
            self.delete_entry(query_hash)
            return None
        return entry

    def delete_entry(self, query_hash: str) -> None:
        if hasattr(self.kv_client, "delete"):
            self.kv_client.delete(self._key(query_hash))


class SwarmNectarCache:
    """Decentralized Nectar Cache for MoE Agent Mesh over Kafka AMCP."""

    def __init__(
        self,
        default_ttl_s: float = DEFAULT_NECTAR_TTL_S,
        persistence_path: str | Path | None = None,
        sync_backend: MultiNodeNectarSyncBackend | None = None,
    ) -> None:
        self.default_ttl_s = default_ttl_s
        self._persistence = FileNectarPersistence(persistence_path) if persistence_path is not None else None
        self._sync_backend = sync_backend
        self._cache: dict[str, NectarEntry] = self._persistence.load() if self._persistence is not None else {}
        self._total_queries: int = 0
        self._cache_hits: int = 0
        self._tokens_saved_estimate: int = 0
        self._last_lookup_latency_ms: float = 0.0

    def reload_from_disk(self) -> int:
        """Reload unexpired entries from the persistence backend if configured."""
        if self._persistence is None:
            return len(self._cache)
        disk_entries = self._persistence.load()
        self._cache.update(disk_entries)
        return len(self._cache)

    def flush_to_disk(self) -> None:
        """Persist active entries atomically to disk if persistence is configured."""
        if self._persistence is not None:
            self._persistence.save(self._cache)

    def put(
        self,
        tool_name: str,
        query: str | Mapping[str, Any],
        content: dict[str, Any],
        discovered_by_bee: str,
        nectar_richness: float = 0.90,
        ttl_s: float | None = None,
    ) -> NectarEntry:
        """Store verified tool result in the nectar cache and format Kafka food source event."""
        q_hash = canonical_query_hash(tool_name, query)
        entry_id = f"nectar-{q_hash[:16]}"
        entry = NectarEntry(
            nectar_id=entry_id,
            tool_name=tool_name.strip().lower(),
            query_hash=q_hash,
            content=content,
            nectar_richness=max(0.0, min(1.0, float(nectar_richness))),
            discovered_by_bee=discovered_by_bee.strip(),
            created_utc=time.time(),
            ttl_s=ttl_s if ttl_s is not None else self.default_ttl_s,
        )
        self._cache[q_hash] = entry
        if self._sync_backend is not None:
            self._sync_backend.put_entry(entry)
        self.flush_to_disk()
        return entry

    def get(
        self,
        tool_name: str,
        query: str | Mapping[str, Any],
        requesting_bee: str = "anonymous",
    ) -> NectarEntry | None:
        """Retrieve cached nectar entry if present and unexpired."""
        t0 = time.perf_counter()
        self._total_queries += 1
        q_hash = canonical_query_hash(tool_name, query)
        entry = self._cache.get(q_hash)
        if entry is None and self._sync_backend is not None:
            remote_entry = self._sync_backend.get_entry(q_hash)
            if remote_entry is not None:
                self._cache[q_hash] = remote_entry
                entry = remote_entry
        if entry is None and self._persistence is not None:
            self.reload_from_disk()
            entry = self._cache.get(q_hash)

        if entry is None:
            self._last_lookup_latency_ms = round((time.perf_counter() - t0) * 1000.0, 4)
            return None

        if entry.is_expired:
            del self._cache[q_hash]
            if self._sync_backend is not None:
                self._sync_backend.delete_entry(q_hash)
            self.flush_to_disk()
            self._last_lookup_latency_ms = round((time.perf_counter() - t0) * 1000.0, 4)
            return None

        # Cache Hit!
        self._cache_hits += 1
        self._tokens_saved_estimate += 450  # Average tokens per tool generation avoided
        entry.record_access()
        if self._sync_backend is not None:
            self._sync_backend.put_entry(entry)
        self.flush_to_disk()
        self._last_lookup_latency_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        return entry

    def build_kafka_food_source_event(self, entry: NectarEntry) -> dict[str, Any]:
        """Construct canonical CloudEvent dict for publishing to hivemind.mesh.food_sources.v1."""
        return {
            "topic": TOPIC_FOOD_SOURCES,
            "event_type": "io.hivemind.mesh.food_source_updated",
            "nectar_id": entry.nectar_id,
            "tool_name": entry.tool_name,
            "query_hash": entry.query_hash,
            "discovered_by_bee": entry.discovered_by_bee,
            "nectar_richness": entry.nectar_richness,
            "access_count": entry.access_count,
            "hmt_royalties": entry.hmt_royalties_earned,
            "created_utc": entry.created_utc,
        }

    def stats(self) -> dict[str, Any]:
        """Return cache health, hit rate, and economic royalties distribution."""
        active_entries = sum(1 for e in self._cache.values() if not e.is_expired)
        hit_rate = round(self._cache_hits / max(1, self._total_queries) * 100.0, 2)
        total_royalties = round(sum(e.hmt_royalties_earned for e in self._cache.values()), 4)

        return {
            "active_entries": active_entries,
            "total_queries": self._total_queries,
            "cache_hits": self._cache_hits,
            "hit_rate_pct": hit_rate,
            "tokens_saved_estimate": self._tokens_saved_estimate,
            "total_hmt_royalties_distributed": total_royalties,
            "last_lookup_latency_ms": self._last_lookup_latency_ms,
            "sync_backend_enabled": self._sync_backend is not None,
        }
