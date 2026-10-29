# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit Tests for Swarm Nectar Cache (P2P Knowledge & Food Source Sharing over Kafka)."""

import time
import pytest

from hivemind.swarm.swarm_nectar_cache import (
    MICRO_ROYALTY_HMT,
    TOPIC_FOOD_SOURCES,
    SwarmNectarCache,
    canonical_query_hash,
)


def test_canonical_query_hashing_consistency():
    """Verify hash determinism across whitespace and case variations."""
    h1 = canonical_query_hash("web_search", "Lithium Battery Solid-State")
    h2 = canonical_query_hash("WEB_SEARCH", "lithium battery solid-state")
    assert h1 == h2

    # Different query yields different hash
    h3 = canonical_query_hash("web_search", "Sodium Ion Battery")
    assert h1 != h3


def test_nectar_cache_put_get_hit_and_royalties():
    """Verify storing, retrieving, hit counting, and micro-royalty attribution in HMT."""
    cache = SwarmNectarCache(default_ttl_s=3600.0)

    entry = cache.put(
        tool_name="web_search",
        query="solid-state battery cathode stability",
        content={"snippets": ["Li6PS5Cl electrolyte shows 450 Wh/kg"]},
        discovered_by_bee="hivemind/qwen35-4b",
        nectar_richness=0.95,
    )

    assert entry.nectar_id.startswith("nectar-")
    assert entry.discovered_by_bee == "hivemind/qwen35-4b"
    assert entry.access_count == 0
    assert entry.hmt_royalties_earned == 0.0

    # Peer bee 1 accesses the nectar
    hit1 = cache.get("web_search", "solid-state battery cathode stability", requesting_bee="hivemind/phi4-mini")
    assert hit1 is not None
    assert hit1.access_count == 1
    assert hit1.hmt_royalties_earned == MICRO_ROYALTY_HMT

    # Peer bee 2 accesses the nectar
    hit2 = cache.get("web_search", "solid-state battery cathode stability", requesting_bee="hivemind/gemma4-e4b")
    assert hit2 is not None
    assert hit2.access_count == 2
    assert hit2.hmt_royalties_earned == round(2 * MICRO_ROYALTY_HMT, 6)

    # Cache miss
    miss = cache.get("web_search", "unseen query", requesting_bee="hivemind/qwen35-4b")
    assert miss is None

    stats = cache.stats()
    assert stats["active_entries"] == 1
    assert stats["total_queries"] == 3
    assert stats["cache_hits"] == 2
    assert stats["hit_rate_pct"] == 66.67
    assert stats["tokens_saved_estimate"] == 900
    assert stats["total_hmt_royalties_distributed"] == round(2 * MICRO_ROYALTY_HMT, 4)


def test_nectar_cache_ttl_expiration():
    """Verify expired nectar entries are cleaned up and return None."""
    cache = SwarmNectarCache(default_ttl_s=0.05)  # 50ms TTL

    cache.put(
        tool_name="scipy_solver",
        query="stiff_ode_convergence",
        content={"drift": 1e-6},
        discovered_by_bee="hivemind/bee-rusty-sundials",
        ttl_s=0.05,
    )

    # Immediate access works
    assert cache.get("scipy_solver", "stiff_ode_convergence") is not None

    # Wait for expiration
    time.sleep(0.08)
    assert cache.get("scipy_solver", "stiff_ode_convergence") is None
    assert cache.stats()["active_entries"] == 0


def test_kafka_food_source_event_builder():
    """Verify CloudEvent structure matches AMCP v1.5 requirements."""
    cache = SwarmNectarCache()
    entry = cache.put(
        tool_name="json_extractor",
        query="battery_spec_v1",
        content={"capacity": "450Wh/kg"},
        discovered_by_bee="hivemind/qwen35-4b",
    )

    ev = cache.build_kafka_food_source_event(entry)
    assert ev["topic"] == TOPIC_FOOD_SOURCES
    assert ev["event_type"] == "io.hivemind.mesh.food_source_updated"
    assert ev["nectar_id"] == entry.nectar_id
    assert ev["discovered_by_bee"] == "hivemind/qwen35-4b"
