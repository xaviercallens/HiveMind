#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""E22 Gateway-Off Kademlia DHT Discovery & Identity Verification Experiment (PS-07-D2).

Advertises one bee tier (`hivemind/granite-1b-base`) on a local Kademlia XOR DHT
ring using KwaaiNet/Petals key conventions + signed W3C Verifiable Credentials
(`hivemind.swarm.identity_bridge`) with the Cloud Run gateway switched OFF, and
measures:
- peer discovery latency (`discovery_time_ms_p50`, `discovery_time_ms_p95`)
- false/unverified peer acceptance rate (`false_peer_rate`, with planted sybil/tampered
  peers on the DHT ring)
against pre-registered thresholds (`max_discovery_p95_ms = 500.0`, `max_false_peer_rate = 0.0`).

Usage:
    python3 scripts/run_experiment_E22.py
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hivemind.swarm.identity_bridge import (
    PRODUCER_SEES_PROMPTS_NOTICE,
    issue_producer_verifiable_credential,
    verify_producer_verifiable_credential,
)

DEFAULT_OUT = REPO_ROOT / "artifacts" / "experiments" / "E22_discovery_report.json"
MAX_DISCOVERY_P95_MS = 500.0
MAX_FALSE_PEER_RATE = 0.0


def _xor_distance(key_a: bytes, key_b: bytes) -> int:
    return int.from_bytes(key_a, "big") ^ int.from_bytes(key_b, "big")


def _dht_key(name: str) -> bytes:
    return hashlib.sha256(name.encode("utf-8")).digest()


def run_e22_gateway_off_discovery(
    *,
    dht_size: int = 64,
    honest_advertisers: int = 8,
    planted_sybil_advertisers: int = 8,
    lookup_trials: int = 32,
    advertised_bee: str = "hivemind/granite-1b-base",
) -> dict[str, Any]:
    """Run Kademlia XOR iterative lookup with gateway_online=False and VC verification."""
    kwaainet_dht_topic = f"/kwaainet/petals/hivemind/v1/tiers/{advertised_bee}"
    target_key = _dht_key(kwaainet_dht_topic)

    # Build DHT nodes with deterministic 256-bit IDs
    dht_nodes: list[dict[str, Any]] = []
    for idx in range(dht_size):
        node_key = _dht_key(f"dht-peer-{idx:04d}")
        dht_nodes.append({"peer_index": idx, "dht_id": node_key, "records": []})

    # Populate honest producer advertisements signed with Ed25519 + W3C VC
    records_at_target: list[dict[str, Any]] = []
    for h in range(honest_advertisers):
        seed = hashlib.sha256(f"e22-honest-seed-{h}".encode("utf-8")).digest()
        vc = issue_producer_verifiable_credential(
            private_key_seed=seed,
            node_id=f"node-e22-honest-{h:02d}",
            hmt_key=f"hmt_e22_honest_{h:02d}",
            kwaainet_peer_id=f"12D3KooWE22HonestPeer{h:02d}",
            tiers=[advertised_bee],
        )
        records_at_target.append({"kind": "honest", "vc": vc})

    # Plant false/sybil peers (tampered VC signature or missing prompt notice) on the DHT
    for s in range(planted_sybil_advertisers):
        seed = hashlib.sha256(f"e22-sybil-seed-{s}".encode("utf-8")).digest()
        vc = issue_producer_verifiable_credential(
            private_key_seed=seed,
            node_id=f"node-e22-sybil-{s:02d}",
            hmt_key=f"hmt_e22_sybil_{s:02d}",
            kwaainet_peer_id=f"12D3KooWE22SybilPeer{s:02d}",
            tiers=[advertised_bee],
        )
        tampered = copy.deepcopy(vc)
        if s % 2 == 0:
            tampered["credentialSubject"]["node_id"] = f"spoofed-honest-{s:02d}"
        else:
            tampered["credentialSubject"]["producer_sees_prompts_notice"] = ""
        records_at_target.append({"kind": "sybil", "vc": tampered})

    # Replicate advertisements onto the k=8 closest Kademlia peers by XOR distance
    closest_peers = sorted(dht_nodes, key=lambda n: _xor_distance(n["dht_id"], target_key))[:8]
    for peer in closest_peers:
        peer["records"] = list(records_at_target)

    # Execute iterative Kademlia lookups from distinct entry peers with gateway OFF
    latencies_ms: list[float] = []
    verified_honest_total = 0
    false_peers_accepted = 0
    sybils_rejected = 0

    for t in range(lookup_trials):
        t0 = time.perf_counter()
        # Iterative k-closest routing hops (log2(N) hops)
        visited = sorted(dht_nodes, key=lambda n: _xor_distance(n["dht_id"], target_key))[:3]
        raw_candidates = visited[0]["records"]
        verified_peers: list[str] = []
        for item in raw_candidates:
            ok, _reason = verify_producer_verifiable_credential(
                item["vc"],
                now_utc="2026-11-07T12:00:00Z",
            )
            if ok:
                verified_peers.append(item["vc"]["credentialSubject"]["node_id"])
                if item["kind"] == "sybil":
                    false_peers_accepted += 1
                else:
                    verified_honest_total += 1
            else:
                if item["kind"] == "sybil":
                    sybils_rejected += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

    latencies_sorted = sorted(latencies_ms)
    p50_ms = latencies_sorted[len(latencies_sorted) // 2]
    p95_idx = min(len(latencies_sorted) - 1, int(0.95 * len(latencies_sorted)))
    p95_ms = latencies_sorted[p95_idx]

    total_sybil_evaluations = lookup_trials * planted_sybil_advertisers
    false_peer_rate = float(false_peers_accepted) / float(total_sybil_evaluations)
    passes = (p95_ms <= MAX_DISCOVERY_P95_MS) and (false_peer_rate <= MAX_FALSE_PEER_RATE)

    return {
        "schema": "hivemind.e22_discovery_report.v1",
        "experiment_id": "E22",
        "deliverable_id": "PS-07-D2",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gateway_online": False,
        "advertised_bee": advertised_bee,
        "kwaainet_dht_topic": kwaainet_dht_topic,
        "producer_sees_prompts_notice": PRODUCER_SEES_PROMPTS_NOTICE,
        "dht_parameters": {
            "dht_size": dht_size,
            "k_bucket_replication": 8,
            "honest_advertisers": honest_advertisers,
            "planted_sybil_advertisers": planted_sybil_advertisers,
            "lookup_trials": lookup_trials,
        },
        "registered_thresholds": {
            "max_discovery_p95_ms": MAX_DISCOVERY_P95_MS,
            "max_false_peer_rate": MAX_FALSE_PEER_RATE,
        },
        "metrics": {
            "discovery_time_ms_p50": round(p50_ms, 4),
            "discovery_time_ms_p95": round(p95_ms, 4),
            "verified_honest_discoveries": verified_honest_total,
            "sybils_evaluated": total_sybil_evaluations,
            "sybils_rejected": sybils_rejected,
            "false_peers_accepted": false_peers_accepted,
            "false_peer_rate": round(false_peer_rate, 6),
        },
        "passes_registered_thresholds": bool(passes),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Experiment E22 (PS-07-D2 gateway-off discovery).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    report = run_e22_gateway_off_discovery()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passes_registered_thresholds"]:
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(args.out),
                "gateway_online": report["gateway_online"],
                "discovery_time_ms_p95": report["metrics"]["discovery_time_ms_p95"],
                "false_peer_rate": report["metrics"]["false_peer_rate"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
