# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Timing run for the expert registry, residency planner and hierarchical router at 100 / 1000 / 5000 experts.

Not collected by pytest (no ``test_`` prefix). Run:  PYTHONPATH=src python tests/bench_expert_registry_scale.py
The experts are SYNTHETIC: generated ids, keywords and adapter hashes (sha256 of the id), sizes and Zipf demand drawn
from a seeded RNG. Only the wall-clock timings are measurements; nothing here says how good any real expert is.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from hivemind.core.expert_registry import (
    LIVE,
    LORA_DOMAIN,
    RETIRED,
    SHADOW,
    ExpertRecord,
    ExpertRegistry,
    HierarchicalExpertRouter,
    ServingNode,
    plan_adapter_residency,
    validate_residency_plan,
)

BASE = "Qwen/Qwen3-8B"


def bench(n_experts: int, n_requests: int = 2000, seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)
    n_clusters = max(8, n_experts // 50)
    t0 = time.perf_counter()
    reg = ExpertRegistry()
    for i in range(n_experts):
        c = i % n_clusters
        eid = f"syn-{i}"
        reg.register(ExpertRecord(eid, LORA_DOMAIN, f"cl{c}", BASE, hashlib.sha256(eid.encode()).hexdigest(),
                                  rng.randint(30, 160), keywords=(f"topic{c}", f"kw{i}", f"kw{(i * 7) % n_experts}")))
    t_register = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(n_experts):
        eid = f"syn-{i}"
        reg.transition(eid, SHADOW, "bench")
        r = i % 10
        if r < 8:
            reg.transition(eid, LIVE, "bench")
        elif r == 8:
            reg.transition(eid, RETIRED, "bench")
    t_lifecycle = time.perf_counter() - t0

    adapters = reg.residency_inputs()
    ranked = sorted(adapters)
    rng.shuffle(ranked)
    demand = {a: 1.0 / (k + 1) for k, a in enumerate(ranked)}  # Zipf(1) over a random order
    nodes = [ServingNode(f"node{j}", 15360, BASE, 9000, 32) for j in range(16)]
    t0 = time.perf_counter()
    plan = plan_adapter_residency(nodes, adapters, demand)
    t_plan = time.perf_counter() - t0
    violations = validate_residency_plan(plan, nodes, adapters)

    t0 = time.perf_counter()
    router = HierarchicalExpertRouter(reg, BASE, resident=plan.hot)
    t_router_build = time.perf_counter() - t0
    prompts: List[str] = []
    for _ in range(n_requests):
        words = [f"topic{rng.randrange(n_clusters)}", f"kw{rng.randrange(n_experts)}", "please", "explain"]
        prompts.append(" ".join(words))
    t0 = time.perf_counter()
    reasons: Dict[str, int] = {}
    for p in prompts:
        reason = router.route(p).reason
        reasons[reason] = reasons.get(reason, 0) + 1
    t_route = time.perf_counter() - t0
    for p in prompts:  # property re-checked on every decision of the run
        d = router.route(p)
        assert d.expert_id is None or reg.get(d.expert_id).state == LIVE

    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        path = reg.save(Path(tmp) / "registry.json")
        t_save = time.perf_counter() - t0
        size = path.stat().st_size
        t0 = time.perf_counter()
        ExpertRegistry.load(path)
        t_load = time.perf_counter() - t0

    return {
        "experts": n_experts,
        "clusters": n_clusters,
        "live": len(reg.ids_in_state(LIVE)),
        "register_s": round(t_register, 4),
        "lifecycle_s": round(t_lifecycle, 4),
        "plan_s": round(t_plan, 4),
        "plan_ordering": plan.ordering,
        "hot": len(plan.hot),
        "served_demand_fraction": round(plan.served_demand_fraction, 4),
        "plan_violations": len(violations),
        "router_build_s": round(t_router_build, 4),
        "route_requests": n_requests,
        "route_us_per_request": round(1e6 * t_route / n_requests, 1),
        "route_reasons": dict(sorted(reasons.items())),
        "save_s": round(t_save, 4),
        "load_s": round(t_load, 4),
        "json_bytes": size,
    }


if __name__ == "__main__":
    print(json.dumps({"python": sys.version.split()[0], "machine": platform.machine(), "processor": platform.processor(),
                      "platform": platform.platform()}))
    for n in (100, 1000, 5000):
        print(json.dumps(bench(n)))
