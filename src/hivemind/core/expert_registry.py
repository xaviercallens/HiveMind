# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Season 21 (`HM-S21-MOESCALE`): an expert registry, adapter residency planner and hierarchical router for many experts.

The Season 3-5 tables (``SEASON_3_DOMAIN_EXPERT_SPECS``, ``SERVING_EXPERT_SPECS``) are Python dicts with a handful of
entries, routed by scoring every keyword of every domain on every prompt. That does not survive hundreds or thousands of
experts: there is no lifecycle (a half-evaluated adapter is as routable as a proven one), no binding between an adapter
and the base model it was trained on, and no notion of which adapters actually fit in a GPU at once. This module adds the
three missing pieces, all pure Python with no heavy imports:

1. :class:`ExpertRegistry` holds experts of every kind (``lora_domain`` adapters, ``serving`` experts that call a model
   server, ``byob`` bring-your-own-bee candidates) under one lifecycle ``candidate -> shadow -> live -> retired`` with only
   the transitions in :data:`LEGAL_TRANSITIONS`. Adapters are content-addressed by ``(base_model_id, sha256)``: the same
   bytes cannot be registered twice, and :meth:`ExpertRegistry.check_attachable` refuses to put an adapter on any base
   but its own. The registry persists to JSON with an atomic write (temp file, fsync, ``os.replace``) and re-validates the
   whole file, including that every expert's transition history replays legally, when it loads.
2. :func:`plan_adapter_residency` decides which adapters are hot (resident in a node's multi-LoRA slots, as llama-server
   ``--lora`` or vLLM ``--max-loras`` expose them) and which are cold (loaded on demand). It is a deterministic greedy
   bin packing run under two orders (demand, then demand per MiB) keeping whichever serves more demand. Greedy is not
   optimal for this knapsack-like problem; :func:`validate_residency_plan` checks the hard constraints (VRAM, slots, base
   match, each adapter at most once) independently of the planner.
3. :class:`HierarchicalExpertRouter` routes ``prompt -> cluster -> expert``. The cluster is chosen by whole-word keyword
   hits (same semantics as ``btm_moe.keyword_route``, but each distinct keyword is tested once per prompt, single-word
   keywords by a set lookup). Inside the cluster, QueenBee recruitment weights (``RecruitmentReport.weights``) rank the
   live experts when given, keyword hits otherwise. A cold winner falls back to the best resident live expert of the
   same cluster, then to the base model. Only ``live`` experts are ever candidates.

Nothing here loads an adapter or calls a model; the planner and router are bookkeeping over the inputs they are given.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import (
    Any,
    Collection,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

REGISTRY_SCHEMA = "hivemind-expert-registry/v1"

CANDIDATE, SHADOW, LIVE, RETIRED = "candidate", "shadow", "live", "retired"
LIFECYCLE_STATES: Tuple[str, ...] = (CANDIDATE, SHADOW, LIVE, RETIRED)
# shadow -> candidate sends an adapter back for another evaluation round; live -> shadow demotes without retiring.
LEGAL_TRANSITIONS: Dict[str, frozenset[str]] = {
    CANDIDATE: frozenset({SHADOW, RETIRED}),
    SHADOW: frozenset({LIVE, CANDIDATE, RETIRED}),
    LIVE: frozenset({SHADOW, RETIRED}),
    RETIRED: frozenset(),
}

LORA_DOMAIN, SERVING, BYOB = "lora_domain", "serving", "byob"
EXPERT_KINDS: frozenset[str] = frozenset({LORA_DOMAIN, SERVING, BYOB})
ADAPTER_KINDS: frozenset[str] = frozenset({LORA_DOMAIN, BYOB})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SIMPLE_WORD_RE = re.compile(r"^[a-z0-9_]+$")
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


class IllegalTransitionError(ValueError):
    """A lifecycle change that :data:`LEGAL_TRANSITIONS` does not allow."""


class AdapterBaseMismatchError(ValueError):
    """An adapter offered to a base model other than the one it was trained on."""


@dataclass(frozen=True)
class ExpertRecord:
    """One expert. Adapter kinds carry ``adapter_sha256`` and ``adapter_size_mib``; a serving expert carries neither."""

    expert_id: str
    kind: str
    cluster_id: str
    base_model_id: str
    adapter_sha256: Optional[str] = None
    adapter_size_mib: int = 0
    keywords: Tuple[str, ...] = ()
    source: str = ""
    state: str = CANDIDATE
    history: Tuple[Tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.expert_id or not self.cluster_id or not self.base_model_id:
            raise ValueError("expert_id, cluster_id and base_model_id must be non-empty")
        if self.kind not in EXPERT_KINDS:
            raise ValueError(f"unknown expert kind {self.kind!r}; expected one of {sorted(EXPERT_KINDS)}")
        if self.state not in LIFECYCLE_STATES:
            raise ValueError(f"unknown lifecycle state {self.state!r}")
        if self.kind in ADAPTER_KINDS:
            if self.adapter_sha256 is None or not _SHA256_RE.match(self.adapter_sha256):
                raise ValueError(f"{self.kind} expert {self.expert_id!r} needs a lowercase hex sha256 of its adapter")
            if isinstance(self.adapter_size_mib, bool) or not isinstance(self.adapter_size_mib, int) or self.adapter_size_mib <= 0:
                raise ValueError(f"{self.kind} expert {self.expert_id!r} needs an integer adapter_size_mib > 0")
        elif self.adapter_sha256 is not None or self.adapter_size_mib != 0:
            raise ValueError(f"serving expert {self.expert_id!r} has no adapter: adapter_sha256 must be None and size 0")
        object.__setattr__(self, "keywords", tuple(dict.fromkeys(k.strip().lower() for k in self.keywords if k.strip())))
        object.__setattr__(self, "history", tuple(tuple(h) for h in self.history))  # type: ignore[misc]
        replay_history(self.history, self.state)

    @property
    def has_adapter(self) -> bool:
        return self.kind in ADAPTER_KINDS

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["keywords"] = list(self.keywords)
        data["history"] = [list(h) for h in self.history]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExpertRecord":
        return cls(
            expert_id=str(data["expert_id"]),
            kind=str(data["kind"]),
            cluster_id=str(data["cluster_id"]),
            base_model_id=str(data["base_model_id"]),
            adapter_sha256=data.get("adapter_sha256"),
            adapter_size_mib=data.get("adapter_size_mib", 0),
            keywords=tuple(str(k) for k in data.get("keywords", ())),
            source=str(data.get("source", "")),
            state=str(data["state"]),
            history=tuple((str(a), str(b), str(c)) for a, b, c in data.get("history", ())),
        )


def replay_history(history: Sequence[Tuple[str, str, str]], final_state: str) -> None:
    """Raise :class:`IllegalTransitionError` unless ``history`` is a legal chain from ``candidate`` to ``final_state``."""
    state = CANDIDATE
    for step in history:
        if len(step) != 3:
            raise IllegalTransitionError(f"history entry must be (from, to, reason), got {step!r}")
        src, dst, _reason = step
        if src != state or dst not in LEGAL_TRANSITIONS.get(src, frozenset()):
            raise IllegalTransitionError(f"history step {src!r} -> {dst!r} is illegal after state {state!r}")
        state = dst
    if state != final_state:
        raise IllegalTransitionError(f"history ends in {state!r} but the record says {final_state!r}")


class ExpertRegistry:
    """Experts indexed by id, state, cluster and adapter content hash."""

    def __init__(self) -> None:
        self._experts: Dict[str, ExpertRecord] = {}
        self._by_state: Dict[str, set[str]] = {s: set() for s in LIFECYCLE_STATES}
        self._by_cluster: Dict[str, set[str]] = {}
        self._by_sha: Dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._experts)

    def __contains__(self, expert_id: object) -> bool:
        return expert_id in self._experts

    def _index(self, rec: ExpertRecord) -> None:
        self._experts[rec.expert_id] = rec
        self._by_state[rec.state].add(rec.expert_id)
        self._by_cluster.setdefault(rec.cluster_id, set()).add(rec.expert_id)
        if rec.adapter_sha256 is not None:
            self._by_sha[rec.adapter_sha256] = rec.expert_id

    def _admit(self, rec: ExpertRecord) -> None:
        if rec.expert_id in self._experts:
            raise ValueError(f"expert {rec.expert_id!r} is already registered")
        if rec.adapter_sha256 is not None and rec.adapter_sha256 in self._by_sha:
            owner = self._experts[self._by_sha[rec.adapter_sha256]]
            if owner.base_model_id != rec.base_model_id:
                raise AdapterBaseMismatchError(
                    f"adapter {rec.adapter_sha256[:12]} is bound to base {owner.base_model_id!r}, not {rec.base_model_id!r}"
                )
            raise ValueError(f"adapter {rec.adapter_sha256[:12]} is already registered as {owner.expert_id!r}")
        self._index(rec)

    def register(self, record: ExpertRecord) -> ExpertRecord:
        """Add a new expert. It always enters as ``candidate`` with an empty history (a promoted record is reset)."""
        rec = replace(record, state=CANDIDATE, history=())
        self._admit(rec)
        return rec

    def get(self, expert_id: str) -> ExpertRecord:
        return self._experts[expert_id]

    def transition(self, expert_id: str, to_state: str, reason: str) -> ExpertRecord:
        """Move an expert along the lifecycle; raises :class:`IllegalTransitionError` for anything not allowed."""
        rec = self._experts[expert_id]
        if to_state not in LEGAL_TRANSITIONS[rec.state]:
            raise IllegalTransitionError(f"{expert_id!r}: {rec.state!r} -> {to_state!r} is not a legal transition")
        if not reason.strip():
            raise ValueError("a transition needs a reason")
        new = replace(rec, state=to_state, history=rec.history + ((rec.state, to_state, reason.strip()),))
        self._by_state[rec.state].discard(expert_id)
        self._by_state[to_state].add(expert_id)
        self._experts[expert_id] = new
        return new

    def ids_in_state(self, state: str) -> List[str]:
        if state not in LIFECYCLE_STATES:
            raise ValueError(f"unknown lifecycle state {state!r}")
        return sorted(self._by_state[state])

    def clusters(self) -> List[str]:
        return sorted(self._by_cluster)

    def experts_in_cluster(self, cluster_id: str, state: Optional[str] = None) -> List[ExpertRecord]:
        ids = self._by_cluster.get(cluster_id, set())
        if state is not None:
            ids = ids & self._by_state[state]
        return [self._experts[i] for i in sorted(ids)]

    def all_experts(self) -> List[ExpertRecord]:
        return [self._experts[i] for i in sorted(self._experts)]

    def find_by_adapter(self, adapter_sha256: str) -> Optional[ExpertRecord]:
        owner = self._by_sha.get(adapter_sha256)
        return None if owner is None else self._experts[owner]

    def check_attachable(self, expert_id: str, base_model_id: str) -> ExpertRecord:
        """Return the expert if its adapter may be attached to ``base_model_id``; raise otherwise."""
        rec = self._experts[expert_id]
        if not rec.has_adapter:
            raise ValueError(f"{expert_id!r} is a serving expert and has no adapter to attach")
        if rec.base_model_id != base_model_id:
            raise AdapterBaseMismatchError(f"{expert_id!r} was trained on {rec.base_model_id!r}, not {base_model_id!r}")
        return rec

    def verify_adapter_bytes(self, expert_id: str, data: bytes) -> bool:
        """True when ``data`` hashes to the sha256 the expert was registered with."""
        rec = self._experts[expert_id]
        return rec.adapter_sha256 is not None and hashlib.sha256(data).hexdigest() == rec.adapter_sha256

    def residency_inputs(self, states: Collection[str] = (LIVE,)) -> Dict[str, "AdapterFootprint"]:
        """Footprints of the adapter experts in ``states``, ready for :func:`plan_adapter_residency`."""
        out: Dict[str, AdapterFootprint] = {}
        for state in sorted(states):
            for eid in self.ids_in_state(state):
                rec = self._experts[eid]
                if rec.has_adapter:
                    out[eid] = AdapterFootprint(size_mib=rec.adapter_size_mib, base_model_id=rec.base_model_id)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"schema": REGISTRY_SCHEMA, "experts": [r.to_dict() for r in self.all_experts()]}

    def save(self, path: Path | str) -> Path:
        """Write the registry as JSON atomically: a crash leaves either the old file or the new one, never half of one."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), sort_keys=True, indent=1).encode("utf-8")
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, target)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        return target

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExpertRegistry":
        if data.get("schema") != REGISTRY_SCHEMA:
            raise ValueError(f"not an expert registry file (schema {data.get('schema')!r})")
        reg = cls()
        for item in data.get("experts", []):
            reg._admit(ExpertRecord.from_dict(item))
        return reg

    @classmethod
    def load(cls, path: Path | str) -> "ExpertRegistry":
        """Read and fully re-validate a saved registry (kinds, hashes, duplicates, history replay)."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def import_serving_experts(self, base_model_id: str, cluster_id: str = "code") -> List[str]:
        """Register every entry of ``serving_experts.SERVING_EXPERT_SPECS`` not yet present, as a ``serving`` candidate."""
        from hivemind.core.serving_experts import (
            SERVING_EXPERT_SPECS,  # imports torch via btm_moe, so only on demand
        )

        added: List[str] = []
        for name, spec in sorted(SERVING_EXPERT_SPECS.items()):
            if name in self._experts:
                continue
            self.register(ExpertRecord(name, SERVING, cluster_id, base_model_id, keywords=tuple(spec["keywords"]),
                                       source="serving_experts.SERVING_EXPERT_SPECS"))
            added.append(name)
        return added


# ---------------------------------------------------------------------------
# Residency planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterFootprint:
    """What placing one adapter costs: its VRAM and the base model it must sit on."""

    size_mib: int
    base_model_id: str

    def __post_init__(self) -> None:
        if isinstance(self.size_mib, bool) or not isinstance(self.size_mib, int) or self.size_mib <= 0:
            raise ValueError("size_mib must be an integer > 0")
        if not self.base_model_id:
            raise ValueError("base_model_id must be non-empty")


@dataclass(frozen=True)
class ServingNode:
    """A GPU serving one resident base model with up to ``max_lora_slots`` adapters loaded beside it.

    ``base_resident_mib`` is everything that is not an adapter: base weights, KV cache reserve, runtime overhead.
    """

    node_id: str
    vram_mib: int
    base_model_id: str
    base_resident_mib: int
    max_lora_slots: int

    def __post_init__(self) -> None:
        if not self.node_id or not self.base_model_id:
            raise ValueError("node_id and base_model_id must be non-empty")
        for name in ("vram_mib", "base_resident_mib", "max_lora_slots"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")

    @property
    def adapter_budget_mib(self) -> int:
        return self.vram_mib - self.base_resident_mib


@dataclass(frozen=True)
class ResidencyPlan:
    """Hot set per node, cold adapters with the reason they are cold, and how much demand the hot set serves."""

    placements: Mapping[str, Tuple[str, ...]]
    hot: Mapping[str, str]
    cold: Mapping[str, str]
    node_free_mib: Mapping[str, int]
    node_free_slots: Mapping[str, int]
    blocked_nodes: Mapping[str, str]
    placed_demand: float
    total_demand: float
    ordering: str
    unknown_demand_ids: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def served_demand_fraction(self) -> float:
        return self.placed_demand / self.total_demand if self.total_demand > 0.0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "placements": {n: list(a) for n, a in sorted(self.placements.items())},
            "cold": dict(sorted(self.cold.items())),
            "node_free_mib": dict(sorted(self.node_free_mib.items())),
            "node_free_slots": dict(sorted(self.node_free_slots.items())),
            "blocked_nodes": dict(sorted(self.blocked_nodes.items())),
            "placed_demand": self.placed_demand,
            "total_demand": self.total_demand,
            "served_demand_fraction": self.served_demand_fraction,
            "ordering": self.ordering,
            "unknown_demand_ids": list(self.unknown_demand_ids),
        }


def _greedy_place(
    order: Sequence[str],
    nodes: Sequence[ServingNode],
    adapters: Mapping[str, AdapterFootprint],
    demand: Mapping[str, float],
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, int], Dict[str, int]]:
    usable = [n for n in nodes if n.adapter_budget_mib >= 0]
    free = {n.node_id: n.adapter_budget_mib for n in usable}
    slots = {n.node_id: n.max_lora_slots for n in usable}
    by_base: Dict[str, List[str]] = {}
    for n in usable:
        by_base.setdefault(n.base_model_id, []).append(n.node_id)
    hot: Dict[str, str] = {}
    cold: Dict[str, str] = {}
    for aid in order:
        fp = adapters[aid]
        if demand.get(aid, 0.0) <= 0.0:
            cold[aid] = "zero_demand"
            continue
        same_base = by_base.get(fp.base_model_id, [])
        if not same_base:
            cold[aid] = "no_node_with_base"
            continue
        fits = [nid for nid in same_base if slots[nid] > 0 and free[nid] >= fp.size_mib]
        if not fits:
            cold[aid] = "no_capacity"
            continue
        best = min(fits, key=lambda nid: (-free[nid], nid))  # worst fit: spread load, keep big holes for big adapters
        hot[aid] = best
        free[best] -= fp.size_mib
        slots[best] -= 1
    return hot, cold, free, slots


def plan_adapter_residency(
    nodes: Sequence[ServingNode],
    adapters: Mapping[str, AdapterFootprint],
    demand: Mapping[str, float],
) -> ResidencyPlan:
    """Choose the hot adapter set per node by demand-weighted greedy bin packing.

    Each adapter is placed at most once, only on a node holding its base model, never past the node's VRAM budget
    (``vram_mib - base_resident_mib``) or its LoRA slot count. Two deterministic orders are tried, by demand and by demand
    per MiB, and the one serving more demand is kept (ties keep the demand order). Adapters with zero demand stay cold.
    A node whose base alone exceeds its VRAM is listed in ``blocked_nodes`` and receives nothing.
    """
    node_ids = [n.node_id for n in nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("node ids must be unique")
    for aid, d in demand.items():
        if not isinstance(d, (int, float)) or isinstance(d, bool) or not math.isfinite(float(d)) or d < 0:
            raise ValueError(f"demand for {aid!r} must be a finite number >= 0, got {d!r}")
    blocked = {n.node_id: f"base_resident_mib {n.base_resident_mib} exceeds vram_mib {n.vram_mib}"
               for n in nodes if n.adapter_budget_mib < 0}
    total = float(sum(float(demand.get(a, 0.0)) for a in adapters))
    orders = {
        "demand": sorted(adapters, key=lambda a: (-float(demand.get(a, 0.0)), adapters[a].size_mib, a)),
        "demand_per_mib": sorted(adapters, key=lambda a: (-float(demand.get(a, 0.0)) / adapters[a].size_mib, a)),
    }
    plans: List[ResidencyPlan] = []
    for name, order in orders.items():
        hot, cold, free, slots = _greedy_place(order, nodes, adapters, demand)
        placed = float(sum(float(demand.get(a, 0.0)) for a in hot))
        placements = {nid: tuple(sorted(a for a, n in hot.items() if n == nid)) for nid in free}
        plan = ResidencyPlan(placements, hot, cold, free, slots, blocked, placed, total, name,
                             tuple(sorted(set(demand) - set(adapters))))
        plans.append(plan)
    best = plans[0]
    for plan in plans[1:]:
        if plan.placed_demand > best.placed_demand:
            best = plan
    return best


def validate_residency_plan(
    plan: ResidencyPlan, nodes: Sequence[ServingNode], adapters: Mapping[str, AdapterFootprint]
) -> List[str]:
    """Independent check of a plan's hard constraints; returns the violations (empty when the plan is sound)."""
    violations: List[str] = []
    by_id = {n.node_id: n for n in nodes}
    seen: Dict[str, str] = {}
    for nid, placed in plan.placements.items():
        node = by_id.get(nid)
        if node is None:
            violations.append(f"unknown node {nid!r}")
            continue
        if nid in plan.blocked_nodes and placed:
            violations.append(f"blocked node {nid!r} holds adapters")
        if len(placed) > node.max_lora_slots:
            violations.append(f"node {nid!r} holds {len(placed)} adapters but has {node.max_lora_slots} slots")
        used = 0
        for aid in placed:
            fp = adapters.get(aid)
            if fp is None:
                violations.append(f"unknown adapter {aid!r} on {nid!r}")
                continue
            if aid in seen:
                violations.append(f"adapter {aid!r} placed on both {seen[aid]!r} and {nid!r}")
            seen[aid] = nid
            if fp.base_model_id != node.base_model_id:
                violations.append(f"adapter {aid!r} (base {fp.base_model_id!r}) on node {nid!r} (base {node.base_model_id!r})")
            used += fp.size_mib
        if used > node.adapter_budget_mib:
            violations.append(f"node {nid!r} uses {used} MiB of adapters over its {node.adapter_budget_mib} MiB budget")
    overlap = set(seen) & set(plan.cold)
    if overlap:
        violations.append(f"adapters both hot and cold: {sorted(overlap)}")
    return violations


# ---------------------------------------------------------------------------
# Hierarchical routing
# ---------------------------------------------------------------------------


def _boundary_pattern(kw: str) -> "re.Pattern[str]":
    head = r"(?<![a-z0-9_])" if (kw[0].isalnum() or kw[0] == "_") else ""
    tail = r"(?![a-z0-9_])" if (kw[-1].isalnum() or kw[-1] == "_") else ""
    return re.compile(head + re.escape(kw) + tail)


def keyword_hits(prompt_lower: str, keywords: Iterable[str]) -> frozenset[str]:
    """Keywords occurring in ``prompt_lower`` as whole words, with ``btm_moe._keyword_hit`` semantics.

    A keyword made only of ``[a-z0-9_]`` is a hit exactly when it is one maximal ``[a-z0-9_]+`` run of the prompt, so it is
    found by a set lookup; other keywords (phrases, punctuation) use the boundary regex.
    """
    tokens: Optional[set[str]] = None
    hits: set[str] = set()
    for kw in keywords:
        if not kw:
            continue
        if _SIMPLE_WORD_RE.match(kw):
            if tokens is None:
                tokens = set(_TOKEN_RE.findall(prompt_lower))
            if kw in tokens:
                hits.add(kw)
            continue
        if _boundary_pattern(kw).search(prompt_lower) is not None:
            hits.add(kw)
    return frozenset(hits)


@dataclass(frozen=True)
class RouteDecision:
    """Where one request goes. ``expert_id`` is None exactly when the target is the base model."""

    target: str  # "expert" or "base"
    expert_id: Optional[str]
    cluster_id: Optional[str]
    reason: str
    load_on_demand: bool = False


class HierarchicalExpertRouter:
    """Two-level router ``cluster -> expert`` over the live experts of a registry snapshot.

    ``resident`` names the adapter experts that are hot (e.g. ``plan.hot``); serving experts are always resident (they
    are model servers, not adapters). ``resident=None`` treats every live expert as resident (single-node setting).
    Only adapters bound to ``base_model_id`` are eligible. Rebuild the router after registry changes; it is a snapshot.
    """

    def __init__(
        self,
        registry: ExpertRegistry,
        base_model_id: str,
        resident: Optional[Collection[str]] = None,
        allow_cold_load: bool = False,
    ) -> None:
        if not base_model_id:
            raise ValueError("base_model_id must be non-empty")
        self.base_model_id = base_model_id
        self.allow_cold_load = allow_cold_load
        self._live: Dict[str, List[ExpertRecord]] = {}
        self._resident: set[str] = set()
        self._cluster_keywords: Dict[str, frozenset[str]] = {}
        resident_set = None if resident is None else set(resident)
        for eid in registry.ids_in_state(LIVE):
            rec = registry.get(eid)
            if rec.has_adapter and rec.base_model_id != base_model_id:
                continue
            self._live.setdefault(rec.cluster_id, []).append(rec)
            if not rec.has_adapter or resident_set is None or eid in resident_set:
                self._resident.add(eid)
        kw_clusters: Dict[str, List[str]] = {}
        for cid in sorted(self._live):
            kws = frozenset(k for r in self._live[cid] for k in r.keywords)
            self._cluster_keywords[cid] = kws
            for k in kws:
                kw_clusters.setdefault(k, []).append(cid)
        # Inverted index: a request costs O(prompt tokens + phrase keywords), not O(all keywords of all experts).
        self._kw_clusters: Dict[str, Tuple[str, ...]] = {k: tuple(v) for k, v in kw_clusters.items()}
        self._simple_keywords = frozenset(k for k in kw_clusters if _SIMPLE_WORD_RE.match(k))
        self._phrase_patterns = [(k, _boundary_pattern(k)) for k in sorted(kw_clusters) if not _SIMPLE_WORD_RE.match(k)]

    def prompt_hits(self, prompt: str) -> frozenset[str]:
        """Keywords of live experts that ``prompt`` hits; equal to ``keyword_hits`` over all of them, but indexed."""
        low = prompt.lower()
        hits = set(_TOKEN_RE.findall(low)) & self._simple_keywords
        hits.update(k for k, pat in self._phrase_patterns if pat.search(low) is not None)
        return frozenset(hits)

    @property
    def live_count(self) -> int:
        return sum(len(v) for v in self._live.values())

    def select_cluster(self, prompt: str) -> Tuple[Optional[str], frozenset[str]]:
        """Cluster with the most distinct keyword hits (ties by name), or None when nothing matches."""
        hits = self.prompt_hits(prompt)
        if not hits:
            return None, hits
        score: Dict[str, int] = {}
        for k in hits:
            for cid in self._kw_clusters[k]:
                score[cid] = score.get(cid, 0) + 1
        best = min(score, key=lambda c: (-score[c], c))
        return best, hits

    def route(
        self,
        prompt: str,
        cluster_id: Optional[str] = None,
        recruitment: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> RouteDecision:
        """Route one request. ``recruitment`` is ``RecruitmentReport.weights`` (cluster -> expert -> weight)."""
        if cluster_id is None:
            cluster_id, hits = self.select_cluster(prompt)
            if cluster_id is None:
                return RouteDecision("base", None, None, "no_cluster_match")
        else:
            hits = self.prompt_hits(prompt) & self._cluster_keywords.get(cluster_id, frozenset())
        candidates = self._live.get(cluster_id, [])
        if not candidates:
            return RouteDecision("base", None, cluster_id, "no_live_expert_in_cluster")
        weights = (recruitment or {}).get(cluster_id, {})
        ranked = sorted(
            candidates,
            key=lambda r: (-float(weights.get(r.expert_id, 0.0)), -len(hits.intersection(r.keywords)), r.expert_id),
        )
        top = ranked[0]
        if top.expert_id in self._resident:
            return RouteDecision("expert", top.expert_id, cluster_id, "best_resident")
        if self.allow_cold_load:
            return RouteDecision("expert", top.expert_id, cluster_id, "cold_load_on_demand", load_on_demand=True)
        for rec in ranked[1:]:
            if rec.expert_id in self._resident:
                return RouteDecision("expert", rec.expert_id, cluster_id, "fallback_resident_same_cluster")
        return RouteDecision("base", None, cluster_id, "no_resident_expert_in_cluster")
