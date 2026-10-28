"""Expert registry, adapter residency planning and hierarchical routing for many experts (HM-S21-MOESCALE)."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hivemind.core.btm_moe import _keyword_hit
from hivemind.core.expert_registry import (
    BYOB,
    CANDIDATE,
    LIVE,
    LORA_DOMAIN,
    RETIRED,
    SERVING,
    SHADOW,
    AdapterBaseMismatchError,
    AdapterFootprint,
    ExpertRecord,
    ExpertRegistry,
    HierarchicalExpertRouter,
    IllegalTransitionError,
    ResidencyPlan,
    ServingNode,
    keyword_hits,
    plan_adapter_residency,
    replay_history,
    validate_residency_plan,
)

BASE = "Qwen/Qwen3-8B"
OTHER_BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"


def sha(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def lora(eid: str, cluster: str = "code", base: str = BASE, size: int = 100, keywords: tuple[str, ...] = ()) -> ExpertRecord:
    return ExpertRecord(eid, LORA_DOMAIN, cluster, base, adapter_sha256=sha(eid + base), adapter_size_mib=size, keywords=keywords)


def promote(reg: ExpertRegistry, eid: str, to: str) -> None:
    path = {SHADOW: [SHADOW], LIVE: [SHADOW, LIVE], RETIRED: [RETIRED], CANDIDATE: []}[to]
    for step in path:
        reg.transition(eid, step, f"test promotes to {step}")


# ---------------------------------------------------------------- lifecycle


def test_legal_lifecycle_records_history_and_updates_state_index() -> None:
    reg = ExpertRegistry()
    reg.register(lora("rust"))
    reg.transition("rust", SHADOW, "passed offline eval")
    reg.transition("rust", LIVE, "shadow win rate above gate")
    rec = reg.transition("rust", RETIRED, "superseded")
    assert rec.state == RETIRED
    assert [h[:2] for h in rec.history] == [(CANDIDATE, SHADOW), (SHADOW, LIVE), (LIVE, RETIRED)]
    assert reg.ids_in_state(RETIRED) == ["rust"] and reg.ids_in_state(LIVE) == []


@pytest.mark.parametrize("path, bad", [([], LIVE), ([SHADOW, LIVE], CANDIDATE), ([RETIRED], SHADOW), ([RETIRED], LIVE)])
def test_illegal_transitions_are_rejected_and_leave_state_unchanged(path: List[str], bad: str) -> None:
    reg = ExpertRegistry()
    reg.register(lora("x"))
    for step in path:
        reg.transition("x", step, "setup")
    before = reg.get("x").state
    with pytest.raises(IllegalTransitionError):
        reg.transition("x", bad, "should fail")
    assert reg.get("x").state == before
    assert reg.ids_in_state(before) == ["x"]


def test_transition_requires_a_reason_and_shadow_can_go_back_to_candidate() -> None:
    reg = ExpertRegistry()
    reg.register(lora("x"))
    reg.transition("x", SHADOW, "eval")
    with pytest.raises(ValueError, match="reason"):
        reg.transition("x", CANDIDATE, "   ")
    assert reg.transition("x", CANDIDATE, "needs another round").state == CANDIDATE
    with pytest.raises(ValueError, match="unknown lifecycle"):
        reg.ids_in_state("dormant")


def test_register_resets_a_promoted_record_to_candidate() -> None:
    hist = ((CANDIDATE, SHADOW, "a"), (SHADOW, LIVE, "b"))
    rec = ExpertRecord("x", LORA_DOMAIN, "c", BASE, sha("x"), 10, state=LIVE, history=hist)
    reg = ExpertRegistry()
    out = reg.register(rec)
    assert out.state == CANDIDATE and out.history == ()
    assert reg.ids_in_state(CANDIDATE) == ["x"]


def test_replay_history_rejects_forged_chains() -> None:
    replay_history(((CANDIDATE, SHADOW, "a"),), SHADOW)  # positive control: no exception
    with pytest.raises(IllegalTransitionError):
        replay_history((), LIVE)
    with pytest.raises(IllegalTransitionError):
        replay_history(((CANDIDATE, LIVE, "skip shadow"),), LIVE)
    with pytest.raises(IllegalTransitionError):  # each step is legal on its own but the chain is broken
        replay_history(((CANDIDATE, SHADOW, "a"), (LIVE, RETIRED, "b")), RETIRED)
    with pytest.raises(IllegalTransitionError):
        replay_history(((CANDIDATE, SHADOW),), SHADOW)  # type: ignore[arg-type]


# ---------------------------------------------------------------- records and content addressing


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"kind": LORA_DOMAIN, "adapter_sha256": None, "adapter_size_mib": 10}, "sha256"),
        ({"kind": LORA_DOMAIN, "adapter_sha256": sha("a").upper(), "adapter_size_mib": 10}, "sha256"),
        ({"kind": BYOB, "adapter_sha256": sha("a"), "adapter_size_mib": 0}, "adapter_size_mib"),
        ({"kind": BYOB, "adapter_sha256": sha("a"), "adapter_size_mib": True}, "adapter_size_mib"),
        ({"kind": SERVING, "adapter_sha256": sha("a"), "adapter_size_mib": 0}, "no adapter"),
        ({"kind": SERVING, "adapter_sha256": None, "adapter_size_mib": 5}, "no adapter"),
        ({"kind": "wizard", "adapter_sha256": None, "adapter_size_mib": 0}, "unknown expert kind"),
    ],
)
def test_record_validation_rejects_bad_records(kwargs: Dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ExpertRecord("e", cluster_id="c", base_model_id=BASE, **kwargs)  # type: ignore[arg-type]
    ok = ExpertRecord("e", SERVING, "c", BASE)
    assert ok.has_adapter is False and ok.state == CANDIDATE


def test_record_rejects_empty_ids_and_unknown_state_and_normalises_keywords() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ExpertRecord("", SERVING, "c", BASE)
    with pytest.raises(ValueError, match="unknown lifecycle"):
        ExpertRecord("e", SERVING, "c", BASE, state="dormant")
    rec = ExpertRecord("e", SERVING, "c", BASE, keywords=(" Rust ", "rust", "", "Borrow"))
    assert rec.keywords == ("rust", "borrow")


def test_same_adapter_bytes_cannot_be_registered_twice_or_bound_to_another_base() -> None:
    reg = ExpertRegistry()
    reg.register(ExpertRecord("a", LORA_DOMAIN, "c", BASE, sha("blob"), 10))
    with pytest.raises(ValueError, match="already registered as 'a'"):
        reg.register(ExpertRecord("b", BYOB, "c", BASE, sha("blob"), 10))
    with pytest.raises(AdapterBaseMismatchError):
        reg.register(ExpertRecord("c", BYOB, "c", OTHER_BASE, sha("blob"), 10))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(ExpertRecord("a", LORA_DOMAIN, "c", BASE, sha("other"), 10))
    assert len(reg) == 1 and reg.find_by_adapter(sha("blob")).expert_id == "a"  # type: ignore[union-attr]
    assert reg.find_by_adapter(sha("missing")) is None


def test_check_attachable_refuses_a_different_base_and_serving_experts() -> None:
    reg = ExpertRegistry()
    reg.register(lora("a"))
    reg.register(ExpertRecord("srv", SERVING, "code", BASE))
    assert reg.check_attachable("a", BASE).expert_id == "a"
    with pytest.raises(AdapterBaseMismatchError):
        reg.check_attachable("a", OTHER_BASE)
    with pytest.raises(ValueError, match="no adapter"):
        reg.check_attachable("srv", BASE)


def test_verify_adapter_bytes_detects_tampering() -> None:
    reg = ExpertRegistry()
    data = b"lora-a-weights"
    reg.register(ExpertRecord("a", LORA_DOMAIN, "c", BASE, hashlib.sha256(data).hexdigest(), 1))
    reg.register(ExpertRecord("srv", SERVING, "c", BASE))
    assert reg.verify_adapter_bytes("a", data) is True
    assert reg.verify_adapter_bytes("a", data + b"!") is False
    assert reg.verify_adapter_bytes("srv", data) is False


def test_cluster_and_residency_inputs_views() -> None:
    reg = ExpertRegistry()
    for eid, cl in (("a", "code"), ("b", "code"), ("c", "bio")):
        reg.register(lora(eid, cluster=cl, size=7))
    reg.register(ExpertRecord("srv", SERVING, "code", BASE))
    for eid in ("a", "c", "srv"):
        promote(reg, eid, LIVE)
    assert reg.clusters() == ["bio", "code"]
    assert [r.expert_id for r in reg.experts_in_cluster("code", LIVE)] == ["a", "srv"]
    assert [r.expert_id for r in reg.experts_in_cluster("code")] == ["a", "b", "srv"]
    inputs = reg.residency_inputs()
    assert sorted(inputs) == ["a", "c"] and inputs["a"].size_mib == 7  # serving expert has no footprint
    assert "a" in reg and "zz" not in reg


# ---------------------------------------------------------------- persistence


def _small_registry() -> ExpertRegistry:
    reg = ExpertRegistry()
    reg.register(lora("a", keywords=("rust",)))
    reg.register(lora("b", cluster="bio"))
    reg.register(ExpertRecord("srv", SERVING, "code", BASE, keywords=("python",)))
    promote(reg, "a", LIVE)
    promote(reg, "b", RETIRED)
    return reg


def test_save_load_roundtrip_preserves_everything(tmp_path: Path) -> None:
    reg = _small_registry()
    path = reg.save(tmp_path / "sub" / "registry.json")
    loaded = ExpertRegistry.load(path)
    assert loaded.to_dict() == reg.to_dict()
    assert loaded.get("a").state == LIVE and loaded.get("a").history == reg.get("a").history
    assert [p.name for p in path.parent.iterdir()] == ["registry.json"]  # no temp file left behind


def test_load_rejects_forged_state_wrong_schema_and_duplicate_adapters(tmp_path: Path) -> None:
    data = _small_registry().to_dict()
    forged = json.loads(json.dumps(data))
    forged["experts"][1]["state"] = LIVE  # retired expert flipped back to live without history
    (tmp_path / "f.json").write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(IllegalTransitionError):
        ExpertRegistry.load(tmp_path / "f.json")
    with pytest.raises(ValueError, match="schema"):
        ExpertRegistry.from_dict({**data, "schema": "something-else/v9"})
    dup = json.loads(json.dumps(data))
    dup["experts"][1]["adapter_sha256"] = dup["experts"][0]["adapter_sha256"]
    with pytest.raises(ValueError, match="already registered"):
        ExpertRegistry.from_dict(dup)


def test_failed_replace_keeps_the_old_file_and_removes_the_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "registry.json"
    _small_registry().save(target)
    before = target.read_bytes()
    reg = _small_registry()
    reg.register(lora("new"))

    def broken_replace(src: str, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        reg.save(target)
    assert target.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["registry.json"]


def test_import_serving_experts_from_the_existing_table() -> None:
    reg = ExpertRegistry()
    added = reg.import_serving_experts(BASE)
    assert added == ["Python_Code_GWAYA"]
    assert reg.get("Python_Code_GWAYA").kind == SERVING and reg.get("Python_Code_GWAYA").state == CANDIDATE
    assert reg.import_serving_experts(BASE) == []


# ---------------------------------------------------------------- residency planning


def test_planner_respects_slots_vram_and_base() -> None:
    nodes = [ServingNode("t4", 15360, BASE, 15000, 2), ServingNode("other", 24000, OTHER_BASE, 10000, 8)]
    adapters = {f"a{i}": AdapterFootprint(100, BASE) for i in range(5)}
    adapters["big"] = AdapterFootprint(400, BASE)
    adapters["foreign"] = AdapterFootprint(10, "Llama-3")
    demand = {f"a{i}": float(10 - i) for i in range(5)} | {"big": 100.0, "foreign": 5.0}
    plan = plan_adapter_residency(nodes, adapters, demand)
    assert plan.placements["t4"] == ("a0", "a1")  # 360 MiB budget, 2 slots: big (400) cannot fit
    assert plan.cold["big"] == "no_capacity" and plan.cold["foreign"] == "no_node_with_base"
    assert plan.placements["other"] == ()
    assert plan.node_free_mib["t4"] == 160 and plan.node_free_slots["t4"] == 0
    assert validate_residency_plan(plan, nodes, adapters) == []


def test_density_order_wins_when_it_serves_more_demand() -> None:
    nodes = [ServingNode("n", 1100, BASE, 1000, 10)]
    adapters = {"A": AdapterFootprint(100, BASE), "B": AdapterFootprint(50, BASE), "C": AdapterFootprint(50, BASE)}
    plan = plan_adapter_residency(nodes, adapters, {"A": 10.0, "B": 9.0, "C": 9.0})
    assert plan.ordering == "demand_per_mib" and plan.placements["n"] == ("B", "C")
    assert plan.placed_demand == pytest.approx(18.0) and plan.served_demand_fraction == pytest.approx(18 / 28)


def test_planner_zero_demand_blocked_node_and_unknown_ids() -> None:
    nodes = [ServingNode("bad", 8000, BASE, 9000, 4), ServingNode("ok", 8000, BASE, 7000, 4)]
    adapters = {"hot": AdapterFootprint(10, BASE), "idle": AdapterFootprint(10, BASE)}
    plan = plan_adapter_residency(nodes, adapters, {"hot": 1.0, "retired-one": 3.0})
    assert "bad" in plan.blocked_nodes and "bad" not in plan.placements
    assert plan.hot == {"hot": "ok"} and plan.cold == {"idle": "zero_demand"}
    assert plan.unknown_demand_ids == ("retired-one",)
    d = plan.to_dict()
    assert d["placements"] == {"ok": ["hot"]} and d["served_demand_fraction"] == 1.0
    empty = plan_adapter_residency([], {}, {})
    assert empty.served_demand_fraction == 0.0 and empty.hot == {}


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf, True, "3"])
def test_planner_rejects_bad_demand(bad: object) -> None:
    with pytest.raises(ValueError, match="demand"):
        plan_adapter_residency([], {"a": AdapterFootprint(1, BASE)}, {"a": bad})  # type: ignore[dict-item]
    assert plan_adapter_residency([], {"a": AdapterFootprint(1, BASE)}, {"a": 0}).cold == {"a": "zero_demand"}


def test_planner_and_types_reject_bad_nodes_and_footprints() -> None:
    with pytest.raises(ValueError, match="unique"):
        plan_adapter_residency([ServingNode("n", 1, BASE, 0, 1)] * 2, {}, {})
    with pytest.raises(ValueError, match="vram_mib"):
        ServingNode("n", -1, BASE, 0, 1)
    with pytest.raises(ValueError, match="non-empty"):
        ServingNode("", 1, BASE, 0, 1)
    with pytest.raises(ValueError, match="size_mib"):
        AdapterFootprint(0, BASE)
    with pytest.raises(ValueError, match="base_model_id"):
        AdapterFootprint(1, "")
    assert ServingNode("n", 100, BASE, 40, 1).adapter_budget_mib == 60


def test_validator_catches_every_kind_of_bad_plan() -> None:
    nodes = [ServingNode("n", 1000, BASE, 900, 1), ServingNode("blk", 10, BASE, 20, 4)]
    adapters = {"a": AdapterFootprint(80, BASE), "b": AdapterFootprint(80, BASE), "f": AdapterFootprint(5, OTHER_BASE)}
    bad = ResidencyPlan(
        placements={"n": ("a", "b", "f"), "ghost": ("a",), "blk": ("a", "zz")},
        hot={"a": "n", "b": "n", "f": "n"},
        cold={"a": "zero_demand"},
        node_free_mib={}, node_free_slots={}, blocked_nodes={"blk": "over"},
        placed_demand=0.0, total_demand=0.0, ordering="hand-made",
    )
    violations = validate_residency_plan(bad, nodes, adapters)
    text = "\n".join(violations)
    for needle in ("unknown node 'ghost'", "blocked node 'blk'", "has 1 slots", "over its 100 MiB budget",
                   "on node 'n' (base", "placed on both", "unknown adapter 'zz'", "both hot and cold"):
        assert needle in text, needle
    good = plan_adapter_residency(nodes, adapters, {"a": 1.0, "b": 2.0, "f": 1.0})
    assert validate_residency_plan(good, nodes, adapters) == []


node_st = st.builds(
    ServingNode,
    node_id=st.sampled_from([f"n{i}" for i in range(6)]),
    vram_mib=st.integers(0, 4000),
    base_model_id=st.sampled_from([BASE, OTHER_BASE]),
    base_resident_mib=st.integers(0, 4000),
    max_lora_slots=st.integers(0, 6),
)


@settings(max_examples=150, deadline=None)
@given(
    nodes=st.lists(node_st, max_size=6, unique_by=lambda n: n.node_id),
    sizes=st.lists(st.tuples(st.integers(1, 900), st.sampled_from([BASE, OTHER_BASE]), st.floats(0, 100)), max_size=30),
)
def test_property_plans_never_overfill_and_cover_every_adapter(nodes: List[ServingNode], sizes: List[tuple]) -> None:
    adapters = {f"a{i}": AdapterFootprint(s, b) for i, (s, b, _) in enumerate(sizes)}
    demand = {f"a{i}": d for i, (_, _, d) in enumerate(sizes)}
    plan = plan_adapter_residency(nodes, adapters, demand)
    assert validate_residency_plan(plan, nodes, adapters) == []
    assert set(plan.hot) | set(plan.cold) == set(adapters) and not set(plan.hot) & set(plan.cold)
    assert plan.placed_demand <= plan.total_demand + 1e-9
    assert all(v >= 0 for v in plan.node_free_mib.values()) and all(v >= 0 for v in plan.node_free_slots.values())
    assert plan_adapter_residency(nodes, adapters, demand).to_dict() == plan.to_dict()  # deterministic


# ---------------------------------------------------------------- keyword matching


@settings(max_examples=300, deadline=None)
@given(
    text=st.text(alphabet="ab_1 .-:\n", max_size=30),
    keywords=st.lists(st.text(alphabet="ab_1 .-:", min_size=1, max_size=4), max_size=6),
)
def test_property_keyword_hits_matches_btm_moe_semantics(text: str, keywords: List[str]) -> None:
    kws = [k.strip().lower() for k in keywords if k.strip()]
    expected = {k for k in kws if _keyword_hit(k, text.lower())}
    assert keyword_hits(text.lower(), kws) == frozenset(expected)


def test_keyword_hits_whole_words_only() -> None:
    assert keyword_hits("trust the arc of rust", ["rust", "arc", "list comprehension", ".py"]) == frozenset({"rust", "arc"})
    assert keyword_hits("a list comprehension in main.py", ["list comprehension", ".py", "lis", ""]) == frozenset(
        {"list comprehension", ".py"}
    )


# ---------------------------------------------------------------- hierarchical routing


def _routing_registry() -> ExpertRegistry:
    reg = ExpertRegistry()
    reg.register(lora("rust-sec", "rust", keywords=("rust", "borrow", "unsafe")))
    reg.register(lora("rust-async", "rust", keywords=("rust", "tokio", "async")))
    reg.register(lora("rust-old", "rust", keywords=("rust", "tokio", "async", "borrow", "unsafe")))
    reg.register(lora("rust-cand", "rust", keywords=("rust", "tokio", "async", "borrow")))
    reg.register(lora("bio-qa", "bio", keywords=("patient", "clinical")))
    reg.register(lora("bio-other-base", "bio", base=OTHER_BASE, keywords=("patient", "clinical", "trial")))
    reg.register(ExpertRecord("py-srv", SERVING, "python", OTHER_BASE, keywords=("python", "pytest")))
    for eid in ("rust-sec", "rust-async", "bio-qa", "bio-other-base", "py-srv"):
        promote(reg, eid, LIVE)
    promote(reg, "rust-old", RETIRED)
    promote(reg, "rust-cand", SHADOW)
    return reg


def test_router_picks_cluster_then_expert_by_keywords() -> None:
    router = HierarchicalExpertRouter(_routing_registry(), BASE)
    d = router.route("rust tokio async runtime question")
    assert (d.target, d.cluster_id, d.expert_id, d.reason) == ("expert", "rust", "rust-async", "best_resident")
    d = router.route("is this unsafe borrow in rust sound?")
    assert d.expert_id == "rust-sec"
    assert router.live_count == 4  # bio-other-base is on another base and excluded


def test_router_never_returns_retired_shadow_or_foreign_base_experts() -> None:
    router = HierarchicalExpertRouter(_routing_registry(), BASE)
    # rust-old and rust-cand match every keyword here but are retired / shadow
    d = router.route("rust tokio async borrow unsafe")
    assert d.expert_id in {"rust-sec", "rust-async"}
    d = router.route("clinical trial patient")  # bio-other-base has more hits but is bound to another base
    assert d.expert_id == "bio-qa"


def test_router_serving_expert_is_resident_regardless_of_base() -> None:
    router = HierarchicalExpertRouter(_routing_registry(), BASE, resident=[])
    d = router.route("my pytest fails in python")
    assert (d.target, d.expert_id, d.reason) == ("expert", "py-srv", "best_resident")
    d = router.route("clinical patient")  # only adapter in bio is cold and nothing else is resident
    assert (d.target, d.expert_id, d.cluster_id, d.reason) == ("base", None, "bio", "no_resident_expert_in_cluster")


def test_router_recruitment_weights_override_keywords() -> None:
    router = HierarchicalExpertRouter(_routing_registry(), BASE)
    prompt = "rust tokio async"
    assert router.route(prompt).expert_id == "rust-async"
    weights = {"rust": {"rust-sec": 0.4, "rust-async": 0.1, "rust-old": 9.0}}
    d = router.route(prompt, recruitment=weights)
    assert d.expert_id == "rust-sec"  # rust-old has the largest weight but is retired


def test_router_cold_winner_falls_back_to_resident_or_loads_on_demand() -> None:
    reg = _routing_registry()
    router = HierarchicalExpertRouter(reg, BASE, resident=["rust-sec"])
    d = router.route("rust tokio async")
    assert (d.expert_id, d.reason, d.load_on_demand) == ("rust-sec", "fallback_resident_same_cluster", False)
    eager = HierarchicalExpertRouter(reg, BASE, resident=["rust-sec"], allow_cold_load=True)
    d = eager.route("rust tokio async")
    assert (d.expert_id, d.reason, d.load_on_demand) == ("rust-async", "cold_load_on_demand", True)


def test_router_base_fallbacks_and_explicit_cluster() -> None:
    router = HierarchicalExpertRouter(_routing_registry(), BASE)
    d = router.route("what is the weather today")
    assert (d.target, d.expert_id, d.cluster_id, d.reason) == ("base", None, None, "no_cluster_match")
    d = router.route("anything", cluster_id="astro")
    assert (d.target, d.reason) == ("base", "no_live_expert_in_cluster")
    d = router.route("tokio", cluster_id="rust")
    assert d.expert_id == "rust-async"
    with pytest.raises(ValueError, match="base_model_id"):
        HierarchicalExpertRouter(ExpertRegistry(), "")


@settings(max_examples=100, deadline=None)
@given(
    experts=st.lists(
        st.tuples(
            st.sampled_from(["c0", "c1", "c2"]),
            st.sampled_from([CANDIDATE, SHADOW, LIVE, RETIRED]),
            st.sampled_from([LORA_DOMAIN, SERVING]),
            st.lists(st.sampled_from(["alpha", "beta", "gamma", "delta"]), max_size=3),
            st.booleans(),
        ),
        max_size=20,
    ),
    prompt_words=st.lists(st.sampled_from(["alpha", "beta", "gamma", "delta", "noise"]), max_size=5),
    weights=st.dictionaries(st.integers(0, 19), st.floats(0, 1), max_size=5),
    cold_load=st.booleans(),
)
def test_property_every_request_gets_a_live_resident_expert_or_the_base(
    experts: List[tuple], prompt_words: List[str], weights: Dict[int, float], cold_load: bool
) -> None:
    reg = ExpertRegistry()
    resident: List[str] = []
    for i, (cluster, state, kind, kws, is_res) in enumerate(experts):
        eid = f"e{i}"
        if kind == SERVING:
            reg.register(ExpertRecord(eid, SERVING, cluster, BASE, keywords=tuple(kws)))
        else:
            reg.register(lora(eid, cluster, keywords=tuple(kws)))
        promote(reg, eid, state)
        if is_res:
            resident.append(eid)
    recruitment = {c: {f"e{i}": w for i, w in weights.items()} for c in ("c0", "c1", "c2")}
    router = HierarchicalExpertRouter(reg, BASE, resident=resident, allow_cold_load=cold_load)
    d = router.route(" ".join(prompt_words), recruitment=recruitment)
    assert d.target in {"expert", "base"}
    if d.target == "base":
        assert d.expert_id is None
    else:
        rec = reg.get(d.expert_id)  # type: ignore[arg-type]
        assert rec.state == LIVE
        assert d.load_on_demand or rec.kind == SERVING or rec.expert_id in resident
        assert d.load_on_demand == (d.reason == "cold_load_on_demand")


@settings(max_examples=200, deadline=None)
@given(
    text=st.text(alphabet="ab_1 .-:", max_size=30),
    keywords=st.lists(st.text(alphabet="ab_1 .-:", min_size=1, max_size=4), min_size=1, max_size=8),
)
def test_property_router_index_equals_the_plain_keyword_scan(text: str, keywords: List[str]) -> None:
    reg = ExpertRegistry()
    reg.register(ExpertRecord("e", SERVING, "c", BASE, keywords=tuple(keywords)))
    promote(reg, "e", LIVE)
    router = HierarchicalExpertRouter(reg, BASE)
    expected = keyword_hits(text.lower(), reg.get("e").keywords)
    assert router.prompt_hits(text) == expected
    assert router.select_cluster(text)[0] == ("c" if expected else None)
