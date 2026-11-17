# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Real token shards for the CrowdGPT MoE round: offline tests with a fake datasets-server and a toy encoder."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hivemind.swarm.crowdgpt_real_shards import build_real_expert_shards, row_to_text

PLAN = {
    "decontamination_benchmarks": ["allenai/ai2_arc"],
    "experts": [
        {"expert_id": "E0", "assigned_cohort": "GCP Spot Tesla T4", "accepted_sources": [{"dataset_id": "web/a", "config": "default"}]},
        {"expert_id": "E1", "assigned_cohort": "RTX 2070 coding", "accepted_sources": [{"dataset_id": "code/b", "config": "default"}]},
        {"expert_id": "E3", "assigned_cohort": "Formal", "accepted_sources": [{"dataset_id": "allenai/ai2_arc", "config": "ARC-Easy"}]},
    ],
}


def _encode(text: str) -> list[int]:
    return [ord(c) % 97 for c in text]


class FakeServer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def __call__(self, ds: str, cfg: str, split: str, offset: int, length: int):
        self.calls.append((ds, offset, length))
        return [{"text": f"{ds} row {i} " * 3, "id": i} for i in range(offset, min(offset + length, 400))]


def _build(tmp_path: Path, fake: FakeServer, **kw):
    return build_real_expert_shards(PLAN, _encode, 97, cache_dir=tmp_path / "rows", fetch=fake,
                                    train_tokens_per_expert=kw.get("train", 500), heldout_tokens_per_expert=kw.get("held", 100),
                                    heldout_rows=kw.get("held_rows", 5))


def test_benchmark_sources_are_never_fetched_and_their_expert_is_reported(tmp_path: Path) -> None:
    fake = FakeServer()
    _s0, _s1, _h, manifest = _build(tmp_path, fake)
    assert all(ds != "allenai/ai2_arc" for ds, _o, _l in fake.calls)
    e3 = next(e for e in manifest["experts"] if e["expert_id"] == "E3")
    assert e3["skipped_reason"] and e3["train_tokens"] == 0


def test_heldout_rows_precede_training_rows_and_never_overlap(tmp_path: Path) -> None:
    fake = FakeServer()
    _s0, _s1, _h, manifest = _build(tmp_path, fake)
    for e in manifest["experts"]:
        for src in e["sources"]:
            assert src["heldout_rows"][1] <= src["train_rows"][0]
    held_offsets = {(ds, o) for ds, o, _l in fake.calls if o < 5}
    train_offsets = {(ds, o) for ds, o, _l in fake.calls if o >= 5}
    assert held_offsets and train_offsets and not (held_offsets & train_offsets)


def test_learners_follow_the_cohorts_and_budgets(tmp_path: Path) -> None:
    s0, s1, held, manifest = _build(tmp_path, FakeServer())
    assert s0.dtype == torch.long and s0.numel() == 500 and s1.numel() == 500 and held.numel() == 200
    assert {e["expert_id"]: e["learner"] for e in manifest["experts"]}["E0"] == 0
    assert {e["expert_id"]: e["learner"] for e in manifest["experts"]}["E1"] == 1
    assert manifest["evidence"] == "measured" and manifest["learner_tokens"] == [500, 500]


def test_rows_are_cached_so_a_rerun_makes_no_request(tmp_path: Path) -> None:
    first = FakeServer()
    a = _build(tmp_path, first)
    second = FakeServer()
    b = _build(tmp_path, second)
    assert second.calls == [] and torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])


def test_tokens_outside_the_vocabulary_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        build_real_expert_shards(PLAN, lambda s: [1000], 97, cache_dir=tmp_path / "r", fetch=FakeServer(),
                                 train_tokens_per_expert=10, heldout_tokens_per_expert=5, heldout_rows=2)


def test_an_unreachable_source_is_recorded_not_fatal_unless_a_learner_ends_up_empty(tmp_path: Path) -> None:
    def down(ds, cfg, split, offset, length):
        if ds == "code/b":
            raise OSError("unreachable")
        return FakeServer()(ds, cfg, split, offset, length)

    with pytest.raises(ValueError, match="empty learner shard"):
        build_real_expert_shards(PLAN, _encode, 97, cache_dir=tmp_path / "r2", fetch=down,
                                 train_tokens_per_expert=50, heldout_tokens_per_expert=10, heldout_rows=2)


def test_row_to_text_keeps_content_and_drops_metadata_and_long_arrays() -> None:
    assert row_to_text({"text": " hello ", "url": "x"}) == "hello"
    t = row_to_text({"problem": "2+2?", "generated_solution": "4", "id": 7, "flux": list(range(500)), "tags": ["a", "b"]})
    assert "problem: 2+2?" in t and "generated_solution: 4" in t and "tags: a, b" in t
    assert "id:" not in t and "flux" not in t



def test_local_cache_fetcher_reads_cached_files_in_order_and_falls_back(tmp_path) -> None:
    """--local-data: cached files are read like the datasets-server (same offsets), other datasets fall back."""
    import json as _json

    from hivemind.swarm import crowdgpt_real_shards as rs

    snap = tmp_path / "datasets--sahil2801--CodeAlpaca-20k" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "code_alpaca_20k.json").write_text(_json.dumps([{"instruction": f"task {i}", "output": f"code {i}"} for i in range(50)]))
    calls = []
    fetch = rs.LocalHFCacheFetcher(roots=[tmp_path], fallback=lambda *a: calls.append(a) or [{"text": "remote"}])
    rows = fetch("sahil2801/CodeAlpaca-20k", "default", "train", 40, 5)
    assert [r["instruction"] for r in rows] == [f"task {i}" for i in range(40, 45)]
    assert fetch("sahil2801/CodeAlpaca-20k", "default", "train", 48, 10)[-1]["instruction"] == "task 49"
    assert fetch.origin_of("sahil2801/CodeAlpaca-20k") == "local_hf_cache"
    assert fetch("bigcode/self-oss-instruct-sc2-exec-filter-50k", "default", "train", 0, 3) == [{"text": "remote"}]
    assert calls and fetch.origin_of("bigcode/self-oss-instruct-sc2-exec-filter-50k") == "datasets_server"


def test_local_extra_sources_come_first_and_carry_a_licence() -> None:
    from hivemind.swarm import crowdgpt_real_shards as rs

    plan = {"experts": [{"expert_id": "E2_math_reasoning", "accepted_sources": [{"dataset_id": "nvidia/OpenMathInstruct-2"}]},
                        {"expert_id": "E0_general_fineweb", "accepted_sources": [{"dataset_id": "HuggingFaceFW/fineweb-edu"}]}]}
    out = rs.with_local_extra_sources(plan)
    assert [s["dataset_id"] for s in out["experts"][0]["accepted_sources"]] == ["AI-MO/NuminaMath-CoT", "nvidia/OpenMathInstruct-2"]
    assert out["experts"][0]["accepted_sources"][0]["license"] == "apache-2.0"
    assert out["experts"][1]["accepted_sources"] == plan["experts"][1]["accepted_sources"]
    assert plan["experts"][0]["accepted_sources"] == [{"dataset_id": "nvidia/OpenMathInstruct-2"}]  # input not mutated
    assert not any("gsm8k" in d or "sciq" in d or "proofnet" in d for d in rs.LOCAL_DATASET_FILES)
