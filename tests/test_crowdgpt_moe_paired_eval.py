# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for the IP-300 paired base vs MoE held-out evaluation (chunk selection, pairing, bootstrap, decision)."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("crowdgpt_moe_paired_eval", REPO / "scripts" / "crowdgpt_moe_paired_eval.py")
pe = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(pe)


def _rows(n_total: int, short_every: int = 0):
    """A fake source: row i has text 'w<i> ' repeated; every ``short_every``-th row is too short."""
    def fetch(ds, cfg, split, off, length):
        out = []
        for i in range(off, min(off + length, n_total)):
            words = 3 if (short_every and i % short_every == 0) else 100
            out.append({"text": " ".join(f"w{i}" for _ in range(words))})
        return out
    return fetch


def _encode(text: str):
    return [len(w) % 50 for w in text.split()]


def _manifest(sources_e0, sources_e1=None, skipped=False):
    experts = [{"expert_id": "E0", "sources": sources_e0, "train_tokens": 10, "skipped_reason": None}]
    if sources_e1 is not None:
        experts.append({"expert_id": "E1", "sources": sources_e1, "train_tokens": 10, "skipped_reason": None})
    if skipped:
        experts.append({"expert_id": "E3", "sources": [], "train_tokens": 0,
                        "skipped_reason": "every source is on the decontamination benchmark list"})
    return {"experts": experts}


def _src(ds, end, origin="datasets_server"):
    return {"dataset_id": ds, "config": "default", "split": "train", "origin": origin,
            "heldout_rows": [0, 40], "train_rows": [40, end]}


def test_selection_starts_after_training_and_skips_short_rows():
    man = _manifest([_src("a/a", 300)])
    chunks, rep = pe.select_test_chunks(man, {"datasets_server": _rows(1000, short_every=5)}, _encode,
                                        chunks_per_expert=10, chunk_tokens=64)
    rows = [c["row"] for c in chunks]
    assert len(chunks) == 10
    assert min(rows) >= 300  # never a training or ratchet row
    assert all(r % 5 != 0 for r in rows)  # short rows skipped
    assert rows == sorted(rows) and rows[0] == 301
    assert all(len(c["ids"]) == 64 for c in chunks)
    assert rep[0]["sources"][0]["rows_too_short"] == 3  # rows 300, 305, 310


def test_exhausted_source_and_untrained_expert_are_reported_not_substituted():
    man = _manifest([_src("a/a", 370)], skipped=True)
    chunks, rep = pe.select_test_chunks(man, {"datasets_server": _rows(370)}, _encode, chunks_per_expert=5)
    assert chunks == []
    assert rep[0]["skipped_reason"] == "no unseen rows after the training range"
    assert rep[0]["sources"][0]["exhausted"] is True
    assert rep[1]["skipped_reason"].startswith("not trained")


def test_second_source_read_only_when_first_runs_short():
    man = _manifest([_src("a/a", 100), _src("b/b", 40)])
    fetch = {"datasets_server": lambda ds, *a: _rows(103 if ds == "a/a" else 1000)(ds, *a)}
    chunks, rep = pe.select_test_chunks(man, fetch, _encode, chunks_per_expert=5)
    assert [(c["dataset_id"], c["row"]) for c in chunks] == [("a/a", 100), ("a/a", 101), ("a/a", 102), ("b/b", 40), ("b/b", 41)]
    man2 = _manifest([_src("a/a", 100), _src("b/b", 40)])
    chunks2, rep2 = pe.select_test_chunks(man2, {"datasets_server": _rows(10_000)}, _encode, chunks_per_expert=5)
    assert {c["dataset_id"] for c in chunks2} == {"a/a"} and len(rep2[0]["sources"]) == 1


def test_scan_cap_limits_rows():
    man = _manifest([_src("a/a", 100)])
    chunks, rep = pe.select_test_chunks(man, {"datasets_server": _rows(10_000, short_every=1)}, _encode,
                                        chunks_per_expert=5, scan_cap_rows=250)
    assert chunks == [] and rep[0]["sources"][0]["rows_scanned"] == 250


def test_benchmark_source_is_refused():
    man = _manifest([_src("openai/openai_humaneval", 100)])
    with pytest.raises(ValueError, match="benchmark"):
        pe.select_test_chunks(man, {"datasets_server": _rows(1000)}, _encode)


def test_routes_by_manifest_origin():
    seen = []

    def local(ds, *a):
        seen.append("local")
        return _rows(1000)(ds, *a)

    man = _manifest([_src("a/a", 100, origin="local_hf_cache")])
    pe.select_test_chunks(man, {"local_hf_cache": local, "datasets_server": None}, _encode, chunks_per_expert=2)
    assert seen == ["local"]


def test_bootstrap_is_reproducible_and_brackets_the_mean():
    d = [(-0.05 + 0.01 * ((i * 37) % 11 - 5)) for i in range(240)]
    a = pe.paired_bootstrap_ci(d, n_resamples=2000, seed=20261118)
    b = pe.paired_bootstrap_ci(d, n_resamples=2000, seed=20261118)
    assert a == b
    mean, lo, hi = a
    assert lo < mean < hi and hi < 0
    assert pe.paired_bootstrap_ci(d, n_resamples=2000, seed=1) != a


def test_decision_rule():
    assert pe.decide(240, -0.03, -0.01) == "PASS"
    assert pe.decide(240, 0.01, 0.03) == "HARM"
    assert pe.decide(240, -0.01, 0.02) == "NO_GAIN"
    assert pe.decide(240, -0.01, 0.0) == "NO_GAIN"  # upper bound must be strictly below 0
    assert pe.decide(199, -0.03, -0.01) == "INSUFFICIENT_N"


def test_domain_guard_and_b3():
    ok = [{"base_nll": 5.0, "moe_nll": 4.9}, {"base_nll": 4.0, "moe_nll": 4.04}]
    bad = [{"base_nll": 5.0, "moe_nll": 4.9}, {"base_nll": 4.0, "moe_nll": 4.0401}]
    assert pe.domain_guard(ok) is True
    assert pe.domain_guard(bad) is False
    assert pe.b3_supported("PASS", True) is True
    assert pe.b3_supported("PASS", False) is False
    assert pe.b3_supported("NO_GAIN", True) is False


def test_sign_test():
    r = pe.sign_test([-1, -1, -1, 1, 0])
    assert (r["moe_lower"], r["moe_higher"], r["ties"]) == (3, 1, 1)
    assert r["p_two_sided"] == pytest.approx(2 * (1 + 4) / 16)


def test_per_expert_pairing():
    chunks = [{"expert_id": "E0"}, {"expert_id": "E1"}, {"expert_id": "E0"}]
    out = pe.per_expert_summary(chunks, [5.0, 4.0, 3.0], [4.0, 4.5, 3.0])
    assert [e["expert_id"] for e in out] == ["E0", "E1"]
    assert out[0]["base_nll"] == 4.0 and out[0]["moe_nll"] == 3.5 and out[0]["mean_diff"] == -0.5
    assert out[1]["mean_diff"] == 0.5 and out[1]["n"] == 1


def _small_model():
    worker = pe._load_worker()
    model, _ = worker._load_or_build_model(cache_dir=None, use_small_arch=True)
    return worker, model


def test_chunk_nll_matches_manual_cross_entropy():
    _, model = _small_model()
    model.eval()
    ids = [(i * 13 + 7) % 256 for i in range(64)]
    got = pe.chunk_nll(model, ids)
    x = torch.tensor([ids])
    with torch.no_grad():
        logits = model(x)[0]  # the model's own forward: (logits, loss, kv)
    want = F.cross_entropy(logits[0, :-1].float(), x[0, 1:]).item()
    assert got == pytest.approx(want, abs=1e-4)


SPLIT = {"split_mode": "lora_domain_experts", "num_experts": 8, "top_k": 2, "routing_mode": "sigma_topk",
         "lora_rank": 8, "resident_expert_indices": [0, 1, 2, 3, 4, 5, 6, 7], "num_router_fragments": 4}


def test_zero_delta_upcycle_equals_dense_and_loaded_buckets_change_loss(tmp_path):
    from safetensors.torch import save_file

    from hivemind.swarm.crowdgpt_moe_opendiloco import extract_moe_trainable_buckets

    worker, model = _small_model()
    ids = [(i * 13 + 7) % 256 for i in range(64)]
    dense = pe.chunk_nll(model, ids)
    pe.upcycle_and_load(model, SPLIT, None, worker.load_init_buckets)
    assert pe.chunk_nll(model, ids) == pytest.approx(dense, abs=1e-5)

    # a checkpoint with non-zero LoRA B changes the loss; a wrong-shaped one is refused by the worker's loader
    buckets = extract_moe_trainable_buckets(model, num_router_fragments=4)
    g = torch.Generator().manual_seed(0)
    moved = {k: v + 0.05 * torch.randn(v.shape, generator=g) for k, v in buckets.items()}
    ck = tmp_path / "ck.safetensors"
    save_file({k: v.contiguous() for k, v in moved.items()}, str(ck))
    _, model2 = _small_model()
    pe.upcycle_and_load(model2, SPLIT, str(ck), worker.load_init_buckets)
    assert pe.chunk_nll(model2, ids) != pytest.approx(dense, abs=1e-5)
    bad = tmp_path / "bad.safetensors"
    k0 = next(iter(moved))
    save_file({**{k: v.contiguous() for k, v in moved.items()}, k0: torch.zeros(3)}, str(bad))
    _, model3 = _small_model()
    with pytest.raises(ValueError):
        pe.upcycle_and_load(model3, SPLIT, str(bad), worker.load_init_buckets)


def test_main_refuses_when_cuda_visible(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    rc = pe.main(["--rows-cache-dir", str(tmp_path), "--tokenizer-json", "x", "--out", str(tmp_path / "o.json")])
    assert rc == 2 and not (tmp_path / "o.json").exists()


def test_preregistration_spec_is_frozen_and_consistent():
    spec = json.loads((REPO / "docs" / "E_B3_MOE_PAIRED_PREREGISTRATION.json").read_text(encoding="utf-8"))
    assert spec["frozen"] is True
    assert spec["arms"]["moe"]["checkpoint_sha256"].startswith("0778b9ee4279ee6a")
    assert spec["chunking"]["chunks_per_expert"] * spec["chunking"]["expected_experts"] >= spec["chunking"]["min_total_chunks"] >= 200
    assert set(spec["data"]["excluded_benchmarks"]) == set(pe.BANNED_BENCHMARKS)
    assert os.path.basename(spec["data"]["manifest"]) == "real_data_manifest.json"


# ---------------------------------------------------------------------------
# E-B3-MOE-2 mid-row follow-up (chunk_offset)
# ---------------------------------------------------------------------------

def _lens(lengths):
    """A fake source whose row i has ``lengths[i]`` words (each encodes to one id)."""
    def fetch(ds, cfg, split, off, length):
        return [{"text": " ".join(f"w{i}x{j}" for j in range(lengths[i]))} for i in range(off, min(off + length, len(lengths)))]
    return fetch


def test_offset_zero_selects_exactly_the_944_chunks():
    man = _manifest([_src("a/a", 300)])
    fetch = {"datasets_server": _rows(1000, short_every=5)}
    old, rep_old = pe.select_test_chunks(man, fetch, _encode, chunks_per_expert=10, chunk_tokens=64)
    new, rep_new = pe.select_test_chunks(man, fetch, _encode, chunks_per_expert=10, chunk_tokens=64, chunk_offset=0)
    assert [(c["row"], c["ids"]) for c in old] == [(c["row"], c["ids"]) for c in new]
    assert pe.chunk_ids_sha256(old) == pe.chunk_ids_sha256(new) and rep_old == rep_new
    assert all(c["score_from"] == 0 and len(c["ids"]) == 64 for c in new)


def test_midrow_needs_offset_plus_tokens_and_keeps_the_context():
    lengths = [200] * 300 + [100, 127, 128, 300, 64, 129] + [50] * 100
    man = _manifest([_src("a/a", 300)])
    chunks, rep = pe.select_test_chunks(man, {"datasets_server": _lens(lengths)}, lambda t: list(range(len(t.split()))),
                                        chunks_per_expert=3, chunk_tokens=64, chunk_offset=64)
    assert [c["row"] for c in chunks] == [302, 303, 305]  # rows with < 128 ids are skipped, next eligible taken
    assert all(len(c["ids"]) == 128 and c["score_from"] == 64 for c in chunks)
    assert chunks[0]["ids"] == list(range(128))  # ids[0:128]: the row's opening stays as context
    assert rep[0]["sources"][0]["rows_too_short"] == 3


def test_midrow_hash_covers_all_128_ids():
    a = [{"expert_id": "E0", "dataset_id": "d", "row": 1, "ids": list(range(128)), "score_from": 64}]
    b = [{**a[0], "ids": [*range(10), 999, *range(11, 128)]}]
    assert pe.chunk_ids_sha256(a) != pe.chunk_ids_sha256(b)


def test_masked_score_matches_manual_cross_entropy_and_is_causal():
    _, model = _small_model()
    model.eval()
    ids = [(i * 13 + 7) % 256 for i in range(128)]
    x = torch.tensor([ids])
    with torch.no_grad():
        logits = model(x)[0][0].float()
    mid = pe.chunk_nll(model, ids, score_from=64)
    assert mid == pytest.approx(F.cross_entropy(logits[63:127], x[0, 64:]).item(), abs=1e-4)  # 64 targets ids[64..127]
    start_masked = pe.chunk_nll(model, ids, score_to=64)
    assert start_masked == pytest.approx(F.cross_entropy(logits[0:63], x[0, 1:64]).item(), abs=1e-4)
    # causality: later tokens do not change earlier predictions (the run's causality gate)
    assert start_masked == pytest.approx(pe.chunk_nll(model, ids[:64]), abs=1e-4)
    assert pe.chunk_nll(model, ids) == pe.chunk_nll(model, ids, score_from=0)
    with pytest.raises(ValueError):
        pe.chunk_nll(model, ids, score_from=128)


def test_eval_arm_uses_score_from_and_rowstart_control():
    _, model = _small_model()
    ids = [(i * 7 + 3) % 256 for i in range(128)]
    chunks = [{"expert_id": "E0", "ids": ids, "score_from": 64}]
    assert pe.eval_arm(model, chunks)[0] == pytest.approx(pe.chunk_nll(model, ids, score_from=64), abs=1e-6)
    assert pe.eval_arm(model, chunks, rowstart_tokens=64)[0] == pytest.approx(pe.chunk_nll(model, ids[:64]), abs=1e-6)


def test_zero_delta_upcycle_equals_dense_on_midrow_score():
    worker, model = _small_model()
    ids = [(i * 13 + 7) % 256 for i in range(128)]
    dense = pe.chunk_nll(model, ids, score_from=64)
    pe.upcycle_and_load(model, SPLIT, None, worker.load_init_buckets)
    assert pe.chunk_nll(model, ids, score_from=64) == pytest.approx(dense, abs=1e-5)


def test_per_expert_minimum_excludes_small_experts_from_the_verdict_and_guard():
    chunks = [{"expert_id": "E0"}] * 200 + [{"expert_id": "E1"}] * 5
    keep = pe.qualifying_experts(chunks, 20)
    assert keep == {"E0"}
    ix = [i for i, c in enumerate(chunks) if c["expert_id"] in keep]
    base = [5.0] * 205
    moe = [4.9 + 0.001 * (i % 7) for i in range(200)] + [6.0] * 5  # E1 much worse, but below the minimum
    r = pe.make_analyser(chunks, ix, keep, min_total=200, n_resamples=500)(moe, base)
    assert r["n"] == 200 and r["verdict"] == "PASS" and r["domain_guard_ok"] is True
    assert [e["qualifies"] for e in r["per_expert"]] == [True, False]
    r2 = pe.make_analyser(chunks, ix[:150], keep, min_total=200, n_resamples=500)(moe, base)
    assert r2["verdict"] == "INSUFFICIENT_N"


def test_format_content_label():
    assert pe.format_content_label("PASS", -0.20) == "MOSTLY_CONTENT"
    assert pe.format_content_label("PASS", -0.1653) == "MOSTLY_CONTENT"
    assert pe.format_content_label("PASS", -0.10) == "MOSTLY_FORMAT_SOME_CONTENT"
    assert pe.format_content_label("NO_GAIN", -0.10) == "FORMAT"
    assert pe.format_content_label("HARM", 0.05) == "FORMAT"
    assert pe.format_content_label("INSUFFICIENT_N", -0.5) == "NONE"


def test_mid_over_start_ratio():
    mid = {"mean_paired_diff": -0.1, "per_expert": [{"expert_id": "E0", "mean_diff": -0.05}, {"expert_id": "E1", "mean_diff": -0.2}]}
    start = {"mean_paired_diff": -0.4, "per_expert": [{"expert_id": "E0", "mean_diff": -0.1}, {"expert_id": "E1", "mean_diff": 0.0}]}
    r = pe.mid_over_start_ratio(mid, start)
    assert r["overall"] == pytest.approx(0.25) and r["E0"] == pytest.approx(0.5) and r["E1"] is None


def test_midrow_preregistration_spec_is_frozen_and_consistent():
    spec = json.loads((REPO / "docs" / "E_B3_MOE_MIDROW_PREREGISTRATION.json").read_text(encoding="utf-8"))
    first = json.loads((REPO / "docs" / "E_B3_MOE_PAIRED_PREREGISTRATION.json").read_text(encoding="utf-8"))
    assert spec["frozen"] is True
    ch = spec["chunking"]
    assert ch["chunk_offset"] == 64 and ch["chunk_tokens"] == 64 and ch["min_row_tokens"] == 128
    assert ch["min_total_chunks"] == 200 and ch["min_chunks_per_expert"] == 20 and ch["scan_cap_rows_per_source"] == 2000
    # only the chunk position changes: same base, primary checkpoint, split, manifest and benchmark exclusions as #944
    assert spec["arms"]["base"] == first["arms"]["base"]
    assert spec["arms"]["moe"]["checkpoint_sha256"] == first["arms"]["moe"]["checkpoint_sha256"]
    assert spec["arms"]["moe"]["split"] == first["arms"]["moe"]["split"]
    assert spec["data"]["manifest_sha256"] == first["data"]["manifest_sha256"]
    assert set(spec["data"]["excluded_benchmarks"]) == set(pe.BANNED_BENCHMARKS)
    assert spec["arms"]["moe_secondary"]["checkpoint_sha256"].startswith("58696cfe5660bd91")
    assert "-0.1653" in spec["interpretation_rule"]["reference"]
