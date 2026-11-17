# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for the S21 base-model catalog, VRAM estimator, promotion gate and upgrade plan.

Only the network is mocked (an injected HTTP getter serving synthetic Hub responses). Every check has a negative control.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import math
import urllib.error
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hivemind.core import base_model_catalog as bmc

REPO = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO / "configs" / "base_models_s21.json"
SHA = "a" * 40


# --- a fake Hub (network is the only thing mocked) ---------------------------------------------------------------------------

def _safetensors_blob(tensors: Mapping[str, tuple[list[int], str]]) -> bytes:
    header = {k: {"shape": s, "dtype": d, "data_offsets": [0, 0]} for k, (s, d) in tensors.items()}
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header).encode()
    return len(raw).to_bytes(8, "little") + raw


TOY_TENSORS: dict[str, tuple[list[int], str]] = {
    "model.embed_tokens.weight": [[100, 8], "BF16"],
    "model.layers.0.self_attn.q_proj.weight": [[8, 8], "BF16"],
    "model.layers.0.mlp.up_proj.weight": [[16, 8], "BF16"],
    "model.layers.0.mlp.down_proj.weight": [[8, 16], "BF16"],
    "model.layers.0.input_layernorm.weight": [[8], "BF16"],
    "model.layers.0.linear_attn.conv1d.weight": [[24, 1, 4], "BF16"],
    "model.visual.blocks.0.attn.qkv.weight": [[24, 8], "BF16"],
    "mtp.fc.weight": [[8, 16], "BF16"],
    "lm_head.weight": [[100, 8], "BF16"],
}
TOY_TOTAL = 800 + 64 + 128 + 128 + 8 + 96 + 192 + 128 + 800
TOY_CONFIG = {"architectures": ["ToyForCausalLM"], "model_type": "toy", "hidden_size": 8, "num_hidden_layers": 1,
              "intermediate_size": 16, "vocab_size": 100, "num_attention_heads": 2, "num_key_value_heads": 1,
              "tie_word_embeddings": False}


class FakeHub:
    def __init__(self, models: Mapping[str, dict[str, Any]], config_status: int = 200) -> None:
        self.models = models
        self.config_status = config_status
        self.calls: list[str] = []

    def __call__(self, url: str, headers: Mapping[str, str]) -> tuple[int, bytes]:
        self.calls.append(url)
        for mid, spec in self.models.items():
            if url == f"{bmc.HF_BASE}/api/models/{mid}?blobs=true":
                return 200, json.dumps(spec["info"]).encode()
            if url == f"{bmc.HF_BASE}/{mid}/resolve/{SHA}/config.json":
                return self.config_status, json.dumps(spec["config"]).encode()
            if url.startswith(f"{bmc.HF_BASE}/{mid}/resolve/{SHA}/") and url.endswith(".safetensors"):
                blob = spec["blob"]
                rng = headers["Range"].split("=")[1].split("-")
                return 206, blob[int(rng[0]): int(rng[1]) + 1]
        return 401, b'{"error":"Invalid username or password."}'


def _toy_info(total: int = TOY_TOTAL) -> dict[str, Any]:
    return {"id": "toy/Toy-1", "sha": SHA, "lastModified": "2026-01-01T00:00:00.000Z", "gated": False, "pipeline_tag": "text-generation",
            "library_name": "transformers", "cardData": {"license": "apache-2.0"}, "tags": ["license:apache-2.0"],
            "safetensors": {"total": total, "parameters": {"BF16": total}},
            "siblings": [{"rfilename": "config.json", "size": 10}, {"rfilename": "model.safetensors", "size": 4242, "lfs": {"sha256": "b" * 64}}]}


def _toy_hub(**kw: Any) -> FakeHub:
    return FakeHub({"toy/Toy-1": {"info": _toy_info(**kw), "config": TOY_CONFIG, "blob": _safetensors_blob(TOY_TENSORS)}})


NOW = datetime(2026, 10, 28, 9, 0, tzinfo=UTC)


# --- 1. catalog ----------------------------------------------------------------------------------------------------------------

def test_classify_tensor_components_and_negative_controls() -> None:
    assert bmc.classify_tensor("model.layers.3.mlp.gate_proj.weight", [10, 4]) == "linear"
    assert bmc.classify_tensor("model.visual.blocks.0.attn.qkv.weight", [10, 4]) == "vision"
    assert bmc.classify_tensor("mtp.layers.0.mlp.up_proj.weight", [10, 4]) == "mtp"
    assert bmc.classify_tensor("lm_head.weight", [10, 4]) == "embedding"
    # negatives: a 1-D norm, a 3-D conv and a bias are never treated as quantisable linear weights
    assert bmc.classify_tensor("model.norm.weight", [10]) == "other"
    assert bmc.classify_tensor("model.layers.0.linear_attn.conv1d.weight", [8, 1, 4]) == "other"
    assert bmc.classify_tensor("model.layers.0.self_attn.q_proj.bias", [10, 4]) == "other"


def test_summarize_header_exact_split() -> None:
    s = bmc.summarize_header(TOY_TENSORS)
    assert s["params_total"] == TOY_TOTAL
    assert s["params_text_linear"] == 64 + 128 + 128
    assert s["params_embedding"] == 1600
    assert s["params_vision"] == 192 and s["params_mtp"] == 128
    assert s["params_text_other"] == 8 + 96
    assert s["lora_in_plus_out_all_linear"] == 16 + 24 + 24
    assert s["lora_in_plus_out_mlp"] == 48
    assert s["max_linear_elements"] == 128
    assert sum(s[k] for k in ("params_text_linear", "params_embedding", "params_text_other", "params_vision", "params_mtp")) == s["params_total"]


def test_read_safetensors_header_and_failure() -> None:
    hub = _toy_hub()
    out = bmc.read_safetensors_header(hub, f"{bmc.HF_BASE}/toy/Toy-1/resolve/{SHA}/model.safetensors")
    assert set(out) == set(TOY_TENSORS)
    assert out["mtp.fc.weight"] == ([8, 16], "BF16")
    with pytest.raises(RuntimeError, match="length prefix"):
        bmc.read_safetensors_header(hub, f"{bmc.HF_BASE}/nope/x.safetensors")

    def short_second(url: str, headers: Mapping[str, str]) -> tuple[int, bytes]:
        return (206, (5).to_bytes(8, "little")) if headers["Range"] == "bytes=0-7" else (404, b"")

    with pytest.raises(RuntimeError, match="header unavailable"):
        bmc.read_safetensors_header(short_second, "u")


def test_fetch_entry_exists_and_not_found() -> None:
    hub = _toy_hub()
    entry = bmc.fetch_catalog_entry("toy/Toy-1", bmc.INCUMBENT, hub, "2026-10-28T09:00:00Z")
    assert entry["exists"] is True and entry["revision_sha"] == SHA
    assert entry["license"] == "apache-2.0" and entry["safetensors_total_bytes"] == 4242
    assert entry["header"]["params_total"] == entry["api_parameters_total"] == TOY_TOTAL
    assert entry["architecture"]["head_dim"] == 4 and entry["architecture"]["n_full_attention_layers"] == 1
    missing = bmc.fetch_catalog_entry("Qwen/Does-Not-Exist", bmc.CANDIDATE, hub, "t")
    assert missing["exists"] is False and missing["http_status"] == 401
    assert "revision_sha" not in missing


def test_fetch_entry_license_from_tags_and_config_failure() -> None:
    info = _toy_info()
    info["cardData"] = {}
    hub = FakeHub({"toy/Toy-1": {"info": info, "config": TOY_CONFIG, "blob": _safetensors_blob(TOY_TENSORS)}})
    assert bmc.fetch_catalog_entry("toy/Toy-1", bmc.CANDIDATE, hub, "t")["license"] == "apache-2.0"
    broken = FakeHub({"toy/Toy-1": {"info": _toy_info(), "config": TOY_CONFIG, "blob": b""}}, config_status=404)
    with pytest.raises(RuntimeError, match="config.json unavailable"):
        bmc.fetch_catalog_entry("toy/Toy-1", bmc.CANDIDATE, broken, "t")


def test_build_catalog_splits_missing_and_validates() -> None:
    cat = bmc.build_catalog(_toy_hub(), [("toy/Toy-1", bmc.INCUMBENT), ("Qwen/Ghost", bmc.CANDIDATE)], now=NOW)
    assert [m["model_id"] for m in cat["models"]] == ["toy/Toy-1"]
    assert [m["model_id"] for m in cat["not_found"]] == ["Qwen/Ghost"]
    assert cat["retrieved_at"] == "2026-10-28T09:00:00Z"
    assert bmc.validate_catalog(cat) == []


def test_validate_catalog_rejects_each_defect() -> None:
    good = bmc.build_catalog(_toy_hub(), [("toy/Toy-1", bmc.INCUMBENT)], now=NOW)
    assert bmc.validate_catalog(good) == []
    cases: list[tuple[str, Any]] = []
    c = copy.deepcopy(good); c["models"][0]["revision_sha"] = "main"; cases.append(("40-hex", c))
    c = copy.deepcopy(good); c["models"][0]["api_parameters_total"] = TOY_TOTAL + 1; cases.append(("!= Hub", c))
    c = copy.deepcopy(good); c["models"][0]["role"] = bmc.CANDIDATE; cases.append(("no incumbent", c))
    c = copy.deepcopy(good); c["models"].append(copy.deepcopy(c["models"][0])); cases.append(("duplicate", c))
    c = copy.deepcopy(good); c["schema"] = "v0"; cases.append(("schema", c))
    c = copy.deepcopy(good); c["models"][0]["license"] = None; cases.append(("license", c))
    c = copy.deepcopy(good); c["models"][0]["header"] = {}; cases.append(("header summary", c))
    c = copy.deepcopy(good); c["models"][0]["retrieved_at"] = ""; cases.append(("retrieval date", c))
    c = copy.deepcopy(good); c["models"][0]["api_parameters_total"] = None; cases.append(("no Hub safetensors total", c))
    for needle, bad in cases:
        problems = bmc.validate_catalog(bad)
        assert any(needle in p for p in problems), (needle, problems)


def test_committed_catalog_is_valid_and_fetched(tmp_path: Path) -> None:
    cat = bmc.load_catalog(CATALOG_PATH)
    ids = {m["model_id"] for m in cat["models"]}
    assert {bmc.INCUMBENT_ID, "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"} <= ids
    assert all(bmc._SHA_RE.match(m["revision_sha"]) and m["retrieved_at"] for m in cat["models"])
    assert not ids & {m["model_id"] for m in cat["not_found"]}
    bad = copy.deepcopy(cat)
    bad["models"][0]["header"]["params_total"] += 1
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="Hub safetensors total"):
        bmc.load_catalog(p)


class _Resp:
    status = 200

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *a: object) -> None:
        return None

    def read(self) -> bytes:
        return b"ok"


def test_urllib_get_returns_status_and_body(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    def ok(req: Any, timeout: int) -> _Resp:
        seen.append(req)
        return _Resp()

    monkeypatch.setattr(bmc.urllib.request, "urlopen", ok)
    assert bmc.urllib_get("https://huggingface.co/x", {"Range": "bytes=0-7"}) == (200, b"ok")
    assert seen[0].get_header("Range") == "bytes=0-7"

    def fail(req: Any, timeout: int) -> _Resp:
        raise urllib.error.HTTPError("u", 401, "no", None, io.BytesIO(b"denied"))  # type: ignore[arg-type]

    monkeypatch.setattr(bmc.urllib.request, "urlopen", fail)
    assert bmc.urllib_get("https://huggingface.co/x", {}) == (401, b"denied")

    def fail_nobody(req: Any, timeout: int) -> _Resp:
        raise urllib.error.HTTPError("u", 404, "no", None, None)  # type: ignore[arg-type]

    monkeypatch.setattr(bmc.urllib.request, "urlopen", fail_nobody)
    assert bmc.urllib_get("https://huggingface.co/x", {}) == (404, b"")


# --- 2. VRAM estimator ---------------------------------------------------------------------------------------------------------

def _geom(model_id: str) -> bmc.ModelGeometry:
    cat = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return bmc.geometry_from_entry(next(m for m in cat["models"] if m["model_id"] == model_id))


def test_gguf_bytes_per_weight_from_block_layout() -> None:
    assert bmc.gguf_bytes_per_weight("Q4_0") == pytest.approx(18 / 32)
    assert bmc.gguf_bytes_per_weight("Q8_0") == pytest.approx(34 / 32)
    with pytest.raises(ValueError):
        bmc.gguf_bytes_per_weight("Q3_K")


def test_training_estimate_components() -> None:
    g = _geom(bmc.INCUMBENT_ID)
    est = bmc.estimate_qlora_training(g, bmc.TrainRecipe(seq_len=512, micro_batch=1))
    assert est.kind == "ESTIMATE" and est.mode == "qlora_nf4_training"
    assert est.components_mib["nf4_weights"] == pytest.approx(g.params_text_linear * 0.53 / 2**20, abs=0.1)
    assert est.components_mib["fp16_aux_towers"] == 0.0  # the coder has no vision/MTP tower
    assert est.total_mib == pytest.approx(sum(est.components_mib.values()), abs=0.5)
    q = _geom("Qwen/Qwen3.5-4B")
    with_aux = bmc.estimate_qlora_training(q, bmc.TrainRecipe())
    without = bmc.estimate_qlora_training(q, bmc.TrainRecipe(load_aux_towers=False))
    assert with_aux.total_mib - without.total_mib == pytest.approx((q.params_vision + q.params_mtp) * 2 / 2**20, abs=0.2)
    mlp_only = bmc.estimate_qlora_training(q, bmc.TrainRecipe(targets="mlp"))
    assert mlp_only.components_mib["lora_weights_grads_optimizer"] < without.components_mib["lora_weights_grads_optimizer"]


@settings(max_examples=40, deadline=None)
@given(seq=st.integers(16, 4096), mb=st.integers(1, 8), rank=st.integers(1, 64))
def test_training_estimate_monotone_in_tokens_and_rank(seq: int, mb: int, rank: int) -> None:
    g = _geom("Qwen/Qwen3.5-0.8B")
    base = bmc.estimate_qlora_training(g, bmc.TrainRecipe(seq_len=seq, micro_batch=mb, lora_rank=rank)).total_mib
    assert bmc.estimate_qlora_training(g, bmc.TrainRecipe(seq_len=seq * 2, micro_batch=mb, lora_rank=rank)).total_mib > base
    assert bmc.estimate_qlora_training(g, bmc.TrainRecipe(seq_len=seq, micro_batch=mb + 1, lora_rank=rank)).total_mib > base
    assert bmc.estimate_qlora_training(g, bmc.TrainRecipe(seq_len=seq, micro_batch=mb, lora_rank=rank + 8)).total_mib > base
    ckpt = bmc.estimate_qlora_training(g, bmc.TrainRecipe(seq_len=seq, micro_batch=mb, lora_rank=rank, gradient_checkpointing=True))
    assert ckpt.total_mib < base


def test_serving_estimate_orders_quants_and_context() -> None:
    g = _geom("Qwen/Qwen3.5-9B")
    q4 = bmc.estimate_gguf_serving(g, bmc.ServeConfig("Q4_0", 4096))
    q8 = bmc.estimate_gguf_serving(g, bmc.ServeConfig("Q8_0", 4096))
    long = bmc.estimate_gguf_serving(g, bmc.ServeConfig("Q4_0", 32768))
    assert q8.total_mib > q4.total_mib and long.total_mib > q4.total_mib
    assert q4.components_mib["linear_attention_state"] > 0  # Qwen3.5 is hybrid: recurrent state is counted
    inc = bmc.estimate_gguf_serving(_geom(bmc.INCUMBENT_ID), bmc.ServeConfig("Q4_0", 4096))
    assert inc.components_mib["linear_attention_state"] == 0.0  # a pure full-attention model has none
    assert q4.to_dict()["kind"] == "ESTIMATE"


def test_recipe_and_serve_config_reject_bad_input() -> None:
    for kw in ({"seq_len": 0}, {"targets": "everything"}, {"optimizer": "lion"}, {"act_scale": 0.0}):
        with pytest.raises(ValueError):
            bmc.TrainRecipe(**kw)
    with pytest.raises(ValueError):
        bmc.ServeConfig("Q4_0", 0)
    with pytest.raises(ValueError):
        bmc.ServeConfig("FP4", 4096)
    assert bmc.TrainRecipe(gradient_checkpointing=True).label.endswith("ckpt")


def test_fit_verdict_boundaries() -> None:
    assert bmc.fit_verdict(1000.0, 1200.0).verdict == bmc.FITS
    assert bmc.fit_verdict(1000.0, 1199.0).verdict == bmc.MARGINAL
    assert bmc.fit_verdict(1000.0, 999.0).verdict == bmc.DOES_NOT_FIT
    assert not bmc.fit_verdict(1000.0, 1100.0).fits
    assert bmc.fit_verdict(1000.0, 1000.0, margin=0.0).fits
    with pytest.raises(ValueError):
        bmc.fit_verdict(1000.0, 0.0)


def test_calibrate_from_measurement_records_ratio(tmp_path: Path) -> None:
    est = bmc.estimate_qlora_training(_geom("Qwen/Qwen3.5-0.8B"), bmc.TrainRecipe())
    rec = bmc.calibrate_from_measurement(est, est.total_mib * 1.1, measured_kind="process_peak", gpu="Tesla T4",
                                         source="nvidia-smi log /x", now=NOW)
    assert rec.ratio_measured_to_estimate == pytest.approx(1.1, abs=1e-3) and rec.within_margin
    over = bmc.calibrate_from_measurement(est, est.total_mib * 1.5, measured_kind="process_peak", gpu="Tesla T4", source="s")
    assert not over.within_margin  # negative control: an estimator undershoot beyond the margin is flagged
    alloc = bmc.calibrate_from_measurement(est, 1000.0, measured_kind="torch_max_allocated", gpu="Tesla T4", source="s")
    assert alloc.estimate_mib == pytest.approx(est.total_mib - bmc.CONTEXT_ALLOWANCE_MIB, abs=0.2)
    for kw in ({"measured_peak_mib": 0.0}, {"measured_peak_mib": math.inf}, {"measured_kind": "guess"}, {"source": " "}):
        args = {"measured_peak_mib": 900.0, "measured_kind": "process_peak", "gpu": "T4", "source": "s", **kw}
        with pytest.raises(ValueError):
            bmc.calibrate_from_measurement(est, args.pop("measured_peak_mib"), **args)
    zero = bmc.VramEstimate("m", "qlora_nf4_training", "c", {"cuda_context_allowance": 307.0}, 307.0)
    with pytest.raises(ValueError, match="positive"):
        bmc.calibrate_from_measurement(zero, 100.0, measured_kind="torch_max_allocated", gpu="T4", source="s")
    path = tmp_path / "cal.json"
    assert bmc.append_calibration(path, rec) == 1
    assert bmc.append_calibration(path, over) == 2
    assert json.loads(path.read_text())[0]["gpu"] == "Tesla T4"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON list"):
        bmc.append_calibration(path, rec)


# --- 3. promotion gate ---------------------------------------------------------------------------------------------------------

def _pack(n: int, correct: set[int]) -> dict[str, bool]:
    return {f"item{i}": i in correct for i in range(n)}


def test_mcnemar_known_values_and_errors() -> None:
    assert bmc.mcnemar_exact_p(0, 0) == 1.0
    assert bmc.mcnemar_exact_p(0, 10) == pytest.approx(2 / 1024)
    assert bmc.mcnemar_exact_p(5, 5) == 1.0
    with pytest.raises(ValueError):
        bmc.mcnemar_exact_p(-1, 2)


@settings(max_examples=60, deadline=None)
@given(b=st.integers(0, 60), c=st.integers(0, 60))
def test_mcnemar_matches_verified_update_gate(b: int, c: int) -> None:
    from hivemind.swarm.verified_update_gate import exact_paired_p

    assert bmc.mcnemar_exact_p(b, c) == pytest.approx(exact_paired_p(b, c), rel=1e-12)
    assert bmc.mcnemar_exact_p(b, c) == bmc.mcnemar_exact_p(c, b)


@settings(max_examples=50, deadline=None)
@given(outcomes=st.lists(st.booleans(), min_size=50, max_size=200))
def test_identical_outcomes_never_promote(outcomes: list[bool]) -> None:
    pack = {f"i{k}": v for k, v in enumerate(outcomes)}
    guard = {f"g{k}": v for k, v in enumerate(outcomes[:20])}
    r = bmc.promotion_gate(pack, dict(pack), guard, dict(guard))
    assert r.decision == bmc.HOLD and not r.promoted
    assert r.wins == r.losses == 0 and r.p_two_sided == 1.0


def test_clearly_better_candidate_promotes() -> None:
    incumbent = _pack(100, set(range(40)))
    candidate = _pack(100, set(range(70)))  # 30 wins, 0 losses
    guard = _pack(30, set(range(25)))
    r = bmc.promotion_gate(candidate, incumbent, guard, dict(guard))
    assert r.decision == bmc.PROMOTE and r.promoted
    assert r.wins == 30 and r.losses == 0 and r.p_two_sided < 1e-6


def test_better_candidate_with_guard_regression_holds() -> None:
    incumbent, candidate = _pack(100, set(range(40))), _pack(100, set(range(70)))
    guard_inc, guard_cand = _pack(30, set(range(25))), _pack(30, set(range(24)))  # one guard loss
    r = bmc.promotion_gate(candidate, incumbent, guard_cand, guard_inc)
    assert r.decision == bmc.HOLD and any("guard-pack regression" in x for x in r.reasons)
    lenient = bmc.promotion_gate(candidate, incumbent, guard_cand, guard_inc, bmc.PromotionPolicy(tolerated_guard_net_loss=1))
    assert lenient.decision == bmc.PROMOTE
    big_loss = bmc.promotion_gate(candidate, incumbent, _pack(40, set()), _pack(40, set(range(12))),
                                  bmc.PromotionPolicy(tolerated_guard_net_loss=20))
    assert big_loss.decision == bmc.HOLD and any("significant guard-pack regression" in x for x in big_loss.reasons)


def test_worse_or_insignificant_candidate_holds() -> None:
    incumbent = _pack(100, set(range(70)))
    worse = _pack(100, set(range(40)))
    guard = _pack(30, set(range(25)))
    r = bmc.promotion_gate(worse, incumbent, guard, dict(guard))
    assert r.decision == bmc.HOLD and r.losses == 30
    slight = bmc.promotion_gate(_pack(100, set(range(72))), incumbent, guard, dict(guard))  # 2 wins, 0 losses: p = 0.5
    assert slight.decision == bmc.HOLD and slight.p_two_sided == pytest.approx(0.5)


def test_unpaired_or_small_packs_are_insufficient() -> None:
    guard = _pack(30, set())
    r = bmc.promotion_gate(_pack(100, set(range(70))), _pack(99, set()), guard, dict(guard))
    assert r.decision == bmc.INSUFFICIENT and any("item ids differ" in x for x in r.reasons)
    small = bmc.promotion_gate(_pack(10, set(range(10))), _pack(10, set()), guard, dict(guard))
    assert small.decision == bmc.INSUFFICIENT and small.to_dict()["n_items"] == 10
    with pytest.raises(ValueError, match="booleans"):
        bmc.promotion_gate({"a": 1}, {"a": True}, guard, dict(guard))  # type: ignore[dict-item]
    for kw in ({"alpha": 1.0}, {"min_items": 0}, {"tolerated_guard_net_loss": -1}):
        with pytest.raises(ValueError):
            bmc.PromotionPolicy(**kw)


# --- 4. upgrade plan -----------------------------------------------------------------------------------------------------------

def test_gpu_tiers_match_s19_hardware_tiers() -> None:
    from hivemind.core.qwen35_usage_moe_orchestrator import QWEN35_HARDWARE_TIERS

    budgets = {t.tier_id: t.budget_mib for t in bmc.GPU_TIERS}
    for tier_id in ("std_8gb", "big_12gb", "xl_16gb_plus"):
        assert budgets[tier_id] == QWEN35_HARDWARE_TIERS[tier_id]["max_wddm_safe_mib"]
    assert budgets["t4_serving_resident"] < budgets["t4_serving_stopped"]


def test_plan_is_derived_and_monotone_in_budget() -> None:
    cat = bmc.load_catalog(CATALOG_PATH)
    plan = bmc.recommend_upgrade_plan(cat)
    tiers = sorted(plan["tiers"], key=lambda t: t["budget_mib"])
    size = {m["model_id"]: m["header"]["params_total"] for m in cat["models"]}
    picks = [size[t["train_candidate"]] if t["train_candidate"] else 0 for t in tiers]
    assert picks == sorted(picks)  # more memory never selects a smaller training candidate
    for t in tiers:
        for mid, row in t["models"].items():
            if row["train"]:
                assert row["train"]["required_with_margin_mib"] <= t["budget_mib"]
            else:  # negative control: no rung of the ladder fits with the margin
                g = bmc.geometry_from_entry(next(m for m in cat["models"] if m["model_id"] == mid))
                assert all(not bmc.fit_verdict(bmc.estimate_qlora_training(g, r).total_mib, t["budget_mib"]).fits
                           for r in bmc.TRAIN_LADDER)
        if t["train_candidate"]:
            assert t["models"][t["train_candidate"]]["role"] == bmc.CANDIDATE
    assert set(plan["not_found"]) == {m["model_id"] for m in cat["not_found"]}


def test_plan_keeps_incumbent_when_nothing_fits_and_rejects_bad_catalog() -> None:
    cat = bmc.load_catalog(CATALOG_PATH)
    tiny = bmc.recommend_upgrade_plan(cat, tiers=(bmc.GpuTier("tiny", 100, "test"),))
    assert tiny["tiers"][0]["train_candidate"] is None
    assert tiny["tiers"][0]["action"].startswith("KEEP_INCUMBENT")
    bad = copy.deepcopy(cat)
    bad["models"] = [m for m in bad["models"] if m["role"] != bmc.INCUMBENT]
    with pytest.raises(ValueError, match="no incumbent"):
        bmc.recommend_upgrade_plan(bad)


def test_main_prints_plan(capsys: pytest.CaptureFixture[str]) -> None:
    assert bmc.main(["--catalog", str(CATALOG_PATH)]) == 0
    text = capsys.readouterr().out
    assert "t4_serving_stopped" in text and "ESTIMATE" in text
    assert bmc.main(["--catalog", str(CATALOG_PATH), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["tiers"]) == len(bmc.GPU_TIERS)


def test_refresh_script_writes_validated_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    spec = importlib.util.spec_from_file_location("refresh_bmc", REPO / "scripts" / "refresh_base_model_catalog.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    good = bmc.build_catalog(_toy_hub(), [("toy/Toy-1", bmc.INCUMBENT)], now=NOW)
    monkeypatch.setattr(mod, "build_catalog", lambda http_get: good)
    out = tmp_path / "c.json"
    assert mod.main(["--out", str(out)]) == 0
    assert json.loads(out.read_text())["models"][0]["model_id"] == "toy/Toy-1"
    bad = copy.deepcopy(good)
    bad["models"][0]["revision_sha"] = "main"
    monkeypatch.setattr(mod, "build_catalog", lambda http_get: bad)
    out2 = tmp_path / "c2.json"
    assert mod.main(["--out", str(out2)]) == 1
    assert not out2.exists()


@settings(max_examples=60, deadline=None)
@given(n=st.integers(1, 80), k=st.integers(0, 80))
def test_binomial_half_tail_bounds_and_symmetry(n: int, k: int) -> None:
    assert bmc.binom_half_lower_tail(-1, n) == 0.0
    assert bmc.binom_half_lower_tail(n, n) == 1.0
    k = min(k, n - 1)
    # P(X <= k) + P(X <= n - k - 1) = 1 for a fair coin, by symmetry
    assert bmc.binom_half_lower_tail(k, n) + bmc.binom_half_lower_tail(n - k - 1, n) == pytest.approx(1.0, abs=1e-12)
