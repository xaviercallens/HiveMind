"""Tests for scripts/qlora_vram_estimate.py: synthetic configs, monotonicity, checkpointing, and the calibration points (CPU, no GPU, no weights)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "qlora_vram_estimate.py"
_spec = importlib.util.spec_from_file_location("qlora_vram_estimate", _PATH)
q = importlib.util.module_from_spec(_spec)
sys.modules["qlora_vram_estimate"] = q
_spec.loader.exec_module(q)  # type: ignore[union-attr]

# config.json values of the three calibration models (copied from the local cache; embedded so the test needs no cache)
GRANITE = {"hidden_size": 1024, "num_hidden_layers": 24, "vocab_size": 49152, "intermediate_size": 512, "num_attention_heads": 16,
           "num_key_value_heads": 8, "tie_word_embeddings": True, "num_local_experts": 32, "num_experts_per_tok": 8}
QWEN3_4B = {"hidden_size": 2560, "num_hidden_layers": 36, "vocab_size": 151936, "intermediate_size": 9728, "num_attention_heads": 32,
            "num_key_value_heads": 8, "head_dim": 128, "tie_word_embeddings": True}
PHI4_MINI = {"hidden_size": 3072, "num_hidden_layers": 32, "vocab_size": 200064, "intermediate_size": 8192, "num_attention_heads": 24,
             "num_key_value_heads": 8, "tie_word_embeddings": True}
TINY_DENSE = {"hidden_size": 256, "num_hidden_layers": 4, "vocab_size": 1000, "intermediate_size": 512, "num_attention_heads": 4,
              "num_key_value_heads": 4, "tie_word_embeddings": True}
TINY_MOE = {**TINY_DENSE, "num_local_experts": 8, "num_experts_per_tok": 2, "intermediate_size": 128}


def specs():
    return {"granite-3.1-1b-a400m-base": q.spec_from_config(GRANITE, "granite-3.1-1b-a400m-base"),
            "Qwen3-4B-Instruct-2507": q.spec_from_config(QWEN3_4B, "Qwen3-4B-Instruct-2507"),
            "Phi-4-mini-instruct": q.spec_from_config(PHI4_MINI, "Phi-4-mini-instruct")}


def test_parameter_counts_match_the_known_sizes():
    g = q.spec_from_config(GRANITE, "g")
    assert 1.30e9 < g.total_params < 1.36e9  # "1.3B total" in docs/T4_NIGHT_MODEL_DECISION.md
    assert 0.40e9 < g.active_params < 0.46e9  # "0.4B active"
    assert g.fused_expert_params > 0.85 * g.total_params  # the caveat: almost everything is fused experts, kept 16-bit
    assert 1_376_256 == q.lora_params(g, q.Recipe())  # docs/BOINC_CORE_NODE.md: 1,376,256 trainable parameters at rank 8 on q,k,v,o


def test_weights_only_matches_measured_resident_sizes():
    for key, measured in (("granite-3.1-1b-a400m-base", 2444), ("Qwen3-4B-Instruct-2507", 2578), ("Phi-4-mini-instruct", 2764)):
        m = specs()[key]
        b = q.breakdown(m, replace(q.BASE_RECIPE, targets=()), 0.0)
        assert abs(b["resident_weights"] - measured) / measured < 0.03, (key, b["resident_weights"])


def test_moe_experts_stay_16_bit_and_do_not_get_mlp_adapters():
    m = q.spec_from_config(TINY_MOE, "m")
    b = q.breakdown(m, q.Recipe(), 1.0)
    assert b["fp16_weights"] >= m.fused_expert_params * 2 / q.MIB
    all_lin = q.lora_params(m, q.Recipe(targets=q.ALL_LINEAR))
    assert all_lin == q.lora_params(m, q.Recipe())  # gate/up/down add nothing on a MoE
    dense = q.spec_from_config(TINY_DENSE, "d")
    assert q.lora_params(dense, q.Recipe(targets=q.ALL_LINEAR)) > q.lora_params(dense, q.Recipe())


@pytest.mark.parametrize("cfg", [TINY_DENSE, TINY_MOE])
def test_monotone_in_rank_sequence_and_micro_batch(cfg):
    m = q.spec_from_config(cfg, "m")
    base = q.Recipe(rank=8, seq_len=128, micro_batch=2)
    p0 = q.breakdown(m, base)["process_peak"]
    assert q.breakdown(m, replace(base, rank=16))["process_peak"] > p0
    assert q.breakdown(m, replace(base, rank=64))["process_peak"] > q.breakdown(m, replace(base, rank=16))["process_peak"]
    assert q.breakdown(m, replace(base, seq_len=256))["process_peak"] > p0
    assert q.breakdown(m, replace(base, micro_batch=4))["process_peak"] > p0
    assert q.breakdown(m, replace(base, targets=q.ALL_LINEAR))["process_peak"] >= p0


def test_checkpointing_lowers_activations_and_peak():
    m = specs()["Qwen3-4B-Instruct-2507"]
    r = q.Recipe(seq_len=512, micro_batch=4)
    off, on = q.breakdown(m, r), q.breakdown(m, replace(r, checkpointing=True))
    assert on["activations_layers"] < off["activations_layers"]
    assert on["process_peak"] < off["process_peak"]
    assert on["logits"] == off["logits"]  # checkpointing does not shrink the logits


def test_optimizer_state_ordering_and_validation():
    m = q.spec_from_config(TINY_DENSE, "d")
    r = q.Recipe(rank=64, targets=q.ALL_LINEAR)
    peaks = {o: q.breakdown(m, replace(r, optimizer=o))["lora_state"] for o in q.OPTIMIZERS}
    assert peaks["adamw"] > peaks["sgd_momentum"] > peaks["adamw8bit"] > peaks["sgd"]
    assert peaks["adamw"] > peaks["sgd"]
    with pytest.raises(ValueError):
        q.breakdown(m, replace(r, optimizer="lion"))
    with pytest.raises(ValueError):
        q.breakdown(m, replace(r, micro_batch=0))


def test_calibration_points_reproduced_within_the_stated_error():
    cal = q.calibrate(specs())
    assert cal["available"]
    assert cal["loo_max_abs_error"] <= q.THRESHOLD  # if this ever fails the doc must say "order of magnitude only"
    for p in cal["points"]:
        assert abs(p["in_sample_rel_error"]) <= q.THRESHOLD, p
        assert abs(p["loo_rel_error"]) <= q.THRESHOLD, p
        assert abs(p["resident_rel_error"]) <= 0.03, p
    # the two out-of-memory points must be predicted at or above the allocator cap they hit (the estimate must not call them comfortable)
    for p in cal["points"]:
        if p["censored_oom_lower_bound"]:
            assert p["in_sample_pred_mib"] >= q.CAP_MIB, p
    assert q.OVERHEAD_MIB == 3716 - 3409


def test_calibration_reports_poor_when_the_error_is_large():
    # a deliberately wrong measurement must flip the verdict (the script must not hide a bad fit)
    bad = tuple(replace(p, measured_mib=p.measured_mib * 0.5) if p.key == "granite-1b-a400m" else p for p in q.CALIBRATION_POINTS)
    old = q.CALIBRATION_POINTS
    q.CALIBRATION_POINTS = bad
    try:
        cal = q.calibrate(specs())
    finally:
        q.CALIBRATION_POINTS = old
    assert cal["poor"] and "order-of-magnitude" in cal["verdict"]


def test_table_has_every_budget_and_is_ordered_by_budget():
    s = specs()
    cal = q.calibrate(s)
    rows = q.table(s, cal["act_scale"], max(cal["loo_max_abs_error"], q.THRESHOLD))
    assert len(rows) == len(s) * len(q.RECIPES)
    order = {"NO": 0, "marginal": 1, "fits": 2}
    for r in rows:
        v = [order[r[f"tier_{b}"]] for b in q.BUDGETS]
        assert v == sorted(v)  # a bigger budget never fits less
    granite_r0 = next(r for r in rows if r["model"].startswith("granite") and r["recipe"].startswith("R0"))
    assert granite_r0["tier_4300"] == "fits"
    qwen_r0 = next(r for r in rows if r["model"].startswith("Qwen3-4B") and r["recipe"].startswith("R0"))
    assert qwen_r0["tier_4300"] == "NO"


def test_cli_json_runs_without_a_cache(tmp_path, capsys):
    rc = q.main(["--hub", str(tmp_path), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["calibration"]["available"] is False and out["table"] == []


def test_cli_config_mode(tmp_path, capsys):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(TINY_DENSE))
    assert q.main(["--hub", str(tmp_path), "--config", str(p), "--rank", "16", "--seq", "256", "--checkpointing"]) == 0
    assert "process_peak" in json.loads(capsys.readouterr().out)["breakdown_mib"]


@pytest.mark.skipif(not os.path.isdir(q.HF_HUB_DEFAULT), reason="local Hugging Face cache not present")
def test_cache_presets_agree_with_embedded_configs():
    cal_cache = q.calibrate(q.load_presets())
    cal_emb = q.calibrate(specs())
    assert cal_cache["available"]
    assert abs(cal_cache["act_scale"] - cal_emb["act_scale"]) < 1e-6
