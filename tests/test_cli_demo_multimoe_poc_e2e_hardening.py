# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-control boundary tests for demo_hf_domain_swarm,
multi_moe_hf_campaign, poc_rtx2070, and simulate_e2e."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from hivemind.cli.demo_hf_domain_swarm import (
    _execute_single_domain_on_loaded_model,
    _format_astrophysics_hf_dataset_subset,
)
from hivemind.cli.multi_moe_hf_campaign import (
    _build_seeded_campaign_peft_model,
    _extract_lora_tensors_cpu,
    _load_lora_tensors_into_model,
    _try_fetch_live_hf_rows,
    main as multimoe_main,
    run_multi_moe_hf_campaign,
)
from hivemind.cli.poc_rtx2070 import (
    _create_tokenized_parquet_shard,
    fetch_hf_domain_subset as poc_fetch_hf_domain_subset,
    run_huggingface_rtx2070_poc,
)
from hivemind.cli.simulate_e2e import (
    estimate_workunit_tflops,
    run_end_to_end_simulation,
)


def test_demo_hf_domain_swarm_rejects_invalid_astrophysics_and_execution_inputs() -> None:
    for bad_tok in (True, 123, ["token"]):
        with pytest.raises(ValueError, match="hf_token must be a string or None"):
            _format_astrophysics_hf_dataset_subset(
                "mzrk36/exoplanets-unified-dataset",
                num_samples=4,
                hf_token=bad_tok,  # type: ignore[arg-type]
            )

    model = nn.Linear(4, 4)
    base_kwargs = {
        "exp_index": 1,
        "dataset_id": "sahil2801/CodeAlpaca-20k",
        "domain_name": "Python_Systems_Code",
        "eval_prompt": "### Instruction:\n",
        "consumer_node_id": "node-1",
        "provider_node_ids": ["node-2", "node-3"],
        "model": model,
        "tokenizer": object(),
        "initial_lora_state": {"w": torch.ones(4)},
        "base_model_sha256": "a" * 64,
        "nf4_mb": 12.5,
        "store": object(),
        "ledger": object(),
        "daemon": object(),
        "validator": object(),
        "outer_opt": object(),
        "lora_rank": 8,
        "inner_steps": 4,
        "seq_len": 16,
        "num_samples": 12,
    }
    for override, match in (
        ({"exp_index": 0}, "exp_index"),
        ({"exp_index": True}, "exp_index"),
        ({"dataset_id": "  "}, "dataset_id"),
        ({"domain_name": ""}, "domain_name"),
        ({"eval_prompt": "   "}, "eval_prompt"),
        ({"consumer_node_id": ""}, "consumer_node_id"),
        ({"base_model_sha256": ""}, "base_model_sha256"),
        ({"provider_node_ids": []}, "provider_node_ids"),
        ({"provider_node_ids": ["node-2", ""]}, "provider_node_ids"),
        ({"provider_node_ids": "node-2"}, "provider_node_ids"),
        ({"model": "not-a-module"}, "model must be a torch.nn.Module"),
        ({"initial_lora_state": {}}, "initial_lora_state"),
        ({"nf4_mb": -1.0}, "nf4_mb"),
        ({"nf4_mb": float("nan")}, "nf4_mb"),
        ({"nf4_mb": True}, "nf4_mb"),
        ({"lora_rank": 0}, "lora_rank"),
        ({"inner_steps": True}, "inner_steps"),
        ({"seq_len": 0}, "seq_len"),
        ({"num_samples": 0}, "num_samples"),
    ):
        with pytest.raises(ValueError, match=match):
            _execute_single_domain_on_loaded_model(**{**base_kwargs, **override})  # type: ignore[arg-type]


def test_multi_moe_hf_campaign_helpers_and_entrypoint_validation() -> None:
    for bad_ds in ("", "   ", True, 123):
        with pytest.raises(ValueError, match="dataset_id"):
            _try_fetch_live_hf_rows(bad_ds)  # type: ignore[arg-type]
    for bad_rows in (0, True, 2.5):
        with pytest.raises(ValueError, match="max_rows"):
            _try_fetch_live_hf_rows("qiaojin/PubMedQA", max_rows=bad_rows)  # type: ignore[arg-type]
    for bad_timeout in (0.0, -1.0, float("nan"), True):
        with pytest.raises(ValueError, match="timeout_sec"):
            _try_fetch_live_hf_rows("qiaojin/PubMedQA", timeout_sec=bad_timeout)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="seed"):
        _build_seeded_campaign_peft_model(-1, torch.device("cpu"))
    with pytest.raises(ValueError, match="seed"):
        _build_seeded_campaign_peft_model(True, torch.device("cpu"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="device"):
        _build_seeded_campaign_peft_model(42, "cpu")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="model must be a torch.nn.Module"):
        _extract_lora_tensors_cpu("not-a-model")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model must be a torch.nn.Module"):
        _load_lora_tensors_into_model("not-a-model", {"w": torch.ones(2)})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="state must be a non-empty Mapping"):
        _load_lora_tensors_into_model(nn.Linear(2, 2), {})  # type: ignore[arg-type]

    for bad_kwargs in (
        {"steps_per_domain": 4.0},
        {"steps_per_domain": "4"},
        {"domain_limit": 4.0},
        {"domain_limit": "4"},
        {"learning_rate": "0.01"},
        {"sequence_length": 1},
        {"sequence_length": 256},
        {"min_quorum_k": 2.0},
        {"grace_margin_gamma": "0.5"},
        {"quorum_merge_mode": ["rda"]},
    ):
        with pytest.raises(ValueError):
            run_multi_moe_hf_campaign(preferred_device="cpu", output_path=None, **bad_kwargs)  # type: ignore[arg-type]

    for bad_argv in ("--tier quick", b"--tier", 123):
        with pytest.raises(ValueError, match="argv must be a sequence"):
            multimoe_main(bad_argv)  # type: ignore[arg-type]


def test_poc_rtx2070_and_simulate_e2e_reject_non_int_and_raw_string_texts(tmp_path: Path) -> None:
    for bad_samples in (4.5, "24", True, 0):
        with pytest.raises(ValueError, match="num_samples"):
            poc_fetch_hf_domain_subset("sahil2801/CodeAlpaca-20k", num_samples=bad_samples)  # type: ignore[arg-type]
    for bad_token in (True, 123):
        with pytest.raises(ValueError, match="hf_token"):
            poc_fetch_hf_domain_subset("sahil2801/CodeAlpaca-20k", hf_token=bad_token)  # type: ignore[arg-type]

    dummy_tok = SimpleNamespace(encode=lambda text, add_special_tokens=False: [1, 2, 3, 4])
    with pytest.raises(ValueError, match="texts"):
        _create_tokenized_parquet_shard(
            dummy_tok,
            "Code",
            "raw-string-not-list",  # type: ignore[arg-type]
            tmp_path / "s.parquet",
        )
    for bad_kwargs in (
        {"seq_len": 16.0},
        {"num_sequences": 4.0},
    ):
        with pytest.raises(ValueError):
            _create_tokenized_parquet_shard(
                dummy_tok,
                "Code",
                ["def f(): return 1"],
                tmp_path / "s.parquet",
                **bad_kwargs,  # type: ignore[arg-type]
            )

    shard_path, sha, sig = _create_tokenized_parquet_shard(
        dummy_tok,
        "Code",
        ["def f(): return 1"],
        tmp_path / "ok.parquet",
        seq_len=8,
        num_sequences=2,
    )
    assert shard_path.exists()
    assert len(sha) == 64 and len(sig) == 64

    for bad_poc_kwargs in (
        {"hf_subset_size": 24.0},
        {"lora_rank_r": 16.0},
        {"lora_alpha": 32.0},
        {"inner_steps_H": 12.0},
        {"spot_check_interval_s": 6.0},
    ):
        with pytest.raises(ValueError):
            run_huggingface_rtx2070_poc(**bad_poc_kwargs)  # type: ignore[arg-type]

    valid_wu = SimpleNamespace(inner_steps_H=10, seq_len=32)
    assert estimate_workunit_tflops(valid_wu, total_params=100_000_000, micro_batch_size=2) > 0.0
    for bad_wu in (
        SimpleNamespace(inner_steps_H=10.0, seq_len=32),
        SimpleNamespace(inner_steps_H=10, seq_len=32.0),
    ):
        with pytest.raises(ValueError, match="inner_steps_H and seq_len"):
            estimate_workunit_tflops(bad_wu)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="total_params"):
        estimate_workunit_tflops(valid_wu, total_params=100.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="micro_batch_size"):
        estimate_workunit_tflops(valid_wu, micro_batch_size=2.0)  # type: ignore[arg-type]

    for bad_sim_kwargs in (
        {"inner_steps_H": 20.0},
        {"spot_check_interval_s": 10.0},
    ):
        with pytest.raises(ValueError):
            run_end_to_end_simulation(**bad_sim_kwargs)  # type: ignore[arg-type]
