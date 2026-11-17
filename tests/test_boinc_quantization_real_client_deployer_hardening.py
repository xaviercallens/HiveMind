# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and invariant hardening tests for quantization, BOINC real_client, and RTX BOINC deployer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from hivemind.boinc.real_client.app import q16_digest
from hivemind.boinc.real_client.moe_app import (
    eval_by_domain,
    eval_loss,
    infer,
    load_peft_model,
    state_sha,
    train,
)
from hivemind.boinc.real_client.moe_work import (
    build_train_spec,
    interleave,
    load_custom_shards,
)
from hivemind.boinc.rtx_boinc_deployer import run_rtx_boinc_client_cycle
from hivemind.config import HardwareProfile
from hivemind.core.quantization import (
    attach_qlora_adapters,
    load_hf_qlora_model,
    stream_load_hf_causal_lm,
)


def test_quantization_attach_stream_and_load_validation() -> None:
    tiny = nn.Linear(8, 8)
    with pytest.raises(ValueError, match="base_model"):
        attach_qlora_adapters("not-a-module")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rank_r"):
        attach_qlora_adapters(tiny, rank_r=0)
    with pytest.raises(ValueError, match="lora_alpha"):
        attach_qlora_adapters(tiny, lora_alpha=0)
    with pytest.raises(ValueError, match="target_modules"):
        attach_qlora_adapters(tiny, target_modules=[""])
    with pytest.raises(ValueError, match="gradient_checkpointing"):
        attach_qlora_adapters(tiny, gradient_checkpointing="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        attach_qlora_adapters(tiny, seed=True)  # type: ignore[arg-type]

    dev = torch.device("cpu")
    with pytest.raises(ValueError, match="model_id"):
        stream_load_hf_causal_lm("   ", dev, torch.float32)
    with pytest.raises(ValueError, match="device"):
        stream_load_hf_causal_lm("Qwen/Qwen2.5-0.5B", "cpu", torch.float32)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="compute_dtype"):
        stream_load_hf_causal_lm("Qwen/Qwen2.5-0.5B", dev, "float32")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="apply_nf4"):
        stream_load_hf_causal_lm("Qwen/Qwen2.5-0.5B", dev, torch.float32, apply_nf4="yes")  # type: ignore[arg-type]

    prof = HardwareProfile(
        name="rtx2070",
        description="RTX 2070 8GB",
        target_vram_gb=8.0,
        available_vram_budget_gb=6.5,
        hf_model_id="Qwen/Qwen2.5-0.5B",
    )
    with pytest.raises(ValueError, match="profile"):
        load_hf_qlora_model("not-a-profile")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="device"):
        load_hf_qlora_model(prof, device="cpu")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        load_hf_qlora_model(prof, seed=False)  # type: ignore[arg-type]


def test_boinc_real_client_app_q16_digest_validation() -> None:
    digest = q16_digest({"a": torch.ones(2, 2)})
    assert len(digest) == 64

    with pytest.raises(ValueError, match="state"):
        q16_digest({})
    with pytest.raises(ValueError, match="state"):
        q16_digest("not-a-dict")  # type: ignore[arg-type]


def test_boinc_real_client_moe_app_validation(tmp_path: Path) -> None:
    valid_spec = {
        "model_id": "ibm-granite/granite-3.1-1b-a400m-base",
        "init_seed": 42,
        "rank": 8,
        "alpha": 16,
        "prompts": ["Hello world"],
        "max_new_tokens": 8,
    }
    with pytest.raises(TypeError, match="spec"):
        load_peft_model("not-a-dict")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model_id"):
        load_peft_model({**valid_spec, "model_id": " "})
    with pytest.raises(ValueError, match="init_seed"):
        load_peft_model({**valid_spec, "init_seed": True})
    with pytest.raises(ValueError, match="rank"):
        load_peft_model({**valid_spec, "rank": 0})
    with pytest.raises(ValueError, match="adapter_state"):
        load_peft_model(valid_spec, adapter_state={})

    with pytest.raises(TypeError, match="spec"):
        eval_by_domain(None, "not-a-dict")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="eval_sets"):
        eval_by_domain(None, {})

    with pytest.raises(ValueError, match="tokens"):
        eval_loss(None, [])
    with pytest.raises(ValueError, match="batch"):
        eval_loss(None, [[1, 2, 3]], batch=0)

    sha = state_sha({"lora_A": torch.ones(2, 2)})
    assert len(sha) == 64
    with pytest.raises(ValueError, match="state"):
        state_sha({})

    ckpt = tmp_path / "ckpt.pt"
    with pytest.raises(TypeError, match="spec"):
        train("not-a-dict", ckpt)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ckpt"):
        train(valid_spec, "not-a-path")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="crash_after_segment"):
        train(valid_spec, ckpt, crash_after_segment=0)
    with pytest.raises(ValueError, match="lazy"):
        train(valid_spec, ckpt, lazy="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="start_state"):
        train(valid_spec, ckpt, start_state={})

    with pytest.raises(TypeError, match="spec"):
        infer("not-a-dict", {"w": torch.ones(2)})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="adapter_state"):
        infer(valid_spec, {})
    with pytest.raises(ValueError, match="prompts"):
        infer({**valid_spec, "prompts": []}, {"w": torch.ones(2)})
    with pytest.raises(ValueError, match="max_new_tokens"):
        infer({**valid_spec, "max_new_tokens": 0}, {"w": torch.ones(2)})


def test_boinc_real_client_moe_work_validation() -> None:
    out = interleave([[[1, 2], [3, 4]], [[5, 6]]])
    assert out == [[1, 2], [5, 6], [3, 4]]

    with pytest.raises(ValueError, match="parts"):
        interleave("not-a-list")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="parts"):
        interleave(["not-a-list"])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="custom_dir"):
        load_custom_shards("   ")
    with pytest.raises(ValueError, match="secret"):
        load_custom_shards("/tmp/nonexistent", secret=b"")

    with pytest.raises(ValueError, match="wu_id"):
        build_train_spec("   ")
    with pytest.raises(ValueError, match="shard_index"):
        build_train_spec("wu-1", shard_index=-1)
    with pytest.raises(ValueError, match="model_id"):
        build_train_spec("wu-1", model_id="")
    with pytest.raises(ValueError, match="inner_steps"):
        build_train_spec("wu-1", inner_steps=0)
    with pytest.raises(ValueError, match="lr"):
        build_train_spec("wu-1", lr=0.0)
    with pytest.raises(ValueError, match="rng_seed"):
        build_train_spec("wu-1", rng_seed=True)  # type: ignore[arg-type]


def test_rtx_boinc_client_cycle_validation() -> None:
    with pytest.raises(ValueError, match="server_url"):
        run_rtx_boinc_client_cycle(server_url="ftp://invalid")
    with pytest.raises(ValueError, match="handle"):
        run_rtx_boinc_client_cycle(handle="   ")
    with pytest.raises(ValueError, match="node_id"):
        run_rtx_boinc_client_cycle(node_id="   ")
    with pytest.raises(ValueError, match="preferred_domain"):
        run_rtx_boinc_client_cycle(preferred_domain="")
    with pytest.raises(ValueError, match="device"):
        run_rtx_boinc_client_cycle(device=" ")
