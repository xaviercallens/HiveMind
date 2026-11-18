"""Negative-control tests for CLI campaign, RTX 2070 PoC, E2E simulation, and demo perplexity hardening."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import hivemind.cli.poc_rtx2070 as poc_mod
from hivemind.cli.demo_hf_domain_swarm import _evaluate_batch_perplexity
from hivemind.cli.multi_moe_hf_campaign import run_multi_moe_hf_campaign
from hivemind.cli.poc_rtx2070 import (
    _create_tokenized_parquet_shard,
    _export_shared_swarm_expert,
)
from hivemind.cli.simulate_e2e import (
    estimate_workunit_tflops,
    run_end_to_end_simulation,
)


def test_run_multi_moe_hf_campaign_rejects_invalid_seed_stream_and_output_path(tmp_path: Path) -> None:
    out_file = tmp_path / "report.json"
    with pytest.raises(ValueError, match="sequence_length"):
        run_multi_moe_hf_campaign(sequence_length="96", output_path=out_file)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        run_multi_moe_hf_campaign(seed=-1, output_path=out_file)
    with pytest.raises(ValueError, match="seed"):
        run_multi_moe_hf_campaign(seed=True, output_path=out_file)
    with pytest.raises(ValueError, match="seed"):
        run_multi_moe_hf_campaign(seed="42", output_path=out_file)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="live_hf_stream"):
        run_multi_moe_hf_campaign(live_hf_stream=1, output_path=out_file)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="output_path"):
        run_multi_moe_hf_campaign(output_path="not_a_path")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="output_path"):
        run_multi_moe_hf_campaign(output_path=True)  # type: ignore[arg-type]
    assert not out_file.exists()


def test_create_tokenized_parquet_shard_rejects_invalid_shard_path_and_rng_seed(tmp_path: Path) -> None:
    dummy_tok = SimpleNamespace(encode=lambda text, add_special_tokens=False: [10, 11, 12, 13, 14, 15])
    shard_file = tmp_path / "shard.parquet"
    out_path, sha256_hex, sig = _create_tokenized_parquet_shard(
        dummy_tok, "Code", ["def f(x): return x + 1"], shard_file, seq_len=4, num_sequences=2, rng_seed=7
    )
    assert out_path == shard_file
    assert len(sha256_hex) == 64
    assert len(sig) > 0

    with pytest.raises(ValueError, match="shard_path"):
        _create_tokenized_parquet_shard(
            dummy_tok, "Code", ["def f(x): return x + 1"], "not_a_path", seq_len=4, num_sequences=2  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="rng_seed"):
        _create_tokenized_parquet_shard(
            dummy_tok, "Code", ["def f(x): return x + 1"], shard_file, seq_len=4, num_sequences=2, rng_seed=True
        )
    with pytest.raises(ValueError, match="rng_seed"):
        _create_tokenized_parquet_shard(
            dummy_tok, "Code", ["def f(x): return x + 1"], shard_file, seq_len=4, num_sequences=2, rng_seed=-1
        )


def test_export_shared_swarm_expert_rejects_invalid_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(poc_mod, "ARTIFACTS_ROOT", tmp_path)
    valid_lora = {"lora_A": torch.ones((2, 4), dtype=torch.float32)}
    valid_traj = {"w0": {"steps": 4}}

    cid, export_path = _export_shared_swarm_expert("Code", "sahil2801/CodeAlpaca-20k", valid_lora, valid_traj)
    assert cid.startswith("cid://sha256/expert-code-")
    assert export_path.exists()

    with pytest.raises(ValueError, match="domain_name"):
        _export_shared_swarm_expert("   ", "sahil2801/CodeAlpaca-20k", valid_lora, valid_traj)
    with pytest.raises(ValueError, match="dataset_id"):
        _export_shared_swarm_expert("Code", "", valid_lora, valid_traj)
    with pytest.raises(ValueError, match="lora_state"):
        _export_shared_swarm_expert("Code", "sahil2801/CodeAlpaca-20k", {}, valid_traj)
    with pytest.raises(ValueError, match="worker_trajectories"):
        _export_shared_swarm_expert("Code", "sahil2801/CodeAlpaca-20k", valid_lora, [])  # type: ignore[arg-type]


def test_estimate_workunit_tflops_and_run_end_to_end_simulation_reject_invalid_inputs() -> None:
    valid_wu = SimpleNamespace(inner_steps_H=8, seq_len=32)
    assert estimate_workunit_tflops(valid_wu, total_params=100_000_000, micro_batch_size=2) > 0.0

    with pytest.raises(ValueError, match="workunit"):
        estimate_workunit_tflops(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="workunit"):
        estimate_workunit_tflops(SimpleNamespace(inner_steps_H=8))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inner_steps_H"):
        estimate_workunit_tflops(SimpleNamespace(inner_steps_H=True, seq_len=32))
    with pytest.raises(ValueError, match="seq_len"):
        estimate_workunit_tflops(SimpleNamespace(inner_steps_H=8, seq_len=True))

    with pytest.raises(ValueError, match="use_hf_hub"):
        run_end_to_end_simulation(use_hf_hub="yes")  # type: ignore[arg-type]


def test_evaluate_batch_perplexity_rejects_invalid_model_batch_and_pad_token_id() -> None:
    class _TinyDummyLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
        ) -> SimpleNamespace:
            return SimpleNamespace(loss=self.w * 1.25)

    model = _TinyDummyLM()
    batch = torch.tensor([[1, 2, 0]], dtype=torch.long)
    ppl = _evaluate_batch_perplexity(model, batch, pad_token_id=0)
    assert ppl > 1.0

    with pytest.raises(ValueError, match="model"):
        _evaluate_batch_perplexity("not_a_module", batch)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="token_batch"):
        _evaluate_batch_perplexity(model, [[1, 2]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="token_batch"):
        _evaluate_batch_perplexity(model, torch.empty((0,), dtype=torch.long))
    with pytest.raises(ValueError, match="pad_token_id"):
        _evaluate_batch_perplexity(model, batch, pad_token_id=True)
    with pytest.raises(ValueError, match="pad_token_id"):
        _evaluate_batch_perplexity(model, batch, pad_token_id=-1)
