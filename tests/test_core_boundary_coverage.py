"""CPU boundary contracts; execute only in the reviewed remote quarantine."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import numpy as np
import pytest
import torch
from torch import nn

import hivemind.config as config
from hivemind.compress import lattice_quantizer as lattice
from hivemind.core import btm_moe, btm_upcycle, diloco_inner, quantization
from hivemind.core.diloco_outer import DiLoCoOuterOptimizer
from hivemind.core.fomoe_sigma_routing import (
    sigmoid_independent_topk_routing,
    skip_token_flops_per_token,
)
from hivemind.data import token_training_eval as token_eval
from hivemind.security.gpu_execution_circuit_breaker import GPUExecutionCircuitBreaker


_TOKEN_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_TOKEN")


@pytest.fixture
def isolated_token_environment(monkeypatch):
    for key in _TOKEN_KEYS:
        # Record absent keys too, so direct resolver writes are removed at teardown.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    # Replace only config's os reference; changing global os.name breaks pathlib on Linux.
    monkeypatch.setattr(config, "os", SimpleNamespace(name="nt", environ=os.environ))


def test_cache_setup_loads_both_dotenv_files_and_creates_returned_directories(monkeypatch, tmp_path):
    dotenv = SimpleNamespace(load_dotenv=Mock())
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    monkeypatch.setattr(config, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(config, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    directory_keys = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME",
                      "UV_CACHE_DIR", "TMP", "TEMP")
    for key in directory_keys:
        monkeypatch.setenv(key, "old-cache")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setenv("HF_HUB_DISABLE_SYMLINKS_WARNING", "0")

    result = config.ensure_local_drive_caches()

    assert dotenv.load_dotenv.call_args_list == [
        call(config.PROJECT_ROOT / ".env", override=False),
        call(Path.home() / ".env", override=False),
    ]
    for key in directory_keys:
        assert os.environ[key] == result[key]
        assert Path(result[key]).is_dir()
        assert Path(result[key]).is_relative_to(tmp_path / "cache")
    assert (tmp_path / "artifacts").is_dir()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] == "0"


@pytest.mark.parametrize("hive", ["user", "machine"])
def test_registry_token_skips_missing_blank_and_nonstring_values(
    monkeypatch, isolated_token_environment, hive,
):
    context = MagicMock()
    registry_key = context.__enter__.return_value
    registry = SimpleNamespace(HKEY_CURRENT_USER="user", HKEY_LOCAL_MACHINE="machine")

    def open_key(candidate, subkey):
        if candidate != hive:
            raise OSError("inaccessible hive")
        return context

    registry.OpenKey = Mock(side_effect=open_key)
    registry.QueryValueEx = Mock(side_effect=[FileNotFoundError("missing"), (123, 1),
                                             (" \t ", 1), ("  hf_registry  ", 1)])
    hub = SimpleNamespace(get_token=Mock(side_effect=AssertionError("registry must win")))
    monkeypatch.setitem(sys.modules, "winreg", registry)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    assert config.get_hf_token() == "hf_registry"
    assert os.environ["HF_TOKEN"] == "hf_registry"
    assert registry.QueryValueEx.call_args_list == [call(registry_key, key) for key in _TOKEN_KEYS]
    assert registry.OpenKey.call_args_list == [
        call("user", "Environment"),
        *([call("machine", r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")]
          if hive == "machine" else []),
    ]
    context.__exit__.assert_called_once()
    hub.get_token.assert_not_called()


@pytest.mark.parametrize("hub_failure", [False, True])
def test_token_lookup_falls_back_after_both_registry_hives_fail(
    monkeypatch, isolated_token_environment, hub_failure,
):
    registry = SimpleNamespace(HKEY_CURRENT_USER="user", HKEY_LOCAL_MACHINE="machine",
                               OpenKey=Mock(side_effect=OSError("denied")))
    get_token = Mock(side_effect=RuntimeError("no cached credentials")) if hub_failure else Mock(return_value="hf_cached")
    monkeypatch.setitem(sys.modules, "winreg", registry)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(get_token=get_token))

    assert config.get_hf_token() == (None if hub_failure else "hf_cached")
    assert registry.OpenKey.call_count == 2
    get_token.assert_called_once_with()
    assert os.environ.get("HF_TOKEN") == (None if hub_failure else "hf_cached")


@pytest.mark.parametrize("decoder,shape,message", [
    (lattice.nearest_a2, (2, 3), "A2 lives in dimension 2"),
    (lattice.nearest_e8, (2, 7), "E8 lives in dimension 8"),
])
def test_lattice_decoders_reject_wrong_dimensions(decoder, shape, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        decoder(np.zeros(shape))


def test_a2_membership_rejects_wrong_dimension():
    with pytest.raises(ValueError, match="A2 lives in dimension 2"):
        lattice.is_lattice_point("A2", np.zeros((2, 3)))


@pytest.mark.parametrize("operation", [
    lambda: lattice.is_lattice_point("unknown", np.zeros((2, 2))),
    lambda: lattice.factorised_rate_bits_per_dim("unknown", np.zeros((2, 2))),
    lambda: lattice.LatticeDeltaQuantizer("unknown", scale=1.0),
])
def test_unknown_lattice_kind_is_reported_in_keyerror(operation):
    with pytest.raises(KeyError) as caught:
        operation()
    assert caught.value.args == ("unknown",)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_lattice_quantizer_requires_finite_positive_scale(scale):
    with pytest.raises(ValueError, match="scale must be positive and finite"):
        lattice.LatticeDeltaQuantizer("A2", scale=scale)


def test_unconditional_entropy_matches_known_binary_distribution():
    assert lattice._cond_entropy_bits(np.array([0, 0, 1, 1])) == pytest.approx(1.0)
    assert lattice._cond_entropy_bits(np.ones(4)) == pytest.approx(0.0)


def test_sign_error_feedback_reuses_residual_without_aliasing_the_result():
    quantizer = lattice.SignDeltaQuantizer(error_feedback=True)
    first = quantizer.quantize(torch.tensor([1.0, 3.0]))
    torch.testing.assert_close(first.dequantized, torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(first.residual, torch.tensor([-1.0, 1.0]))
    assert first.relative_mse == pytest.approx(0.2)
    first.residual.zero_()  # The quantizer owns an independent carry buffer.
    second = quantizer.quantize(torch.zeros(2))
    torch.testing.assert_close(second.dequantized, torch.tensor([-1.0, 1.0]))
    torch.testing.assert_close(second.residual, torch.zeros(2))
    torch.testing.assert_close(first.dequantized + second.dequantized, torch.tensor([1.0, 3.0]))
    assert second.relative_mse == 0.0
    assert second.rate_bits_per_dim == 1.0


@pytest.mark.parametrize("target", [-0.1, 0.1])
def test_scale_calibration_rejects_unreachable_rate_for_constant_zero_sample(target):
    with pytest.raises(ValueError, match="outside the reachable range"):
        lattice.calibrate_scale_for_rate("Z", torch.zeros(8), target)


@pytest.mark.parametrize("iterations", [0, 1])
def test_scale_calibration_iteration_limit_returns_the_actual_measured_rate(iterations):
    sample = torch.tensor([-1.0, 1.0])
    scale, rate = lattice.calibrate_scale_for_rate("Z", sample, 0.5, block=1, tol=0.0, iters=iterations)
    expected_scale = 1.0 if iterations == 0 else math.sqrt(1000.0)
    assert scale == pytest.approx(expected_scale)
    measured = lattice.LatticeDeltaQuantizer("Z", scale, block=1).quantize(sample)
    assert rate == pytest.approx(measured.rate_bits_per_dim)
    assert rate == (1.0 if iterations == 0 else 0.0)


@pytest.mark.parametrize("clusters", [0, -1])
def test_even_cluster_sizes_requires_a_positive_cluster_count(clusters):
    with pytest.raises(ValueError, match="num_clusters must be >= 1"):
        btm_moe._even_cluster_sizes(4, clusters)


@pytest.mark.parametrize("sizes", [[], [2, 0], [-1, 2]])
def test_hierarchical_router_rejects_empty_clusters(sizes):
    with pytest.raises(ValueError, match="every macro cluster needs at least one expert"):
        btm_moe.HierarchicalMoERouter(hidden_size=2, cluster_sizes=sizes)


def test_hierarchical_router_reports_largest_ragged_cluster():
    router = btm_moe.HierarchicalMoERouter(hidden_size=2, cluster_sizes=[1, 3, 2], top_k=2)
    assert router.experts_per_domain == 3
    assert router.num_experts == 6
    assert router.cluster_offsets == [0, 1, 4]
    assert router.top_k == 1


@pytest.mark.parametrize("kwargs,message", [
    ({"codec": "unknown"}, "codec must be one of"),
    ({"target_bits_per_dim": 0.0}, "target_bits_per_dim must be > 0"),
    ({"target_bits_per_dim": -1.0}, "target_bits_per_dim must be > 0"),
])
def test_kv_cache_rejects_invalid_codec_or_rate(kwargs, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        btm_moe.FP8PagedKVCache(**kwargs)


@pytest.mark.parametrize("codec,kind,block,bits", [
    ("a2_rope_plane", "A2", 2, 4.0),
    ("a2_8bit", "A2", 2, 8.0),
    ("e8_lattice", "E8", 8, 4.0),
])
def test_kv_lattice_fallback_quantizes_with_variance_based_scale(monkeypatch, codec, kind, block, bits):
    calibrate = Mock(side_effect=ValueError("target outside reachable range"))
    monkeypatch.setattr(lattice, "calibrate_scale_for_rate", calibrate)
    tensor = torch.tensor([[-2.0, -1.0, 1.0, 2.0, 0.0, 0.5, -0.5, 1.5]])
    stored, scale, rate, error = btm_moe.FP8PagedKVCache._encode_decode_with_codec(tensor, codec)
    fallback = max(1e-6, float(tensor.reshape(-1).std()) / 2.0 ** max(0.5, bits - 0.5))
    expected = lattice.LatticeDeltaQuantizer(kind, fallback, block=block).quantize(tensor)
    torch.testing.assert_close(stored, expected.dequantized.half().reshape(tensor.shape))
    torch.testing.assert_close(scale, torch.tensor([fallback], dtype=torch.float16))
    assert rate == pytest.approx(expected.rate_bits_per_dim)
    assert error == pytest.approx(expected.relative_mse)
    args, kwargs = calibrate.call_args
    assert args[0] == kind
    torch.testing.assert_close(args[1], tensor.reshape(-1))
    assert kwargs == {"target_bits_per_dim": bits, "block": block}
    assert calibrate.call_count == 1


@pytest.mark.parametrize("deltas,weights,mode,message", [
    ([], [], "tsv", "at least one attention delta is required"),
    ([torch.ones(2, 2)], [], "tsv", "weights and deltas must have the same length"),
    ([torch.ones(2, 2)], [1.0], "unknown", "Unsupported attention merge mode"),
])
def test_attention_delta_merge_rejects_invalid_requests(deltas, weights, mode, message):
    with pytest.raises(ValueError, match=message):
        btm_upcycle._merge_attention_projection_deltas(deltas, weights, mode)


def test_invalid_routing_mode_is_rejected_before_creating_a_sparse_block():
    with pytest.raises(ValueError, match="routing_mode must be"):
        btm_upcycle.SparseMoEMLPBlock(2, [nn.Identity()], ["A"], routing_mode="unknown")


def test_model_rejects_routing_mode_without_mutating_existing_blocks():
    block = btm_upcycle.SparseMoEMLPBlock(2, [nn.Identity()], ["A"], top_k=1)
    model = btm_upcycle.UpcycledSparseMoEModel(nn.Sequential(block), ["A"], top_k=1)
    with pytest.raises(ValueError, match="routing_mode must be"):
        model.set_routing_mode("unknown")
    assert block.routing_mode == "softmax"


def test_sigma_router_without_projection_uses_balanced_half_gates():
    block = btm_upcycle.SparseMoEMLPBlock(
        2, [nn.Identity(), nn.Identity()], ["A", "B"], top_k=2, routing_mode="sigma_moe",
    )
    block.router = nn.Identity()  # No wg or micro_gates: exercise the neutral-logit fallback.
    inputs = torch.tensor([[[1.0, 2.0], [-1.0, 3.0]]])
    torch.testing.assert_close(block(inputs), inputs)
    torch.testing.assert_close(block.expert_token_counts, torch.tensor([2, 2]))
    assert block.total_routed_assignments == 4
    assert block.ghost_skipped_assignments == 0
    assert block.last_ghost_assignment_fraction == 0.0
    assert block.last_aux_loss.item() == 0.0


@pytest.mark.parametrize("kwargs,message", [
    ({"hidden_size": 0, "loop_steps": 2}, "hidden_size >= 1 and loop_steps >= 1 required"),
    ({"hidden_size": 2, "loop_steps": 0}, "hidden_size >= 1 and loop_steps >= 1 required"),
    ({"hidden_size": 2, "loop_steps": 2, "entropy_coef": -0.1}, "entropy_coef and budget_coef must be >= 0"),
    ({"hidden_size": 2, "loop_steps": 2, "budget_coef": -0.1}, "entropy_coef and budget_coef must be >= 0"),
])
def test_depth_prior_rejects_invalid_dimensions_or_regularization(kwargs, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        btm_upcycle.LearnedDepthPrior(**kwargs)


def test_unique_weight_bytes_counts_parameter_buffer_alias_storage_once():
    module = nn.Module()
    module.register_parameter("weight", nn.Parameter(torch.tensor([1.0, 2.0])))
    module.register_buffer("alias", module.weight.detach().view(2))
    module.register_buffer("independent", module.weight.detach().clone())
    assert btm_upcycle._unique_weight_bytes(module) == 2 * 2 * torch.tensor(0.0).element_size()


@pytest.mark.parametrize("invalid", [None, "not-a-number"])
def test_upcycle_rejects_non_numeric_reputation_before_touching_model(invalid):
    model = nn.Identity()
    with pytest.raises(ValueError, match="must be a finite non-negative number") as caught:
        btm_upcycle.upcycle_branches_to_sparse_moe(
            model, {"A": {}, "B": {}}, branch_reputations={"A": invalid},
        )
    assert isinstance(caught.value.__cause__, (TypeError, ValueError))
    assert list(model.parameters()) == []


@pytest.mark.parametrize("mode", ["tsv", "iso_cts"])
@pytest.mark.parametrize("model_kind", ["bare", "wrapped"])
def test_upcycle_without_attention_adapters_preserves_base_projections(mode, model_kind, monkeypatch):
    monkeypatch.setattr(quantization, "configure_deterministic_runtime", lambda *args, **kwargs: None)
    with torch.random.fork_rng(devices=[]):
        # Seed only the CPU generator; torch.manual_seed also affects CUDA generators.
        torch.default_generator.manual_seed(17)
        base, _, _ = quantization.build_local_qwen2_backbone(
            seed=17, vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=16,
            device=torch.device("cpu"), use_4bit_nf4_if_cuda=False,
        )
        layer = base.model.layers[0]
        saved = {name: getattr(layer.self_attn, name).weight.detach().clone()
                 for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
        inputs = torch.tensor([[[0.1] * 8, [0.2] * 8]])
        with torch.no_grad():
            expected_ffn = layer.mlp(inputs).clone()
        source_model = base
        if model_kind == "wrapped":
            source_model = nn.Module()
            source_model.base_model = base
        model, report = btm_upcycle.upcycle_branches_to_sparse_moe(
            source_model, {"A": {}, "B": {}}, lora_rank_r=2, lora_alpha=4, top_k=2,
            target_device=torch.device("cpu"), in_place=True, attention_merge_mode=mode,
        )
    for name, weight in saved.items():
        torch.testing.assert_close(getattr(model.backbone.model.layers[0].self_attn, name).weight, weight)
    block, = model.get_moe_blocks()
    torch.testing.assert_close(block(inputs), expected_ffn)
    assert report.attention_layers_merged == 4
    assert report.ffn_expert_modules_created == 2
    assert report.attention_merge_mode == mode


class _QuadraticAdapter(nn.Module):
    """A real differentiable CPU objective for testing loop state and preemption."""

    def __init__(self):
        super().__init__()
        self.lora_weight = nn.Parameter(torch.tensor([1.0, 2.0]))

    def forward(self, input_ids, labels):
        return SimpleNamespace(loss=self.lora_weight.square().mean())


def test_inner_loop_thermal_sleep_and_preemption_capture_uncheckpointed_state(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(diloco_inner, "configure_deterministic_runtime", lambda *args, **kwargs: None)
    model = _QuadraticAdapter()
    breaker = GPUExecutionCircuitBreaker(
        simulated_telemetry={"temperature_c": 80.0, "power_draw_watts": 35.0,
                             "power_limit_watts": 175.0, "on_battery_power": False},
    )
    sleep = Mock()
    monkeypatch.setattr(diloco_inner, "time", SimpleNamespace(
        sleep=sleep, perf_counter=diloco_inner.time.perf_counter,
    ))
    preempt = Mock(side_effect=[False, True])
    with torch.random.fork_rng(devices=[]):
        result = diloco_inner.run_diloco_inner_loop(
            model=model, token_sequences=torch.tensor([[1, 2, 3, 4]]), worker_id="boundary-worker",
            branch_name="Code", inner_steps_H=3, checkpoint_interval_s=2, lr=0.01,
            weight_decay=0.0, should_preempt_fn=preempt, circuit_breaker=breaker,
            enable_thermal_sleep=True,
        )
    assert result.steps_completed == 1
    assert result.preempted_by_daemon is True
    assert result.thermal_throttled_steps == 2
    assert result.emergency_dump_path is None
    assert result.circuit_breaker_reason is None
    assert set(result.micro_checkpoints) == {0}
    assert sleep.call_args_list == [call(0.1), call(0.1)]
    assert preempt.call_count == 2
    torch.testing.assert_close(result.theta_init["lora_weight"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(result.theta_final["lora_weight"], model.lora_weight.detach())
    assert torch.all(result.theta_final["lora_weight"] < result.theta_init["lora_weight"])
    torch.testing.assert_close(result.raw_delta["lora_weight"],
                               result.theta_init["lora_weight"] - result.theta_final["lora_weight"])
    assert result.initial_loss == result.final_loss == 2.5


@pytest.mark.parametrize("period", [True, False, 0, -1, 1.5, "2"])
def test_outer_constructor_rejects_nonpositive_or_noninteger_restart_period(period):
    with pytest.raises(ValueError, match="momentum_restart_period must be an integer >= 1 or None"):
        DiLoCoOuterOptimizer(momentum_restart_period=period)


@pytest.mark.parametrize("period", [True, False, 0, -1, 1.5, "2"])
def test_outer_step_rejects_restart_override_before_creating_branch_state(period):
    optimizer = DiLoCoOuterOptimizer()
    with pytest.raises(ValueError, match="momentum_restart_period must be an integer >= 1 or None"):
        optimizer.step("Code", 1, {}, {}, momentum_restart_period=period)
    assert optimizer.velocities == optimizer.cpu_buckets == optimizer.cgad_optimizers == {}


def _matching_workers(theta):
    return {f"worker-{i}": {key: torch.full_like(value, 0.5) for key, value in theta.items()}
            for i in range(4)}


def test_outer_flat_bucket_rejects_consistent_new_shapes_without_mutating_old_state():
    optimizer = DiLoCoOuterOptimizer(use_cpu_flat_bucket=True, max_byzantine_f=0)
    theta = {"w": torch.tensor([2.0, 3.0])}
    optimizer.step("Code", 1, theta, _matching_workers(theta))
    bucket = optimizer.cpu_buckets["Code"]
    master = bucket.flat_master_cpu.clone()
    velocity = bucket.flat_velocity_cpu.clone()
    changed = {"w": torch.ones(3)}
    with pytest.raises(ValueError, match="branch parameters must match the existing flat bucket"):
        optimizer.step("Code", 2, changed, _matching_workers(changed))
    torch.testing.assert_close(bucket.flat_master_cpu, master)
    torch.testing.assert_close(bucket.flat_velocity_cpu, velocity)
    assert optimizer.cpu_buckets["Code"] is bucket


def test_cgad_restart_flushes_first_moment_and_preserves_variance_and_other_branch():
    optimizer = DiLoCoOuterOptimizer(
        outer_optimizer_mode="cosine_gated_adam_decay", max_byzantine_f=0,
        outer_lr_eta=0.1, weight_decay=0.0, momentum_restart_period=2,
    )
    theta = {"w": torch.tensor([2.0, 3.0])}
    first = optimizer.step("Code", 1, theta, _matching_workers(theta))
    optimizer.step("Other", 1, theta, _matching_workers(theta))
    other_m = optimizer.cgad_optimizers["Other"].m_buffers["w"].clone()
    cgad = optimizer.cgad_optimizers["Code"]
    variance_before = cgad.v_buffers["w"].clone()
    second = optimizer.step("Code", 2, first.theta_updated, _matching_workers(theta))
    torch.testing.assert_close(cgad.m_buffers["w"], torch.full((2,), 0.05))
    torch.testing.assert_close(cgad.v_buffers["w"], 0.99 * variance_before + 0.01 * 0.25)
    assert cgad.step_count["w"] == 2
    expected_step = 0.1 * (0.5 / 1.9) / (0.5 + 1e-8)
    torch.testing.assert_close(second.theta_updated["w"], first.theta_updated["w"] - expected_step)
    torch.testing.assert_close(optimizer.cgad_optimizers["Other"].m_buffers["w"], other_m)
    torch.testing.assert_close(second.velocity_updated["w"], cgad.m_buffers["w"])


@pytest.mark.parametrize("shape,message", [
    ((3,), "router_logits must be [tokens, experts]"),
    ((1, 2, 3), "router_logits must be [tokens, experts]"),
    ((0, 3), "router_logits must be non-empty"),
    ((2, 0), "router_logits must be non-empty"),
])
def test_sigma_routing_rejects_invalid_rank_or_empty_axes(shape, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        sigmoid_independent_topk_routing(torch.empty(shape), top_k=1)


@pytest.mark.parametrize("override,message", [
    ({"ffn_flops_per_layer": [], "gate_flops_per_layer": [], "local_experts_per_layer": [],
      "total_experts_per_layer": []}, "at least one layer is required"),
    ({"empirical_ghost_fraction_per_layer": []}, "empirical_ghost_fraction_per_layer must have one entry per layer"),
    ({"dense_flops": -1.0}, "dense_flops must be >= 0 and top_k >= 1"),
    ({"top_k": 0}, "dense_flops must be >= 0 and top_k >= 1"),
    ({"ffn_flops_per_layer": [-1.0]}, "FLOP counts must be >= 0"),
    ({"gate_flops_per_layer": [-1.0]}, "FLOP counts must be >= 0"),
])
def test_skip_token_flop_report_rejects_empty_or_negative_inputs(override, message):
    kwargs = dict(dense_flops=10.0, ffn_flops_per_layer=[20.0], gate_flops_per_layer=[2.0],
                  top_k=1, local_experts_per_layer=[1], total_experts_per_layer=[2])
    kwargs.update(override)
    with pytest.raises(ValueError, match=re.escape(message)):
        skip_token_flops_per_token(**kwargs)


def test_training_evaluation_detects_actual_frozen_parameter_mutation(monkeypatch, tmp_path):
    evaluate = token_eval.evaluate_tokens
    evaluations = []

    def corrupt_after_final_evaluation(model, batch):
        result = evaluate(model, batch)
        evaluations.append(result)
        if len(evaluations) == 4:
            with torch.no_grad():
                next(param for param in model.parameters() if not param.requires_grad).add_(1.0)
        return result

    monkeypatch.setattr(token_eval, "evaluate_tokens", corrupt_after_final_evaluation)
    with pytest.raises(RuntimeError, match="frozen base parameters changed during adapter training"):
        token_eval.run_token_training_evaluation(tmp_path, steps=0)
    assert len(evaluations) == 4
    assert not (tmp_path / "adapter.safetensors").exists()
    assert not (tmp_path / "report.json").exists()


def test_training_evaluation_rejects_corrupted_sender_cas_bytes(monkeypatch, tmp_path):
    read_chunk = token_eval.ContentAddressedSwarmStore.read_chunk
    reads = []

    def corrupted_read(store, cid):
        reads.append(cid)
        return read_chunk(store, cid) + b"tampered-in-transit"

    monkeypatch.setattr(token_eval.ContentAddressedSwarmStore, "read_chunk", corrupted_read)
    with pytest.raises(ValueError, match="sender CAS payload SHA-256 mismatch"):
        token_eval.run_token_training_evaluation(tmp_path, steps=0)
    assert len(reads) == 1
    assert (tmp_path / "adapter.safetensors").is_file()
    assert not (tmp_path / "report.json").exists()


def test_training_evaluation_rejects_fresh_peer_with_different_base_bytes(monkeypatch, tmp_path):
    build = token_eval.build_model
    builds = []

    def wrong_recipient(seed, base_bytes=None):
        builds.append((seed, base_bytes))
        return build(seed if base_bytes is None else seed + 1)

    monkeypatch.setattr(token_eval, "build_model", wrong_recipient)
    with pytest.raises(ValueError, match="fresh recipient base model SHA-256 mismatch"):
        token_eval.run_token_training_evaluation(tmp_path, steps=0, seed=23)
    assert len(builds) == 2
    assert builds[0] == (23, None)
    assert builds[1] == (23, (tmp_path / "base_model.safetensors").read_bytes())
    assert not (tmp_path / "report.json").exists()


def test_training_evaluation_cli_prints_selected_report_fields(monkeypatch, tmp_path, capsys):
    report = {"holdout": {"improved": False}, "transfer": {"exact_adapter_match": True}}
    evaluate = Mock(return_value=report)
    monkeypatch.setattr(token_eval, "run_token_training_evaluation", evaluate)
    monkeypatch.setattr(sys, "argv", ["token-training-eval", "--output-dir", str(tmp_path),
                                      "--steps", "0", "--seed", "7"])
    token_eval.main()
    evaluate.assert_called_once_with(tmp_path, steps=0, seed=7)
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"report": str(tmp_path / "report.json"), **report}
    assert captured.err == ""


class _TorchCudaAdapter(SimpleNamespace):
    """Replace CUDA adapter calls while delegating tensor work to real CPU PyTorch."""

    def __getattr__(self, name):
        return getattr(torch, name)


@pytest.fixture
def cpu_work_with_cuda_cleanup(monkeypatch):
    # Real PyTorch/optimizer internals must never query a driver. Only the modules'
    # cleanup adapter sees CUDA as available, a valid case even for CPU workloads.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cuda = Mock(spec=["is_available", "empty_cache"])
    cuda.is_available.return_value = True
    adapter = _TorchCudaAdapter(cuda=cuda)
    monkeypatch.setattr(diloco_inner, "torch", adapter)
    monkeypatch.setattr(quantization, "torch", adapter)
    monkeypatch.setattr(diloco_inner, "configure_deterministic_runtime", lambda *args, **kwargs: None)
    return cuda, adapter


@pytest.mark.parametrize("allocated,ceiling,fits", [
    (0.25, 0.5, True),
    (0.5, 0.5, False),
    (0.75, 0.5, False),
])
def test_footprint_report_uses_available_device_telemetry_and_strict_ceiling(
    monkeypatch, allocated, ceiling, fits,
):
    # Mock the device/measurement interface, without changing any CPU tensor's device.
    parameter_handle = SimpleNamespace(device=torch.device("cuda:2"))
    model = SimpleNamespace(parameters=Mock(side_effect=lambda: iter([parameter_handle])))
    measure = Mock(return_value={"total_gb": 0.125, "cuda_allocated_gb": allocated})
    cuda = Mock(spec=["is_available"])
    cuda.is_available.return_value = True
    monkeypatch.setattr(btm_moe, "torch", _TorchCudaAdapter(cuda=cuda))
    monkeypatch.setattr(btm_moe, "measure_moe_vram_footprint_gb", measure)

    report = btm_moe._footprint_report(model, ceiling)

    assert report == {
        "parameter_footprint_gb": 0.125,
        "vram_ceiling_gb": ceiling,
        "vram_measured_gb": allocated,
        "fits_under_vram_ceiling": fits,
        "vram_measurement": "measured: torch.cuda.memory_allocated on the model device",
    }
    measure.assert_called_once_with(model)
    model.parameters.assert_called_once_with()
    cuda.is_available.assert_called_once_with()


def test_lora_export_detaches_then_transfers_cuda_handle_and_converts_cpu_values():
    source = torch.tensor([-2.5, 7.25], dtype=torch.float64)
    parameter = Mock(spec=["device", "requires_grad", "detach"])
    parameter.device = torch.device("cuda:3")
    parameter.requires_grad = False
    parameter.detach.return_value.cpu.return_value = source
    frozen = Mock(spec=["requires_grad", "detach"])
    frozen.requires_grad = False
    model = SimpleNamespace(named_parameters=Mock(return_value=[
        ("layer.lora_A", parameter), ("frozen.weight", frozen),
    ]))

    state = quantization.extract_lora_state_dict(model)

    assert set(state) == {"layer.lora_A"}
    assert parameter.mock_calls == [call.detach(), call.detach().cpu()]
    frozen.detach.assert_not_called()
    model.named_parameters.assert_called_once_with()
    exported = state["layer.lora_A"]
    assert exported.device == torch.device("cpu")
    assert exported.dtype == torch.float32
    assert exported.requires_grad is False
    torch.testing.assert_close(exported, torch.tensor([-2.5, 7.25]))
    exported.zero_()
    torch.testing.assert_close(source, torch.tensor([-2.5, 7.25], dtype=torch.float64))


def test_inner_preemption_saves_checkpoint_before_cleaning_cuda_cache_twice(
    monkeypatch, tmp_path, cpu_work_with_cuda_cleanup,
):
    cuda, adapter = cpu_work_with_cuda_cleanup
    model = _QuadraticAdapter()
    breaker = GPUExecutionCircuitBreaker()
    evaluate = Mock(return_value=SimpleNamespace(should_preempt=False, should_throttle=False))
    monkeypatch.setattr(breaker, "evaluate", evaluate)
    events = []
    build_optimizer = diloco_inner.build_inner_optimizer

    def tracked_optimizer(**kwargs):
        optimizer = build_optimizer(**kwargs)
        step = optimizer.step

        def tracked_step(*args, **step_kwargs):
            result = step(*args, **step_kwargs)
            events.append("optimizer_step")
            return result

        monkeypatch.setattr(optimizer, "step", tracked_step)
        return optimizer

    def save_checkpoint(*args, **kwargs):
        torch.save(*args, **kwargs)
        events.append("checkpoint_saved")

    def release_cache():
        assert model.lora_weight.grad is None
        events.append("empty_cache")

    monkeypatch.setattr(diloco_inner, "build_inner_optimizer", tracked_optimizer)
    adapter.save = save_checkpoint
    cuda.empty_cache.side_effect = release_cache
    with torch.random.fork_rng(devices=[]):
        result = diloco_inner.run_diloco_inner_loop(
            model=model, token_sequences=torch.tensor([[1, 2, 3, 4]]), worker_id="cleanup-worker",
            branch_name="Code", inner_steps_H=3, checkpoint_interval_s=2, lr=0.01,
            weight_decay=0.0, should_preempt_fn=Mock(side_effect=[False, True]),
            circuit_breaker=breaker, emergency_checkpoint_dir=tmp_path,
        )

    assert events == ["optimizer_step", "checkpoint_saved", "empty_cache", "empty_cache"]
    assert cuda.mock_calls == [call.is_available(), call.empty_cache(),
                               call.is_available(), call.empty_cache()]
    assert result.steps_completed == 1
    assert result.preempted_by_daemon is True
    assert result.emergency_dump_path == tmp_path / "emergency_cleanup-worker_step1.pt"
    saved = torch.load(result.emergency_dump_path, weights_only=True)
    assert saved["step"] == 1
    assert saved["worker_id"] == "cleanup-worker"
    assert saved["branch_name"] == "Code"
    assert saved["optimizer_state"]["state"]
    torch.testing.assert_close(saved["lora_state"]["lora_weight"], result.theta_final["lora_weight"])
    torch.testing.assert_close(result.theta_final["lora_weight"], model.lora_weight.detach())
    assert torch.all(result.theta_final["lora_weight"] < result.theta_init["lora_weight"])
    torch.testing.assert_close(result.raw_delta["lora_weight"],
                               result.theta_init["lora_weight"] - result.theta_final["lora_weight"])


def test_inner_completion_and_replay_clean_cache_and_produce_identical_cpu_updates(
    monkeypatch, cpu_work_with_cuda_cleanup,
):
    cuda, _ = cpu_work_with_cuda_cleanup
    model, replay_model = _QuadraticAdapter(), _QuadraticAdapter()
    breaker = GPUExecutionCircuitBreaker()
    monkeypatch.setattr(breaker, "evaluate", Mock(return_value=SimpleNamespace(
        should_preempt=False, should_throttle=False,
    )))
    initial = {"lora_weight": model.lora_weight.detach().clone()}
    tokens = torch.tensor([[1, 2, 3, 4]])
    phases = []
    current = model

    def release_cache():
        assert current.lora_weight.grad is None
        phases.append("train" if current is model else "replay")

    cuda.empty_cache.side_effect = release_cache
    with torch.random.fork_rng(devices=[]):
        trained = diloco_inner.run_diloco_inner_loop(
            model=model, token_sequences=tokens, worker_id="cleanup-worker", branch_name="Code",
            inner_steps_H=1, checkpoint_interval_s=1, lr=0.01, weight_decay=0.0,
            circuit_breaker=breaker,
        )
        current = replay_model
        replayed = diloco_inner.replay_inner_segment(
            model=replay_model, start_lora_state=initial, token_sequences=tokens,
            start_step=0, end_step=1, lr=0.01, weight_decay=0.0,
        )

    assert phases == ["train", "replay"]
    assert cuda.mock_calls == [call.is_available(), call.empty_cache(),
                               call.is_available(), call.empty_cache()]
    assert trained.steps_completed == 1
    assert trained.preempted_by_daemon is False
    assert set(trained.micro_checkpoints) == {0, 1}
    torch.testing.assert_close(trained.theta_final["lora_weight"], torch.tensor([0.99, 1.99]))
    torch.testing.assert_close(replayed, trained.theta_final)
    torch.testing.assert_close(replay_model.lora_weight.detach(), trained.theta_final["lora_weight"])


def test_stream_loader_cleans_cuda_cache_after_loading_and_freezing_real_cpu_weights(
    monkeypatch, tmp_path, cpu_work_with_cuda_cleanup,
):
    import huggingface_hub
    from safetensors.torch import save_file
    from transformers import AutoConfig

    cuda, _ = cpu_work_with_cuda_cleanup
    cfg = quantization.Qwen2Config(
        vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=16,
        tie_word_embeddings=False,
    )
    checkpoint = tmp_path / "tiny.safetensors"
    construct = quantization.Qwen2ForCausalLM
    created = []

    def record_construction(config):
        model = construct(config)
        created.append(model)
        return model

    def release_cache():
        model, = created
        assert all(param.device.type == "cpu" and not param.requires_grad for param in model.parameters())

    configuration = Mock(return_value=cfg)
    download = Mock(return_value=str(checkpoint))
    monkeypatch.setattr(AutoConfig, "from_pretrained", configuration)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    cuda.empty_cache.side_effect = release_cache
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(19)
        reference = construct(cfg)
        saved = {key: value.detach().cpu().clone() for key, value in reference.state_dict().items()}
        save_file(saved, str(checkpoint))
        monkeypatch.setattr(quantization, "Qwen2ForCausalLM", record_construction)
        loaded, is_nf4, compressed_mb = quantization.stream_load_hf_causal_lm(
            model_id="fixture/tiny-cpu", device=torch.device("cpu"),
            compute_dtype=torch.float32, apply_nf4=False,
        )

    assert loaded is created[0]
    torch.testing.assert_close(loaded.state_dict(), saved)
    assert is_nf4 is False
    assert compressed_mb == loaded._nf4_compressed_mb == 0.0
    assert cuda.mock_calls == [call.is_available(), call.empty_cache()]
    configuration.assert_called_once_with("fixture/tiny-cpu", token=None)
    download.assert_called_once_with(repo_id="fixture/tiny-cpu", filename="model.safetensors", token=None)


def test_synthetic_cpu_training_preserves_unpaired_reference_tensor(monkeypatch):
    from hivemind.data.hf_real_trainer import train_single_domain_lora_adapter

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    reference = {
        "x.lora_A": torch.zeros(4, 8, dtype=torch.float16),
        "x.lora_B": torch.zeros(8, 4, dtype=torch.float16),
        "extra.reference_weight": torch.tensor([[0.5, -0.5], [1.0, -1.0]], dtype=torch.float16),
    }
    original = {key: value.clone() for key, value in reference.items()}
    with torch.random.fork_rng(devices=[]):
        trained, metrics = train_single_domain_lora_adapter(
            "Rust_Systems_Security", reference, lora_rank_r=4, lora_alpha=8,
            inner_steps=4, lr=0.11, topk_ratio=1.0,
        )

    assert set(trained) == set(original)
    for key, value in reference.items():
        torch.testing.assert_close(value, original[key], rtol=0.0, atol=0.0)
    assert metrics["measurement_kind"] == "synthetic_regression"
    assert metrics["inner_steps"] == metrics["requested_inner_steps"] == 4
    assert math.isfinite(metrics["initial_loss"]) and math.isfinite(metrics["final_loss"])
    assert metrics["final_loss"] < metrics["initial_loss"]
    assert torch.count_nonzero(trained["x.lora_B"]).item() > 0
    torch.testing.assert_close(trained["extra.reference_weight"], original["extra.reference_weight"],
                               rtol=0.0, atol=0.0)
