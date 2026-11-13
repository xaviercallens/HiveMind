# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Refusal, round-trip and offline-sync tests for the Crowd-v1 live runtime.

Complements `test_crowdgpt_live_runtime.py`: config validation, the CUDA placement guard with an
injected `mem_get_info`, untied heads and disabled Engram, flat-tensor and `safetensors` loading refusals,
deterministic generation edges (empty prompt, EOS, top-p collapse), the Hugging Face sync client against a
fake `urlopen` (no network), and the rejected-round rollback of `execute_multi_gpu_crowdgpt_round`.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest
from safetensors.torch import save_file
import torch

from hivemind.swarm import crowdgpt_live_runtime as rt
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdGPTHuggingFaceSync,
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
    execute_multi_gpu_crowdgpt_round,
    load_crowd_v1_safetensors_into_model,
    select_safe_execution_device,
)


def _cfg(**over: Any) -> CrowdV1RuntimeConfig:
    base: dict[str, Any] = dict(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        max_position_embeddings=32,
        engram_buckets=31,
    )
    base.update(over)
    return CrowdV1RuntimeConfig(**base)


def _model(seed: int = 20261112, **over: Any) -> CrowdV1WithEngram:
    torch.manual_seed(seed)
    model = CrowdV1WithEngram(_cfg(**over))
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim >= 2:
                p.mul_(0.05)
    return model


# ---------------------------------------------------------------------------
# Config & device guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"vocab_size": 4}, "vocab_size must be an integer >= 8"),
        ({"hidden_size": True}, "hidden_size must be an integer"),
        ({"engram_buckets": -1}, "engram_buckets must be an integer >= 0"),
        ({"head_dim": 7}, "head_dim must be even"),
        ({"rope_theta": 1.0}, "rope_theta"),
        ({"rope_theta": float("inf")}, "rope_theta"),
        ({"rope_theta": True}, "rope_theta"),
        ({"layer_norm_eps": 0.0}, "layer_norm_eps"),
        ({"layer_norm_eps": float("nan")}, "layer_norm_eps"),
        ({"tie_word_embeddings": 1}, "tie_word_embeddings"),
    ],
)
def test_config_rejects_invalid_numerics(over: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _cfg(**over)


def test_device_guard_rejects_negative_estimate_and_queries_cuda_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError, match="estimated_base_vram_mib"):
            select_safe_execution_device(estimated_base_vram_mib=bad, cuda_available=False)  # type: ignore[arg-type]

    mib = 1024 * 1024
    free = {"mib": 5_000}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_a, **_k: (free["mib"] * mib, 15_360 * mib))
    t4_tight = select_safe_execution_device(cuda_available=True, compute_capability=(7, 5), gpu_name="Tesla T4")
    assert t4_tight.device == "cpu" and t4_tight.vram_free_mib == 5_000 and t4_tight.vram_safety_floor_mib == 4_300
    assert "CPU_FALLBACK_VRAM_GUARD" in t4_tight.reason
    free["mib"] = 9_000
    t4_ok = select_safe_execution_device(cuda_available=True, compute_capability=(7, 5), gpu_name="Tesla T4")
    assert t4_ok.device == "cuda" and t4_ok.dtype_name == "float16" and t4_ok.estimated_model_vram_mib == 1350


# ---------------------------------------------------------------------------
# Model structure, flat tensors and placement
# ---------------------------------------------------------------------------


def test_disabled_engram_contributes_exact_zeros() -> None:
    model = _model(engram_buckets=0)
    assert model.engram.enabled is False
    ids = torch.tensor([[1, 2, 3]])
    out = model.engram(ids)
    assert out.shape == (1, 3, 32) and torch.count_nonzero(out) == 0
    with torch.no_grad():
        logits, loss, _ = model(ids, labels=ids)
    assert logits.shape == (1, 3, 64) and loss is not None and torch.isfinite(loss)


def test_untied_head_is_a_separate_trunk_parameter_and_round_trips() -> None:
    model = _model(tie_word_embeddings=False)
    params = model.base_parameters()
    assert params[-1] is model.head.weight and model.head.weight is not model.wte.weight
    flat = model.export_flat_base_bf16()
    assert flat.numel() == sum(p.numel() for p in params)
    other = _model(seed=1, tie_word_embeddings=False)
    other.load_flat_tensors(flat)
    assert torch.equal(other.export_flat_base_bf16(), flat)


def test_load_flat_tensors_refuses_wrong_sizes_and_loads_engram_exactly() -> None:
    model = _model()
    flat = model.export_flat_base_bf16()
    with pytest.raises(ValueError, match="base elements"):
        model.load_flat_tensors(flat[:-1])
    eng = torch.arange(model.engram.table.weight.numel(), dtype=torch.float32) / 1000.0
    with pytest.raises(ValueError, match="engram elements"):
        model.load_flat_tensors(flat, flat_engram=eng[:-1])
    model.load_flat_tensors(flat, flat_engram=eng.reshape(31, 32))
    assert torch.equal(model.engram.table.weight.detach().reshape(-1), eng)
    # Engram tensors are ignored on a model without an Engram table
    no_eng = _model(engram_buckets=0)
    before = no_eng.engram.table.weight.detach().clone()
    no_eng.load_flat_tensors(no_eng.export_flat_base_bf16(), flat_engram=torch.ones(5))
    assert torch.equal(no_eng.engram.table.weight, before)


def test_place_for_inference_casts_untied_head_and_can_keep_engram_on_device() -> None:
    model = _model(tie_word_embeddings=False)
    model.place_for_inference(device="cpu", dtype=torch.float64, offload_engram_to_cpu=False)
    assert model.head.weight.dtype == torch.float64 and model.blocks[0].attn.q.weight.dtype == torch.float64
    assert model.engram.table.weight.dtype == torch.float64
    tied = _model()
    tied.place_for_inference(device="cpu", dtype=torch.float64)
    assert tied.head.weight is tied.wte.weight and tied.engram.table.weight.dtype == torch.bfloat16


def test_forward_rejects_sequences_beyond_the_context_including_the_kv_cache() -> None:
    model = _model()
    with torch.no_grad():
        with pytest.raises(ValueError, match="max_position_embeddings"):
            model(torch.zeros(1, 33, dtype=torch.long))
        _, _, kvs = model(torch.zeros(1, 30, dtype=torch.long), use_cache=True)
        assert kvs is not None and len(kvs) == 2
        with pytest.raises(ValueError, match="Sequence length 33"):
            model(torch.zeros(1, 3, dtype=torch.long), past_kvs=kvs, use_cache=True)


# ---------------------------------------------------------------------------
# Generation edges
# ---------------------------------------------------------------------------


def test_generate_handles_empty_prompt_and_stops_on_eos() -> None:
    model = _model()
    model.eval()
    out = model.generate_ids([], max_new_tokens=3, temperature=0.0)
    assert 1 <= len(out) <= 3 and all(0 <= t < 64 for t in out)
    stop = model.generate_ids([5, 6], max_new_tokens=10, temperature=0.0, eos_token_ids=range(64))
    assert len(stop) == 1  # every token is EOS, so generation ends after the first one


def test_tiny_top_p_collapses_sampling_to_greedy_and_full_top_p_samples_within_vocab() -> None:
    model = _model()
    model.eval()
    greedy = model.generate_ids([3, 4, 5], max_new_tokens=6, temperature=0.0, eos_token_ids=())
    torch.manual_seed(0)
    nucleus = model.generate_ids([3, 4, 5], max_new_tokens=6, temperature=0.7, top_p=1e-9, eos_token_ids=())
    assert nucleus == greedy
    torch.manual_seed(0)
    a = model.generate_ids([3, 4, 5], max_new_tokens=6, temperature=1.0, top_p=1.0, eos_token_ids=())
    torch.manual_seed(0)
    b = model.generate_ids([3, 4, 5], max_new_tokens=6, temperature=1.0, top_p=1.0, eos_token_ids=())
    assert a == b and len(a) == 6 and all(0 <= t < 64 for t in a)  # seeded multinomial is reproducible


# ---------------------------------------------------------------------------
# safetensors loading
# ---------------------------------------------------------------------------


def _state_of(model: CrowdV1WithEngram) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone().contiguous() for k, v in model.named_parameters()}


def test_safetensors_loader_skips_tied_head_and_hashes_the_file(tmp_path: Path) -> None:
    src = _model(seed=3)
    state = _state_of(src)
    state["head.weight"] = state["wte.weight"].clone()  # exporters often write the tied head explicitly
    path = tmp_path / "model.safetensors"
    save_file(state, str(path))
    dst = _model(seed=4)
    digest = load_crowd_v1_safetensors_into_model(dst, path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert dst.head.weight is dst.wte.weight
    assert all(torch.equal(dict(dst.named_parameters())[k], v) for k, v in _state_of(src).items())


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda s: s.__setitem__("bogus.weight", torch.zeros(2)), "unexpected=\\['bogus.weight'\\]"),
        (lambda s: s.pop("ln.bias"), "missing=\\['ln.bias'\\]"),
        (lambda s: s.__setitem__("ln.bias", torch.zeros(7)), "Shape mismatch for ln.bias"),
    ],
)
def test_safetensors_loader_refuses_incompatible_files(tmp_path: Path, mutate: Any, match: str) -> None:
    state = _state_of(_model(seed=5))
    mutate(state)
    path = tmp_path / "bad.safetensors"
    save_file(state, str(path))
    with pytest.raises(ValueError, match=match):
        load_crowd_v1_safetensors_into_model(_model(seed=6), path)


# ---------------------------------------------------------------------------
# Hugging Face sync client (offline, fake urlopen)
# ---------------------------------------------------------------------------


class _Resp(io.BytesIO):
    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _FakeHub:
    """URL -> bytes table standing in for huggingface.co; unknown URLs raise like an offline host."""

    def __init__(self, routes: dict[str, bytes]) -> None:
        self.routes = dict(routes)
        self.calls: list[str] = []
        self.user_agents: list[str | None] = []

    def __call__(self, req: Any, timeout: float = 0.0) -> _Resp:
        url = req.full_url
        self.calls.append(url)
        self.user_agents.append(req.get_header("User-agent"))
        if url not in self.routes:
            raise OSError(f"offline: {url}")
        return _Resp(self.routes[url])


REV = "0123456789abcdef0123"
REV_URL = f"https://huggingface.co/{rt.CROWDGPT_HF_REPO}/resolve/{REV}/model.safetensors"


def _sync(tmp_path: Path) -> CrowdGPTHuggingFaceSync:
    return CrowdGPTHuggingFaceSync(tmp_path / "cache", refresh_interval_s=1.0)


def test_hf_sync_downloads_revision_pinned_weights_and_prunes_old_ones(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hub = _FakeHub({
        rt.CROWDGPT_STATUS_URL: json.dumps({"hfRevision": REV, "globalStep": 512}).encode(),
        rt.CROWDGPT_TOKENIZER_URL: b'{"model": "bpe"}',
        REV_URL: b"W" * 1000,
    })
    monkeypatch.setattr(rt.urllib.request, "urlopen", hub)
    sync = _sync(tmp_path)
    assert sync.refresh_interval_s == 30.0  # polling is clamped to protect the Hub
    (tmp_path / "cache").mkdir()
    stale = tmp_path / "cache" / "model_step_100.safetensors"
    stale.write_bytes(b"old")
    locked = tmp_path / "cache" / "model_locked.safetensors"
    locked.mkdir()  # cannot be unlinked: pruning must not fail the sync

    info = sync.ensure_latest_files()
    assert info["hf_revision"] == REV and info["global_step"] == 512
    assert info["weights_path"] == tmp_path / "cache" / f"model_{REV[:12]}.safetensors"
    assert info["weights_path"].read_bytes() == b"W" * 1000
    assert info["tokenizer_path"].read_bytes() == b'{"model": "bpe"}'
    assert not stale.exists() and locked.is_dir()
    assert not list((tmp_path / "cache").glob(".tmp_crowd_*"))
    assert all(ua == "hivemind-crowd-v1-sync/1.0" for ua in hub.user_agents)

    n_calls = len(hub.calls)
    again = sync.ensure_latest_files()  # inside the refresh interval: cached status, files already present
    assert again["weights_path"] == info["weights_path"] and len(hub.calls) == n_calls
    sync.ensure_latest_files(force_check=True)
    assert hub.calls[n_calls:] == [rt.CROWDGPT_STATUS_URL]


def test_hf_sync_falls_back_to_main_weights_when_revision_url_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hub = _FakeHub({
        rt.CROWDGPT_STATUS_URL: json.dumps({"hfRevision": REV, "globalStep": 7}).encode(),
        rt.CROWDGPT_TOKENIZER_URL: b"{}",
        rt.CROWDGPT_MODEL_SAFETENSORS_URL: b"main-weights",
    })
    monkeypatch.setattr(rt.urllib.request, "urlopen", hub)
    info = _sync(tmp_path).ensure_latest_files()
    assert info["weights_path"].read_bytes() == b"main-weights"
    assert hub.calls.index(REV_URL) < hub.calls.index(rt.CROWDGPT_MODEL_SAFETENSORS_URL)


def test_hf_sync_uses_step_tag_on_main_and_tolerates_non_dict_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hub = _FakeHub({
        rt.CROWDGPT_STATUS_URL: json.dumps({"hfRevision": "main", "globalStep": 348}).encode(),
        rt.CROWDGPT_TOKENIZER_URL: b"{}",
        rt.CROWDGPT_MODEL_SAFETENSORS_URL: b"m",
    })
    monkeypatch.setattr(rt.urllib.request, "urlopen", hub)
    sync = _sync(tmp_path)
    info = sync.ensure_latest_files()
    assert info["weights_path"].name == "model_step_348.safetensors"
    assert REV_URL not in hub.calls

    hub.routes[rt.CROWDGPT_STATUS_URL] = b"[1, 2, 3]"
    assert sync.fetch_remote_status() == {} and sync.last_status == {}
    hub.routes.pop(rt.CROWDGPT_STATUS_URL)
    with pytest.raises(OSError, match="offline"):
        sync.fetch_remote_status()


# ---------------------------------------------------------------------------
# Verified round
# ---------------------------------------------------------------------------


def test_round_refuses_short_or_2d_shards() -> None:
    for bad in (torch.zeros(31, dtype=torch.long), torch.zeros(4, 16, dtype=torch.long)):
        with pytest.raises(ValueError, match="at least 32"):
            execute_multi_gpu_crowdgpt_round(_model(), bad, hiveminder_handle="h", prefer_cuda=False)


def test_round_rejected_by_the_ratchet_restores_the_previous_head() -> None:
    model = _model()
    before = model.export_flat_base_bf16()
    shard = torch.tensor([(i * 3 + 1) % 64 for i in range(96)], dtype=torch.long)
    out = execute_multi_gpu_crowdgpt_round(model, shard, hiveminder_handle="alice", prefer_cuda=False,
                                           learning_rate=1e-2, max_regression_tolerance=-1e9)
    assert out["accepted"] is False and out["ratchet_decision"]["accepted"] is False
    assert out["ratchet_decision"]["reason"].startswith("HELD_PREVIOUS_HEAD")
    assert out["delta_l2"] > 0.0  # training did move the weights before the gate refused them
    assert torch.equal(model.export_flat_base_bf16(), before)
    assert out["device_decision"]["device"] == "cpu" and out["hardware_headers"]["X-Client-Backend"] == "CPU"
    assert len(out["compressed_delta_sha256"]) == 64 and out["compressed_delta_bytes"] > 0
