# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Edge-case, refusal and round-trip tests for the GCP Spot T4 `GCPCrowd` DiLoCo engine.

Covers the branches the end-to-end tests in `test_gcp_t4_crowd_diloco.py` do not reach: input validation,
the `/opt/hivemind/t4` path guard (including symlinks), every `T4GovernorYieldGuard` trigger through the
injectable probes and a patched `urlopen`, `weights_only` segment loading, LoRA adapter round trips, the
untied-head layouts, resume-from-segment paths, the >4M-element streaming Multi-Krum / RMS guard, and the
multi-round outer hub (velocity reuse, rejection damping). Everything runs offline on CPU.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hivemind.swarm import gcp_t4_crowd_diloco as gcp  # noqa: E402
from hivemind.swarm.crowdgpt_bridge import (  # noqa: E402
    LossRatchetGate,
    check_weight_rms_collapse_guard,
    multi_krum_select_and_aggregate,
)
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram  # noqa: E402
from hivemind.swarm.gcp_t4_crowd_diloco import (  # noqa: E402
    CrowdLoRALinear,
    GcpCrowdDiLoCoOuterHub,
    GcpT4DiLoCoInnerResult,
    T4GovernorYieldGuard,
    assert_not_protected_t4_path,
    chunked_causal_lm_cross_entropy,
    compute_gcp_t4_spot_crowd_budget,
    extract_crowd_lora_state_dict,
    forward_crowd_v1_checkpointed,
    inject_crowd_lora_adapters,
    load_crowd_lora_state_dict,
    load_resumable_diloco_segment,
    project_lora_to_flat_trunk_delta,
    render_gcp_t4_diloco_systemd_unit,
    run_gcp_t4_diloco_inner_loop,
    save_crowd_lora_safetensors,
    save_resumable_diloco_segment,
    select_t4_diloco_recipe,
    unwrap_crowd_lora_adapters,
)

BIG_DIM = 4_194_305  # one element above the in-memory threshold of the streaming Krum / RMS paths


def _tiny(seed: int = 20261112, *, tie: bool = True) -> CrowdV1WithEngram:
    torch.manual_seed(seed)
    cfg = CrowdV1RuntimeConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        max_position_embeddings=64,
        engram_buckets=64,
        tie_word_embeddings=tie,
    )
    model = CrowdV1WithEngram(cfg)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim >= 2:
                p.mul_(0.02)
    return model


def _shard(n: int = 200) -> torch.Tensor:
    return torch.tensor([(i * 7 + 3) % 64 for i in range(n)], dtype=torch.long)


def _cpu_decision(mode: str) -> gcp.T4DiLoCoRecipeDecision:
    return select_t4_diloco_recipe(requested_mode=mode, num_layers=2, window_blocks=1, lora_rank=2, prefer_cuda=False)


def _base_numel(model: CrowdV1WithEngram) -> int:
    return sum(p.numel() for p in model.base_parameters())


# ---------------------------------------------------------------------------
# Governance: path guard, budget, systemd unit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   "])
def test_path_guard_rejects_empty_paths(bad: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        assert_not_protected_t4_path(bad)


@pytest.mark.parametrize(
    "forbidden",
    [
        "/opt/hivemind/x/../t4/model.pt",
        "/opt/./hivemind//t4",
        "C:\\opt\\hivemind\\t4\\ckpt.pt",
        "/../opt/hivemind/t4/a",
    ],
)
def test_path_guard_normalises_traversal_and_windows_spellings(forbidden: str) -> None:
    with pytest.raises(PermissionError):
        assert_not_protected_t4_path(forbidden)


def test_path_guard_allows_lookalike_siblings_and_relative_paths() -> None:
    for ok in ("/opt/hivemind/t4x", "/opt/hivemind/t40/a", "relative/dir", "/opt/hivemind"):
        assert assert_not_protected_t4_path(ok) == Path(ok)


def test_path_guard_follows_symlinks_into_the_protected_dir(tmp_path: Path) -> None:
    """A symlink into /opt/hivemind/t4 is refused (the refusal used to be swallowed by an `except OSError`)."""
    link = tmp_path / "innocent_link"
    link.symlink_to("/opt/hivemind/t4")  # only the link is created; nothing is written under /opt
    with pytest.raises(PermissionError, match="resolved path"):
        assert_not_protected_t4_path(link / "checkpoint.pt")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"baseline_monthly_usd": -1.0},
        {"baseline_monthly_usd": float("nan")},
        {"budget_ceiling_usd": 0.5},
        {"scheduled_window_hours_per_day": True},
        {"extra_spot_hours_per_day": float("inf")},
        {"extra_spot_hours_per_day": "2"},
    ],
)
def test_budget_rejects_invalid_numbers(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="finite number"):
        compute_gcp_t4_spot_crowd_budget(**kwargs)


def test_budget_rejects_more_than_24_hours_and_accepts_lowercase_spot() -> None:
    with pytest.raises(ValueError, match="24.0"):
        compute_gcp_t4_spot_crowd_budget(scheduled_window_hours_per_day=20.0, extra_spot_hours_per_day=5.0)
    plan = compute_gcp_t4_spot_crowd_budget(provisioning_model=" spot ", baseline_monthly_usd=200.0)
    assert plan.provisioning_model == "SPOT"
    # Already over the ceiling: no extra hours fit, and the plan is flagged
    assert plan.max_additional_hours_per_day_within_budget == 0.0
    assert plan.within_budget is False


def test_systemd_unit_refuses_protected_work_dir_and_normalises_paths() -> None:
    with pytest.raises(PermissionError):
        render_gcp_t4_diloco_systemd_unit(work_dir="/opt/hivemind/t4/gcpcrowd")
    unit = render_gcp_t4_diloco_systemd_unit(repo_dir="C:\\repo\\HiveMind", work_dir="/data/gcpcrowd", user="svc")
    assert "User=svc" in unit
    assert "Environment=PYTHONPATH=C:/repo/HiveMind/src" in unit
    assert "--work-dir /data/gcpcrowd" in unit
    assert f"--min-free-vram-mib {gcp.DEFAULT_MIN_FREE_VRAM_MIB_T4}" in unit
    assert "NoNewPrivileges=yes" in unit and "ProtectSystem=full" in unit


# ---------------------------------------------------------------------------
# Chunked cross-entropy & LoRA adapters
# ---------------------------------------------------------------------------


def test_chunked_cross_entropy_rejects_bad_shapes_and_chunk_sizes() -> None:
    h = torch.zeros(2, 5, 4)
    w = torch.zeros(7, 4)
    y = torch.zeros(2, 5, dtype=torch.long)
    with pytest.raises(ValueError, match=r"\[B, T, D\]"):
        chunked_causal_lm_cross_entropy(h[0], w, y)
    with pytest.raises(ValueError, match="must match"):
        chunked_causal_lm_cross_entropy(h, w, y[:, :4])
    for bad in (0, True, 2.5):
        with pytest.raises(ValueError, match="token_chunk_size"):
            chunked_causal_lm_cross_entropy(h, w, y, token_chunk_size=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 2"):
        chunked_causal_lm_cross_entropy(h[:, :1], w, y[:, :1])


def test_chunked_cross_entropy_all_ignored_is_zero_and_skips_ignored_chunks() -> None:
    torch.manual_seed(3)
    h = torch.randn(1, 9, 4, requires_grad=True)
    w = torch.randn(11, 4)
    all_ignored = torch.full((1, 9), -100, dtype=torch.long)
    zero = chunked_causal_lm_cross_entropy(h, w, all_ignored)
    assert zero.item() == 0.0 and zero.dtype == torch.float32
    zero.backward()
    assert h.grad is not None and torch.count_nonzero(h.grad) == 0

    labels = torch.randint(0, 11, (1, 9))
    labels[0, 1:5] = -100  # the first chunk of 4 shifted targets is entirely ignored
    ref = F.cross_entropy(F.linear(h.detach(), w)[0, :-1], labels[0, 1:], ignore_index=-100)
    got = chunked_causal_lm_cross_entropy(h.detach(), w, labels, token_chunk_size=4, use_checkpoint=False)
    assert torch.allclose(got, ref, atol=1e-6)


def test_crowd_lora_linear_validation_and_step0_identity() -> None:
    with pytest.raises(TypeError):
        CrowdLoRALinear(nn.Conv1d(2, 2, 1))  # type: ignore[arg-type]
    base = nn.Linear(6, 5, bias=True)
    for bad_rank in (0, True, 1.5):
        with pytest.raises(ValueError, match="rank"):
            CrowdLoRALinear(base, rank=bad_rank)  # type: ignore[arg-type]
    for bad_alpha in (0.0, -1.0, float("nan"), False):
        with pytest.raises(ValueError, match="alpha"):
            CrowdLoRALinear(base, alpha=bad_alpha)  # type: ignore[arg-type]

    lora = CrowdLoRALinear(base, rank=2, alpha=4.0)
    assert lora.weight is base.weight
    assert not base.weight.requires_grad and not base.bias.requires_grad
    assert lora.scaling == 2.0
    x = torch.randn(3, 6)
    assert torch.equal(lora(x), base(x))  # B == 0 at init
    assert torch.count_nonzero(lora.delta_weight_fp32()) == 0


def test_lora_reinject_with_new_rank_rewraps_and_same_rank_keeps_adapters() -> None:
    model = _tiny()
    inject_crowd_lora_adapters(model, rank=2, alpha=4.0)
    q0 = model.blocks[0].attn.q
    assert isinstance(q0, CrowdLoRALinear) and q0.rank == 2
    inject_crowd_lora_adapters(model, rank=2, alpha=4.0)
    assert model.blocks[0].attn.q is q0  # same rank/alpha: adapter reused, not stacked
    trainable = inject_crowd_lora_adapters(model, rank=3, alpha=6.0)
    q1 = model.blocks[0].attn.q
    assert isinstance(q1, CrowdLoRALinear) and q1.rank == 3 and q1.base is q0.base
    assert not isinstance(q1.base, CrowdLoRALinear)
    # 2 blocks x 7 projections x (A, B) + final LayerNorm weight and bias
    assert len(trainable) == 2 * 7 * 2 + 2
    assert all(p.requires_grad for p in trainable)


def test_unwrap_with_merge_folds_the_low_rank_update_into_base_weights() -> None:
    model = _tiny()
    inject_crowd_lora_adapters(model, rank=2, alpha=4.0)
    with torch.no_grad():
        for blk in model.blocks:
            for sub, proj in gcp.LORA_TARGET_ATTRS:
                getattr(getattr(blk, sub), proj).lora_B.normal_(0.0, 0.1)
    w_before = model.blocks[1].mlp.w2.base.weight.detach().clone()
    expected = w_before + model.blocks[1].mlp.w2.delta_weight_fp32()
    unwrap_crowd_lora_adapters(model, merge_into_base=True)
    assert type(model.blocks[1].mlp.w2) is nn.Linear
    assert torch.allclose(model.blocks[1].mlp.w2.weight, expected, atol=1e-7)

    plain = _tiny()
    ref = plain.blocks[0].attn.q.weight.detach().clone()
    inject_crowd_lora_adapters(plain, rank=2, alpha=4.0)
    with torch.no_grad():
        plain.blocks[0].attn.q.lora_B.fill_(1.0)
    unwrap_crowd_lora_adapters(plain, merge_into_base=False)
    assert torch.equal(plain.blocks[0].attn.q.weight, ref)  # no merge: base untouched


def test_lora_state_dict_round_trip_is_exact() -> None:
    src = _tiny(seed=1)
    inject_crowd_lora_adapters(src, rank=2, alpha=4.0, seed=11)
    with torch.no_grad():
        for p in src.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.05)
    state = extract_crowd_lora_state_dict(src)
    assert set(state) >= {"ln.weight", "ln.bias", "blocks.0.attn.q.lora_A", "blocks.1.mlp.w3.lora_B"}
    assert all(v.dtype == torch.float32 and v.device.type == "cpu" for v in state.values())

    dst = _tiny(seed=1)
    inject_crowd_lora_adapters(dst, rank=2, alpha=4.0, seed=99)
    assert not torch.equal(extract_crowd_lora_state_dict(dst)["blocks.0.attn.q.lora_A"], state["blocks.0.attn.q.lora_A"])
    load_crowd_lora_state_dict(dst, state)
    again = extract_crowd_lora_state_dict(dst)
    assert again.keys() == state.keys()
    assert all(torch.equal(again[k], state[k]) for k in state)

    # A partial state only touches the keys it carries
    load_crowd_lora_state_dict(dst, {"ln.bias": torch.full_like(state["ln.bias"], 0.25)})
    assert torch.all(dst.ln.bias == 0.25)
    assert torch.equal(dst.ln.weight.detach(), state["ln.weight"])


def test_save_lora_safetensors_refuses_protected_dir_and_hash_matches_file(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    model = _tiny()
    inject_crowd_lora_adapters(model, rank=2, alpha=4.0)
    with pytest.raises(PermissionError):
        save_crowd_lora_safetensors(model, "/opt/hivemind/t4/lora.safetensors")
    dest = tmp_path / "nested" / "gcpcrowd_lora.safetensors"
    digest = save_crowd_lora_safetensors(model, dest)
    assert digest == hashlib.sha256(dest.read_bytes()).hexdigest()
    loaded = load_file(str(dest))
    ref = extract_crowd_lora_state_dict(model)
    assert loaded.keys() == ref.keys()
    assert all(loaded[k].dtype == torch.bfloat16 for k in loaded)
    assert all(torch.equal(loaded[k], ref[k].to(torch.bfloat16)) for k in ref)
    assert not list(tmp_path.joinpath("nested").glob(".tmp_lora_*"))  # temp file renamed away


def test_project_lora_on_untied_partial_model_places_deltas_at_trunk_offsets() -> None:
    model = _tiny(tie=False)
    ln_before = (model.ln.weight.detach().clone(), model.ln.bias.detach().clone())
    inject_crowd_lora_adapters(model, rank=2, alpha=4.0)
    # leave block 1's `k` and `w1` as plain Linear layers so both branches of the projection run
    model.blocks[1].attn.k = model.blocks[1].attn.k.base
    model.blocks[1].mlp.w1 = model.blocks[1].mlp.w1.base
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.01)

    flat = project_lora_to_flat_trunk_delta(model, ln_before=ln_before)
    params = model.base_parameters()
    assert flat.dtype == torch.bfloat16 and flat.numel() == sum(p.numel() for p in params)
    assert params[-1] is model.head.weight  # untied head is the last trunk segment

    offsets: dict[int, tuple[int, int]] = {}
    off = 0
    for p in params:
        offsets[id(p)] = (off, off + p.numel())
        off += p.numel()

    def seg(p: torch.Tensor) -> torch.Tensor:
        lo, hi = offsets[id(p)]
        return flat[lo:hi]

    q0 = model.blocks[0].attn.q
    assert torch.equal(seg(q0.weight), q0.delta_weight_fp32().reshape(-1).to(torch.bfloat16))
    assert torch.count_nonzero(seg(model.blocks[1].attn.k.weight)) == 0
    assert torch.count_nonzero(seg(model.blocks[1].mlp.w1.weight)) == 0
    assert torch.count_nonzero(seg(model.wte.weight)) == 0
    assert torch.count_nonzero(seg(model.head.weight)) == 0
    expected_ln = (model.ln.weight.detach() - ln_before[0]).to(torch.bfloat16)
    assert torch.equal(seg(model.ln.weight), expected_ln)

    no_ln = project_lora_to_flat_trunk_delta(model)
    assert torch.count_nonzero(no_ln[offsets[id(model.ln.weight)][0] : offsets[id(model.ln.bias)][1]]) == 0


# ---------------------------------------------------------------------------
# Recipe selector & checkpointed forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"requested_mode": "fp8_magic"}, "requested_mode"),
        ({"round_index": -1}, "round_index"),
        ({"round_index": True}, "round_index"),
        ({"num_layers": 0}, "num_layers"),
        ({"window_blocks": 0}, "window_blocks"),
    ],
)
def test_recipe_selector_rejects_invalid_arguments(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        select_t4_diloco_recipe(prefer_cuda=False, **kwargs)


def test_recipe_selector_cpu_modes_keep_their_shapes() -> None:
    bw = select_t4_diloco_recipe(requested_mode="block_window_diloco", round_index=3, num_layers=5, window_blocks=2,
                                 prefer_cuda=False)
    assert bw.device == "cpu" and bw.recipe == "block_window_diloco"
    assert bw.window_block_indices == (1, 2)  # (3 * 2) % 5 = 1
    ql = select_t4_diloco_recipe(requested_mode="qlora_graft_diloco", lora_rank=4, num_layers=3, prefer_cuda=False)
    assert ql.lora_rank == 4 and ql.window_block_indices == (0, 1, 2)
    cpu = select_t4_diloco_recipe(requested_mode="cpu_fp32", cuda_available=True, num_layers=3)
    assert cpu.recipe == "full_trunk_8bit_ckpt" and cpu.can_train_on_gpu is False and cpu.lora_rank == 0


def test_recipe_selector_queries_cuda_memory_when_not_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    mib = 1024 * 1024
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_a, **_k: (12_000 * mib, 15_360 * mib))
    dec = select_t4_diloco_recipe(cuda_available=True, compute_capability=(8, 0), gpu_name="NVIDIA A10G")
    assert dec.vram_free_mib == 12_000 and dec.vram_total_mib == 15_360
    assert dec.recipe == "full_trunk_8bit_ckpt"
    assert dec.frozen_dtype == "bfloat16"  # Ampere keeps native bf16 for frozen weights
    assert dec.projected_free_vram_mib == 12_000 - gcp.EST_VRAM_FULL_TRUNK_8BIT_CKPT_MIB
    # A stricter operator override raises the floor and pushes the choice down to the lighter recipe
    strict = select_t4_diloco_recipe(cuda_available=True, compute_capability=(7, 5), gpu_name="Tesla T4",
                                     vram_info_mib=(12_000, 15_360), min_free_vram_mib_override=6_000)
    assert strict.vram_floor_mib == 6_000 and strict.recipe == "block_window_diloco"
    assert strict.frozen_dtype == "float16"


def test_checkpointed_forward_rejects_long_sequences_and_matches_model_logits() -> None:
    model = _tiny()
    model.eval()
    too_long = torch.zeros(1, model.cfg.max_position_embeddings + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="max_position_embeddings"):
        forward_crowd_v1_checkpointed(model, too_long)
    ids = _shard(20).reshape(2, 10)
    with torch.no_grad():
        logits, loss = forward_crowd_v1_checkpointed(model, ids, labels=None)
        ref_logits, _, _ = model(ids)
    assert loss is None and logits.shape == (2, 10, model.cfg.vocab_size)
    assert torch.allclose(logits, ref_logits, atol=1e-5)


# ---------------------------------------------------------------------------
# Serving-first yield guard
# ---------------------------------------------------------------------------


class _FakeResp(io.BytesIO):
    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _url_of(req: Any) -> str:
    return req if isinstance(req, str) else req.full_url


def test_parse_nginx_stub_status_excludes_the_status_request_itself() -> None:
    body = "Active connections: 4\nserver accepts handled requests\n 10 10 20\nReading: 0 Writing: 3 Waiting: 1\n"
    assert gcp._parse_nginx_in_flight(body) == 2
    assert gcp._parse_nginx_in_flight("Reading: 0 Writing: 0 Waiting: 0") == 0
    assert gcp._parse_nginx_in_flight("Reading: 0 Writing: x Waiting: 1") == 0
    assert gcp._parse_nginx_in_flight("no status here") == 0


def test_guard_yields_on_governor_in_flight_and_unreadable_state(tmp_path: Path) -> None:
    state = tmp_path / "governor_state.json"
    state.write_text(json.dumps({"state": "serving", "in_flight": 3, "gpu_temp_c": 61}), encoding="utf-8")
    res = T4GovernorYieldGuard(governor_state_file=state, gpu_telemetry_fn=lambda: (9_000, 40)).check()
    assert res.should_yield and res.trigger == "governor_in_flight"
    assert res.in_flight_requests == 3 and res.governor_state == "serving" and res.gpu_temp_c == 61

    state.write_text("{not json", encoding="utf-8")
    lenient = T4GovernorYieldGuard(governor_state_file=state, gpu_telemetry_fn=lambda: (9_000, 40)).check()
    assert lenient.should_yield is False and lenient.governor_state == "unreadable"
    strict = T4GovernorYieldGuard(governor_state_file=state, require_governor_training_state=True).check()
    assert strict.should_yield and strict.trigger == "governor_state_unreadable"

    missing = T4GovernorYieldGuard(governor_state_file=tmp_path / "absent.json", require_governor_training_state=True)
    assert missing.check().trigger == "governor_state_standalone"


def test_guard_yields_on_nginx_traffic_from_probe_and_from_stub_status(monkeypatch: pytest.MonkeyPatch) -> None:
    res = T4GovernorYieldGuard(nginx_probe_fn=lambda: 2, gpu_telemetry_fn=lambda: (9_000, 40)).check()
    assert res.should_yield and res.trigger == "nginx_in_flight_inference" and res.in_flight_requests == 2

    seen: list[str] = []

    def fake_urlopen(req: Any, timeout: float = 0.0) -> _FakeResp:
        seen.append(_url_of(req))
        return _FakeResp(b"Active connections: 3\nReading: 0 Writing: 3 Waiting: 0\n")

    monkeypatch.setattr(gcp.urllib.request, "urlopen", fake_urlopen)
    via_http = T4GovernorYieldGuard(nginx_status_url="http://127.0.0.1:8091/nginx_status",
                                    gpu_telemetry_fn=lambda: (9_000, 40)).check()
    assert via_http.trigger == "nginx_in_flight_inference" and via_http.in_flight_requests == 2
    assert seen == ["http://127.0.0.1:8091/nginx_status"]


def test_guard_fails_closed_when_nginx_is_unreachable_during_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def broken_urlopen(*_a: Any, **_k: Any) -> _FakeResp:
        raise OSError("connection refused")

    monkeypatch.setattr(gcp.urllib.request, "urlopen", broken_urlopen)
    state = tmp_path / "governor_state.json"
    state.write_text(json.dumps({"state": "training", "in_flight": 0}), encoding="utf-8")
    strict = T4GovernorYieldGuard(governor_state_file=state, require_governor_training_state=True,
                                  nginx_status_url="http://127.0.0.1:8091/nginx_status",
                                  gpu_telemetry_fn=lambda: (9_000, 40))
    res = strict.check()
    assert res.should_yield and res.trigger == "nginx_status_unreachable_fail_closed"
    assert res.governor_state == "training"
    # Without the governor requirement an unreachable status page is not by itself a reason to yield
    lenient = T4GovernorYieldGuard(nginx_status_url="http://127.0.0.1:8091/nginx_status",
                                   gpu_telemetry_fn=lambda: (9_000, 40))
    assert lenient.check().should_yield is False


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        ({"preempted": b"TRUE", "maintenance-event": b"NONE"}, True),
        ({"preempted": b"FALSE", "maintenance-event": b"TERMINATE_ON_HOST_MAINTENANCE"}, True),
        ({"preempted": b"FALSE", "maintenance-event": b"NONE"}, False),
        ({}, False),  # metadata server unreachable: both probes raise
    ],
)
def test_guard_reads_gce_spot_metadata(monkeypatch: pytest.MonkeyPatch, answers: dict[str, bytes], expected: bool) -> None:
    headers_seen: list[str | None] = []

    def fake_urlopen(req: Any, timeout: float = 0.0) -> _FakeResp:
        headers_seen.append(req.get_header("Metadata-flavor"))
        for suffix, body in answers.items():
            if _url_of(req).endswith(suffix):
                return _FakeResp(body)
        raise OSError("no metadata server")

    monkeypatch.setattr(gcp.urllib.request, "urlopen", fake_urlopen)
    guard = T4GovernorYieldGuard(check_gce_spot_metadata=True, gpu_telemetry_fn=lambda: (9_000, 40))
    res = guard.check()
    assert res.should_yield is expected and res.spot_preempted is expected
    if expected:
        assert res.trigger == "gce_spot_preemption_30s_notice"
    assert headers_seen and all(h == "Google" for h in headers_seen)


def test_guard_caches_spot_preemption_and_rate_limits_metadata_polls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def probe() -> bool:
        calls["n"] += 1
        return calls["n"] == 1

    sticky = T4GovernorYieldGuard(spot_probe_fn=probe, spot_poll_interval_s=0.25, gpu_telemetry_fn=lambda: (9_000, 40))
    assert sticky.check().spot_preempted is True
    assert sticky.check().spot_preempted is True  # a preemption notice is never forgotten
    assert calls["n"] == 1

    quiet = {"n": 0}

    def never() -> bool:
        quiet["n"] += 1
        return False

    t = {"now": 1_000.0}
    monkeypatch.setattr(gcp.time, "monotonic", lambda: t["now"])
    limited = T4GovernorYieldGuard(spot_probe_fn=never, spot_poll_interval_s=10.0, gpu_telemetry_fn=lambda: (9_000, 40))
    limited.check()
    limited.check()
    assert quiet["n"] == 1  # second check inside the poll interval reuses the cached answer
    t["now"] += 11.0
    limited.check()
    assert quiet["n"] == 2
    # Without a probe or metadata checks there is never a spot signal
    assert T4GovernorYieldGuard(gpu_telemetry_fn=lambda: (9_000, 40))._probe_spot_preempted() is False


def test_guard_thermal_and_vram_floor_triggers() -> None:
    hot = T4GovernorYieldGuard(gpu_telemetry_fn=lambda: (12_000, 85)).check()
    assert hot.should_yield and hot.trigger == "gpu_thermal_guard_85C" and hot.vram_free_mib == 12_000
    low = T4GovernorYieldGuard(gpu_telemetry_fn=lambda: (4_299, 50)).check()
    assert low.should_yield and low.trigger == "vram_floor_breach_4299MiB_lt_4300MiB"
    ok = T4GovernorYieldGuard(gpu_telemetry_fn=lambda: (4_300, 50)).check()
    assert ok.should_yield is False and ok.trigger is None
    disabled = T4GovernorYieldGuard(min_free_vram_mib=0, gpu_telemetry_fn=lambda: (10, 50)).check()
    assert disabled.should_yield is False


def test_guard_reclaims_allocator_cache_before_declaring_a_cuda_floor_breach(monkeypatch: pytest.MonkeyPatch) -> None:
    mib = 1024 * 1024
    free = {"mib": 4_000}
    emptied = {"n": 0}

    def empty_cache() -> None:
        emptied["n"] += 1
        free["mib"] = 6_000

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_a, **_k: (free["mib"] * mib, 15_360 * mib))
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    res = T4GovernorYieldGuard().check()
    assert emptied["n"] == 1
    assert res.should_yield is False and res.vram_free_mib == 6_000

    free["mib"] = 4_000
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.__setitem__("n", emptied["n"] + 1))
    still_low = T4GovernorYieldGuard().check()
    assert still_low.should_yield and still_low.trigger.startswith("vram_floor_breach_4000MiB")


def test_guard_without_gpu_reports_unbounded_free_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    res = T4GovernorYieldGuard().check()
    assert res.should_yield is False and res.vram_free_mib == 99_999 and res.gpu_temp_c == 40


def test_evacuate_clears_optimizer_state_and_releases_cuda_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model(torch.zeros(1, 4, dtype=torch.long))[0].sum().backward()
    opt.step()
    assert len(opt.state) > 0
    emptied = {"n": 0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.__setitem__("n", emptied["n"] + 1))
    ms = gcp._evacuate_model_to_cpu(model, opt)
    assert ms >= 0.0 and len(opt.state) == 0 and emptied["n"] == 1
    assert all(p.device.type == "cpu" for p in model.parameters())


# ---------------------------------------------------------------------------
# Resumable segments
# ---------------------------------------------------------------------------


def test_segment_round_trip_is_exact_including_fp32_master_and_optimizer_state(tmp_path: Path) -> None:
    torch.manual_seed(5)
    before = torch.randn(40).to(torch.bfloat16)
    current = torch.randn(40).to(torch.bfloat16)
    fp32 = torch.randn(40)
    lora = {"blocks.0.attn.q.lora_A": torch.randn(2, 8)}
    opt_state = {
        "state": {0: {"step": torch.tensor(3.0), "exp_avg": torch.randn(4), "exp_avg_sq": torch.rand(4)}},
        "param_groups": [{"lr": 1e-3, "params": [0], "betas": (0.9, 0.999)}],
    }
    dest = save_resumable_diloco_segment(
        tmp_path / "deep" / "seg.pt",
        round_index=4,
        completed_steps=3,
        target_steps=8,
        recipe="block_window_diloco",
        step_losses=[3.5, 3.25, 3.0],
        tokens_processed=96,
        base_before_bf16=before,
        current_base_bf16=current,
        lora_state=lora,
        trigger="operator_hold_file",
        current_base_fp32=fp32,
        optimizer_state=opt_state,
    )
    assert dest.exists() and not list(dest.parent.glob(".tmp_seg_*"))
    seg = load_resumable_diloco_segment(dest)
    assert seg["round_index"] == 4 and seg["completed_steps"] == 3 and seg["target_steps"] == 8
    assert seg["recipe"] == "block_window_diloco" and seg["trigger"] == "operator_hold_file"
    assert seg["step_losses"] == [3.5, 3.25, 3.0] and seg["tokens_processed"] == 96
    assert torch.equal(seg["base_before_bf16"], before) and torch.equal(seg["current_base_bf16"], current)
    assert torch.equal(seg["current_base_fp32"], fp32)
    assert torch.equal(seg["lora_state"]["blocks.0.attn.q.lora_A"], lora["blocks.0.attn.q.lora_A"])
    st = seg["optimizer_state"]["state"][0]
    assert torch.equal(st["exp_avg"], opt_state["state"][0]["exp_avg"])
    assert seg["optimizer_state"]["param_groups"][0]["betas"] == (0.9, 0.999)  # tuples keep their type


def test_segment_loader_refuses_protected_paths_wrong_schema_and_pickled_code(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        load_resumable_diloco_segment("/opt/hivemind/t4/segment.pt")
    with pytest.raises(PermissionError):
        save_resumable_diloco_segment(
            "/opt/hivemind/t4/segment.pt", round_index=0, completed_steps=0, target_steps=1, recipe="x",
            step_losses=[], tokens_processed=0, base_before_bf16=torch.zeros(1), current_base_bf16=torch.zeros(1),
        )

    wrong = tmp_path / "wrong.pt"
    torch.save({"schema": "something.else.v1"}, wrong)
    with pytest.raises(ValueError, match="Invalid or unsupported"):
        load_resumable_diloco_segment(wrong)
    not_dict = tmp_path / "list.pt"
    torch.save([1, 2, 3], not_dict)
    with pytest.raises(ValueError, match="Invalid or unsupported"):
        load_resumable_diloco_segment(not_dict)

    evil = tmp_path / "evil.pt"
    torch.save({"schema": "hivemind.gcpcrowd.diloco_segment.v1", "payload": _Unsafe()}, evil)
    with pytest.raises(pickle.UnpicklingError):
        load_resumable_diloco_segment(evil)
    assert _Unsafe.reduced is False


class _Unsafe:
    """Pickling this records a call to a non-allowlisted global; `weights_only=True` must refuse it."""

    reduced = False

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (_mark_reduced, ())


def _mark_reduced() -> None:
    _Unsafe.reduced = True


def test_optimizer_state_to_cpu_preserves_container_types() -> None:
    state = {"a": [torch.ones(2), 3], "b": (torch.zeros(1), "x"), "c": 1.5}
    out = gcp._optimizer_state_to_cpu(state)
    assert isinstance(out["a"], list) and isinstance(out["b"], tuple)
    assert torch.equal(out["a"][0], torch.ones(2)) and out["a"][1] == 3 and out["b"][1] == "x" and out["c"] == 1.5


# ---------------------------------------------------------------------------
# Inner loop: validation, VRAM yield, untied layouts, resume paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shard", "kwargs", "match"),
    [
        (torch.zeros(4, 8, dtype=torch.long), {}, "1D tensor"),
        (torch.zeros(15, dtype=torch.long), {}, "at least 16"),
        (None, {"inner_steps": 0}, "inner_steps"),
        (None, {"inner_steps": True}, "inner_steps"),
        (None, {"batch_size": 0}, "batch_size"),
        (None, {"grad_accum_steps": 0}, "grad_accum_steps"),
    ],
)
def test_inner_loop_rejects_invalid_arguments(shard: torch.Tensor | None, kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        run_gcp_t4_diloco_inner_loop(_tiny(), _shard() if shard is None else shard, prefer_cuda=False, **kwargs)


def test_inner_loop_refuses_protected_work_dir() -> None:
    with pytest.raises(PermissionError):
        run_gcp_t4_diloco_inner_loop(_tiny(), _shard(), prefer_cuda=False, work_dir="/opt/hivemind/t4/work")


def test_inner_loop_yields_before_starting_when_vram_is_insufficient() -> None:
    model = _tiny()
    inject_crowd_lora_adapters(model, rank=2, alpha=4.0)  # leftover adapters must be removed, not trained
    weights_before = [p.detach().clone() for p in model.base_parameters()]
    decision = select_t4_diloco_recipe(cuda_available=True, vram_info_mib=(5_000, 15_360), compute_capability=(7, 5),
                                       gpu_name="Tesla T4", num_layers=2)
    assert decision.recipe == "yield_insufficient_vram" and decision.can_train_on_gpu is False
    res = run_gcp_t4_diloco_inner_loop(model, _shard(), recipe_override=decision, inner_steps=3)
    assert res.preempted is True and res.preemption_trigger == "vram_guard_pre_start"
    assert res.steps_completed == 0 and res.step_losses == [] and res.target_steps == 3
    assert res.delta_bf16.numel() == _base_numel(model) and torch.count_nonzero(res.delta_bf16) == 0
    assert not isinstance(model.blocks[0].attn.q, CrowdLoRALinear)
    assert all(torch.equal(a, b) for a, b in zip(weights_before, model.base_parameters()))


@pytest.mark.parametrize("mode", ["full_trunk_8bit_ckpt", "block_window_diloco", "qlora_graft_diloco"])
def test_inner_loop_on_untied_head_model_returns_full_trunk_delta(mode: str) -> None:
    model = _tiny(tie=False)
    head_before = model.head.weight.detach().clone()
    res = run_gcp_t4_diloco_inner_loop(model, _shard(), recipe_override=_cpu_decision(mode), inner_steps=1,
                                       seq_len=16, use_8bit_adamw=False, use_grad_checkpointing=False,
                                       pack_stream=False)
    assert res.steps_completed == 1 and len(res.step_losses) == 1
    assert res.delta_bf16.numel() == _base_numel(model)
    assert res.compressed_delta_stream == b"" and res.compressed_delta_sha256 == ""
    assert res.submit_metadata["hivemindRecipe"] == mode and res.submit_metadata["preempted"] is False
    assert res.hardware_headers["X-Client-Backend"] == "CPU"
    head_lo = _base_numel(model) - model.head.weight.numel()
    head_delta = res.delta_bf16[head_lo:]
    if mode == "full_trunk_8bit_ckpt":
        assert torch.count_nonzero(head_delta) > 0  # the untied head trains with the full trunk
    else:
        assert torch.count_nonzero(head_delta) == 0  # frozen in block-window and LoRA recipes
        assert torch.equal(model.head.weight.detach().float(), head_before)
    if mode == "qlora_graft_diloco":
        assert res.lora_state is not None and "blocks.0.attn.q.lora_A" in res.lora_state
        assert not isinstance(model.blocks[0].attn.q, CrowdLoRALinear)  # adapters removed after export


def test_qlora_preempt_at_step_zero_then_resume_from_segment(tmp_path: Path) -> None:
    decision = _cpu_decision("qlora_graft_diloco")
    model = _tiny()

    class _Always:
        def check(self) -> gcp.PreemptionCheckResult:
            return gcp.PreemptionCheckResult(True, "operator_hold_file", 0, "held", False, 0, 0)

    first = run_gcp_t4_diloco_inner_loop(model, _shard(), learner_id="t4", recipe_override=decision, inner_steps=2,
                                         seq_len=16, work_dir=tmp_path, yield_guard=_Always(),  # type: ignore[arg-type]
                                         use_8bit_adamw=False)
    assert first.preempted and first.preemption_trigger == "operator_hold_file"
    assert first.steps_completed == 0 and first.step_losses == []
    assert first.loss_after == first.loss_before  # nothing trained, nothing claimed
    assert first.yield_latency_ms is not None and first.segment_checkpoint_path is not None
    seg = load_resumable_diloco_segment(first.segment_checkpoint_path)
    assert seg["recipe"] == "qlora_graft_diloco" and seg["completed_steps"] == 0
    assert "current_base_fp32" not in seg  # the LoRA recipe never touches the trunk
    assert set(seg["lora_state"]) >= {"blocks.0.attn.q.lora_A", "ln.weight"}

    clear = T4GovernorYieldGuard(gpu_telemetry_fn=lambda: (9_000, 40))
    second = run_gcp_t4_diloco_inner_loop(model, _shard(), learner_id="t4", recipe_override=decision, inner_steps=2,
                                          seq_len=16, work_dir=tmp_path, yield_guard=clear, use_8bit_adamw=False)
    assert second.preempted is False and second.steps_completed == 2 and len(second.step_losses) == 2
    assert not first.segment_checkpoint_path.exists()  # consumed and removed on completion
    assert second.segment_checkpoint_path is None


def test_resume_survives_an_optimizer_state_from_a_different_layout(tmp_path: Path) -> None:
    model = _tiny()
    decision = _cpu_decision("full_trunk_8bit_ckpt")
    base = model.export_flat_base_bf16()
    seg_path = tmp_path / "segment_t4_r0002.pt"
    save_resumable_diloco_segment(
        seg_path, round_index=2, completed_steps=1, target_steps=3, recipe=decision.recipe, step_losses=[4.0],
        tokens_processed=16, base_before_bf16=base, current_base_bf16=base,
        current_base_fp32=torch.cat([p.detach().float().reshape(-1) for p in model.base_parameters()]),
        optimizer_state={"state": {}, "param_groups": [{"lr": 1e-3, "params": [0]}]},
    )
    res = run_gcp_t4_diloco_inner_loop(model, _shard(), learner_id="t4", round_index=2, recipe_override=decision,
                                       inner_steps=3, seq_len=16, work_dir=tmp_path, use_8bit_adamw=False)
    assert res.preempted is False and res.steps_completed == 3
    assert len(res.step_losses) == 3 and res.step_losses[0] == 4.0  # resumed history kept, two new steps run
    assert not seg_path.exists()


def test_resume_ignores_a_segment_for_another_recipe(tmp_path: Path) -> None:
    model = _tiny()
    base = model.export_flat_base_bf16()
    seg_path = tmp_path / "segment_t4_r0000.pt"
    save_resumable_diloco_segment(seg_path, round_index=0, completed_steps=5, target_steps=6, recipe="block_window_diloco",
                                  step_losses=[9.0] * 5, tokens_processed=80, base_before_bf16=base,
                                  current_base_bf16=base)
    res = run_gcp_t4_diloco_inner_loop(model, _shard(), learner_id="t4", recipe_override=_cpu_decision("full_trunk_8bit_ckpt"),
                                       inner_steps=1, seq_len=16, work_dir=tmp_path, use_8bit_adamw=False)
    assert res.steps_completed == 1 and len(res.step_losses) == 1 and 9.0 not in res.step_losses


def test_inner_loop_follows_crowdgpt_crosscheck_cohort_targets() -> None:
    meta = {
        "taskId": "cohort-task-7",
        "globalStep": 400,
        "crossCheck": {"enabled": True, "targetSeed": 77, "targetLocalSteps": 2, "targetBatchSize": 2,
                       "targetAccumSteps": 1, "targetAutocast": "fp32", "targetOptimizer": "adamw"},
    }
    res = run_gcp_t4_diloco_inner_loop(_tiny(), _shard(), recipe_override=_cpu_decision("block_window_diloco"),
                                       inner_steps=9, batch_size=1, seq_len=16, task_metadata=meta,
                                       use_8bit_adamw=False)
    assert res.target_steps == 2 and res.steps_completed == 2
    sm = res.submit_metadata
    assert sm["taskId"] == "cohort-task-7" and sm["crossCheckUsed"] is True and sm["actualSeed"] == 77
    assert sm["actualSteps"] == 2 and sm["actualBatchSize"] == 2
    assert res.tokens_processed == 2 * 2 * 16
    assert res.hardware_headers["X-Client-Batch-Size"] == "2"
    assert res.hardware_headers["X-Client-Optim"] == "adamw"


# ---------------------------------------------------------------------------
# Streaming (>4M element) Multi-Krum and RMS guard
# ---------------------------------------------------------------------------


def _big_vectors(n_honest: int, seed: int = 9) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    center = torch.randn(BIG_DIM, generator=g) * 1e-2
    return [(center + torch.randn(BIG_DIM, generator=g) * 1e-4).to(torch.bfloat16) for _ in range(n_honest)]


def test_streaming_multi_krum_rejects_a_large_outlier_and_caps_token_weights() -> None:
    honest = _big_vectors(4)
    outlier = torch.full((BIG_DIM,), 5.0, dtype=torch.bfloat16)
    deltas = honest[:2] + [outlier] + honest[2:]
    res = gcp._streaming_multi_krum_bf16(deltas, f_byzantine=1, token_weights=[100, 100, 100, 100, 1000],
                                         chunk_elements=1_048_576)
    assert res.rejected_indices == [2] and res.selected_indices == [0, 1, 3, 4]
    assert res.byzantine_filter_active is True
    assert res.aggregated_delta_bf16.dtype == torch.bfloat16 and res.aggregated_delta_bf16.numel() == BIG_DIM
    # median of selected weights is 100 -> the 1000-token claim is capped at 125
    assert res.effective_tokens == 100 + 100 + 100 + 125
    assert res.krum_scores[2] > max(res.krum_scores[i] for i in res.selected_indices)
    # the aggregate stays inside the honest cluster
    assert float(res.aggregated_delta_bf16.float().abs().max()) < 0.1

    # The in-memory path agrees on the same geometry at small scale
    small = [d[:4096] for d in deltas]
    ref = multi_krum_select_and_aggregate(small, f_byzantine=1, token_weights=[100, 100, 100, 100, 1000])
    assert ref.rejected_indices == res.rejected_indices


def test_streaming_multi_krum_single_learner_and_non_finite_input() -> None:
    (only,) = _big_vectors(1, seed=2)
    res = gcp._streaming_multi_krum_bf16([only], f_byzantine=1, token_weights=[0])
    assert res.selected_indices == [0] and res.byzantine_filter_active is False
    assert res.effective_tokens == 1 and torch.equal(res.aggregated_delta_bf16, only)
    assert res.aggregated_delta_bf16.data_ptr() != only.data_ptr()  # a copy, not an alias

    a, b, c = _big_vectors(3, seed=4)
    b = b.clone()
    b[123] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        gcp._streaming_multi_krum_bf16([a, b, c], f_byzantine=0, token_weights=[1, 1, 1])


def test_streaming_rms_guard_matches_the_in_memory_guard_on_every_outcome() -> None:
    g = torch.Generator().manual_seed(17)
    old = (torch.randn(BIG_DIM, generator=g) * 0.04).to(torch.bfloat16)
    cases = {
        "ACCEPTED_RMS_HEALTHY": (old.float() * 1.02).to(torch.bfloat16),
        "REJECTED_ZERO_PARTICIPANT_WEIGHT_HALVING": (old.float() * 0.5).to(torch.bfloat16),
        "REJECTED_RMS_COLLAPSE": (old.float() * 0.3 + torch.randn(BIG_DIM, generator=g) * 0.01).to(torch.bfloat16),
        "REJECTED_RMS_EXPLOSION": (old.float() * 2.0).to(torch.bfloat16),
    }
    for expected_reason, new in cases.items():
        got = gcp._streaming_weight_rms_guard(old, new)
        ref = check_weight_rms_collapse_guard(old, new)
        assert got.reason.startswith(expected_reason), got.reason
        assert ref.reason.startswith(expected_reason), ref.reason
        assert got.accepted is ref.accepted is (expected_reason == "ACCEPTED_RMS_HEALTHY")
        assert got.rms_ratio == pytest.approx(ref.rms_ratio, rel=1e-9)
        assert got.rms_old == pytest.approx(ref.rms_old, rel=1e-9)
        assert got.halving_residual_rms == pytest.approx(ref.halving_residual_rms, rel=1e-6, abs=1e-12)

    bad = old.clone()
    bad[-1] = float("inf")
    nf = gcp._streaming_weight_rms_guard(old, bad)
    assert nf.accepted is False and nf.reason == "REJECTED_NON_FINITE_WEIGHTS"


# ---------------------------------------------------------------------------
# Outer hub: validation, multi-round Nesterov, rejection damping, receipts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "kwargs", "exc", "match"),
    [
        ((torch.zeros(0),), {}, ValueError, "non-empty"),
        (([1.0, 2.0],), {}, ValueError, "non-empty"),
        ((torch.ones(4),), {"outer_lr": 0.0}, ValueError, "outer_lr"),
        ((torch.ones(4),), {"outer_lr": True}, ValueError, "outer_lr"),
        ((torch.ones(4),), {"outer_lr": float("nan")}, ValueError, "outer_lr"),
        ((torch.ones(4),), {"outer_momentum": 1.0}, ValueError, "outer_momentum"),
        ((torch.ones(4),), {"outer_momentum": -0.1}, ValueError, "outer_momentum"),
        ((torch.ones(4),), {"work_dir": "/opt/hivemind/t4/hub"}, PermissionError, "protected"),
    ],
)
def test_outer_hub_constructor_refusals(args: tuple[Any, ...], kwargs: dict[str, Any], exc: type, match: str) -> None:
    with pytest.raises(exc, match=match):
        GcpCrowdDiLoCoOuterHub(*args, **kwargs)


def _result(model: CrowdV1WithEngram, learner: str, delta: torch.Tensor, steps: int = 1) -> GcpT4DiLoCoInnerResult:
    return GcpT4DiLoCoInnerResult(
        learner_id=learner, hiveminder_handle=learner, device_suffix="cpu", round_index=0,
        recipe_decision=_cpu_decision("full_trunk_8bit_ckpt"), steps_completed=steps, target_steps=1,
        tokens_processed=64, loss_before=1.0, loss_after=1.0, step_losses=[1.0], delta_bf16=delta,
    )


def test_outer_hub_step_refusals() -> None:
    model = _tiny()
    hub = GcpCrowdDiLoCoOuterHub(model.export_flat_base_bf16())
    held = _shard(80)
    with pytest.raises(ValueError, match="At least one"):
        hub.step([], eval_model=model, held_out_tokens=held, prefer_cuda_eval=False)
    with pytest.raises(ValueError, match="held_out_tokens"):
        hub.step([("a", torch.zeros(4), 1)], eval_model=model, held_out_tokens=held[:10], eval_window_len=64,
                 prefer_cuda_eval=False)
    with pytest.raises(ValueError, match="held_out_tokens"):
        hub.step([("a", torch.zeros(4), 1)], eval_model=model, held_out_tokens=held.reshape(2, 40),
                 prefer_cuda_eval=False)
    zero_step = _result(model, "idle", torch.zeros(_base_numel(model), dtype=torch.bfloat16), steps=0)
    with pytest.raises(ValueError, match="No non-empty learner deltas"):
        hub.step([zero_step], eval_model=model, held_out_tokens=held, eval_window_len=16, prefer_cuda_eval=False)


def test_outer_hub_reuses_velocity_and_damps_it_on_a_rejected_round(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    model = _tiny()
    dim = _base_numel(model)
    hub = GcpCrowdDiLoCoOuterHub(model.export_flat_base_bf16(), outer_lr=0.5, outer_momentum=0.9, f_byzantine=0,
                                 max_regression_tolerance=1e9, work_dir=tmp_path / "hub")
    held = _shard(80)
    g = torch.Generator().manual_seed(23)

    def small_delta() -> torch.Tensor:
        return (torch.randn(dim, generator=g) * 1e-4).to(torch.bfloat16)

    r0 = hub.step([("t4", small_delta(), 64), ("rtx2070", small_delta(), 32)], eval_model=model, held_out_tokens=held,
                  eval_window_len=16, eval_num_windows=2, prefer_cuda_eval=False, pack_upstream_stream=False)
    assert r0.accepted and r0.round_index == 0 and hub.velocity_bf16 is not None
    assert r0.checkpoint_path is not None and r0.checkpoint_path.exists()
    assert torch.equal(load_file(str(r0.checkpoint_path))["weights"], hub.theta_bf16)
    assert r0.checkpoint_sha256 == hub.current_checkpoint_sha256()
    assert len(r0.upstream_crowdgpt_stream_sha256) == 64 and r0.loss_check_receipt is None

    v0 = hub.velocity_bf16.clone()
    theta0 = hub.theta_bf16.clone()
    items = [_result(model, "t4", small_delta()), _result(model, "peer", small_delta())]
    r1 = hub.step(items, eval_model=model, held_out_tokens=held, eval_window_len=16, eval_num_windows=2,
                  prefer_cuda_eval=False, save_safetensors=False, clear_input_deltas=True)
    assert r1.accepted and r1.round_index == 1 and r1.checkpoint_path is None
    assert all(it.delta_bf16.numel() == 0 for it in items)  # inputs released for RAM
    # Nesterov velocity carries momentum: v1 = mu * v0 + agg (checked in fp32 within bf16 rounding)
    agg = r1.krum_result.aggregated_delta_bf16.float()
    assert torch.allclose(hub.velocity_bf16.float(), 0.9 * v0.float() + agg, atol=1e-6, rtol=1e-2)
    assert not torch.equal(hub.theta_bf16, theta0)

    v1 = hub.velocity_bf16.clone()
    sha1 = hub.current_checkpoint_sha256()
    hub.ratchet = LossRatchetGate(max_regression_tolerance=-1e9)  # force the held-out gate to refuse
    r2 = hub.step([("t4", small_delta(), 64)], eval_model=model, held_out_tokens=held, eval_window_len=16,
                  eval_num_windows=2, prefer_cuda_eval=False)
    assert r2.accepted is False and r2.ratchet_decision.accepted is False and r2.rms_guard.accepted is True
    assert hub.current_checkpoint_sha256() == sha1 and r2.checkpoint_sha256 == sha1
    assert torch.equal(hub.velocity_bf16, (0.5 * v1.float()).to(torch.bfloat16))
    assert torch.equal(model.export_flat_base_bf16(), hub.theta_bf16)  # eval model restored to the held head

    receipts = sorted((tmp_path / "hub").glob("gcpcrowd_outer_round_*.json"))
    assert [p.name for p in receipts] == [f"gcpcrowd_outer_round_{i:04d}.json" for i in range(3)]
    last = json.loads(receipts[-1].read_text(encoding="utf-8"))
    assert last["accepted"] is False and last["checkpoint_sha256"] == sha1
    assert len(hub.history) == 3 and hub.round_index == 3
