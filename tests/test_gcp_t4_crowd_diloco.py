# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Comprehensive unit & integration tests for the GCP Spot Tesla T4 `GCPCrowd` DiLoCo Engine."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from hivemind.router.opentrainer import (  # noqa: E402
    build_crowdgpt_trainer_producer_plan,
    is_allowed_opentrainer_campaign_id,
)
from hivemind.swarm.crowdgpt_bridge import unpack_crowdgpt_delta_stream  # noqa: E402
from hivemind.swarm.crowdgpt_live_runtime import (  # noqa: E402
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
)
from hivemind.swarm.crowdgpt_loss_check import LossCheckReceipt  # noqa: E402
from hivemind.swarm.gcp_t4_crowd_diloco import (  # noqa: E402
    GCPCROWD_CAMPAIGN_ID,
    GcpCrowdDiLoCoOuterHub,
    T4GovernorYieldGuard,
    assert_not_protected_t4_path,
    chunked_causal_lm_cross_entropy,
    compute_gcp_t4_spot_crowd_budget,
    run_gcp_t4_diloco_inner_loop,
    save_crowd_lora_safetensors,
    select_t4_diloco_recipe,
    inject_crowd_lora_adapters,
)


def _tiny_crowd_model(seed: int = 20261110) -> CrowdV1WithEngram:
    torch.manual_seed(seed)
    cfg = CrowdV1RuntimeConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        max_position_embeddings=128,
        engram_buckets=128,
    )
    model = CrowdV1WithEngram(cfg)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim >= 2:
                p.mul_(0.02)
    return model


def test_protected_t4_path_hard_rejection(tmp_path: Path) -> None:
    for forbidden in (
        "/opt/hivemind/t4",
        "/opt/hivemind/t4/",
        "/opt/hivemind/t4/scripts/t4_governor.py",
        "/opt/hivemind/t4/checkpoints/../expected_code_hashes.txt",
        "D:/opt/hivemind/t4/model.safetensors",
    ):
        with pytest.raises(PermissionError, match="/opt/hivemind/t4"):
            assert_not_protected_t4_path(forbidden)

    allowed_linux = assert_not_protected_t4_path("/mnt/disks/disk-socrateai-local-1/gcpcrowd")
    assert "gcpcrowd" in str(allowed_linux)
    allowed_tmp = assert_not_protected_t4_path(tmp_path / "gcpcrowd")
    assert allowed_tmp == tmp_path / "gcpcrowd"


def test_gcp_t4_spot_budget_governance_within_150_usd() -> None:
    # Default piggybacked training inside scheduled 4.5h/day window costs $0.00 incremental
    plan = compute_gcp_t4_spot_crowd_budget()
    assert plan.within_budget is True
    assert plan.incremental_monthly_usd == 0.0
    assert plan.projected_total_monthly_usd == 120.62
    assert plan.max_additional_hours_per_day_within_budget >= 5.0

    # Extra 2h/day Spot still fits under $150/mo
    plan_extra_2h = compute_gcp_t4_spot_crowd_budget(extra_spot_hours_per_day=2.0)
    assert plan_extra_2h.within_budget is True
    assert plan_extra_2h.projected_total_monthly_usd < 150.0

    # Extra 8h/day Spot exceeds $150/mo and is flagged
    plan_over = compute_gcp_t4_spot_crowd_budget(extra_spot_hours_per_day=8.0)
    assert plan_over.within_budget is False
    assert plan_over.projected_total_monthly_usd > 150.0

    # On-demand provisioning is strictly rejected by AGENTS.md Rule 2.3
    with pytest.raises(ValueError, match="SPOT"):
        compute_gcp_t4_spot_crowd_budget(provisioning_model="STANDARD")


def test_chunked_causal_lm_cross_entropy_exact_value_and_gradients() -> None:
    torch.manual_seed(7)
    b, t, d, v = 2, 29, 32, 197
    h_ref = torch.randn(b, t, d, dtype=torch.float32, requires_grad=True)
    w_ref = torch.randn(v, d, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, v, (b, t), dtype=torch.long)
    labels[0, 5] = -100

    logits = F.linear(h_ref, w_ref)
    loss_ref = F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, v),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    loss_ref.backward()

    h_chk = h_ref.detach().clone().requires_grad_(True)
    w_chk = w_ref.detach().clone().requires_grad_(True)
    loss_chk = chunked_causal_lm_cross_entropy(
        h_chk,
        w_chk,
        labels,
        token_chunk_size=11,
        use_checkpoint=True,
    )
    loss_chk.backward()

    assert torch.allclose(loss_chk, loss_ref, atol=1e-6, rtol=1e-5)
    assert h_chk.grad is not None and h_ref.grad is not None
    assert w_chk.grad is not None and w_ref.grad is not None
    assert torch.allclose(h_chk.grad, h_ref.grad, atol=1e-5, rtol=1e-4)
    assert torch.allclose(w_chk.grad, w_ref.grad, atol=1e-5, rtol=1e-4)


def test_adaptive_t4_recipe_selector_enforces_4300_mib_floor() -> None:
    # 1. Idle T4 in training window (12,500 MiB free / 15,360 MiB total) -> full_trunk_8bit_ckpt
    r_full = select_t4_diloco_recipe(
        requested_mode="auto",
        round_index=0,
        num_layers=24,
        window_blocks=6,
        cuda_available=True,
        vram_info_mib=(12_500, 15_360),
        compute_capability=(7, 5),
        gpu_name="Tesla T4",
    )
    assert r_full.can_train_on_gpu is True
    assert r_full.recipe == "full_trunk_8bit_ckpt"
    assert r_full.frozen_dtype == "float16"
    assert r_full.trainable_dtype == "float32"
    assert r_full.projected_free_vram_mib >= 4300

    # 2. Shared T4 with llama-server resident (7,500 MiB free) -> block_window_diloco rotating over 4 rounds
    covered_blocks: set[int] = set()
    for rnd in range(4):
        r_win = select_t4_diloco_recipe(
            requested_mode="auto",
            round_index=rnd,
            num_layers=24,
            window_blocks=6,
            cuda_available=True,
            vram_info_mib=(7_500, 15_360),
            compute_capability=(7, 5),
            gpu_name="Tesla T4",
        )
        assert r_win.can_train_on_gpu is True
        assert r_win.recipe == "block_window_diloco"
        assert r_win.projected_free_vram_mib >= 4300
        covered_blocks.update(r_win.window_block_indices)
    assert covered_blocks == set(range(24))

    # 3. Tight T4 headroom (6,200 MiB free) -> qlora_graft_diloco
    r_lora = select_t4_diloco_recipe(
        requested_mode="auto",
        round_index=0,
        cuda_available=True,
        vram_info_mib=(6_200, 15_360),
        compute_capability=(7, 5),
        gpu_name="Tesla T4",
    )
    assert r_lora.can_train_on_gpu is True
    assert r_lora.recipe == "qlora_graft_diloco"
    assert r_lora.projected_free_vram_mib >= 4300

    # 4. Heavily loaded T4 (5,500 MiB free -> 5500 - 1750 = 3750 < 4300) -> refuses GPU!
    r_yield = select_t4_diloco_recipe(
        requested_mode="auto",
        round_index=0,
        cuda_available=True,
        vram_info_mib=(5_500, 15_360),
        compute_capability=(7, 5),
        gpu_name="Tesla T4",
    )
    assert r_yield.can_train_on_gpu is False
    assert r_yield.recipe == "yield_insufficient_vram"


def test_governor_yield_guard_and_resumable_segment_recovery(tmp_path: Path) -> None:
    hold = tmp_path / "HOLD"
    hold.write_text("", encoding="utf-8")
    g_hold = T4GovernorYieldGuard(hold_file=hold)
    assert g_hold.check().should_yield is True
    assert g_hold.check().trigger == "operator_hold_file"
    hold.unlink()

    g_spot = T4GovernorYieldGuard(spot_probe_fn=lambda: True, spot_poll_interval_s=0.01)
    chk_spot = g_spot.check()
    assert chk_spot.should_yield is True
    assert chk_spot.spot_preempted is True
    assert chk_spot.trigger == "gce_spot_preemption_30s_notice"

    # Mid-round preemption at step 3 of 6 via nginx in-flight inference, followed by seamless resume!
    model = _tiny_crowd_model()
    shard = torch.tensor([(i * 5 + 3) % 120 for i in range(256)], dtype=torch.long)
    calls = {"n": 0}

    def _nginx_probe() -> int:
        calls["n"] += 1
        return 2 if calls["n"] == 4 else 0

    guard = T4GovernorYieldGuard(
        nginx_probe_fn=_nginx_probe,
        min_free_vram_mib=0,
    )
    part1 = run_gcp_t4_diloco_inner_loop(
        model,
        shard,
        learner_id="gcp-t4-node1",
        round_index=2,
        inner_steps=6,
        batch_size=2,
        seq_len=32,
        learning_rate=3e-3,
        requested_mode="cpu_fp32",
        work_dir=tmp_path / "gcpcrowd",
        yield_guard=guard,
    )
    assert part1.preempted is True
    assert part1.preemption_trigger == "nginx_in_flight_inference"
    assert part1.steps_completed == 3
    assert part1.segment_checkpoint_path is not None and part1.segment_checkpoint_path.exists()
    assert part1.yield_latency_ms is not None and part1.yield_latency_ms < 1000.0

    # Clear nginx traffic and resume from the saved segment
    guard_clear = T4GovernorYieldGuard(nginx_probe_fn=lambda: 0, min_free_vram_mib=0)
    part2 = run_gcp_t4_diloco_inner_loop(
        model,
        shard,
        learner_id="gcp-t4-node1",
        round_index=2,
        inner_steps=6,
        batch_size=2,
        seq_len=32,
        learning_rate=3e-3,
        requested_mode="cpu_fp32",
        work_dir=tmp_path / "gcpcrowd",
        resume_from_segment=True,
        yield_guard=guard_clear,
    )
    assert part2.preempted is False
    assert part2.steps_completed == 6
    assert len(part2.step_losses) == 6
    assert part2.loss_after < part2.loss_before
    assert not part1.segment_checkpoint_path.exists()


def test_heterogeneous_t4_and_rtx2070_diloco_outer_nesterov_with_byzantine_rejection_and_loss_receipt(
    tmp_path: Path,
) -> None:
    model = _tiny_crowd_model(seed=42)
    pattern = torch.tensor([(i * 3 + 7) % 64 for i in range(64)], dtype=torch.long)
    stream = pattern.repeat(8)
    shard_t4 = stream[:320].clone()
    shard_2070 = torch.roll(stream, shifts=4)[:320].clone()
    shard_peer = torch.roll(stream, shifts=8)[:320].clone()
    held_out = stream[:192].clone()

    secret = hashlib.sha256(b"t4-test-evaluator-secret").digest()
    work_dir = tmp_path / "outer_hub"
    hub = GcpCrowdDiLoCoOuterHub(
        model.export_flat_base_bf16(),
        outer_lr=0.7,
        outer_momentum=0.9,
        f_byzantine=1,
        work_dir=work_dir,
        evaluator_id="hivemind-t4",
        evaluator_secret=secret,
    )

    # Learner 0: GCP T4 running block_window_diloco
    model.load_flat_tensors(hub.theta_bf16)
    res_t4 = run_gcp_t4_diloco_inner_loop(
        model,
        shard_t4,
        learner_id="gcp-t4-hermes-node1",
        round_index=0,
        inner_steps=8,
        batch_size=2,
        seq_len=32,
        learning_rate=8e-4,
        requested_mode="block_window_diloco",
        window_blocks=2,
        prefer_cuda=False,
    )

    # Learner 1: Local RTX 2070 running qlora_graft_diloco projected into trunk delta space
    model.load_flat_tensors(hub.theta_bf16)
    res_2070 = run_gcp_t4_diloco_inner_loop(
        model,
        shard_2070,
        learner_id="xavgamsrv-rtx2070",
        round_index=0,
        inner_steps=8,
        batch_size=2,
        seq_len=32,
        learning_rate=1.5e-3,
        requested_mode="qlora_graft_diloco",
        lora_rank=8,
        prefer_cuda=False,
    )

    # Learner 2: Honest community GPU running full_trunk_8bit_ckpt
    model.load_flat_tensors(hub.theta_bf16)
    res_peer = run_gcp_t4_diloco_inner_loop(
        model,
        shard_peer,
        learner_id="community-rtx4070",
        round_index=0,
        inner_steps=8,
        batch_size=2,
        seq_len=32,
        learning_rate=8e-4,
        requested_mode="cpu_fp32",
        prefer_cuda=False,
    )

    # Learner 3 + Learner 4: one more honest learner + 1 Byzantine poisoner (N=5 >= 2*1 + 3 for f=1 Krum!)
    honest_clone = ("community-rtx3090", res_t4.delta_bf16.clone(), res_t4.tokens_processed)
    poison_delta = (-50.0 * res_t4.delta_bf16.float()).to(torch.bfloat16)
    byzantine = ("byzantine-poisoner", poison_delta, 999_999)

    outer = hub.step(
        [res_t4, res_2070, res_peer, honest_clone, byzantine],
        eval_model=model,
        held_out_tokens=held_out,
        eval_window_len=48,
        eval_num_windows=3,
        prefer_cuda_eval=False,
    )

    assert "byzantine-poisoner" in outer.rejected_learners
    assert "byzantine-poisoner" not in outer.selected_learners
    assert outer.rms_guard.accepted is True
    assert outer.ratchet_decision.accepted is True
    assert outer.accepted is True
    assert outer.loss_after < outer.loss_before
    assert outer.checkpoint_path is not None and outer.checkpoint_path.exists()
    assert outer.loss_check_receipt is not None

    # Verify cryptographic HMAC signature on the T4 LossCheckReceipt
    receipt_obj = LossCheckReceipt.from_json(json.dumps(outer.loss_check_receipt))
    assert receipt_obj.evaluator_id == "hivemind-t4"
    assert receipt_obj.verify_signature(secret) is True

    # Verify packed CrowdGPT delta stream from inner learner unpacks cleanly
    unpacked = unpack_crowdgpt_delta_stream(res_t4.compressed_delta_stream)
    assert unpacked.metadata["hivemindRecipe"] == "block_window_diloco"
    assert unpacked.base_delta_bf16.numel() == res_t4.delta_bf16.numel()

    # Verify standalone LoRA safetensors export guard
    inject_crowd_lora_adapters(model, rank=4, alpha=8.0)
    lora_file = tmp_path / "gcpcrowd_lora.safetensors"
    lora_sha = save_crowd_lora_safetensors(model, lora_file)
    assert len(lora_sha) == 64 and lora_file.exists()


def test_t4_governor_markers_opentrainer_campaign_and_worker_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import t4_governor

    assert "gcp_t4_crowd_diloco" in t4_governor.PIPELINE_MARKERS
    cmds = {
        "21": b"python scripts/gcp_t4_crowd_diloco_worker.py --mode daemon",
        "22": b"python -m hivemind.boinc.real_client.boinc_server_scripts validate x",
        "23": b"llama-server -m bee",
    }
    real_path = t4_governor.Path

    class _CrossPlatformFakePath(type(real_path("/"))):
        def read_bytes(self):
            parts = str(self).replace("\\", "/").strip("/").split("/")
            pid = parts[1] if len(parts) >= 2 else ""
            if pid not in cmds:
                raise OSError
            return cmds[pid].replace(b" ", b"\0")

    monkeypatch.setattr(t4_governor, "Path", _CrossPlatformFakePath)
    procs = [("21", "python", "2350"), ("22", "python", "1200"), ("23", "llama-server", "4838")]
    assert t4_governor.pipeline_gpu_pids(procs) == ["21", "22"]
    # the crowd worker is part of the training pipeline (not "someone asking"), but it is NOT BOINC's task: counting its
    # memory as BOINC's would make the governor treat it as free and resume BOINC on top of it
    assert t4_governor.boinc_gpu_pids(procs) == []
    monkeypatch.setattr(t4_governor, "Path", real_path)
    assert is_allowed_opentrainer_campaign_id(GCPCROWD_CAMPAIGN_ID) is True
    plan = build_crowdgpt_trainer_producer_plan(
        recent_rounds=[{"run_id": GCPCROWD_CAMPAIGN_ID, "tokens_trained": 16384}],
    )
    assert GCPCROWD_CAMPAIGN_ID in plan["live_campaign_telemetry"]["allowed_campaign_ids"]
    assert plan["live_campaign_telemetry"]["crowdgpt_tokens_trained"] == 16384

    spec = importlib.util.spec_from_file_location(
        "gcp_t4_crowd_diloco_worker",
        ROOT / "scripts" / "gcp_t4_crowd_diloco_worker.py",
    )
    assert spec and spec.loader
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    assert worker.main(["--mode", "budget-check"]) == 0
    budget_out = json.loads(capsys.readouterr().out)
    assert budget_out["within_budget"] is True

    unit_path = tmp_path / "hivemind-gcpcrowd-diloco.service"
    assert worker.main(["--mode", "render-systemd", "--work-dir", str(tmp_path / "work"), "--systemd-out", str(unit_path)]) == 0
    unit_txt = unit_path.read_text(encoding="utf-8")
    assert "--min-free-vram-mib 4300" in unit_txt
    assert "/opt/hivemind/t4/" not in unit_txt.split("ExecStart=")[1]

    rc = worker.main(
        [
            "--mode",
            "co-train-round",
            "--small-arch",
            "--cpu-only",
            "--rounds",
            "1",
            "--inner-steps",
            "4",
            "--seq-len",
            "32",
            "--lr",
            "5e-4",
            "--work-dir",
            str(tmp_path / "co_train_work"),
        ]
    )
    assert rc == 0
    co_out = json.loads(capsys.readouterr().out)
    assert co_out["status"] == "ok"
    assert len(co_out["rounds"]) == 1
    assert co_out["rounds"][0]["accepted"] is True
    assert co_out["rounds"][0]["loss_after"] < co_out["rounds"][0]["loss_before"]
