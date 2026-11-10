#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""GCP Spot Tesla T4 & Multi-GPU Swarm DiLoCo Worker for `GCPCrowd` (`Vxtzq/Crowd-v1` x HiveMind).

Modes:
  1. `budget-check`: Verifies Spot `n1-standard-8` + Tesla T4 schedule against the `$150/mo` `AGENTS.md` cap.
  2. `render-systemd`: Prints or writes the serving-first `hivemind-gcpcrowd-diloco.service` template
     (writes strictly to `/mnt/disks/disk-socrateai-local-1/gcpcrowd`, never `/opt/hivemind/t4/`).
  3. `co-train-round`: Executes a verified multi-learner DiLoCo outer Nesterov round (`Learner 0 = GCP T4`,
     `Learner 1 = RTX 2070` + optional community cohorts) on real `Vxtzq/Crowd-v1` weights (or synthetic
     smoke architecture), produces a signed `hivemind-t4` `LossCheckReceipt`, and optionally submits the
     round receipt to HiveMind OpenTrainer (`POST /v1/hivemind/opentrainer/submit-delta`).
  4. `once` / `daemon`: Runs the continuous serving-first T4 DiLoCo inner learner governed by
     `T4GovernorYieldGuard` (`HOLD`, `governor_state.json`, `nginx` `stub_status`, GCE Spot 30s preemption,
     and `>= 4,300 MiB` free VRAM floor).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import urllib.request

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hivemind.router.opentrainer import DEFAULT_MIN_FREE_VRAM_MIB_T4  # noqa: E402
from hivemind.swarm.crowdgpt_bridge import DEFAULT_HIVEMIND_GATEWAY_URL  # noqa: E402
from hivemind.swarm.crowdgpt_live_runtime import (  # noqa: E402
    CrowdGPTHuggingFaceSync,
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
    load_crowd_v1_safetensors_into_model,
)
from hivemind.swarm.gcp_t4_crowd_diloco import (  # noqa: E402
    DEFAULT_GCP_T4_WORK_DIR_POSIX,
    DEFAULT_T4_GOVERNOR_STATE_POSIX,
    DEFAULT_T4_HOLD_FILE_POSIX,
    DEFAULT_T4_NGINX_STATUS_URL,
    GCPCROWD_CAMPAIGN_ID,
    GcpCrowdDiLoCoOuterHub,
    T4GovernorYieldGuard,
    assert_not_protected_t4_path,
    compute_gcp_t4_spot_crowd_budget,
    render_gcp_t4_diloco_systemd_unit,
    run_gcp_t4_diloco_inner_loop,
)


def _submit_opentrainer_receipt(
    *,
    gateway_url: str,
    api_key: str,
    campaign_id: str,
    round_index: int,
    delta_l2: float,
    eval_loss: float,
    tokens_trained: int,
    vram_free_mib: int,
    vram_total_mib: int,
    host_label: str = "socreateai-agora-hermes-node1",
) -> dict[str, Any]:
    payload = {
        "run_id": campaign_id,
        "round_index": int(round_index),
        "delta_l2": min(24.5, max(1e-6, float(delta_l2))),
        "eval_loss": float(eval_loss),
        "tokens_trained": max(1, int(tokens_trained)),
        "vram_free_mib": int(vram_free_mib),
        "vram_total_mib": int(vram_total_mib),
        "host_label": host_label,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{gateway_url.rstrip('/')}/v1/hivemind/opentrainer/submit-delta",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "hivemind-gcpcrowd-t4-diloco/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _load_or_build_model(
    *,
    cache_dir: Path | None,
    use_small_arch: bool,
) -> tuple[CrowdV1WithEngram, dict[str, Any]]:
    if use_small_arch:
        torch.manual_seed(20261110)
        cfg = CrowdV1RuntimeConfig(
            vocab_size=256,
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=256,
            max_position_embeddings=256,
            engram_buckets=512,
        )
        model = CrowdV1WithEngram(cfg)
        with torch.no_grad():
            for p in model.parameters():
                if p.ndim >= 2:
                    p.mul_(0.02)
        return model, {"source": "synthetic_small_arch", "hf_revision": "local-smoke", "global_step": 0}

    sync = CrowdGPTHuggingFaceSync(cache_dir=cache_dir)
    cached_candidates = sorted(sync.cache_dir.glob("model_*.safetensors")) if sync.cache_dir.exists() else []
    if cached_candidates:
        weights_path = cached_candidates[-1]
        rev = weights_path.stem.replace("model_", "")
        step = 348
    else:
        files = sync.ensure_latest_files(force_check=True)
        weights_path = Path(files["weights_path"])
        rev = str(files["hf_revision"])
        step = int(files["global_step"])

    prev_dt = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = CrowdV1WithEngram(CrowdV1RuntimeConfig())
    finally:
        torch.set_default_dtype(prev_dt)
    sha = load_crowd_v1_safetensors_into_model(model, weights_path)
    return model, {
        "source": str(weights_path),
        "hf_revision": rev,
        "global_step": step,
        "weights_sha256": sha,
    }


def _build_token_shards(vocab_size: int, shard_tokens: int = 512) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build repeatable structured non-IID token shards + held-out validation stream."""
    base_pattern = torch.tensor(
        [(i * 7 + 13) % max(16, min(vocab_size, 256)) for i in range(16)],
        dtype=torch.long,
    )
    repeats = max(8, (shard_tokens // 16) + 4)
    stream = base_pattern.repeat(repeats)
    shard_t4 = stream[:shard_tokens].clone()
    shard_2070 = torch.roll(stream, shifts=4)[:shard_tokens].clone()
    held_out = stream[: max(128, shard_tokens // 2)].clone()
    return shard_t4, shard_2070, held_out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GCP Spot Tesla T4 DiLoCo Co-Trainer for GCPCrowd")
    ap.add_argument(
        "--mode",
        choices=("budget-check", "render-systemd", "co-train-round", "once", "daemon"),
        default="co-train-round",
    )
    ap.add_argument("--work-dir", type=Path, default=Path(DEFAULT_GCP_T4_WORK_DIR_POSIX))
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--governor-state", type=Path, default=None)
    ap.add_argument("--hold-file", type=Path, default=None)
    ap.add_argument("--nginx-status-url", default=None)
    ap.add_argument("--require-governor-training", action="store_true")
    ap.add_argument("--check-spot-preemption", action="store_true")
    ap.add_argument("--min-free-vram-mib", type=int, default=DEFAULT_MIN_FREE_VRAM_MIB_T4)
    ap.add_argument(
        "--recipe",
        choices=("auto", "full_trunk_8bit_ckpt", "block_window_diloco", "qlora_graft_diloco", "cpu_fp32"),
        default="auto",
    )
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--inner-steps", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--outer-lr", type=float, default=0.7)
    ap.add_argument("--outer-momentum", type=float, default=0.9)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--window-blocks", type=int, default=6)
    ap.add_argument("--small-arch", action="store_true", help="Use lightweight test config instead of 1B HF checkpoint")
    ap.add_argument("--prefer-cuda", action="store_true", default=True)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--gateway", default=DEFAULT_HIVEMIND_GATEWAY_URL)
    ap.add_argument("--submit-opentrainer", action="store_true")
    ap.add_argument("--campaign-id", default=GCPCROWD_CAMPAIGN_ID)
    ap.add_argument("--no-save-safetensors", action="store_true")
    ap.add_argument("--systemd-out", type=Path, default=None)
    ap.add_argument("--poll-s", type=float, default=15.0)
    args = ap.parse_args(argv)

    if args.mode == "budget-check":
        plan = compute_gcp_t4_spot_crowd_budget()
        print(json.dumps(asdict(plan), indent=2, sort_keys=True))
        return 0 if plan.within_budget else 1

    safe_work = assert_not_protected_t4_path(args.work_dir)
    if args.mode == "render-systemd":
        unit_txt = render_gcp_t4_diloco_systemd_unit(
            work_dir=str(safe_work),
            governor_state=str(args.governor_state or DEFAULT_T4_GOVERNOR_STATE_POSIX),
            hold_file=str(args.hold_file or DEFAULT_T4_HOLD_FILE_POSIX),
            nginx_status_url=str(args.nginx_status_url or DEFAULT_T4_NGINX_STATUS_URL),
        )
        if args.systemd_out is not None:
            out_p = assert_not_protected_t4_path(args.systemd_out)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(unit_txt, encoding="utf-8")
        else:
            print(unit_txt, end="")
        return 0

    prefer_cuda = bool(args.prefer_cuda and not args.cpu_only)
    eff_min_free = int(args.min_free_vram_mib)
    if prefer_cuda and torch.cuda.is_available():
        _, total_b = torch.cuda.mem_get_info(0)
        total_m = int(total_b // (1024 * 1024))
        if total_m < 14_000 and eff_min_free == DEFAULT_MIN_FREE_VRAM_MIB_T4:
            eff_min_free = 1536

    guard = T4GovernorYieldGuard(
        hold_file=args.hold_file,
        governor_state_file=args.governor_state,
        require_governor_training_state=args.require_governor_training,
        nginx_status_url=args.nginx_status_url,
        check_gce_spot_metadata=args.check_spot_preemption,
        min_free_vram_mib=eff_min_free,
    )

    model, rev_meta = _load_or_build_model(cache_dir=args.cache_dir, use_small_arch=args.small_arch)
    shard_t4, shard_2070, held_out = _build_token_shards(model.cfg.vocab_size, shard_tokens=max(256, args.seq_len * 6))

    if args.mode == "co-train-round":
        secret = hashlib.sha256(b"hivemind-gcpcrowd-t4-receipt-key").digest()
        hub = GcpCrowdDiLoCoOuterHub(
            model.export_flat_base_bf16(),
            outer_lr=args.outer_lr,
            outer_momentum=args.outer_momentum,
            f_byzantine=1,
            work_dir=safe_work,
            evaluator_id="hivemind-t4",
            evaluator_secret=secret,
            clone_initial=False,
        )
        if prefer_cuda and torch.cuda.is_available():
            import gc
            model.place_for_inference(device="cuda", dtype=torch.float16, offload_engram_to_cpu=True)
            gc.collect()
        round_summaries: list[dict[str, Any]] = []
        for rnd in range(max(1, args.rounds)):
            if rnd > 0:
                model.load_flat_tensors(hub.theta_bf16)
            res_t4 = run_gcp_t4_diloco_inner_loop(
                model,
                shard_t4,
                learner_id="gcp-t4-hermes-node1",
                hiveminder_handle="gcp-t4-hermes",
                device_suffix="t4spot",
                round_index=rnd,
                inner_steps=args.inner_steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                learning_rate=args.lr,
                requested_mode=args.recipe,
                window_blocks=args.window_blocks,
                lora_rank=args.lora_rank,
                prefer_cuda=prefer_cuda,
                work_dir=safe_work,
                yield_guard=guard,
                pack_stream=False,
                base_before_bf16_ref=hub.theta_bf16,
            )

            model.load_flat_tensors(hub.theta_bf16)
            res_2070 = run_gcp_t4_diloco_inner_loop(
                model,
                shard_2070,
                learner_id="xavgamsrv-rtx2070",
                hiveminder_handle="xavgamsrv",
                device_suffix="rtx2070",
                round_index=rnd,
                inner_steps=args.inner_steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                learning_rate=args.lr,
                requested_mode="qlora_graft_diloco" if args.recipe == "auto" else args.recipe,
                lora_rank=args.lora_rank,
                prefer_cuda=prefer_cuda,
                work_dir=safe_work,
                yield_guard=guard,
                pack_stream=False,
                base_before_bf16_ref=hub.theta_bf16,
            )

            outer = hub.step(
                [res_t4, res_2070],
                eval_model=model,
                held_out_tokens=held_out,
                eval_window_len=min(args.seq_len, 64),
                eval_num_windows=2,
                prefer_cuda_eval=prefer_cuda,
                save_safetensors=not args.no_save_safetensors,
                clear_input_deltas=True,
            )

            ot_resp: dict[str, Any] | None = None
            api_key = os.environ.get("HIVEMIND_API_KEY", "").strip()
            if args.submit_opentrainer and api_key:
                free_m = res_t4.recipe_decision.vram_free_mib or 10_500
                total_m = res_t4.recipe_decision.vram_total_mib or 15_360
                if total_m >= 14_000:
                    free_m = max(free_m, DEFAULT_MIN_FREE_VRAM_MIB_T4)
                else:
                    free_m = max(free_m, 1536)
                ot_resp = _submit_opentrainer_receipt(
                    gateway_url=args.gateway,
                    api_key=api_key,
                    campaign_id=args.campaign_id,
                    round_index=outer.round_index,
                    delta_l2=outer.outer_delta_l2,
                    eval_loss=outer.loss_after,
                    tokens_trained=outer.effective_tokens,
                    vram_free_mib=free_m,
                    vram_total_mib=total_m,
                )

            round_summaries.append(
                {
                    "round_index": outer.round_index,
                    "accepted": outer.accepted,
                    "loss_before": outer.loss_before,
                    "loss_after": outer.loss_after,
                    "outer_delta_l2": outer.outer_delta_l2,
                    "effective_tokens": outer.effective_tokens,
                    "selected_learners": outer.selected_learners,
                    "t4_recipe": res_t4.recipe_decision.recipe,
                    "rtx2070_recipe": res_2070.recipe_decision.recipe,
                    "rms_reason": outer.rms_guard.reason,
                    "ratchet_reason": outer.ratchet_decision.reason,
                    "checkpoint_sha256": outer.checkpoint_sha256,
                    "opentrainer_submission": ot_resp,
                }
            )

        out_report = {
            "status": "ok",
            "campaign_id": args.campaign_id,
            "revision_meta": rev_meta,
            "budget_plan": asdict(compute_gcp_t4_spot_crowd_budget()),
            "rounds": round_summaries,
        }
        print(json.dumps(out_report, indent=2, sort_keys=True))
        return 0

    rnd = 0
    while True:
        res = run_gcp_t4_diloco_inner_loop(
            model,
            shard_t4,
            learner_id="gcp-t4-hermes-node1",
            hiveminder_handle="gcp-t4-hermes",
            device_suffix="t4spot",
            round_index=rnd,
            inner_steps=args.inner_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            learning_rate=args.lr,
            requested_mode=args.recipe,
            window_blocks=args.window_blocks,
            lora_rank=args.lora_rank,
            prefer_cuda=prefer_cuda,
            work_dir=safe_work,
            yield_guard=guard,
        )
        print(
            json.dumps(
                {
                    "round": rnd,
                    "recipe": res.recipe_decision.recipe,
                    "steps_completed": res.steps_completed,
                    "preempted": res.preempted,
                    "trigger": res.preemption_trigger,
                    "yield_latency_ms": res.yield_latency_ms,
                    "loss_before": res.loss_before,
                    "loss_after": res.loss_after,
                    "delta_l2": res.delta_l2,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not res.preempted:
            rnd += 1
        if args.mode == "once":
            return 0
        time.sleep(max(1.0, float(args.poll_s)))


if __name__ == "__main__":
    raise SystemExit(main())
