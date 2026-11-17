#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""CLI Daemon & Runner for the Permanent Low-Cost VM OpenTrainer Orchestrator.

Modes:
  1. `orchestrator`: Runs the permanent CPU-only OpenTrainer HTTP + AMCP/Kafka orchestrator daemon
     on a low-cost Spot VM (`e2-small` at ~$3.65/mo or local port `:8096`).
  2. `learner`: Runs an outbound HTTP GPU/CPU learner (e.g. on RTX 2070) against an OpenTrainer
     orchestrator URL without requiring the Tesla T4.
  3. `self-test`: Executes an end-to-end T4-offline collaborative QLoRA MoE + Outer DiLoCo run
     combining the local GPU (`cuda:0` NF4 when available) and the OpenTrainer CPU Sentinel
     (`t4_online=False`), writing `artifacts/opentrainer_lowcost_vm_verification.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e8_qlora_moe_round as qlora_round  # noqa: E402
from hivemind.swarm import federation_agents as fed  # noqa: E402
from hivemind.swarm.amcp_kafka_transport import sign_envelope  # noqa: E402
from hivemind.swarm.opentrainer_orchestrator import (  # noqa: E402
    PIN_FILES,
    LowCostVMProfile,
    OpenTrainerConfig,
    PermanentOpenTrainerOrchestrator,
    create_opentrainer_app,
    execute_http_learner_round,
    resolve_opentrainer_keyring,
)


def run_self_test(
    *,
    rounds: int = 3,
    inner_steps: int = 20,
    device: str = "auto",
    quant: str = "nf4",
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Execute a real T4-offline collaborative training run (GPU learner + low-cost VM CPU sentinel)."""
    actual_device = "cuda" if (device in ("cuda", "auto") and torch.cuda.is_available()) else "cpu"
    actual_quant = quant if actual_device == "cuda" else "none"
    keyring = resolve_opentrainer_keyring()

    out_file = out_path or (REPO_ROOT / "artifacts" / "opentrainer_lowcost_vm_verification.json")
    ckpt_file = out_file.with_suffix(".ckpt.pt")
    if ckpt_file.exists():
        ckpt_file.unlink()

    cfg = OpenTrainerConfig(
        run_id="opentrainer-t4free-selftest",
        learners=("l0", "l1"),
        quorum=1,
        rounds=rounds,
        seed=1,
        inner_steps=inner_steps,
        inner_lr=1e-2,
        batch_size=4,
        shard_size=32,
        outer_lr=0.5,
        outer_mu=0.5,
        cpu_sentinel_when_t4_offline=True,
        cpu_sentinel_learner_id="l0",
        cpu_sentinel_inner_steps=10,
        checkpoint_path=ckpt_file,
        report_path=out_file,
    )
    orch = PermanentOpenTrainerOrchestrator(cfg, keyring=keyring, vm_profile=LowCostVMProfile())
    pin = fed.code_pin(PIN_FILES)
    base = qlora_round.build_base_state(cfg.seed)
    base_sha = fed.tensors_sha256(base)

    gpu_model = qlora_round.load_model(base, actual_quant, actual_device, cfg.seed)
    evalset = qlora_round.make_eval(len(cfg.learners), cfg.seed)
    shard_l1 = qlora_round.make_shard(1, cfg.shard_size, cfg.seed)
    gpu_name = torch.cuda.get_device_name(0) if actual_device == "cuda" else "CPU"
    if actual_device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for rnd in range(rounds):
        offer_env = orch.get_signed_offer_envelope()
        offer_payload = offer_env["data"]["payload"]
        start_state = fed.check_offer(offer_payload, my_pin=pin, my_base_sha256=base_sha)

        # 1. T4 is offline -> low-cost VM runs its lightweight CPU sentinel for shard l0
        orch.run_cpu_sentinel_step_if_needed()

        # 2. Local RTX 2070 (or community GPU) trains shard l1 in NF4 on CUDA
        out_round = qlora_round.learner_round(
            gpu_model,
            start_state,
            shard_l1,
            evalset,
            steps=cfg.inner_steps,
            lr=cfg.inner_lr,
            batch=cfg.batch_size,
            seed=cfg.seed,
            rnd=rnd,
            learner_id=1,
            device=actual_device,
            shuffle_labels=False,
        )
        delta = out_round["delta"]
        tokens = int(out_round["tokens"])
        eval_before = float(out_round["eval_loss_global"])
        peak_vram = float(torch.cuda.max_memory_allocated() / 2**20) if actual_device == "cuda" else 0.0
        norm = fed.check_delta(delta, start_state, cfg.max_delta_l2)
        res_ev = fed.build_result(
            run_id=cfg.run_id,
            rnd=rnd,
            learner_id="l1",
            host=f"rtx2070-{gpu_name.replace(' ', '_')}",
            tokens=tokens,
            delta=delta,
            delta_l2=norm,
            eval_loss_global=eval_before,
            pin=pin,
            global_sha256=str(offer_payload["global_sha256"]),
            vram_peak_mib=peak_vram,
        )
        orch.submit_result_envelope(sign_envelope(res_ev, keyring, ttl_s=900.0))

    if ckpt_file.exists():
        ckpt_file.unlink()

    report = orch.build_report()
    report["self_test_device"] = actual_device
    report["self_test_gpu_name"] = gpu_name
    report["self_test_quant"] = actual_quant
    report["loss_decreased"] = bool(report["current_eval_loss"] < report["initial_eval_loss"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HiveMind Low-Cost VM Permanent OpenTrainer Orchestrator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_orch = sub.add_parser("orchestrator", help="Run the permanent OpenTrainer HTTP/AMCP orchestrator server")
    p_orch.add_argument("--host", default="0.0.0.0")
    p_orch.add_argument("--port", type=int, default=8096)
    p_orch.add_argument("--run-id", default="opentrainer-perm-001")
    p_orch.add_argument("--learners", nargs="+", default=["l0", "l1"])
    p_orch.add_argument("--quorum", type=int, default=1)
    p_orch.add_argument("--rounds", type=int, default=500)
    p_orch.add_argument("--seed", type=int, default=1)
    p_orch.add_argument("--inner-steps", type=int, default=20)
    p_orch.add_argument("--inner-lr", type=float, default=1e-2)
    p_orch.add_argument("--outer-lr", type=float, default=0.5)
    p_orch.add_argument("--outer-mu", type=float, default=0.5)
    p_orch.add_argument("--round-deadline-s", type=float, default=120.0)
    p_orch.add_argument("--checkpoint", default="artifacts/opentrainer/checkpoint.pt")
    p_orch.add_argument("--out", default="artifacts/opentrainer/opentrainer_status.json")

    p_learn = sub.add_parser("learner", help="Run an outbound HTTP learner against the OpenTrainer orchestrator")
    p_learn.add_argument("--orchestrator-url", default="http://127.0.0.1:8096")
    p_learn.add_argument("--learner-id", default="l1")
    p_learn.add_argument("--index", type=int, default=1)
    p_learn.add_argument("--quant", choices=("none", "nf4"), default="nf4")
    p_learn.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    p_learn.add_argument("--host-label", default="")
    p_learn.add_argument("--poll-interval-s", type=float, default=3.0)
    p_learn.add_argument("--max-rounds", type=int, default=0, help="0 = run until orchestrator completes")

    p_test = sub.add_parser("self-test", help="Run T4-offline collaborative training verification")
    p_test.add_argument("--rounds", type=int, default=3)
    p_test.add_argument("--inner-steps", type=int, default=20)
    p_test.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    p_test.add_argument("--quant", choices=("none", "nf4"), default="nf4")
    p_test.add_argument("--out", default="artifacts/opentrainer_lowcost_vm_verification.json")

    args = parser.parse_args(argv)
    if args.cmd == "self-test":
        rep = run_self_test(
            rounds=args.rounds,
            inner_steps=args.inner_steps,
            device=args.device,
            quant=args.quant,
            out_path=Path(args.out),
        )
        print(json.dumps(rep, indent=2))
        return 0 if rep["loss_decreased"] and rep["completed"] else 1

    if args.cmd == "orchestrator":
        import uvicorn

        cfg = OpenTrainerConfig(
            run_id=args.run_id,
            learners=tuple(args.learners),
            quorum=args.quorum,
            rounds=args.rounds,
            seed=args.seed,
            inner_steps=args.inner_steps,
            inner_lr=args.inner_lr,
            outer_lr=args.outer_lr,
            outer_mu=args.outer_mu,
            round_deadline_s=args.round_deadline_s,
            checkpoint_path=Path(args.checkpoint),
            report_path=Path(args.out),
        )
        orch = PermanentOpenTrainerOrchestrator(cfg)
        app = create_opentrainer_app(orch)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    if args.cmd == "learner":
        keyring = resolve_opentrainer_keyring()
        done = 0
        while True:
            res = execute_http_learner_round(
                args.orchestrator_url,
                learner_id=args.learner_id,
                learner_index=args.index,
                keyring=keyring,
                quant=args.quant,
                device="cuda" if args.device in ("cuda", "auto") and torch.cuda.is_available() else "cpu",
                host_label=args.host_label or None,
            )
            print(json.dumps(res), flush=True)
            if res.get("status") == "aborted":
                return 0
            done += 1
            if args.max_rounds > 0 and done >= args.max_rounds:
                return 0
            time.sleep(args.poll_interval_s)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
