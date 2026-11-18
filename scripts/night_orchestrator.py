#!/usr/bin/env python3
"""Run the night expert-training schedule on the T4 VM, one subject after another, with hard time budgets and a guaranteed restore.

Plan and hypotheses: docs/NIGHT_MOE_TRAINING_PLAN.md and artifacts/experiments/NIGHT_MOE_prereg.json. Data: scripts/night_pack_build.py
(built beforehand, on CPU). Trainer: scripts/e17_train.py (QLoRA, assistant-token loss). Evaluation: scripts/night_eval.py through a
llama-server on the VM (base GGUF + ``--lora`` adapter), the same engine and settings for the base and for every adapter.

Modes:
  --dry-run     print the schedule and check every precondition that does not need the GPU; changes nothing
  --baselines   evaluate the BASE model on every pack (run this before the night, while the GPU is free)
  --run         the night: for each subject train, train a shuffled-label control, convert, serve, evaluate, then restore the producer
Every stage has a time budget; an overrun is recorded as TIMEOUT and the schedule moves on. State is in OUT/state.json so a
rerun resumes after the last finished stage. A stage that cannot run reports BLOCKED. Nothing is simulated.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
INSTANCE, ZONE = "socreateai-agora-hermes-node1", "us-east4-b"
D = "/mnt/disks/disk-socrateai-local-1/hivemind-train"
LLAMA = "/home/callensxavier_gmail_com/gwaya-lite/llama.cpp"
BASE_GGUF = "/mnt/disks/disk-socrateai-local-1/leanmaster/ollama/blobs/sha256-60e05f2100071479f596b964f89f510f057ce397ea22f2833a0cfe029bfc2463"
BASE_HF = "Qwen/Qwen2.5-Coder-7B-Instruct"
PORT = 8293
GATEWAY = "https://hivemind-openrouter-1003063861791.us-central1.run.app"
ENDPOINT = "https://34-48-48-154.sslip.io"
# train hours are CAPS (the trainer saves what it has when the cap is hit); epochs/size are the plan, the cap protects the night
SUBJECTS = {
    "maths": {"pairs": 1000, "epochs": 2, "train_cap_h": 0.75, "eval_packs": ["maths", "physics_transfer"]},
    "exoplanet": {"pairs": 1600, "epochs": 2, "train_cap_h": 0.6, "eval_packs": ["exoplanet", "astro_transfer"]},
    "stars": {"pairs": 974, "epochs": 3, "train_cap_h": 0.5, "eval_packs": ["stars", "astro_transfer"]},
    "cyber": {"pairs": 1200, "epochs": 2, "train_cap_h": 1.0, "eval_packs": ["cyber"]},
}
EVAL_PACKS = ["maths", "physics_transfer", "astro_transfer", "exoplanet", "stars", "cyber"]
CONTROL_CAP_H, CONVERT_MIN, SERVE_MIN, STAGE_SLACK_MIN = 0.25, 6, 8, 5
CKPT_EVERY = 10  # optimizer steps between resumable checkpoints: the GPU host is a Spot VM that can be stopped mid-run
# evaluation time budgets in minutes (estimates x ~1.6; real timings are written to state.json), and items used for the control model
EVAL_MIN = {"maths": 40, "physics_transfer": 25, "astro_transfer": 15, "exoplanet": 25, "stars": 15, "cyber": 25}
CONTROL_LIMIT = {"maths": 100}
INTERFERENCE_LIMIT = 100


def open_tunnel() -> subprocess.Popen:
    """Forward the VM's llama-server port to this machine (the evaluator runs here: its sandbox-free scoring needs no VM CPU)."""
    return subprocess.Popen(["gcloud", "compute", "ssh", INSTANCE, "--zone", ZONE, "--", "-N", "-L", f"{PORT}:127.0.0.1:{PORT}", "-o", "ServerAliveInterval=30", "-o", "ExitOnForwardFailure=yes"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def vm(cmd: str) -> list[str]:
    return ["gcloud", "compute", "ssh", INSTANCE, "--zone", ZONE, "--command", cmd]


def scp(local: list[Path], remote_dir: str) -> list[str]:
    return ["gcloud", "compute", "scp", "--zone", ZONE, *map(str, local), f"{INSTANCE}:{remote_dir}/"]


@dataclass
class Stage:
    name: str
    argv: list[str]
    timeout_min: float
    note: str = ""
    must_succeed: bool = False  # a failed required stage skips the rest of that subject, never the restore


@dataclass
class Plan:
    stages: list[Stage] = field(default_factory=list)

    def add(self, *a, **k) -> None:
        self.stages.append(Stage(*a, **k))


def train_cmd(pairs: str, out: str, epochs: int, cap_h: float) -> str:
    return (f"cd {D} && HF_HOME={D}/hf HF_HUB_OFFLINE=1 ./venv/bin/python e17_train.py --base {BASE_HF} --pairs {pairs} --out {out} "
            f"--epochs {epochs} --lr 1e-4 --accum 16 --warmup 10 --max-len 768 --seed 17 --max-hours {cap_h} --ckpt-every {CKPT_EVERY}")


def convert_cmd(adapter: str) -> str:
    return (f"cd {D} && BASE=$(ls -d {D}/hf/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/*) && HF_HOME={D}/hf HF_HUB_OFFLINE=1 "
            f"./venv/bin/python {LLAMA}/convert_lora_to_gguf.py {adapter} --base $BASE --outfile {adapter}.gguf")


def serve_cmd(adapter_gguf: str | None) -> str:
    lora = f"--lora {adapter_gguf}" if adapter_gguf else ""
    return (f"pkill -f 'llama-serve[r] .*--port {PORT}' ; sleep 2; cd {D} && (setsid nohup {LLAMA}/build/bin/llama-serve\"\"r -m {BASE_GGUF} {lora} -c 4096 -ngl 99 -np 1 "
            f"--host 127.0.0.1 --port {PORT} > serve_{PORT}.log 2>&1 < /dev/null &); for i in $(seq 1 180); do curl -s -m2 localhost:{PORT}/health | grep -q ok && echo UP && exit 0; sleep 2; done; echo DOWN; exit 1")


STOP_SERVER = f"pkill -f 'llama-serve[r] .*--port {PORT}'; sleep 3; nvidia-smi --query-gpu=memory.used --format=csv,noheader"
RESTORE = (f"cd ~/hivemind-producer && export HIVEMIND_API_KEY=$(cat api_key) HIVEMIND_ENDPOINT_TOKEN=$(cat ~/.hivemind_producer/endpoint_token) && "
           f"(setsid nohup python3 hivemind_producer_agent.py --gateway {GATEWAY} --endpoint {ENDPOINT} --model coder=qwen2.5-coder:7b-instruct "
           f"--model coder-lite=qwen2.5-coder:3b --cost-per-1k 0.4 --capacity-1k 60 --execution-class t4 > agent.log 2>&1 < /dev/null &); sleep 20; tail -n2 agent.log")


def eval_argv(py: str, packs_dir: Path, pack: str, tag: str, out: Path, baseline: Path | None, limit: int = 0) -> list[str]:
    argv = [py, str(HERE / "night_eval.py"), "--endpoint", f"http://127.0.0.1:{PORT}", "--pack", str(packs_dir / pack), "--tag", tag, "--out", str(out)]
    argv += ["--baseline", str(baseline)] if baseline else []
    return argv + (["--limit", str(limit)] if limit else [])


def size_stage_timeout_min(
    pack: str,
    *,
    measured_sec_per_item: dict[str, float] | None = None,
    n_items: int | None = None,
    safety_factor: float = 1.6,
) -> float:
    """Compute per-cell evaluation timeout in minutes from measured per-item latency (HM-S14-04 / #168).

    In NIGHT-MOE-1, interference cells had a hardcoded 12-minute (720 s) timeout and `eval_maths` timed out for three
    adapters (`cyber`, `exoplanet`, `stars`). Sizing from measured latency with a safety factor (floored at `EVAL_MIN[pack]`)
    eliminates `TIMEOUT` cells across the full pack-by-adapter grid.
    """
    floor_min = float(EVAL_MIN.get(pack, 20))
    if measured_sec_per_item and pack in measured_sec_per_item and n_items is not None and n_items > 0:
        est_min = (float(measured_sec_per_item[pack]) * int(n_items) * float(safety_factor)) / 60.0
        return max(floor_min, round(est_min, 1))
    return floor_min


def build_plan(
    mode: str,
    subjects: list[str],
    packs_dir: Path,
    out: Path,
    py: str,
    interference: bool = False,
    *,
    full_interference_grid: bool = False,
    measured_sec_per_item: dict[str, float] | None = None,
) -> Plan:
    p = Plan()
    if mode == "baselines":
        p.add("serve_base", vm(serve_cmd(None)), SERVE_MIN, must_succeed=True)
        for pack in EVAL_PACKS:
            p.add(f"eval_base_{pack}", eval_argv(py, packs_dir, pack, f"base_{pack}", out / "base", None), EVAL_MIN[pack], f"base on {pack}")
        p.add("stop_server", vm(STOP_SERVER), 2)
        return p
    p.add("free_gpu_check", vm("[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 1500 ] && echo EMPTY || (echo BUSY; exit 1)"), 2, must_succeed=True)
    for s in subjects:
        c = SUBJECTS[s]
        pk = packs_dir / s
        p.add(f"{s}:mkdir", vm(f"mkdir -p {D}/packs/{s} {D}/adapters"), 1, must_succeed=True)
        p.add(f"{s}:push", scp([pk / "train.jsonl", pk / "train_shuffled.jsonl"], f"{D}/packs/{s}"), 5, must_succeed=True)
        p.add(f"{s}:train", vm(train_cmd(f"packs/{s}/train.jsonl", f"adapters/{s}", c["epochs"], c["train_cap_h"])), c["train_cap_h"] * 60 + STAGE_SLACK_MIN + 6, must_succeed=True)
        p.add(f"{s}:train_control", vm(train_cmd(f"packs/{s}/train_shuffled.jsonl", f"adapters/{s}_ctrl", 1, CONTROL_CAP_H)), CONTROL_CAP_H * 60 + STAGE_SLACK_MIN + 6, "shuffled-label negative control")
        for tag, ad in ((s, f"adapters/{s}"), (f"{s}_ctrl", f"adapters/{s}_ctrl")):
            p.add(f"{tag}:convert", vm(convert_cmd(ad)), CONVERT_MIN, must_succeed=(tag == s))
            p.add(f"{tag}:serve", vm(serve_cmd(f"{D}/{ad}.gguf")), SERVE_MIN, must_succeed=False)
            for pack in (c["eval_packs"] if tag == s else [s]):
                base = out / "base" / f"base_{pack}_journal.jsonl"
                lim = 0 if full_interference_grid else (CONTROL_LIMIT.get(pack, 0) if tag != s else 0)
                t_min = size_stage_timeout_min(pack, measured_sec_per_item=measured_sec_per_item) if full_interference_grid else EVAL_MIN[pack] * (0.5 if lim else 1)
                p.add(f"{tag}:eval_{pack}", eval_argv(py, packs_dir, pack, f"{tag}_{pack}", out / "adapters", base if base.exists() else None, lim), t_min)
            p.add(f"{tag}:stop", vm(STOP_SERVER), 2)
    if interference or full_interference_grid:  # does the maths adapter hurt cyber answers, and so on: every adapter on every OTHER pack
        for s in subjects:
            p.add(f"x_{s}:serve", vm(serve_cmd(f"{D}/adapters/{s}.gguf")), SERVE_MIN)
            for pack in (x for x in EVAL_PACKS if x != s and x not in SUBJECTS[s]["eval_packs"]):
                base = out / "base" / f"base_{pack}_journal.jsonl"
                lim = 0 if full_interference_grid else INTERFERENCE_LIMIT
                t_min = size_stage_timeout_min(pack, measured_sec_per_item=measured_sec_per_item) if full_interference_grid else 12
                p.add(f"x_{s}:eval_{pack}", eval_argv(py, packs_dir, pack, f"x_{s}_{pack}", out / "interference", base if base.exists() else None, lim), t_min)
            p.add(f"x_{s}:stop", vm(STOP_SERVER), 2)
    p.add("restore_producer", vm(RESTORE), 3, "ALWAYS runs: the router has no live producer until this succeeds")
    return p


def preconditions(packs_dir: Path, subjects: list[str], out: Path) -> list[str]:
    problems = []
    for s in subjects + [x for x in EVAL_PACKS if x not in subjects]:
        for f in ("manifest.json", "eval.jsonl") + (("train.jsonl", "train_shuffled.jsonl") if s in SUBJECTS and s in subjects else ()):
            if not (packs_dir / s / f).exists():
                problems.append(f"missing {packs_dir / s / f}")
    for pack in subjects:
        n = sum(1 for _ in (packs_dir / pack / "train.jsonl").open()) if (packs_dir / pack / "train.jsonl").exists() else 0
        if n < 50:
            problems.append(f"{pack}: only {n} training rows (the trainer needs 50)")
        elif n != SUBJECTS[pack]["pairs"]:
            problems.append(f"{pack}: pack has {n} training rows but the plan says {SUBJECTS[pack]['pairs']}; rebuild the pack or edit SUBJECTS")
    return problems


def run_stage(st: Stage) -> tuple[str, str]:
    t0 = time.time()
    try:
        r = subprocess.run(st.argv, capture_output=True, text=True, timeout=st.timeout_min * 60)
        status = "OK" if r.returncode == 0 else f"FAILED({r.returncode})"
        if r.returncode == 4 and "e17_train.py" in " ".join(st.argv):  # the trainer hit its own time cap and saved the adapter: usable, but flagged
            status = "OK_TRUNCATED"
        tail = (r.stdout + r.stderr)[-600:]
    except subprocess.TimeoutExpired as exc:
        status, tail = "TIMEOUT", ((exc.stdout or b"")[-300:].decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else "")
    return status, f"{round(time.time() - t0)} s | {tail.strip()[-400:]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--baselines", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--packs", required=True, help="directory written by night_pack_build.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", default=",".join(SUBJECTS))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--interference", action="store_true", help="add the cross-subject interference stage after the subjects")
    ap.add_argument("--deadline-h", type=float, default=10.0, help="hard stop for new stages; the restore still runs")
    args = ap.parse_args()
    subjects = [s for s in args.subjects.split(",") if s]
    unknown = [s for s in subjects if s not in SUBJECTS]
    if unknown:
        raise SystemExit(f"unknown subject(s) {unknown}; choose from {list(SUBJECTS)}")
    packs, out = Path(args.packs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mode = "baselines" if args.baselines else "run"
    plan = build_plan(mode, subjects, packs, out, args.python, args.interference)
    problems = preconditions(packs, subjects, out)
    total_budget = sum(s.timeout_min for s in plan.stages) / 60
    if args.dry_run:
        for s in plan.stages:
            print(f"{s.timeout_min:6.1f} min  {s.name:28s} {s.note}")
        print(f"\nupper bound if every stage used its whole budget: {total_budget:.1f} h (deadline {args.deadline_h} h; stages past the deadline are skipped, the restore still runs)")
        print("preconditions:", "OK" if not problems else problems)
        return 0 if not problems else 2
    if problems:
        print("BLOCKED:", problems)
        return 2
    tunnel = open_tunnel()
    time.sleep(10)
    try:
        return execute(plan, args, out)
    finally:
        tunnel.terminate()


def execute(plan: Plan, args: argparse.Namespace, out: Path) -> int:
    state_path = out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    t_start, skipped_subject = time.time(), ""
    for st in plan.stages:
        subject = st.name.split(":")[0]
        if st.name in state and state[st.name]["status"].startswith("OK"):
            continue
        if st.name != "restore_producer" and (time.time() - t_start) / 3600 > args.deadline_h:
            state[st.name] = {"status": "SKIPPED_DEADLINE"}
            continue
        if skipped_subject and subject.split("_ctrl")[0] == skipped_subject and st.name != "restore_producer":
            state[st.name] = {"status": "SKIPPED_AFTER_FAILURE"}
            continue
        status, info = run_stage(st)
        state[st.name] = {"status": status, "info": info, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        state_path.write_text(json.dumps(state, indent=1))
        print(f"{st.name}: {status} ({info[:80]})", flush=True)
        if not status.startswith("OK") and st.must_succeed and st.name != "restore_producer":
            skipped_subject = subject.split("_ctrl")[0]
    state_path.write_text(json.dumps(state, indent=1))
    return 0 if state.get("restore_producer", {}).get("status") == "OK" else 4


if __name__ == "__main__":
    sys.exit(main())
