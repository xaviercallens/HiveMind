# NIGHT-INTENSE-1 guarded probe queue: runbook

Tool: `scripts/night_probe_queue.py` (standard library, Python 3.10), a thin queue around `scripts/t4_model_probe.py`. Pre-registration: `artifacts/experiments/NIGHT_INTENSE_1_prereg.json`; plan: `docs/NIGHT_INTENSE_PLAN.md` sections 2, 3, 6, 7. Governor policy is unchanged (Option A).

**What it is.** A memory and speed FEASIBILITY probe of the four pre-registered arms on random token ids (3 segments of 10 optimiser steps each). It reports whether an arm loads and trains one short run under a VRAM cap, the torch peak, seconds per step and tokens per second.
**What it is not.** It does not measure held-out quality (no bits-per-byte, no data), does not train a block, and does not decide WIN / NO_GAIN. The 3.0 h training run of the prereg is a separate later step and is **NOT AVAILABLE**. Not tested on the GPU: this tool has never been run against the real card by the change that added it.

## The arms (read from the prereg file, nothing hard-coded)

| Arm | Model | rank / alpha | seq | micro-batch | checkpointing | LoRA targets | Tier cap (MiB) | Estimator alloc / process peak (MiB) |
|---|---|---|---|---|---|---|---|---|
| A0 | ibm-granite/granite-3.1-1b-a400m-base | 8 / 16 | 128 | 2 | no | q,k,v,o | 3950 (N0) | 2843 / 3150 |
| A1 | ibm-granite/granite-3.1-1b-a400m-base | 32 / 64 | 256 | 2 | no | q,k,v,o | 3950 (N0) | 3285 / 3592 |
| A2 | ibm-granite/granite-3.1-3b-a800m-base | 16 / 32 | 256 | 2 | yes | q,k,v,o | 9650 (N1) | 6407 / 6714 |
| A3 | Qwen/Qwen3-4B-Instruct-2507 | 16 / 32 | 256 | 1 | yes | q,k,v,o,gate,up,down | 9650 (N1) | 3615 / 3922 |

Per-arm cap = `min(--max-cap-mib (9500), free VRAM now - 600 MiB)`. The "Tier cap" column is the allocator cap that reproduces the tier budget (governor's 4300 MiB rule minus the ~350 MiB CUDA context for N0 = 3950; 10,000 MiB minus context for N1 = 9650); it is applied only with `--tier-cap`. By default an N0 arm therefore runs under the larger cap and the result says whether it stayed within its budget: each `arm_*.json` carries `tier_budget_mib` (4300 or 10000) and `fits_tier_budget` (true if the probe's own nvidia-smi memory peak, `measured.nvidia_smi_probe_peak_used_mib`, is within the budget; false if it is above, or if it ran out of memory although the cap allowed the whole budget; null if not determined). Use `--tier-cap` to run the pass/fail test under the policy's own limit instead (an OOM then gives no peak number). Estimator figures have a calibration error up to about 20 % and are biased low for the Granite MoE; they are predictions, not measurements. Known caveat (`docs/T4_NIGHT_MODEL_DECISION.md`): with transformers 5.17 the fused MoE experts of A2 stay 16-bit (about 6.0 GiB of weights), and an earlier probe of that model ran out of memory while loading under a 3,950 MiB cap; whether it loads under 9,650 MiB is exactly what A2 tests. Not tested.

## Before you start (the owner's steps)

1. Be inside 23:00 to 06:30 Europe/Paris, or the queue refuses to start (exit 2).
1b. **BOINC wait.** Once its HOLD exists the governor only sets BOINC to "never"; a training task that was already running can take 0.8 to 60 s to leave the GPU. Before each arm the queue waits up to 90 s (`--boinc-wait-s`) for BOINC GPU tasks (the governor's `boinc_gpu_pids`) to go. It never kills one: if it is still there the arm is BLOCKED with a reason naming BOINC. The wait is recorded as `preflight.boinc_wait_s`.
2. The queue never unloads anything. **Any resident Ollama model blocks every arm**, including the N0 arms. So for the intended run the coder must already be gone (explicit unload as in plan section 6 step 2, or its keep-alive expired), then the 30 quiet minutes. The queue does not do this for you.
3. Only the allow-listed GPU process may hold GPU memory (default name `llama-server`, the bee server). Anything else blocks (`--allow-gpu NAME`, repeatable, to change).
4. Dry-run first (read-only, see below).

## Commands

Interpreter: the earlier probes ran on the node's BOINC virtualenv (`/mnt/disks/disk-socrateai-local-1/boinc/venv/bin/python`, the interpreter `hivemind-t4-workgen.service` uses). That the venv contains `peft`, `bitsandbytes` and `transformers` is **Not verified** here (nothing was run); check with `... -c "import peft, bitsandbytes, transformers"` before the night. Nothing is installed by this tool. The queue itself needs only the standard library; pass the venv interpreter to the probe subprocess with `--python`.

```bash
cd /path/to/HiveMind      # the checkout that contains scripts/
PY=/mnt/disks/disk-socrateai-local-1/boinc/venv/bin/python

python3 scripts/night_probe_queue.py --list                         # arm table, no GPU, no network
python3 scripts/night_probe_queue.py --dry-run --python $PY         # every precondition; creates nothing
python3 scripts/night_probe_queue.py --python $PY --arms A0,A1      # real run, tonight's N0 arms
python3 scripts/night_probe_queue.py --python $PY --arms A2         # N1 arm, only after the coder is unloaded
```

Results go to `/mnt/disks/disk-socrateai-local-1/night_probe/<UTC timestamp>/` (`--out-dir` to choose; pointing at an existing directory resumes it: arms already OK, OOM or CHECKPOINTING_UNSUPPORTED are skipped, other statuses are retried). Files: `arm_<ID>.json`, `arm_<ID>.probe.stdout.log`, `arm_<ID>.probe.stderr.log`, `summary.json`.

`--dry-run` reads the clock, the HOLD/lock state, nginx `stub_status` and Ollama `/api/ps` (GET only) and runs the same read-only `nvidia-smi` queries the governor runs every second. It writes no file, creates no directory, no lock and no HOLD, and starts no probe. Exit 0 if everything would pass, 3 if something would stop the run.

Other options: `--tier-cap`, `--boinc-wait-s`, `--max-cap-mib`, `--min-cap-mib` (1500: below it the arm is BLOCKED), `--max-gpu-temp-c` (83), `--arm-timeout-s` (1800), `--ignore-window` (supervised tests only; recorded in the results), `--hold-file`, `--lock-file`, `--status-url`, `--ollama-url`.

## Statuses

| Status | Meaning |
|---|---|
| OK | The probe finished: load, 3 segments, JSON with torch peaks and speed. Says the arm fits that cap on random tokens; nothing about quality. |
| OOM | Out of memory under the cap (inside the probe, or the probe crashed with a CUDA out-of-memory message). A conclusive result: the arm does not fit that cap. |
| CHECKPOINTING_UNSUPPORTED | The model or loader could not enable gradient checkpointing (or did not report it as enabled). Reported, never ignored. |
| BLOCKED | A precondition failed, nothing was started: resident Ollama model, non-allow-listed GPU process, GPU utilisation >= 10 %, nginx request in flight, nginx or Ollama unreadable (fails closed; "connection refused" from Ollama counts as nothing resident), or cap below the minimum. The queue stops after a BLOCKED arm. |
| ABORTED_BY_TRAFFIC | While the probe ran: a user request in flight on nginx, a new GPU process (not the probe or its children, not present at the start, not allow-listed), or Ollama loading a model. The probe's whole process group was sent SIGTERM, then SIGKILL after 2 s. The queue stops. |
| TOO_HOT | GPU temperature above the limit (83 C) before or during the arm. The queue stops. |
| TIMEOUT | The arm ran longer than `--arm-timeout-s`. |
| ERROR | Any other failure (unsupported architecture, crash). |
| INTERRUPTED | The queue received SIGTERM or SIGINT. |
| OUTSIDE_WINDOW | The window closed before or while the arm ran (a running arm is stopped). The queue stops. |

Exit codes: 0 all selected arms ended OK, OOM or CHECKPOINTING_UNSUPPORTED; 1 some other status; 2 outside the window; 4 another queue holds the lock; 130 interrupted.

## How a stop is detected

A monitor thread ticks every second: one `nvidia-smi` sample (memory, temperature, power; the maxima are recorded), one GET of nginx `stub_status` and one GET of Ollama `/api/ps`. Request in flight = `writing - 1` from `stub_status` (this poll itself excluded), exactly the governor's definition. Two GPU signals mirror the governor's other-GPU-busy check for a running task: (a) any compute process that is not the probe or one of its descendants, was not present at the start and is not allow-listed is treated as traffic; (b) `nvidia-smi pmon -c 1 -s u` parsed with the governor's `parse_pmon`: any process other than the probe's own with SM utilisation >= 5 % (the governor's `max(10 // 2, 5)`) is traffic. (b) is what notices a direct call to the allow-listed, already resident bee server, which skips nginx. `pmon` takes about a second, so a tick can last about 2 s and the worst case from request to dead probe is about 4 s (not measured on the card). Stop = SIGTERM to the probe's process group, SIGKILL after 2 s; the polling interval is 1 s. Measured only with fakes and with a real throwaway process tree in the tests (child and grandchild dead, well under 3 s); on the real card: Not tested.

## Safety contract

1. **Window.** Starts only inside 23:00-06:30 Europe/Paris (zoneinfo, DST-safe); re-checked before each arm and every second during an arm.
2. **HOLD.** Creates the governor's HOLD file (`/mnt/disks/disk-socrateai-local-1/hivemind_telemetry/HOLD`) for the whole run and removes it in `finally`, `atexit` and on SIGTERM/SIGINT. A HOLD that existed before the run is the owner's: it is not removed, and the results say `hold_preexisting: true`. SIGKILL of the queue itself cannot be caught and would leave its HOLD in place: the governor then stays paused (fail-safe), see rollback.
3. **Never evict.** Ollama is only ever read (`GET /api/ps`). Nothing is unloaded, no process is killed except the queue's own probe.
4. **Traffic guard.** As above.
5. **Sampler and temperature limit.** As above.
6. **One queue.** `flock` on `/mnt/disks/disk-socrateai-local-1/night_probe/queue.lock`.
7. Governor policy, nginx, systemd, `/opt`, `/etc` and the BOINC client are not touched.

## Stop it safely

`kill -TERM <queue pid>` (or Ctrl-C). The probe group is stopped (SIGTERM, SIGKILL after 2 s), the current arm is recorded INTERRUPTED, HOLD (if the queue made it) and the lock are released, and `summary.json` is written. Do not `kill -9` the queue.

## Afterwards

1. HOLD gone, unless it was yours: `ls /mnt/disks/disk-socrateai-local-1/hivemind_telemetry/HOLD` must say "No such file" and `summary.json` should show `hold_removed_by_queue: true` (or `hold_preexisting: true`).
2. GPU free of the probe: `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader`.
3. Governor state: the state file `.../hivemind_telemetry/governor_state.json` (the governor decides to train again only after its 1800 s quiet rule; this tool never touches it).
4. Coder reload time, if you unloaded it: the first request after the stop is cold (52 to 120 s measured earlier, plan M6). Time it before real users arrive.
5. Read `summary.json` and each `arm_*.json`: `measured.torch_peak_allocated_mib` against `estimate.alloc_peak_mib` (`estimate_vs_measured`), `measured.nvidia_smi_peak_used_mib`, `max_temp_c`, `max_power_w`, `seconds_per_step`, `tokens_per_second`. If the estimator could not be loaded or the model's `config.json` is not in the local cache the estimate says `available: false` with the reason.

## Rollback

Delete nothing but this tool's own files: `rm` a stray HOLD only if `summary.json` or the HOLD file's content (`night_probe_queue pid N`) shows the queue created it and that pid is dead; remove a stale lock by checking no queue runs (the lock is a `flock`, it releases when the process ends, so the file itself is harmless). To undo the change entirely, revert the PR; `scripts/t4_model_probe.py` stays backward compatible (default flags give the historic spec, rank 8, alpha 16).

## Not tested

- Any run on the real GPU, with the real nginx, Ollama or bee server: Not tested.
- Gradient checkpointing on the PEFT/bitsandbytes model through `replay_inner_segment` (the probe reports CHECKPOINTING_UNSUPPORTED if it fails): Not tested.
- `--targets` for A3 (the probe sets `moe_app.LORA_TARGETS` before loading): Not tested on a real model.
- The probe's peak figures versus the estimator for the new arms: Not tested.
- Dependencies of the BOINC venv: Not verified.
- Quality, held-out loss, the 3.0 h block, TTFT measurements: not part of this tool; NOT AVAILABLE.
