# Running HiveMind work on a real BOINC client (GPU)

HiveMind's `hivemind.boinc` package is a *modified-BOINC* coordinator (Python). This page documents the first
integration with the **stock BOINC client**: `hivemind.boinc.real_client`, tested on this project's Tesla T4.

## What exists

| Piece | File | Role |
| --- | --- | --- |
| GPU science app | `src/hivemind/boinc/real_client/app.py` | Seeded, deterministic mini DiLoCo inner loop (low-rank adapter). Atomic checkpoints, `boinc_finish_called` file, `--inject-fault` red-team flag. |
| Validator bridge | `…/bridge.py` | Replays the workunit on CPU and GPU, compares adapter states with the existing Q16 verifier (`validator.py`; exact within a class, ≤ 1 unit across classes), credits via `verified_credit` (cobblestones). |
| Local project server | `…/project_server.py` | Scheduler RPC, upload handler and download endpoints in the formats of the BOINC sources. **Loopback only**. |
| Client setup | `…/client_setup.py` | Private data dir, `app_info.xml` (anonymous platform, `<coproc>` NVIDIA), `cc_config.xml`, account file. |
| Runner | `scripts/boinc_real_e2e.py` | The end-to-end test below. |

## How to reproduce

```bash
# stock client, unpacked (no systemd service, no daemon on the GPU)
apt-get download boinc-client libboinc7
dpkg -x boinc-client_*.deb ROOT && dpkg -x libboinc7_*.deb ROOT
python3 scripts/boinc_real_e2e.py --boinc-root ROOT --work /some/disk/run
```

## Result (2026-10-25, run 3; all eight criteria of `artifacts/boinc_real/ACCEPTANCE.md`, written before the run)

* Client `BOINC 7.18.1` logged `CUDA: NVIDIA GPU 0: Tesla T4 (… 8141 GFLOPS peak)` and `Found app_info.xml; using anonymous platform`; its scheduler request carried the T4 in `<coprocs>`.
* It downloaded the workunit over HTTP, ran the app on the T4, uploaded through the upload-handler protocol, and reported `exit_status 0`.
* The honest result was **validated** (GPU replay bit-exact, CPU replay within 1 Q16 unit) and credited 0.384 cobblestones.
* The forged result (app told to report a perturbed adapter) differed from the CPU replay by 656 Q16 units and was **rejected with 0 credit**.
* Preemption: the client and its app were `SIGKILL`ed mid-task; after restart the task **resumed from step 700** and its final digest is **bit-identical** to an uninterrupted run (credit 2.88 cobblestones).
* Afterwards no BOINC process and no process on the GPU remained.

Evidence: `artifacts/boinc_real/e2e_report.json`, `client_stdout.log` (verbatim client log).

## Findings from the development runs (kept, not hidden)

1. **Run 1 looped.** The client restarted the task endlessly: `Task … exited with zero status but no 'finished' file`. An app must leave `boinc_finish_called` (what `boinc_finish()` writes in `api/boinc_api.cpp`) even without linking libboinc. Fixed in `app.py`; the log of the looping run is `dev_run1_premature_exit_loop.log`.
2. **Run 2 scored A8 false** because the runner's own CUDA context (the validator replays on the GPU in-process) was counted as a leftover. The check now excludes the runner's PID; the client left nothing behind in either run.
3. The client passes `--device N` to GPU apps; the app ignores it (single GPU).

## Limits — read before citing

* This is **not the production BOINC server** (no MySQL/Apache/daemons, no code signing, no `script_validator`). The app reaches the client through the *anonymous platform* mechanism, and the scheduler/upload formats were derived from the BOINC sources (commit `20798b8`), not tested against a real server. A real-server run is the next step (`sched/script_validator.cpp` and `script_assimilator.cpp` exist and can call this Python bridge).
* Single machine, single GPU, one client, three workunits; the test shows the integration works, not scalability, throughput or fairness.
* The app is a small synthetic adapter-training task, not an LLM workload; the credit numbers are for the toy task.
* The fault control is a flag in the app (a volunteer cheating is simulated); real adversaries are out of scope here.
* A stock client does not need, and this test does not use, the Python coordinator in `coordinator_api.py`.
* PTS/cobblestones here are non-monetary accounting units, as in the rest of the project.
