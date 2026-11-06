# NIGHT-INTENSE-1 feasibility probes: measured results, 2026-10-28 (23:01 to 23:07 Paris)

Source: the queue's own result files (`arm_A*.json`, `summary.json`) in `night_probe/20261028T2201*`, `T2202*` and `T2205*` on the data disk. Every number below is read from those files by `scripts/night_probe_queue.py` and summarised by `night_probe_report.py`.

**What this is.** A memory and speed feasibility measurement on random token ids. It is **not** a quality measurement, and no held-out loss was computed. No training step exists yet to test quality.

## Arms

| arm | model | rank | seq | micro-batch | checkpointing | status | estimate alloc / process (MiB) | measured alloc (MiB) | error vs alloc | probe peak per nvidia-smi (MiB) | s/step | tok/s | max temp (C) | max power (W) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | ibm-granite/granite-3.1-1b-a400m-base | 8 | None | 2 | False | OK | 2843 / 3150 | 3409 | 19.9% | 3710 | 0.964 | 266 | 77 | 68.08 |
| A1 | ibm-granite/granite-3.1-1b-a400m-base | 32 | None | 2 | False | OK | 3285 / 3592 | 4567 | 39.0% | 4916 | 1.149 | 445 | 81 | 97.29 |
| A2 | ibm-granite/granite-3.1-3b-a800m-base | 16 | None | 2 | True | OK | 6407 / 6714 | 6523 | 1.8% | 6964 | 3.183 | 161 | 82 | 84.41 |
| A3 | Qwen/Qwen3-4B-Instruct-2507 | 16 | None | 1 | True | OK | 3615 / 3922 | 3567 | -1.3% | 4146 | 1.796 | 143 | 82 | 110.35 |

## What the numbers say

- **All four arms ran to a conclusive OK status.** None was blocked, aborted or out of memory. The queue stopped nothing, and HOLD was created and removed by the queue (`hold_created_by_queue: true`, `hold_preexisting: false`).
- **The estimator was not accurate on the small Granite-1B arms.** A0 was 19.9 percent above the estimate and A1 39.0 percent above. A2 (1.8 percent) and A3 (-1.3 percent) were close. The earlier calibration error bound of 17.6 percent is therefore too optimistic at low rank and long sequence, and the estimator should not be used for the Granite-1B recipes until it is recalibrated.
- **Granite-1B rank 32 at sequence 256 does not fit the 4,300 MiB night budget.** A1 measured 4,567 MiB allocated, so under the governor's floor it would not run at night. It only ran because the probe was capped at 9,500 MiB. Rank 8 (A0, 3,409 MiB) is the only Granite-1B recipe measured under 4,300 MiB at this sequence length.
- **Granite-3B with checkpointing (A2) and Qwen3-4B with checkpointing (A3) both fit the 10,000 MiB big-profile budget** and were close to the estimate. Neither fits the 4,300 MiB night budget.
- **Temperature.** Peaks were 77 to 82 C, below the 83 C limit used by the queue, but within 1 C of it for A2 and A3, so a longer run could reach the limit.
- **Power readings need checking before use.** A3 reported 110 W and A2 84 W. The T4 is rated at 70 W, so these values are not plausible as card power. They may be a sensor or unit issue. They are recorded as reported and must not be used for an energy claim.

## Not tested

- Quality (held-out loss, bits per byte): **Not tested.**
- Throughput over a full night, thermal behaviour over hours, and interaction with the live bee server under user traffic: **Not tested.**
- The gradient-checkpointing path through the real trainer step (the probe measures a forward and backward pass at the recorded settings only): **Not tested.**
- The Qwen3-4B tokenizer is not in the local cache (only the config, weights and index are). A real training step would need it: **Not tested.**

## Decision this supports (not taken here)

Under the governor's current 4,300 MiB night floor, no candidate in this set fits except a rank-8 Granite-1B recipe. A night with a larger base needs the owner's decision on the budget (Option B in `NIGHT_INTENSE_PLAN.md`), a recalibrated estimator, and a quality measurement before any training run is planned.
