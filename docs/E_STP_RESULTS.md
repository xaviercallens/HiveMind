# E-STP-1 and E-STP-2: results (17 November 2026)

**Verdicts, exactly as pre-registered (Amendment A1, 2026-11-15):**

| Experiment | Model | Verdict |
|---|---|---|
| E-STP-1 | Crowd-v1 (1.0 B), LoRA, lr 2e-5 | **uninformative** |
| E-STP-2 | Qwen3-0.6B, LoRA, lr 2e-5 | **no_gain** |
| M4 gate (OpenTrainer `stp_lora`) | | **closed** |

**What this means.**
- **STP:** Semantic Tube Prediction, the auxiliary loss proposed in `docs/HIVEMIND_JEPA_LLM_PROPOSAL.md`, did not improve fine-tuning accuracy at any data fraction for Qwen3-0.6B.
- **E-STP-2:** a negative verdict, so it ends this branch of the program (`docs/CITIZEN_AI_JEPA_PROGRAM.md`). The M4 step, turning on `stp_lora` in OpenTrainer, does not happen.
- **E-STP-1:** tested nothing about STP, because Crowd-v1 did not learn the task at lr 2e-5 (0 % exact match in every run). E-STP-1b (`docs/E_STP_1B_PREREGISTRATION.md`) repeated it at a calibrated learning rate: **fail_guard_rail** (below), which ends the Crowd-v1 branch.

The analysis is `scripts/crowdgpt_t4/estp_criteria.py`. Its output is `artifacts/experiments/E_STP_criteria_2026-11-17.json`, committed unchanged. The per-run receipts are in `artifacts/experiments/E_STP_runs_2026-11-17.json`: 36 runs, all `done`. Predictions, logs and samples are left out to keep the file small.

## E-STP-2 (Qwen3-0.6B): exact-match accuracy, %

3 seeds per arm. SFT is the baseline; STP is SFT plus the STP loss.

| Training fraction | SFT (seeds 0, 1, 2) | STP (seeds 0, 1, 2) | SFT mean | STP mean | STP − SFT | pooled SD | gain > 2 SD |
|---|---|---|---|---|---|---|---|
| 1 | 87.9, 87.4, 87.1 | 88.2, 86.6, 86.7 | 87.47 | 87.17 | −0.30 | 0.70 | no |
| 1/4 | 81.4, 79.9, 80.9 | 81.2, 80.7, 80.8 | 80.73 | 80.90 | +0.17 | 0.57 | no |
| 1/16 (primary) | 61.2, 59.7, 61.3 | 59.7, 61.1, 60.0 | 60.73 | 60.27 | −0.47 | 0.82 | no |

**Guard rail:** STP far-token loss may not rise by more than 0.05 nat. It **passes** at every fraction; STP's far loss is in fact slightly lower:

| Fraction | SFT | STP |
|---|---|---|
| 1 | 2.884 | 2.865 |
| 1/4 | 2.917 | 2.890 |
| 1/16 | 2.904 | 2.868 |

**Health clause:** **not evaluable**. The pre-registration names a `health()` check that the frozen code does not define, so this clause is reported, not silently dropped.

All three fractions are interpretable. No fraction, including the primary 1/16, shows a gain larger than 2 pooled SDs, so the verdict is **no_gain**.

## E-STP-1 (Crowd-v1, 1.0 B)

- **Accuracy:** 0.0 % exact match in all 18 runs (both arms, 3 fractions, 3 seeds).
- **Far-token loss:** 3.27–3.30, identical between the arms to within 0.0002.
- **Verdict:** under A1, every fraction is uninformative, so the verdict is **uninformative**. The cause is underfitting at lr 2e-5, which E-STP-1b addresses.

## Safety

- **VRAM:** every run kept at least **7,319 MiB** free on the T4 (project floor 4,300 MiB). Each Qwen run hit its own 2,700 MiB process cap once at the start of evaluation; the queue waited, resumed the run from its checkpoint and finished it. No guard aborted any run.
- **Isolation:** the runs used the cross-project GPU lease. The AutoevolveAI crons were paused and the Granite producer held while they ran.

## E-STP-1b (Crowd-v1 at a calibrated learning rate), 18 November

**Verdict: fail_guard_rail.** This is a negative result, so it ends the Crowd-v1 branch of the program. The M4 gate stays **closed**:
- this analysis covers Crowd-v1 only, so it prints the M4 gate as "undecided" and E-STP-2 as "incomplete";
- E-STP-2 already returned no_gain, and P2 reads the two jointly.

**Phase 1 (learning-rate calibration):**
- Three SFT probes on the full data, seed 0, training only. Final 200-step training loss (CE):

  | Learning rate | Final CE |
  |---|---|
  | 1e-4 | 2.306 |
  | 3e-4 | 2.178 |
  | 1e-3 | **2.155** |

- 1e-3, the lowest, was selected (`artifacts/experiments/E_STP_1B_lr_selection_2026-11-17.json`).

**Phase 2: exact-match accuracy, %** (3 seeds per arm, lr 1e-3):

| Training fraction | SFT (seeds 0, 1, 2) | STP (seeds 0, 1, 2) | SFT mean | STP mean | STP − SFT | pooled SD | gain > 2 SD |
|---|---|---|---|---|---|---|---|
| 1 | 14.1, 15.0, 16.2 | 14.8, 14.8, 16.4 | 15.10 | 15.33 | +0.23 | 0.99 | no |
| 1/4 | 14.1, 14.2, 14.7 | 14.7, 14.7, 14.8 | 14.33 | 14.73 | +0.40 | 0.23 | no |
| 1/16 (primary) | 10.2, 9.8, 10.5 | 10.5, 9.8, 11.1 | 10.17 | 10.47 | +0.30 | 0.52 | no |

**Far-token loss** (guard rail: STP at most SFT + 0.05 nat):

| Fraction | SFT | STP | Change | Guard rail |
|---|---|---|---|---|
| 1 | 8.793 | 9.257 | +0.46 | **fail** |
| 1/4 | 8.206 | 9.576 | +1.37 | **fail** |
| 1/16 | 12.321 | 12.202 | −0.12 | pass |

**Summary:**
- At the calibrated learning rate, Crowd-v1 learns the task: 10–16 % exact match, against 0 % at 2e-5.
- STP's small accuracy edge stays within noise at every fraction.
- STP raises far-token loss well past the guard rail at two of three fractions.
- `health()` is again not evaluable.

**Files:**
- `artifacts/experiments/E_STP_1B_criteria_2026-11-18.json`: the analysis output, unchanged;
- `artifacts/experiments/E_STP_1B_runs_2026-11-18.json`: 18 phase-2 runs and 3 probes, without predictions, logs or samples.

**Safety:** at least **7,519 MiB** free in every run, and no guard aborted any run.
