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
- **E-STP-1:** tested nothing about STP, because Crowd-v1 did not learn the task at lr 2e-5 (0 % exact match in every run). E-STP-1b (`docs/E_STP_1B_PREREGISTRATION.md`) repeats it at a calibrated learning rate. It is running and will be reported separately.

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
