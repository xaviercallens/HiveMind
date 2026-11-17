# E-STP-1b: Crowd-v1 with a calibrated learning rate (pre-registration, written 2026-11-16 ~19:45 Paris)

**Status:** a proposal. It runs only with the owner's approval, and only after E-STP-2 has finished on the T4.
It is written **before** any E-STP-2 (Qwen3-0.6B) result and before E-STP-1's final verdict.

## Why (facts seen so far, stated so the motivation is on record)
- E-STP-1 (frozen recipe from `docs/HIVEMIND_JEPA_LLM_PROPOSAL.md` §4) on Crowd-v1:
  - **8 of 18 runs are done, and all score 0.0 % exact match**: SFT and STP, fractions 1, ¼ and 1/16, seeds 0–1.
    Exact-prefix accuracy is 0.6–1.3 %.
  - The training log of `crowd_f1_sft_s0` shows the answer cross-entropy going **8.17 → about 3.0 nats** over 2,000
    steps at lr 2e-5. The predictions start like the right regex but never stop.
  - The end token is correct: id 151643 is `<|endoftext|>` in Crowd-v1's Qwen-compatible tokenizer.
- **Reading:** the paper's lr (2e-5, chosen for fully trained models) underfits Crowd-v1, an early pre-training
  checkpoint. Under Amendment A1, every Crowd-v1 fraction will then be "uninformative", so E-STP-1 cannot test STP on
  Crowd-v1 at all.
- **E-STP-1 is not changed or re-scored.** Its verdict stands as computed by `estp_criteria.py`. E-STP-1b is a new,
  separately labelled experiment.

## Design

### Phase 1: learning-rate calibration (SFT only; no STP; nothing compared)
- Model `crowd`, arm `sft`, fraction 1, seed 0. Everything else as E-STP-1: epochs 4, bs 8, accum 2, rank 64.
- **lr ∈ {1e-4, 3e-4, 1e-3}** (2e-5 is already measured by E-STP-1). Run with `--skip-eval`.
- **Selection rule, with no test-set access:**
  - candidates are the lrs whose run is `done`, never stopped for non-finite loss, and whose mean training CE over the
    last 200 steps is below its first-200-step mean;
  - pick the one with the **lowest mean training CE over the last 200 steps**;
  - if no candidate exists, stop: report "Crowd-v1 does not fit NL-RX-SYNTH with LoRA r64 in 4 epochs at any probed
    lr", and do not run Phase 2.
- Cost: 3 runs × ~30 min.

### Phase 2: E-STP-1's design at the selected lr
- 18 runs: fractions 1, ¼, 1/16 × arms SFT and STP (λ = 0.02) × seeds 0–2, at the Phase 1 lr. All else as E-STP-1.
- **Results directory:** `results/estp1b/`, never mixed with `results/estp/`.
- **Analysis:** `scripts/crowdgpt_t4/estp_criteria.py --results results/estp1b` with model `crowd`. Amendment A1's rules
  apply **unchanged**:
  - 5 % SFT floor per fraction;
  - a gain > 2 × pooled sd at the primary fraction 1/16, or at two interpretable fractions;
  - guard rail: STP far-loss ≤ SFT + 0.05 nat;
  - `health()` reported as not evaluable.
- Cost: about 9 hours (one T4 night).

## What E-STP-1b can and cannot say
- A success says STP helps Crowd-v1 **at a learning rate calibrated on SFT**. It does not say the paper's recipe
  transfers.
- A failure, or an uninformative result, closes the Crowd-v1 branch of the program, as `CITIZEN_AI_JEPA_PROGRAM.md`
  requires.
- The M4 gate (OpenTrainer `stp_lora`) still also depends on E-STP-2, read jointly as P2 specifies.

## Running rules
These are the same as E-STP-1:
- through the frozen runner's code, with the T4 guard;
- free VRAM ≥ 4,300 MiB;
- other GPU jobs on the T4 go through `scripts/t4_gpu_lease.py` (#930), or are paused, as decided by the owner;
- no deployment and no GCP spend.
