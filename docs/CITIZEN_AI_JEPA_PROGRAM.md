# Citizen-AI JEPA-LLM program: goal, milestones, gates (written 2026-11-15, before any E-STP result)

**Long-term goal (the owner's):** a citizen AI served and trained by about **1,000 consumer GPU nodes**, built on
data- and compute-efficient JEPA-style language models, through HiveMind's OpenTrainer (training) and OpenRouter
(serving).

**What a loop can and cannot do.** The loop advances the *program* one gated milestone at a time. It does not recruit
volunteers, spend money, deploy GCP resources, or wire anything into production without a passed gate. Recruiting
1,000 nodes is a human, community effort; the loop's job is to make each step evidence-based and ready.

## Milestones and gates

| # | Milestone | Done when | Gate to go further |
|---|---|---|---|
| M0 | Proposal: paper (`paper/hivemind_jepa_evolution/`), design docs, library (#920), gated runner | PRs open, runner waiting | — |
| M1 | **E-STP-1** (Crowd-v1, data fractions 1, 1/4, 1/16; 3 seeds) | all 18 runs `done`, criteria computed as pre-registered | STP > SFT by 2 × pooled sd at ≥ 1 fraction, guard rail and health respected |
| M2 | **E-STP-2** (Qwen3-0.6B, same design) | 18 runs done | read with M1 (model vs recipe) |
| M3 | **E-STP-3** (raw Python-code text, document-aware STP) | protocol written *before* running, then runs done | held-out code loss improves, no guard failure |
| M4 | **OpenTrainer task type `stp_lora`** (signed `JepaSpec`, replay-verified) as a PR | code + CPU tests + replay-equality test | only if M1 or M2 passed and M3 shows no harm |
| M5 | **OpenRouter catalog entries** for the new model types (STP-tuned experts, JEPA embedding/router model) as a PR, catalog files only | PR with entries marked `experimental`, no deployment | only after a model from M4 beats its SFT twin on held-out evals |
| M6 | **Scale plan to 1,000 consumer GPUs**: paper section updated with measured per-node cost, bandwidth and validation numbers from M1–M5 | paper v2 | owner decision |

## Rules for every tick

- **T4 courtesy:**
  - experiments run only through `run_estp.py` (gated: not before Mon 16 Nov 07:00 Paris; governor `held` + `HOLD`
    present; no foreign GPU process > 1 GB);
  - free VRAM ≥ 4,300 MiB always;
  - never stop other sessions' processes; the CrowdGPT client belongs to the "v3.6.0 experimentation t4" session.
- **No GCP spend, no deployment.** Catalog or task PRs are code and docs only; the owner merges and deploys.
- **Pre-registration:** the criteria in `docs/HIVEMIND_JEPA_LLM_PROPOSAL.md` §4 are final. Post-hoc analyses are
  labeled as such.
- **Honest reporting:** a negative result ends that branch of the program. Report it, and switch effort to the data-mix
  lever (#907 P0).

## Stop condition for the loop

Stop when M3 is complete and M4's gate has been decided (passed: open the M4 PR and stop; failed: write the negative
result into the paper and stop), or when the owner says stop. M5 and M6 are follow-up loops.
