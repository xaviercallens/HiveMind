# NIGHT-MOE-3: a routed three-expert MoE on the RTX 2070 (8 GB)

Status: **draft, not preregistered.** Freeze it as `artifacts/experiments/NIGHT_MOE_3_prereg.json` before the first training step.
Evidence base: `docs/NIGHT_MOE_RESULTS.md` and `docs/NIGHT_MOE_2_PLAN.md` (NIGHT-MOE-1, one night on one T4). Figures below are from those files; anything marked **estimate** or **unmeasured** has not been run.
Owner decision (2026-11-04): the maths expert is worth the GPU time despite the NIGHT-MOE-1 null.

## 1. Why this design

- Only label tasks with a clean signal trained well (stars 0.489 -> 0.893, adapter beat its shuffled control 193 to 7).
- Maths trained on terse gold text got worse (0.785 -> 0.715). Cyber text with unverified targets was a null.
- "MoE" in NIGHT-MOE-1 meant a set of adapters; no router or gate was trained or tested. This run tests the routed system.
- The 7B QLoRA run needed 10.23 GiB on the T4, so it does not fit 8 GB. The base here is Qwen2.5-1.5B, then 3B, both 4-bit.

## 2. Experts (each has a verifier)

| Expert | Data | Note |
|---|---|---|
| stars | label task, split by star | the one result that cleared its control; replication on a new GPU and base |
| exoplanet | split by star, plus a logistic-regression baseline | ruled VOID in NIGHT-MOE-1 by rule, adapter beat control 309 to 17 post hoc; rerun with stricter controls (owner decision pending on inclusion) |
| maths | the base model's own solutions, kept only when they match the GSM8K answer | no gold text; most likely to show no gain, and a null is reported as one |

Cyber is dropped unless a source with verifiable targets is found.

## 3. Procedure

1. 1-step smoke test on each base; record the measured peak VRAM before any training.
2. Preregistration file frozen first (hypotheses, n, seeds, decision rules, the multiplicity family).
3. Controls for every expert: a shuffled-label control and a cross-domain adapter control. Two seeds for any headline gain. Full packs, no `--limit`, n >= 300 per pack.
4. Router, in order: keyword or classifier router (measure routing accuracy, then end-to-end accuracy); embedding router only if the first loses more than the preregistered margin; a learned gate only if both fall short.

## 4. Hypotheses

- **H1:** each adapter beats its cross-domain-recipe control on its own pack (exact paired McNemar, p < 0.05, positive difference).
- **H2:** the adapter beats the base on items the base parsed; unparsable counts are reported next to accuracy.
- **H3:** the routed system scores at least as well as the best single adapter per pack and loses no more than a preregistered margin on any other pack. This is the test for calling it an MoE.
- **H4:** descriptive only: does 3B raise the effect sizes over 1.5B (one seed each, no claim).

Success is H1 and H3 holding on at least two experts. An expert that fails its control is reported VOID, not dropped. A routed system that loses to the best single adapter is reported as a failed MoE claim.

## 5. Constraints

No use of the T4 governor, the HOLD file or any hivemind unit; no Qwen3.5 training; results are real tool output only; the run is a branch plus PR, merged by the gatekeeper.

## 6. Not claimed

No result exists yet. Whether 3B fits in 8 GB, whether exoplanet has enough data, and disk space on the 2070 are unmeasured.
