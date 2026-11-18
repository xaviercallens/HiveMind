# E-B3-MOE-PAIRED: pre-registration (IP-300)

**Registered 2026-11-18, before any test number was computed.** The frozen spec is
[`E_B3_MOE_PAIRED_PREREGISTRATION.json`](E_B3_MOE_PAIRED_PREREGISTRATION.json). If this page and the JSON disagree, the
JSON wins. Neither file is edited after the run; results go in a separate section at the end of this page.

## Question

Does HiveMind's MoE LoRA add-on lower the next-token loss of the frozen base `Vxtzq/Crowd-v1` on real rows that
training **and** the training ratchet's own evaluation never read? The ratchet reported 5.385 → 5.230 (−2.88 %), but it
measured on the same held-out rows it used to accept or reject rounds, so that number was used for selection and cannot
decide B3.

## Arms (paired, same chunks)

| Arm | What runs |
|---|---|
| base | dense `Vxtzq/Crowd-v1`, revision `b63212a843d662dadbec35f8544b6e0dafa6e72b`, weights sha256 `1f8214ca…2209f` |
| MoE | the same model upcycled in place (`lora_domain_experts`, 8 experts, top-2 `sigma_topk`, LoRA rank 8, 24 layers, 4 router fragments) with `hf_moe_best/crowd_v1_moe_lora_experts_best.safetensors` loaded (sha256 `0778b9ee4279ee6a70f88e5fdbd02237f620499b1daee5ef8b927d2c3af781f8`) |

Newer checkpoints are not used even if they appear. The model is built and loaded with the worker's own code path
(`scripts/gcp_t4_crowd_diloco_worker.py`: `_load_or_build_model`, `load_init_buckets`, `load_moe_trainable_buckets`,
`forward_crowd_v1_checkpointed`).

**Base-equivalence gate:** after the base arm, the model is upcycled with fresh zero-delta buckets. Its loss on chunk 0
must equal the dense loss within 1e-3 nats, otherwise the run aborts before the MoE arm.

## Test rows: independent of training and of the ratchet

- Source of truth: the run's `real_data_manifest.json` (sha256 `f4f13461…4e913`). All four runs in the checkpoint's lineage
  list identical row ranges.
- The ratchet evaluated on rows `[0, 40)` of each source, and training read the rows from 40 up to `train_rows[1]`.
  **Test rows start at `train_rows[1]`**, are read in increasing order, and the scan stops after 2,000 rows per source.
- Rows are fetched by the manifest's origin: the local HF cache in file order (as in training) or the datasets-server.
  Before registration, the last 100 training rows of each local source were re-read and compared with the run's cached
  copy: all four matched, and no cached block starts at `train_rows[1]`. The script repeats this check and aborts on a
  mismatch.
- A later source of the same expert is read only if the earlier ones run short.
- The benchmarks `allenai/ai2_arc`, `google-research-datasets/mbpp` and `openai/openai_humaneval` are refused.
- **Expected skips:**
  - E3 logic has no source and was never trained.
  - E4 physics: `maximus-21/PhysicsQA` has 370 rows, and training read up to row 370, so no unseen rows are left. It is
    reported as skipped, not substituted.

## Chunks and metric

- **One chunk per test row:** the first 64 token ids of the first 40 rows per expert that have at least 64 ids. That is
  6 experts × 40 = **240 chunks**; at least 200 are required.
- **Per-chunk metric:** mean next-token NLL in nats over the 63 shifted positions. Each chunk runs at batch size 1, with
  `eval()` and `no_grad`.
- **Compute:** CPU only, fp32 trunk, 4 threads, `nice 19`. The GPU is never touched.
- **Paired difference:** `d_i = NLL_moe_i − NLL_base_i`. Negative means the add-on helps.

## Statistics and decision rule

- **Primary statistic:** paired bootstrap of `mean(d)`, resampling chunks i.i.d. with 10,000 resamples. The RNG is
  `numpy.random.default_rng(20261118)` and the CI is the percentile 95 % interval.
- **Also reported:**
  - per-expert means (base, MoE, d, relative change);
  - an exact two-sided sign test with ties dropped, informational only.

| Verdict | Condition |
|---|---|
| INSUFFICIENT_N | fewer than 200 chunks |
| **PASS** | CI upper bound < 0 |
| **HARM** | CI lower bound > 0 |
| **NO_GAIN** | otherwise |

**Domain guard** (from the B3 definition in `docs/MVP_TARGET.md`, "no domain worse than +1%"): every evaluated expert
must have mean `NLL_moe ≤ 1.01 × mean NLL_base`.

**Effect on B3:** this evaluation supports B3 only if the verdict is PASS **and** the domain guard holds. Every other
outcome leaves B3 failing. The verdict is reported exactly as the rule gives it, whatever it is.

`scripts/mvp_scorecard.py` (`check_b3`) reads only `GRANITE_ADAPTER_*gate*.json`, so this experiment does not change
the scorecard wiring.

## Receipt

The receipt is `artifacts/experiments/E_B3_MOE_PAIRED_<date>.json`. It records:

- the sha256 of the spec;
- the pre-registration commit and the script commit;
- the row indices used per source;
- the sha256 of the chunk token ids;
- the per-chunk NLLs;
- start and end times;
- a `deviations` list.

---

## Results (added after the run; the sections above and the JSON spec are unchanged)

- **Order of events:**
  - pre-registration commit `ef51535300f3187c04c22adcdf3a206fb29e33bc`, pushed 2026-11-18 12:46:18 UTC;
  - script commit `b76570fc` pushed next;
  - the evaluation started 12:49:19 UTC and finished 12:59:40 UTC (621 s wall, CPU only, 4 threads, fp32 trunk).
- **Receipt:** [`artifacts/experiments/E_B3_MOE_PAIRED_2026-11-18.json`](../artifacts/experiments/E_B3_MOE_PAIRED_2026-11-18.json),
  with spec sha256 `0a22082e…57c14`, chunk-id sha256 `8eabd337…92a9b92` and `deviations: []`.
- **Integrity checks:**
  - every registered hash matched;
  - the row-order check passed on all 4 local sources;
  - base-equivalence gate: the zero-delta upcycled model gave 4.715309 on chunk 0, identical to dense.

**Test rows used** (all at or after each source's `train_rows[1]`):

| Expert | Source | Rows used | Notes |
|---|---|---|---|
| E0 general | fineweb-edu | 440–479 | |
| E1 code | CodeAlpaca-20k | 3244–3314 | 75 scanned, 35 had < 64 tokens; self-oss-instruct not needed |
| E2 math | NuminaMath-CoT | 340–379 | |
| E3 logic | — | — | skipped: never trained |
| E4 physics | PhysicsQA | — | skipped: no rows after 370 |
| E5 exoplanets | exoplanet-transit-detection | 140–179 | |
| E6 stars | sspp-dr9-stellar-spectra | 540–579 | |
| E7 security | Code_Vulnerability_Security_DPO | 940–979 | |

**N = 240 chunks** (6 experts × 40).

| | Base NLL | MoE NLL | Mean d | Change | MoE lower / higher |
|---|---|---|---|---|---|
| E0 general | 4.6290 | 4.5870 | −0.0419 | −0.91 % | 27 / 13 |
| E1 code | 5.1974 | 4.0355 | −1.1619 | −22.36 % | 40 / 0 |
| E2 math | 4.8411 | 4.7034 | −0.1376 | −2.84 % | 35 / 5 |
| E5 exoplanets | 6.1476 | 5.9895 | −0.1581 | −2.57 % | 33 / 7 |
| E6 stars | 3.5218 | 3.3214 | −0.2004 | −5.69 % | 40 / 0 |
| E7 security | 5.1801 | 4.8966 | −0.2835 | −5.47 % | 40 / 0 |
| **All** | **4.9195** | **4.5889** | **−0.3306** | | **215 / 25** |

- **Mean paired difference:** −0.3306 nats, bootstrap 95 % CI **[−0.3865, −0.2770]** (10,000 resamples, seed 20261118).
- **Sign test:** p = 7.2e-39 (informational).
- **Verdict under the registered rule: PASS** (the CI upper bound is below 0).
- **Domain guard:** holds, since no expert got worse.
- So this evaluation **supports B3** for the MoE add-on as registered.

**Reading the numbers (not part of the decision rule):**

- **Chunk position:**
  - Each chunk is the first 64 tokens of a row, in the `row_to_text` serialisation used in training (for example
    `instruction: … output: …`).
  - Part of the gain, most clearly the −22 % on E1, where CodeAlpaca rows are short templated records, is likely the
    add-on learning that per-source format, not general domain skill.
  - The ratchet measured contiguous windows inside rows instead.
  - So these magnitudes are not comparable with the ratchet's −2.88 %, and they should not be quoted as a general
    quality gain of Crowd-v1.
- **Scorecard:** the MVP scorecard is not changed. `check_b3` reads only the Granite train-to-serve gate artifacts, which
  still fail. Wiring this receipt into B3 (and into the router's serve gate) is an owner decision.
