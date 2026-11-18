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
