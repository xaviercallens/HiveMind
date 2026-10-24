# HiveMind T4 benchmark campaign (2026-10-23/24)

Hardware: Tesla T4 (15 GB, sm_75), 8 vCPU, 31.5 GB RAM; torch 2.7.1+cu118, transformers 5.17.0. Data: Hugging Face WikiText-2, fineweb-edu, codeparrot-clean-valid, GSM8K, French Wikipedia, PubMed, CNN/DailyMail and LAMBADA (E13/E13d/E14 domains), MBPP and HumanEval (local cache). Models: Qwen2.5-0.5B-Instruct (fp16), Qwen2.5-Coder-7B-Instruct (fp16, embeddings on CPU), Gemma-4-E4B (bf16, per-layer embeddings on CPU), Granite-3.1-1B-A400M and 3B MoE (fp16), Qwen2.5-Coder-1.5B-Instruct (fp16, E15) and Qwen2.5-Coder 1.5B / 3B / 7B Q4_K_M via Ollama (E16).

Every experiment below was pre-registered before it ran according to the file modification times (files in `artifacts/experiments/*prereg*.json`; note that the `registered_utc` stamps of several of them post-date the stage timestamps of their runs, an unresolved inconsistency recorded in `docs/RESEARCH_PROGRAM.md`; deviations and amendments are recorded in the files, E14 and the E13c/E13d diagnostics are labelled exploratory). Numbers are copied from the receipts listed at the end; 95 % intervals are df = 2 t-intervals over three seeds. A refuted or inconclusive prediction is reported as such. No peer review of record exists for any of this; PTS / $HIVE are non-monetary and no reward is attached to any experiment. The first queue ran 2026-10-23 20:50 to 2026-10-24 06:50 UTC (sections 1-6); the extension queue (sections 1b and 6-10) ran after it.

## 1. Campaign timeline

- Started 2026-10-23 20:50 UTC, budget 10.0 h, wall time 10.0 h.
- GPU utilisation (NVML, 30 s samples, 1200 samples): mean 64.3 %, peak memory 14.81 GiB.

| job | status | wall (min) | mean GPU util % | what it was |
| :-- | :-- | --: | --: | :-- |
| `E11_qwen0.5b` | exit 1 | 1.2 | 26.0 | crashed at the end (a code bug: KLT table is a dict); fixed and retried |
| `E11_qwen7b` | exit 1 | 1.8 | 31.3 | same crash; retried |
| `E11_gemma4-e4b` | exit 1 | 5.3 | 26.8 | same crash; retried |
| `E11_qwen0.5b_retry` | ok | 3.3 | 72.9 | registered E11 run (amendment 1) |
| `E11_gemma4-e4b_retry` | ok | 51.9 | 97.7 | registered E11 run |
| `E11_qwen7b_retry` | ok | 15.7 | 73.2 | registered E11 run |
| `E11_aggregate` | exit 1 | 0.2 | - | report written; exit 1 = a negative control failed on the Qwen models (reported) |
| `E12_replay` | exit 1 | 0.3 | 0.0 | crashed in 20 s (wrong torch API path); fixed and retried |
| `E10b_lattice_05b` | exit -15 | 23.9 | 99.7 | first execution, stopped by the lead after one cell (not a registered rule) |
| `E12_replay_retry` | ok | 109.9 | 10.6 | registered E12 run |
| `E10b_amended` | exit 1 | 92.1 | 98.5 | amendment-2 run; exit 1 = positive control failed, run INVALID by its registered rule |
| `E11b_qwen0.5b_fineweb-edu_w1024` | ok | 5.6 | 64.5 | E11b exploratory |
| `E11b_qwen7b_fineweb-edu_w1024` | ok | 21.8 | 76.6 | E11b exploratory |
| `E11b_gemma4-e4b_fineweb-edu_w1024` | ok | 78.3 | 90.6 | E11b exploratory |
| `E13_qwen0.5b` | ok | 41.8 | 86.1 | registered E13 (Qwen2.5-0.5B, 18 cells); all controls passed |
| `E13_granite1b` | exit 3 | 3.9 | 28.4 | stopped by the registered time fail-fast (2.69 s/step with eager experts + checkpointing) |
| `E13_granite3b` | ok | 79.3 | 34.1 | registered E13 feasibility run (Granite-3B, 1 seed) |
| `E13_granite_step_bench` | ok | 2.3 | 34.0 | Granite step-time microbenchmark (decided amendment 4) |
| `E13_qwen0.5b_m2proc` | exit 3 | 1.0 | 43.5 | stopped by the learning gate (M=2 learns less); waiver registered (amendment 5) |
| `E13c_qwen0.5b_rotstraggle` | exit 1 | 15.4 | 89.7 | exploratory diagnostic; exit 1 only because the identity control is absent from its S0+S2 grid |
| `E13_qwen0.5b_m2proc_v2` | exit 1 | 3.9 | 72.4 | two real OS processes: S1 == S0 bit-identical; exit 1 only because the codec-rate control is n/a |
| `E13_granite1b_v2` | killed at deadline | 41.1 | 47.6 | Granite-1B with grouped_mm (amendment 4); cut by the deadline, resumed in the extension queue |
| `E11b_qwen0.5b_code_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen7b_code_w1024` | not started (deadline reached) | - | - |  |
| `E11b_gemma4-e4b_code_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen0.5b_gsm8k_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen7b_gsm8k_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen0.5b_news_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen7b_news_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen0.5b_fiction_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen7b_fiction_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen0.5b_pubmed_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen7b_pubmed_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen0.5b_wiki-fr_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen7b_wiki-fr_w1024` | not started (deadline reached) | - | - |  |
| `E11b_gemma4-e4b_gsm8k_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen0.5b_wikitext2_w4096` | not started (deadline reached) | - | - |  |
| `E11b_gemma4-e4b_wikitext2_w4096` | not started (deadline reached) | - | - |  |
| `E11b_aggregate` | not started (deadline reached) | - | - |  |

Successful jobs used 6.83 GPU-hours. Failed attempts are listed and explained below; they are not hidden.

## 1b. Extension queue timeline

- Extension queue started 2026-10-24 06:50 UTC (right after the first queue's deadline), budget 9.0 h; last job ended 2026-10-24 12:28 UTC, i.e. 5.63 h of wall time.
- GPU utilisation (NVML, 676 samples): mean 74.5 %, peak memory 6.68 GiB.

| job | status | start (UTC) | wall (min) | mean GPU util % | what it was |
| :-- | :-- | :-- | --: | --: | :-- |
| `E13_granite1b_v2` | ok | 06:50 | 15.6 | 50.2 | resumed (--resume) from the first queue's partial run: Granite-3.1-1B, cells S0 S1 S2, 3 seeds, grouped_mm (amendment 4); completes the registered E13 Granite-1B result |
| `E14_qwen0.5b` | exit 1 | 07:06 | 27.1 | 88.8 | E14 exploratory (Qwen2.5-0.5B, S0 S2 S6 S7, 3 seeds); report written. exit 1 = the runner's rule controls_all_passed requires every control to be true and two (S1 identity, raw wire bytes) are not evaluated (null) in this grid; every evaluated control passed |
| `E13d_qwen0.5b_m6` | ok | 07:33 | 42.4 | 90.5 | E13d exploratory replication (amendment 6): 6 workers on 6 domains, cells S0 S1 S2 S5, all controls passed |
| `E15_gwaya_rft` | ok | 08:15 | 111.1 | 76.2 | E15: GWAYA-gated RFT of Qwen2.5-Coder-1.5B (stages baseline, loop_base, rft, train, tuned, controls), all controls passed |
| `E16_gwaya_lite_t4` | ok | 10:07 | 73.5 | 86.1 | E16: GWAYA-Lite tiers 1.5B/3B/7B Q4_K_M via Ollama (PR #21 P1/P2), run valid |
| `E14_granite1b` | exit 1 | 11:20 | 68.1 | 49.2 | E14 exploratory (Granite-3.1-1B MoE, S0 S2 S6 S7, 3 seeds); exit 1 for the same reason as E14_qwen0.5b (null controls), every evaluated control passed |
| `E11b_qwen0.5b_fineweb-edu_w1024` | skipped (output exists) | - | - | - | skipped: output exists (this fineweb-edu configuration ran in the first queue) |
| `E11b_qwen7b_fineweb-edu_w1024` | skipped (output exists) | - | - | - | skipped: output exists (this fineweb-edu configuration ran in the first queue) |
| `E11b_gemma4-e4b_fineweb-edu_w1024` | skipped (output exists) | - | - | - | skipped: output exists (this fineweb-edu configuration ran in the first queue) |

The six jobs above used 5.63 GPU-hours in total (4 ended ok, 2 with exit 1 for the null-control reason explained in the table).
The remaining E11b sweep jobs of the first queue (code, GSM8K, news, fiction, PubMed, French Wikipedia, 4096-token contexts) were not re-queued here: they were stopped on purpose, so E11b stays partial (the three fineweb-edu 1024-token configurations) and exploratory, and its aggregate report does not exist.
Explanations of the non-ok rows of the first queue: `E13_granite1b` was aborted by the registered time fail-fast (2.69 s/step projected 3.09 h > 3.0 h; amendment 4 switched to grouped_mm, 0.78 s/step); `E13_qwen0.5b_m2proc` was aborted by the registered learning gate (M = 2 learns only 0.0253 nats by round 3 < 0.05) and re-run as `E13_qwen0.5b_m2proc_v2` under the waiver of amendment 5; `E13_granite1b_v2` was cut by the deadline and resumed above.

## 2. E11 — lattice KV-cache codecs vs FP8 / scalar (RQ10, HM-S10-05)

**Decision (pre-registered, amendment 1), valid models only:** P11a **refuted on qwen0.5b (one rate-matched cell, R = 4; R = 2 and 3 are not rate-matched) and on gemma4-e4b**; P11b **refuted on gemma4-e4b** (inconclusive on qwen0.5b). **The qwen7b row is INVALID (negative control `negative_A2at8_equals_fp8` failed: -0.4722 ± 0.1975 nats) and its verdicts are withdrawn; no 7B number is quoted.** Controls all passed: False. Receipt `d47e0956eef3a831`.

| model | dtype | peak GiB | baseline NLL | ΔNLL FP8 | ΔNLL INT4 | P11a | P11b | controls |
| :-- | :-- | --: | --: | --: | --: | :-- | :-- | :-- |
| qwen0.5b | float16 | 1.295 | 2.785 | +0.0413 ± 0.0095 | +2.410 ± 0.461 | refuted | inconclusive | failed: negative_A2at8_equals_fp8 |
| qwen7b | float16 | 14.228 | - | - | - | INVALID, withdrawn | INVALID, withdrawn | failed: negative_A2at8_equals_fp8 |
| gemma4-e4b | bfloat16 | 9.484 | 2.092 | +0.0017 ± 0.0013 | +0.026 ± 0.008 | refuted | refuted (E8 already beats Z without the transform) | all pass |

Paired differences at matched measured rate (ΔNLL(Z) − ΔNLL(lattice), nats/token, positive = lattice better):

| model | R | Z − A₂ (none) | Z − E₈ (none) | Z − E₈ (KLT) |
| :-- | --: | :-- | :-- | :-- |
| qwen0.5b | 2 | -2.5171 ± 1.3890 (invalid (rate not matched)) | +0.3725 ± 0.4861 (CI includes 0) | +0.8861 ± 0.4717 (E8 better) |
| qwen0.5b | 3 | -3.0592 ± 0.1803 (invalid (rate not matched)) | -0.5382 ± 0.0795 (E8 worse) | -1.1913 ± 0.6485 (E8 worse) |
| qwen0.5b | 4 | -0.0939 ± 0.1785 (CI includes 0) | -0.4394 ± 0.0857 (E8 worse) | +0.0648 ± 0.2290 (CI includes 0) |
| gemma4-e4b | 2 | +0.0057 ± 0.0115 (CI includes 0) | +0.0271 ± 0.0058 (E8 better) | +0.0061 ± 0.0114 (invalid (rate not matched)) |
| gemma4-e4b | 3 | +0.0039 ± 0.0065 (CI includes 0) | +0.0051 ± 0.0068 (CI includes 0) | -0.0008 ± 0.0118 (invalid (rate not matched)) |
| gemma4-e4b | 4 | +0.0054 ± 0.0091 (CI includes 0) | +0.0028 ± 0.0024 (E8 better) | +0.0015 ± 0.0068 (invalid (rate not matched)) |

Descriptive only (decides nothing): does a 4-bit codec reach FP8 (paired ΔNLL difference CI includes 0) at about half the bytes?

| model | A₂@4 | E₈@4 + KLT | Z@4 | INT4 |
| :-- | :-- | :-- | :-- | :-- |
| qwen0.5b | +0.3602 ± 0.0660 | +0.2751 ± 0.1615 | +0.2662 ± 0.1919 | +2.3686 ± 0.4698 |
| gemma4-e4b | +0.0012 ± 0.0079 ✓ | +0.0003 ± 0.0030 ✓ | +0.0066 ± 0.0014 | +0.0240 ± 0.0078 |

## 3. E12 — spot-check replay divergence across hardware (RQ11, HM-S10-08)

**Decision (pre-registered):** P12-same **supported** (cell B (same T4, separate process) - a proxy for same class, not a second T4); P12-cross **supported** (cell C (T4 vs this host's x86 CPU, float32 both)); fixed-point clause: not decided (no fixed-point kernel); fp64 + 2^-20 snapping proxy reported in cell D. Controls all passed: True. ε = 0.0001.

| cell | max ‖Δθ‖∞ | min ‖Δθ‖∞ | segments above ε |
| :-- | --: | --: | --: |
| A_same_process_T4 | 0.000e+00 | 0.000e+00 | 0/9 |
| B_separate_process_T4 | 0.000e+00 | 0.000e+00 | 0/9 |
| C_cross_T4_vs_cpu | 6.260e-04 | 2.050e-04 | 9/9 |
| D_proxy_fp64_snap_T4_vs_cpu | 6.342e-04 | 1.984e-04 | 9/9 |
| E_tamper | 2.961e+00 | 2.619e+00 | 9/9 |

Not run on this host: RTX 2070 (sm_75, other host), TPU-VM CPU, a second T4.
Determinism warnings captured per configuration (count): {'cuda_fp32': 0, 'cuda_fp32_subprocess': 0, 'cpu_fp32': 0, 'cuda_fp64_snap': 0}.

## 4. E10b — lattice outer-delta codec at 0.5B (RQ9, HM-S10-01)

**Decision (pre-registered, amendment 2):** P10a **invalid (controls failed: positive_sign_hurts); computed: supported**; P10b **invalid (controls failed: positive_sign_hurts); computed: inconclusive (CI includes 0)**; P10c **invalid (controls failed: positive_sign_hurts); computed: supported**. Controls all passed: False.

Distortion on real 0.5B LoRA deltas (descriptive whatever the validity): relMSE(E₈)/relMSE(ℤ⁸) = 0.869 ± 0.003 per seed [0.8685, 0.8685, 0.8703] (Gaussian prediction 0.86; E10 toy 0.81).

| arm | final held-out loss | rel. MSE | rate (bits/dim) |
| :-- | :-- | --: | --: |
| none | +2.9459 ± 0.0017 | +0.0000 | +32.000 |
| Z8@1+EF | +2.9460 ± 0.0011 | +0.5153 | +0.991 |
| E8@1+EF | +2.9459 ± 0.0009 | +0.4478 | +1.001 |
| sign | +2.9459 ± 0.0025 | +0.3553 | +1.000 |
| E8@6 | +2.9459 ± 0.0016 | +0.0003 | +6.014 |

## 5. E11b — exploratory sweep over Hugging Face domains and context length

E11b report not found (sweep not aggregated).

## 6. E13 — single-GPU swarm simulation (Qwen2.5-0.5B dense; Granite-3.1 MoE 1B and 3B)

Single-GPU swarm over real localhost HTTP (E8's localhost negative control): **no WAN claim**; latencies are GPU time-sharing artefacts. Peers are separate OS processes only where the memory preflight allowed it (see the hosting-fallback count).

### qwen0.5b

Controls all passed: True; failed controls: []; cells failed: []. Hosting fallback cells: 15.

| cell | seeds | final held-out loss (per seed) | admitted | late | upload ratio (actual / ideal) | round latency p50 (s) |
| :-- | --: | :-- | --: | --: | :-- | --: |
| S0 sync_baseline | 3 | 2.2334, 2.2312, 2.2308 | 1.00 | 0 | - | 19.8 |
| S1 swarm_identity | 3 | 2.2334, 2.2312, 2.2308 | 1.00 | 0 | 1.00x / 1.00x | 20.2 |
| S2 swarm_all_on | 3 | 2.3856, 2.3801, 2.3506 | 0.50 | 27 | 11.03x / 16.05x | 16.8 |
| S3 no_codec | 3 | 2.3419, 2.3441, 2.3647 | 0.50 | 25 | 1.00x / 1.00x | 16.5 |
| S4 no_rda | 3 | 2.3249, 2.3481, 2.3175 | 0.50 | 26 | 11.03x / 16.05x | 16.8 |
| S5 no_quorum | 3 | 2.2437, 2.2377, 2.2370 | 0.96 | 0 | 11.00x / 16.05x | 24.1 |

Registered predictions: P13a **supported**; P13b **refuted** (retention -0.051 ± 0.350, bar 0.5); P13c **supported** (upload 0.0907 of raw, ideal 0.0623, bar 0.30).

Each mechanism off (descriptive; retention = share of the synchronous improvement kept):

| cell | mean retention | admitted | late |
| :-- | --: | --: | --: |
| S3 no_codec | 0.114 | 0.50 | 25 |
| S4 no_rda | 0.264 | 0.50 | 26 |
| S5 no_quorum | 0.942 | 0.96 | 0 |

### granite1b

Controls all passed: True; failed controls: []; cells failed: []. Hosting fallback cells: 0.

| cell | seeds | final held-out loss (per seed) | admitted | late | upload ratio (actual / ideal) | round latency p50 (s) |
| :-- | --: | :-- | --: | --: | :-- | --: |
| S0 sync_baseline | 3 | 2.1122, 2.1125, 2.1182 | 1.00 | 0 | - | 46.6 |
| S1 swarm_identity | 3 | 2.1122, 2.1125, 2.1182 | 1.00 | 0 | 1.00x / 1.00x | 47.0 |
| S2 swarm_all_on | 3 | 2.1946, 2.2008, 2.2178 | 0.67 | 15 | 10.97x / 16.04x | 42.7 |

Registered predictions: P13a **supported**; P13b **refuted** (retention -0.658 ± 0.665, bar 0.5); P13c **supported** (upload 0.0912 of raw, ideal 0.0623, bar 0.30).
MoE routing of the final global model on each domain's held-out tokens (descriptive; mean over seeds; entropy = normalised expert-usage entropy per layer, imbalance = max/mean expert load, top-25% = share of routed assignments on the 25% most-used experts):

| cell | domain | seeds | entropy | load imbalance | top-25% share |
| :-- | :-- | --: | --: | --: | --: |
| S0 | code | 3 | 0.913 | 3.00 | 0.516 |
| S0 | pubmed | 3 | 0.946 | 2.63 | 0.455 |
| S0 | wiki-fr | 3 | 0.930 | 3.02 | 0.499 |
| S1 | code | 3 | 0.913 | 3.00 | 0.516 |
| S1 | pubmed | 3 | 0.946 | 2.63 | 0.455 |
| S1 | wiki-fr | 3 | 0.930 | 3.02 | 0.499 |
| S2 | code | 3 | 0.913 | 2.99 | 0.517 |
| S2 | pubmed | 3 | 0.946 | 2.65 | 0.458 |
| S2 | wiki-fr | 3 | 0.929 | 3.03 | 0.501 |


### granite3b

Controls all passed: True; failed controls: []; cells failed: []. Hosting fallback cells: 0.

| cell | seeds | final held-out loss (per seed) | admitted | late | upload ratio (actual / ideal) | round latency p50 (s) |
| :-- | --: | :-- | --: | --: | :-- | --: |
| S0 sync_baseline | 1 | 1.7185 | 1.00 | 0 | - | 187.1 |
| S1 swarm_identity | 1 | 1.7185 | 1.00 | 0 | 1.00x / 1.00x | 186.3 |
| S2 swarm_all_on | 1 | 1.8934 | 0.50 | 8 | 11.04x / 16.03x | 153.0 |

Registered predictions: P13a **supported**; P13b **descriptive (n=1)** (per seed {'1': -0.6975575286211182}); P13c **supported** (upload 0.0906 of raw, ideal 0.0624, bar 0.30).
MoE routing of the final global model on each domain's held-out tokens (descriptive; mean over seeds; entropy = normalised expert-usage entropy per layer, imbalance = max/mean expert load, top-25% = share of routed assignments on the 25% most-used experts):

| cell | domain | seeds | entropy | load imbalance | top-25% share |
| :-- | :-- | --: | --: | --: | --: |
| S0 | code | 1 | 0.890 | 3.69 | 0.583 |
| S0 | gsm8k | 1 | 0.904 | 3.66 | 0.557 |
| S0 | pubmed | 1 | 0.927 | 3.36 | 0.504 |
| S0 | wiki-fr | 1 | 0.916 | 3.58 | 0.529 |
| S1 | code | 1 | 0.890 | 3.69 | 0.583 |
| S1 | gsm8k | 1 | 0.904 | 3.66 | 0.557 |
| S1 | pubmed | 1 | 0.927 | 3.36 | 0.504 |
| S1 | wiki-fr | 1 | 0.916 | 3.58 | 0.529 |
| S2 | code | 1 | 0.887 | 3.76 | 0.589 |
| S2 | gsm8k | 1 | 0.903 | 3.70 | 0.561 |
| S2 | pubmed | 1 | 0.926 | 3.42 | 0.507 |
| S2 | wiki-fr | 1 | 0.915 | 3.63 | 0.533 |


### E13c — rotating straggler (exploratory diagnostic, decides nothing)

Same as S2 but the delayed worker rotates (3, 2, 1) instead of always being the GSM8K worker. Retention of the synchronous improvement: +0.402 ± 0.304 (per seed {1: 0.5332002329341676, 2: 0.29108089995191744, 3: 0.3812219410475062}), against the registered persistent-straggler value in the table above.

### E13 with M=2 separate OS processes (identity control only)

S1 (2 OS processes over HTTP) equals its S0: passed = True, rows = [{'available': True, 'max_abs_diff': 0.0, 'tolerance': 1e-06, 'passed': True, 'bit_identical': True, 'identity_noise_floor': {'available': True, 'max_abs_diff': 0.0, 'bit_identical': True, 'twin_final_loss': 2.3666575253009796, 'note': 'diagnostic only; the identity rule stays S1 vs S0 <= 1e-6'}}].


### E13d — six-domain, six-worker replication (exploratory, amendment 6)

Exploratory replication registered in amendment 6 (changes no registered E13 result): Qwen2.5-0.5B, **6 workers on 6 Hugging Face domains** (code, wiki-fr, pubmed, gsm8k, news, fiction), K = 3 of 6 for S2, K = M = 6 for S5, the registered straggler (worker 5, fiction) and churn, thread-hosted peers (memory preflight), 3 seeds. Controls all passed: True; cells failed: []. Receipt `18d8e5da22530302`.

| cell | final held-out loss (per seed) | admitted | late | upload ratio (actual / ideal) | round latency p50 (s) |
| :-- | :-- | --: | --: | :-- | --: |
| S0 sync_baseline | 2.6210, 2.6192, 2.6216 | 1.00 | 0 | - | 29.3 |
| S1 swarm_identity | 2.6210, 2.6192, 2.6216 | 1.00 | 0 | 1.00x / 1.00x | 29.8 |
| S2 swarm_all_on | 2.6967, 2.6941, 2.7287 | 0.50 | 50 | 11.04x / 16.06x | 29.3 |
| S5 no_quorum | 2.6275, 2.6231, 2.6321 | 0.97 | 0 | 11.03x / 16.06x | 31.2 |

Per-domain final held-out loss (mean over seeds):

| cell | code | wiki-fr | pubmed | gsm8k | news | fiction |
| :-- | --: | --: | --: | --: | --: | --: |
| S0 | 2.029 | 2.716 | 2.985 | 1.245 | 3.245 | 3.504 |
| S1 | 2.029 | 2.716 | 2.985 | 1.245 | 3.245 | 3.504 |
| S2 | 2.165 | 2.735 | 2.948 | 1.357 | 3.287 | 3.747 |
| S5 | 2.051 | 2.731 | 2.972 | 1.243 | 3.263 | 3.505 |

Replication statuses by the registered rules: P13a **supported** (S1 equals S0 bit-identically); P13b **inconclusive** (retention +0.290 ± 0.391, interval [-0.100, +0.681] includes the 0.5 bar; per seed [0.373, 0.388, 0.109]); P13c **supported** (upload 0.0905 of raw, ideal 0.0623). Without the quorum (S5, K = M) retention is 0.942.
Reading: the registered E13 result (P13b refuted at M = 4) is **not replicated as a refutation** at M = 6 — it is inconclusive — and the point estimate is positive; the K = M cell again keeps almost all of the synchronous improvement. Three seeds give a wide interval; nothing here supports a claim for WAN or volunteer swarms.


## 7. E14 — fair quorum and deferred carry for the swarm (exploratory; Qwen2.5-0.5B and Granite-3.1-1B)

**Exploratory follow-up of E13** (pre-registered in `E14_swarm_fairness_prereg.json` before any S6/S7 run; decides no registered E13 prediction). Two admission policies against the plain K = ceil(M/2) swarm S2: **S6 fair quorum** (a worker excluded from one closed round must be in the next quorum; the round waits for it, capped) and **S7 deferred carry** (a late delta is queued and merged into the next round with weight token_weight / (1 + staleness), staleness <= 2). S2 is re-run in the same invocation so that S2, S6, S7 are paired seed by seed. Hypotheses by the registered rule: H14a (S6) / H14b (S7) are **supported** if the df = 2 95 % t-interval (t = 4.302652729911275) of the per-seed difference loss(S2) − loss(cell) is above 0, **refuted** if below 0, **inconclusive** if it includes 0. Two hypotheses per model, no multiple-comparison correction. One GPU, thread-hosted peers over localhost HTTP, 6 rounds × 20 inner steps, 3 seeds: **no WAN claim**.

### qwen0.5b

Report receipt `975d81b7b503e8e2`; runner's controls_all_passed = False (not evaluated in this grid: raw_wire_bytes_exact, s1_equals_s0; failed: none; S0 twin bit-identical: True); E14 controls valid: True.

| cell | final held-out loss (per seed) | mean | admitted | late | upload ratio (actual / ideal) | round latency p50 / p95 (s) |
| :-- | :-- | --: | --: | --: | :-- | :-- |
| S0 sync_baseline | 2.2334, 2.2312, 2.2308 | 2.2318 | 1.00 | 0 | - | 19.5 / 20.1 |
| S2 swarm_all_on | 2.3856, 2.3801, 2.3506 | 2.3721 | 0.50 | 27 | 11.03x / 16.05x | 16.5 / 22.4 |
| S6 fair_quorum | 2.2458, 2.2464, 2.2518 | 2.2480 | 0.68 | 20 | 11.01x / 16.05x | 21.0 / 29.5 |
| S7 deferred_carry | 2.2963, 2.2843, 2.3039 | 2.2949 | 0.50 | 6 | 11.02x / 16.05x | 16.3 / 20.3 |

Paired differences (S2 − cell, nats; positive = the cell is better than S2) and the registered statuses:

| hypothesis | cell | per seed | mean ± half-width | interval | status |
| :-- | :-- | :-- | :-- | :-- | :-- |
| H14a | S6 | +0.1398, +0.1337, +0.0988 | +0.1241 ± 0.0549 | [+0.0692, +0.1790] | **supported** |
| H14b | S7 | +0.0893, +0.0958, +0.0467 | +0.0772 ± 0.0662 | [+0.0110, +0.1435] | **supported** |

Retention of the synchronous improvement ρ = (L_round0 − L_cell)/(L_round0 − L_S0) (viability bar 0.5 of P13b; the K = M value on Qwen2.5-0.5B in E13 was 0.94):

| cell | per seed | mean ± half-width |
| :-- | :-- | :-- |
| S2 | -0.154, -0.110, +0.109 | -0.051 ± 0.350 |
| S6 | +0.906, +0.887, +0.844 | +0.879 ± 0.079 |
| S7 | +0.523, +0.604, +0.457 | +0.528 ± 0.184 |

Costs (descriptive): round latency p50 S0 19.5 s, S2 16.5 s, S6 21.0 s, S7 16.3 s — S6 / S2 = 1.28, S7 / S2 = 0.99 (one time-shared GPU, so these are scheduling artefacts, not WAN latencies). S6 waited an extra 79.2 s in total (10 rounds with extra wait, 13 forced admissions, 0 wait timeouts); S7 carried 18 deltas (368640 tokens, staleness histogram {'1': 16, '2': 2}, 0 discarded as too stale). Late discards S2 / S6 / S7: 27 / 20 / 6.

Replication check: the E13 registered S2 on this model had final losses [2.3856, 2.3801, 2.3506]; the S2 re-run inside E14 gave [2.3856, 2.3801, 2.3506] (identical to 4 decimals).

### granite1b

Report receipt `5f9eac4939b1333c`; runner's controls_all_passed = False (not evaluated in this grid: raw_wire_bytes_exact, s1_equals_s0; failed: none; S0 twin bit-identical: True); E14 controls valid: True.

| cell | final held-out loss (per seed) | mean | admitted | late | upload ratio (actual / ideal) | round latency p50 / p95 (s) |
| :-- | :-- | --: | --: | --: | :-- | :-- |
| S0 sync_baseline | 2.1122, 2.1125, 2.1182 | 2.1143 | 1.00 | 0 | - | 46.2 / 47.8 |
| S2 swarm_all_on | 2.1946, 2.1804, 2.1923 | 2.1891 | 0.67 | 15 | 10.97x / 16.04x | 42.7 / 65.3 |
| S6 fair_quorum | 2.1692, 2.1754, 2.1752 | 2.1733 | 0.76 | 10 | 10.97x / 16.04x | 50.4 / 66.7 |
| S7 deferred_carry | 2.1946, 2.2053, 2.2149 | 2.2049 | 0.67 | 3 | 10.97x / 16.04x | 43.3 / 77.1 |

Paired differences (S2 − cell, nats; positive = the cell is better than S2) and the registered statuses:

| hypothesis | cell | per seed | mean ± half-width | interval | status |
| :-- | :-- | :-- | :-- | :-- | :-- |
| H14a | S6 | +0.0254, +0.0050, +0.0171 | +0.0158 ± 0.0255 | [-0.0097, +0.0413] | **inconclusive** |
| H14b | S7 | +0.0001, -0.0249, -0.0226 | -0.0158 ± 0.0342 | [-0.0500, +0.0184] | **inconclusive** |

Retention of the synchronous improvement ρ = (L_round0 − L_cell)/(L_round0 − L_S0) (viability bar 0.5 of P13b; the K = M value on Qwen2.5-0.5B in E13 was 0.94):

| cell | per seed | mean ± half-width |
| :-- | :-- | :-- |
| S2 | -0.451, -0.202, -0.459 | -0.371 ± 0.362 |
| S6 | -0.004, -0.114, -0.123 | -0.080 ± 0.165 |
| S7 | -0.450, -0.643, -0.903 | -0.665 ± 0.565 |

Costs (descriptive): round latency p50 S0 46.2 s, S2 42.7 s, S6 50.4 s, S7 43.3 s — S6 / S2 = 1.18, S7 / S2 = 1.01 (one time-shared GPU, so these are scheduling artefacts, not WAN latencies). S6 waited an extra 77.8 s in total (5 rounds with extra wait, 5 forced admissions, 0 wait timeouts); S7 carried 12 deltas (245760 tokens, staleness histogram {'1': 12}, 0 discarded as too stale). Late discards S2 / S6 / S7: 15 / 10 / 3.

Replication check: the E13 registered S2 on this model had final losses [2.1946, 2.2008, 2.2178]; the S2 re-run inside E14 gave [2.1946, 2.1804, 2.1923] (per-seed values differ: thread scheduling is not deterministic between runs).

**Reading.** qwen0.5b: H14a **supported**, H14b **supported** (retention S2 -0.05, S6 +0.88, S7 +0.53; S6 round latency 1.28x S2); granite1b: H14a **inconclusive**, H14b **inconclusive** (retention S2 -0.37, S6 -0.08, S7 -0.67; S6 round latency 1.18x S2). Where a hypothesis is inconclusive its interval includes 0: that is absence of a detected effect with 3 seeds, not evidence of none. Whether the dense-model support transfers to the pretrained MoE is therefore not established at this size. The fixed constants (MAX_EXCLUSION 1, MAX_STALENESS 2) were not tuned.


## 8. E15 — GWAYA-gated RFT of a coding assistant under 8 GB of VRAM

**GWAYA-gated rejection-sampling fine-tuning (RFT) of Qwen/Qwen2.5-Coder-1.5B-Instruct** (rank-16 LoRA, 7.5 GiB per-process cap on the T4), pre-registered with one amendment. GWAYA here is a fail-closed gate: a candidate is accepted only if the vendored zero-stub AST audit is clean **and** the sandboxed task tests pass. Controls all passed: True. Receipt `678b5a52b87918c7`; design hash `67f22d0074135648`.

Stages (NVML power × time for energy; whole-GPU, descriptive):

| stage | status | wall (min) | peak reserved (GiB) | peak allocated (GiB) | energy (Wh) |
| :-- | :-- | --: | --: | --: | --: |
| baseline | ok | 8.3 | 4.18 | 3.67 | 9.1 |
| loop_base | ok | 13.3 | 4.24 | 3.71 | 14.7 |
| rft | ok | 67.4 | 4.47 | 3.42 | 74.8 |
| train | ok | 6.1 | 4.15 | 3.78 | 6.7 |
| tuned | ok | 15.1 | 4.36 | 3.75 | 16.6 |
| controls | ok | 0.14 | - | - | - |

Conditions (pass@1 = all tests pass; Wilson 95 % interval; MBPP-sanitized n = 257, HumanEval n = 164 for the *h* rows):

| condition | what | n | passes | pass@1 [Wilson 95 %] | accepted by the gate | generated tokens | GPU s | correct per GPU-min |
| :-- | :-- | --: | --: | :-- | --: | --: | --: | --: |
| C0 | base, greedy | 257 | 169 | 65.8 % [59.8, 71.3] | 169 | 19285 | 228 | 44.4 |
| C1 | base + GWAYA loop | 257 | 187 | 72.8 % [67.0, 77.8] | 187 | 74808 | 1001 | 11.2 |
| C2 | RFT-tuned, greedy | 257 | 171 | 66.5 % [60.6, 72.0] | 171 | 14391 | 198 | 51.7 |
| C3 | tuned + GWAYA loop | 257 | 190 | 73.9 % [68.2, 78.9] | 190 | 53995 | 689 | 16.6 |
| C0h | HumanEval base | 164 | 110 | 67.1 % [59.6, 73.8] | 110 | 20703 | 254 | 26.0 |
| C2h | HumanEval tuned | 164 | 111 | 67.7 % [60.2, 74.4] | 111 | 16014 | 186 | 35.8 |

Paired comparisons (exact two-sided McNemar from the discordant counts; the p recomputed here equals the report's):

| comparison | difference (points) | b only (second wins) | a only (first wins) | McNemar p | registered status |
| :-- | --: | --: | --: | --: | :-- |
| C2 − C0 (H15a) | +0.78 | 15 | 13 | 0.851 | **inconclusive** |
| C1 − C0 (H15b) | +7.00 | 19 | 1 | 4.01e-05 | **supported** |
| C3 − C1 (H15c) | +1.17 | 12 | 9 | 0.664 | descriptive |
| C3 − C0 (H15c) | +8.17 | 26 | 5 | 0.000192 | descriptive |
| HumanEval C2h − C0h (H15e) | +0.61 | 12 | 11 | 1 | descriptive |

H15d (peak reserved <= 8.0 GiB in every stage): **supported** (maximum 4.47 GiB).
Reading: the GWAYA inference loop moves the base by +7.0 points (19 problems gained, 1 lost; H15b supported); RFT-tuned greedy moves it by +0.8 points (p = 0.85; H15a inconclusive, i.e. no detected gain from the fine-tuning) and by +0.6 points on HumanEval (p = 1.00); stacking the loop on the tuned model vs on the base: +1.2 points (p = 0.66). The loop costs 4.4x the GPU seconds of greedy decoding (C1 1001 s vs C0 228 s; the winner fell back to the first greedy candidate on 53 of 257 problems). One training seed, one model size: no training-seed interval.

RFT data (rejection sampling with the gate, all tests of the train problem):

- 462 train problems after decontamination (464 raw, 2 removed for prompt-text overlap with eval); **359 / 462 problems yielded at least one accepted sample** (fail-fast threshold 150); mean yield 0.5417 accepted per sample; **673 kept (problem, solution) pairs**; unverified samples 0; 329948 generated tokens in 3957 GPU s (67.4 min wall, 74.8 Wh).
- Training: 129 steps over 673 pairs (3 epochs), 18464768 LoRA parameters on 196 linears, epoch mean CE [0.1562, 0.096, 0.0612], skipped non-finite steps 0, adapter sha256 `a56bf67224bcf9b3`.

Controls (the run is valid iff all pass): gate accepts 100/100 MBPP reference solutions and 164/164 HumanEval canonical solutions; 12/12 planted stubs rejected; hostile-candidate containment probes all contained = True; no leakage (id overlap 0, text overlap 0, 2 identical reference solutions under different prompts reported); peak reserved <= 8 GiB = True.

**Sandbox caveat.** The gate is the vendored GWAYA zero-stub auditor plus our own user/mount/net/pid-namespace sandbox with rlimits (no seccomp, no cgroups, shared kernel), not GWAYA's Bubblewrap oracle. It is a defence for a 1.5B model generating Python functions, not for adversarial third-party code, and results are labelled GWAYA-style. The vendored auditor can reject legitimate code (it flags any `pass` outside an except handler or class body), which lowers the RFT yield.


## 9. E16 — GWAYA-Lite tiers on the T4 (PR #21 P1 / P2)

**GWAYA-Lite on a Tesla T4** (GWAYA draft PR #21 criteria P1 and P2 measured as a specification): Qwen2.5-Coder 1.5B / 3B / 7B-Instruct, Q4_K_M, served by Ollama 0.34.1, MBPP-sanitized n = 257, same gate and sandbox as E15. `run_valid` = True (idle GPU, gate and unload controls). Receipt `6c9e904bcd9b52bc`.

Measured table (generated from the receipt into `docs/community/GWAYA_T4_DEPLOYMENT.md` by `scripts/fill_deployment_guide.py`, reproduced here unchanged):

Measured on a Tesla T4 with Ollama 0.34.1, Q4_K_M, MBPP-sanitized n = 257; `run_valid` = True. VRAM = `nvidia-smi` memory.used delta over the idle baseline (CUDA context included).

| Tier | num_ctx | VRAM (MiB) | PR estimate | ratio | decode tok/s | prompt-eval tok/s | on GPU | pass@1 raw R | pass@1 loop L | L-R, gained/lost (McNemar p) | precision of VERIFIED |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.5B | 4096 | 1651 | 1782 | 0.93 | 130.5 | 2841 | 100% | 64.6% | 73.2% | +8.6 pts, 22/0 (p = 4.8e-07) | 92.6% |
| 1.5B | 8192 | 2107 | 1894 | 1.11 | 127.9 | 2769 | 100% | - | - | - | - |
| 3B | 4096 | 2713 | 2584 | 1.05 | 68.3 | 1541 | 100% | 72.8% | 80.2% | +7.4 pts, 19/0 (p = 3.8e-06) | 92.8% |
| 3B | 8192 | 3297 | 2728 | 1.21 | 67.1 | 1580 | 100% | - | - | - | - |
| 7B | 4096 | 5483 | 5290 | 1.04 | 28.5 | 812 | 100% | 81.7% | 85.6% | +3.9 pts, 10/0 (p = 2.0e-03) | 94.8% |
| 7B | 8192 | 6387 | 5514 | 1.16 | 28.2 | 827 | 100% | 81.7% (raw only) | - | - | - |

Registered statuses (E16 pre-registration): H16a (7B within 6656 MiB) 4k **supported**, 8k **supported**; H16b (estimator within 0.80-1.25) **adequate**; H16c (loop does not lower pass@1) {'1.5b': 'supported', '3b': 'supported', '7b': 'supported'}; H16d (loop >= +5.0 points, p < 0.05) {'1.5b': 'supported', '3b': 'supported', '7b': 'inconclusive'}; H16e (precision of VERIFIED >= 0.80) {'1.5b': 'supported', '3b': 'supported', '7b': 'supported'}; H16f (7B decode >= 20 tok/s) **met** (28.5 tok/s).

Escalation ladder (descriptive): 1.5B loop -> 3B loop -> 7B loop reaches 85.2% against 85.6% for the 7B loop alone (paired p = 1.00), escalating 21% of the problems to the 3B and 12% to the 7B, using 78% of the 7B-only model time.
Caveats: the gate is the vendored GWAYA zero-stub auditor plus a namespace sandbox, not GWAYA's Bubblewrap oracle; VERIFIED means audit clean and the visible test passed, and 5-7% of those solutions still fail the hidden tests; one runtime (Ollama), one seed set, one benchmark.

Registered hypotheses (E16 pre-registration), statuses copied from the report:

| hypothesis | claim | status |
| :-- | :-- | :-- |
| H16a | 7B Q4 fits within 6656 MiB (nvidia-smi delta incl. CUDA context) | num_ctx 4096: **supported** (5483 MiB), num_ctx 8192: **supported** (6387 MiB) |
| H16b | measured/estimated VRAM ratio in [0.80, 1.25] for every tier × ctx | **adequate** (ratios 1.5b@4096: 0.93, 1.5b@8192: 1.11, 3b@4096: 1.05, 3b@8192: 1.21, 7b@4096: 1.04, 7b@8192: 1.16) |
| H16c | the loop does not lower pass@1 | 1.5b: **supported**, 3b: **supported**, 7b: **supported** |
| H16d | loop gain >= +5.0 points with McNemar p < 0.05 | 1.5b: **supported**, 3b: **supported**, 7b: **inconclusive** |
| H16e | precision of VERIFIED >= 0.80 vs hidden tests | 1.5b: **supported** (92.6 %), 3b: **supported** (92.8 %), 7b: **supported** (94.8 %) |
| H16f (descriptive) | 7B decode >= 20 tok/s at 4096 | **met** (28.5 tok/s) |

Loop effect per tier (paired by problem; exact McNemar):

| tier | raw R | loop L | L − R (points) | gained / lost | McNemar p | loop extra model time (s) | determinism (10 repeats identical) |
| :-- | :-- | :-- | --: | --: | --: | --: | --: |
| 1.5b | 64.6 % [58.6, 70.2] | 73.2 % [67.4, 78.2] | +8.6 | 22 / 0 | 4.8e-07 | 249 | 1.0 |
| 3b | 72.8 % [67.0, 77.8] | 80.2 % [74.9, 84.6] | +7.4 | 19 / 0 | 3.8e-06 | 365 | 1.0 |
| 7b | 81.7 % [76.5, 86.0] | 85.6 % [80.8, 89.4] | +3.9 | 10 / 0 | 0.002 | 585 | 1.0 |

Context invariance (7B raw): num_ctx 8192 gives 210/257 against 210/257 at 4096 (paired difference +0.0).

Escalation ladder (descriptive, no status): 1.5B loop → 3B loop on FAILED → 7B loop on FAILED reaches 219/257 (85.2 %) against 220/257 (85.6 %) for the 7B loop alone (paired difference -0.4 points, McNemar p = 1.00); it escalates 54 problems (21 %) to the 3B and 31 (12 %) to the 7B and uses 1227 s of model time against 1566 s (78 %).

Controls: idle GPU at start 0 MiB (limit 1500); gate accepts 100/100 references and containment = True; unload returned to 0 MiB (baseline 0); every probe 100 % on the GPU = True; probe contamination clean = True.

**Caveats.** One T4 and one runtime (Ollama, Q4_K_M only); the 6656 MiB PR ceiling is a Windows WDDM reserve, tested here as a threshold on Linux, not as a physical limit; the gate is the E15 gate (vendored zero-stub auditor + namespace sandbox, not GWAYA's Bubblewrap oracle); VERIFIED means audit clean and the *visible* test passed, and 5-7 % of those solutions still fail the hidden tests; Ollama greedy decoding is not guaranteed bit-stable (10 repeated problems per tier were identical here). Publishing anything on the GWAYA repository or its PR is left to the owner.


## 10. GWAYA-Lite deployment on the T4

`scripts/gwaya_lite.py` (small-first ladder Qwen2.5-Coder 1.5B → 3B → 7B Q4_K_M behind the fail-closed gate, guide `docs/community/GWAYA_LITE_ASSISTANT.md`) was deployed as a local OpenAI-compatible service on the T4 at `http://127.0.0.1:8099` (loopback only). Live check `artifacts/gwaya_lite_live_check.json` (receipt `8a8125a929e2e894`, written 2026-10-24T12:31:02Z), `all_ok` = True:

- health: HTTP 200 {"status": "ok", "service": "gwaya-lite", "ladder": ["1.5b", "3b", "7b"], "ctx": 4096, "auth": false}; models: HTTP 200; listening on loopback: True.

| task | tests | verdict | tier that answered | candidates | repair rounds | generated tokens | wall (s) | gate verdict |
| :-- | --: | :-- | :-- | --: | --: | --: | --: | :-- |
| is_palindrome | 4 | **VERIFIED** | 1.5b | 1 | 0 | 71 | 12.38 | ACCEPTED |
| longest_valid_parentheses | 7 | **VERIFIED** | 1.5b | 3 | 0 | 341 | 2.78 | ACCEPTED |
| fizzbuzz | 3 | **VERIFIED** | 3b | 7 | 3 | 397 | 16.12 | ACCEPTED |

- A request without tests was answered **AUDITED** (never VERIFIED without tests = True).
- Streaming: 13 SSE data chunks, `[DONE]` terminator = True.
- Foreign `Host` header → HTTP 403; foreign `Origin` header → HTTP 403.
- After the requests: GPU memory used 1639 MiB; resident Ollama models [['qwen2.5-coder:1.5b', 1514]] (name, MiB).

Scope: a smoke check of three small tasks, not a benchmark (the benchmark is E16 above). The service shares the E15 sandbox limits (not GWAYA's Bubblewrap oracle); the editor-integration examples in the guide are untested in real editors; it must stay on loopback unless an API key is configured.


## 11. Receipts

| artifact | sha256 (first 12) |
| :-- | :-- |
| `E11_kv_lattice_prereg.json` | `f79e06aeb2f1` |
| `E11_kv_lattice_prereg_amendment_1.json` | `2794f8884932` |
| `E11_kv_lattice_report.json` | `be45db7d9106` |
| `E12_replay_divergence_prereg.json` | `9d039cca758d` |
| `E12_replay_divergence_report.json` | `d3c112bf71a3` |
| `E10b_lattice_05b_prereg.json` | `e5102af84574` |
| `E10b_lattice_05b_report.json` | `b65920ad8b78` |
| `E11b_kv_lattice_sweep_prereg.json` | `5c68ffed8f88` |
| `E11b_kv_lattice_sweep_report.json` | `-` |
| `E13_swarm_local_prereg.json` | `4e9a2d800204` |
| `E13_swarm_local_prereg_amendment_1.json` | `9e9ae076864a` |
| `E13_swarm_local_prereg_amendment_2.json` | `820e6ba8a24c` |
| `E13_swarm_local_prereg_amendment_3.json` | `5ed406d70bd4` |
| `E13_swarm_local_prereg_amendment_4.json` | `d82a46694e0d` |
| `E13_swarm_local_prereg_amendment_5.json` | `456ed2a8d1db` |
| `E13_swarm_local_prereg_amendment_6.json` | `3052a7a446c1` |
| `E13_swarm_local_report.json` | `6e412186ff05` |
| `E13_swarm_local_predictions.json` | `6dc1cd833bad` |
| `E13_swarm_local_report_m6.json` | `88e8f3cdf50e` |
| `E13_swarm_local_predictions_m6.json` | `aa5a458e94db` |
| `E14_swarm_fairness_prereg.json` | `42b794fd4c52` |
| `E14_report_qwen0.5b.json` | `bb9683a4da15` |
| `E14_report_granite1b.json` | `ceb4787a59e2` |
| `E14_hypotheses.json` | `034737aeaa8f` |
| `E15_gwaya_rft_prereg.json` | `a7f601f6b964` |
| `E15_gwaya_rft_prereg_amendment_1.json` | `7c26d13eb823` |
| `E15_gwaya_rft_report.json` | `b44ef54c9d36` |
| `E15_hypotheses.json` | `a46220fa3813` |
| `E16_gwaya_lite_t4_prereg.json` | `2f032305ecb7` |
| `E16_gwaya_lite_t4_report.json` | `4cdce7c88770` |
| `artifacts/gwaya_lite_live_check.json` | `c6884e7686ee` |
| `T4_benchmark_queue_log.json` | `a98d675de634` |
| `T4_benchmark_queue_ext_log.json` | `677a4d0d4234` |
