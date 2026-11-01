# Project HiveMind — Master `TODO.md`: What Has Really Been Achieved vs. What Remains

**Last Updated**: November 1, 2026 (Foundation Paper **v4.3.0** Alignment)  
**Companion Documents**:
- Foundation Paper: [`paper/hivemind_foundation_article.tex`](paper/hivemind_foundation_article.tex) & [`paper/hivemind_foundation_article.pdf`](paper/hivemind_foundation_article.pdf) (v4.3.0, 75 pages, 84/84 provenance gates passed)
- Code & Execution Audit Plan: [`docs/IMPROVEMENT_PLAN_2026-11-01.md`](docs/IMPROVEMENT_PLAN_2026-11-01.md) (`IP-01`..`IP-125`)
- Real Two-Host E8 Audit & Discrepancy Reconciliation: [`docs/E8_TWO_HOST_REAL_TEST.md`](docs/E8_TWO_HOST_REAL_TEST.md) & [`artifacts/experiments/e8_real/RESULTS_T4_SIDE.md`](artifacts/experiments/e8_real/RESULTS_T4_SIDE.md)
- Wave 3 Nocturnal T4 Campaign Report: [`artifacts/night_sci_wave3/NIGHT_SCI_WAVE3_campaign_report.json`](artifacts/night_sci_wave3/NIGHT_SCI_WAVE3_campaign_report.json)

---

## 1. Executive Summary: Epistemic Honesty & Current State

This document provides an engineering- and referee-grade accounting of **what has actually been implemented, executed on real hardware, and verified** in Project HiveMind versus **what remains simulated, single-host, or open on the research backlog**.

### Three-Tier Epistemic Stratification
1. **Verified on Real Multi-Host / Production Infrastructure**:
   - **Two-Host Cross-OS DiLoCo (`E8 Stage 1`, `PR #407`)**: Executed across `socreateai-agora-hermes-node1` (Debian 12 Linux, NVIDIA Tesla T4 16 GB) and `XAVGAMSRV` (Windows 11 Build 26200, physical NVIDIA GeForce RTX 2070 8 GB) over an SSH reverse tunnel (`K=2`, RDA merge, 12 outer rounds, 400 inner steps, seeds 1–3). Paired final-loss difference vs. single-host baseline: **`+4.8e-07 ± 8.3e-07`** (loopback control: **`+0.000e+00`**); steady-state round latency: **`p50 = 1.20 s`**, **`p95 = 3.79 s`**.
   - **Two-Host Cloud WAN Pilot (`E8-pilot`)**: Executed between local workstation and GCP TPU VM (`us-west4-a`, 151 ms ICMP RTT, ~320 ms HTTP tunnel RTT), reproducing the single-host trajectory at `K=2` (`-0.00000 ± 0.00000`, `p50 = 4.1 s`, `p95 = 14.2 s`).
   - **Live GCP Cloud Run + Tesla T4 + BOINC Production Swarm (`v3.9.0`–`v4.3.0`)**:
     - **Registered Accounts vs. Physical GPUs (`PR #409`, `PR #426`, `PR #430`)**: **47 registered accounts** in the v3.9.0 snapshot (expanded to **58–69 registered accounts** on the live `hivemind-openrouter` SQLite database, 36.0% active in 24h) spanning **all 4 system roles**: 8 dedicated GPU Producers, 14 Prosumers, 22 inference Consumers, and 3 Admins.
     - **Attested BOINC & Nocturnal MoE Workunits**: **100 assimilated BOINC workunits** (`5,632.11 Cobblestones` ≈ `2.433 PFLOPs`, 98 chained 6-domain QLoRA rounds lowering held-out NLL `2.2705 -> 2.1380`, `-5.8%`) + **24 nocturnal multi-producer MoE workunits** (`4,525.0 TFLOPs-hours` cumulative attested compute) with a **0.0% (`0/124`) execution-class-matched false rejection rate**.
     - **Live OpenRouter Inference Ledger**: **1,427 billed completions** (`240,162` tokens, mean endpoint `p50 = 196.8 ms`, 0 5xx errors). Note: live `hivemind/coder` requests (`850` completions) were served by the primary Tesla T4 Ollama daemon (`34.44.142.201:11434`), while the `5,462 MiB` (`5,483 MiB` in E16) RTX 2070 figure measures local consumer GPU residency for the 4-bit `Qwen2.5-Coder-7B` tier.
2. **Verified on Single-Host Real GPU Hardware (NVIDIA Tesla T4 16 GB & RTX 2070 8 GB)**:
   - **Wave 3 Scientific MoE Bee (`15 = 10 + 5` Divisions, `NIGHT-SCI-WAVE3-2026`)**: Live nocturnal T4 training & paired evaluation on `Qwen/Qwen3-4B-Instruct-2507` across 5 new scientific expert divisions (`bee_quantum_fluid`, `bee_astrophysics_k3`, `bee_genetic_elenchus`, `bee_topology_tda_lean`, `bee_numeric_solver_sundials`), lifting held-out 50-item scientific benchmark accuracy from **`58.00%` (`29/50`) to `84.00%` (`42/50`, `+26.00%` lift, McNemar `p = 0.001312`, `b=13, c=0`)**, **`0.0%` wrong-verified energy rate**, **`5,842.0 MiB` peak VRAM**, and **`9,518.0 MiB >= 4,300 MiB` free T4 VRAM** (Gates `G1`–`G3` all **PASS**).
   - **Controlled GPU & Mechanism Benchmarks (`H1`–`H11`, `E1`–`E7`, `E10`–`E20`, RTX 2070 PoC)**:
     - RTX 2070 4-branch `Qwen2.5-0.5B` QLoRA-MoE upcycling within **`1,509.5 MB` peak VRAM**.
     - `E15`/`E16` coding evaluation on full MBPP-sanitized (`n=257`, `85.6%` pass@1 on 7B, `85.2%` on small-first cascade) and HumanEval (`n=164`, `67.7%`).
     - `NIGHT-MOE-1` (`hivemind/bee-stars`) on GAIA/Hipparcos stellar classification (`n=317`, `48.9% -> 89.3%`, `+40.4%`, McNemar `p < 1e-4`).
     - `E20` (`P20a`–`P20c`): `680.0 MiB` Adaptive Multi-Split ML guard (`94.34%–98.29%` step-time reduction), `225.4x` Top-1% 2D-DCT + EF21 compression (`3e-8` wire drift), and `31.45x` 6-region Super-Peer ingress reduction + `62.5%` SQLite write-IOPS coalescing.
3. **Verified by Formal Kernel (`Lean 4`)**:
   - **`161` zero-`sorry` discrete arithmetic & necessity lemmas** in [`lean4/HiveMindSpec.lean`](lean4/HiveMindSpec.lean) + **`3` Mathlib real-inner-product theorems** in [`lean4/mathlib/RDAMultiplier.lean`](lean4/mathlib/RDAMultiplier.lean) (**`164` total declarations**).
   - *Explicit boundary*: The 161 lemmas in `HiveMindSpec.lean` are over $\mathbb{N}/\mathbb{Z}$ (byte budgets, partition non-overlap, EF21 telescoping sums, necessity counterexamples) and do not prove real-valued stochastic convergence or C++/CUDA equivalence.

---

## 2. What Has Really Been Achieved & Verified (Detailed Ledger)

### 2.1 October 31 – November 1, 2026 Audit & Wave 3 Pull Requests (`PRs #368–#439`)

| PR / Commit | Subsystem / Files | What Was Really Achieved & Verified | Status |
| :--- | :--- | :--- | :--- |
| **`PR #407`** (`9fc7623`) | [`scripts/run_experiment_E8_pilot.py`](scripts/run_experiment_E8_pilot.py), [`docs/E8_TWO_HOST_REAL_TEST.md`](docs/E8_TWO_HOST_REAL_TEST.md), [`artifacts/experiments/e8_real/`](artifacts/experiments/e8_real/RESULTS_T4_SIDE.md) | Executed real two-host cross-OS `E8 Stage 1` DiLoCo between Linux Tesla T4 (`socreateai-agora-hermes-node1`) and Windows 11 RTX 2070 (`XAVGAMSRV`) over SSH reverse tunnel (`+4.8e-07 ± 8.3e-07` vs. single-host; `p50 = 1.20 s`). Fixed 4 harness bugs (`HM_E8_DEVICE`, `HM_INNER_STEPS`, round-0 cold-start exclusion, `E8_REPORT_PATH`). | **Merged** |
| **`PR #408`** (`14bd515`) | [`src/hivemind/core/idle_worker.py`](src/hivemind/core/idle_worker.py), [`tests/test_idle_worker_cuda_truthfulness.py`](tests/test_idle_worker_cuda_truthfulness.py) | Closed synthetic gradient fallback loophole: `idle_worker.py` now requires `real_gpu_verified=True` before accruing any training steps or points (synthetic mode requires explicit `HIVEMIND_ENABLE_SYNTHETIC_GRADIENTS=1`). | **Merged** |
| **`PR #409`** (`f3b6a04`), **`PR #426`** (`48fc6f5`) | [`deploy/gcp_openrouter/app.py`](deploy/gcp_openrouter/app.py), [`deploy/gcp_ruche/app.py`](deploy/gcp_ruche/app.py) | Reconciled "47 nodes" wording across live UIs and APIs: explicitly distinguishes total registered accounts (all 4 roles) from active 24h physical GPU Producers/Prosumers. | **Merged** |
| **`PR #412`** (`bf26c2d`) | [`src/hivemind/moe/wave3_scientific_verifiers.py`](src/hivemind/moe/wave3_scientific_verifiers.py), [`scripts/run_night_sci_wave3_campaign.py`](scripts/run_night_sci_wave3_campaign.py) | Implemented Wave 3 Scientific MoE Bee (`15 = 10 + 5` divisions on `Qwen/Qwen3-4B-Instruct-2507`), 5-term deterministic scientific energy verifier $E_{\text{wave3}}$, T4 pre-flight dry-run guard, and nocturnal training orchestrator. | **Merged** |
| **`PR #428`** (`6cd2a8d`) | [`src/hivemind/core/gwaya.py`](src/hivemind/core/gwaya.py), [`tests/test_gwaya_lite.py`](tests/test_gwaya_lite.py) | Replaced the last two `pass` stubs in `gwaya.py` with thread-safe `ContextVar` session tracking and SHA-256 weight verification (`190/190` zero-stub files in `src/hivemind/`). | **Merged** |
| **`PR #429`** (`dfef620`) | [`src/hivemind/moe/wave3_scientific_verifiers.py`](src/hivemind/moe/wave3_scientific_verifiers.py), [`tests/test_moe_load_balance_and_wave3_verifiers_hardening.py`](tests/test_moe_load_balance_and_wave3_verifiers_hardening.py) | Preserved PyTorch autograd graph on Switch-Transformer load-balancing loss $\mathcal{L}_{\text{aux}}$ (`IP-35`, removed `.item()` detachment) and hardened `DeterministicScientificEnergyVerifier` against `NaN`/`Inf`/`bool` injection and omitted domain witness keys (`25/25` tests pass). | **Merged** |
| **`PR #430`** (`3c75a66`), **`PR #433`** (`5f44192`) | [`deploy/gcp_openrouter/app.py`](deploy/gcp_openrouter/app.py), [`deploy/gcp_ruche/app.py`](deploy/gcp_ruche/app.py), [`tests/test_la_ruche_and_hivemind_at_home.py`](tests/test_la_ruche_and_hivemind_at_home.py) | Added live `/api/swarm/overview` telemetry endpoint (15-MoE `WeightedRoundRobin` router stats + registered vs. 24h active node breakdown) and bilingual (`FR/EN`) Citizen AI Manifesto (`/manifesto`). | **Merged** |
| **`PR #432`** (`3058fc8`) | [`src/hivemind/verify/zk_opoi_v5_circuit.py`](src/hivemind/verify/zk_opoi_v5_circuit.py), [`tests/test_zk_opoi_v5_forgery.py`](tests/test_zk_opoi_v5_forgery.py) | Hardened `ZKOPoIV5FixedPointCircuit` with strict `witness_commitment_sha256` verification, HMAC-SHA256 challenge binding (`witness_hmac_sha256`), and `gradient_norm_q16` bounds checks; updated docstrings to state honest Q16.16 Freivalds/Schwartz-Zippel scope vs. open Halo2/Plonky3 circuit (`HM-S30-01`). | **Merged** |
| **`PR #435`** (`c9a6842`) | [`deploy/gcp_ruche/app.py`](deploy/gcp_ruche/app.py), [`tests/test_ruche_no_unverified_points.py`](tests/test_ruche_no_unverified_points.py) | Removed fixed `250.0`-point client-supplied credit trust on `/api/boinc/submit`; credits are now computed deterministically server-side from verified hardware tier and compute duration (`14/14` tests pass). | **Merged** |
| **`PR #436`–`#439`** | [`src/hivemind/core/auth.py`](src/hivemind/core/auth.py), [`src/hivemind/data/poison_guard.py`](src/hivemind/data/poison_guard.py), [`src/hivemind/core/queen_bee.py`](src/hivemind/core/queen_bee.py), [`src/hivemind/moe/upcycle.py`](src/hivemind/moe/upcycle.py) | Landed `PR-IP-01`..`PR-IP-09` from `docs/IMPROVEMENT_PLAN_2026-11-01.md`: auth & token validation hardening, poison-guard robustness, QueenBee orchestration safety, and MoE upcycling/verification checks. | **Merged** |

---

### 2.2 Wave 3 Nocturnal Tesla T4 Campaign (`NIGHT-SCI-WAVE3-2026`) — Verified Live Results

Executed on `socreateai-agora-hermes-node1` (NVIDIA Tesla T4, `15,360 MiB` total VRAM) across 12 Decoupled DiLoCo outer rounds × 100 inner steps (`1,200` inner steps per division, `K=6` RDA + EF21 Top-1% 2D-DCT at `225.4x` compression; full report in [`artifacts/night_sci_wave3/NIGHT_SCI_WAVE3_campaign_report.json`](artifacts/night_sci_wave3/NIGHT_SCI_WAVE3_campaign_report.json)):

- **Gate G1 (Paired Accuracy Lift & McNemar Significance)**: **PASS** — Base `Qwen3-4B` achieved **`58.00%`** (`29/50`) vs. Wave 3 15-MoE Bee **`84.00%`** (`42/50`), an absolute lift of **`+26.00%`** (`b = 13` discordant improvements, `c = 0` regressions, exact two-sided McNemar **`p = 0.001312 < 0.02`**).
- **Gate G2 (Zero Wrong-Verified Energy Rate)**: **PASS** — **`0.0%`** (`0/50`) wrong solutions accepted by $E_{\text{wave3}}(x,y) \le 10^{-3}$.
- **Gate G3 (Tesla T4 VRAM Headroom Guard)**: **PASS** — Peak training VRAM reached **`5,842.0 MiB`** (`<= 6,200 MiB` cap), maintaining **`9,518.0 MiB >= 4,300 MiB`** free VRAM on the Tesla T4 with **0 OOM events** and **100% (`60/60`) inner-round loss monotonicity**:
  - `bee_quantum_fluid` (Div. 10): Loss `1.9181 -> 0.9037` (`-52.9%`), Accuracy `60.0% -> 90.0%` (`+30.0%`, `b=3, c=0`), Peak VRAM `5,712 MiB`.
  - `bee_astrophysics_k3` (Div. 11): Loss `1.8169 -> 0.8732` (`-51.9%`), Accuracy `60.0% -> 80.0%` (`+20.0%`, `b=2, c=0`), Peak VRAM `5,842 MiB`.
  - `bee_genetic_elenchus` (Div. 12): Loss `1.7410 -> 0.8489` (`-51.2%`), Accuracy `50.0% -> 80.0%` (`+30.0%`, `b=3, c=0`), Peak VRAM `5,655 MiB`.
  - `bee_topology_tda_lean` (Div. 13): Loss `1.6820 -> 0.8229` (`-51.1%`), Accuracy `60.0% -> 80.0%` (`+20.0%`, `b=2, c=0`), Peak VRAM `5,769 MiB`.
  - `bee_numeric_solver_sundials` (Div. 14): Loss `1.6080 -> 0.7793` (`-51.5%`), Accuracy `60.0% -> 90.0%` (`+30.0%`, `b=3, c=0`), Peak VRAM `5,699 MiB`.

---

## 3. Open Pull Requests In Flight (Ready for Review / Merge)

| Open PR | Branch | Scope & Verification Status | Action Needed |
| :--- | :--- | :--- | :--- |
| **`PR #434`** | `fix/security-l2-channel-turn-ledger-2026-11-01` | Hardens `L2StateChannelManager` (`hmac.compare_digest` + single-settlement replay guard), `TURNAllocationManager` (RFC 5766 private/loopback/metadata SSRF block + per-client quotas), and `HiveSwarmLedger` (`RLock` + atomic `os.replace`). `9/9` tests pass in [`tests/test_l2_turn_ledger_hardening.py`](tests/test_l2_turn_ledger_hardening.py). | Review & Merge |
| **`PR #441`** | `pr-ip-10-12-13` | Implements `PR-IP-10`, `PR-IP-12`, and `PR-IP-13` from [`docs/IMPROVEMENT_PLAN_2026-11-01.md`](docs/IMPROVEMENT_PLAN_2026-11-01.md) (training loop edge-case guards, transport resilience, and router health probes). | Review & Merge |
| **`PR #442`** | `pr-ip-11-docs` | Implements `PR-IP-11` documentation and `ROADMAP.md` alignment from [`docs/IMPROVEMENT_PLAN_2026-11-01.md`](docs/IMPROVEMENT_PLAN_2026-11-01.md). | Review & Merge |
| **`PR #443`** | `pr-ip-14-tests` | Implements `PR-IP-14` expanded unit/integration test coverage from [`docs/IMPROVEMENT_PLAN_2026-11-01.md`](docs/IMPROVEMENT_PLAN_2026-11-01.md). | Review & Merge |
| **`PR #410`** | `fix/ip02-purge-tracked-artifacts` | Purges redundant tracked binary/log artifacts identified in `IP-02` while preserving all provenance-locked JSON artifacts required by `paper/autoevolve_peer_review_pipeline.py`. | Verify provenance gate & Merge |

---

## 4. What Remains to Be Achieved (Prioritized Backlog)

### 4.1 P0 Owner & Infrastructure Governance Actions (`IP-01`, `IP-72`, `IP-73`, `IP-119`, `IP-125`)
- [ ] **`IP-01` (Credential & Token Hygiene)**: Rotate any historic API tokens or admin credentials that ever appeared in local shell scripts or environment files, and verify secret scanning in CI.
- [ ] **`IP-72` (GitHub Branch Protection on `main`)**: Enable required status checks (`pytest`, `lake build`, `autoevolve_peer_review_pipeline.py`) and disallow force-pushes on `main`.
- [ ] **`IP-73` (Automated Zenodo DOI Deposit Secret)**: Configure `ZENODO_ACCESS_TOKEN` in GitHub Actions repository secrets so [`scripts/publish_hivemind_paper_zenodo.py`](scripts/publish_hivemind_paper_zenodo.py) can automatically mint/update the Zenodo deposit on release tags.
- [ ] **`IP-119` (Cloud Run Production Rollout of v4.3.0 Endpoints)**: Deploy the latest `hivemind-openrouter` (`--cpu=1 --memory=1Gi --min-instances=1 --max-instances=1 --no-cpu-throttling --cpu-boost --concurrency=80`) and `hivemind-at-home-ruche` (`--min-instances=0 --cpu-throttling --cpu-boost`) revisions so `/api/swarm/overview`, `/manifesto`, and server-side BOINC credit calculation (`PR #430`, `PR #433`, `PR #435`) are live on all public URLs.
- [ ] **`IP-125` (GCP Cost Governance Audit & Cleanup)**: Run [`scripts/gcp_cost_cleanup.sh`](scripts/gcp_cost_cleanup.sh) to enforce Artifact Registry retention (`keepCount: 2`, `7d`), verify zero idle Cloud TPUs or unattached static IPs, and keep `darkmatter-k3-db` at `--activation-policy=NEVER` when idle.

### 4.2 P0 Real Multi-Host GPU Training & Byzantine WAN Campaigns (`Issue #405`, `E8 Stage 2`, `E9`)
- [ ] **`E8 Stage 2` — Real Multi-Host GPU QLoRA DiLoCo (`Issue #405`, `HM-S9-01`)**:
  - Upgrade the two-host cross-OS `E8 Stage 1` CPU harness (`PR #407`) to run **4-bit NF4 `Qwen2.5-0.5B` / `Qwen3-4B` QLoRA** on physical GPUs across `socreateai-agora-hermes-node1` (Tesla T4 16 GB) and `XAVGAMSRV` (Windows 11 RTX 2070 8 GB) plus $\ge 3$ volunteer GPU producers ($M \ge 5$).
  - Implement true cross-round asynchronous learner continuation in `DecoupledDiLoCoQuorumSyncer` so straggler deltas that miss quorum $K < M$ are carried forward and merged in round $t+1$ with token-weighted Radial-Directional Averaging (resolving the round-synchronous discard defect identified in `E8-pilot` cell 2).
  - Include transport RTT explicitly in the adaptive grace window controller ($\xi_{\text{grace}} = \gamma\,\xi_{\text{slack}} + \widehat{\text{RTT}}_{p95}$, resolving `E8-pilot` cell 3).
- [ ] **`E9` — Live Multi-Host Byzantine Red-Team Campaign ($n \ge 5$ Multi-Krum & ALIE Coalitions)**:
  - Execute a 3-seed, $n \ge 5$ ($n \ge 2f + 3$) multi-host WAN experiment injecting sign-flip, Gaussian noise, and A-Little-Is-Enough (`ALIE`) inner-product manipulation attacks alongside honest GPU learners to measure the empirical false-positive and false-negative rates of Multi-Krum + $2\sigma$ cosine filtering + 10% micro-checkpoint spot-checks under non-IID shard drift.

### 4.3 P1 Cryptographic, Verification & Formal Proof Milestones (`HM-S30-01`, `IP-75`–`IP-78`)
- [ ] **`HM-S30-01` — True Halo2 / Plonky3 R1CS/AIR `zk-SNARK` Circuit**:
  - While `ZKOPoIV5FixedPointCircuit` (`PR #432`) provides a fast $\mathcal{O}(d)$ deterministic Q16.16 Freivalds / Schwartz-Zippel linear-projection and HMAC-SHA256 commitment verifier, it is **not** yet a succinct polynomial IOP / R1CS zk-SNARK. Implement a real Halo2 or Plonky3 lookup-table circuit proving quantized forward/backward LoRA rank-$r$ matrix multiplication with $\mathcal{O}(1)$ proof size ($\le 4\text{ KB}$) and sub-5 ms verification latency.
- [ ] **BOINC Assimilator & Cross-Architecture Validator Hardening (`IP-75`–`IP-78`)**:
  - Implement a true integer/fixed-point (`Q16.16` / `INT8`) deterministic forward/backward replay kernel so cross-architecture spot-checks (e.g., Turing `sm_75` vs. Ada `sm_89` vs. AMD ROCm) do not require strict architecture-matched validator pools (`E12` follow-up).
  - Automate continuous Quorum-of-2 BOINC workunit generation and canonical result assimilation into the global `FlatBucketOuterOptimizer` checkpoint chain with cryptographic receipt pinning.
- [ ] **Lean 4 Specification Strengthening (`lean4/HiveMindSpec.lean`)**:
  - Replace the two vacuous/tautological Season 15 declarations noted in Paper Section 14 (`Theorem 125` and `Theorem 127`, where one conjunct is a tautology or a hypothesis is unused) with non-vacuous arithmetic necessity theorems while preserving `161/161` zero-`sorry` compilation.
  - Extend `lean4/mathlib/RDAMultiplier.lean` (`3` Mathlib theorems) to formalize the discrete EF21 contractive error-feedback bound over real inner-product spaces.

### 4.4 P1/P2 Model Evaluation & Scaling Backlog (`PR-IP-15`..`PR-IP-19`)
- [ ] **Full-Split Standard Benchmark Suites Across All 35 + 5 Models**:
  - Run full-split zero-shot / 5-shot **MMLU**, **GSM8K**, **MATH-500**, and **GPQA-Diamond** evaluations across all 24 nocturnal MoE adapters and all 15 Wave 3 Scientific MoE Bee divisions (`Qwen/Qwen3-4B-Instruct-2507`), reporting paired bootstrap confidence intervals and McNemar $p$-values against the unadapted base models.
- [ ] **Follow-Up on Invalid / Inconclusive Pre-Registered Experiments (`E10b`, `E11`)**:
  - **`E10b` (0.5B Lattice Outer-Delta Codec)**: Re-run `E10b` on `Qwen2.5-0.5B` with $\ge 24$ outer rounds and $\ge 100$ inner steps per round plus per-round dynamic scale adaptation so the codec distortion advantage (`0.79x` $E_8$ vs. $\mathbb{Z}^8$) can be resolved above trajectory noise.
  - **`E11` (`Qwen2.5-Coder-7B` KV-Cache Negative Control)**: Diagnose why the `FP8` negative control on `Qwen2.5-Coder-7B` exceeded the pre-registered perplexity tolerance (`+12.7%`), and evaluate per-channel outlier-aware `FP8` / `KIVI` scaling.
- [ ] **End-to-End 128-Expert 3-Tier Hierarchical MoE Training (`H3`, `H5`, `E4`)**:
  - Scale from the current 15-division (`NIGHT-SCI-WAVE3-2026`) and 24-expert nocturnal campaigns to simultaneous training and Branch-Train-Mix upcycling of all $4 \times 4 \times 8 = 128$ domain experts under the 3-tier hierarchical router with `FoMoE` Ghost-Expert `Skip-Token` routing.
- [ ] **Remaining `docs/IMPROVEMENT_PLAN_2026-11-01.md` Engineering PRs (`PR-IP-15` to `PR-IP-19`)**:
  - Complete the remaining modular refactoring, observability metrics export, and end-to-end chaos/fuzz testing tasks cataloged in `PR-IP-15` through `PR-IP-19`.
