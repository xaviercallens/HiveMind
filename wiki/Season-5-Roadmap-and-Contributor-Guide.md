# 🗺️ Season 7 (`Wave 6`) Roadmap & Contributor Guide (`v1.8.0`)

All **48 Pull Requests across Waves 1, 2, 3, 4, and 5 (`Seasons 1–6`) + Dependabot** have been reviewed, verified, and merged into `main`. We have now opened **8 Season 7 / Wave 6 Community Tasks (`HM-S7-01` through `HM-S7-08` — `24,000 PTS` Compute-Credit Pool)** on **[GitHub Discussions #83](https://github.com/xaviercallens/HiveMind/discussions/83)**, **[GitHub Discussions #85](https://github.com/xaviercallens/HiveMind/discussions/85)**, and **[GitHub Issues](https://github.com/xaviercallens/HiveMind/issues)**!

---

## ⚖️ Non-Monetary Compute-Reciprocity Charter (`0.00 EUR / USD`)

> **Important Notice for All Human & AI Contributors**:
> - **No Financial Payment (`0.00 EUR / USD`)**: Project HiveMind is a non-commercial, open-science engineering initiative. There is **no cash payment, salary, fiat payout, or speculative cryptocurrency value** attached to any task, PR, or benchmark submission.
> - **Compute-Reciprocity Points Only (`PTS` / `$HIVE`)**: Earned points (`PTS` / `$HIVE`) are tracked in [`docs/CONTRIBUTOR_TASKS_AND_BOUNTIES.md`](https://github.com/xaviercallens/HiveMind/blob/main/docs/CONTRIBUTOR_TASKS_AND_BOUNTIES.md) and the deterministic ledger (`src/hivemind/economy/ledger.py`). They confer **only reciprocal computing processing benefits**—priority distributed Sparse MoE inference routing, higher QLoRA fine-tuning job quotas, and multi-guild co-mining bandwidth—when the **live production HiveMind Swarm** is deployed and used by a large community.

---

## 1. Open Season 7 (`Wave 6`) Tasks (`HM-S7-01` through `HM-S7-08` — `24,000 PTS` Pool)

| Task ID | Track & Target Scope | Deliverable Summary | Suggested Contributors | Compute-Credit Points (`PTS`) |
| :---: | :--- | :--- | :--- | :---: |
| **`HM-S7-01`** | **Track 1: Sparse Activation Checkpointing (`Issue #66`)** (`src/hivemind/core/btm_moe.py`, `src/hivemind/core/qlora_4bit.py`) | Selective Top-2 Expert activation recomputation reducing 8K-context QLoRA backward activation memory by `>= 35%` while keeping `< 2.2 GB` peak VRAM on 8 GB GPUs | `@clark719664`, `@pedromineo`, Open | **`+3,500 PTS`** |
| **`HM-S7-02`** | **Track 2: Live Multi-Region WAN GameDay & UPnP/PCP Nat Mapping** (`src/hivemind/p2p/nat_traversal.py`, `src/hivemind/p2p/quic_link.py`) | Opt-in multi-host WAN benchmark across real residential NATs with automatic UPnP IGD / RFC 6887 PCP port mapping fallback before TURN `ChannelBind` | `@clark719664`, Open | **`+3,000 PTS`** |
| **`HM-S7-03`** | **Track 3: Asynchronous Pipeline-Parallel Expert Sharding** (`src/hivemind/core/swarm_fabric.py`, `src/hivemind/core/btm_moe.py`) | Micro-batch token pipelining across 2–4 consumer GPUs over QUIC 0-RTT with automatic straggler bypass when a peer enters game-preemption mode | `@pedromineo`, `@clark719664`, Open | **`+3,500 PTS`** |
| **`HM-S7-04`** | **Track 4: Zero-Copy Safetensors MMap & Delta-Adapter Streaming** (`src/hivemind/swarm/hf_community_hub.py`, `src/hivemind/swarm/dht_storage.py`) | Memory-mapped `safetensors` differential LoRA patch streaming (`Delta-LoRA`) cutting peer hot-swap disk I/O and RAM footprint by `50%+` | `@pedromineo`, Open | **`+2,500 PTS`** |
| **`HM-S7-05`** | **Track 5: Verifiable Compute-Credit Redemption & Quota Scheduler** (`src/hivemind/economy/ledger.py`, `src/hivemind/cli/swarm_dashboard.py`) | Production Priority Compute Queue (`Deficit Round-Robin + Compute-Points Weighting`) translating contributor `PTS` into priority inference & QLoRA job slots | `@clark719664`, `@radialmonster`, Open | **`+3,000 PTS`** |
| **`HM-S7-06`** | **Track 6: Community GPU Hardware Telemetry & Profiling Suite** (`src/hivemind/cli/community_gpu_benchmark.py`, `scripts/run_community_gpu_benchmark.py`) | Extend the opt-in GPU benchmark with DirectML/ROCm/Metal backend detection, thermal throttling telemetry, and anonymized JSON leaderboard export | `@radialmonster`, `@pedromineo`, Open | **`+2,500 PTS`** |
| **`HM-S7-07`** | **Track 7: Hack-Miner Season 7 (`Missions 25–28`) & Live Telemetry Replay** (`src/hivemind/cli/hf_hack_miner.py`, `src/hivemind/cli/georing_webgl.py`) | Missions 25–28 (`Distributed RAG Synthesis`, `CUDA Warp-Specialized MoE`, `Cross-Guild ZK Audit`, `64-Node WAN Chaos`) + timeline scrubber in `:8090` | `@pedromineo`, `@radialmonster`, Open | **`+2,500 PTS`** |
| **`HM-S7-08`** | **Track 8: Lean 4 Season 7 Formal Proofs (`Theorems 55–62`)** (`lean4/HiveMindSpec.lean`, `paper/hivemind_whitepaper.tex`) | 8 machine-checked Lean 4 theorems covering selective activation checkpointing bounds, deficit round-robin compute-point fairness, and pipeline straggler safety (`0 sorry`) | `@clark719664`, Open | **`+3,500 PTS`** |
| **`HM-S7-09`** | **Track 9: Operation Polyglot-Singularity-128 Multi-MoE & 8-Domain HF Campaign** (`src/hivemind/cli/multi_moe_hf_campaign.py`, `scripts/run_multi_moe_hf_campaign.py`) | Run & extend the 6-stage real-token Causal-LM (`Qwen2ForCausalLM` + `PEFT` LoRA) 8-domain Hugging Face training, `EF21-DiLoCo` outer sync, Byzantine-filtered `TIES` sign-consensus (`SI-04`), and 128-Expert 3-Tier Sparse MoE upcycling campaign | `@clark719664`, `@pedromineo`, `@radialmonster`, Open | **`+3,500 PTS`** |
| **`HM-S7-10`** | **Track 10: Live Hugging Face Streaming & Multi-GPU MoE Router Calibration** (`src/hivemind/data/fixtures/polyglot_science_8domain_v1.json`, `src/hivemind/core/btm_moe.py`) | Expand `--live-hf-stream` Parquet/Datasets streaming across all 8 Hugging Face subjects and calibrate `ThreeTierHierarchicalMoERouter` Super-Cluster routing | `@pedromineo`, `@clark719664`, Open | **`+3,500 PTS`** |

---

## 2. Historical Wave Summary (`Waves 1–5`: `48/48` PRs Merged)

- **Waves 1–3 (`Seasons 1–4`)**: 25 PRs merged — 4-bit NF4 QLoRA, Sparse 32-Expert MoE, Kademlia DHT, Multi-Krum & ZK-OPoI v1–v2, Steam Background Overlay, 3D WebGL Globe, 40 Lean 4 proofs.
- **Wave 4 (`Season 5`, `PRs #52–#62`)**: 64-Expert 2-Tier MoE, FP8 KV-Cache Paging, Interactive $O(\log L)$ Bisection Fraud Game v3, Live HF Hub API + Multi-Peer 64KB Streaming, Guild Wars Missions 17–20, Multi-Host WAN QUIC+TURN Benchmark, Lean 4 Theorems 41–48.
- **Wave 5 (`Season 6`, `PRs #63–#82`)**: EF21-DiLoCo Stateful Error-Feedback (`3.2x` speedup), 128-Expert 3-Tier Hierarchical MoE (`16x` sparsity), Multi-Round Parallel 4-Way Fraud Game (`ZK-OPoI v4`) + 5% Guild Insurance Pool, BitTorrent Rarest-First Chunk Scheduler, Elo Guild Matchmaking (`Missions 21–24`), RFC 5766 TURN `ChannelBind` (`0x4000`), Lean 4 `77` machine-checked `theorem` declarations (`0 sorry`).

---

## 3. Conflict-Free Pull Request Workflow (Auto-Reviewed Every 3 Hours)

1. **Claim a Task in GitHub Discussions:** Comment `/claim HM-S7-XX` in **[Discussion #85](https://github.com/xaviercallens/HiveMind/discussions/85)** or **[Discussion #83](https://github.com/xaviercallens/HiveMind/discussions/83)**.
2. **Create an Isolated Topic Branch:**
   ```bash
   git checkout -b feat/s7-activation-checkpointing
   ```
3. **Run the Full Verification Gate Locally (`300+` Tests + `77` Lean 4 Theorems):**
   ```bash
   PYTHONPATH=. pytest tests/ -q
   cd lean4 && lake build
   ```
4. **Open a Pull Request against `main`:** Reference `Closes #<issue>` or `Implements HM-S7-XX` / `Implements HM-S8-XX`. The **3-Hour Autonomous Community Orchestrator** (`scripts/hivemind_autonomous_community_orchestrator.py`) automatically scans, verifies, and merges clean PRs every 3 hours (`0 */3 * * *`).

---

## 🧬 4. Season 8 (`Wave 7` — "Operation Iso-Sparse-FoMoE") Tasks (`HM-S8-01`..`HM-S8-10`)

Season 8 leverages **8 cloned open-source GitHub repositories** (`one-covenant/SparseLoCo`, `shoroi/isoloco`, `PrimeIntellect-ai/OpenDiLoCo`, `PrimeIntellect-ai/prime-diloco`, `bloc97/DeMo`, `google-deepmind/asyncdiloco`, `pmerolla/fomoe`, `learning-at-home/hivemind`) and **22 arXiv papers (2023–2026)** integrated in `src/hivemind/swarm/diloco_nextgen.py`. See **[Acknowledgments & Academic Citations](Acknowledgments-and-Academic-Citations)** for complete attributions and BibTeX.

| Task ID | Branch Prefix | Leveraged GitHub / arXiv Work | Deliverable Summary | Reciprocal Compute Points (`0.00 EUR/USD`) |
| :--- | :--- | :--- | :--- | :---: |
| **`HM-S8-01`** | `feat/s8-pa-cgad-worker-wiring` | `CGAD` (`arXiv:2605.09126`) + `asyncdiloco` (`arXiv:2401.09135`) | Wire `CosineGatedAdamDecayOuterOptimizer` (`SI-06`) into `RealPeerTrainerNode` & `multi_moe_hf_campaign.py` | `3,500 PTS / $HIVE` |
| **`HM-S8-02`** | `feat/s8-prime-cpu-flat-bucket-outer` | `prime-diloco` + `OpenDiLoCo` (`arXiv:2407.07852`) | Wire `CPUOffloadedFlatBucketOuterState` (`SI-06B`, `0 B` extra GPU VRAM) into `AsyncNodeWorker` | `3,500 PTS / $HIVE` |
| **`HM-S8-03`** | `feat/s8-isoloco-tsv-upcycle-merge` | `shoroi/isoloco` (`arXiv:2607.03011`) + `TSV` (`arXiv:2412.00081`) | Wire `ReputationWeightedIsoCTSMerger` (`SI-07`) & `TSVDecorrelatedMerger` (`SI-07B`) into 128-Expert BTM upcycling | `3,500 PTS / $HIVE` |
| **`HM-S8-04`** | `feat/s8-pulseloco-sparseloco-2bit-wire` | `one-covenant/SparseLoCo` (`arXiv:2508.15706`) + `PULSELoCo` (`arXiv:2602.03839`) | Wire `ComputeVisibleSparseLoCoCompressor` (`SI-09B`, `158x` wire compression) into `EF21DiLoCoErrorFeedbackBuffer` | `3,500 PTS / $HIVE` |
| **`HM-S8-05`** | `feat/s8-demo-chunked-dct-ef21` | `bloc97/DeMo` (`arXiv:2411.19870`) + `SparseLoCo` | Integrate `ChunkedOrthonormalDCTTransform` (`SI-09C`) with per-worker error feedback | `3,000 PTS / $HIVE` |
| **`HM-S8-06`** | `feat/s8-fomoe-partial-replica-skip-token` | `FoMoE` (`arXiv:2606.19025`) | Partial expert replication (`O_e = 4`, `32` local experts per 8 GB GPU) & Ghost-Expert `Skip-Token` (`SI-09A`) | `3,500 PTS / $HIVE` |
| **`HM-S8-07`** | `feat/s8-skip-attention-car-dma-streamer` | `pmerolla/fomoe` (`Skip-Attention` + `CAR`) | Wire `SkipAttentionSpeculativeRouter` (`SI-10`, `~85.8%` prefetch accuracy + `CAR` fallback) into `PinnedRAMExpertDMAStreamer` | `3,500 PTS / $HIVE` |
| **`HM-S8-08`** | `feat/s8-heloco-autoloco-cadence-bridge` | `HeLoCo` (`arXiv:2606.00271`) + `AutoLoCo` (`arXiv:2609.36662`) | Bridge `AdaptiveDiLoCoCadenceController` (`SI-03`) with `AutoLoCoIntervalCorrector` & `HeLoCoDirectionCorrector` (`SI-08`) | `3,000 PTS / $HIVE` |
| **`HM-S8-09`** | `feat/s8-muon-newton-schulz-inner-lora` | `MuLoCo` (Thérien et al., `arXiv:2505.23725`) + Belilovsky, Horoi, Oyallon & Thérien (`arXiv:2610.19064`) | Integrate `MuonNewtonSchulzLoRAPreconditioner` (`SI-11`) into local QLoRA inner loop | `3,000 PTS / $HIVE` |
| **`HM-S8-10`** | `formal/s8-isoloco-dct-newton-schulz-lean4` | `lean4/HiveMindSpec.lean` (`Theorems 83–90`) | Extend `Season10_OuterLoopDiLoCoEcosystem` in Lean 4 from `82` to `90` machine-checked theorems (`0 sorry`) | `3,500 PTS / $HIVE` |

---

## 🛰️ 5. Season 9 (`Wave 8 — Operation Decoupled-Quorum`) & Season 10 (`Wave 9 — Operation Lattice-Codec`) Tasks & GPU Campaigns (`v2.5.0`)

With all `63/63` Pull Requests merged across Waves 1–7 and `128` machine-checked Lean 4 theorems (`0 sorry`, `128/128` `#print axioms` lines) verified in `lean4/HiveMindSpec.lean`, Season 9 and Season 10 focus on **Decoupled Quorum (`K < M`)**, **Continuous Async Straggler Folding (`HM-S9-13`)**, **Pairwise Delta-Cosine Radial-Directional Averaging (`HM-S9-14`)**, **FoMoE $\sigma$-MoE + Peri-LN (`HM-S9-02`)**, **Looped Orthogonal Input Injection & Terminal-KV Sharing (`HM-S9-03`)**, **Conway-Sloane $E_8$ / $A_2$ Hexagonal Lattice Wire & Paged KV-Cache Codecs (`HM-S10-01`, `HM-S10-05`)**, and **Execution-Class Two-Statistic Spot-Check & `Q16` Fixed-Point Digests (`HM-S10-08`)**:

| Task ID | Branch Prefix | Target Subsystem | Deliverable & GPU Campaign Verification | Reciprocal Compute Points (`0.00 EUR/USD`) |
| :--- | :--- | :--- | :--- | :---: |
| **`HM-S9-01`** | `feat/s9-decoupled-syncer-campaign-wiring` | `src/hivemind/cli/multi_moe_hf_campaign.py` | Wire `DecoupledDiLoCoQuorumSyncer` (`SI-12`) & `AdaptiveGraceWindowController` into `DiLoCoOuterOptimizer` & Stage 7 (`CAMP-S9-E5-QUORUM-RDA-GPU`) | `3,500 PTS / $HIVE` |
| **`HM-S9-02`** | `feat/s9-sigma-moe-peri-ln-block` | `src/hivemind/core/btm_upcycle.py` | FoMoE `sigma_moe` sigmoid independent Top-$k$ routing + Peri-LN & measured ghost-expert Skip-Token FLOP report (`CAMP-S9-E7-SIGMA-LOOPED-MOE-GPU`) | `3,500 PTS / $HIVE` |
| **`HM-S9-03`** | `feat/s9-looped-learned-depth-prior-orthoinj` | `src/hivemind/core/btm_upcycle.py` | `LoopedExpertBackbone` with `orthogonal_input_injection`, `LearnedDepthPrior`, and `evaluate_terminal_kv_sharing` (`CAMP-S9-E7-SIGMA-LOOPED-MOE-GPU`) | `3,500 PTS / $HIVE` |
| **`HM-S9-04`** | `feat/s9-factored-gossip-mix1-mix2` | `src/hivemind/swarm/diloco_nextgen.py` | `FactoredGossipMix1Mix2Coordinator` (`Mix1` non-blocking backbone + `Mix2` JS-disagreement-gated sensitive blocks) (`CAMP-S10-E12-HETERO-SPOTCHECK-WAN-GPU`) | `3,000 PTS / $HIVE` |
| **`HM-S9-05`** | `feat/s9-outer-momentum-restart` | `src/hivemind/core/diloco_outer.py` | Periodic outer-momentum restarting (`K_restart`) across Nesterov, CPU Flat-Bucket & PA-CGAD (`CAMP-S9-E5-QUORUM-RDA-GPU`) | `2,500 PTS / $HIVE` |
| **`HM-S9-06`** | `feat/s9-darl-data-leasing` | `src/hivemind/swarm/dht_storage.py` | Heartbeat-backed CAS data shard leases (`TTL = alpha * t_step + RTT`, `committed == granted - reclaimed`) (`CAMP-S10-E12-HETERO-SPOTCHECK-WAN-GPU`) | `3,000 PTS / $HIVE` |
| **`HM-S9-07`** | `feat/s9-multi-host-wan-gameday-real` | `scripts/run_e8_pilot_wan.sh` | 3–8 real machines over residential WAN (`K < M` quorum, RDA on, p50/p95/p99 outer-round latency) | `3,500 PTS / $HIVE` |
| **`HM-S9-08`** | `formal/s9-rda-norm-and-quorum-liveness` | `lean4/HiveMindSpec.lean` | Season 15 Lean 4 Formal Proofs (`Theorems 121–128`, `128/128` `#print axioms`, `0 sorry`) | `3,000 PTS / $HIVE` |
| **`HM-S9-13`** | `feat/s9-continuous-async-learner-quorum` | `src/hivemind/swarm/decoupled_diloco.py` | `ContinuousAsyncLearnerCoordinator` (`drop` vs `deferred_token_weighted` vs `eager_one_step` with `0` dropped tokens) | `3,500 PTS / $HIVE` |
| **`HM-S9-14`** | `feat/s9-campaign-rda-delta-cosine-diagnostic` | `src/hivemind/swarm/decoupled_diloco.py` | `analyze_campaign_rda_and_delta_cosine` (pairwise delta cosine $\bar{c}$, empirical vs theoretical $r_t$) | `3,000 PTS / $HIVE` |
| **`HM-S10-01`** | `feat/s10-e8-lattice-wire-codec` | `src/hivemind/swarm/diloco_nextgen.py` | Conway-Sloane `E8`/`Z`/`sign` `LatticeDeltaQuantizer` wired into `ComputeVisibleSparseLoCoCompressor` (`CAMP-S10-E11-LATTICE-KV-WIRE-GPU`) | `3,500 PTS / $HIVE` |
| **`HM-S10-02`** | `feat/s10-orchestrator-lean-axioms-gate` | `src/hivemind/cli/community_orchestrator.py` | One-`#print-axioms`-per-theorem Lean 4 gate & Scientific-Lead protected file guard | `3,000 PTS / $HIVE` |
| **`HM-S10-05`** | `feat/s10-a2-hexagonal-kv-cache-codecs` | `src/hivemind/compress/lattice_quantizer.py`, `src/hivemind/core/btm_moe.py` | Conway-Sloane 2D hexagonal `A2` decoder & 5 pluggable codecs in `FP8PagedKVCache` (`CAMP-S10-E11-LATTICE-KV-WIRE-GPU`) | `3,500 PTS / $HIVE` |
| **`HM-S10-08`** | `feat/s10-execution-class-two-stat-spot-check` | `src/hivemind/boinc/validator.py` | Execution-class validator assignment, two-statistic replay rule (`linf` + tensor fraction) & `Q16` fixed-point digest (`CAMP-S10-E12-HETERO-SPOTCHECK-WAN-GPU`) | `3,500 PTS / $HIVE` |


