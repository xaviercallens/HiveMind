# 🐝 Community MoE Training, Custom Bees & Open-Source Dataset Integration (`v3.8.0`)

> **⚖️ Strictly Non-Monetary Compute Reciprocity:** All training and compute credits (HMT Reflexion Points) are symbolic accounting units for prioritizing inference and swarm compute. No fiat, crypto, or payment is involved (`0.00 EUR / 0.00 USD`).

---

## 1. Vision: The "Bring Your Own Bee & Dataset" (BYOB) Architecture

Project HiveMind opens up its decentralized Mixture-of-Experts (MoE) infrastructure to the entire open-source community: **Producers, researchers, university labs, and domain specialists can bring their own specialized Bee requests and their own datasets** to create dedicated micro-experts hosted and served across the global swarm.

Whether you need:
- **Formal Verification & Theorem Proving:** Lean 4 tactics, Isabelle/HOL, Coq, or Metamath (`proofnet`, `miniF2F`).
- **Low-Level Systems & Kernel Performance:** Zero-copy Rust network drivers, Triton GPU kernels, SIMD acceleration.
- **Frontier Physics & Astrophysics:** Kepler/TESS transit analysis, JWST atmospheric spectra, quantum lattice simulations.
- **Biomedical & Life Sciences:** PubMedQA evidence retrieval, genomic variant pathogenicity, clinical drug interaction analysis.
- **Cybersecurity & Smart Contracts:** Automated CTI threat intelligence, SWE-bench vulnerability patching, Solidity EVM audits.
- **Cultural & Regional Linguistics:** Academic French prose, minority languages, dialect translation.

HiveMind provides the automated decentralized training pipeline, cryptographic verification gates, and swarm deployment.

---

## 2. Hardware Capabilities for Community Producers (< 8 GB VRAM)

Every producer running a consumer GPU or edge accelerator can participate in MoE adapter training without running out of memory:

| Base Model | Architecture | Parameter Count | 4-bit NF4 Footprint | Peak Measured VRAM (Training) | Target Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`google/gemma-3-1b-it`** | Gemma 3 | 1.2 B | ~1.1 GB | **1 331.7 MB** | Entry-level GPUs, Laptops, Edge devices |
| **`google/gemma-2-2b-it`** | Gemma 2 | 2.6 B | ~1.8 GB | **2 043.2 MB** | GTX 1660, RTX 2060, Apple M1/M2 |
| **`google/gemma-3-4b-it`** | Gemma 3 | 4.1 B | ~2.7 GB | **3 089.0 MB** | RTX 3060, RTX 2070 (8 GB) |
| **`Qwen/Qwen2.5-Coder-7B-Instruct`** | Qwen 2.5 | 7.6 B | ~4.7 GB | **5 005.9 MB** | RTX 2070, RTX 3070, RTX 4070 |
| **`Qwen/Qwen2.5-7B-Instruct`** | Qwen 2.5 | 7.6 B | ~4.7 GB | **5 027.4 MB** | RTX 2070, RTX 3080, Apple M3 Pro |
| **`Qwen/Qwen3.5-9B-Base`** | Qwen 3.5 | 9.0 B | ~5.4 GB | **5 717.2 MB** | RTX 3080/3090, RTX 4070/4090 |
| **`google/gemma-2-9b-it`** | Gemma 2 | 9.2 B | ~5.8 GB | **6 285.4 MB** | RTX 3090, RTX 4080/4090 |

### Dual-Mode Autopilot Execution
1. **Interactive Daytime Priority (07:00–22:00 CET):**
   - The GPU prioritizes low-latency user chat completions (`hivemind/coder`, `hivemind/bee-stars`).
   - Hardware monitors enforce an instant **$< 1.5$s preemption circuit-breaker** when gaming or local 3D rendering starts.
2. **Nocturnal QLoRA Self-Refinement (22:00–07:00 CET):**
   - Dispatches continuous micro-workunits (`40–50` inner steps) across active community nodes.
   - Enforces the **Paired McNemar Non-Regression Gate** ($p < 0.05$ with Edwards' continuity correction).

---

## 3. The 24-Expert Hugging Face Catalog

The standard HiveMind swarm routes across 24 micro-experts organized into 4 macro-clusters:

- **Cluster 00 (Systems & Formal):** `maths_formal` (`hoskinson-center/proofnet`), `code_rust_kernel` (`bigcode/the-stack-v2-dedup`), `code_python_perf` (`HuggingFaceH4/CodeAlpaca_20K`), `code_feedback_debug` (`m-a-p/CodeFeedback-Filtered-Instruction`), `systems_distributed_p2p` (`bigcode/santacoder`), `cuda_kernel_opt` (`neuralmagic/LLM_Compression_Code`).
- **Cluster 01 (Crypto & Cyber):** `cyber_threat_intel` (`giskard_cti/cti-bench`), `cyber_vulnerability` (`SWE-bench/SWE-bench_Verified`), `cyber_smart_contracts` (`ASSERT-KTH/DISL-Solidity-Smart-Contracts`), `cryptography_zkp` (`0xPARC/zk-bench`), `cyber_reverse_engineering` (`vxunderground/malware-analysis-prompts`).
- **Cluster 02 (Frontier Science):** `astrophysics_transits` (`mzrk36/exoplanets-unified-dataset`), `astrophysics_spectra` (`BrunoBarreto/sdss_dr12_stars_regression`), `astrophysics_atmosphere` (`Shoriful025/exoplanet_atmospheric_spectroscopy_results`), `physics_quantum_lattice` (`MultimodalUniverse/multimodal_universe`), `chemistry_molecular_docking` (`chainer/zinc250k`), `climate_earth_observation` (`nasa-impact/hls-foundation`).
- **Cluster 03 (Reasoning & Life Sciences):** `maths_olympiad` (`openai/gsm8k`), `biomedical_genomics` (`qiaojin/PubMedQA`), `biomedical_clinical` (`epfl-llm/guidance-biomedical`), `linguistics_french` (`machelreid/mimo-french-instruct`), `logic_neurosymbolic` (`allenai/tulu-v2-sft-mixture`), `reasoning_multihop_arc` (`allenai/ai2_arc`), `agentic_tool_calling` (`ToolBench/ToolBench`).

---

## 4. Public & Open-Source Charter (Mandatory Agreement)

Project HiveMind is founded on the principle of **Zero-Trust Open Science**:
1. **Public Domain Weights:** All trained MoE adapter weights, delta checkpoints, and benchmark evaluations are deposited under **permissive open-source licenses (Apache 2.0 / MIT / OpenRAIL)** in the public SocrateAI GCS Vault and Hugging Face Hub.
2. **No Proprietary Lock-In:** Neither Project HiveMind nor the contributor may patent, copyright-restrict, paywall, or privatize the trained adapters.
3. **Reproducibility & Provenance:** Every workunit must include its dataset provenance hash, random seed, training hyper-parameters, and evaluation metrics published to Zenodo / GitHub.

---

## 5. ⚖️ Prorated Fair-Share Leaderboard & PoUW Reflexion Points

The Cyber-Bee Contributor Leaderboard ([`https://hivemindathome.github.io/`](https://hivemindathome.github.io/)) ranks contributors on a **prorated fair-share basis**:
- **Prorated Hourly Yield:** `compute_points_earned / idle_gpu_hours_shared` (~`112–115 pts/hr`).
- **Hardware Equality:** A consumer GPU (e.g. RTX 2070 with 8 GB VRAM) contributing steadily during nocturnal quiet hours earns the **exact same prorated hourly rate per verified workunit** as an enterprise 3090/4090 rig.
- **Idempotent Settlement:** Points are calculated deterministically:
  $$\text{Reward} = 10.0\text{ HMT (base)} + 0.0125 \times \text{steps} + 5.0\text{ HMT (gate bonus)}$$
  and credited to the append-only SQLite hash-chained ledger (`verify_chain() == True`).

---

## 6. How to Propose a Custom Bee (Call for HiveMinders)

1. **Step 1 — Open a GitHub Issue:**
   - Tag: `[Bee-Proposal]`
   - Specify:
     - Target domain title and codename (e.g., `quantum_topological_codes`)
     - Hugging Face dataset identifier (e.g., `org/quantum-circuits-dataset`)
     - Evaluation metrics (accuracy, hold-out perplexity reduction, or formal theorem count)
2. **Step 2 — Swarm Workunit Generation:**
   - The coordinator generates deterministic auto-improvement workunits and assigns them to connected producers during European night or idle windows.
3. **Step 3 — Gate Verification & Upcycling:**
   - When the candidate adapter beats the baseline on the Paired McNemar Gate ($p < 0.05$) without regression, it is published to the public vault and added to the live MoE router!
