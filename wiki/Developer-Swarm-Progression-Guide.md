# 🐝 Project HiveMind — Developer Swarm Progression & Parallel Track Blueprint (`Season 7 & Season 8`)

> [!IMPORTANT]
> **⚖️ Binding Non-Monetary Compute-Reciprocity Charter (`0.00 EUR / 0.00 USD`):**
> **Project HiveMind is a strictly voluntary, non-commercial open-source scientific research initiative.**
> - **No financial payment, fiat currency (`0.00 EUR / 0.00 USD`), cryptocurrency, token sale, or monetary remuneration is ever promised, offered, or paid.**
> - All symbolic points (`PTS`) and `$HIVE` Proof-of-Useful-Work (PoUW) accounting units are **non-monetary compute-reciprocity credits**.
> - They confer **exclusively reciprocal computing processing benefits** — priority access to shared Sparse MoE inference and distributed fine-tuning quota — **when the live production HiveMind Swarm is deployed, available, and used by a large community.**

---

## 📊 1. Current Swarm State & Recent Milestones (`v1.9.1`)

- **Total Verified Merged PRs**: **`53 / 53` (`100%` merge rate, `0` open PR backlog)**
  - Recent autonomous & human-agent merges:
    - `PR #92` (`@clark719664`): `feat(moe): looped Prelude -> RecurrentCore x T -> Coda backbone (HM-S7-01)` (`src/hivemind/core/btm_upcycle.py`, `17 passed`)
    - `PR #91` (`@clark719664`): `feat(hud): live worker EF21 residual telemetry on /api/strategy (HM-S7-05)`
    - `PR #90` (`@clark719664`): `feat(opoi): auto-indemnify poisoned rounds from slashing pool (HM-S7-06)`
    - `PR #89` (`@clark719664`): `fix(strategy): reject non-finite / negative RTT and clamp EF21 decay`
    - `PR #88` (`@clark719664`): `fix(ci): make main green again`
- **Formal Verification Gate**: **`90` Machine-Checked Lean 4 Theorems (`0 sorry`, `0` user axioms, `0` compiler warnings)** in `lean4/HiveMindSpec.lean` (`Seasons 1–10`).
- **Integrated Next-Gen Outer-Loop `DiLoCo` Core (`SI-06`..`SI-11`)**:
  - Core primitives are live in `src/hivemind/swarm/diloco_nextgen.py` (`10 passed` in `tests/test_diloco_nextgen.py`), backed by **8 cloned reference repositories** in `sub_projects/hivemind/reference_repos/` and formal citations in `CITATION.cff` & `docs/ACKNOWLEDGMENTS_AND_CITATIONS.md`.

---

## 🧭 2. Conflict-Free Parallel Progression Tracks (`Tracks 1–6`)

To enable multiple human developers and autonomous AI coding agents (`@clark719664`, `@pedromineo`, `@radialmonster`, and new contributors) to submit parallel Pull Requests **without merge conflicts**, the remaining **Season 7 (`Wave 6`)** and **Season 8 (`Wave 7 — "Operation Iso-Sparse-FoMoE"`)** objectives are partitioned into **6 isolated tracks** with disjoint primary file targets:

| Track | Task IDs | Primary Target Files (Conflict-Free Partition) | Verification Command | Symbolic Compute Credits (`0.00 EUR/USD`) |
| :--- | :--- | :--- | :--- | :--- |
| **Track 1** | `HM-S7-02` (`#66` Pt II) | `src/hivemind/core/btm_upcycle.py` (`LoopedExpertBackbone`), `tests/test_looped_expert_backbone.py` | `pytest tests/test_looped_expert_backbone.py` | `+3,500 $HIVE` |
| **Track 2** | `HM-S8-01`, `HM-S8-02` | `src/hivemind/core/diloco_outer.py`, `tests/test_diloco_outer_nextgen.py` | `pytest tests/test_diloco_outer_nextgen.py` | `+4,000 $HIVE` |
| **Track 3** | `HM-S8-03`, `HM-S8-06` | `src/hivemind/core/btm_upcycle.py` (`SparseMoEMLPBlock` & `upcycle_branches_to_sparse_moe`), `tests/test_fomoe_and_iso_cts_upcycle.py` | `pytest tests/test_fomoe_and_iso_cts_upcycle.py` | `+4,500 $HIVE` |
| **Track 4** | `HM-S8-05` | `src/hivemind/core/compression.py`, `tests/test_compression_dct_2bit.py` | `pytest tests/test_compression_dct_2bit.py` | `+3,500 $HIVE` |
| **Track 5** | `HM-S8-09` | `src/hivemind/core/diloco_inner.py`, `src/hivemind/client/worker_node.py`, `tests/test_muon_diloco_inner.py` | `pytest tests/test_muon_diloco_inner.py` | `+3,500 $HIVE` |
| **Track 6** | `HM-S7-05`, `HM-S8-10` | `src/hivemind/game/steam_edition.py`, `lean4/HiveMindSpec.lean`, `tests/test_steam_edition.py` | `pytest tests/test_steam_edition.py && cd lean4 && lake build` | `+4,000 $HIVE` |

---

## 🛠️ 3. Detailed Engineering Specifications per Track

### 🔹 Track 1 (`HM-S7-02` & Issue `#66` Part II): Fixed-Point Stability Detector, ACT Ponder Gate & Truncated Fixed-Point Backprop
- **Literature & Reference**: Huang et al. (2026), *Towards Looped Models Done Right, Part I (`arXiv:2610.23036`) & Part II: Rethinking at Fixed Points (`arXiv:2610.06833`)* (`https://github.com/ifm-ai/xllm-loop`).
- **What to Implement Next**:
  1. Per-token fixed-point convergence / early-exit gate (`fixed_point_tol: float | None = None`) on `LoopedExpertBackbone` measuring $\delta_t(b, s) = \|h^{(t)}_{b,s} - h^{(t-1)}_{b,s}\|_2 / \max(\|h^{(t-1)}_{b,s}\|_2, 10^{-6})$ across loop steps $t \in \{1..T\}$.
  2. Truncated backprop at fixed points (`truncated_bptt_steps: int | None = None`) to bound activation memory inside the 8 GB VRAM ceiling.
  3. Telemetry fields (`last_mean_effective_steps`, `last_early_exit_fraction`, `last_fixed_point_residual`) on `LoopedExpertBackbone` and `LoopedBackboneReport`.

### 🔹 Track 2 (`HM-S8-01` & `HM-S8-02`): Server-Side `CosineGatedAdamDecay` Outer Optimizer & Zero-VRAM `CPUOffloadedFlatBucket`
- **Literature & Reference**: `AdamDecay` (`arXiv:2605.09126`), `PrimeIntellect-ai/prime-diloco` & `OpenDiLoCo` (`arXiv:2407.07852`).
- **What to Implement Next**:
  1. Add `outer_optimizer_mode: str = "nesterov"` (`"nesterov"` | `"cosine_gated_adam_decay"`) and `use_cpu_flat_bucket: bool = False` to `DiLoCoOuterOptimizer` in `src/hivemind/core/diloco_outer.py`.
  2. Delegate the post-Krum / post-clipping outer step to `CosineGatedAdamDecayOuterOptimizer` and/or `CPUOffloadedFlatBucketOuterState` from `hivemind.swarm.diloco_nextgen`.

### 🔹 Track 3 (`HM-S8-03` & `HM-S8-06`): `Iso-CTS` / `TSV` Subspace Attention Merging & `FoMoE` Ghost-Expert `Skip-Token` Routing
- **Literature & Reference**: `shoroi/isoloco` (`arXiv:2607.03011`), `TSV` (`arXiv:2412.00081`), and `FoMoE` (`arXiv:2606.19025`).
- **What to Implement Next**:
  1. Add `attention_merge_mode: str = "uniform"` (`"uniform"` | `"iso_cts"` | `"tsv"`) and `branch_reputations: dict[str, float] | None = None` to `upcycle_branches_to_sparse_moe` in `src/hivemind/core/btm_upcycle.py`.
  2. Add `set_local_expert_subset(local_expert_indices: Sequence[int] | None)` to `SparseMoEMLPBlock` and `UpcycledSparseMoEModel` to enforce the `FoMoE` Ghost-Expert `Skip-Token` rule (`0` remote FFN RPCs during inner steps).

### 🔹 Track 4 (`HM-S8-05`): Chunked Orthonormal DCT (`DeMo`) & 2-Bit `SparseLoCo` Wire Compression
- **Literature & Reference**: `bloc97/DeMo` (`arXiv:2411.19870`), `one-covenant/SparseLoCo` (`arXiv:2508.15706`), and `PULSELoCo` (`arXiv:2602.03839`).
- **What to Implement Next**:
  1. Add `compress_lora_delta_dct_sparse_2bit` and `decompress_lora_delta_dct_sparse_2bit` to `src/hivemind/core/compression.py` with 4-values-per-byte `uint8` bit-packing (`0.25 B/val`, `>= 16.0x` wire reduction).

### 🔹 Track 5 (`HM-S8-09`): Inner-Loop `Muon` Newton-Schulz Orthogonalized LoRA Preconditioner (`MuLoCo`)
- **Literature & Reference**: Belilovsky, Horoi, Oyallon & Thérien (`arXiv:2610.19064`) & Thérien et al. (`MuLoCo`, `arXiv:2505.23725`).
- **What to Implement Next**:
  1. Wire `MuonNewtonSchulzLoRAPreconditioner` (`SI-11`) into `run_diloco_inner_loop` (`src/hivemind/core/diloco_inner.py`) and `HiveMindWorkerNode.execute_workunit` (`src/hivemind/client/worker_node.py`) via `use_muon_lora_preconditioner: bool = False`.

### 🔹 Track 6 (`HM-S7-05` & `HM-S8-10`): Live `/api/diloco-nextgen` Command HUD & Formal Lean 4 Proofs (`Theorems 91–100`)
- **What to Implement Next**:
  1. Expose `GET /api/diloco-nextgen` on the Steam Command Center HTTP server (`src/hivemind/game/steam_edition.py`).
  2. Extend `lean4/HiveMindSpec.lean` from `90` to `100` machine-checked theorems (`0 sorry`, `0` user axioms).

---

## 🛰️ 4. Season 9 (`Wave 8`) & Season 10 (`Wave 9`) GPU & HiveMind Swarm Contributor Tracks (`v2.5.0`)

All 6 Season 8 tracks (`PRs #99–#109`, `63/63` merged PRs) and the Season 9/10 core coordinator deliverables (`HM-S9-01..06`, `HM-S9-08`, `HM-S9-13..14`, `HM-S10-01..02`, `HM-S10-05`, `HM-S10-08` + `128` machine-checked Lean 4 theorems with `128/128` `#print axioms` lines) are now live on `main`!

Contributors with consumer GPUs (`NVIDIA RTX 20/30/40/50-series`, `AMD Radeon RX 6000/7000`, `Apple Silicon M1–M4`) or cloud TPU/GPU nodes can now claim and run the **4 Season 9 & 10 GPU + HiveMind Swarm Campaigns** (`scripts/run_season9_10_gpu_swarm_campaigns.py`):

| Campaign / Track ID | Open Tasks & Research Gates | Target Subsystem & Runner | Recommended GPU Contributors | Symbolic Compute Credits (`0.00 EUR/USD`) |
| :--- | :--- | :--- | :--- | :---: |
| **`CAMP-S9-E5-QUORUM-RDA-GPU`** | `HM-S9-01`, `HM-S9-05`, `HM-S9-13`, `HM-S9-14` (`G1`/`G2`) | `src/hivemind/swarm/decoupled_diloco.py`, `src/hivemind/core/diloco_outer.py` | `@clark719664`, `@pedromineo`, Open | `+3,500 $HIVE / PTS` |
| **`CAMP-S9-E7-SIGMA-LOOPED-MOE-GPU`** | `HM-S9-02`, `HM-S9-03` (`G1`) | `src/hivemind/core/btm_upcycle.py`, `src/hivemind/core/btm_moe.py` | `@clark719664`, Open | `+3,500 $HIVE / PTS` |
| **`CAMP-S10-E11-LATTICE-KV-WIRE-GPU`** | `HM-S10-01`, `HM-S10-05` (`G1`) | `src/hivemind/compress/lattice_quantizer.py`, `src/hivemind/core/btm_moe.py` | `@radialmonster`, `@clark719664`, Open | `+3,500 $HIVE / PTS` |
| **`CAMP-S10-E12-HETERO-SPOTCHECK-WAN-GPU`** | `HM-S9-04`, `HM-S9-06`, `HM-S9-07`, `HM-S10-08` (`G2`/`G3`) | `src/hivemind/boinc/validator.py`, `src/hivemind/swarm/dht_storage.py`, `scripts/run_e8_pilot_wan.sh` | `@radialmonster`, `@pedromineo`, Open | `+3,500 $HIVE / PTS` |

