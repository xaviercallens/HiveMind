# Reine-Cortex (QueenCortex) — Active MoE Campaign Architecture & Implementation Plan

> **Non-Monetary Charter (`AGPL-3.0`):** The HiveMind Reine-Cortex (`QueenCortex`) NeuroController subsystem executes under the **`0.00 EUR / 0.00 USD`** Personal-PC & Volunteer Compute-Reciprocity Charter (`LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`), strictly on CPU (`0 MiB` GPU VRAM, `0` gradient steps), with zero cryptocurrency wallets, zero external payment links, and zero unauthorized lateral egress. Root `ROADMAP.md` is untouched; NeuroController status and milestones live in this document and [`docs/NEURO_CONTROLLER_ROADMAP.md`](./NEURO_CONTROLLER_ROADMAP.md).

---

## 1. End-to-End Reine-Cortex (`Gelée Royale`) Closed-Loop MoE Campaign Pipeline

```text
  1. Butinage (Les Abeilles Butineuses)
     Volunteer DiLoCo / QLoRA / CrowdLoRA nodes gather domain Nectar
     (code, Lean 4, astro, physics, cyber) into La Ruche shared vault
          |
          v
  2. Trophallaxie & Gelée Royale WSP (Les Abeilles Nourrices, 0 MiB VRAM)
     - Preflight MP & Byzantine Multi-Krum filter (NaN/Inf & spectral spike rejection)
     - Sequential chain incremental task-vector extraction (delta_d = W_d - W_{d-1})
     - Trophallaxis TIES sign-consensus election (density = 0.25)
     - Low-RSS associative AlphaEdit P_0 projection: (scale * B) @ (A @ P_0)
       applied BEFORE outer DiLoCo momentum accumulation (M_t = beta * P_0(M_{t-1}) + Delta_t^{RJ}, D1)
     - Kronecker-factored Riemannian null-space transport (R_step, R_proj, D2 / Theorem 3.7)
       & geodesic KL trust-region clamping on curved deep layers
     - Optimal-Transport Hungarian + Procrustes cross-arm alignment & P_0 fusion (D3 / Theorem 3.8)
       resolving permuted/rotated router-expert co-adaptation mismatch prior to fusion
     - Selective ghost sinks (w_B = 0), OOF multi-gate routing & ker(G_D0) Fisher/folded RMSNorm
          |
          v
  3. Operculation d'Alvéole ANSE & Alvéole Vierge
     - Monotone ANSE ratchet (Delta E < 0 and D0 / worst_domain <= +1.0%)
     - ANSE Maximum Pain (1e6) atomic bitwise rollback across weights + momentum buffers
     - Continuous SHA-256 hash-chained round receipts & single-shot Alvéole Vierge lock
          |
          v
  4. Phéromones Royales Prochain Round (QueenPheromoneDirective)
     - Freeze converged Anchor Combs (Alvéoles Ancres) bitwise across rounds
     - Sink/recycle pure corruptors (Mitosis D4) and reset corruptor momentum buffers
     - Toggle router_frozen (fragile KL mismatch) vs router_unfrozen_coadapted / OT alignment (D3)
     - Emit normalized Boltzmann/UCB bee_foraging_domain_weights toward high-deficit domains
          |
          v
  5. 4-Bit NF4 Edge Export, Proof-Carrying ZK-WSP & Wald SPRT Canary
     - Prose-Hessian compensated NF4 edge blocks under 6,656 MiB 8 GB VRAM ceiling (D4)
     - Proof-carrying ZK-WSP spot-checks + Lean 4 attestation receipts (D5)
     - Sequential Wald SPRT canary routing with instant bitwise rollback
```

### Operator Mapping to Cleanroom Modules (`51` Merged PRs `#978`–`#1059` + Current `NEURAL-CTRL-DYN-45`)

| Campaign Stage | Primary Module(s) | Algebraic / Thermodynamic Invariant |
| :--- | :--- | :--- |
| **Butinage & La Ruche Vault** | `la_ruche_vault.py`, `crowdgpt_moe_opendiloco.py`, `volunteer_wsp_workunit_exporter.py` | Content-addressed `Alvéoles`, AGPL-3.0 allowlisted shards, `0.00 EUR / 0.00 USD` compute reciprocity |
| **Preflight & Byzantine Filter** | `marchenko_pastur_covariance_guard.py`, `byzantine_wsp_delta_aggregator.py`, `representation_collapse_wsp_guard.py` | Oracle shrinkage $\Sigma_{\text{shrunk}} = (1-\rho)M + \rho \frac{\text{tr}(M)}{d}I$, Multi-Krum filters poisoned deltas |
| **Sequential Chain & Inter-Round Trophallaxie (`D1`)** | `inter_round_royal_jelly_trophallaxis.py` (`#1056`), `sequential_chain_ties_nullspace_rescuer.py` (`#1052`), `queen_cortex_moe_campaign_governor.py` (`#1051`), `ties_dare_moe_merger.py`, `rank_preserving_nullspace_input_injector.py`, `folded_norm_ece_calibrator.py` | Consecutive chain differencing $\Delta W_d = W_d - W_{d-1}$, associative low-RSS $(s B)(A P_0)$, pre-momentum $P_0$ projection ($M_t = \beta P_0(M_{t-1}) + \Delta_t^{\text{RJ}}$, $\|M_t V_0\|_F = 0$), and $\ker(G_{D_0})$ diagonal-Fisher + folded $1/\tau_{\text{temp}}$ calibration |
| **Kronecker Riemannian Null-Space Transport (`D2`)** | `kronecker_riemannian_nullspace_transport.py` (`#1059`), `natural_gradient_trust_wsp_surgeon.py` (`#1025`) | Damped K-FAC $(G_{\text{cov}} \otimes A_{\text{cov}})$ whitened null-space operators $R_{\text{step}}, R_{\text{proj}}$, isometric layer parallel transport, Theorem 3.7 unconditional descent ($\le 0$), geodesic KL trust bound $\frac{1}{2}\operatorname{tr}(G \Delta W A \Delta W^\top) \le \delta_{\text{KL}}$ |
| **Optimal-Transport Cross-Arm Alignment (`D3`)** | **`optimal_transport_moe_arm_fuser.py`** (`NEURAL-CTRL-DYN-45`), `procrustes_wsp_skill_transfer.py` (`#1023`), `domain_matched_moe_arm_composer.py` (`#1043`) | Exact $O(E^3)$ Hungarian bijection $\pi^*$ + SVD Orthogonal Procrustes $R_{B \to A}$ & LoRA rank alignment $S_r \in O(r)$, Theorem 3.8 exact permutation-transport MoE invariance (`RCD_D3`), $P_0$ orthogonal residual synergy + 2-matmul affine LayerNorm |
| **Causal Attribution & Co-Adaptation** | `prefix_cached_causal_attribution.py`, `self_healing_checkpoint_rescuer.py`, `router_freeze_cross_arm_surgeon.py` | Prefix-cached $L + L(L+1)/2$ LOEO knockouts; anchor layers & combs preserved bitwise |
| **Operculation ANSE & Phéromones Royales** | `inter_round_royal_jelly_trophallaxis.py`, `queen_cortex_moe_campaign_governor.py`, `monotone_wsp_stage_composer.py`, `auto_wsp_campaign_scheduler.py`, `lean4_wsp_theorem_registry.py` | Commit iff $\Delta E < 0$ & $D_0 \le +1.0\%$; atomic bitwise rollback (`1e6` pain); signed `QueenPheromoneDirective`; `188` zero-`sorry` Lean 4 declarations (DOIs `10.5281/zenodo.24553269` & `10.5281/zenodo.24542441`) |
| **Gen-3 Queued Wave (`D4`–`D5`)** | `alveole_mitosis_hessian_nf4_exporter.py` (`D4`), `zk_wsp_federated_hive_verifier.py` (`D5`) | *Alvéole* mitosis + Hessian NF4 `< 6,656 MiB` (`D4`) and `ZK-WSP` multi-hive federation (`D5`), both empirically validated in `PILOT11` |
| **4-Bit NF4 Export & Wald SPRT Canary** | `quantization_compensated_wsp.py`, `auto_open_weight_orchestrator.py`, `shadow_canary_wsp_traffic_router.py` | Hessian-compensated NF4 edge export under `6,656 MiB` VRAM + Wald SPRT shadow canary |

---

## 2. Four-Hour Periodic Verification & Evolution Workflow Stages

Each NeuroController improvement slice progresses through five deterministic local CPU verification stages on a 4-hour periodic cadence (zero remote GPU SSH, zero `.lock.json` locked holdout vault re-evaluation, zero duplicate local pilot processes):

1. **Stage 1 — Local Empirical Ratchet Audit & Slice Triage:**
   - Confirms local CPU isolation (`CUDA_VISIBLE_DEVICES=""`) and checks active local CPU processes (`pgrep -af '[p]ilot[0-9]'` respected `PID 458296` without launching duplicates) while auditing certified receipts (`PILOT1`..`PILOT11`) and leaving every `.lock.json` holdout vault untouched.
   - Syncs merged PR states (`51` merged NeuroController code PRs `#978`–`#1059` + Foundation Paper v4.4.0 `#1054`) and checks out the active `NEURAL-CTRL-DYN-*` branch cleanly off `origin/main`.
2. **Stage 2 — Cleanroom Module, 5-Scenario Test Suite & Tracking Docs Authoring:**
   - Implements the target module in `src/hivemind/swarm/` with AGPL-3.0 copyright header, zero stubs, zero forbidden patterns, and zero network/subprocess egress.
   - Authors `5` deterministic CPU scenarios (`tests/test_*_scenarios.py`, CI-safe `try/except ImportError` guard around optional `gwaya`) targeting `>= 95.0%` standalone line coverage, and updates both `docs/NEURO_CONTROLLER_ROADMAP.md` and `docs/NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md`.
3. **Stage 3 — Independent Scenario Verification, Standalone Coverage, Ruff & 5-Layer Security Gate:**
   - Runs scenario tests alongside repository compliance suites (`test_agpl3_ip_protection.py`, `test_static_numbers_have_artifacts.py`, `test_no_lateral_movement.py`).
   - Measures standalone line coverage (`python -m coverage run` without `COVERAGE_CORE=ctrace`), runs `ruff check`, runs `ZeroStubAudit`, verifies `pr_gatekeeper.forbidden_scan`, checks `OwnerOrderGuardrail`, and validates `PRSecurityScanner` with `0` findings.
4. **Stage 4 — Review-Ready Slice Packaging & Telemetry Receipt Logging:**
   - Stages verified slice files cleanly with owner identity and logs JSONL telemetry receipts (leaving final commit/publication exclusively to the Phase 4 workflow).
5. **Stage 5 — Plan & Roadmap Progress Sync:**
   - Records latest empirical CPU verdicts, standalone coverage, and Gen-3 `D1`–`D5` completion status in tracking documents.

---

## 3. Delivered Status of `NEURAL-CTRL-DYN-43` (`D1`, `#1056`) & `NEURAL-CTRL-DYN-44` (`D2`, `#1059`)

- **`NEURAL-CTRL-DYN-43` (Gen-3 Direction `D1`, Merged `#1056`):** [`src/hivemind/swarm/inter_round_royal_jelly_trophallaxis.py`](../src/hivemind/swarm/inter_round_royal_jelly_trophallaxis.py) + [`tests/test_inter_round_royal_jelly_trophallaxis_scenarios.py`](../tests/test_inter_round_royal_jelly_trophallaxis_scenarios.py) (`97%` standalone coverage).
- **`NEURAL-CTRL-DYN-44` (Gen-3 Direction `D2`, Merged `#1059`):** [`src/hivemind/swarm/kronecker_riemannian_nullspace_transport.py`](../src/hivemind/swarm/kronecker_riemannian_nullspace_transport.py) + [`tests/test_kronecker_riemannian_nullspace_transport_scenarios.py`](../tests/test_kronecker_riemannian_nullspace_transport_scenarios.py) (`99%` standalone coverage).

---

## 4. Implementation Status of `NEURAL-CTRL-DYN-45` (Gen-3 Direction `D3`)

- **Target Module:** [`src/hivemind/swarm/optimal_transport_moe_arm_fuser.py`](../src/hivemind/swarm/optimal_transport_moe_arm_fuser.py) — **Implemented**
- **Scenario Suite:** [`tests/test_optimal_transport_moe_arm_fuser_scenarios.py`](../tests/test_optimal_transport_moe_arm_fuser_scenarios.py) — **Implemented (`5` deterministic CPU scenarios: `S1`..`S5`)**

### Implemented Capabilities & Mathematical Guarantees (Zenodo `10.5281/zenodo.24542441` §6 Direction `D3` / Theorem 3.8 & Lean 4 `RCD_D3`)

1. **`solve_exact_hungarian_assignment(...)` & `compute_cross_arm_expert_similarity_matrix(...)`:**
   - Exact `O(E^3)` Kuhn-Munkres Hungarian assignment solving the optimal `1`-to-`1` bijection `perm` between `Arm_A` and `Arm_B` expert slots using combined probe router-logit correlation, Procrustes-aligned functional expert output cosine similarity, and optional aligned parameter cosine similarity (requiring zero ground-truth permutation labels).
2. **`match_moe_experts_hungarian_procrustes(...)`, `align_lora_rank_basis_procrustes(...)` & `fuse_aligned_moe_layer_weights(...)`:**
   - Computes debiased/linear CKA and SVD Orthogonal Procrustes rotation `R_{B->A}` aligning hidden spaces before functional matching and weight transport (`router_B[perm] @ R_{B->A}`, `W_in_B @ R_{B->A}`, `R_{B->A}^T @ W_out_B`), aligns LoRA factor rank bases `(B_B S, S^T A_B)` via exact `S in O(r)` SVD, prevents random-init cross-term shrinkage via QR-SVD LoRA fusion, projects updates strictly into the general-prose AlphaEdit null-space `P_0`, and applies orthogonal residual synergy across arms.
3. **`execute_verified_ot_moe_arm_fusion(...)` & `verify_ot_moe_fusion_receipt_provenance(...)`:**
   - Executes deterministic CPU cross-arm MoE fusion (`0 MiB` VRAM, `0` gradient steps) across `CrowdV1WithEngram` (`CrowdLoRAMoEMLP`, `CrowdFineGrainedMoEMLP`) and HF-style MoE transformers with optional 2-matmul affine LayerNorm + folded temperature calibration, committing iff `ANSE Delta E < 0` against donor arms and `D0 / worst_domain <= +1.0%` under `ANSE_MAXIMUM_PAIN` (`1e6`) atomic bitwise rollback and SHA-256 provenance receipts.
