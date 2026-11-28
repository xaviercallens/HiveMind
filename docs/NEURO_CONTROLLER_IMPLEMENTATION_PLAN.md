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
     - Conditional Convex P_0 + <1% OFC S_val multi-adapter blending (DYN-51, PILOT15):
       lambda = 0 (zero stripping) when D0 <= +0.5% benign/synergistic transfer;
       smooth ramp to (delta_w @ P_0)_{S_val^perp} only when D0 > trigger_threshold
     - Low-RSS associative AlphaEdit P_0 projection: (scale * B) @ (A @ P_0)
       applied BEFORE outer DiLoCo momentum accumulation (M_t = beta * P_0(M_{t-1}) + Delta_t^{RJ}, D1)
     - Kronecker-factored Riemannian null-space transport (R_step, R_proj, D2 / Theorem 3.7)
       & geodesic KL trust-region clamping on curved deep layers
     - Optimal-Transport Hungarian + Procrustes cross-arm alignment & P_0 fusion (D3 / Theorem 3.8)
       resolving permuted/rotated router-expert co-adaptation mismatch prior to fusion
     - Dynamic Alvéole Mitosis recycling causal-knockout corruptor cells into high-deficit
       specialist domains inside P_0 with BOS sink stability guard (D4)
     - Selective ghost sinks (w_B = 0), OOF multi-gate routing, Empirical-Bayes James-Stein
       rank-k ker(G_D0) intersect S_val^perp Fisher-Norm + signed-tau calibration (DYN-48, DYN-51)
       distinguishing under-confident Base (tau > 1.0) vs over-confident Instruct (tau < 1.0),
       Multi-Scale Spectral-Anisotropy Causal-vs-MLM WSP Operator Regime Router (DYN-50),
       & Effective-DOF BIC/MDL + Paired-Variance Arm Promoter (DYN-52, PILOT12/PILOT16)
       blocking high-DOF selection overfitting (1,536-DOF -> JS-Rank-4 / 1-DOF fallback)
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
     - Diagonal prose-Hessian compensated NF4 edge blocks + P_0 low-rank residuals
       under the 6,656 MiB 8 GB consumer GPU WDDM envelope (D4)
     - Proof-carrying ZK-WSP spot-checks + Lean 4 attestation + Multi-Krum federation (D5)
     - Sequential Wald SPRT canary routing with instant bitwise rollback
```

### Operator Mapping to Cleanroom Modules (`58` Merged Code PRs `#978`–latest + Paper v4.4.0 `#1054` + Current `NEURAL-CTRL-DYN-52`)

| Campaign Stage | Primary Module(s) | Algebraic / Thermodynamic Invariant |
| :--- | :--- | :--- |
| **Butinage & La Ruche Vault** | `la_ruche_vault.py`, `crowdgpt_moe_opendiloco.py`, `volunteer_wsp_workunit_exporter.py` | Content-addressed `Alvéoles`, AGPL-3.0 allowlisted shards, `0.00 EUR / 0.00 USD` compute reciprocity |
| **Preflight & Byzantine Filter** | `marchenko_pastur_covariance_guard.py`, `byzantine_wsp_delta_aggregator.py`, `representation_collapse_wsp_guard.py` | Oracle shrinkage $\Sigma_{\text{shrunk}} = (1-\rho)M + \rho \frac{\text{tr}(M)}{d}I$, Multi-Krum filters poisoned deltas |
| **Effective-DOF BIC/MDL Arm Promotion (`PILOT12` / `PILOT16`)** | **`effective_dof_bic_arm_selector.py`** (`NEURAL-CTRL-DYN-52`) | Computes $k_{\text{eff}} = \sum c_j$ vs $1.0$ vs $d_{\text{full}}$ and exact penalty $\frac{k_{\text{eff}} \ln n}{2 n}$; higher-DOF arms displace lower-DOF incumbents iff improvement exceeds $\Delta\text{BIC} + z \cdot \text{SE}_{\text{paired}}$ under `ANSE Delta E < 0` & `D0 <= +1.0%` |
| **Conditional Multi-Adapter `P_0`/OFC & Signed-`tau` Calibration (`PILOT15`/`PILOT16`)** | `conditional_p0_ofc_adapter_calibrator.py` (`NEURAL-CTRL-DYN-51`) | Early-layer `<1%` OFC vs massive-activation `ofc_iou` confound audit; conditional convex blend $(1-\lambda)\Delta W + \lambda(\Delta W P_0)_{S_{\text{val}}^\perp}$ ($\lambda=0$ when $D_0 \le +0.5\%$); exact linearity $L(\tau,\alpha) = \tau L_{\text{base}} + \alpha L_{\Delta w}$ separating Instruct ($\tau < 1.0$) vs Base ($\tau > 1.0$) inside $\ker(G_{D_0}) \cap S_{\text{val}}^\perp$ |
| **Multi-Scale Spectral & Regime Routing** | `multiscale_spectral_anisotropy_router.py` (`#1078`, `NEURAL-CTRL-DYN-50`) | Profiles `top1_energy_frac`, `top8_energy_frac`, massive-activation ratio & self-suppression `median(|w|)/|w_i|`, and `probe_tokens_per_dof`; routes to `james_stein_rank_k_fisher` (`PILOT12`), `folded_scalar_rmsnorm_only` (`PILOT9`), `p0_lora_plus_folded_norm` (`PILOT3A`/`PILOT15` conditional `P_0`), or `abstain_insufficient_probe_tokens` (`PILOT14`) |
| **Sequential Chain & Inter-Round Trophallaxie (`D1`)** | `inter_round_royal_jelly_trophallaxis.py` (`#1056`), `sequential_chain_ties_nullspace_rescuer.py` (`#1052`), `queen_cortex_moe_campaign_governor.py` (`#1051`), `ties_dare_moe_merger.py`, `rank_preserving_nullspace_input_injector.py`, `folded_norm_ece_calibrator.py`, `james_stein_spectral_norm_surgeon.py` (`#1073`, `NEURAL-CTRL-DYN-48`) | Consecutive chain differencing $\Delta W_d = W_d - W_{d-1}$, associative low-RSS $(s B)(A P_0)$, pre-momentum $P_0$ projection ($M_t = \beta P_0(M_{t-1}) + \Delta_t^{\text{RJ}}$, $\|M_t V_0\|_F = 0$), and Empirical-Bayes James-Stein rank-$k$ $\ker(G_{D_0})$ Fisher norm ($c_j = \max(0, 1 - \sigma_j^2 / (\mu_j^2 + \sigma_j^2/n))$) + 1-DOF folded $1/\tau_{\text{temp}}$ fallback |
| **Kronecker Riemannian Null-Space Transport (`D2`)** | `kronecker_riemannian_nullspace_transport.py` (`#1059`), `natural_gradient_trust_wsp_surgeon.py` (`#1025`) | Damped K-FAC $(G_{\text{cov}} \otimes A_{\text{cov}})$ whitened null-space operators $R_{\text{step}}, R_{\text{proj}}$, isometric layer parallel transport, Theorem 3.7 unconditional descent ($\le 0$), geodesic KL trust bound $\frac{1}{2}\operatorname{tr}(G \Delta W A \Delta W^\top) \le \delta_{\text{KL}}$ |
| **Optimal-Transport Cross-Arm Alignment (`D3`)** | `optimal_transport_moe_arm_fuser.py` (`#1062`), `procrustes_wsp_skill_transfer.py` (`#1023`), `domain_matched_moe_arm_composer.py` (`#1043`) | Exact $O(E^3)$ Hungarian bijection $\pi^*$ + SVD Orthogonal Procrustes $R_{B \to A}$ & LoRA rank alignment $S_r \in O(r)$, Theorem 3.8 exact permutation-transport MoE invariance (`RCD_D3`), $P_0$ orthogonal residual synergy + 2-matmul affine LayerNorm |
| **Dynamic Alvéole Mitosis & Hessian NF4 Export (`D4`)** | `alveole_mitosis_hessian_nf4_exporter.py` (`#1066`), `quantization_compensated_wsp.py` (`#993`), `attention_sink_kv_surgeon.py` (`#1005`) | LOEO corruptor recycling into high-deficit domains inside $P_0$ ($\| \Delta_{\text{mitosis}} V_0 \|_F = 0$), BOS sink guard, diagonal prose-Hessian NF4 low-rank residual inside $P_0$, `<= 6,656 MiB` WDDM envelope (`RCD_D4`) |
| **Causal Attribution & Co-Adaptation** | `prefix_cached_causal_attribution.py`, `self_healing_checkpoint_rescuer.py`, `router_freeze_cross_arm_surgeon.py` | Prefix-cached $L + L(L+1)/2$ LOEO knockouts; anchor layers & combs preserved bitwise |
| **Operculation ANSE & Phéromones Royales** | `inter_round_royal_jelly_trophallaxis.py`, `queen_cortex_moe_campaign_governor.py`, `monotone_wsp_stage_composer.py`, `auto_wsp_campaign_scheduler.py`, `lean4_wsp_theorem_registry.py` | Commit iff $\Delta E < 0$ & $D_0 \le +1.0\%$; atomic bitwise rollback (`1e6` pain); signed `QueenPheromoneDirective`; zero-`sorry` Lean 4 proofs (DOIs `10.5281/zenodo.24553269` & `10.5281/zenodo.24542441`) |
| **Proof-Carrying ZK-WSP Multi-Hive (`D5`) & Wald SPRT Canary** | `zk_wsp_federated_hive_verifier.py` (`#1069`), `auto_open_weight_orchestrator.py`, `shadow_canary_wsp_traffic_router.py` | Fiat-Shamir `ZK-WSP` subspace spot-checks (`RCD_D5`) + `Lean4WSPAttestationReceipt` cross-binding + Multi-Krum in $P_0$ ($n \ge 2f + 3$) + sequential Wald SPRT shadow canary |
| **Unified Gen-3 `D1`–`D5` Synergy Orchestrator** | `reine_cortex_gen3_orchestrator.py` (`#1076`, `NEURAL-CTRL-DYN-49`) | End-to-end deterministic stage ratchet (`D5 -> D3 -> D2 -> D1 -> D4`) enforcing $\Delta E < 0$, $D_0 \le +1.0\%$, `<= 6,656 MiB` WDDM ceiling, per-stage atomic bitwise rollback, and chained SHA-256 Merkle provenance (`PILOT11` `-0.014684` nats, PPL `-1.458%`, `D0 -0.248%`, `PASS`) |

---

## 2. Four-Hour Periodic Verification & Evolution Workflow Stages

Each NeuroController improvement slice progresses through five deterministic local CPU verification stages on a 4-hour periodic cadence (zero remote GPU SSH, zero `.lock.json` locked holdout vault re-evaluation, zero duplicate local pilot processes):

1. **Stage 1 — Local Empirical Ratchet Audit & Slice Triage:**
   - Confirms local CPU isolation (`CUDA_VISIBLE_DEVICES=""`) while auditing certified receipts (`PILOT1`..`PILOT12` + `PILOT13` spec + `PILOT14` ModernBERT-base finding + `PILOT15` Qwen2.5-0.5B-Instruct multi-adapter OFC receipt + `PILOT16` Qwen2.5-1.5B scaling receipt + `PILOT17` Qwen2.5-0.5B-Instruct NF4 D4 Hessian OFC JS-Norm `76.32%` recovery receipt + active `PILOT18` PID `10906` without starting duplicates) and leaving every `.lock.json` holdout vault untouched.
   - Syncs merged PR states (`58` merged NeuroController code PRs `#978`–latest + Foundation Paper v4.4.0 `#1054`) and checks out the active `NEURAL-CTRL-DYN-*` branch cleanly off `origin/main`.
2. **Stage 2 — Cleanroom Module, 5-Scenario Test Suite & Tracking Docs Authoring:**
   - Implements the target module in `src/hivemind/swarm/` with AGPL-3.0 copyright header, zero stubs, zero forbidden patterns, and zero network/subprocess egress.
   - Authors `5` deterministic CPU scenarios (`tests/test_*_scenarios.py`, CI-safe `try/except ImportError` guard around optional `gwaya`) targeting `>= 95.0%` standalone line coverage, and updates both `docs/NEURO_CONTROLLER_ROADMAP.md` and `docs/NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md`.
3. **Stage 3 — Independent Scenario Verification, Standalone Coverage, Ruff & 5-Layer Security Gate:**
   - Runs scenario tests alongside repository compliance suites (`test_agpl3_ip_protection.py`, `test_static_numbers_have_artifacts.py`, `test_no_lateral_movement.py`).
   - Measures standalone line coverage (`python -m coverage run` without `COVERAGE_CORE=ctrace`), runs `ruff check`, runs `ZeroStubAudit`, verifies `pr_gatekeeper.forbidden_scan`, checks `OwnerOrderGuardrail`, and validates `PRSecurityScanner` with `0` findings.
4. **Stage 4 — Review-Ready Slice Packaging & Telemetry Receipt Logging:**
   - Stages verified slice files cleanly with owner identity and logs JSONL telemetry receipts (leaving final commit/publication exclusively to the Phase 4 workflow).
5. **Stage 5 — Plan & Roadmap Progress Sync:**
   - Records latest empirical CPU verdicts, standalone coverage, and slice completion status in tracking documents.

---

## 3. Delivered Status of Prior Merged Slices (`NEURAL-CTRL-DYN-38`..`51`)

- **`NEURAL-CTRL-DYN-38` (`#1040`):** `prefix_cached_causal_attribution.py` (`99%` standalone coverage).
- **`NEURAL-CTRL-DYN-39` (`#1043`):** `domain_matched_moe_arm_composer.py` (`99%` standalone coverage).
- **`NEURAL-CTRL-DYN-40` (`#1049`):** `rank_preserving_nullspace_input_injector.py` (`97.93%` standalone coverage).
- **`NEURAL-CTRL-DYN-41` (`#1051`):** [`src/hivemind/swarm/queen_cortex_moe_campaign_governor.py`](../src/hivemind/swarm/queen_cortex_moe_campaign_governor.py) (`99.28%` standalone coverage).
- **`NEURAL-CTRL-DYN-42` (`#1052`):** [`src/hivemind/swarm/sequential_chain_ties_nullspace_rescuer.py`](../src/hivemind/swarm/sequential_chain_ties_nullspace_rescuer.py) (`100.0%` standalone coverage).
- **`NEURAL-CTRL-DYN-43` (Gen-3 Direction `D1`, Merged `#1056`):** [`src/hivemind/swarm/inter_round_royal_jelly_trophallaxis.py`](../src/hivemind/swarm/inter_round_royal_jelly_trophallaxis.py) + [`tests/test_inter_round_royal_jelly_trophallaxis_scenarios.py`](../tests/test_inter_round_royal_jelly_trophallaxis_scenarios.py) (`97%` standalone coverage).
- **`NEURAL-CTRL-DYN-44` (Gen-3 Direction `D2`, Merged `#1059`):** [`src/hivemind/swarm/kronecker_riemannian_nullspace_transport.py`](../src/hivemind/swarm/kronecker_riemannian_nullspace_transport.py) + [`tests/test_kronecker_riemannian_nullspace_transport_scenarios.py`](../tests/test_kronecker_riemannian_nullspace_transport_scenarios.py) (`99%` standalone coverage).
- **`NEURAL-CTRL-DYN-45` (Gen-3 Direction `D3`, Merged `#1062`):** [`src/hivemind/swarm/optimal_transport_moe_arm_fuser.py`](../src/hivemind/swarm/optimal_transport_moe_arm_fuser.py) + [`tests/test_optimal_transport_moe_arm_fuser_scenarios.py`](../tests/test_optimal_transport_moe_arm_fuser_scenarios.py) (`O(E^3)` Hungarian + Orthogonal Procrustes cross-arm MoE fuser).
- **`NEURAL-CTRL-DYN-46` (Gen-3 Direction `D4`, Merged `#1066`):** [`src/hivemind/swarm/alveole_mitosis_hessian_nf4_exporter.py`](../src/hivemind/swarm/alveole_mitosis_hessian_nf4_exporter.py) + [`tests/test_alveole_mitosis_hessian_nf4_exporter_scenarios.py`](../tests/test_alveole_mitosis_hessian_nf4_exporter_scenarios.py) (Dynamic *Alvéole* Mitosis corruptor recycler & diagonal prose-Hessian 4-bit NF4 edge exporter).
- **`NEURAL-CTRL-DYN-47` (Gen-3 Direction `D5`, Merged `#1069`):** [`src/hivemind/swarm/zk_wsp_federated_hive_verifier.py`](../src/hivemind/swarm/zk_wsp_federated_hive_verifier.py) + [`tests/test_zk_wsp_federated_hive_verifier_scenarios.py`](../tests/test_zk_wsp_federated_hive_verifier_scenarios.py) (Proof-carrying `ZK-WSP` subspace spot-checks, Lean 4 theorem attestation, and Multi-Krum multi-hive federation).
- **`NEURAL-CTRL-DYN-48` (Empirical-Bayes James-Stein Spectral-Shrinkage Surgeon, Merged `#1073`, grounded in `PILOT9`, `PILOT12`, and `PILOT14`):** [`src/hivemind/swarm/james_stein_spectral_norm_surgeon.py`](../src/hivemind/swarm/james_stein_spectral_norm_surgeon.py) + [`tests/test_james_stein_spectral_norm_surgeon_scenarios.py`](../tests/test_james_stein_spectral_norm_surgeon_scenarios.py) (`99.08%` standalone coverage — probe-DOF / spectral-anisotropy / massive-activation guard, top-`k` James-Stein `ker(G_D0)` Fisher-Norm shrinkage, 1-DOF folded-tau fallback, and ANSE atomic bitwise rollback).
- **`NEURAL-CTRL-DYN-49` (Unified Gen-3 `D1`–`D5` Orchestrator, Merged `#1076`, grounded in `PILOT11`):** [`src/hivemind/swarm/reine_cortex_gen3_orchestrator.py`](../src/hivemind/swarm/reine_cortex_gen3_orchestrator.py) + [`tests/test_reine_cortex_gen3_orchestrator_scenarios.py`](../tests/test_reine_cortex_gen3_orchestrator_scenarios.py) (`100.0%` standalone coverage, `410/410` statements — deterministic stage-ratcheted `D5 -> D3 -> D2 -> D1 -> D4` multi-hive campaign orchestration with chained SHA-256 Merkle provenance).
- **`NEURAL-CTRL-DYN-50` (Multi-Scale Spectral-Anisotropy Profiler & Causal-vs-MLM WSP Operator Regime Router, Merged `#1078`, grounded in `PILOT3A/3B`, `PILOT9`, `PILOT12`, `PILOT14`, `PILOT15`):** [`src/hivemind/swarm/multiscale_spectral_anisotropy_router.py`](../src/hivemind/swarm/multiscale_spectral_anisotropy_router.py) + [`tests/test_multiscale_spectral_anisotropy_router_scenarios.py`](../tests/test_multiscale_spectral_anisotropy_router_scenarios.py) (`97.13%` standalone coverage, `22` passed — profiles pre-norm SVD concentration, massive-activation self-suppression, and probe tokens/DOF to select safe JS rank-`k`, 1-DOF folded-tau, conditional `P_0` LoRA, or MLM abstention).
- **`NEURAL-CTRL-DYN-51` (Conditional Convex `P_0`/OFC Multi-Adapter Blender & Signed-`tau` Norm Calibrator, Merged, grounded in `PILOT15` & `PILOT16`):** [`src/hivemind/swarm/conditional_p0_ofc_adapter_calibrator.py`](../src/hivemind/swarm/conditional_p0_ofc_adapter_calibrator.py) + [`tests/test_conditional_p0_ofc_adapter_calibrator_scenarios.py`](../tests/test_conditional_p0_ofc_adapter_calibrator_scenarios.py) (`97.0%` standalone coverage — conditional `lambda = 0` benign preservation vs smooth convex `(delta_w @ P_0)_{S_val^perp}` protection, `<1%` OFC vs massive-activation IoU audit, and exact logit-linearity signed-`tau` + cross-fitted James-Stein norm calibration).

---

## 4. Implementation Status of `NEURAL-CTRL-DYN-52` (Effective-DOF BIC/MDL Arm Selector & Two-Stage Variance-Guarded James-Stein Promoter)

- **Target Module:** [`src/hivemind/swarm/effective_dof_bic_arm_selector.py`](../src/hivemind/swarm/effective_dof_bic_arm_selector.py) — **Implemented**
- **Scenario Suite:** [`tests/test_effective_dof_bic_arm_selector_scenarios.py`](../tests/test_effective_dof_bic_arm_selector_scenarios.py) — **Implemented (`5` deterministic CPU scenarios: `S1`..`S5`)**

### Implemented Capabilities & Mathematical Guarantees (`PILOT12` & `PILOT16`)

1. **Effective Degrees of Freedom & Exact MDL/BIC Penalty (`compute_effective_dof_and_bic_penalty`):**
   - Quantifies effective model complexity as `k_eff = sum(c_j)` for James-Stein shrinkage arms vs `full_dim` (`1,536` in `PILOT16`) for unregularized Full-DOF vs `1.0` for 1-DOF folded-`tau` scalar (and `0.0` for base identity), computing exact per-token MDL/BIC complexity penalty `(k_eff * log(n_sel_tokens)) / (2 * n_sel_tokens)`.
2. **Two-Stage Variance-Guarded James-Stein Promoter (`select_variance_guarded_champion`):**
   - Stage 1 enforces `D0 <= +1.0%` (`general_guard_rel`), risk improvement over base, and zero-SNR contraction detection (`sum(c_j) == 0` -> scalar fallback).
   - Stage 2 sorts feasible arms by ascending `effective_dof` and permits a higher-DOF challenger to displace a lower-DOF incumbent only when `risk(incumbent) - risk(challenger) > (bic(challenger) - bic(incumbent)) + z * paired_se` and `bic_penalized_objective(challenger) < 0`, directly preventing `PILOT16`-style noisy `+0.000287`-nat selection overfitting (`1,536` DOF dropping `-25.6%` to `-0.006150` on confirm while `CrossFit JS-Rank-4` achieved `#1` NLL `-0.007602` & `#1` ECE `-39.21%`).
3. **End-to-End ANSE-Ratcheted Promotion & Bitwise Rollback (`execute_verified_bic_promoted_wsp`):**
   - Fits 1-DOF scalar, cross-fitted James-Stein rank-`k` (`ker(G_D0) intersect S_val^perp`), and Full-DOF control candidates on `probe_chunks`, applies two-stage BIC + paired-variance selection on `selection_chunks`, verifies `ANSE Delta E < 0` and `D0 <= +1.0%` on held-out `eval_chunks` with atomic bitwise rollback (`ANSE_MAXIMUM_PAIN = 1e6`) and canonical SHA-256 provenance receipts (`verify_bic_arm_selector_receipt_provenance`).
