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
     - Dynamic Alvéole Mitosis recycling causal-knockout corruptor cells into high-deficit
       specialist domains inside P_0 with BOS sink stability guard (D4)
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
     - Diagonal prose-Hessian compensated NF4 edge blocks + P_0 low-rank residuals
       under the 6,656 MiB 8 GB consumer GPU WDDM envelope (D4)
     - Proof-carrying ZK-WSP spot-checks + Lean 4 attestation + Multi-Krum federation (D5)
     - Sequential Wald SPRT canary routing with instant bitwise rollback
```

### Operator Mapping to Cleanroom Modules (`54` Merged PRs `#978`–`#1069` + Current `NEURAL-CTRL-DYN-49`)

| Campaign Stage | Primary Module(s) | Algebraic / Thermodynamic Invariant |
| :--- | :--- | :--- |
| **Butinage & La Ruche Vault** | `la_ruche_vault.py`, `crowdgpt_moe_opendiloco.py`, `volunteer_wsp_workunit_exporter.py` | Content-addressed `Alvéoles`, AGPL-3.0 allowlisted shards, `0.00 EUR / 0.00 USD` compute reciprocity |
| **Preflight & Byzantine Filter** | `marchenko_pastur_covariance_guard.py`, `byzantine_wsp_delta_aggregator.py`, `representation_collapse_wsp_guard.py` | Oracle shrinkage $\Sigma_{\text{shrunk}} = (1-\rho)M + \rho \frac{\text{tr}(M)}{d}I$, Multi-Krum filters poisoned deltas |
| **Sequential Chain & Inter-Round Trophallaxie (`D1`)** | `inter_round_royal_jelly_trophallaxis.py` (`#1056`), `sequential_chain_ties_nullspace_rescuer.py` (`#1052`), `queen_cortex_moe_campaign_governor.py` (`#1051`), `ties_dare_moe_merger.py`, `rank_preserving_nullspace_input_injector.py`, `folded_norm_ece_calibrator.py` | Consecutive chain differencing $\Delta W_d = W_d - W_{d-1}$, associative low-RSS $(s B)(A P_0)$, pre-momentum $P_0$ projection ($M_t = \beta P_0(M_{t-1}) + \Delta_t^{\text{RJ}}$, $\|M_t V_0\|_F = 0$), and $\ker(G_{D_0})$ diagonal-Fisher + folded $1/\tau_{\text{temp}}$ calibration |
| **Kronecker Riemannian Null-Space Transport (`D2`)** | `kronecker_riemannian_nullspace_transport.py` (`#1059`), `natural_gradient_trust_wsp_surgeon.py` (`#1025`) | Damped K-FAC $(G_{\text{cov}} \otimes A_{\text{cov}})$ whitened null-space operators $R_{\text{step}}, R_{\text{proj}}$, isometric layer parallel transport, Theorem 3.7 unconditional descent ($\le 0$), geodesic KL trust bound $\frac{1}{2}\operatorname{tr}(G \Delta W A \Delta W^\top) \le \delta_{\text{KL}}$ |
| **Optimal-Transport Cross-Arm Alignment (`D3`)** | `optimal_transport_moe_arm_fuser.py` (`#1062`), `procrustes_wsp_skill_transfer.py` (`#1023`), `domain_matched_moe_arm_composer.py` (`#1043`) | Exact $O(E^3)$ Hungarian bijection $\pi^*$ + SVD Orthogonal Procrustes $R_{B \to A}$ & LoRA rank alignment $S_r \in O(r)$, Theorem 3.8 exact permutation-transport MoE invariance (`RCD_D3`), $P_0$ orthogonal residual synergy + 2-matmul affine LayerNorm |
| **Dynamic Alvéole Mitosis & Hessian NF4 Export (`D4`)** | `alveole_mitosis_hessian_nf4_exporter.py` (`#1066`), `quantization_compensated_wsp.py` (`#993`), `attention_sink_kv_surgeon.py` (`#1005`) | LOEO corruptor recycling into high-deficit domains inside $P_0$ ($\| \Delta_{\text{mitosis}} V_0 \|_F = 0$), BOS sink guard, diagonal prose-Hessian NF4 low-rank residual inside $P_0$, `<= 6,656 MiB` WDDM envelope (`RCD_D4`) |
| **Causal Attribution & Co-Adaptation** | `prefix_cached_causal_attribution.py`, `self_healing_checkpoint_rescuer.py`, `router_freeze_cross_arm_surgeon.py` | Prefix-cached $L + L(L+1)/2$ LOEO knockouts; anchor layers & combs preserved bitwise |
| **Operculation ANSE & Phéromones Royales** | `inter_round_royal_jelly_trophallaxis.py`, `queen_cortex_moe_campaign_governor.py`, `monotone_wsp_stage_composer.py`, `auto_wsp_campaign_scheduler.py`, `lean4_wsp_theorem_registry.py` | Commit iff $\Delta E < 0$ & $D_0 \le +1.0\%$; atomic bitwise rollback (`1e6` pain); signed `QueenPheromoneDirective`; zero-`sorry` Lean 4 proofs (DOIs `10.5281/zenodo.24553269` & `10.5281/zenodo.24542441`) |
| **Proof-Carrying ZK-WSP Multi-Hive (`D5`) & Wald SPRT Canary** | `zk_wsp_federated_hive_verifier.py` (`#1069`), `auto_open_weight_orchestrator.py`, `shadow_canary_wsp_traffic_router.py` | Fiat-Shamir `ZK-WSP` subspace spot-checks (`RCD_D5`) + `Lean4WSPAttestationReceipt` cross-binding + Multi-Krum in $P_0$ ($n \ge 2f + 3$) + sequential Wald SPRT shadow canary |
| **Unified Gen-3 `D1`–`D5` Synergy Orchestrator** | **`reine_cortex_gen3_orchestrator.py`** (`NEURAL-CTRL-DYN-49`) | End-to-end deterministic stage ratchet (`D5 -> D3 -> D2 -> D1 -> D4`) enforcing $\Delta E < 0$, $D_0 \le +1.0\%$, `<= 6,656 MiB` WDDM ceiling, per-stage atomic bitwise rollback, and chained SHA-256 Merkle provenance (`PILOT11` `-0.014684` nats, PPL `-1.458%`, `D0 -0.248%`, `PASS`) |

---

## 2. Four-Hour Periodic Verification & Evolution Workflow Stages

Each NeuroController improvement slice progresses through five deterministic local CPU verification stages on a 4-hour periodic cadence (zero remote GPU SSH, zero `.lock.json` locked holdout vault re-evaluation, zero duplicate local pilot processes):

1. **Stage 1 — Local Empirical Ratchet Audit & Slice Triage:**
   - Confirms local CPU isolation (`CUDA_VISIBLE_DEVICES=""`) and verifies active local CPU processes (`pgrep -af '[p]ilot[0-9]'` returned `NO_ACTIVE_PILOT_PROCESS`) while auditing certified receipts (`PILOT1`..`PILOT12` + `PILOT13` spec + `PILOT14` ModernBERT-base finding) and leaving every `.lock.json` holdout vault untouched.
   - Syncs merged PR states (`54` merged NeuroController PRs `#978`–`#1069` incl. Foundation Paper v4.4.0 `#1054`) and checks out the active `NEURAL-CTRL-DYN-*` branch cleanly off `origin/main`.
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

## 3. Delivered Status of Prior Merged Slices (`NEURAL-CTRL-DYN-38`..`47`)

- **`NEURAL-CTRL-DYN-38` (`#1040`):** `prefix_cached_causal_attribution.py` (`99%` standalone coverage).
- **`NEURAL-CTRL-DYN-39` (`#1043`):** `domain_matched_moe_arm_composer.py` (`99%` standalone coverage).
- **`NEURAL-CTRL-DYN-40` (`#1049`):** `rank_preserving_nullspace_input_injector.py` (`97.93%` standalone coverage).
- **`NEURAL-CTRL-DYN-41` (`#1051`):** [`src/hivemind/swarm/queen_cortex_moe_campaign_governor.py`](../src/hivemind/swarm/queen_cortex_moe_campaign_governor.py) (`99.28%` standalone coverage).
- **`NEURAL-CTRL-DYN-42` (`#1052`):** [`src/hivemind/swarm/sequential_chain_ties_nullspace_rescuer.py`](../src/hivemind/swarm/sequential_chain_ties_nullspace_rescuer.py) (`100.0%` standalone coverage).
- **`NEURAL-CTRL-DYN-43` (Gen-3 Direction `D1`, Merged `#1056`):** [`src/hivemind/swarm/inter_round_royal_jelly_trophallaxis.py`](../src/hivemind/swarm/inter_round_royal_jelly_trophallaxis.py) + [`tests/test_inter_round_royal_jelly_trophallaxis_scenarios.py`](../tests/test_inter_round_royal_jelly_trophallaxis_scenarios.py) (`97%` standalone coverage).
- **`NEURAL-CTRL-DYN-44` (Gen-3 Direction `D2`, Merged `#1059`):** [`src/hivemind/swarm/kronecker_riemannian_nullspace_transport.py`](../src/hivemind/swarm/kronecker_riemannian_nullspace_transport.py) + [`tests/test_kronecker_riemannian_nullspace_transport_scenarios.py`](../tests/test_kronecker_riemannian_nullspace_transport_scenarios.py) (`99%` standalone coverage).
- **`NEURAL-CTRL-DYN-45` (Gen-3 Direction `D3`, Merged `#1062`):** [`src/hivemind/swarm/optimal_transport_moe_arm_fuser.py`](../src/hivemind/swarm/optimal_transport_moe_arm_fuser.py) + [`tests/test_optimal_transport_moe_arm_fuser_scenarios.py`](../tests/test_optimal_transport_moe_arm_fuser_scenarios.py) (`O(E^3)` Hungarian + Orthogonal Procrustes cross-arm MoE fuser).
- **`NEURAL-CTRL-DYN-46` (Gen-3 Direction `D4`, Merged `#1066`):** [`src/hivemind/swarm/alveole_mitosis_hessian_nf4_exporter.py`](../src/hivemind/swarm/alveole_mitosis_hessian_nf4_exporter.py) + [`tests/test_alveole_mitosis_hessian_nf4_exporter_scenarios.py`](../tests/test_alveole_mitosis_hessian_nf4_exporter_scenarios.py) (Dynamic *Alvéole* Mitosis corruptor recycler & diagonal prose-Hessian 4-bit NF4 edge exporter).
- **`NEURAL-CTRL-DYN-47` (Gen-3 Direction `D5`, Merged `#1069`):** [`src/hivemind/swarm/zk_wsp_federated_hive_verifier.py`](../src/hivemind/swarm/zk_wsp_federated_hive_verifier.py) + [`tests/test_zk_wsp_federated_hive_verifier_scenarios.py`](../tests/test_zk_wsp_federated_hive_verifier_scenarios.py) (Proof-carrying `ZK-WSP` subspace spot-checks, Lean 4 theorem attestation, and Multi-Krum federation).

---

## 4. Implementation Status of `NEURAL-CTRL-DYN-49` (Unified Gen-3 `D1`–`D5` Orchestrator)

- **Target Module:** [`src/hivemind/swarm/reine_cortex_gen3_orchestrator.py`](../src/hivemind/swarm/reine_cortex_gen3_orchestrator.py) — **Implemented (`100.0%` standalone line coverage, `410/410` statements)**
- **Scenario Suite:** [`tests/test_reine_cortex_gen3_orchestrator_scenarios.py`](../tests/test_reine_cortex_gen3_orchestrator_scenarios.py) — **Implemented (`5` deterministic CPU scenarios: `S1`..`S5`, `0` ruff errors, `0` security findings)**

### Implemented Capabilities & Mathematical Guarantees (`PILOT11` Gen-3 `D1`–`D5` Synergy)

1. **Five-Direction Ordered Stage Composition (`execute_gen3_five_direction_campaign`):**
   - Executes `D5` (`ZK-WSP` Byzantine multi-hive intake) -> `D3` (Hungarian/Procrustes cross-arm MoE fusion) -> `D2` (Kronecker Riemannian `P_0` transport) -> `D1` (inter-round Royal-Jelly trophallaxis & `QueenPheromoneDirective` emission) -> `D4` (Dynamic *Alvéole* mitosis + diagonal prose-Hessian 4-bit NF4 export `<= 6,656 MiB` WDDM envelope).
2. **Per-Stage ANSE Thermodynamic Ratchet & Atomic Bitwise Rollback:**
   - Commits each stage iff `ANSE Delta E < 0`, general-prose drift `D0 <= +1.0%`, and WDDM footprint `<= 6,656 MiB` (for `D4`). Regressive or guard-violating stages trigger `ANSE_MAXIMUM_PAIN` (`1e6`) atomic bitwise rollback while preserving all beneficial stages; if all stages fail guard, the full pipeline restores the baseline checkpoint bitwise (`final_state_sha256 == base_state_sha256`).
3. **Chained SHA-256 Merkle Provenance (`compute_merkle_root_sha256`, `verify_gen3_campaign_merkle_chain`, `verify_gen3_campaign_receipt_provenance`):**
   - Links every stage entry (`stage_ledger[0..4]`) via `prev_stage_merkle_sha256` from `GENESIS_MERKLE_SHA256`, binds every stage to its Lean 4 theorem ID and sub-receipt SHA-256, verifies `QueenPheromoneDirective` signatures, and seals the campaign receipt with a deterministic binary Merkle root `merkle_root_sha256`.
