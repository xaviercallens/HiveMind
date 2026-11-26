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
     - Selective ghost sinks (w_B = 0), OOF multi-gate routing & ker(G_D0) Fisher/folded RMSNorm
          |
          v
  3. Operculation d'Alvéole ANSE & Alvéole Vierge
     - Monotone ANSE ratchet (Delta E < 0 and D0 / worst_domain <= +1.0%)
     - ANSE Maximum Pain (1e6) atomic bitwise rollback on guardrail violation
     - Single-shot SHA-256 disjoint holdout vault verification (Alvéole Vierge)
          |
          v
  4. Phéromones Royales Prochain Round (QueenPheromoneDirective)
     - Freeze converged Anchor Combs (Alvéoles Ancres); sink/recycle corruptors (Mitosis D4)
     - Toggle router_frozen (fragile KL mismatch) vs router_unfrozen_coadapted / OT alignment (D3)
     - Emit normalized Boltzmann/UCB bee_foraging_domain_weights toward high-deficit domains
          |
          v
  5. 4-Bit NF4 Edge Export, Proof-Carrying ZK-WSP & Wald SPRT Canary
     - Prose-Hessian compensated NF4 edge blocks under 6,656 MiB 8 GB VRAM ceiling (D4)
     - Proof-carrying ZK-WSP spot-checks + Lean 4 attestation receipts (D5)
     - Sequential Wald SPRT canary routing with instant bitwise rollback
```

### Operator Mapping to Cleanroom Modules (Gen-2 Delivered & Gen-3 Planned)

| Campaign Stage | Primary Module(s) | Algebraic / Thermodynamic Invariant |
| :--- | :--- | :--- |
| **Butinage & La Ruche Vault** | `la_ruche_vault.py`, `crowdgpt_moe_opendiloco.py`, `volunteer_wsp_workunit_exporter.py` | Content-addressed `Alvéoles`, AGPL-3.0 allowlisted shards, `0.00 EUR / 0.00 USD` compute reciprocity |
| **Preflight & Byzantine Filter** | `marchenko_pastur_covariance_guard.py`, `byzantine_wsp_delta_aggregator.py`, `representation_collapse_wsp_guard.py` | Oracle shrinkage $\Sigma_{\text{shrunk}} = (1-\rho)M + \rho \frac{\text{tr}(M)}{d}I$, Multi-Krum filters $f < (n-2)/2$ poisoned deltas |
| **Sequential Chain Trophallaxie & Gelée Royale WSP** | **`sequential_chain_ties_nullspace_rescuer.py`**, `queen_cortex_moe_campaign_governor.py`, `ties_dare_moe_merger.py`, `rank_preserving_nullspace_input_injector.py`, `folded_norm_ece_calibrator.py` | Consecutive chain differencing $\Delta W_d = W_d - W_{d-1}$, TIES sign election, associative low-RSS $(s B)(A P_0)$ in $O(r d_{\text{in}}^2 + d_{\text{out}} r d_{\text{in}})$, and safe folded $1/\tau_{\text{temp}}$ RMSNorm calibration |
| **Causal Attribution & Co-Adaptation** | `prefix_cached_causal_attribution.py`, `self_healing_checkpoint_rescuer.py`, `domain_matched_moe_arm_composer.py` | Prefix-cached $L + L(L+1)/2$ LOEO knockouts; anchor layers & combs preserved bitwise |
| **Operculation ANSE & Alvéole Vierge** | `monotone_wsp_stage_composer.py`, `auto_wsp_campaign_scheduler.py`, `lean4_wsp_theorem_registry.py` | Commit iff $\Delta E < 0$ & $D_0 \le +1.0\%$; atomic bitwise rollback (`1e6` pain); Lean 4 verified invariants (`0 sorry`, DOI `10.5281/zenodo.24542441`) |
| **Gen-3 Next Wave (`D1`–`D5`)** | `inter_round_royal_jelly_trophallaxis.py` (`D1`), `kronecker_riemannian_nullspace_transport.py` (`D2`), `optimal_transport_moe_arm_fuser.py` (`D3`), `alveole_mitosis_hessian_nf4_exporter.py` (`D4`), `zk_wsp_federated_hive_verifier.py` (`D5`) | Inter-round pheromone propagation (`D1`), Riemannian Fisher-Rao null-space transport (`D2`), Hungarian+Procrustes cross-arm fusion (`D3`), *Alvéole* mitosis + Hessian NF4 `< 6,656 MiB` (`D4`), and `ZK-WSP` multi-hive federation (`D5`) |
| **4-Bit NF4 Export & Wald SPRT Canary** | `quantization_compensated_wsp.py`, `auto_open_weight_orchestrator.py`, `shadow_canary_wsp_traffic_router.py` | Hessian-compensated NF4 edge export under `6,656 MiB` VRAM + Wald SPRT shadow canary |

---

## 2. Gated NeuroController Verification & Evolution Stages

Each NeuroController improvement slice advances through five deterministic local CPU verification stages (zero remote GPU access, zero locked vault re-evaluation):

1. **Stage 1 — Local Empirical Ratchet Audit & Slice Triage:**
   - Audits local CPU-only (`CUDA_VISIBLE_DEVICES=""`) empirical receipts (`PILOT1`..`PILOT10`) while keeping every single-shot holdout vault lock (`.lock.json`) immutable.
   - Syncs merged PR states in the NeuroController backlog and selects the next cleanroom `NEURAL-CTRL-DYN-*` capability grounded in measured MoE bottlenecks.
2. **Stage 2 — Cleanroom Module, 5-Scenario Test Suite & Repository Docs Authoring:**
   - Implements the target module in `src/hivemind/swarm/` with AGPL-3.0 copyright header, zero stubs, zero forbidden patterns, and zero network/subprocess egress.
   - Authors `5` deterministic CPU scenarios (`tests/test_*_scenarios.py`, CI-safe `try/except ImportError` guard around optional `gwaya`) targeting `>= 95.0%` standalone line coverage, and updates both `docs/NEURO_CONTROLLER_ROADMAP.md` and `docs/NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md`.
3. **Stage 3 — Independent Scenario Verification, Standalone Coverage, Ruff & 5-Layer Security Gate:**
   - Runs scenario tests alongside repository compliance suites (`test_agpl3_ip_protection.py`, `test_static_numbers_have_artifacts.py`, `test_no_lateral_movement.py`).
   - Measures standalone line coverage (`python -m coverage run` without `COVERAGE_CORE=ctrace`), runs `ruff check`, runs `ZeroStubAudit`, verifies `pr_gatekeeper.forbidden_scan`, checks `OwnerOrderGuardrail`, and validates `PRSecurityScanner` with `0` findings.
4. **Stage 4 — Review-Ready Slice Packaging & Telemetry Receipt Logging:**
   - Stages verified slice files cleanly with owner identity and logs JSONL telemetry receipts.
5. **Stage 5 — Roadmap & Implementation Progress Sync:**
   - Records latest verified coverage, empirical holdout verdicts, and Gen-3 `D1`–`D5` progression across tracking documents.

---

## 3. Implementation Status of `NEURAL-CTRL-DYN-42`

- **Target Module:** [`src/hivemind/swarm/sequential_chain_ties_nullspace_rescuer.py`](../src/hivemind/swarm/sequential_chain_ties_nullspace_rescuer.py) — **Implemented**
- **Scenario Suite:** [`tests/test_sequential_chain_ties_nullspace_rescuer_scenarios.py`](../tests/test_sequential_chain_ties_nullspace_rescuer_scenarios.py) — **Implemented (`5` deterministic CPU scenarios: `S1`..`S5`)**
- **Prior Slices:** `NEURAL-CTRL-DYN-40` ([`rank_preserving_nullspace_input_injector.py`](../src/hivemind/swarm/rank_preserving_nullspace_input_injector.py), merged `#1049`) and `NEURAL-CTRL-DYN-41` (`queen_cortex_moe_campaign_governor.py`, merged `#1051`).

### Implemented Capabilities & Mathematical Guarantees (`PILOT7` / `PILOT8` Grounding)

1. **`extract_incremental_chain_task_vectors(chain_lora_factors, *, p_null=None)`:**
   - Extracts pure step-$d$ domain increments $\Delta W_d = W_d - W_{d-1}$ from consecutive sequentially trained LoRA chain checkpoints `(B_d, A_d, scale_d)`. When `p_null` is provided, computes each cumulative step directly inside $P_0$ via associative low-RSS factorization without ever materializing unprojected dense $B_d A_d$.
2. **`factorized_associative_nullspace_delta(b_scaled, a_mat, p_null)`:**
   - Evaluates $(s B)(A P_0)$ in $O(r d_{\text{in}}^2 + d_{\text{out}} r d_{\text{in}})$ FLOPs instead of dense $O(d_{\text{out}} d_{\text{in}}^2)$, verified numerically equivalent (`< 1e-5` max abs diff) while cutting peak intermediate RAM for multi-billion-parameter MoEs (`-64%` RSS measured in `PILOT8`).
3. **`ties_elect_and_project_increments(task_vectors, p_null, *, density=0.25, step_weights=None)`:**
   - Trims per-step increments to top-`density` magnitude, elects coordinatewise consensus signs (cancelling conflicting step oscillations cleanly), averages sign-aligned entries, and right-projects into the prose null-space $P_0$.
4. **`execute_verified_chain_rescue(...)` & `verify_chain_rescue_receipt_provenance(...)`:**
   - Combines optional peak donor projection, TIES-elected incremental chain vectors in $P_0$, and safe folded RMSNorm temperature calibration (`tau_temp >= 1.0`, zero inference overhead) under strict monotone `ANSE Delta E < 0` and `D0 / worst_domain <= +1.0%` guards, enforcing atomic bitwise rollback (`final_state_sha256 == initial_state_sha256`, `anse_energy = 1e6`) on any infeasible guard and returning a canonical SHA-256 provenance receipt.
