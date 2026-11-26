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
     - Trophallaxis TIES sign-consensus election (density_top_p = 0.25)
     - AlphaEdit P_0 = I - U_0 U_0^T prose-nullspace projection & selective ghost sinks (w_B = 0)
     - Closed-form ker(G_D0) diagonal-Fisher RMSNorm/LayerNorm + 1/tau_temp calibration
       (rescuing drifting outer rounds kept=[] before rejection)
          |
          v
  3. Operculation d'Alvéole ANSE & Alvéole Vierge
     - Monotone ANSE ratchet (Delta E < 0 and D0 / worst_domain <= +1.0%)
     - ANSE Maximum Pain (1e6) atomic bitwise rollback on guardrail violation
     - Single-shot SHA-256 disjoint holdout vault verification (Alvéole Vierge)
          |
          v
  4. Phéromones Royales Prochain Round (QueenPheromoneDirective)
     - Freeze converged Anchor Combs (Alvéoles Ancres); sink/reseed pure corruptors
     - Toggle router_frozen (fragile KL mismatch) vs router_unfrozen_coadapted
     - Emit normalized Boltzmann/UCB bee_foraging_domain_weights toward high-deficit domains
          |
          v
  5. 4-Bit NF4 Export & Wald SPRT Canary
     - Diagonal-Hessian NF4 edge blocks under 6,656 MiB 8 GB VRAM ceiling
     - Sequential Wald SPRT canary routing with instant bitwise rollback
```

### Operator Mapping to Cleanroom Modules

| Campaign Stage | Primary Module(s) | Algebraic / Thermodynamic Invariant |
| :--- | :--- | :--- |
| **Butinage & La Ruche Vault** | `la_ruche_vault.py`, `crowdgpt_moe_opendiloco.py`, `volunteer_wsp_workunit_exporter.py` | Content-addressed `Alvéoles`, AGPL-3.0 allowlisted shards, `0.00 EUR / 0.00 USD` compute reciprocity |
| **Preflight & Byzantine Filter** | `marchenko_pastur_covariance_guard.py`, `byzantine_wsp_delta_aggregator.py`, `representation_collapse_wsp_guard.py` | Oracle shrinkage $\Sigma_{\text{shrunk}} = (1-\rho)M + \rho \frac{\text{tr}(M)}{d}I$, Multi-Krum filters $f < (n-2)/2$ poisoned deltas |
| **Trophallaxie & Gelée Royale WSP** | **`queen_cortex_moe_campaign_governor.py`**, `ties_dare_moe_merger.py`, `rank_preserving_nullspace_input_injector.py`, `folded_norm_ece_calibrator.py` | TIES sign election + $P_0$ projection + $\ker(G_{D_0})$ diagonal-Fisher step satisfying $\|G_{D_0} \Delta w\|_\infty < 10^{-4}$ and folded $1/\tau_{\text{temp}}$ |
| **Causal Attribution & Co-Adaptation** | `prefix_cached_causal_attribution.py`, `self_healing_checkpoint_rescuer.py`, `domain_matched_moe_arm_composer.py` | Prefix-cached $L + L(L+1)/2$ LOEO knockouts; anchor layers & combs preserved bitwise |
| **Operculation ANSE & Alvéole Vierge** | **`queen_cortex_moe_campaign_governor.py`**, `monotone_wsp_stage_composer.py`, `auto_wsp_campaign_scheduler.py`, `lean4_wsp_theorem_registry.py` | Commit iff $\Delta E < 0$ & $D_0 \le +1.0\%$; atomic bitwise rollback (`1e6` pain); Lean 4 verified invariants (`0 sorry`) |
| **Phéromones Royales Next-Round Steering** | **`queen_cortex_moe_campaign_governor.py`** (`emit_queen_pheromone_campaign_directive`) | Signed `QueenPheromoneDirective`: `anchor_experts_to_freeze`, `corruptor_experts_to_sink_or_reseed`, `router_training_mode`, Boltzmann/UCB `bee_foraging_domain_weights` |
| **4-Bit NF4 Export & Wald SPRT Canary** | `quantization_compensated_wsp.py`, `auto_open_weight_orchestrator.py`, `shadow_canary_wsp_traffic_router.py` | Hessian-compensated NF4 edge export under `6,656 MiB` VRAM + Wald SPRT shadow canary |

---

## 2. Periodic 4-Hour NeuroController Workflow Stages

The 4-hour periodic NeuroController workflow advances through five gated stages in deterministic sequence:

1. **Stage 1 — CPU Weight-Improvement Ratchet Audit & Slice Triage:**
   - Audits CPU-only (`CUDA_VISIBLE_DEVICES=""`, `MemAvailable >= 12.0 GiB`) experiment directories without re-evaluating certified holdout vaults (`skip=0, 160, 320, 480, 640, 800`, `pilot3a`, `pilot3b`, `pilot4b`, `pilot7`).
   - Checks active process locks (avoiding duplicate CPU runners while `pilot9_llm_guided_qwen25_3b_wsp.py` PID `125930` is running) before triggering downstream ratchets.
   - Syncs merged PR states in `neural_control_backlog.json` and prepares the next pending `NEURAL-CTRL-*` slice on a clean branch off `origin/main`.
2. **Stage 2 — Cleanroom Module, 5-Scenario Test Suite & Repository Docs Authoring:**
   - Implements the target module in `src/hivemind/swarm/` with AGPL-3.0 copyright header, zero stubs, zero forbidden patterns, and zero network/subprocess egress.
   - Authors 5 deterministic CPU scenarios (`tests/test_*_scenarios.py`, CI-safe `try/except ImportError` guard around optional `gwaya`) targeting `>= 95.0%` standalone line coverage, and updates both `docs/NEURO_CONTROLLER_ROADMAP.md` and `docs/NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md`.
3. **Stage 3 — Independent Scenario Verification, Standalone Coverage, Ruff & 5-Layer Security Gate:**
   - Runs scenario tests alongside repository compliance suites (`test_agpl3_ip_protection.py`, `test_static_numbers_have_artifacts.py`, `test_no_lateral_movement.py`).
   - Measures standalone line coverage (`python -m coverage run` without `COVERAGE_CORE=ctrace`), runs `ruff check`, runs `ZeroStubAudit`, verifies `pr_gatekeeper.forbidden_scan`, checks `OwnerOrderGuardrail`, and validates `PRSecurityScanner` with `0` findings.
4. **Stage 4 — Review-Ready Slice Packaging & Telemetry Receipt Logging:**
   - Stages verified slice files cleanly with owner identity and logs JSONL telemetry receipts (leaving final publication and release actions exclusively to the Phase 4 owner workflow).
5. **Stage 5 — Plan & Roadmap Progress Sync:**
   - Records latest empirical T4 / CPU verdicts, standalone coverage, and slice completion status in tracking documents.

---

## 3. Implementation Status of `NEURAL-CTRL-DYN-41`

- **Target Module:** [`src/hivemind/swarm/queen_cortex_moe_campaign_governor.py`](../src/hivemind/swarm/queen_cortex_moe_campaign_governor.py) — **Implemented**
- **Scenario Suite:** [`tests/test_queen_cortex_moe_campaign_governor_scenarios.py`](../tests/test_queen_cortex_moe_campaign_governor_scenarios.py) — **Implemented (5 deterministic CPU scenarios: S1..S5)**
- **Prior Slice (`NEURAL-CTRL-DYN-40`):** [`src/hivemind/swarm/rank_preserving_nullspace_input_injector.py`](../src/hivemind/swarm/rank_preserving_nullspace_input_injector.py) — **Merged (`#1049`)**

### Implemented Capabilities & Mathematical Guarantees (`PILOT1`..`PILOT9` Grounding)

1. **`distill_forager_nectar_to_royal_jelly(base_experts, forager_bee_deltas, p_null_by_expert, *, w_norm_base, normed_unit_probe, head_weight, probe_targets, is_general_mask, ties_density=0.25)`:**
   - Combines TIES sign-consensus aggregation (`trim_and_elect_sign_ties`), AlphaEdit `P_0` prose-nullspace projection, anchor/ghost-sink handling, and closed-form `ker(G_D0)` diagonal-Fisher RMSNorm/LayerNorm calibration (`compute_ker_g_d0_fisher_norm_step`) with `< 1e-4` exact first-order prose chunk-gradient orthogonality (`|G_D0 @ delta_w|_inf`), rescuing discarded/drifted outer rounds in `0 MiB` GPU VRAM.
2. **`emit_queen_pheromone_campaign_directive(expert_causal_gains_nats, expert_prose_harm_nats, router_expert_kl_mismatch, domain_nlls, domain_eces, *, anchor_gain_floor_nats=-0.01, coadapt_kl_ceiling=0.05, temperature=0.25)`:**
   - Issues a SHA-256-signed `QueenPheromoneDirective` partitioning `anchor_experts_to_freeze`, `corruptor_experts_to_sink_or_reseed`, and `active_specialist_experts`, toggling `router_frozen` vs `router_unfrozen_coadapted` on co-adaptation KL fragility, and concentrating normalized Boltzmann/UCB `bee_foraging_domain_weights` on the highest-NLL/ECE specialist domains.
3. **`execute_queen_cortex_campaign_round(...)`:**
   - Runs full closed-loop La Ruche MoE campaign governance (supporting both `CrowdV1WithEngram` / MoE `nn.Module` checkpoints and structured block mappings), auditing raw unguarded rounds against Royal-Jelly distillation under `ANSE Delta E < 0` and `D0 <= +1.0%` (`Operculation d'Alvéole`), enforcing ANSE Maximum Pain (`1e6`) atomic bitwise rollback (`final_state_sha256 == rollback_expected_sha256`), and verifying provenance via `verify_queen_cortex_receipt_provenance`.
