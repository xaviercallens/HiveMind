# HiveMind NeuroController — Architecture & Implementation Plan

> **Non-Monetary Charter (`AGPL-3.0`):** The HiveMind NeuroController subsystem executes under the **`0.00 EUR / 0.00 USD`** Personal-PC & Volunteer Compute-Reciprocity Charter (`LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`), strictly on CPU (`0 MiB` GPU VRAM, `0` gradient steps), with zero cryptocurrency wallets, zero external payment links, and zero unauthorized lateral egress. Root `ROADMAP.md` is untouched; NeuroController status and milestones live in this document and [`docs/NEURO_CONTROLLER_ROADMAP.md`](./NEURO_CONTROLLER_ROADMAP.md).

---

## 1. End-to-End NeuroController Operator Pipeline

```text
  Preflight MP Guard
   (Ledoit-Wolf Marchenko-Pastur shrinkage, spectral outlier scan, NaN/Inf quarantine)
         |
         v
  Byzantine Null-Space Filter
   (Multi-Krum peer filtering, spectral spike rejection, P_0 = I - U_0 U_0^T projection)
         |
         v
  Causal Anchor Protection + Selective Sinks + OOF Multi-Gate + QR-SVD/ESD + Folded Norm Calibration
   (Leave beneficial anchors bitwise intact; FoMoE w_B=0 ghost sinks; 2-fold OOF SNR gating;
    rank-r QR-SVD & heavy-tail ESD winsorization; <0.001% folded RMSNorm/LayerNorm 1/tau_temp)
         |
         v
  Monotone ANSE Ratchet
   (Per-stage Delta E < 0 & general-prose E0/D0 <= +1.0% guard; atomic bitwise rollback + 1e6 pain)
         |
         v
  Single-Shot Holdout Vault
   (Pairwise SHA-256 split disjointness across probe/selection/confirm/vault + immutable lock)
         |
         v
  4-Bit NF4 Export & SPRT Canary
   (Diagonal-Hessian NF4 edge blocks under 6,656 MiB 8 GB VRAM ceiling + sequential SPRT routing)
```

### Operator Mapping to Cleanroom Modules

| Pipeline Stage | Primary Module(s) | Algebraic / Thermodynamic Invariant |
| :--- | :--- | :--- |
| **Preflight MP Guard** | `marchenko_pastur_covariance_guard.py`, `self_healing_weight_repair.py`, `representation_collapse_wsp_guard.py` | Oracle shrinkage $\Sigma_{\text{shrunk}} = (1-\rho)M + \rho \frac{\text{tr}(M)}{d}I$, $\|G^*\|_2 \le 1.05$, RankMe preservation |
| **Byzantine Null-Space Filter** | `byzantine_wsp_delta_aggregator.py`, `sequential_nullspace_memory.py`, `volunteer_wsp_workunit_exporter.py` | Multi-Krum filters $f < (n-2)/2$ poisoned peers inside orthonormal prose null-space $P_0$ |
| **Causal Attribution & Selective Sinks** | `layer_localized_attribution.py`, `cross_layer_residual_nullspace.py`, `self_healing_checkpoint_rescuer.py` | Beneficial anchor layers ($h_\ell \le -0.010$ nats) remain $0$-bit immutable; rogue `w_B = 0` gives exact $\Delta y = 0$ |
| **OOF Multi-Gate & Cross-Arm Synergy** | `sigma_router_gate_control.py`, `orthogonal_multigate_router_surgeon.py`, `router_freeze_cross_arm_surgeon.py`, **`domain_matched_moe_arm_composer.py`** | 2-fold OOF SNR gating & domain-matched donor assignment inside $P_0$ strictly beating every individual donor arm |
| **Spectral & RepE Weight Surgery** | `lora_svd_mode_surgery.py`, `heavy_tail_esd_weight_surgeon.py`, `thinking_loop_wsp_suppressor.py`, `contrastive_steering_weight_compiler.py` | Thin QR-SVD & Hill-MLE ESD (`pl_alpha in [2, 4]`) + sub-0.02% anti-loop updates inside $P_0$ |
| **Folded Final-Norm Calibration** | `logit_calibration_temperature_surgeon.py`, `folded_norm_ece_calibrator.py`, **`domain_matched_moe_arm_composer.py`** | Exact affine 2-matmul search over $(\lambda_{\text{ln}}, \tau_{\text{temp}})$ folding $1/\tau_{\text{temp}}$ into final `RMSNorm`/`LayerNorm` ($25\times$ matmul speedup) |
| **Monotone ANSE Ratchet & Vault** | `monotone_wsp_stage_composer.py`, `auto_wsp_campaign_scheduler.py`, `wsp_formal_invariant_verifier.py`, `lean4_wsp_theorem_registry.py` | Stage commit iff $\Delta E < 0$ and $E_0/D_0 \le +1.0\%$; Lean 4 machine-checked invariants (`0 sorry`) |
| **4-Bit NF4 Export & SPRT Canary** | `quantization_compensated_wsp.py`, `auto_open_weight_orchestrator.py`, `shadow_canary_wsp_traffic_router.py` | NF4 error absorbed into $P_0$-weighted LoRA/Hessian blocks; Wald SPRT progressive traffic ramp |

---

## 2. Periodic 4-Hour NeuroController Workflow Stages

The autonomous 4-hour NeuroController workflow (`30 2,6,10,14,18,22 * * *`) executes five gated phases in strict sequence:

1. **Stage 1 — T4 Weight-Improvement Ratchet & Slice Triage:**
   - Audits T4 CPU-only (`CUDA_VISIBLE_DEVICES=""`, `MemAvailable >= 11.0 GiB`) experiment directories without re-evaluating certified vaults (`skip=0, 160, 320, 480, 640`, `pilot4b`).
   - Checks auto-advance ratchet preconditions before launching downstream queued experiments (avoiding duplicate CPU runners while `pilot5_c3_auto_wsp.py` is active).
   - Syncs merged PR states in `neural_control_backlog.json` and selects the next pending `NEURAL-CTRL-*` slice on a clean branch off `origin/main`.
2. **Stage 2 — Cleanroom Module, 5-Scenario Test Suite & Repository Docs Authoring:**
   - Implements the target module in `src/hivemind/swarm/` with AGPL-3.0 copyright header, zero stubs, zero forbidden patterns, and zero network/subprocess egress.
   - Authors 5 deterministic CPU scenarios (`tests/test_*_scenarios.py`, CI-safe `try/except ImportError` guard around optional `gwaya`) targeting `>= 95.0%` standalone line coverage, plus syncs both `docs/NEURO_CONTROLLER_ROADMAP.md` and `docs/NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md`.
3. **Stage 3 — Independent Scenario Verification, Standalone Coverage, Ruff & 5-Layer Security Gate:**
   - Runs scenario tests alongside repository compliance suites (`test_agpl3_ip_protection.py`, `test_static_numbers_have_artifacts.py`, `test_no_lateral_movement.py`).
   - Measures standalone line coverage (`coverage run --rcfile=/dev/null`), runs `ruff check`, runs `ZeroStubAudit`, verifies `pr_gatekeeper.forbidden_scan`, and simulates `PRSecurityScanner.audit_pull_request(..., author_email=None)` with `0` blocking findings.
4. **Stage 4 — PR Publication & Telemetry Receipt Logging:**
   - Stages only the verified slice files, signs/commits with owner git identity, opens a clean PR on `xaviercallens/HiveMind`, and logs JSONL telemetry receipts.
5. **Stage 5 — Plan & Roadmap Progress Sync:**
   - Records published PR telemetry, standalone coverage, and latest T4 empirical verdicts into tracking artifacts.

---

## 3. Implementation Status of `NEURAL-CTRL-DYN-38`

- **Target Module:** [`src/hivemind/swarm/domain_matched_moe_arm_composer.py`](../src/hivemind/swarm/domain_matched_moe_arm_composer.py) — **Implemented**
- **Scenario Suite:** [`tests/test_domain_matched_moe_arm_composer_scenarios.py`](../tests/test_domain_matched_moe_arm_composer_scenarios.py) — **Implemented (5 deterministic CPU scenarios: S1..S5)**

### Implemented Capabilities & Mathematical Guarantees

1. **`fast_affine_layernorm_temperature_grid(h_pre_ln, base_ln_w, base_ln_b, cand_ln_w, cand_ln_b, head_weight, targets, is_general_mask, *, eps=1e-5, lambda_grid=..., tau_grid=..., max_general_drift_rel=0.006)`:**
   - Exploits exact affine logit linearity $\text{LayerNorm}(h; w_0 + \lambda \, \Delta w, b_0 + \lambda \, \Delta b) W_{\text{head}}^\top / \tau = (L_0 + \lambda \, dL) / \tau$ so evaluating a 50-point 2D $(\lambda_{\text{ln}}, \tau)$ grid requires only 2 `F.linear` head matmuls per chunk batch ($25\times$ speedup over naive 50-matmul evaluation) with exact `< 1e-5` NLL equivalence and identical optimal $(\lambda_{\text{ln}}, \tau)$.
   - Enforces the strict general-prose calibration guard (`max_general_drift_rel = 0.006`) with minimum-drift fallback and folds $1 / \tau$ directly into final LayerNorm weights/biases via `fold_temperature_into_norm_params`.
2. **`select_domain_winning_experts_across_arms(arms_expert_deltas, probe_domain_losses_by_arm, p_null_by_expert, ...)`:**
   - Grounded in `PILOT4A` and `PILOT5` on `CROWDGPT-MOE-C3` (where Arm `A` excels on `E1_code = -5.212%` and Arm `B` excels on `E2_math = -6.499%` and `E6_stars = -43.162%`), assigns each specialist expert slot to the donor arm achieving the deepest probe domain NLL drop.
   - Zeroes ghost sinks, preserves protected anchors bitwise, and projects surviving LoRA (`w1_A, w2_A, w3_A/B`) and dense/SwiGLU deltas strictly into AlphaEdit prose null-space `P_0` so general-prose $(I - P_0)$ leakage vanishes.
3. **`execute_verified_multi_arm_moe_composition(...)`:**
   - Combines `select_domain_winning_experts_across_arms`, `P_0`-projected multi-arm router blending, and `fast_affine_layernorm_temperature_grid` under an explicit monotone ANSE ratchet (`Delta E < 0` and general-prose / worst-domain guard `<= +1.0%`), committing if and only if the composed MoE strictly beats EVERY individual donor arm on ANSE energy (and every feasible donor on `nll_mean`), with atomic bitwise rollback (`final_state_sha256 == expected_rollback_sha256`) and canonical SHA-256 provenance receipts (`verify_composer_receipt_provenance`).
