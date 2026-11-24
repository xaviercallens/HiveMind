# HiveMind Auto Open-Weight Capability (`Season 35 / HM-S35`)

> **Scope:** Training-free, zero-VRAM (`CUDA_VISIBLE_DEVICES=""` CPU float32) Weight-Space Programming (WSP) for open-weight Sparse MoE and LoRA models across Project HiveMind (`0.00 EUR / 0.00 USD` Non-Monetary Compute-Reciprocity Charter, `AGPL-3.0`).

---

## 1. Certified Empirical Baseline (`PILOT1-WSP-ALPHAEDIT-MOE-2026-11-23`)

Evaluated on `C2_final` (`CROWDGPT-MOE-8X-UPCYCLE-V1` on `ibm-granite/granite-3.1-1b-a400m-instruct`, `24` layers, `8` `sigma_topk` experts) across four pairwise-disjoint evaluation splits (`240` 128-token chunks per split, `0` overlapping chunks, `0` gradient steps, `0 MiB` GPU VRAM):

- **Calibration (`probe_skip120`, `n = 240`):** Identified rogue unscored `expert_3` (`||w1_B|| = 42.50`), prose-harming `{0, 1}`, dual-active `expert_5` (`+0.0278` nats specialist contribution when zeroed), and **beneficial anchor Layer `17`** (`e0_harm_nats = -0.2516`) vs. prose-interfering tail layers `18` (`+0.0142`) and `19` (`+0.0137`).
- **Selection (`selection_skip40`, `n = 240`):** **`11 / 12` pre-registered candidates achieved `PASS`** (`E0 <= +1.0%` B3 guard, `mean_diff < 0` with `95%` bootstrap CI `< 0`, `MOSTLY_CONTENT`):
  - `S1` (`WSP_A1_ln_sink`): `-0.5249` nats (`E0 = -0.057%`).
  - `S2` (`WSP_S2_top14_harm_layers`): `-0.5365` nats (`E0 = +0.396%`, protecting Layer `17`).
  - `S3` (`WSP_S3_svd_plus_router`): `-0.5354` nats with **all `6 / 6` domains negative** (`E0 = -0.296%`).
  - `S4` (`WSP_S4_multigate_only`): **`-0.5529` nats** (`E0 = +0.018%`).
- **Confirmation (`confirm_a2_skip80`, `n = 240`):** `WSP_S4_multigate_only` achieved **`-0.5444` nats** (`95% CI: [-0.6152, -0.4743]`, `E0 = +0.478%`, `PASS` — **`6.35x` deeper** than Amendment A2 `e50_r00`'s `-0.0857` nats).
- **Single-Shot Virgin Holdout Vault (`vault_skip160`, `n = 240`, frozen permanently after one evaluation):** **`mean_diff = -0.5550` nats** (`95% CI: [-0.6270, -0.4837]`, `J = 3.8420` vs. base `4.3970`), **`E0 = +0.119%`**, `E1 = -5.433%`, `E2 = -4.603%`, `E5 = -11.789%`, `E6 = -46.694%`, `E7 = -14.651%` — **`PASS` / `MOSTLY_CONTENT`**.

---

## 2. Generation `S5` ("PASS Synergy") Engine (`src/hivemind/swarm/auto_open_weight_engine.py`)

`WSP_S4_multigate_only` still zeroed `{0, 1, 3}` uniformly across all `24` layers (including beneficial anchor Layer `17`) and did not apply `S3`'s tail-layer QR-SVD prose-nullspace projection or `DYN-14`'s folded final-norm temperature calibration. Generation `S5` combines all four `PASS` mechanisms under a monotone per-stage ANSE ratchet:

1. **`S5.1` (`S2 + S1/S4`):** Leave beneficial anchor layers ($h_\ell \le -0.010$ nats) 100% bitwise untouched; zero pure rogue/prose-harming ghost-sink experts only on interfering layers.
2. **`S5.2` (`S4`):** Preserve dual-active specialists ($\Delta \text{spec} \ge 0.012$ nats) and apply 2-fold OOF SNR-gated ($\text{SNR} \ge 0.25$) router suppression strictly orthogonal to all specialist domain centroids.
3. **`S5.3` (`S3`):** On top prose-interfering tail layers, project active specialist right factors into the Marchenko–Pastur-denoised general-prose null-space $P_0 = I - U_0 U_0^\top$ via rank-preserving QR-SVD.
4. **`S5.4` (`DYN-14`):** Fold optimal $(\lambda_{\text{ln}}, 1 / \tau_{\text{temp}})$ directly into final `LayerNorm` / `RMSNorm` parameters with zero inference latency cost.

---

## 3. CLI Usage (`scripts/hivemind_auto_open_weight.py`)

```bash
# 1. Inspect a Sparse MoE checkpoint (.safetensors or .npz) on CPU (0 MiB GPU VRAM)
python scripts/hivemind_auto_open_weight.py diagnose \
  --checkpoint path/to/moe_buckets.safetensors --json

# 2. Apply offline S5 anchor-protected surgery + folded final-norm calibration
python scripts/hivemind_auto_open_weight.py run-s5-synergy \
  --checkpoint path/to/moe_buckets.safetensors \
  --attribution-json path/to/attribution.json \
  --output path/to/moe_buckets_s5.safetensors --json
```

---

## 4. Season 35 Roadmap Tasks (`HM-S35-01` .. `HM-S35-08`)

| Task ID | Wave | Subsystem / PRs | Deliverable (`Auto Open-Weight Capability`) | Verification & Promotion Gate |
| :--- | :---: | :--- | :--- | :--- |
| **`HM-S35-01`** | **W1** | `src/hivemind/swarm/*`, PRs `#978`–`#986`, `#988`–`#991`, `#993`–`#997`, `#999`–`#1003`, `#1007` (**`23` merged**) | **23-Module Cleanroom Weight-Space Programming Foundation**: AlphaEdit null-space (`wsp_alphaedit_control`), `sigma_topk` gate (`sigma_router_gate_control`), causal layer attribution (`layer_localized_attribution`), QR-SVD mode surgery (`lora_svd_mode_surgery`), harness (`hivemind_neural_control_harness`), runtime guard (`adaptive_steering_guard`), self-healing repair (`self_healing_weight_repair`), entropy-loop steering (`entropy_loop_steering`), Marchenko–Pastur guard (`marchenko_pastur_covariance_guard`), unified controller (`unified_wsp_neural_controller`), skill steering (`decision_skill_weight_steering`), TIES-DARE MoE merger (`ties_dare_moe_merger`), 4-bit quantization compensation (`quantization_compensated_wsp`), LAT circuit breaker (`representation_circuit_breaker`), safetensors doctor (`hivemind_wsp_checkpoint_doctor`), cross-layer anchor nullspace (`cross_layer_residual_nullspace`), Wanda/OBS surgeon (`fisher_kfac_diagonal_surgeon`), AlphaEdit-MEMIT (`rome_memit_knowledge_editor`), DeepSeek-V3 router bias balancer (`moe_load_balance_bias_controller`), folded LN temperature surgeon (`logit_calibration_temperature_surgeon`), orthogonal multi-gate preserver (`orthogonal_multigate_router_surgeon`), monotone WSP stage composer (`monotone_wsp_stage_composer`), and Byzantine Multi-Krum WSP delta aggregator (`byzantine_wsp_delta_aggregator`). | **Delivered (`23/23` merged on `main`)**: `>= 95.0%` standalone line coverage and `5` deterministic CPU scenarios per module. |
| **`HM-S35-02`** | **W1** | `src/hivemind/swarm/auto_open_weight_engine.py`, `scripts/hivemind_auto_open_weight.py`, `docs/AUTO_OPEN_WEIGHT_CAPABILITY_PLAN.md` | **Gen-2 `S5` PASS-Synergy Engine & Offline CLI**: Combines `S2` beneficial-anchor layer protection + `S4` selective ghost-sink & 2-fold OOF SNR-gated orthogonal multi-gate + `S3` tail-layer QR-SVD prose-nullspace projection + `DYN-14` folded final-norm temperature calibration under a monotone per-stage ANSE ratchet ($\Delta E_{\text{stage}} < 0$) and `SingleShotVaultLock`. | `uv run pytest tests/test_auto_open_weight_engine_scenarios.py` (`>= 95.0%` coverage, `E(S5) < E(S4) < E(S2) < E(S1) < E(base)`). |
| **`HM-S35-03`** | **W2** | `pilot2_s5_pass_synergy.py` (T4 CPU float32, `22:00–07:00 CET`) | **Pre-Registered T4 `PILOT2-S5-PASS-SYNERGY` on Fresh Disjoint Shards (`skip=200/240/280/320`)**: Benchmark `WSP_S5_anchor_multigate`, `WSP_S5_anchor_multigate_tail_svd`, and `WSP_S5_full_pass_synergy` against `WSP_S4_multigate_only` on `C2_final` without ever re-touching `vault_skip160` or `skip=0`. | `verify_chunk_splits_disjoint` (`0` overlap across all `9` splits); `E0 <= +1.0%` (target `<= 0.0%` on all `6/6` domains); single-shot `vault_skip320` receipt. |
| **`HM-S35-04`** | **W3** | `src/hivemind/swarm/campaign_orchestrator.py`, `scripts/gcp_t4_crowd_diloco_worker.py` | **Automatic Post-Round WSP Hook in `OpenTrainer` & `DreamPhasePlanner` (`HM-S21-03`)**: Run CPU-only `S5` surgery automatically after every nocturnal DiLoCo / QLoRA outer checkpoint; promote sanitized weights iff $\Delta E < 0$ and guard `PASS`. | Signed A/B lineage receipt (`raw_outer` vs. `s5_sanitized`) in `artifacts/auto_open_weight/`; atomic bitwise fallback on `FAIL`. |
| **`HM-S35-05`** | **W3** | `src/hivemind/core/expert_registry.py` (`HM-S21-02`), `#1006` producer bridge | **Zero-Downtime `ExpertRegistry` & Producer Canary Promotion**: Require a valid `AutoOpenWeightResult` (`verdict == "PASS"`) for `candidate -> shadow -> live` promotion on `hivemind/crowd-v1` and `hivemind/moe-coder`. | Shadow producer passes `15/15` router probes + `0` general-domain regressions before `live` routing. |
| **`HM-S35-06`** | **W4** | `src/hivemind/swarm/byzantine_wsp_delta_aggregator.py`, `scripts/hivemind_quickstart.py` | **Decentralized 8 GB Volunteer WSP Calibration Workunits & Byzantine Multi-Krum Aggregation**: Personal home PCs (`std_8gb <= 6,656 MiB` WDDM, `< 1.5 s` Steam yield, `100%` corporate/Amadeus blocked) compute rank-bounded WSP deltas aggregated via `Multi-Krum` inside AlphaEdit prose null-space $P_0$. | Rejects $f < (n-2)/2$ poisoned/spectral-spike peer deltas; `< 50 MB/h` outbound HTTPS `443` only. |
| **`HM-S35-07`** | **W4** | `src/hivemind/swarm/quantization_compensated_wsp.py` | **4-Bit NF4 / GPTQ Quantization-Compensated Edge Export**: Fold `S5` continuous weight edits into 4-bit quantized blocks with Hessian error compensation for `8 GB` consumer GPUs (`RTX 2070` / laptops). | `< 0.02` nat held-out NLL gap vs. FP32 `S5` checkpoint under `6,656 MiB` VRAM ceiling. |
| **`HM-S35-08`** | **W5** | `lean4/HiveMindSpec.lean` (`Theorems 162–167`) | **Machine-Checked Lean 4 WSP Soundness Proofs (`0 sorry`)**: Formalize AlphaEdit null-space idempotence ($P_0^2 = P_0$), preserved-key invariance ($\Delta W K_0 = 0$), orthogonal multi-gate centroid invariance ($u_{0 \perp S}^\top \mu_e = 0$), and ANSE monotone ratchet descent. | `cd lean4 && lake build` (`0 sorry`, `0` user axioms) + independent axiom-line verification. |

