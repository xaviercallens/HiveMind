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

## 4. Roadmap Tasks (`HM-S35-01` .. `HM-S35-08`)

See `ROADMAP.md` §3.25 for the full 5-wave rollout linking `AutoOpenWeightEngine` to `OpenTrainer` post-round hooks (`HM-S35-04`), `ExpertRegistry` canary promotion (`HM-S35-05`), decentralized 8 GB Gamer-Whisper volunteer WSP workunits (`HM-S35-06`), 4-bit NF4/GPTQ quantization compensation (`HM-S35-07`), and Lean 4 formal proofs (`HM-S35-08`).
