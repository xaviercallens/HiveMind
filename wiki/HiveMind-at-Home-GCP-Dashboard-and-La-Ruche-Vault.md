# 📡 HiveMind@home, GCP Cloud Command Deck & 🐝 La Ruche Community MoE Model Vault (`v2.6.0`)

> **Strictly Non-Monetary Compute-Reciprocity Charter (`0.00 EUR / 0.00 USD`):**
> All curated dataset shards, trained QLoRA domain experts, and merged Branch-Train-Mix (BTM) Sparse Mixture-of-Experts (MoE) checkpoints in **La Ruche** are 100% open-source and freely available to the community.

---

## 1. Permanent Public GCP Cloud Run Deployment & Cost Optimization (`$0.00/mo` Idle)

`HiveMind@home`, the **`La Ruche` Community MoE Model Vault**, and the **Federated AI Training & Sparse MoE Inference Gateway** are permanently deployed on **Google Cloud Run** (`gen-lang-client-0625573011`, `us-central1`) with automatic Google-managed TLS certificates and public DNS:

- **Primary Public HTTPS DNS URL:** [`https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app`](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app)
- **Canonical Regional Cloud Run DNS URL:** [`https://hivemind-at-home-ruche-1003063861791.us-central1.run.app`](https://hivemind-at-home-ruche-1003063861791.us-central1.run.app)
- **SocrateAI GCP Low-Cost Cloud Storage (`La Ruche` Pretrained MoE Buckets):**
  - **Public `La Ruche` GCS Bucket (`us-central1` Standard Always-Free Tier):** `gs://socrateai-hivemind-la-ruche-vault`
  - **Direct Public HTTPS GCS Model URL:** [`https://storage.googleapis.com/socrateai-hivemind-la-ruche-vault/community_moe_release/hivemind_la_ruche_community_moe.safetensors`](https://storage.googleapis.com/socrateai-hivemind-la-ruche-vault/community_moe_release/hivemind_la_ruche_community_moe.safetensors)
  - **Direct Public HTTPS GCS Manifest URL:** [`https://storage.googleapis.com/socrateai-hivemind-la-ruche-vault/community_moe_release/ruche_manifest.json`](https://storage.googleapis.com/socrateai-hivemind-la-ruche-vault/community_moe_release/ruche_manifest.json)
  - **SocrateAI DataLake Mirror URI:** `gs://socrateai-datalake-gen-lang-client-0625573011/hivemind_la_ruche_vault`
- **Scale-to-Zero Cost Optimization (`OPTIMAL_SCALE_TO_ZERO_AND_SPOT_ONLY`):**
  - `--min-instances=0 --max-instances=2 --cpu=1 --memory=512Mi --cpu-throttling`
  - **Idle Monthly Cost:** **`$0.00 USD / month`** (scales to `0` containers when idle; CPU is throttled outside active HTTP requests and covered by the GCP Cloud Run Always-Free Tier of 2M requests / 180,000 vCPU-s / 360,000 GiB-s per month).
  - **Container Footprint:** Ultra-lean `python:3.12-slim` + `numpy` + `safetensors` (`safe_open`) + `fastapi` (`1.4 MB` source & `.safetensors` context, `< 90 MiB` resident RAM, `< 1s` cold start).
  - **Fleet-Wide GCP Cost Audit:** All 30 Cloud Run services in `gen-lang-client-0625573011` enforce `minScale=0`, non-active workstation VMs (`xavuntu-desktop-1791007070`, `instance-20260624-120009`) are `TERMINATED` (`$0` compute), and active GPU compute uses `SPOT` preemption (`socreateai-agora-hermes-node1`).

---

## 2. End-User Public HTTPS API: Retrieve `La Ruche` MoE, Run Inference & Join Federated Training

### A. Retrieve the Merged 8-Domain Community MoE Model (`.safetensors`) over HTTPS
```bash
# 1. Inspect the La Ruche Community MoE Manifest
curl -sL https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/la-ruche/manifest | jq .

# 2. Download the merged 8-domain Community BTM Sparse MoE weights (661,688 bytes)
curl -fSL https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/la-ruche/download/community-moe.safetensors \
  -o hivemind_la_ruche_community_moe.safetensors

# 3. Verify the SHA-256 cryptographic digest (expected: aadd48b8164b440c0eedf63865a2a46a87b07d1609d7e01275e14add7aeeb485)
sha256sum hivemind_la_ruche_community_moe.safetensors

# 4. Download any individual Honeycomb Cell (Alvéole) QLoRA expert (e.g. code_python, formal_lean4, physics_sim)
curl -fSL https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/la-ruche/download/cell/code_python \
  -o alveole_code_python.safetensors
```

### B. Run Live Top-2 Sparse MoE Inference (`SigmaMoE-PeriLN + ThreeTierHotColdHierarchy`)
```bash
# Native HiveMind Sparse MoE Inference API
curl -sX POST https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/v1/moe/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Optimize a PyTorch tensor kernel and prove Lean 4 formal theorem soundness", "top_k": 2}' | jq .

# OpenAI-Compatible Chat Completions API (/v1/chat/completions)
curl -sX POST https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hivemind-la-ruche-moe-v2.6",
    "messages": [{"role": "user", "content": "Simulate symplectic Hamiltonian astrophysics orbit and E8 lattice wire quantization in Rust QUIC"}],
    "top_k": 2
  }' | jq .
```

### C. Participate in Federated AI Training (Request Workunit Lease & Submit Outer Delta)
```bash
# 1. Request a signed federated QLoRA / DiLoCo training workunit lease
curl -sX POST https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/v1/federated/workunit/request \
  -H "Content-Type: application/json" \
  -d '{"contributor_handle": "@community-gpu-node", "hardware_device": "NVIDIA RTX 5060 (8GB VRAM)", "preferred_domain": "code_python"}' | jq .

# 2. Submit trained rank-4 QLoRA outer delta for Two-Statistic Spot-Check & Quorum Admission
curl -sX POST https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/v1/federated/workunit/submit \
  -H "Content-Type: application/json" \
  -d '{
    "workunit_id": "WU-S10-CODE_PYTHON-0001",
    "lease_id": "LEASE-DEMO",
    "lease_version_tag": 1,
    "contributor_handle": "@community-gpu-node",
    "domain_name": "code_python",
    "wire_codec": "e8_2bit",
    "train_nll_before": 5.53,
    "train_nll_after": 5.14,
    "holdout_nll_before": 5.49,
    "holdout_nll_after": 5.31
  }' | jq .
```

---

## 3. Architecture: From Distributed Dataset Mining to `La Ruche` Community MoE

Project HiveMind (`v2.6.0`) unifies three community pillars into a single 1-command workflow:

1. **⛏️ Distributed Scientific Dataset Mining Campaign (`CAMP-S10-RUCHE-DATASET-MINING` / `src/hivemind/data/dataset_mining_campaign.py`)**:
   - Curates, deduplicates (exact normalized string check + cross-split 4-gram Jaccard verification), and computes Shannon byte entropy across the **8 HiveMind Specialist Domains**:
     - `code_python` (`@xaviercallens`)
     - `math_reasoning` (`@clark719664`)
     - `physics_sim` (`@radialmonster`)
     - `systems_rust` (`@pedromineo`)
     - `cyber_sec` (`@xaviercallens`)
     - `bio_med` (`@clark719664`)
     - `general_chat` (`@radialmonster`)
     - `formal_lean4` (`@pedromineo`)
   - Pins every dataset shard into `ContentAddressedSwarmStore` (`cid://sha256/...`) and leases each shard to a GPU contributor via **Heartbeat-Backed CAS Data Shard Leasing (`HM-S9-06`)** with zero duplicate shard commits (`Theorem 124`).
   - Trains real byte-tokenized `Qwen2` QLoRA/LoRA adapters on GPU (`cuda` when available, deterministic CPU fallback) and compresses outer pseudo-gradient deltas via **Conway-Sloane Gosset $E_8$ 2-Bit Lattice Quantization (`HM-S10-01`)**.

2. **🐝 `La Ruche` ("The Beehive") Community MoE Model Vault (`src/hivemind/swarm/la_ruche_vault.py`)**:
   - Organized as 8 hexagonal **Honeycomb Cells (`Alvéoles`, `ALVEOLE-01`..`ALVEOLE-08`)** in axial Conway-Sloane $A_2$ lattice coordinates `(q, r)` surrounding the central **Sigma-MoE + 3-Tier Hot/Cold Router Core (`(0, 0)`)**.
   - **Gatekeeper Verification**: Every contributed QLoRA adapter must pass finite-tensor checks (`torch.isfinite`), non-zero Frobenius norm checks, SVD spectral energy analysis (`project_lora_rank_svd_with_report`), and a strict **Holdout Perplexity Improvement Gate** (`holdout_nll_after < holdout_nll_before`) before entering **La Ruche**.
   - **Community BTM Sparse MoE Assembly & Release**: Merges all 8 verified Honeycomb Cells into a single `UpcycledSparseMoEModel` (`hivemind_la_ruche_community_moe.safetensors`), generates a Hugging Face Model Card (`README.md`) and `ruche_manifest.json`, and exposes a 1-line Python loader:
     ```python
     from hivemind.swarm.la_ruche_vault import load_community_moe_from_ruche

     moe_model, manifest = load_community_moe_from_ruche("artifacts/la_ruche_vault")
     ```

3. **📡 `HiveMind@home` SETI-Style & GCP Cloud Command Dashboard (`src/hivemind/boinc/hivemind_at_home_dashboard.py`)**:
   - **SETI@home Real-Time 60 FPS Spectrum & Lattice Waterfall Analyzer**: Visualizes the 8 specialist MoE frequency channels, Gosset $E_8$ (`12.66 dB` SQNR) & Conway-Sloane $A_2$ quantization envelopes, asynchronous $k$-of-$N$ DiLoCo quorum waves, and 5% two-statistic spot-check verification pulses (`95.36%` compute savings).
   - **GCP Cloud Control Plane + Local CUDA Telemetry**: Displays live status of the public Cloud Run gateway (`hivemind-at-home-ruche`), the GCP Cloud TPU VM (`gwenlaya-tpu-1`, `us-west4-a`, `v5litepod-1`, `READY`), the local CUDA node (`NVIDIA GeForce RTX 5060 Laptop GPU`, `8 GB VRAM`), the Season 9 & 10 GPU Campaign receipts (`CAMP-S9/S10 1..4`), and the **128/128 zero-`sorry` Lean 4 proof shield**.
   - **Interactive SVG Hexagonal Beehive (`La Ruche` Explorer) & Live Inference Console**: Click any hexagonal `Alvéole` cell to inspect or download its `.safetensors` weights, or run Top-2 Sparse MoE inference and federated workunit submissions directly in the browser.

---

## 4. Verification & Local Quickstart Commands

### Run the End-to-End Public GCP DNS Client Verification Suite
```bash
uv run python scripts/test_public_gcp_hivemind_client.py --url https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app
```

### Run the Complete Dataset Mining, `La Ruche` Vault & `HiveMind@home` Dashboard Pipeline Locally
```bash
uv run python scripts/run_hivemind_at_home_and_ruche_campaign.py
```
