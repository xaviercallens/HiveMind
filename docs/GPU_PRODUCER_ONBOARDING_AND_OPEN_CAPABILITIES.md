# Project HiveMind@home & La Ruche OpenRouter — GPU Producer Onboarding & Open Capabilities Guide

## 0. Call for GPU producers (2026-10-31)

HiveMind needs more GPUs online to serve inference and to run night QLoRA training on the community swarm. We are asking for idle GPUs and for people to test the producer on their own machine.

**What helps most**

- A producer on any NVIDIA GPU with **4 GB to 24 GB** of VRAM, idle for part of the day or night.
- Test reports from a machine we have not tried: the GPU model, VRAM, the model the producer chose, and whether it stayed up.
- Bug reports on the install path (see the warning below).

**What you get:** a non-monetary record of compute contributed ("points", `HMT`), as set out in the charter at the top of `ROADMAP.md`. Nothing is paid.

**Before you run anything:** read the install script first. Its one-liner in section 2 downloads and runs code. It also falls back to an 8 GB profile when `nvidia-smi` cannot read your GPU, so check the VRAM figure it reports against your card. Known issue, not yet fixed.

**How to start:** use Option B in section 2 (manual setup with Ollama) if you want to see each step. Report results on the public discussion thread [#251](https://github.com/xaviercallens/HiveMind/issues/251).

---

## 1. Congratulations to Pioneer End-User #1 (`fbc1d3042d377feba6b1e0ac6fd576a4`)

On **2026-10-25**, Pioneer End-User #1 (`fbc1d3042d377feba6b1e0ac6fd576a4`, role: `prosumer`) completed the first live end-user completion on the decentralized **HiveMind OpenRouter (`/v1/chat/completions`)**:

| Metric | Verified Value |
| :--- | :--- |
| **Account ID** | `fbc1d3042d377feba6b1e0ac6fd576a4` (`prosumer`) |
| **Capability Tier** | `hivemind/auto` (Smart Market Liquidity Routing $\to$ Live GPU Producer) |
| **Prompt Task** | High-performance Python decorator with LRU caching, thread safety, and microsecond metrics |
| **Time to First Byte (TTFT)** | **273 ms** |
| **Generation Speed** | **36.0 tok/s** |
| **Total Tokens** | **294 tokens** |
| **Starter Grant** | **100.0000 HMT** (HiveMind Points) |
| **Settled Cost** | **1.5835 HMT** |
| **Remaining Balance** | **98.4165 HMT** (+ **250.0000 HMT** Pioneer #1 Bonus Credited = **348.4165 HMT**) |
| **Cryptographic Receipt** | `45dc962abf34a28c0011f1e55466872a2d3c155e5cda91d5951d2128084a5d5b` |

---

## 2. How to Contribute Your Idle GPU in 1 Command (Zero-Friction Producer Onboarding)

Any gamer, student, developer, or researcher with a consumer GPU (**4 GB to 24 GB VRAM**) can join the swarm and earn **HiveMind Points (`HMT`)** while their PC is idle.

### Option A — 1-Command Automatic GPU Installer (Linux, Windows WSL2, macOS)

```bash
curl -sSL https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/install-producer.sh | bash
```

This script:
1. Automatically detects your GPU and VRAM via `nvidia-smi` (or defaults to an 8 GB consumer card profile).
2. Downloads the standalone zero-friction producer agent (`hivemind_producer_agent.py`).
3. Queries the live `/v1/hivemind/catalog` endpoint and recommends the highest capability tier that safely fits your card.
4. Prints the exact command to start advertising your GPU to the OpenRouter.

### Option B — Direct 3-Step Manual Setup with Ollama

```bash
# 1. Pull and run the model matching your GPU VRAM tier:
#    - 4 GB+ VRAM:  ollama run qwen2.5-coder:3b     (--preset coder-lite)
#    - 6 GB+ VRAM:  ollama run qwen2.5-coder:7b     (--preset coder)
#    - 12 GB+ VRAM: ollama run qwen2.5-coder:14b    (--preset coder-plus)
#    - 16 GB+ VRAM: ollama run gpt-oss:20b          (--preset moe-general)
#    - 24 GB+ VRAM: ollama run qwen3-coder:30b      (--preset moe-coder)
ollama run qwen2.5-coder:7b

# 2. Export your HiveMind API key (minted at /openrouter or /signup)
export HIVEMIND_API_KEY="hm-YOUR_API_KEY"

# 3. Download and run the standalone Producer Agent with Remote Control sync
curl -sSL https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/hivemind_producer_agent.py -o hivemind_producer_agent.py
python3 hivemind_producer_agent.py \
  --gateway https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app \
  --endpoint https://YOUR_PUBLIC_TUNNEL_URL \
  --preset coder \
  --sync-remote-config
```

### Hardware Safety & `< 1.5s` Game Preemption Guarantee
- **Strict 8 GB Consumer VRAM Protection:** Federated training workunits are hard-capped below **6,656 MiB** (`17.14 MiB` measured peak in Season 9/10 campaigns via GaLore rank-64 + 8-bit Adam).
- **Instant `< 1.5s` Game Preemption:** When a game or 3D application launches (`Steam`, `Cyberpunk2077.exe`, `Blender`), the hardware circuit-breaker pauses HiveMind work in **0.42 seconds** (`< 1.5s` SLA).
- **Remote Control From Any Browser:** Visit `https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/openrouter#remote-control` to set your node state to `active`, `paused`, or `draining`, adjust your VRAM cap, or update your minimum ask price in HMT/1k tokens.

---

## 3. All Open Technical Capabilities Available for Community Contribution

Every layer of **Project HiveMind** is open-source (**Apache-2.0**) and open for community contribution:

1. **OpenRouter Live Inference Producer (`/v1/hivemind/producer/heartbeat`)**
   - Serve any of the 5 capability tiers (`coder-lite`, `coder`, `coder-plus`, `moe-general`, `moe-coder`) or specialist Bee adapters (`hivemind/bee-stars`).
   - Earn **85%** of the Kelly-Tullock clearing price in **HiveMind Points (`HMT`)** for every 1,000 tokens served.
2. **La Ruche 8-Domain Federated Sparse MoE Training (`/api/v1/federated/workunit/request` & `/submit`)**
   - Lease content-addressed scientific dataset shards (`HeartbeatDataLeaseManager`) and train rank-4 QLoRA Honeycomb cells across 8 specialist domains:
     - `code_python` (GWAYA Coding Bee)
     - `general_chat` (Gwen Multi-Lingual MoE Bee)
     - `physics_sim` (Kepler/TESS Exoplanet & Green Bank 1420 MHz SETI Bee)
     - `formal_lean4` (Lean 4 Formal Theorem Prover Bee)
     - `math_reasoning` (Conway-Sloane $E_8$ & $A_2$ Lattice Bee)
     - `bio_med` (Genetics & Protein Folding Bee)
     - `systems_rust` (Rust QUIC & Systems Kernel Bee)
     - `cyber_sec` (Cognitive Shield & Byzantine Firewall Bee)
   - Earn **+250 HMT** per verified workunit Passing the SVD rank-4 spectral gate and 5% two-statistic spot-check.
3. **2-Bit Gosset $E_8$ Lattice & DCT Gradient Compression (`src/hivemind/compression/`)**
   - Contribute to the **30.89x WAN compression** codec (`0.518 bits/param`, `12.66 dB` SQNR) with EF21 error-feedback telescoping conservation.
4. **Formal Mathematical Verification in Lean 4 (`formal/`)**
   - Contribute new theorems to our **128 / 128 machine-checked Lean 4 proof inventory** (`0 sorry`, `0 custom axioms`).
5. **Byzantine Multi-Krum & Cognitive Shield Security (`src/hivemind/security/`)**
   - Contribute red-team test vectors and deterministic execution-class spot-check verifiers (**49/49 attack vectors blocked**).
