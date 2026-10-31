---
name: amcp-hivemind-swarm
description: Orchestrate distributed multi-agent systems over AMCP (Agent Mesh Communication Protocol) and route coding, scientific math, Lean 4 formal verification, and Vision/Video MoE tasks to the HiveMind P2P GPU Swarm (<1.5 GB VRAM per node, <1.5s Steam game preemption). Compatible with Claude Code, Cursor, GitHub Copilot, Google Antigravity, and OpenAI SDKs.
---

# AMCP & HiveMind Distributed Agent Mesh Skill (`amcp-hivemind-swarm`)

Use this skill whenever the user wants to:
1. **Coordinate distributed or mobile AI agents** using **AMCP (Agent Mesh Communication Protocol)** with multi-agent Bayesian debate (`HM-S33-02`), speculative draft-verify decoding (`HM-S34-01`), and cryptographic Merkle receipts (`HM-S34-04`).
2. **Offload coding, physics, formal Lean 4 proofs, or Vision/Video MoE tasks** to the **HiveMind OpenRouter** (`15+` live specialist Bees) using zero-cost `HMT` reciprocity points (`100 HMT` free on signup).
3. **Turn an idle 8 GB–16 GB consumer GPU into a P2P HiveMind Bee node** under a strict `< 1,536 MiB` (`< 1.5 GB`) citizen VRAM ceiling and `< 1.5s` automatic game preemption (`HM-S34-02`).

---

## 1. Quick Start — Get a Free `hm-...` API Key (100 HMT Reciprocity Points, Zero Credit Card)

1. Register or sign in at:
   - **Direct Signup Portal:** `https://hivemind-openrouter-1003063861791.us-central1.run.app/signup`
   - **La Ruche Community Portal:** `https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/`
2. Copy your `hm-...` API key (new accounts receive **100 HMT** immediately; sharing idle GPU cycles replenishes HMT automatically at a 95% producer share).

---

## 2. Using HiveMind Specialist Bees from Google Antigravity SDK, Python OpenAI SDK, Claude Code & Cursor

### A. Google Antigravity SDK (`pip install google-antigravity`)
```python
import asyncio
from google.antigravity import Agent, LocalOpenAIAgentConfig

async def main():
    config = LocalOpenAIAgentConfig(
        model="hivemind/bee-stars",
        base_url="https://hivemind-openrouter-1003063861791.us-central1.run.app/v1",
    ).lightweight()
    async with Agent(config=config) as agent:
        reply = await agent.chat(
            "Optimize our Symplectic Velocity-Verlet N-Body orbital step for KEPLER-HM-442b."
        )
        print(await reply.text())

asyncio.run(main())
```

### B. Standard OpenAI Python SDK / Cursor / Continue.dev / Cline
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://hivemind-openrouter-1003063861791.us-central1.run.app/v1",
    api_key="hm-YOUR_API_KEY",
)

resp = client.chat.completions.create(
    model="hivemind/moe-coder-polyglot-3tome",  # or hivemind/coder, hivemind/bee-rusty-sundials, hivemind/bee-lean-prover
    messages=[
        {"role": "user", "content": "Design a lock-free AMCP message ring buffer in Rust with zero heap allocations."}
    ],
)
print(resp.choices[0].message.content)
```

---

## 3. Available Specialist Bees on the Mesh

| Model ID | Specialization | Peak Citizen VRAM |
| :--- | :--- | :---: |
| `hivemind/moe-coder-polyglot-3tome` (`hivemind/dev-moe`) | 3-Tome Polyglot Software Engineering (`Python`, `C`, `C++`, `Rust`) across 8 Expert Clusters | `<= 6,180 MiB` (Prosumer) |
| `hivemind/coder` | Fast IDE code generation, refactoring, and pytest synthesis (`Qwen2.5-Coder-7B`) | `<= 5,500 MiB` |
| `hivemind/bee-stars` | Symplectic astrophysics, exoplanet transit CWT (`KEPLER-HM-442b`), N-body dynamics | `<= 1,420 MiB` |
| `hivemind/bee-rusty-sundials` | Stiff ODE/PDE numerical integration (`CVODE` BDF) & zero-allocation Rust FFI | `<= 1,420 MiB` |
| `hivemind/bee-lean-prover` | Lean 4 formal theorem proving, Gosset $E_8$ lattice quantization & invariant audits | `<= 1,480 MiB` |
| `hivemind/moe-image-gen` (`hivemind/vision-moe`) | 7-Division 104-Bee Image Generation, SVG Diagram Blueprint & OCR Critique | `<= 1,410 MiB` |
| `hivemind/video-moe` (`hivemind/moe-video-gen`) | Spatio-Temporal Video Generation (`Wan2.1` + `LTX-Video` + `225.4x` 3D-DCT compression) | `<= 1,355 MiB` |

---

## 4. AMCP Multi-Agent Debate & Byzantine Verification Workflow

When solving high-stakes engineering or scientific problems, orchestrate a **3-Bee AMCP Debate Arena (`HM-S33-02`)**:
1. **Proposer Bee (`hivemind/coder` or `hivemind/moe-coder-polyglot-3tome`):** Synthesizes the candidate implementation and explicit unit test assertions.
2. **Numerical / Physics Auditor Bee (`hivemind/bee-rusty-sundials` or `hivemind/bee-stars`):** Verifies conservation laws ($|\Delta H / H_0| < 10^{-6}$, spectral radius, memory bounds).
3. **Formal Verifier Bee (`hivemind/bee-lean-prover`):** Checks invariant soundness and emits a SHA-256 Merkle-chained receipt before accepting the patch.

---

## 5. Security, Privacy & Hardware Invariants

- **Split Prelude Local Privacy (`P20a`):** Keep PII scrubbing and initial token embedding on the local machine (`<= 1,536 MiB`); only dispatch anonymized sub-tasks or intermediate hidden states to external swarm nodes.
- **Byzantine Resilience:** All decentralized DiLoCo gradient updates enforce `fp32` Multi-Krum filtering, staleness-aware PA-CGAD (`HM-S33-01`), and Top-1% 2D-DCT spectral verification.
- **Instant Game Preemption (`< 1.5s`, `HM-S34-02`):** When a Steam, DirectX, Vulkan, or Proton process starts on a producer's machine, the worker flushes EF21 residuals to pinned host RAM and drops GPU VRAM allocation to `0 MiB` in under `1,500 ms`.
