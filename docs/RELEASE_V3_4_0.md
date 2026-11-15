# HiveMind `v3.4.0` — CrowdGPT-MoE x OpenDiLoCo Co-Training, HMT P2P Transfers & Weekend Swarm Raid

**Release Date:** 2026-11-11  
**Tag:** `v3.4.0`  
**Merged PR:** [#871 — CrowdGPT Trainer/Producer/Prosumer Co-Training, HMT P2P Transfers & Nov 14–15 MoE OpenDiLoCo Raid](https://github.com/xaviercallens/HiveMind/pull/871)

---

## Highlights

1. **Sparse-MoE Split & Decoupled OpenDiLoCo for `Vxtzq/Crowd-v1` (`1.01B` -> `2.18B` Fine-Grained / `1.06B` LoRA-MoE)**
   - [`src/hivemind/swarm/crowdgpt_moe_opendiloco.py`](../src/hivemind/swarm/crowdgpt_moe_opendiloco.py): Upcycles dense `SwiGLU` layers `[6..23]` into **1 Shared Anchor Expert + 8 Routed Domain Specialists (`top-2`)**, supporting both full fine-grained MoE (`2.18B` total / `1.20B` active) and memory-lean LoRA-MoE (`r=16`, `1.06B` total / `1.02B` active) for `8 GB–16 GB` community GPUs.
   - Integrates **CPU-Offloaded FP32 Nesterov Outer Momentum** (`m=0.9`, `lr=0.7`), DeepMind **Decoupled DiLoCo** (`RDA` staleness damping + `PA-CGAD` cosine conflict gating), and cryptographic **Spot-Checked Loss Ratchet** (`LossCheckReceipt` + `S21` `ACCEPTED`-only dataset curation).
   - Live CUDA (`sm_75`) verification on checkpoint `59c016e94875`: held-out cross-entropy loss dropped **`5.3042 -> 5.1742 -> 0.9668`** across 2 consecutive outer rounds with zero-mmap `32 MiB` chunked `safetensors` streaming (`peak VRAM 3,907 MiB`, respecting `AGENTS.md` VRAM guards).

2. **GCP Tesla T4 (`socreateai-agora-hermes-node1`) & Local RTX 2070 Co-Training + Producer Serving**
   - [`src/hivemind/swarm/gcp_t4_crowd_diloco.py`](../src/hivemind/swarm/gcp_t4_crowd_diloco.py) & [`scripts/gcp_t4_crowd_diloco_worker.py`](../scripts/gcp_t4_crowd_diloco_worker.py): Preemptible Spot T4 (`>= 4,300 MiB` free VRAM guard, `/opt/hivemind/t4/` read-only protection) and local RTX 2070 (`>= 1,536 MiB` free VRAM guard) OpenDiLoCo worker + OpenAI-compatible `hivemind/crowd-v1` producer serving.
   - [`src/hivemind/swarm/crowdgpt_live_runtime.py`](../src/hivemind/swarm/crowdgpt_live_runtime.py) & [`src/hivemind/swarm/crowdgpt_loss_check.py`](../src/hivemind/swarm/crowdgpt_loss_check.py): Exact `CrowdForCausalLM` + `CrowdEngram` (`227,865 x 1,536`) execution and CPU-verifiable held-out loss check receipts.

3. **Peer-to-Peer HiveMind Point (`HMT`) Transfers, Registration Grants & Promo Bonuses**
   - [`src/hivemind/router/accounts.py`](../src/hivemind/router/accounts.py) & [`src/hivemind/router/gateway.py`](../src/hivemind/router/gateway.py): Atomic `POST /v1/hivemind/accounts/transfer` (`transfer_points_between_hiveminders`) with idempotency keys, self-transfer guard, and double-entry ledger audit trail, plus boosted signup welcome grants and promo bonus codes.

4. **Weekend Massive Swarm Training Raid (`14–15 November 2026`)**
   - [`src/hivemind/cli/campaign_calendar_quorum.py`](../src/hivemind/cli/campaign_calendar_quorum.py) & [`docs/calendar/hivemind_crowdgpt_weekend_nov14_15_2026.ics`](calendar/hivemind_crowdgpt_weekend_nov14_15_2026.ics): 6-wave global schedule (`Wave 0` Friday Preflight through `Wave 5` Sunday Merge & HF/OpenRouter Promotion).
   - `GET /v1/hivemind/opentrainer/crowdgpt-weekend-raid` live raid telemetry endpoint, GitHub Issue registration template ([`.github/ISSUE_TEMPLATE/04_crowdgpt_weekend_moe_raid.yml`](../.github/ISSUE_TEMPLATE/04_crowdgpt_weekend_moe_raid.yml)), and multi-channel launch kit ([`docs/CROWDGPT_WE_NOV14_15_2026_MASSIVE_TRAINING_KIT.md`](CROWDGPT_WE_NOV14_15_2026_MASSIVE_TRAINING_KIT.md)).

5. **GCP Cost Governance (`AGENTS.md` Compliant)**
   - Projected monthly GCP spend maintained at **`$121.92 / mo`** (`<= $150.00 / mo` hard budget cap).
