# Tesla T4 Canonical Producer Shield & Agent Consolidation (`deploy/t4_producer/`)

## Overview
On `socreateai-agora-hermes-node1` (NVIDIA Tesla T4, `15,360 MiB` total VRAM):
- `llama-server` (`hivemind/bee-stars`, `Qwen2.5-Coder-7B-Instruct-Q4_K_M` + NIGHT-MOE-1 Stars LoRA) is pinned at **`4,838 MiB`**.
- Per `AGENTS.md`, at least **`>= 4,300 MiB` free VRAM** must always be preserved (`max_model_budget = 15,360 - 4,838 - 4,300 = 6,222 MiB`).
- Advertising `gpt-oss:20b` (`13.1 GiB`) or `deepseek-r1:14b` (`8.6 GiB`) on the same Ollama instance causes cold-start eviction/swapping (`HTTP 502` timeouts) and violates the `>= 4,300 MiB` free VRAM guard.
- Running duplicate shield/agent pairs (`hivemind-live-shield-coder` on `:8089` and `hivemind-rtx2070-shield` on `:8091`) against the same `127.0.0.1:11434` Ollama backend causes self-competing heartbeats and split rate limits.

## Canonical Deployment
1. Stop and disable any duplicate secondary shield/agent units pointing to `127.0.0.1:11434`:
   ```bash
   sudo systemctl disable --now hivemind-rtx2070-agent.service hivemind-rtx2070-shield.service
   ```
2. Install and enable the canonical T4 shield and agent units:
   ```bash
   sudo cp deploy/t4_producer/hivemind-live-shield-coder.service /etc/systemd/system/
   sudo cp deploy/t4_producer/hivemind-live-agent-coder.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now hivemind-live-shield-coder.service hivemind-live-agent-coder.service
   ```
