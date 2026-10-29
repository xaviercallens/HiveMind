# HiveMind & SocrateAI Engineering & GCP Cost Governance Rules (`AGENTS.md`)

## 1. Local Tesla T4 Hardware & Directory Protection (`socreateai-agora-hermes-node1`)
- **Protected Directory**: Never modify, overwrite, or delete `/opt/hivemind/t4/` on `socreateai-agora-hermes-node1`.
- **VRAM Headroom Guard**: Always maintain at least **`>= 4,300 MiB` free VRAM** on the local NVIDIA Tesla T4 GPU (`15,360 MiB` total) so core HiveMind inference services are never starved or OOM-killed.

## 2. GCP Cost & Autoscaler Governance (`gen-lang-client-0625573011`)
Whenever deploying, scaling, or auditing GCP infrastructure, strictly enforce the following cost and autoscaler rules (and activate the `gcp-cost-autoscaler-ops` skill in [.agents/skills/gcp-cost-autoscaler-ops/SKILL.md](file:///home/callensxavier_gmail_com/HiveMind/.agents/skills/gcp-cost-autoscaler-ops/SKILL.md)):

1. **Zero Idle Cloud TPUs**:
   - Never provision On-Demand Cloud TPUs (`v5litepod-*`, `v6e-*`) for persistent daemons.
   - Always use `--spot` (`provisioningModel: SPOT`) for batch training or XLA benchmarking and delete the TPU VM immediately when the run completes.

2. **Cloud Run Knative Autoscaler Defaults**:
   - **Single-Writer Stateful Litestream SQLite (`hivemind-openrouter`)**: Must use `--cpu=1 --memory=1Gi --min-instances=1 --max-instances=1 --no-cpu-throttling --cpu-boost --concurrency=80` ([deploy/gcp_openrouter/deploy.sh](file:///home/callensxavier_gmail_com/HiveMind/deploy/gcp_openrouter/deploy.sh)). Never over-allocate `>1 vCPU` to a single-worker (`--workers 1`) Python Uvicorn process.
   - **Stateless Control Plane & HTTP Services (`hivemind-at-home-ruche`, `symbrain-v4-edge`, `runux-math-kernel-service`, etc.)**: Must use `--min-instances=0`, `--cpu-throttling` (request-based CPU billing), and `--cpu-boost`.
   - **Cloud Run L4 GPU Services (`deepseek-prover-v2`, `personashield-api-gpu`, `symbrain-v4-cloud32`)**: Must cap `--min-instances=0 --max-instances=1` and offload sustained/burst traffic to the zero-GCP-cost **100-Producer Community GPU Swarm** (`hivemind-at-home-ruche` + AMCP v1.5 EDA traffic sustainer).

3. **Compute Engine VMs, Static IPs & Persistent Disks**:
   - All non-stateful compute VMs must use `--provisioning-model=SPOT`.
   - Never reserve paid static external IPs (`gcloud compute addresses create`) for internal VPC cluster nodes (such as Kafka KRaft voters in [scripts/amcp_kafka_cluster.py](file:///home/callensxavier_gmail_com/HiveMind/scripts/amcp_kafka_cluster.py), which bind to private VPC IPs `10.150.0.x`). Always release static external IPs before stopping VMs to avoid the `$0.01/hr` (`$7.30/mo`) unused static IP penalty.
   - Keep snapshot schedule retention at `--max-retention-days=3 --on-source-disk-delete=apply-retention-policy` and detach daily snapshot schedules from `TERMINATED` VMs.
   - Keep `darkmatter-k3-db` Cloud SQL stopped (`--activation-policy=NEVER`) when idle, and preserve `socrate-forwarding-rule` (`34.54.227.33`) while `socrate-ai-lab.com` routes through its managed SSL certificate (`socrate-ai-ssl`).
   - Enforce Artifact Registry cleanup policies (`keepCount: 2`, delete older than `7d`) on all container repositories.

## 3. Formal Verification & Testing Standards
- **Lean 4 Proofs**: All new theorems in [lean4/HiveMindSpec.lean](file:///home/callensxavier_gmail_com/HiveMind/lean4/HiveMindSpec.lean) must compile cleanly via `lake env lean HiveMindSpec.lean` with **zero `sorry` axioms**.
- **Pytest Suite**: Run `uv run pytest` on all modified modules and verify 100% pass rate before opening or updating pull requests.
