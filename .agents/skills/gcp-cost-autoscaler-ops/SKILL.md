---
name: gcp-cost-autoscaler-ops
description: >-
  Audits and optimizes Google Cloud Platform (GCP) costs, Cloud Run Knative autoscalers,
  Compute Engine Spot VMs, Cloud TPUs, Persistent Disks, Snapshots, Static IPs, Cloud SQL,
  and Artifact Registry for project gen-lang-client-0625573011 (SocrateAI / HiveMind).
  Use this skill whenever the user asks to audit GCP billing/costs, optimize autoscalers,
  clean up idle VMs/TPUs/disks/snapshots, or deploy new Cloud Run / GCE services cost-effectively.
---

# GCP Cost & Autoscaler Optimization Skill (`gcp-cost-autoscaler-ops`)

This skill provides a repeatable runbook and automated CLI tool for auditing and optimizing GCP infrastructure costs in project `gen-lang-client-0625573011` (`SocrateAI`) while preserving 100% of active HiveMind and SocrateAI production workloads.

## 1. Automated Live Audit & Safe Optimization Script

Always start by running the bundled audit script in read-only mode to inspect the current live state and estimated monthly burn across all GCP resource types:

```bash
python3 .agents/skills/gcp-cost-autoscaler-ops/scripts/gcp_cost_audit.py --project gen-lang-client-0625573011
```

To apply non-destructive autoscaler and hygiene optimizations automatically (snapshot pruning >3 days, Artifact Registry cleanup policy enforcement, unattached static IP detection):

```bash
python3 .agents/skills/gcp-cost-autoscaler-ops/scripts/gcp_cost_audit.py --project gen-lang-client-0625573011 --apply-safe
```

---

## 2. Four-Tier Autoscaler & Cost Architecture

### Tier 1 — Cloud Run Knative Autoscaler Rules
1. **Single-Writer Stateful Litestream Gateway (`hivemind-openrouter` in `us-central1`)**:
   - Runs `litestream replicate -exec "uvicorn server:app --workers 1"`.
   - Because Uvicorn runs a single Python worker process (GIL-bound to 1 core), allocate exactly **1 vCPU and 1 GiB RAM** with `--cpu-boost`:
     ```bash
     gcloud run services update hivemind-openrouter \
       --region=us-central1 --project=gen-lang-client-0625573011 \
       --cpu=1 --memory=1Gi --min-instances=1 --max-instances=1 \
       --no-cpu-throttling --cpu-boost --concurrency=80
     ```
2. **Stateless Control Plane & APIs (`hivemind-at-home-ruche`, `symbrain-v4-edge`, `runux-math-kernel-service`, etc.)**:
   - Must use **request-based billing** (`--cpu-throttling`) and **scale-to-zero** (`--min-instances=0`) with `--cpu-boost`:
     ```bash
     gcloud run services update <SERVICE> \
       --region=<REGION> --project=gen-lang-client-0625573011 \
       --cpu-throttling --cpu-boost --min-instances=0 --max-instances=2
     ```
3. **Cloud Run L4 GPU Services (`deepseek-prover-v2`, `personashield-api-gpu`, `symbrain-v4-cloud32`)**:
   - Keep `--min-instances=0 --max-instances=1` to prevent concurrent multi-GPU cold-start spikes from crawlers or burst retries.
   - Offload sustained inference to the zero-GCP-cost **100-Producer Community GPU Swarm** (`hivemind-at-home-ruche` + AMCP v1.5 EDA sustainer).

### Tier 2 — Cloud TPU Governance
- **Never** leave On-Demand Cloud TPUs (`v5litepod-*`, `v6e-*`) running idle (`$1.20+/hr = ~$876/mo` per chip).
- Always check all TPU zones (`us-west4-a`, `us-central1-a`, `us-east1-d`, `europe-west4-a`):
  ```bash
  for z in us-west4-a us-central1-a us-central1-b us-east1-d europe-west4-a; do
    gcloud compute tpus tpu-vm list --zone=$z --project=gen-lang-client-0625573011
  done
  ```
- If a TPU is needed for batch benchmarking or training, provision with `--spot` and delete immediately upon job completion.

### Tier 3 — Compute Engine VMs, Static IPs & Persistent Disks
1. **Core Production Node (`socreateai-agora-hermes-node1` in `us-east4-b`)**:
   - `n1-standard-8` + `1x Tesla T4` (`SPOT`).
   - **Protected Path**: Never touch `/opt/hivemind/t4/` and always maintain `>= 4,300 MiB` free VRAM on the Tesla T4.
2. **Static External IPs (`$7.30/mo` penalty when attached to `TERMINATED` VMs)**:
   - Internal cluster services (such as Kafka KRaft voters in `scripts/amcp_kafka_cluster.py`) must bind to private VPC IPs (`10.150.0.x`) and never reserve paid static external IPs.
   - Audit and release any `RESERVED` (unattached) or stopped-VM static IPs:
     ```bash
     gcloud compute addresses list --project=gen-lang-client-0625573011
     ```
3. **Snapshot Schedule Retention (`<= 3 days`)**:
   - Keep `default-schedule-1` (`us-east4`) and `default-daily-20261009-204229` (`us-central1`) at `--max-retention-days=3 --on-source-disk-delete=apply-retention-policy`.
   - Detach daily snapshot schedules from `TERMINATED` VMs so stopped disks are not snapshotted daily.

### Tier 4 — Cloud SQL & Artifact Registry Hygiene
1. **Cloud SQL (`darkmatter-k3-db` in `europe-west1`)**:
   - Keep stopped (`--activation-policy=NEVER`) when not actively used. Start on demand with:
     ```bash
     gcloud sql instances patch darkmatter-k3-db --activation-policy=ALWAYS --project=gen-lang-client-0625573011
     ```
2. **Artifact Registry Cleanup Policies**:
   - Ensure every repository retains only the `2` most recent image versions and deletes older revisions after `7d`.
