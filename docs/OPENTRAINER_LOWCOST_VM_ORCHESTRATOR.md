# Permanent Low-Cost VM OpenTrainer Orchestrator (`hm-opentrainer-1`)

Companion to [`docs/FEDERATION_V2_AMCP_AGLETS.md`](FEDERATION_V2_AMCP_AGLETS.md) and [`docs/AMCP_KAFKA_CLUSTER.md`](AMCP_KAFKA_CLUSTER.md).

## 1. Why Decouple the Orchestrator from the Tesla T4?

Previously, collaborative DiLoCo + QLoRA MoE training (`scripts/e8_qlora_moe_round.py` and `scripts/e8_amcp_round.py`) ran its **Coordinator / Orchestrator** directly on the Tesla T4 GPU VM (`socreateai-agora-hermes-node1`). That coupled the entire swarm's training control plane to a GPU node that:
1. Costs `~$110–$180/mo` and can be preempted or stopped to conserve GPU budget.
2. Yields its GPU whenever daytime OpenRouter user inference arrives (`t4_governor.py`).
3. Left community GPUs (such as the local **NVIDIA GeForce RTX 2070** `rtx2070-XAVGAMSRV`) unable to run collaborative training rounds whenever the T4 was offline or busy.

Meanwhile, the **Outer DiLoCo Orchestrator** does not need a GPU:
- It verifies signed `task_claim` and `task_result` envelopes (`code_pin`, `base_sha256`, `global_sha256`, finiteness, `max_delta_l2`),
- Computes the token-weighted Outer DiLoCo Nesterov momentum update (`outer_step`) on CPU tensors (`< 350 MiB` RAM),
- Evaluates the updated global LoRA state on the held-out validation set, and
- Atomically checkpoints `checkpoint.pt` + `opentrainer_status.json` (`temp + fsync + os.replace`).

## 2. Low-Cost Permanent VM Architecture (`~$3.65/mo` Spot `e2-small`)

| Property | Value | Governance Rule (`AGENTS.md`) |
| :--- | :--- | :--- |
| **VM Name** | `hm-opentrainer-1` (or colocated on `hm-kafka-1`) | Dedicated CPU-only orchestrator |
| **Machine Type** | `e2-small` (`2 vCPU`, `2 GiB RAM`, `0 GPUs`) | `~$3.65/mo` Spot (`< 3%` of a GPU VM) |
| **Provisioning Model** | `SPOT` (`--instance-termination-action=STOP`) | Mandatory Spot provisioning |
| **Keepalive** | Cloud Scheduler `hm-opentrainer-1-keepalive` (`*/5 * * * *`) | Auto-restarts stopped Spot VM within 5 min |
| **Network** | Private VPC `10.150.0.14` + WireGuard `10.77.0.6` | Zero paid unused static external IP penalty |
| **Systemd Service** | [`deploy/opentrainer_vm/hivemind-opentrainer.service`](../deploy/opentrainer_vm/hivemind-opentrainer.service) | `User=hivemind`, `NoNewPrivileges=true`, `ProtectSystem=strict` |

## 3. How Collaborative Training Runs Without the Tesla T4

1. **Dual-Transport Aglet Lifecycle (`HTTP/HTTPS Outbound` + `AMCP/Kafka`)**:
   - Community GPUs behind home NAT (like the Windows RTX 2070) connect **outbound only** to `/v1/opentrainer/{offer,claim,result}` or over the AMCP/Kafka bus (`hivemind.amcp.federation.v1`).
   - Every message (`task_offer`, `task_claim`, `task_result`, `task_abort`) is an HMAC-signed AMCP envelope verified by `EnvelopeVerifier`.
2. **T4-Offline Autonomous Quorum (`quorum=1` + Optional CPU Sentinel)**:
   - When the T4 (`l0`) is online and idle, both `l0` (T4) and `l1` (RTX 2070) train their respective domain shards on GPU.
   - When the T4 is offline, preempted, or serving inference, `PermanentOpenTrainerOrchestrator` can either:
     - Close the round on `quorum=1` using the RTX 2070's verified NF4 QLoRA delta alone, **or**
     - Run its built-in lightweight **CPU Sentinel Micro-Learner** (`hm-opentrainer-1-cpu-sentinel`) on shard `l0` so 2-shard cross-domain mixing continues uninterrupted alongside the RTX 2070 (`l1`).

## 4. Quickstart Commands

```bash
# 1. Run the T4-offline self-test on the local RTX 2070 (NF4 CUDA + OpenTrainer CPU sentinel):
python scripts/opentrainer_vm_daemon.py self-test --rounds 3 --inner-steps 20 --device cuda --quant nf4

# 2. Start the permanent OpenTrainer Orchestrator daemon:
python scripts/opentrainer_vm_daemon.py orchestrator --host 0.0.0.0 --port 8096 --quorum 1

# 3. Connect an RTX 2070 or community GPU learner (outbound HTTP only, no T4 needed):
python scripts/opentrainer_vm_daemon.py learner --orchestrator-url http://127.0.0.1:8096 --learner-id l1 --index 1 --quant nf4 --device cuda
```
