# Handover: the RTX 2070 host (XAVGAMSRV) becomes the co-training controller

Written 2026-11-02, when the owner announced that the T4 instance (`socreateai-agora-hermes-node1`) is being stopped. Everything the T4 did for co-training is listed below with what replaces it.

## 1. What stops with the T4, and what keeps running

| Component | After the T4 stops |
|---|---|
| Kafka voters `hm-kafka-1/2/3` (Spot, us-east4-a/b/c) | **keep running** (keepalive by Cloud Scheduler, independent of the T4) |
| Kafka broker node 4 (on the T4) | gone; topics have 3 replicas with min in-sync 2, so they keep working on the voters |
| T4 WireGuard hub (`10.77.0.4`) | gone; the 2070 connects **directly** to the three VMs instead (section 2) |
| Coordinator and learners of the night runs | move to the 2070 (section 3) |
| BOINC server, T4 governor, producers hosted on the T4 | stop (not part of this handover) |
| Results | saved in the data lake: `gs://socrateai-datalake-gen-lang-client-0625573011/hivemind_la_ruche_vault/federation_v2_e8/2026-11-02/` (co-training) and `.../t4_boinc_qlora/2026-11-02/` (T4 QLoRA ledger, returned results, telemetry); also in the repository under `artifacts/experiments/e8_real/` |

## 2. The overlay without the T4

The three Kafka VMs already list the 2070's public key (`WOsqyc4w...`) as peer `10.77.0.5`. Their external addresses are **ephemeral** (no paid static IPs, per `AGENTS.md`) and change when a Spot VM is preempted. Read the current ones with `gcloud compute instances list --filter="name~hm-kafka" --format="table(name,networkInterfaces[0].accessConfigs[0].natIP)"`. On 2026-11-02 they were:

```ini
[Interface]
PrivateKey = <the private key that matches WOsqyc4w...; never send it>
Address = 10.77.0.5/32

[Peer]
# hm-kafka-1
PublicKey = 7nKbE6320ImI+W98cpb+akh0bz8TAc+/UHOKmFHziTk=
AllowedIPs = 10.77.0.1/32
Endpoint = 34.11.62.107:51820
PersistentKeepalive = 25

[Peer]
# hm-kafka-2
PublicKey = 2Udk4EaoSbnCDxJ/eRDa+jjSkiQBijcHPxfB/YmY31E=
AllowedIPs = 10.77.0.2/32
Endpoint = 8.234.157.28:51820
PersistentKeepalive = 25

[Peer]
# hm-kafka-3
PublicKey = 1SUe558dm894iV+Y9ZXDxyNZb/TspyhwKnmScszluis=
AllowedIPs = 10.77.0.3/32
Endpoint = 34.145.228.3:51820
PersistentKeepalive = 25
```

Check: `ping 10.77.0.1`, `Test-NetConnection 10.77.0.1 -Port 9092`. If a VM is preempted, its address changes: update that `Endpoint`.

## 3. Running the coordinator on the 2070

1. Check out the PR #410 branch (or `main` once it is merged). The coordinator now retries a publish through a broker outage instead of crashing, the failure that ended `night-002b` during a voter preemption.
2. Get the resume point: `artifacts/experiments/e8_real/night-002b.ckpt` (round 412, loss 1.420), also in the data lake. Copy it as `night-003.ckpt` to continue from it, or start fresh.
3. Create **new** signing keys on the 2070 (the T4's keys stop with it): one coordinator key `coord` and one key `lk-<id>` per learner, each `python -c "import secrets; print(secrets.token_hex(32))"`. Give each learner its own key plus `coord`, out of band.
4. Run:

```bash
set HIVEMIND_KAFKA_BOOTSTRAP=10.77.0.1:9092,10.77.0.2:9092,10.77.0.3:9092
set HIVEMIND_AMCP_HMAC_KEYS=coord:<hex>,lk-l0:<hex>,lk-t01:<hex>
set HIVEMIND_AMCP_HMAC_ACTIVE_KID=coord
python scripts/e8_amcp_round.py coordinator --run-id night-003 --learners l0 t01 --rounds 1000 --quorum 1 --round-deadline-s 180 --max-retries 40 --pace-s 30 ^
  --inner-steps 40 --inner-lr 1e-2 --outer-lr 0.25 --outer-mu 0.5 --learner-key-prefix lk- --checkpoint night-003.ckpt --resume --out night-003.json
```

The 2070 can also be a learner itself (another process, its own key `lk-l0`, `--device cuda --quant nf4` if bitsandbytes loads on Windows, otherwise `--device cpu --quant none`). The VRAM floor (`--min-free-mib`, 4,300 MiB by default) was set for the T4's 15 GB; on an 8 GB card choose it for what the 2070 must keep for its own use.

## 4. Calendar consequences

Night 3 planned the T4 GPU for the H-Net probe (#448) and the coordinator on the T4 CPU. With the T4 stopped, the probe has no host that night unless the owner names one; the co-training of night 3 runs with the 2070 as coordinator. `docs/community/COTRAINING_NIGHTS.md` should be updated by whoever runs night 3.

## 5. What did not work, so nobody repeats it

- At night the T4 GPU had only 4,611 MiB free (two llama-servers), below what a learner needs under the 4,300 MiB floor: night 2's learners ran on CPU.
- `night-002` plateaued around 1.45; halving the outer learning rate (`night-002b`) brought it to 1.42 before the crash. Untuned.
- The T4's governor yielded about 2 minutes after each night resume while no BOINC task was running (`had_boinc_gpu_task: false`), triggered by 12-28 % GPU use from other processes (Ollama loads a model on demand and unloads it after 10 minutes). Diagnosis only, not acted on.
