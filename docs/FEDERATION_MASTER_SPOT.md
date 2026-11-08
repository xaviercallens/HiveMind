# Federation master on a Spot VM (`hm-fed-master`)

Since 2026-11-02 a low-cost Spot VM is the federation master. It replaces what the T4 did for co-training, so trainers and the RTX 2070 no longer depend on the T4. The OpenRouter itself stays on Cloud Run (`hivemind-openrouter`, whose flags are fixed by `AGENTS.md`); the master is the place trainers connect to and where training sessions are organised.

| | |
|---|---|
| VM | `hm-fed-master`, e2-medium, **Spot** (termination action STOP), us-east4-a, VPC 10.150.0.10, about $7 a month |
| Restart after a preemption | Cloud Scheduler `hm-fed-master-keepalive` (every 5 minutes; a no-op while it runs) |
| Overlay address | `10.77.0.10` (WireGuard `wg-hivemind`, UDP 51820, tag `hm-kafka`) |
| Code | `deploy/amcp/fed_master_startup.sh` (GCE startup script, runs at every boot) |
| Status, without SSH | `gcloud compute instances get-guest-attributes hm-fed-master --zone us-east4-a` (`hivemind/status`, `wg-pubkey`, `session`, `relay`) |
| Results | `gs://socrateai-datalake-gen-lang-client-0625573011/hivemind_la_ruche_vault/federation_v2_e8/sessions/<RUN>/`, copied every 5 minutes |

## 1. What it does

1. **WireGuard hub.** Each member (the 2070, each registered trainer) keeps **one** peer, the master, with `AllowedIPs = 10.77.0.0/24`. The master forwards member traffic only to the overlay and masquerades it (`deploy/amcp/hm-wg-hub.sh`), so the Kafka voters see 10.77.0.10 and need no per-member change, and a member cannot reach the VPC. The master reaches the voters over their stable **internal** addresses, so a voter preemption (which changes its external IP) no longer breaks anyone's tunnel.
2. **Co-training sessions.** The coordinator (`scripts/e8_amcp_round.py coordinator`, per-learner keys `lk-<id>`) runs the session named in metadata `hm-session`. It checkpoints every round and resumes by itself after a crash or a preemption.
3. **HTTPS relay to the 2070 producer (optional).** With `hm-relay-target=10.77.0.5:8787` the master serves `https://<its ip as a-b-c-d>.sslip.io/rtx2070` and forwards to the 2070's producer shield over the overlay. This replaces the `trycloudflare` quick tunnel, whose URL changes at every restart and which failed on 2026-11-02. The shield must listen on the 2070's overlay address (10.77.0.5), not only on 127.0.0.1.

## 2. Joining as a member (the RTX 2070 or a trainer)

1. Generate a WireGuard key pair on your machine and send **only the public key** (and the GPU, VRAM and OS) to the owner, on #431 or through the trainer registration.
2. The owner adds one line `"<id> <public key>"` to the master's `hm-members` metadata (the 2070 is already there as id 5). Within a minute the master adds you as a peer and as a hub member:

   ```bash
   gcloud compute instances add-metadata hm-fed-master --zone us-east4-a --metadata-from-file hm-members=members.txt
   ```

3. Your WireGuard config (Windows: WireGuard app, Import tunnel; Linux: `/etc/wireguard/wg-hivemind.conf`):

   ```ini
   [Interface]
   PrivateKey = <your private key; never send it>
   Address = 10.77.0.<id>/32

   [Peer]
   # hm-fed-master
   PublicKey = 3bk1pCUlVjfqrgalu40DlaBDtMQsx9Mt1zWXrTnF23g=
   AllowedIPs = 10.77.0.0/24
   Endpoint = <master external IP>:51820
   PersistentKeepalive = 25
   ```

   On 2026-11-02 the master's external IP was `34.145.228.3`. It is ephemeral and changes after a preemption. Read it with `gcloud compute instances describe hm-fed-master --zone us-east4-a --format="value(networkInterfaces[0].accessConfigs[0].natIP)"`, or see section 4 for a stable address.

4. Check: `ping 10.77.0.10`, then `Test-NetConnection 10.77.0.1 -Port 9092` (Windows) or `nc -zv 10.77.0.1 9092`.
5. Run your learner with the key the owner gave you out of band (procedure in `docs/FEDERATION_V2_AMCP_AGLETS.md` section 8): `--learner-id <id> --coordinator-key coord`, `HIVEMIND_AMCP_HMAC_ACTIVE_KID=lk-<id>`, `HIVEMIND_KAFKA_BOOTSTRAP=10.77.0.1:9092,10.77.0.2:9092,10.77.0.3:9092`.

## 3. Organising a training session (owner)

```bash
# keys: one coordinator key plus one per learner, 32 random bytes each, never committed
python -c "import secrets; print(secrets.token_hex(32))"
gcloud compute instances add-metadata hm-fed-master --zone us-east4-a \
  --metadata hm-session="night-003|1000|l0 t01 t02|0.25",hm-ckpt-gs=gs://socrateai-datalake-gen-lang-client-0625573011/hivemind_la_ruche_vault/federation_v2_e8/2026-11-02/e8_real/night-002b.ckpt \
  --metadata-from-file hm-fed-keys=fed_keys.txt      # fed_keys.txt: coord:<hex>,lk-l0:<hex>,lk-t01:<hex>,lk-t02:<hex>
```

- `hm-session` is `RUN|ROUNDS|LEARNER IDS|OUTER_LR`. Changing it starts a new session; a finished session is not re-run after a reboot.
- `hm-ckpt-gs` is optional: the resume point copied into a new session (here night 2's round 412).
- Follow it with `get-guest-attributes` (`hivemind/session`) and in the data lake (`sessions/<RUN>/<RUN>.json`, `.log`, `.ckpt`).
- **The signing keys live in instance metadata.** Anyone with read access to the project's VM metadata can read them. Rotate them per session, and remove `hm-fed-keys` when the session ends.

## 4. Stable public address: owner decision

The master's external IP is ephemeral (no paid static IP, per `AGENTS.md`). Each preemption therefore changes the `Endpoint` members must use and the sslip.io hostname of the relay. When the T4 is stopped for good, the existing reservation `darkmatter-static-ip` (34.48.48.154, already paid for) can be **moved to the master** at no extra cost. Then:

- `34.48.48.154:51820` (what `src/hivemind/router/opentrainer.py` already tells newcomers) becomes the master hub;
- `https://34-48-48-154.sslip.io/rtx2070` (the router's default `rtx2070_endpoint_url`) becomes the relay to the 2070.

Steps, after the T4 is stopped: `gcloud compute instances delete-access-config hm-fed-master --zone us-east4-a --access-config-name external-nat`, then `gcloud compute instances add-access-config hm-fed-master --zone us-east4-a --access-config-name external-nat --address 34.48.48.154`. The address is regional (us-east4), so any zone of the region works.

## 5. Validation

`fedm-val-1` (2026-11-02, 17:28–17:38Z), a fresh session (no checkpoint) with fresh per-learner keys:
- the coordinator ran on `hm-fed-master` from metadata, and one CPU learner (`t4val`) ran on the T4;
- both talked only through the real Kafka cluster (voters reached over the overlay; voter 2 was not yet connected, and the quorum of 2 carried it).

Result: **20 of 20 rounds complete**, coordinator eval loss 2.594 → 1.105, no rejected or skipped result, coordinator and learner both rc=0. The checkpoint, results and log were copied by the master to `gs://socrateai-datalake-gen-lang-client-0625573011/hivemind_la_ruche_vault/federation_v2_e8/sessions/fedm-val-1/`. The keys were removed from the metadata afterwards.

Not yet validated: the RTX 2070 joining through the master (its tunnel has to be activated on the 2070 side), the HTTPS relay (needs the 2070 shield on 10.77.0.5) and recovery after a real preemption of the master.
