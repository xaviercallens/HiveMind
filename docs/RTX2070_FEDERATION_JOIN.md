# RTX 2070 (XAVGAMSRV) — join the federation through `hm-fed-master`

Runbook for the agent or person operating the RTX 2070 host (Windows, `XAVGAMSRV`). It turns the measured state
below into the three steps that are missing on the 2070 side. The master side is done; nothing here touches the
router, the T4 or the master.

## Measured state (2026-11-02, 18:50–19:47Z, read-only from the integration host)

| What | Measured | Meaning |
|---|---|---|
| Router fleet (`GET /v1/hivemind/producers/fleet`) | 3 live of 45: `gcp-t4-producer`, `gcp-t4-bee`, `t4-big-gpu-moe-20261028`; every RTX 2070 account offline, `last_heartbeat_utc` ≈ 12:17Z | the 2070 has not heartbeated on its own since the reconnect keepalive stamped it |
| `hm-fed-master` `wg show` (hub 34.145.228.3:51820, overlay 10.77.0.10) | peers 10.77.0.1/.2/.3 handshake every ≤ 2 min; peer `WOsqyc4w…RFk=` = 10.77.0.5 (the 2070): **no endpoint, latest handshake 0** | the 2070's WireGuard tunnel has never come up |
| Relay `https://34-145-228-3.sslip.io/rtx2070/v1/models` | TLS OK, **HTTP 502** (upstream `10.77.0.5:8787` unreachable), per the T4 session since ~18:52Z | the master forwards to the 2070 as soon as the tunnel and the shield answer |

So the only missing pieces are on the 2070 host: (1) the tunnel, (2) the shield listening on the overlay address,
(3) the heartbeat with the relay as endpoint.

## 1. WireGuard tunnel (one peer: the master)

Windows: WireGuard app → *Add tunnel* → *Add empty tunnel*, paste; Linux: `/etc/wireguard/wg-hivemind.conf`.

```ini
[Interface]
PrivateKey = <the 2070's private key — the one whose public key is WOsqyc4wXDumYjyWDgIz+rVBabIccgk3+RDVoxxxRFk=; never paste it anywhere else>
Address = 10.77.0.5/32

[Peer]
# hm-fed-master (Spot e2-medium, us-east4-a)
PublicKey = 3bk1pCUlVjfqrgalu40DlaBDtMQsx9Mt1zWXrTnF23g=
AllowedIPs = 10.77.0.0/24
Endpoint = 34.145.228.3:51820
PersistentKeepalive = 25
```

The master's external IP is ephemeral (Spot); if the handshake never happens, re-read it with
`gcloud compute instances describe hm-fed-master --zone us-east4-a --format="value(networkInterfaces[0].accessConfigs[0].natIP)"`
and update `Endpoint`. If the 2070's key pair was lost, generate a new one and send **only the public key** to the owner
so the master's `hm-members` metadata can be updated (see `docs/FEDERATION_MASTER_SPOT.md`, section 2, on the #410 branch).

**Check:** `ping 10.77.0.10` answers (the master's overlay address) and the WireGuard app shows a recent handshake.

## 2. Producer shield listening on the overlay address

The shield defaults to `127.0.0.1`, which the relay cannot reach. Run it on the overlay address as well:

```powershell
python scripts\hivemind_producer_shield.py --host 10.77.0.5 --port 8787 --upstream http://127.0.0.1:11434 `
  --allow-model <every model this card serves, one --allow-model each>
```

Windows Firewall: allow inbound TCP 8787 **on the WireGuard interface only** (not on the public profile):

```powershell
New-NetFirewallRule -DisplayName "HiveMind shield (WireGuard only)" -Direction Inbound -Protocol TCP -LocalPort 8787 `
  -InterfaceAlias "wg-hivemind" -Action Allow
```

**Check (from the 2070 itself):** `curl http://10.77.0.5:8787/v1/models` answers 401 or 200.
**Check (from anywhere):** `curl https://34-145-228-3.sslip.io/rtx2070/v1/models` answers **401 or 200 instead of 502**.

## 3. Heartbeat on the 2070's own account with the relay as endpoint

```powershell
python scripts\hivemind_producer_agent.py --gateway https://hivemind-openrouter-1003063861791.us-central1.run.app `
  --endpoint https://34-145-228-3.sslip.io/rtx2070 --preset <tier, e.g. coder> --execution-class rtx2070-8gb-turing --once
```

Use the 2070 account's own key and endpoint token exactly as the agent's `--help` documents them (environment or flag);
never put either in a file that is committed, logged or pasted. `--once` prints the gateway's JSON answer; when it is
accepted, run the same command without `--once` (heartbeat every 60 s by default) under the autostart
(`scripts/hivemind_producer_autostart.py`) so it survives reboots.

**Check:** `GET https://hivemind-openrouter-1003063861791.us-central1.run.app/v1/hivemind/producers/fleet` lists the
2070's handle with `status: live` and its endpoint host `34-145-228-3.sslip.io`; a chat probe to a model the 2070
serves answers from it (the integration host's 30-minute monitor reports this).

## Acceptance

Federated means all four hold at the same time, each measured, none assumed:

1. master `wg show`: peer 10.77.0.5 has an endpoint and a handshake younger than 3 minutes;
2. `https://34-145-228-3.sslip.io/rtx2070/v1/models` → 401/200;
3. the fleet endpoint shows the 2070 account live with that endpoint;
4. a chat completion routed to the 2070 returns HTTP 200.

## What not to do

- Do not run `scripts/rtx2070_failover_orchestrator.py` from main against the live router: without #517 a flapping T4
  probe re-points the whole producer fleet to one tunnel (incident of 2026-11-02, #487).
- Do not expose the shield on the public interface; the relay on the master is the only public entry point.
- Do not use trycloudflare or sslip tunnels to the 2070's public address any more; the overlay replaces them.
