# Bringing participants onto the overlay: what is up now, and open-source alternatives to hand-made tunnels

Written 2026-11-01. Companion to `docs/AMCP_KAFKA_CLUSTER.md` and `docs/FEDERATION_V2_AMCP_AGLETS.md`.

## 1. What is up now: the T4 as WireGuard hub

A participant behind NAT (the RTX 2070 host, XAVGAMSRV) needs only **one** WireGuard peer: the T4, which has a stable address. The T4 forwards that participant's traffic to the other overlay addresses (the three Kafka VMs) and masquerades it, so the VMs need no change and the participant never has to track the Spot VMs' changing addresses.

* Script: `deploy/amcp/hm-wg-hub.sh` (installed as `/usr/local/sbin/hm-wg-hub`), wired into `/etc/wireguard/wg-hivemind.conf` as `PostUp`/`PostDown`, so it follows the interface.
* What it forwards: **only** traffic from `10.77.0.5/32` to `10.77.0.0/24`, and the replies. Every other forwarded packet is dropped (the T4 forwarded nothing before), so an overlay member cannot reach the VPC through the T4.
* The 2070 host is a persistent peer of the T4 (`10.77.0.5/32`). UDP 51820 is open to the T4 (network tag `hm-kafka`).
* State on 2026-11-01: the T4 side is ready; **the 2070 host has not completed a handshake yet** (it has to activate its tunnel, section 2).

## 2. The 2070 host: steps to come up (Windows, official WireGuard client)

1. In the WireGuard app, open the tunnel whose public key is `WOsqyc4wXDumYjyWDgIz+rVBabIccgk3+RDVoxxxRFk=` (the key already sent), or import this file and put that tunnel's private key where shown. Never send the private key to anyone.

```ini
[Interface]
PrivateKey = <the private key that matches WOsqyc4w...>
Address = 10.77.0.5/32

[Peer]
# the T4 hub
PublicKey = pSyxjN6PCS8uoRFJnyUPlm2v5myOZDfYnU3i9haH/Dw=
AllowedIPs = 10.77.0.0/24
Endpoint = 34.48.48.154:51820
PersistentKeepalive = 25
```

2. Activate. Check in PowerShell: `ping 10.77.0.4` (the T4), `ping 10.77.0.1` (a Kafka VM, through the hub), `Test-NetConnection 10.77.0.1 -Port 9092`.
3. Join a run as in `docs/FEDERATION_V2_AMCP_AGLETS.md` section 8 (`HIVEMIND_KAFKA_BOOTSTRAP=10.77.0.1:9092,10.77.0.2:9092,10.77.0.3:9092`, the signing key from the operator, then `scripts/e8_amcp_round.py learner ... --night-only`).

If UDP to port 51820 is blocked where the 2070 host is, plain WireGuard cannot work there: see section 4 (wstunnel or Headscale).

## 3. Why hand-made WireGuard does not scale to new participants

Measured this week: every newcomer needs a manual key exchange with an operator, an edit of the hub's peer list, a free UDP path to one port, and the exact repository commit. Two machines that are both behind NAT cannot reach each other without a hub. Revoking someone means editing configs by hand. Kafka has no authentication of its own, so **whoever is on the overlay can read and write every topic**: the peer list is the security boundary, and nobody should hand-maintain it for hundreds of people.

## 4. Open-source options

Requirement: open source, self-hostable, works for a Windows gaming PC behind home NAT, and keeps the Kafka cluster unchanged (brokers advertise `10.77.0.x`).

| Option | What it adds | Fit | Cost / risk |
|---|---|---|---|
| **Plain WireGuard + the T4 hub** (now) | nothing new to run | works today for a few hand-picked machines | manual keys; needs UDP 51820; the hub is a single point of failure |
| **WireGuard over WebSocket/TLS** with an open-source tunnel such as `wstunnel` | WireGuard packets inside an HTTPS-looking TCP connection | for networks that block UDP; keeps everything else | one more binary on each side; the T4's port 443 is already taken by another service, so another port or a route on the existing proxy is needed |
| **Headscale** (self-hosted, Tailscale-compatible control server, BSD-3) **+ Tailscale clients** | automatic key exchange, NAT traversal, relay over HTTPS when direct paths fail (DERP), single-use expiring join keys, per-device revocation, access rules | best newcomer experience: install the client, run one command with a join key. A "subnet router" on the T4 advertises `10.77.0.0/24`, so Kafka stays as it is | needs a public HTTPS endpoint for the control server (the T4 has a stable address; a port and a firewall rule to decide). The Tailscale client core is open source; its Windows and macOS graphical wrappers are not, so a strictly open-source stack would use the command-line client |
| **Nebula** (MIT) | identity by certificates signed by our own CA, lighthouses for discovery, relays | fully open stack; certificates give per-participant identity and expiry, which also answers part of the "one trust domain" problem at the network layer | configuration files rather than a GUI; issuing a certificate per participant is an operator step (scriptable) |
| NetBird | WireGuard mesh with a management service | similar to Headscale | **check the current licences before adoption**: the client and the server parts are licensed differently |
| ZeroTier | mesh networking | would work technically | **excluded**: source-available under the Business Source License, not an OSI open-source licence |

## 5. Recommendation

1. **Now:** the WireGuard hub (done on the T4 side). The 2070 host only has to activate its tunnel (section 2).
2. **Next, for newcomers at scale: Headscale + Tailscale clients**, with the T4 as subnet router for `10.77.0.0/24` and access rules that let members reach only the Kafka brokers (`tcp/9092` on `10.77.0.1-4`). A participant then joins with one command and a single-use, expiring key; revocation is one command. If a strictly open-source client stack matters more than the Windows GUI, use **Nebula** instead (certificates per participant).
3. **In both cases**, add Kafka authentication (SASL per participant) before strangers join: the overlay alone should not be the only lock.

Not done here (each needs an owner decision): running Headscale or Nebula (a new public listener and a firewall rule on a production VM), per-participant Kafka credentials, and a second hub so the T4 is not a single point of failure.
