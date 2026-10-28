# A self-run Kafka cluster with a quorum of three (RTX 2070, T4, one small GCP VM)

Companion to [`AMCP_COLLABORATION.md`](AMCP_COLLABORATION.md). Status 2026-10-28: **designed and proven on one machine; not yet deployed on the real hosts.**

## 1. What is proven, and what is not

| | State | Evidence |
|---|---|---|
| Three-node KRaft quorum, replication 3, min in-sync 2, on loopback | **proven** | `scripts/amcp_kafka_cluster.py local up`; `kafka-metadata-quorum describe` reports voters `[1,2,3]`, a leader, follower lag 0 |
| One node down: writes still acknowledged | **proven** | `scripts/amcp_kafka_quorum_proof.py`: 20 of 20 acknowledged with 3 of 3 up and again with 2 of 3 up |
| Quorum lost (1 of 3 up): writes **refused**, never acknowledged | **proven** | the write timed out after 30 s with zero acknowledged |
| Recovery after the nodes return | **proven** | 19 s to restart two nodes, then 20 of 20 acknowledged |
| No acknowledged event lost, none duplicated, order kept | **proven** | 60 acknowledged, 60 read back, in order; the refused event is absent |
| The project's AMCP transport and CLI against the cluster | **proven** | the live tests pass on it (9 passed) |
| The same on the real hosts, over a network, with the RTX 2070 | **not done** | the hosts were not touched; sections 3 to 5 are the procedure |
| Reachable from the production gateway on Cloud Run | **not solved** | see section 6 |

## 2. Topology and what it actually tolerates

Three voters; a majority is two; **one node may be down.**

| Node id | Host | Address on the private overlay | Character |
|---|---|---|---|
| 1 | small GCP VM (suggested `e2-small`, not Spot, same region as the T4: `us-east4`) | `10.77.0.1` (also the WireGuard hub) | the steady anchor |
| 2 | RTX 2070 machine at home | `10.77.0.2` | behind a home connection; uptime unknown |
| 3 | the T4 VM (`socreateai-agora-hermes-node1`, Spot, automatic restart off) | `10.77.0.3` | can vanish at any time |

| Nodes up | Cluster |
|---|---|
| 3 | healthy |
| any 2 | healthy (proven) |
| 1 | **down for writes** (proven: refused after the timeout). Treat it as down. |

**Honest consequence of this choice.** Until the RTX 2070 joins, the quorum is the small VM plus the Spot T4: if the T4 is preempted the bus is down until it returns. With all three, the bus is down only when two are
out at once, for example a preempted T4 and a home machine that is off. If that is not acceptable, replace the T4 by a **second small non-Spot VM** (a small monthly cost that I did not price: check the console), and keep the T4 and
the RTX 2070 for what they are good at.

## 3. Network: a private overlay, nothing public

Kafka here has **no authentication and no encryption**. It must never listen on a public address. Use a WireGuard overlay `10.77.0.0/24`, hub-and-spoke through the small VM:

1. Open **only UDP 51820** on the small VM (a new firewall rule restricted to that VM by a network tag). No Kafka port is opened anywhere.
2. On each host: `umask 077; wg genkey | tee wg.key | wg pubkey > wg.pub`. **Private keys never leave their host and are never committed.** Exchange public keys only.
3. Hub (`/etc/wireguard/wg-hivemind.conf`): `[Interface] Address=10.77.0.1/24, ListenPort=51820, PrivateKey=...` plus one `[Peer]` per spoke with `AllowedIPs=10.77.0.2/32` (and `.3`).
   Spokes: `[Interface] Address=10.77.0.N/24` and a `[Peer]` for the hub with `Endpoint=<hub address>:51820, AllowedIPs=10.77.0.0/24, PersistentKeepalive=25` (the keepalive lets the home machine sit behind NAT).
4. `systemctl enable --now wg-quick@wg-hivemind`. Check `wg show` and `ping 10.77.0.1` from each spoke.

Everyone who holds a peer key on this overlay is **fully trusted by the cluster** (no ACLs). The peer list is the trust list.

## 4. Install and start a node (on each host)

```bash
# Kafka 3.8.0, verified like mesh/quickstart.sh does it; a JDK 17 or newer
curl -fsSLO https://archive.apache.org/dist/kafka/3.8.0/kafka_2.13-3.8.0.tgz
curl -fsSL  https://archive.apache.org/dist/kafka/3.8.0/kafka_2.13-3.8.0.tgz.sha512      # compare with: sha512sum kafka_2.13-3.8.0.tgz
sudo tar -xzf kafka_2.13-3.8.0.tgz -C /opt && sudo ln -sfn /opt/kafka_2.13-3.8.0 /opt/kafka

# once, on ANY machine: the configuration for all three nodes (this repository)
python scripts/amcp_kafka_cluster.py gen --node 1=10.77.0.1 --node 2=10.77.0.2 --node 3=10.77.0.3 --out ./cluster --data-dir /var/lib/hivemind-kafka
/opt/kafka/bin/kafka-storage.sh random-uuid                       # ONE cluster id, shared by all three

# on each node N: copy clusterN/nodeN.properties to /etc/hivemind-kafka/server.properties, then
sudo -u kafka /opt/kafka/bin/kafka-storage.sh format -t <the shared cluster id> -c /etc/hivemind-kafka/server.properties
sudo cp deploy/amcp/hivemind-kafka-node.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now hivemind-kafka-node
```

Start the three within ten minutes of each other. The generated configuration sets `initial.broker.registration.timeout.ms=600000`: with Kafka's default of 60 seconds, the local test showed a lone node logging
*"unable to register with the controller quorum"* and **exiting** instead of waiting for its peers. With `Restart=always` a node also retries on its own.

Then, from any host on the overlay:

```bash
export HIVEMIND_KAFKA_BOOTSTRAP=10.77.0.1:9092,10.77.0.2:9092,10.77.0.3:9092
/opt/kafka/bin/kafka-metadata-quorum.sh --bootstrap-server 10.77.0.1:9092 describe --status      # CurrentVoters [1,2,3], a leader, lag 0
python scripts/hivemind_amcp.py --dev-keyring provision --replication 3 --min-isr 2
python scripts/hivemind_amcp.py --dev-keyring check
```

## 5. Static quorum, and what that means later

Kafka 3.8 has a **fixed** voter list: controllers cannot be added or removed on a running cluster (that arrived in later releases). So the three addresses above are decided now and every node carries the same list and
the same cluster id. A node that is down is just a voter that is not answering. Replacing a machine means giving the new one the same node id and address, or rebuilding the quorum. The dev broker on `127.0.0.1:9092`
cannot be turned into this cluster; it is a separate single-node cluster with separate data and stays as the local development bus.

## 6. The open problem: the production gateway on Cloud Run

The router (Cloud Run, `us-central1`) would have to reach the brokers. Brokers advertise **overlay** addresses, which Cloud Run cannot route to, and a client that reaches one broker must reach every broker that leads a partition.
Putting a second listener on a VPC address works for the two GCP nodes but not for the home machine. Options, none built: route the overlay through the VPC (custom route to the hub with IP forwarding, plus Direct VPC egress for Cloud Run);
run a small **AMCP relay on the hub** that the gateway calls over HTTPS with its own authentication; or leave the gateway's bus off (`bus.status = not_configured`, as today) and use the cluster for the direct tier only (developers, coding agents, bees).
The last option is the one this runbook assumes until you choose.

## 7. Security, operations, rollback

* No authentication, no TLS on Kafka itself. Before anyone outside the overlay needs access: SASL/SCRAM over TLS (already supported by `KafkaSettings`) and one principal per participant kind with ACLs.
* The log is a bus, not a database: 7-day retention; the durable record of decisions is GitHub and the router's store.
* Data lives on each host's disk. A Spot VM keeps its disk; the process comes back through systemd when the VM returns.
* Rollback: stop and disable the unit, remove `/etc/hivemind-kafka` and the data directory, `systemctl disable --now wg-quick@wg-hivemind` on the T4, delete the VM and the UDP 51820 firewall rule.

## 8. Decisions needed before this goes on the real hosts

1. Create the small GCP VM and the one firewall rule (UDP 51820 only)? Billable, in the production project.
2. Install WireGuard on the T4 production VM (a new network interface, nothing else changes)?
3. Access to the RTX 2070 machine (no route to it exists from here).
4. Keep the Spot T4 as the third voter, or add a second small non-Spot VM?
5. How the gateway reaches the cluster (section 6), or leave it off for now.
