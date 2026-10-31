# A self-run Kafka cluster: three voters (Spot VMs) plus the T4 and the RTX 2070 as brokers

Companion to [`AMCP_COLLABORATION.md`](AMCP_COLLABORATION.md). Status 2026-10-28: **proven on one machine (3-node and 5-node); cloud deployment in progress, see section 9.**

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

## 2. Topology (updated 2026-10-28: three Spot VMs vote, the T4 and the RTX 2070 are broker-only additions)

Why not five voters: Kafka 3.8 fixes the voter list at format time. A voter that is not there yet (the RTX 2070, unreachable from here) still counts toward the majority, so five listed voters would need 3 of the 4 reachable
machines up, which is *worse* than 2 of 3 (at 90% uptime per node: about 0.95 versus 0.97). Broker-only nodes can be added at any time and carry data, so they add resilience for the data without weakening the vote.

| Node id | Host | Role | Overlay address |
|---|---|---|---|
| 1, 2, 3 | three small Spot VMs, one per zone `us-east4-a/b/c`, stop-on-preemption so disks and keys survive | **voter** (broker + controller) | `10.77.0.1-3` |
| 4 | the T4 VM (Spot) | broker only | `10.77.0.4` |
| 5 | RTX 2070 machine at home | broker only | `10.77.0.5` |

Two separate limits, both measured by `scripts/amcp_kafka_quorum_proof.py` on five local nodes (topics replication 5, min in-sync 3):

| | Tolerates | Evidence |
|---|---|---|
| Controller vote (3 voters) | **1** voter down | with voter 3 and broker 5 down, 10 of 10 writes acknowledged |
| Data replicas (5 replicas, min in-sync 3) | **2** replicas down | same run; with a third replica down the write was **refused** (30 s timeout, 0 acknowledged); back up: 10 of 10 |
| Controller vote lost (only voter 1 left, 3 replicas up) | none: writes refused here | observed; also a broker cannot *start* while the vote is lost (timeout when restarting node 5) |
| After everything returned | 40 acknowledged, 40 read back, none lost, none duplicated, order kept | |

So the honest claim is: **any two machines may fail as long as at least two of the three voters stay up; losing two voters stops the bus.** Topics at replication 3 / min in-sync 2 (the default of the three-node
proof) tolerate only one replica down, whatever the node count, so provision at `--replication 5 --min-isr 3` to get the two-failure behaviour for data.

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

## 8. Deployed state (2026-10-31)

| | State |
|---|---|
| Voters 1, 2, 3 | **running**: Spot `e2-small` VMs `hm-kafka-1/2/3` (us-east4-a/b/c, VPC 10.150.0.11-13, project `gen-lang-client-0625573011`), Debian 12, no service account, Kafka 3.8.0 checksum-verified, heap from the unit |
| Node 4 (T4) | **running** as broker-only (observer), unit `hivemind-kafka-t4-node`, data on the data disk, bound to `10.77.0.4`. The dev broker on `127.0.0.1:9092` is untouched and separate |
| Node 5 (RTX 2070) | **not joined**: no route to that machine from here; `deploy/amcp/join_rtx2070.sh` is the procedure (section 9) |
| Overlay | WireGuard `10.77.0.0/24`, full mesh between nodes 1-4 over VPC addresses; UDP 51820 is the only port opened (rule `hm-kafka-wireguard`, tag `hm-kafka`); no Kafka port is open anywhere |
| Keepalive | Cloud Scheduler `hm-kafka-{1,2,3}-keepalive`, every 5 minutes, calls `instances.start` (a no-op on a running VM) with the project's default compute service account, like the existing `hm-t4-keepalive` |
| Topics | provisioned with replication 3 / min in-sync 2 (replication 5 needs five brokers; raise it with a partition reassignment once node 5 joins) |

**Measured on the real hosts** (`scripts/amcp_kafka_real_proof.py`, publishing over the overlay, stopping the real VMs with gcloud): all voters up 10/10 acknowledged; one voter VM stopped 10/10 acknowledged;
two stopped: write refused, 0 acknowledged; both restarted: acknowledged again after **43.5 s**; 21 acknowledged, 21 read back, none lost, none duplicated, in order. After a later full outage the three VMs
came back, re-elected a leader on their own, and all four nodes reported lag 0.

**Things the first real deployment taught** (all fixed in the scripts):
* A Spot VM was preempted **17 minutes** after its first boot and again 10 minutes after restart; by the next check all three were stopped. Spot with no keepalive means the bus is simply down for long periods: the keepalive is required, not optional, and even with it a preempted VM is down until the next 5-minute tick plus boot time.
* Debian 12 with only the headless JRE has no `/usr/lib/jvm/default-java`: the first boot failed with exit status 127. The path is `/usr/lib/jvm/java-17-openjdk-amd64`.
* `archive.apache.org` served the 120 MB tarball at about 260 KB/s (roughly 8 minutes per VM boot that has to fetch it). A mirror or a pre-baked image would shorten recovery of a rebuilt VM.
* Without SSH, the startup script publishes a status line (`hivemind/status`) and the last Kafka log lines (`hivemind/kafka-log`) as **guest attributes**: read them with `gcloud compute instances get-guest-attributes NAME --zone ZONE`. No keys are in them.
* The T4's tunnel and node 4 were first started by hand; the unit file and `systemctl enable hivemind-kafka-t4-node wg-quick@wg-hivemind` make them survive a reboot (the enable step must be run by a person with sudo).

Security notes for this deployment: the VMs sit in the default VPC, where `default-allow-internal` (10.128.0.0/9) and several rules open to `0.0.0.0/0` without target tags (including SSH) apply to every VM in the project, these two as well.
Kafka itself listens only on overlay addresses, so none of that reaches it, but the VMs' SSH exposure is the project's existing posture, not something this change tightened.

## 9. Joining the RTX 2070 (node 5)

1. On the 2070 machine: `bash deploy/amcp/join_rtx2070.sh key` installs WireGuard and a JDK and prints this machine's **public** key.
2. Cloud side (a person with project access): add the T4 to the firewall rule's tag so the 2070 can reach it directly (`gcloud compute instances add-tags socreateai-agora-hermes-node1 --zone us-east4-b --tags hm-kafka`: opens UDP 51820 only). Add the 2070's key as peer 5 (`5 <public key>` with no endpoint) to `hm-wg-peers` on the three VMs, and as a `[Peer]` with `AllowedIPs = 10.77.0.5/32` on the T4.
   Generate node 5's config with `python scripts/amcp_kafka_cluster.py gen --node 1=10.77.0.1 --node 2=10.77.0.2 --node 3=10.77.0.3 --broker 4=10.77.0.4 --broker 5=10.77.0.5 --out ./cluster`.
3. On the 2070: `HUB_KEY_1=.. HUB_KEY_2=.. HUB_KEY_3=.. HUB_KEY_4=.. bash deploy/amcp/join_rtx2070.sh up <VM1 IP> <VM2 IP> <VM3 IP> <T4 IP> lLTifvfsQmypwcwcQutGrA ./cluster/node5.properties` (the VMs' external addresses are the reserved `hm-kafka-1/2/3`; their public keys are in guest attribute `hivemind/wg-pubkey`), then start the broker it prints.
4. Check: `kafka-metadata-quorum.sh --bootstrap-server 10.77.0.1:9092 describe --replication` lists node 5 as an Observer with lag 0. Then, optionally, raise the AMCP topics to replication 5 / min in-sync 3 with `kafka-reassign-partitions.sh` (the five-node proof in section 2 used those settings).

## 10. Still open

1. Reaching the cluster from the production gateway on Cloud Run (section 6): not built. The gateway's bus stays off.
2. Owner decisions: keep Spot for all three voters (cheap, but the bus is down during preemption gaps), or make one voter a non-Spot VM.
3. `systemctl enable` for the T4 units (above), and the RTX 2070 join.
