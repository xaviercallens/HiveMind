#!/bin/bash
# Join the RTX 2070 machine to the HiveMind Kafka overlay as broker-only node 5. Run it ON THE 2070 MACHINE (Linux, Debian or Ubuntu), as a normal user with sudo.
# It does three things and tells you each one first:
#   step 1: install wireguard-tools and a JDK, generate a WireGuard key pair here (the private key never leaves this machine) and PRINT THE PUBLIC KEY;
#   step 2: you send that public key to whoever runs the cloud side (they add it as peer 5, see docs/AMCP_KAFKA_CLUSTER.md section 9) and tell you the four peer addresses
#           (the three VMs and the T4: a broker must reach EVERY other broker, so the 2070 tunnels to all four directly);
#   step 3: with the addresses, it writes the tunnel config, brings the tunnel up, downloads and verifies Kafka, and starts node 5.
# Usage:  bash join_rtx2070.sh key
#         bash join_rtx2070.sh up HUB1_IP HUB2_IP HUB3_IP T4_IP CLUSTER_ID NODE5_PROPERTIES_FILE   (needs HUB_KEY_1..4 in the environment: the public keys of nodes 1-4)
set -euo pipefail
KAFKA_VERSION=3.8.0
BASE="${HIVEMIND_KAFKA_BASE:-$HOME/hivemind-kafka}"
mode="${1:-}"

case "$mode" in
key)
  sudo apt-get install -y -qq wireguard-tools openjdk-17-jre-headless curl
  sudo install -d -m 700 /etc/wireguard
  sudo bash -c 'umask 077; [ -s /etc/wireguard/hm.key ] || wg genkey > /etc/wireguard/hm.key; wg pubkey < /etc/wireguard/hm.key'
  echo "Above is the PUBLIC key of this machine. Send it to the cloud operator. Never send /etc/wireguard/hm.key."
  ;;
up)
  [ "$#" -eq 7 ] || { echo "usage: $0 up HUB1_IP HUB2_IP HUB3_IP T4_IP CLUSTER_ID NODE5_PROPERTIES_FILE" >&2; exit 2; }
  hubs=("$2" "$3" "$4" "$5"); cluster_id="$6"; props="$7"
  for i in 1 2 3 4; do eval "k=\${HUB_KEY_$i:-}"; [ -n "$k" ] || { echo "set HUB_KEY_1 .. HUB_KEY_4 (the public keys of nodes 1-4) first" >&2; exit 2; }; done
  conf=$(mktemp)
  {
    printf '[Interface]\nAddress = 10.77.0.5/24\nListenPort = 51820\nPrivateKey = %s\n' "$(sudo cat /etc/wireguard/hm.key)"
    for i in 1 2 3 4; do eval "k=\${HUB_KEY_$i}"; printf '\n[Peer]\nPublicKey = %s\nAllowedIPs = 10.77.0.%s/32\nEndpoint = %s:51820\nPersistentKeepalive = 25\n' "$k" "$i" "${hubs[$((i-1))]}"; done
  } > "$conf"
  sudo install -m 600 -o root "$conf" /etc/wireguard/wg-hivemind.conf; rm -f "$conf"
  sudo systemctl enable --now wg-quick@wg-hivemind
  ping -c 2 -W 3 10.77.0.1 || { echo "no answer from 10.77.0.1: is this machine's public key added on the cloud side, and UDP 51820 reachable?" >&2; exit 1; }
  mkdir -p "$BASE" && cd "$BASE"
  if [ ! -d "kafka_2.13-${KAFKA_VERSION}" ]; then
    curl -fsSLO "https://archive.apache.org/dist/kafka/${KAFKA_VERSION}/kafka_2.13-${KAFKA_VERSION}.tgz"
    expected=$(curl -fsSL "https://archive.apache.org/dist/kafka/${KAFKA_VERSION}/kafka_2.13-${KAFKA_VERSION}.tgz.sha512" | tr -d ' \n' | sed 's/^.*://' | tr 'A-F' 'a-f')
    actual=$(sha512sum "kafka_2.13-${KAFKA_VERSION}.tgz" | cut -d' ' -f1)
    [ -n "$expected" ] && [ "$expected" = "$actual" ] || { echo "Kafka checksum mismatch: not installing" >&2; exit 1; }
    tar -xzf "kafka_2.13-${KAFKA_VERSION}.tgz"
  fi
  cp "$props" "$BASE/server.properties"
  sed -i "s#^log.dirs=.*#log.dirs=$BASE/data#" "$BASE/server.properties"
  export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
  "kafka_2.13-${KAFKA_VERSION}/bin/kafka-storage.sh" format -t "$cluster_id" -c "$BASE/server.properties" --ignore-formatted
  echo "Formatted. Start it with:  KAFKA_HEAP_OPTS='-Xms128m -Xmx384m' JAVA_HOME=$JAVA_HOME $BASE/kafka_2.13-${KAFKA_VERSION}/bin/kafka-server-start.sh $BASE/server.properties"
  echo "To keep it running across reboots, adapt deploy/amcp/hivemind-kafka-t4-node.service (paths and user)."
  ;;
*)
  echo "usage: $0 key | up HUB1_IP HUB2_IP HUB3_IP T4_IP CLUSTER_ID NODE5_PROPERTIES_FILE" >&2
  exit 2
  ;;
esac
