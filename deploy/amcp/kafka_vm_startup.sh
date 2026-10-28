#!/bin/bash
# GCE startup script for one HiveMind Kafka voter VM (Debian 12). Runs on EVERY boot; everything it does is idempotent.
# Inputs come from instance metadata (none of it is secret):
#   hm-node-id           this node's Kafka node id and its overlay address 10.77.0.<id>
#   hm-cluster-id        the one shared Kafka cluster id
#   hm-server-properties the node's server.properties, produced by scripts/amcp_kafka_cluster.py gen
#   hm-kafka-unit        the systemd unit (deploy/amcp/hivemind-kafka-node.service)
#   hm-wg-peers          one line per other WireGuard peer: "<id> <public key> [<endpoint host:port>]"; re-read every minute
# The WireGuard PRIVATE key is generated here, stays in /etc/wireguard on this VM's disk, and is never printed or sent anywhere. Only the public key is published, as a guest attribute.
set -euo pipefail
md() { curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null || true; }
KAFKA_VERSION=3.8.0

if ! command -v wg >/dev/null || ! [ -d /usr/lib/jvm/default-java ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq wireguard-tools openjdk-17-jre-headless curl ca-certificates
fi
id kafka >/dev/null 2>&1 || useradd --system --home /var/lib/hivemind-kafka --shell /usr/sbin/nologin kafka

if [ ! -d /opt/kafka_2.13-${KAFKA_VERSION} ]; then
  cd /tmp
  curl -fsSLO "https://archive.apache.org/dist/kafka/${KAFKA_VERSION}/kafka_2.13-${KAFKA_VERSION}.tgz"
  expected=$(curl -fsSL "https://archive.apache.org/dist/kafka/${KAFKA_VERSION}/kafka_2.13-${KAFKA_VERSION}.tgz.sha512" | tr -d ' \n' | sed 's/^.*://' | tr 'A-F' 'a-f')
  actual=$(sha512sum "kafka_2.13-${KAFKA_VERSION}.tgz" | cut -d' ' -f1)
  [ -n "$expected" ] && [ "$expected" = "$actual" ] || { echo "kafka tarball checksum mismatch" >&2; exit 1; }
  tar -xzf "kafka_2.13-${KAFKA_VERSION}.tgz" -C /opt && ln -sfn "/opt/kafka_2.13-${KAFKA_VERSION}" /opt/kafka
fi

NODE_ID=$(md hm-node-id)
[ -n "$NODE_ID" ] || { echo "no hm-node-id" >&2; exit 1; }
install -d -m 700 /etc/wireguard
if [ ! -s /etc/wireguard/hm.key ]; then (umask 077; wg genkey > /etc/wireguard/hm.key); fi
wg pubkey < /etc/wireguard/hm.key > /etc/wireguard/hm.pub
curl -fsS -X PUT --data "$(cat /etc/wireguard/hm.pub)" -H 'Metadata-Flavor: Google' \
  "http://metadata.google.internal/computeMetadata/v1/instance/guest-attributes/hivemind/wg-pubkey" >/dev/null 2>&1 || true

cat > /usr/local/bin/hm-wg-sync <<'SYNC'
#!/bin/bash
# Render /etc/wireguard/wg-hivemind.conf from metadata and apply it without dropping the tunnel.
set -euo pipefail
md() { curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null || true; }
ID=$(md hm-node-id)
{
  echo "[Interface]"; echo "Address = 10.77.0.${ID}/24"; echo "ListenPort = 51820"; echo "PrivateKey = $(cat /etc/wireguard/hm.key)"
  md hm-wg-peers | while read -r pid pub ep; do
    [ -n "${pid:-}" ] && [ -n "${pub:-}" ] || continue
    echo; echo "[Peer]"; echo "PublicKey = ${pub}"; echo "AllowedIPs = 10.77.0.${pid}/32"; echo "PersistentKeepalive = 25"
    [ -z "${ep:-}" ] || echo "Endpoint = ${ep}"
  done
} > /etc/wireguard/wg-hivemind.conf.new
chmod 600 /etc/wireguard/wg-hivemind.conf.new
if ! cmp -s /etc/wireguard/wg-hivemind.conf /etc/wireguard/wg-hivemind.conf.new 2>/dev/null; then
  mv /etc/wireguard/wg-hivemind.conf.new /etc/wireguard/wg-hivemind.conf
  if ip link show wg-hivemind >/dev/null 2>&1; then wg syncconf wg-hivemind <(wg-quick strip wg-hivemind); else systemctl enable --now wg-quick@wg-hivemind; fi
else rm -f /etc/wireguard/wg-hivemind.conf.new; fi
# Publish a status line as a guest attribute so the state can be read from outside without SSH. No keys: unit states, handshake ages and the last Kafka log lines only.
put() { curl -fsS -X PUT --data "$2" -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/guest-attributes/hivemind/$1" >/dev/null 2>&1 || true; }
now=$(date +%s)
put status "wg=$(systemctl is-active wg-quick@wg-hivemind || true) kafka=$(systemctl is-active hivemind-kafka-node || true) peers=$(md hm-wg-peers | grep -c . || true) handshake_age_s=[$(wg show wg-hivemind latest-handshakes 2>/dev/null | awk -v n="$now" '{printf "%s:%d ", substr($1,1,6), ($2==0 ? -1 : n-$2)}')] t=${now}"
put kafka-log "$(journalctl -u hivemind-kafka-node -n 6 --no-pager 2>/dev/null | tail -c 1500 || true)"
SYNC
chmod 755 /usr/local/bin/hm-wg-sync
cat > /etc/systemd/system/hm-wg-sync.service <<'U'
[Unit]
Description=Apply HiveMind WireGuard peers from instance metadata
[Service]
Type=oneshot
ExecStart=/usr/local/bin/hm-wg-sync
U
cat > /etc/systemd/system/hm-wg-sync.timer <<'U'
[Unit]
Description=Re-read HiveMind WireGuard peers every minute
[Timer]
OnBootSec=5s
OnUnitActiveSec=60s
[Install]
WantedBy=timers.target
U

install -d -m 755 /etc/hivemind-kafka
md hm-server-properties > /etc/hivemind-kafka/server.properties
md hm-kafka-unit > /etc/systemd/system/hivemind-kafka-node.service
install -d -o kafka -g kafka /var/lib/hivemind-kafka /var/log/hivemind-kafka
if [ ! -f /var/lib/hivemind-kafka/meta.properties ]; then
  sudo -u kafka env JAVA_HOME=/usr/lib/jvm/default-java /opt/kafka/bin/kafka-storage.sh format -t "$(md hm-cluster-id)" -c /etc/hivemind-kafka/server.properties --ignore-formatted
fi
systemctl daemon-reload
systemctl enable --now hm-wg-sync.timer
/usr/local/bin/hm-wg-sync || true
systemctl enable --now hivemind-kafka-node.service
