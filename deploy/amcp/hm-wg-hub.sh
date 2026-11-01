#!/bin/bash
# The T4 as WireGuard hub for overlay members that are not in the Kafka VMs' peer lists (for example a home machine behind NAT).
# Such a member keeps ONE peer, the T4 (a stable address), with AllowedIPs 10.77.0.0/24; the T4 forwards its traffic to the other overlay addresses and masquerades it,
# so the Kafka VMs see 10.77.0.4 and need no change. Nothing else is forwarded: traffic from the overlay to any other network is dropped, so a member cannot reach the VPC.
# Usage (root):  hm-wg-hub.sh up|down|status   [members as 10.77.0.N/32, default 10.77.0.5/32]
# Wired into /etc/wireguard/wg-hivemind.conf as PostUp/PostDown so it follows the interface.
set -euo pipefail
WG=wg-hivemind
NET=10.77.0.0/24
ACTION=${1:-status}
shift || true
MEMBERS=("${@:-10.77.0.5/32}")

rule() {  # rule add|del <table> <chain> <args...>
  local op=$1 table=$2; shift 2
  if iptables -t "$table" -C "$@" 2>/dev/null; then
    [ "$op" = del ] && iptables -t "$table" -D "$@"
  else
    [ "$op" = add ] && iptables -t "$table" -A "$@"
  fi
  return 0
}

case "$ACTION" in
up)
  for m in "${MEMBERS[@]}"; do
    rule add filter FORWARD -i $WG -o $WG -s "$m" -d $NET -j ACCEPT
    rule add filter FORWARD -i $WG -o $WG -d "$m" -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
    rule add nat POSTROUTING -s "$m" -o $WG -j MASQUERADE
  done
  rule add filter FORWARD -j DROP  # last: forward nothing else (this host forwarded nothing before)
  sysctl -q -w net.ipv4.ip_forward=1
  ;;
down)
  sysctl -q -w net.ipv4.ip_forward=0
  rule del filter FORWARD -j DROP
  for m in "${MEMBERS[@]}"; do
    rule del filter FORWARD -i $WG -o $WG -s "$m" -d $NET -j ACCEPT
    rule del filter FORWARD -i $WG -o $WG -d "$m" -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
    rule del nat POSTROUTING -s "$m" -o $WG -j MASQUERADE
  done
  ;;
status)
  echo "ip_forward=$(sysctl -n net.ipv4.ip_forward)"
  iptables -S FORWARD
  iptables -t nat -S POSTROUTING
  ;;
*)
  echo "usage: $0 up|down|status [10.77.0.N/32 ...]" >&2
  exit 2
  ;;
esac
