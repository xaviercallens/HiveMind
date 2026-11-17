#!/bin/bash
# The T4 as WireGuard hub for overlay members that are not in the Kafka VMs' peer lists (for example a home machine behind NAT).
# Such a member keeps ONE peer, the T4 (a stable address), with AllowedIPs 10.77.0.0/24; the T4 forwards its traffic to the other overlay addresses and masquerades it,
# so the Kafka VMs see 10.77.0.4 and need no change. Nothing else is forwarded: traffic from the overlay to any other network is dropped, so a member cannot reach the VPC.
# Usage (root):
#   hm-wg-hub.sh up [members]            enable forwarding, the final DROP, and the members' rules (default member 10.77.0.5/32)
#   hm-wg-hub.sh down [members]          remove the members' rules and the DROP, disable forwarding
#   hm-wg-hub.sh add-member 10.77.0.N/32     allow one more member (rules inserted before the DROP)
#   hm-wg-hub.sh remove-member 10.77.0.N/32  remove one member's rules, leave the hub up
#   hm-wg-hub.sh status
# Wired into /etc/wireguard/wg-hivemind.conf as PostUp/PostDown so it follows the interface.
set -euo pipefail
WG=wg-hivemind
NET=10.77.0.0/24
ACTION=${1:-status}
shift || true
MEMBERS=("${@:-10.77.0.5/32}")

has() { local table=$1; shift; iptables -t "$table" -C "$@" 2>/dev/null; }

member() {  # member add|del <cidr>
  local op=$1 m=$2
  local fwd_out=(FORWARD -i $WG -o $WG -s "$m" -d $NET -j ACCEPT)
  local fwd_back=(FORWARD -i $WG -o $WG -d "$m" -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT)
  local nat=(POSTROUTING -s "$m" -o $WG -j MASQUERADE)
  if [ "$op" = add ]; then
    # ACCEPT rules go to the top of FORWARD so they always come before the final DROP
    has filter "${fwd_back[@]}" || iptables -I FORWARD 1 "${fwd_back[@]:1}"
    has filter "${fwd_out[@]}" || iptables -I FORWARD 1 "${fwd_out[@]:1}"
    has nat "${nat[@]}" || iptables -t nat -A "${nat[@]}"
  else
    while has filter "${fwd_out[@]}"; do iptables -D "${fwd_out[@]}"; done
    while has filter "${fwd_back[@]}"; do iptables -D "${fwd_back[@]}"; done
    while has nat "${nat[@]}"; do iptables -t nat -D "${nat[@]}"; done
  fi
}

case "$ACTION" in
up)
  has filter FORWARD -j DROP || iptables -A FORWARD -j DROP  # last: forward nothing else (this host forwarded nothing before)
  for m in "${MEMBERS[@]}"; do member add "$m"; done
  sysctl -q -w net.ipv4.ip_forward=1
  ;;
down)
  sysctl -q -w net.ipv4.ip_forward=0
  for m in "${MEMBERS[@]}"; do member del "$m"; done
  while has filter FORWARD -j DROP; do iptables -D FORWARD -j DROP; done
  ;;
add-member)
  [ "$#" -ge 1 ] || { echo "add-member needs 10.77.0.N/32" >&2; exit 2; }
  member add "$1"
  ;;
remove-member)
  [ "$#" -ge 1 ] || { echo "remove-member needs 10.77.0.N/32" >&2; exit 2; }
  member del "$1"
  ;;
status)
  echo "ip_forward=$(sysctl -n net.ipv4.ip_forward)"
  iptables -S FORWARD
  iptables -t nat -S POSTROUTING
  ;;
*)
  echo "usage: $0 up|down|add-member|remove-member|status [10.77.0.N/32 ...]" >&2
  exit 2
  ;;
esac
