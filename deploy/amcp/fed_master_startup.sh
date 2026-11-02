#!/bin/bash
# HiveMind federation master (Spot VM): WireGuard hub for the RTX 2070 and registered trainers, coordinator of co-training sessions, HTTPS relay to the
# 2070's producer shield. Replaces what the T4 did for co-training. Runs at every boot (GCE startup script), so a Spot preemption + restart resumes by itself.
#
# Instance metadata it reads (re-read every minute unless noted):
#   hm-branch        repository branch to run (read at first boot)
#   hm-kafka-peers   one line per Kafka voter: "<overlay id> <wg public key> <VPC ip>:51820"
#   hm-members       one line per member (2070, trainers): "<overlay id> <wg public key>"; each gets ONE peer (this hub) with AllowedIPs 10.77.0.0/24
#   hm-session       a co-training session: "RUN|ROUNDS|LEARNER IDS (space separated)|OUTER_LR"; changing it starts a new session, empty = none
#   hm-fed-keys      HIVEMIND_AMCP_HMAC_KEYS for the session ("coord:<hex>,lk-<id>:<hex>,..."): signing keys, never logged or published
#   hm-ckpt-gs       optional gs:// checkpoint to resume a new session from (copied once)
#   hm-results-gs    gs:// prefix; the session's results, logs and checkpoint are copied there every 5 minutes and at the end
#   hm-relay-target  optional "10.77.0.5:8787": the 2070 producer shield, served at https://<this ip as a-b-c-d>.sslip.io/rtx2070
# Status (no SSH needed; create the VM with metadata enable-guest-attributes=TRUE):
#   gcloud compute instances get-guest-attributes hm-fed-master --zone us-east4-a  (hivemind/status, wg-pubkey, session, relay)
set -uo pipefail
md() { curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1" 2>/dev/null || true; }
attr() { md "attributes/$1"; }
put() { curl -fsS -X PUT --data "$2" -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/guest-attributes/hivemind/$1" >/dev/null 2>&1 || true; }
B=/opt/hmmaster
mkdir -p $B/sessions

# ---------------------------------------------------------------------------------------------------------------- install (first boot)
if [ ! -f $B/deps_ok ]; then
  put status "installing"
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv git wireguard-tools iptables curl caddy > $B/apt.log 2>&1
  systemctl disable --now caddy >/dev/null 2>&1 || true  # started below only when a relay target is set
  python3 -m venv $B/venv
  $B/venv/bin/pip install -q --upgrade pip > $B/pip.log 2>&1
  $B/venv/bin/pip install -q torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu >> $B/pip.log 2>&1
  $B/venv/bin/pip install -q transformers==5.17.0 peft==0.21.2 accelerate==1.15.0 confluent-kafka==2.16.1 numpy safetensors >> $B/pip.log 2>&1 && touch $B/deps_ok
  [ -f $B/deps_ok ] || { put status "deps FAILED (see $B/pip.log)"; exit 1; }
fi
if [ ! -d $B/HiveMind ]; then git clone -q --depth 1 --branch "$(attr hm-branch)" https://github.com/xaviercallens/HiveMind.git $B/HiveMind || { put status "clone failed"; exit 1; }; fi
REPO=$B/HiveMind
put commit "$(git -C $REPO log -1 --format=%H)"

# ---------------------------------------------------------------------------------------------------------------- WireGuard hub
install -d -m 700 /etc/wireguard
[ -s /etc/wireguard/hm.key ] || (umask 077; wg genkey > /etc/wireguard/hm.key)
put wg-pubkey "$(wg pubkey < /etc/wireguard/hm.key)"
sysctl -q -w net.ipv4.ip_forward=1

wg_conf() {
  printf '[Interface]\nAddress = 10.77.0.10/24\nListenPort = 51820\nPrivateKey = %s\n' "$(cat /etc/wireguard/hm.key)"
  attr hm-kafka-peers | while read -r id pub ep; do
    [ -n "${pub:-}" ] || continue
    printf '\n[Peer]\nPublicKey = %s\nAllowedIPs = 10.77.0.%s/32\nEndpoint = %s\nPersistentKeepalive = 25\n' "$pub" "$id" "$ep"
  done
  attr hm-members | while read -r id pub; do
    [ -n "${pub:-}" ] || continue
    printf '\n[Peer]\nPublicKey = %s\nAllowedIPs = 10.77.0.%s/32\nPersistentKeepalive = 25\n' "$pub" "$id"
  done
}

sync_wg() {
  local new; new="$(wg_conf)"
  if [ "$new" != "$(cat /etc/wireguard/wg-hivemind.conf 2>/dev/null)" ]; then
    (umask 077; printf '%s\n' "$new" > /etc/wireguard/wg-hivemind.conf)
    if ip link show wg-hivemind >/dev/null 2>&1; then wg syncconf wg-hivemind <(wg-quick strip wg-hivemind); else wg-quick up wg-hivemind > $B/wg.log 2>&1; fi
  fi
  # forward only member -> overlay, masqueraded, so the Kafka voters see 10.77.0.10 and need no per-member peer; drop everything else
  local members=(); while read -r id pub; do [ -n "${pub:-}" ] && members+=("10.77.0.${id}/32"); done < <(attr hm-members)
  [ "${#members[@]}" -eq 0 ] || bash $REPO/deploy/amcp/hm-wg-hub.sh up "${members[@]}" > /dev/null 2>&1
}

# ---------------------------------------------------------------------------------------------------------------- HTTPS relay to the 2070 producer
sync_relay() {
  local target ip host; target="$(attr hm-relay-target)"
  if [ -z "$target" ]; then systemctl stop caddy >/dev/null 2>&1; put relay "off"; return; fi
  ip="$(md network-interfaces/0/access-configs/0/external-ip)"; host="${ip//./-}.sslip.io"
  local cf; cf="$(printf '%s {\n\thandle_path /rtx2070* {\n\t\treverse_proxy %s\n\t}\n\trespond 404\n}\n' "$host" "$target")"
  if [ "$cf" != "$(cat /etc/caddy/Caddyfile 2>/dev/null)" ]; then printf '%s' "$cf" > /etc/caddy/Caddyfile; systemctl restart caddy; fi
  systemctl is-active -q caddy || systemctl start caddy
  put relay "https://${host}/rtx2070 -> ${target} ($(systemctl is-active caddy))"
}

# ---------------------------------------------------------------------------------------------------------------- co-training session
sync_results() {  # $1 = run id
  local gs; gs="$(attr hm-results-gs)"; [ -n "$gs" ] || return 0
  gcloud storage cp -q $B/sessions/"$1".* "${gs%/}/$1/" > /dev/null 2>&1 || true
}

run_session() {  # one coordinator process per session; resumes from its own checkpoint after a crash or a preemption
  local spec="$1" run rounds learners lr
  IFS='|' read -r run rounds learners lr <<< "$spec"
  [ -n "$run" ] && [ -n "$rounds" ] && [ -n "$learners" ] || { put session "bad hm-session: RUN|ROUNDS|LEARNERS|OUTER_LR"; return; }
  local S=$B/sessions/$run
  [ -f "$S.done" ] && { put session "$run finished (rc=$(cat "$S.done"))"; return; }
  if [ ! -f "$S.ckpt" ] && [ -n "$(attr hm-ckpt-gs)" ]; then gcloud storage cp -q "$(attr hm-ckpt-gs)" "$S.ckpt" > /dev/null 2>&1 || true; fi
  put session "running $run (${learners})"
  # shellcheck disable=SC2086  # learner ids are deliberately word-split
  ( cd $REPO && HIVEMIND_KAFKA_BOOTSTRAP=10.77.0.1:9092,10.77.0.2:9092,10.77.0.3:9092 HIVEMIND_AMCP_HMAC_KEYS="$(attr hm-fed-keys)" HIVEMIND_AMCP_HMAC_ACTIVE_KID=coord \
    $B/venv/bin/python -W ignore scripts/e8_amcp_round.py coordinator --run-id "$run" --learners $learners --rounds "$rounds" --quorum 1 \
      --round-deadline-s 180 --max-retries 40 --pace-s 30 --inner-steps 40 --inner-lr 1e-2 --outer-lr "${lr:-0.25}" --outer-mu 0.5 \
      --learner-key-prefix lk- --checkpoint "$S.ckpt" --resume --out "$S.json" > "$S.log" 2>&1 )
  local rc=$?
  sync_results "$run"
  if [ "$rc" -eq 0 ]; then echo 0 > "$S.done"; put session "$run finished rc=0"; else put session "$run stopped rc=$rc; resuming in 60 s"; sleep 60; fi
}

session_loop() {
  while true; do
    local spec; spec="$(attr hm-session)"
    if [ -n "$spec" ] && [ -n "$(attr hm-fed-keys)" ]; then run_session "$spec"; fi
    sleep 30
  done
}

results_loop() { while true; do sleep 300; for f in $B/sessions/*.log; do [ -e "$f" ] && sync_results "$(basename "$f" .log)"; done; done; }

session_loop &
results_loop &
while true; do
  put wg-pubkey "$(wg pubkey < /etc/wireguard/hm.key)"  # re-published every minute: guest attributes may be enabled after boot
  sync_wg
  sync_relay
  put status "wg=$(ip link show wg-hivemind >/dev/null 2>&1 && echo up || echo down) peers=$(wg show wg-hivemind peers 2>/dev/null | grep -c .) handshakes=[$(wg show wg-hivemind latest-handshakes 2>/dev/null | awk -v n="$(date +%s)" '{printf "%s:%d ", substr($1,1,6), ($2==0 ? -1 : n-$2)}')] t=$(date -u +%FT%TZ)"
  sleep 60
done
