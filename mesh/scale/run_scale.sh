#!/bin/bash
# MESH-SCALE-1 harness: N simulated bees on one VM (2 hosts, 4 bee JVMs, one loadgen, one recorder),
# against the local Kafka broker at 127.0.0.1:9092. Pre-registered in
# artifacts/experiments/MESH_SCALE_prereg.json. Never runs a GPU, never binds a port.
#
# usage: run_scale.sh N [duration_s]        (N must be a multiple of 4; prereg runs use duration 120)
#
# Kill timing: bee JVMs kill at load start + KILL_AT seconds, KILL_AT = duration/2 (60 s for the prereg
# 120 s load). Smoke runs at other durations are labelled non-prereg by analyze.py.
#
# Every process started here is recorded in processes.txt and stopped on exit (trap). Only those PIDs
# are signalled. VmHWM of the long-lived JVMs is read before anything is stopped.
set -u

N=${1:?usage: run_scale.sh N [duration_s]}
DUR=${2:-120}
case "$N" in ''|*[!0-9]*) echo "N must be a positive integer" >&2; exit 2 ;; esac
if [ $((N % 4)) -ne 0 ]; then echo "N must be a multiple of 4 (4 bee JVMs)" >&2; exit 2; fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MESH_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
AMCP=/mnt/disks/disk-socrateai-local-1/amcp
JAVA="$AMCP/tools/jdk21/bin/java"
JAR="$MESH_DIR/target/hivemind-bee-mesh-0.1.0-all.jar"
BOOT=127.0.0.1:9092
RATE=200
BEE_JVMS=4
HOSTS=2
KILL_FRACTION=0.10
KILL_AT=$((DUR / 2))
MEM_MIN_KB=$((6 * 1024 * 1024))

PIDS=()
NAMES=()
RC_FILE=""
STOPPED=0
LAST_PID=0

# ---- preflight -------------------------------------------------------------------------------------
MEM_KB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
if [ -z "$MEM_KB" ] || [ "$MEM_KB" -lt "$MEM_MIN_KB" ]; then
  echo "abort: MemAvailable ${MEM_KB:-unknown} kB is below 6 GB" >&2
  exit 3
fi
if ! bash "$AMCP/kafka_local.sh" status | grep -qx "kafka up"; then
  echo "abort: Kafka is not up on $BOOT (run kafka_local.sh start by hand)" >&2
  exit 3
fi
if [ ! -f "$JAR" ]; then
  echo "abort: $JAR is missing (build the mesh first)" >&2
  exit 3
fi

# ---- run directory ---------------------------------------------------------------------------------
TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="hmscale$(date +%s)n$N"
OUT="$AMCP/scale-runs/$TS-N$N"
mkdir -p "$OUT/logs" "$OUT/events"
RC_FILE="$OUT/exit_codes.txt"
: > "$OUT/processes.txt"
cp "$JAR" "$OUT/mesh.jar"
JAR_SHA=$(sha256sum "$OUT/mesh.jar" | awk '{print $1}')
PREREG_SHA=$(sha256sum "$MESH_DIR/../artifacts/experiments/MESH_SCALE_prereg.json" 2>/dev/null | awk '{print $1}')
echo "run dir: $OUT  root: $ROOT  N=$N  duration=${DUR}s  kill_at=${KILL_AT}s"

# ---- pre-create this root's topics before any JVM starts (setup, MESH_SCALE_prereg_amendment_2.json) ----
# Same partition count as the broker default (3, RF 1); the kafka-topics JVM gets an explicit -Xmx256m.
KT="$AMCP/tools/kafka/bin/kafka-topics.sh"
export JAVA_HOME="$AMCP/tools/jdk21"
export KAFKA_HEAP_OPTS="-Xmx256m -Xms64m"
TOPICS=()
for s in $(seq -w 0 31); do TOPICS+=("$ROOT.bee.inbox.s$s"); done
TOPICS+=("$ROOT.task.submit" "$ROOT.task.done" "$ROOT.bee.heartbeat" "$ROOT.bee.lifecycle"
         "$ROOT.bee.mesh.help" "$ROOT.bee.mesh.offer" "$ROOT.host.announce" "$ROOT.control.load"
         "$ROOT.user.reply.lg1")
T_PRE=$(date +%s%3N)
for t in "${TOPICS[@]}"; do
  if ! "$KT" --bootstrap-server "$BOOT" --create --if-not-exists --topic "$t" \
        --partitions 3 --replication-factor 1 >> "$OUT/topics_create.log" 2>&1; then
    echo "abort: could not create topic $t (see $OUT/topics_create.log)" >&2
    exit 4
  fi
done
echo "pre-created ${#TOPICS[@]} topics in $(( $(date +%s%3N) - T_PRE )) ms"

alive() {
  local st
  st=$(awk '/^State:/ {print $2}' "/proc/$1/status" 2>/dev/null)
  [ -n "$st" ] && [ "$st" != "Z" ]
}

vmhwm_kb() {
  awk '/^VmHWM:/ {print $2}' "/proc/$1/status" 2>/dev/null
}

# start_proc NAME XMX_MB ARGS...   sets LAST_PID, records the process
start_proc() {
  local name=$1 xmx=$2
  shift 2
  local t
  t=$(date +%s%3N)
  "$JAVA" "-Xmx${xmx}m" -XX:+ExitOnOutOfMemoryError -jar "$OUT/mesh.jar" "$@" \
    > "$OUT/logs/$name.log" 2>&1 &
  LAST_PID=$!
  PIDS+=("$LAST_PID")
  NAMES+=("$name")
  echo "$name $LAST_PID $xmx $t" >> "$OUT/processes.txt"
}

# stop_all: SIGTERM to every process this script started and still alive, wait up to 60 s, then
# SIGKILL. Records each exit code. Idempotent.
stop_all() {
  if [ "$STOPPED" -eq 1 ]; then return; fi
  STOPPED=1
  local i pid
  for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    if [ "$pid" = "0" ]; then continue; fi
    if alive "$pid"; then kill -TERM "$pid" 2>/dev/null; fi
  done
  for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    if [ "$pid" = "0" ]; then continue; fi   # already reaped and recorded
    local waited=0
    while alive "$pid" && [ "$waited" -lt 60 ]; do sleep 1; waited=$((waited + 1)); done
    if alive "$pid"; then kill -KILL "$pid" 2>/dev/null; echo "${NAMES[$i]} needed SIGKILL" >> "$OUT/stop_notes.txt"; fi
    wait "$pid" 2>/dev/null
    echo "${NAMES[$i]} $pid exit=$?" >> "$RC_FILE"
  done
}

on_exit() {
  if [ "$STOPPED" -eq 0 ]; then
    echo "run interrupted or failed before the normal end; stopping this run's processes" >&2
    stop_all
  fi
}
trap on_exit EXIT

# ---- start: recorder, hosts, bees ------------------------------------------------------------------
start_proc record 256 record --root "$ROOT" --bootstrap "$BOOT" --out "$OUT/announce.jsonl"
REC_PID=$LAST_PID
sleep 2

start_proc host-1 512 host --root "$ROOT" --bootstrap "$BOOT" --host-id h1 \
  --attempt-timeout-ms 10000 --heartbeat-ms 5000
start_proc host-2 512 host --root "$ROOT" --bootstrap "$BOOT" --host-id h2 \
  --attempt-timeout-ms 10000 --heartbeat-ms 5000
sleep 5

PER_JVM=$((N / BEE_JVMS))
for k in $(seq 1 "$BEE_JVMS"); do
  start_proc "bees-$k" 768 bees --root "$ROOT" --bootstrap "$BOOT" --count "$PER_JVM" --tiers coder \
    --max-concurrent 4 --sim-latency-ms 50 --sim-jitter-ms 10 --heartbeat-ms 5000 \
    --kill-after-s "$KILL_AT" --kill-fraction "$KILL_FRACTION" --kill-seed $((20261027 + k)) \
    --event-log "$OUT/events/bees-$k.events.log"
done
LAST_BEE_LAUNCH=$(awk '$1 ~ /^bees-/ {print $4}' "$OUT/processes.txt" | sort -n | tail -1)
echo "all bee JVMs launched; last launch epoch ms $LAST_BEE_LAUNCH"

# ---- wait for registration (both hosts count N live bees) ------------------------------------------
"$JAVA" -Xmx256m -XX:+ExitOnOutOfMemoryError -jar "$OUT/mesh.jar" status --root "$ROOT" --bootstrap "$BOOT" \
  --wait-bees "$N" --hosts "$HOSTS" --timeout 240 > "$OUT/logs/status.log" 2>&1
STATUS_RC=$?
STATUS_END=$(date +%s%3N)
echo "status --wait-bees $N --hosts $HOSTS exit=$STATUS_RC at $STATUS_END" | tee "$OUT/registration.txt"

# ---- load ------------------------------------------------------------------------------------------
start_proc loadgen 512 loadgen --root "$ROOT" --bootstrap "$BOOT" --rate "$RATE" --duration "$DUR" \
  --tier coder --out "$OUT/loadgen.jsonl" --gateway-id lg1
LG_PID=$LAST_PID
LG_PEAK=0
while alive "$LG_PID"; do
  v=$(vmhwm_kb "$LG_PID")
  if [ -n "$v" ] && [ "$v" -gt "$LG_PEAK" ]; then LG_PEAK=$v; fi
  sleep 2
done
wait "$LG_PID"
LG_RC=$?
echo "loadgen exit=$LG_RC sampled_peak_vmhwm_kb=$LG_PEAK" | tee -a "$OUT/registration.txt"
echo "loadgen $LG_PID exit=$LG_RC" >> "$RC_FILE"
for i in "${!PIDS[@]}"; do
  if [ "${PIDS[$i]}" = "$LG_PID" ]; then PIDS[$i]=0; fi
done

# ---- measure, then stop -----------------------------------------------------------------------------
: > "$OUT/peak_rss.txt"
for i in "${!PIDS[@]}"; do
  pid=${PIDS[$i]}
  if alive "$pid"; then
    echo "${NAMES[$i]} $pid vmhwm_kb=$(vmhwm_kb "$pid") alive_at_measure=1" >> "$OUT/peak_rss.txt"
  else
    echo "${NAMES[$i]} $pid vmhwm_kb=NA alive_at_measure=0" >> "$OUT/peak_rss.txt"
  fi
done
echo "loadgen $LG_PID vmhwm_kb=$LG_PEAK sampled=1 (last sample before exit)" >> "$OUT/peak_rss.txt"
KAFKA_PID=$(pgrep -f 'kafka\.Kafka ' | head -1)
KAFKA_LINE=""
KAFKA_XMX_MB=""
KAFKA_HWM_KB=""
if [ -n "$KAFKA_PID" ]; then
  KAFKA_LINE=$(tr '\0' ' ' < "/proc/$KAFKA_PID/cmdline" 2>/dev/null)
  KAFKA_XMX_MB=$(echo "$KAFKA_LINE" | grep -o -- '-Xmx[0-9]*[mMgG]' | head -1 | sed 's/-Xmx//')
  KAFKA_HWM_KB=$(vmhwm_kb "$KAFKA_PID")
fi
echo "kafka pid=${KAFKA_PID:-none} xmx=${KAFKA_XMX_MB:-unknown} vmhwm_kb_lifetime=${KAFKA_HWM_KB:-NA}" >> "$OUT/peak_rss.txt"

stop_all

# ---- collect ----------------------------------------------------------------------------------------
: > "$OUT/oom_grep.txt"
for f in "$OUT"/logs/*.log "$OUT"/events/*.log; do
  [ -f "$f" ] || continue
  c=$(grep -c "OutOfMemoryError" "$f" 2>/dev/null)
  echo "$(basename "$f") OutOfMemoryError=${c:-0}" >> "$OUT/oom_grep.txt"
done

cat > "$OUT/run.json" <<EOF
{
  "experiment_id": "MESH-SCALE-1",
  "run_id": "$TS-N$N",
  "root": "$ROOT",
  "n_bees": $N,
  "duration_s": $DUR,
  "rate_tps": $RATE,
  "tier": "coder",
  "bee_jvms": $BEE_JVMS,
  "bee_xmx_mb": 768,
  "hosts": $HOSTS,
  "host_xmx_mb": 512,
  "recorder_xmx_mb": 256,
  "loadgen_xmx_mb": 512,
  "kafka_xmx_mb": "${KAFKA_XMX_MB:-unknown}",
  "kill_at_s": $KILL_AT,
  "kill_fraction": $KILL_FRACTION,
  "kill_seed_base": 20261027,
  "setup_amendment": 2,
  "topics_precreated": ${#TOPICS[@]},
  "sim_latency_ms": 50,
  "sim_jitter_ms": 10,
  "max_concurrent_per_bee": 4,
  "heartbeat_ms": 5000,
  "attempt_timeout_ms": 10000,
  "last_bee_launch_ms": $LAST_BEE_LAUNCH,
  "status_exit_code": $STATUS_RC,
  "status_end_ms": $STATUS_END,
  "loadgen_exit_code": $LG_RC,
  "jar_sha256": "$JAR_SHA",
  "prereg_sha256": "${PREREG_SHA:-unknown}",
  "kafka_cmdline": "$(echo "$KAFKA_LINE" | sed 's/"/\\"/g')"
}
EOF
echo "done: $OUT"
