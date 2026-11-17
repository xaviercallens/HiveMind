#!/usr/bin/env bash
# HiveMind bee mesh quickstart. One command runs the whole mesh on this machine:
# one Kafka broker (KRaft, loopback), one host, three simulated bees and one HTTP gateway.
#
#   mesh/quickstart.sh          start (default): install tools if missing, build, run, send one chat request
#   mesh/quickstart.sh stop     stop exactly the processes this script recorded, and the Kafka broker it started
#
# Everything lives under $HIVEMIND_MESH_HOME (default $HOME/.hivemind-mesh). Nothing is bound outside 127.0.0.1.
# Read mesh/README.md for what each part does and docs/community/BEE_MESH_QUICKSTART.md for the newcomer tour.
#
# Environment overrides (all optional):
#   HIVEMIND_MESH_HOME        where tools, the AMCP clone, Maven repo, Kafka data, logs and pids go
#   JAVA_HOME_21              use an existing JDK 21 instead of installing Temurin 21
#   MVN_BIN                   use an existing Maven (3.9.x) instead of installing 3.9.9
#   KAFKA_HOME                use an existing Kafka 3.8.x instead of installing 3.8.0
#   AMCP_DIR                  use an existing AMCP clone (it is built as it is, not updated)
#   AMCP_REPO, AMCP_REF       clone this AMCP repository at this ref (default: upstream main)
#   MVN_REPO                  Maven local repository (default $HIVEMIND_MESH_HOME/m2)
#   QUICKSTART_KAFKA_PORT     Kafka client port on 127.0.0.1 (default 9092)
#   QUICKSTART_KAFKA_CONTROLLER_PORT  Kafka KRaft controller port (default: client port + 1)
#   QUICKSTART_GATEWAY_PORT   HTTP gateway port on 127.0.0.1 (default 8097)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESH_HOME="${HIVEMIND_MESH_HOME:-$HOME/.hivemind-mesh}"
TOOLS="$MESH_HOME/tools"
RUN="$MESH_HOME/run"
LOGS="$MESH_HOME/logs"
PIDFILE="$RUN/pids"
ROOTFILE="$RUN/root"
KAFKA_DIR="$MESH_HOME/kafka"

KAFKA_PORT="${QUICKSTART_KAFKA_PORT:-9092}"
KAFKA_CTRL_PORT="${QUICKSTART_KAFKA_CONTROLLER_PORT:-$((KAFKA_PORT + 1))}"
GATEWAY_PORT="${QUICKSTART_GATEWAY_PORT:-8097}"
AMCP_REPO="${AMCP_REPO:-https://github.com/agentmeshcommunicationprotocol/amcpcore.github.io}"
AMCP_REF="${AMCP_REF:-main}"
MVN_REPO="${MVN_REPO:-$MESH_HOME/m2}"
MESH_JAR="$HERE/target/hivemind-bee-mesh-0.1.0-all.jar"

say() { printf '==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

# ---------------------------------------------------------------- tools (JDK 21, Maven 3.9.9, Kafka 3.8.0)

install_jdk() {
  local dl="$TOOLS/dl" dest="$TOOLS/jdk21"
  mkdir -p "$dl" "$dest"
  say "installing JDK 21 (Temurin) into $dest"
  curl -fsSL -o "$dl/jdk21.tar.gz" "https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse"
  local want got
  want=$(curl -fsSL "https://api.adoptium.net/v3/assets/latest/21/hotspot?architecture=x64&image_type=jdk&os=linux&vendor=eclipse" \
    | python3 -c "import sys,json;print(json.load(sys.stdin)[0]['binary']['package']['checksum'])")
  got=$(sha256sum "$dl/jdk21.tar.gz" | cut -d' ' -f1)
  [ "$want" = "$got" ] || die "JDK checksum mismatch (want $want, got $got)"
  tar -xzf "$dl/jdk21.tar.gz" -C "$dest" --strip-components=1
}

install_maven() {
  local dl="$TOOLS/dl" dest="$TOOLS/maven"
  mkdir -p "$dl" "$dest"
  say "installing Maven 3.9.9 into $dest"
  local base="https://archive.apache.org/dist/maven/maven-3/3.9.9/binaries/apache-maven-3.9.9-bin.tar.gz"
  curl -fsSL -o "$dl/maven.tgz" "$base"
  local want got
  want=$(curl -fsSL "$base.sha512" | cut -d' ' -f1)
  got=$(sha512sum "$dl/maven.tgz" | cut -d' ' -f1)
  [ "$want" = "$got" ] || die "Maven checksum mismatch"
  tar -xzf "$dl/maven.tgz" -C "$dest" --strip-components=1
}

install_kafka() {
  local dl="$TOOLS/dl" dest="$TOOLS/kafka"
  mkdir -p "$dl" "$dest"
  say "installing Kafka 3.8.0 into $dest"
  local base="https://archive.apache.org/dist/kafka/3.8.0/kafka_2.13-3.8.0.tgz"
  curl -fsSL -o "$dl/kafka.tgz" "$base"
  # Apache's .sha512 is "file: HEX" over several lines; keep the hex part only
  local want got
  want=$(curl -fsSL "$base.sha512" | tr -d ' \n' | sed 's/.*://' | tr 'A-F' 'a-f')
  got=$(sha512sum "$dl/kafka.tgz" | cut -d' ' -f1)
  [ "$want" = "$got" ] || die "Kafka checksum mismatch"
  tar -xzf "$dl/kafka.tgz" -C "$dest" --strip-components=1
}

ensure_tools() {
  need curl; need tar; need sha256sum; need sha512sum; need python3

  if [ -n "${JAVA_HOME_21:-}" ]; then
    JDK="$JAVA_HOME_21"
  else
    JDK="$TOOLS/jdk21"
    [ -x "$JDK/bin/javac" ] || install_jdk
  fi
  [ -x "$JDK/bin/java" ] || die "no java in $JDK (set JAVA_HOME_21 to a JDK 21)"
  "$JDK/bin/java" -version 2>&1 | grep -q 'version "21' || die "$JDK is not Java 21"

  if [ -n "${MVN_BIN:-}" ]; then
    MVN="$MVN_BIN"
  else
    MVN="$TOOLS/maven/bin/mvn"
    [ -x "$MVN" ] || install_maven
  fi
  [ -x "$MVN" ] || die "Maven not found at $MVN (set MVN_BIN)"

  if [ -n "${KAFKA_HOME:-}" ]; then
    KAFKA="$KAFKA_HOME"
  else
    KAFKA="$TOOLS/kafka"
    [ -x "$KAFKA/bin/kafka-server-start.sh" ] || install_kafka
  fi
  [ -x "$KAFKA/bin/kafka-server-start.sh" ] || die "Kafka not found at $KAFKA (set KAFKA_HOME)"

  say "JDK: $JDK | Maven: $MVN | Kafka: $KAFKA"
}

# ---------------------------------------------------------------- AMCP and the mesh build

mvn_run() { JAVA_HOME="$JDK" "$MVN" -B -Dmaven.repo.local="$MVN_REPO" "$@"; }

build_amcp_and_mesh() {
  need git
  local src
  if [ -n "${AMCP_DIR:-}" ]; then
    src="$AMCP_DIR"
    [ -f "$src/pom.xml" ] || die "AMCP_DIR=$src has no pom.xml"
    say "using the AMCP clone at $src as it is (not updated)"
  else
    src="$MESH_HOME/src/amcpcore"
    if [ ! -f "$src/pom.xml" ]; then
      say "cloning AMCP $AMCP_REPO at $AMCP_REF into $src"
      mkdir -p "$MESH_HOME/src"
      git clone --depth 1 --branch "$AMCP_REF" "$AMCP_REPO" "$src"
    fi
  fi
  cat <<EOF
NOTE: the AMCP Kafka delivery fix is not merged upstream yet. Until it is, point this script at the fix
      branch: AMCP_REPO=<your clone or fork URL> AMCP_REF=fix/kafka-broker-delivery, or AMCP_DIR=<local clone>.
      The clone used here: $src
EOF

  say "building and installing amcp-core and amcp-broker-kafka into $MVN_REPO (log: $LOGS/build-amcp.log)"
  mkdir -p "$LOGS" "$MVN_REPO"
  mvn_run -f "$src/pom.xml" -pl amcp-core,amcp-broker-kafka -am install -DskipTests > "$LOGS/build-amcp.log" 2>&1 \
    || { tail -n 40 "$LOGS/build-amcp.log" >&2; die "AMCP build failed (full log: $LOGS/build-amcp.log)"; }

  say "building the mesh (log: $LOGS/build-mesh.log)"
  mvn_run -f "$HERE/pom.xml" package > "$LOGS/build-mesh.log" 2>&1 \
    || { tail -n 40 "$LOGS/build-mesh.log" >&2; die "mesh build failed (full log: $LOGS/build-mesh.log)"; }
  [ -f "$MESH_JAR" ] || die "expected $MESH_JAR after the build"
}

# ---------------------------------------------------------------- processes

# Each recorded process is one line of $PIDFILE: "<name> <pid>". Kafka is written first, the gateway last.
record() { printf '%s %s\n' "$1" "$2" >> "$PIDFILE"; }

alive() { kill -0 "$1" 2>/dev/null; }

any_recorded_alive() {
  [ -f "$PIDFILE" ] || return 1
  local name pid
  while read -r name pid; do
    alive "$pid" && return 0
  done < "$PIDFILE"
  return 1
}

write_kafka_config() {
  mkdir -p "$KAFKA_DIR/data" "$KAFKA_DIR/logs"
  cat > "$KAFKA_DIR/server.properties" <<EOF
# Written by mesh/quickstart.sh: one KRaft node, loopback only.
process.roles=broker,controller
node.id=1
controller.quorum.voters=1@127.0.0.1:$KAFKA_CTRL_PORT
listeners=PLAINTEXT://127.0.0.1:$KAFKA_PORT,CONTROLLER://127.0.0.1:$KAFKA_CTRL_PORT
advertised.listeners=PLAINTEXT://127.0.0.1:$KAFKA_PORT
controller.listener.names=CONTROLLER
listener.security.protocol.map=PLAINTEXT:PLAINTEXT,CONTROLLER:PLAINTEXT
inter.broker.listener.name=PLAINTEXT
log.dirs=$KAFKA_DIR/data
num.partitions=1
offsets.topic.replication.factor=1
transaction.state.log.replication.factor=1
transaction.state.log.min.isr=1
group.initial.rebalance.delay.ms=0
auto.create.topics.enable=true
log.retention.hours=24
EOF
}

start_kafka() {
  write_kafka_config
  if [ ! -f "$KAFKA_DIR/data/meta.properties" ]; then
    "$KAFKA/bin/kafka-storage.sh" format -t "$("$KAFKA/bin/kafka-storage.sh" random-uuid)" \
      -c "$KAFKA_DIR/server.properties" > /dev/null
  fi
  say "starting Kafka on 127.0.0.1:$KAFKA_PORT (controller $KAFKA_CTRL_PORT, 512 MB heap)"
  JAVA_HOME="$JDK" KAFKA_HEAP_OPTS="-Xms256m -Xmx512m" LOG_DIR="$KAFKA_DIR/logs" \
    nohup "$KAFKA/bin/kafka-server-start.sh" "$KAFKA_DIR/server.properties" > "$LOGS/kafka.out" 2>&1 &
  local pid=$!
  record kafka "$pid"
  for _ in $(seq 1 90); do
    if port_open "$KAFKA_PORT" && "$KAFKA/bin/kafka-broker-api-versions.sh" --bootstrap-server "127.0.0.1:$KAFKA_PORT" >/dev/null 2>&1; then
      return 0
    fi
    alive "$pid" || { tail -n 20 "$LOGS/kafka.out" >&2; die "Kafka exited; see $LOGS/kafka.out and $KAFKA_DIR/logs/server.log"; }
    sleep 1
  done
  die "Kafka did not answer on 127.0.0.1:$KAFKA_PORT within 90 s (see $KAFKA_DIR/logs/server.log)"
}

# Starts one mesh JVM with a 256 MB heap and records its pid.
start_mesh_jvm() {
  local name="$1"; shift
  nohup "$JDK/bin/java" -Xmx256m -jar "$MESH_JAR" "$@" > "$LOGS/$name.log" 2>&1 &
  record "$name" "$!"
}

start_mesh() {
  local root="hmqs$(head -c 4 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  printf '%s\n' "$root" > "$ROOTFILE"
  local bus=(--bootstrap "127.0.0.1:$KAFKA_PORT" --root "$root")

  say "starting the host (topic root $root)"
  start_mesh_jvm host host "${bus[@]}"

  say "starting the gateway on 127.0.0.1:$GATEWAY_PORT"
  start_mesh_jvm gateway gateway --port "$GATEWAY_PORT" "${bus[@]}"

  say "starting 3 simulated bees (one JVM each, coder and general tiers)"
  local i
  for i in 1 2 3; do
    start_mesh_jvm "bee$i" bees --count 1 "${bus[@]}"
  done

  say "waiting until the host sees 3 live bees"
  "$JDK/bin/java" -Xmx256m -jar "$MESH_JAR" status --wait-bees 3 --timeout 90 "${bus[@]}" > "$LOGS/status.out" 2>&1 \
    || { tail -n 20 "$LOGS/status.out" >&2; die "fewer than 3 bees reached the host (see $LOGS/*.log)"; }
}

ask_once() {
  say "sending one chat request to http://127.0.0.1:$GATEWAY_PORT/v1/chat/completions"
  local body resp
  body='{"model":"coder","messages":[{"role":"user","content":"Hello from the bee mesh quickstart"}]}'
  resp=$(curl -sS --max-time 60 "http://127.0.0.1:$GATEWAY_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$body") \
    || die "the gateway did not answer (logs: $LOGS/gateway.log, $LOGS/host.log)"
  printf '%s\n' "$resp" > "$RUN/last-answer.json"
  echo
  echo "Answer:"
  python3 -c 'import json,sys
d = json.load(sys.stdin)
if "choices" not in d:
    print("no choices in reply:", json.dumps(d)); sys.exit(1)
print("  " + d["choices"][0]["message"]["content"])
print("  bee=%s attempts=%s e2eMs=%s" % (d["hivemind"]["beeId"], d["hivemind"]["attempts"], d["hivemind"]["e2eMs"]))' <<< "$resp" \
    || die "unexpected reply: $resp"
  echo
}

next_steps() {
  cat <<EOF
The mesh is running on 127.0.0.1 (Kafka $KAFKA_PORT, gateway $GATEWAY_PORT). Next steps:
  curl -s http://127.0.0.1:$GATEWAY_PORT/mesh/health          # live host announcements
  curl -s http://127.0.0.1:$GATEWAY_PORT/v1/chat/completions -H 'Content-Type: application/json' \\
       -d '{"model":"general","messages":[{"role":"user","content":"your question"}]}'
  Logs: $LOGS    Last answer: $RUN/last-answer.json
  Your own model: restart the bees with --backend-url http://127.0.0.1:<your port> --backend-model <name>
    (only point it at a server you run yourself; see docs/community/BEE_MESH_QUICKSTART.md)
  Stop everything: $0 stop
EOF
}

cmd_start() {
  need python3
  if any_recorded_alive; then die "the mesh from an earlier start is still running; run '$0 stop' first"; fi
  for p in "$KAFKA_PORT" "$KAFKA_CTRL_PORT" "$GATEWAY_PORT"; do
    if port_open "$p"; then die "port $p on 127.0.0.1 is already in use; pick another with QUICKSTART_KAFKA_PORT / QUICKSTART_KAFKA_CONTROLLER_PORT / QUICKSTART_GATEWAY_PORT"; fi
  done
  mkdir -p "$RUN" "$LOGS"
  : > "$PIDFILE"

  ensure_tools
  build_amcp_and_mesh
  start_kafka
  start_mesh
  ask_once
  next_steps
}

kill_recorded() {
  local name="$1" pid="$2" sig
  case "$name" in
    kafka) sig="$KAFKA_DIR/server.properties" ;;
    *) sig="hivemind-bee-mesh" ;;
  esac
  if ! alive "$pid"; then say "$name (pid $pid) already gone"; return 0; fi
  if ! tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "$sig"; then
    warn "pid $pid is no longer $name (its command line changed); not touched"
    return 0
  fi
  say "stopping $name (pid $pid)"
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 40); do alive "$pid" || return 0; sleep 0.5; done
  warn "$name (pid $pid) did not stop in 20 s; sending KILL"
  kill -KILL "$pid" 2>/dev/null || true
}

cmd_stop() {
  if [ ! -f "$PIDFILE" ]; then say "nothing recorded under $RUN; nothing to stop"; return 0; fi
  local names=() pids=() name pid
  while read -r name pid; do
    names+=("$name"); pids+=("$pid")
  done < "$PIDFILE"
  # stop in reverse start order: gateway and bees and host first, Kafka last
  local i
  for ((i = ${#names[@]} - 1; i >= 0; i--)); do
    kill_recorded "${names[$i]}" "${pids[$i]}"
  done
  rm -f "$PIDFILE"
  local left=0
  for i in "${!pids[@]}"; do
    if alive "${pids[$i]}"; then warn "${names[$i]} (pid ${pids[$i]}) is still running"; left=1; fi
  done
  if [ "$left" -eq 0 ]; then say "stopped: no process of this quickstart is left running"; fi
  if port_open "$KAFKA_PORT"; then warn "something still answers on 127.0.0.1:$KAFKA_PORT (not started by this quickstart, or Kafka still shutting down)"; fi
}

case "${1:-start}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  *) die "usage: $0 [start|stop]" ;;
esac
