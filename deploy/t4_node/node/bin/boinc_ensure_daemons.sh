#!/bin/bash
# Keep the BOINC server daemons alive.
#   --wait-db SECONDS   block until the project database answers on its socket (exit 1 on timeout)
#   --verify            after `bin/start`: check that every required daemon runs; if not, wait for the database and start again (up to HM_MAX_TRIES)
#   --watchdog          periodic check: if the daemons unit is active but a daemon has died, start it again
# Why: after a Spot preemption the database needs seconds of crash recovery. `bin/start` ran before its socket existed, so the feeder and the
# transitioner logged "boinc_db.open failed" and exited, and nothing started them again: the scheduler answered "feeder not running" and no work
# was served (2026-10-26 21:46Z and 2026-10-27 04:16Z). The validators/assimilators survive a failed first connect; those two do not.
set -u
SOCK="${HM_DB_SOCK:-/run/mysqld/mysqld.sock}"
ADMIN="${HM_MARIADB_ADMIN:-/usr/bin/mariadb-admin}"
ENVSH="${HM_ENV_SCRIPT:-/opt/hivemind/t4/bin/boinc_project_env.sh}"
STEP="${HM_STEP_S:-5}"            # whole seconds between database polls
DB_WAIT="${HM_DB_WAIT_S:-120}"    # longest wait for the database before giving up on one attempt
TRIES="${HM_MAX_TRIES:-5}"
SETTLE="${HM_SETTLE_S:-20}"       # daemons detach asynchronously: give them time to appear before judging
UNIT_ACTIVE="${HM_UNIT_ACTIVE_CMD:-systemctl is-active --quiet hivemind-boinc-daemons.service}"
REQUIRED=('^feeder ' '^transitioner ' '^file_deleter '
          '^script_validator .*--app hivemind_qlora_moe( |$)' '^script_assimilator .*--app hivemind_qlora_moe( |$)'
          '^script_validator .*--app hivemind_moe_infer( |$)' '^script_assimilator .*--app hivemind_moe_infer( |$)')

log() { echo "boinc_ensure_daemons: $*"; }
# `mariadb-admin ping` exits 0 whenever the server answers (even "access denied"), 1 when it cannot connect
db_up() { [ -S "$SOCK" ] && "$ADMIN" --no-defaults --socket="$SOCK" ping >/dev/null 2>&1; }
wait_db() {
  local limit=$1 waited=0
  while ! db_up; do
    [ "$waited" -ge "$limit" ] && return 1
    sleep "$STEP"; waited=$((waited + STEP))
  done
}
missing() { local p; for p in "${REQUIRED[@]}"; do pgrep -f -- "$p" >/dev/null || echo "$p"; done; }

ensure() {
  local tries=$1 attempt miss
  for ((attempt = 1; attempt <= tries; attempt++)); do
    miss="$(missing)"
    if [ -z "$miss" ]; then log "all required daemons are running"; return 0; fi
    log "attempt $attempt/$tries: not running: $(echo "$miss" | tr '\n' ' ')"
    if ! wait_db "$DB_WAIT"; then log "database not answering on $SOCK: not starting daemons into a dead database"; return 2; fi
    "$ENVSH" bin/start >/dev/null 2>&1 || log "bin/start returned $?"
    sleep "$SETTLE"
  done
  miss="$(missing)"
  if [ -z "$miss" ]; then log "all required daemons are running"; return 0; fi
  log "still not running after $tries attempts: $(echo "$miss" | tr '\n' ' ')"
  return 1
}

case "${1:-}" in
  --wait-db)
    wait_db "${2:-120}" && { log "database is answering"; exit 0; }
    log "database did not answer within ${2:-120}s"; exit 1 ;;
  --verify)
    sleep "$SETTLE"
    ensure "$TRIES"; exit $? ;;
  --watchdog)
    if ! $UNIT_ACTIVE; then log "daemons unit is not active (stopped on purpose, or still starting): nothing to do"; exit 0; fi
    ensure 2; rc=$?
    [ "$rc" -eq 2 ] && exit 0   # database down: its own unit restarts it; a warning is logged above
    exit "$rc" ;;
  *) echo "usage: $0 --wait-db SECONDS | --verify | --watchdog" >&2; exit 64 ;;
esac
