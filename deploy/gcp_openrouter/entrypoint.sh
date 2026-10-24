#!/bin/sh
# Container entrypoint: restore the SQLite ledger from GCS (if a replica exists), then run the gateway under
# Litestream, which streams every committed transaction back to the bucket (about 1 s behind) and flushes on SIGTERM.
set -eu

DB="${HIVEMIND_ROUTER_DB:-/data/router.db}"
REPLICA="${HIVEMIND_ROUTER_REPLICA_URL:-}"
PORT="${PORT:-8080}"

if [ -z "$REPLICA" ]; then
  echo "HIVEMIND_ROUTER_REPLICA_URL is not set: running with an EPHEMERAL ledger" >&2
  exec uvicorn server:app --host 0.0.0.0 --port "$PORT" --workers 1
fi

# -if-db-not-exists: never overwrite a live file.  -if-replica-exists: a first start has nothing to restore.
litestream restore -if-db-not-exists -if-replica-exists -o "$DB" "$REPLICA"
exec litestream replicate -exec "uvicorn server:app --host 0.0.0.0 --port $PORT --workers 1" "$DB" "$REPLICA"
