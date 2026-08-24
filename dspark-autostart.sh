#!/usr/bin/env bash
# Autostart wrapper for the DeepSeek V4 Flash DSpark cluster.
# Waits for docker + the worker node to be reachable, then (re)starts the stack.
# Designed to run from a systemd unit (dspark.service) at boot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"

WORKER_HOST="$(sed -n 's/^WORKER_HOST=//p' "$ENV_FILE" 2>/dev/null || true)"
WORKER_HOST="${WORKER_HOST:-andrejs@spark2}"

log() { echo "[dspark-autostart] $*"; }

# --- 1. Wait for the local docker daemon -------------------------------
for _ in $(seq 1 60); do
  docker info >/dev/null 2>&1 && break
  sleep 5
done
docker info >/dev/null 2>&1 || { log "FATAL: docker daemon not ready"; exit 1; }
log "docker daemon ready"

# --- 2. Wait for the worker node to be reachable over SSH --------------
log "waiting for worker $WORKER_HOST ..."
for _ in $(seq 1 120); do
  if ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$WORKER_HOST" "true" >/dev/null 2>&1; then
    log "worker $WORKER_HOST reachable"
    break
  fi
  sleep 10
done
ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" "true" >/dev/null 2>&1 \
  || { log "FATAL: worker $WORKER_HOST never became reachable"; exit 1; }

# --- 3. Bring the stack up (stop first to be idempotent on retries) -----
log "stopping any previous stack (if present)..."
"$SCRIPT_DIR/stop-deepseek-v4-flash-dspark.sh" >/dev/null 2>&1 || true

log "starting DSpark stack..."
"$SCRIPT_DIR/start-deepseek-v4-flash-dspark.sh"
log "DSpark stack is up"
