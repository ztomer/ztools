#!/bin/bash
# Run weekend eval on all installed models, one at a time.
# Usage: eval/run_weekend_eval.sh [--skip MODEL ...]

set -euo pipefail

# Resolved from this script's own location, not from $PWD: it is run from the repo
# root by hand and from elsewhere by anything that wraps it.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DIR="${DIR:-/tmp/weekend_eval_results}"
mkdir -p "$DIR"

CLEANUP_DONE=false

# This script no longer owns a server process, so it no longer kills one on the way
# out. It used to track an OSAURUS_PID it had started itself; the server is now
# established through tools/osaurus_one.sh, which is machine-wide by design. Killing
# it here would tear down a server that outlives this run and that another session
# may be using -- the opposite of what interrupting one eval should do.
cleanup() {
    if [ "$CLEANUP_DONE" = true ]; then
        return
    fi
    CLEANUP_DONE=true
    echo ""
    echo "=== Interrupted — cleaning up ==="
    echo "Cleanup complete. Results in $DIR"
    exit 130
}

trap cleanup SIGINT SIGTERM

# Get all models from the running server
MODELS=$(osaurus list 2>/dev/null || curl -s http://localhost:1337/models | python3 -c "
import sys, json
for m in json.load(sys.stdin)['data']:
    print(m['id'])
")

if [ -z "$MODELS" ]; then
  echo "No models found. Is Osaurus running?"
  exit 1
fi

# Parse --skip arguments
SKIP=()
for arg in "$@"; do
  if [ "$arg" != "--skip" ]; then
    SKIP+=("$arg")
  fi
done

for m in $MODELS; do
  # Check if model was skipped
  skip_model=0
  for s in "${SKIP[@]}"; do
    if [ "$m" = "$s" ]; then
      echo "=== Skipping: $m ==="
      skip_model=1
      break
    fi
  done
  [ "$skip_model" -eq 1 ] && continue

  echo "=== $m: weekend_transient ==="
  python3 -m eval --model "$m" --task weekend_transient --quick 2>&1 | tee "$DIR/${m}_transient.txt"

  # Check if server died (no response or connection error)
  #
  # Recovered through tools/osaurus_one.sh, never by starting one here. The
  # predecessor ran `osaurus serve &` on any failed curl, which was wrong twice
  # over. It never checked whether a server was ALREADY running, so one
  # transient curl failure left TWO of them competing for the same GPU and RAM --
  # contention the sample guard cannot see, since it reads swap and compressor and
  # not the GPU, so the ruined timing is filed as CLEAN. And it bypassed the GPU
  # lock, so it could start a server on another session's measurement. osaurus_one.sh
  # is idempotent, enforces exactly one, and takes the lock -- which also means it
  # REFUSES rather than restarting under a peer that is mid-eval.
  if ! curl -sf http://localhost:1337/models > /dev/null 2>&1; then
    echo "⚠ Server unreachable after $m — restarting..."
    if ! "$ROOT/tools/osaurus_one.sh" --restart >/dev/null; then
      echo "⚠ Server failed to restart — check Osaurus, or whether a peer holds the GPU"
      exit 1
    fi
  fi

  echo "=== $m: weekend_fixed ==="
  python3 -m eval --model "$m" --task weekend_fixed --quick 2>&1 | tee "$DIR/${m}_fixed.txt"

  if ! curl -sf http://localhost:1337/models > /dev/null 2>&1; then
    echo "⚠ Server unreachable after $m — restarting..."
    if ! "$ROOT/tools/osaurus_one.sh" --restart >/dev/null; then
      echo "⚠ Server failed to restart — check Osaurus, or whether a peer holds the GPU"
      exit 1
    fi
  fi
done

echo "=== ALL DONE ==="
echo "Results in $DIR"
