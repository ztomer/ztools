#!/bin/bash
# Run weekend eval on all installed models, one at a time.
# Usage: eval/run_weekend_eval.sh [--skip MODEL ...]

set -euo pipefail

DIR="${DIR:-/tmp/weekend_eval_results}"
mkdir -p "$DIR"

# Track background processes for cleanup
OSAURUS_PID=""
CLEANUP_DONE=false

cleanup() {
    if [ "$CLEANUP_DONE" = true ]; then
        return
    fi
    CLEANUP_DONE=true
    echo ""
    echo "=== Interrupted — cleaning up ==="
    if [ -n "$OSAURUS_PID" ] && kill -0 "$OSAURUS_PID" 2>/dev/null; then
        echo "Stopping Osaurus (PID $OSAURUS_PID)..."
        kill "$OSAURUS_PID" 2>/dev/null || true
        sleep 1
        kill -9 "$OSAURUS_PID" 2>/dev/null || true
    fi
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
  if ! curl -sf http://localhost:1337/models > /dev/null 2>&1; then
    echo "⚠ Server unreachable after $m — restarting..."
    osaurus serve &>/dev/null &
    OSAURUS_PID=$!
    sleep 15
    if ! curl -sf http://localhost:1337/models > /dev/null 2>&1; then
      echo "⚠ Server failed to restart — check Osaurus"
      kill "$OSAURUS_PID" 2>/dev/null || true
      exit 1
    fi
    OSAURUS_PID=""
  fi

  echo "=== $m: weekend_fixed ==="
  python3 -m eval --model "$m" --task weekend_fixed --quick 2>&1 | tee "$DIR/${m}_fixed.txt"

  if ! curl -sf http://localhost:1337/models > /dev/null 2>&1; then
    echo "⚠ Server unreachable after $m — restarting..."
    osaurus serve &>/dev/null &
    OSAURUS_PID=$!
    sleep 15
    if ! curl -sf http://localhost:1337/models > /dev/null 2>&1; then
      echo "⚠ Server failed to restart — check Osaurus"
      kill "$OSAURUS_PID" 2>/dev/null || true
      exit 1
    fi
    OSAURUS_PID=""
  fi
done

echo "=== ALL DONE ==="
echo "Results in $DIR"
