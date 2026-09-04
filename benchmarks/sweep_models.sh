#!/bin/zsh
# Sweep zeval across installed models, ONE AT A TIME, restarting the server
# between models.
#
# Three ways this has silently produced garbage, each fixed below:
#
# 1. `timeout 3600` killed every model mid-run at exactly the hour. Removed --
#    `ev` already applies per-task timeouts, which is the right level.
#
# 2. A run was marked done on `SWEEP-DONE`, then on `exit=0`. Neither is
#    evidence a run HAPPENED: the overnight sweep had eight models exit 0 with
#    every task scoring 0, in about a second each, and the resume logic would
#    have skipped all of them permanently. Done now requires a scored task with
#    a non-zero result.
#
# 3. Memory exhaustion. Osaurus keeps a model resident, so loading them back to
#    back climbed 62% -> 76% and the machine began swapping (1M pageouts). The
#    prefill probe fell from 1,190 chars/sec to 156, every task hit the 900s
#    timeout, and then models stopped loading at all. Restarting the server
#    between models returns memory to the OS.

set -u
OUT="${SWEEP_OUT:-$HOME/.config/ztools/sweep}"
mkdir -p "$OUT"
EV=/Users/ztomer/Projects/ztools/.venv/bin/ev
PY=/Users/ztomer/Projects/ztools/.venv/bin/python3
cd "${TMPDIR:-/tmp}" || exit 1

# Ordering, revised now that the control has served its purpose. Running the
# known-good models first proved the harness was sound -- gemma, foundation and
# qwen-27b all completed with real spreads while the exotic families produced
# nothing -- so the instrument is no longer in question and those five are done.
#
# What remains runs unknowns-first, with ONE exception at the end: qwen3.6-35b
# has already shown it can consume hours without producing a score, and now that
# timeouts are derived from measured rates its per-task ceiling is large. Last
# place means it cannot starve the four models we still know nothing about.
MODELS=(
  muse-glimmer-30b-jang_6m
  bonsai-27b-ternary-jang
  ornith-1.0-9b-mxfp8
  ornith-1.0-35b-jang_4m
  gemma-4-12b-it-mxfp8
  qwen3.6-27b-mxfp8-mtp
  gemma-4-e4b-it-8bit
  gemma-4-e2b-it-8bit
  foundation
  qwen3.6-35b-a3b-mxfp8-mtp
)

# A model counts as done only with evidence it actually ran: a clean exit AND at
# least one task scoring above zero. An all-zero run is a failed run.
ran_for_real() {
  local log="$1"
  [[ -s "$log" ]] || return 1
  grep -q "SWEEP-DONE exit=0" "$log" || return 1
  grep -qE "[·✗] [a-z_]+: ([1-9][0-9]?|100)%" "$log"
}

restart_server() {
  "$PY" - <<'PYEOF'
import sys, time
sys.path.insert(0, "/Users/ztomer/Projects/ztools/references")
from lib.osaurus_server import restart_server, is_server_running
restart_server()
for _ in range(60):
    if is_server_running():
        print("  server back up")
        break
    time.sleep(2)
else:
    print("  WARNING: server did not come back")
PYEOF
}

for m in $MODELS; do
  log="$OUT/$m.log"
  if ran_for_real "$log"; then
    echo "skip $m (already has real results)"
    continue
  fi

  echo "=== $m : restarting server to reclaim memory ==="
  restart_server
  free=$(memory_pressure 2>/dev/null | grep -o "free percentage: [0-9]*" | grep -o "[0-9]*")
  echo "=== $m : started $(date +%H:%M:%S), memory free ${free:-?}% ==="

  "$EV" --model "$m" > "$log" 2>&1
  echo "SWEEP-DONE exit=$? $(date +%H:%M:%S)" >> "$log"

  if ran_for_real "$log"; then
    echo "=== $m : finished $(date +%H:%M:%S) OK ==="
  else
    echo "=== $m : finished $(date +%H:%M:%S) BUT PRODUCED NO REAL SCORES ==="
  fi
done
echo "SWEEP-COMPLETE"
