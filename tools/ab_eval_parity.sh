#!/usr/bin/env bash
# A/B eval parity: run the SAME taxes task snapshot through the Python eval
# and the Rust eval loop, on the same model and the same server, and compare
# the coarse verdict buckets.
#
# The two stacks score differently -- Python uses graded validators (0-100 per
# rubric), the Rust loop scores a fraction of boolean checks -- so score NUMBERS
# are not expected to match. What must match is the coarse classification:
#   ok/partial  vs  ok/partial   -> agree
#   fail        vs  fail          -> agree
# Anything else is a parity break and fails this script.
#
# Usage: tools/ab_eval_parity.sh [model] [task]
#   model defaults to gemma-4-e2b-it-8bit (2B: cheapest honest reading);
#   task defaults to taxes_qa.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-gemma-4-e2b-it-8bit}"
TASK="${2:-taxes_qa}"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

echo "============================================================"
echo "  EVAL A/B PARITY: $TASK  x  $MODEL"
echo "============================================================"

# One healthy server or none of this means anything.
"$ROOT/tools/osaurus_one.sh" --check || {
    echo "✗ osaurus_one.sh --check failed; fix the server first"
    exit 1
}

# GPU + single-server lock for the whole comparison (mkdir-atomic per
# tools/gpu_lock.sh; macOS ships no flock).
source "$ROOT/tools/gpu_lock.sh"

acquire_gpu_lock() {
    gpu_lock_acquire "ab_eval_parity" || {
        echo "✗ GPU lock held by: $(gpu_lock_holder 2>/dev/null || echo unknown)"
        exit 3
    }
}

OUT_DIR="$(mktemp -d /tmp/ab_eval_parity.XXXXXX)"
trap 'rm -rf "$OUT_DIR"' EXIT

# --- Python side ------------------------------------------------------------
acquire_gpu_lock
(
    cd "$ROOT"
    PYTHONPATH="$ROOT/references" \
    MLX_MODELS_DIR=/tmp/nonexistent \
    OLLAMA_BASE_URL=http://127.0.0.1:1 \
    EVAL_SIGNALS_DIR="$OUT_DIR" \
    "$PY" - <<PYEOF || exit 1
import json, sys
sys.path.insert(0, 'references')
from eval.run import run_eval
from eval.tasks_core import TASKS

task = TASKS['$TASK']
results = run_eval('$MODEL', tasks={'$TASK': task})
r = results[0]
# Write the file directly: run_eval's console output shares stdout and would
# corrupt a redirected capture.
with open('$OUT_DIR/py.json', 'w') as f:
    json.dump({
        'score': r['quality_score'],
        'status': r['status'],
        'time': r.get('time'),
        'error': r.get('error'),
    }, f)
PYEOF
)
if [ $? -ne 0 ]; then
    echo "✗ python side failed"; cat "$OUT_DIR/py.json" 2>/dev/null
    gpu_lock_release 2>/dev/null || true
    exit 1
fi

# --- Rust side --------------------------------------------------------------
# The machine-wide shared CARGO_TARGET_DIR keeps artifacts OUT of rust/target;
# ask cargo where they actually live.
TARGET_DIR="$(cd "$ROOT/rust" && cargo metadata --format-version 1 2>/dev/null | "$PY" -c "import json,sys; print(json.load(sys.stdin)['target_directory'])")"
RUST_BIN="$TARGET_DIR/debug/ztools"
[ -x "$RUST_BIN" ] || (cd "$ROOT/rust" && cargo build --quiet)

env EVAL_SIGNALS_DIR="$OUT_DIR" "$RUST_BIN" \
    model-eval --suite full --task "$TASK" --model "$MODEL" --json-output \
    --tasks-dir "$ROOT/eval_tasks/data" \
    > "$OUT_DIR/rust.json"
if [ $? -ne 0 ]; then
    echo "✗ rust side failed"; cat "$OUT_DIR/rust.json" 2>/dev/null
    gpu_lock_release 2>/dev/null || true
    exit 1
fi
gpu_lock_release 2>/dev/null || true

# --- Compare -----------------------------------------------------------------
echo ""
echo "--- results ---"
echo "python: $(cat "$OUT_DIR/py.json")"
echo "rust:   $(cat "$OUT_DIR/rust.json")"
echo ""

py_status=$("$PY" -c "import json; print(json.load(open('$OUT_DIR/py.json'))['status'])")
rust_status=$("$PY" -c "
import json
rows = json.load(open('$OUT_DIR/rust.json'))
print(rows[0]['status'])
")

bucket() {
    case "$1" in
        ok|partial) echo "not-fail" ;;
        *) echo "fail" ;;
    esac
}

py_bucket=$(bucket "$py_status")
rust_bucket=$(bucket "$rust_status")

if [ "$py_bucket" = "$rust_bucket" ]; then
    echo "→ verdict buckets AGREE: python=$py_status ($py_bucket), rust=$rust_status ($rust_bucket)"
    exit 0
else
    echo "✗ PARITY BREAK: python=$py_status ($py_bucket) but rust=$rust_status ($rust_bucket)"
    exit 1
fi
