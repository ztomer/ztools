#!/usr/bin/env bash
# Re-run nemotron with a raised ceiling, once the current sweep finishes.
#
# WHY. The sweep's per-model ceiling is 14400s (4h). nemotron averaged 838s/task
# under the corrected 32000-token budgets -- it reasons enormously, up to 77k
# characters on the largest prompt -- so it got through 17 of 24 tasks and was
# TRUNCATED. A model measured on 17 tasks cannot be ranked against models measured
# on 24, and the 7 it missed are the LARGEST prompts (taxes_synthesis at 10,238
# chars), i.e. exactly the ones that would drag its average.
#
# 24 x 838s = 20,112s, and the missing 7 are slower than average, so the ceiling is
# set to 36000s (10h) rather than a number that only just fits.
#
# The per-TASK timeout is not the problem and is left alone: it derives from
# measured rates and currently sits at ~7055s for every task.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
# shellcheck source=/dev/null
[ -f tui/lib.sh ] && . tui/lib.sh || { info(){ echo "→ $*"; }; ok(){ echo "✓ $*"; }
                                       warn(){ echo "⚠ $*"; }; die(){ echo "✗ $*" >&2; exit 1; }; }

MODEL="${RERUN_MODEL:-nemotron-3.5-lightning-30b-a3b-mxfp8}"
CEILING="${RERUN_TIMEOUT:-36000}"
LOG="${TMPDIR:-/tmp}/ztools-sweep/rerun-$MODEL.log"
mkdir -p "$(dirname "$LOG")"

# 1. Wait for the sweep, so two evals never share the GPU.
while pgrep -f "sweep_models.sh" >/dev/null 2>&1; do
  sleep 60
done
info "sweep finished; preparing $MODEL re-run"

# 2. Do not measure on a contended machine. This is the check that was missing when a
#    leaked daemon held 31GB and every timing taken under it was wrong -- and note that
#    osaurus_one.sh proves only that ONE osaurus runs, nothing about the other 300
#    processes. Refuse rather than record a number we would have to retract.
./tools/osaurus_one.sh >/dev/null || die "could not establish exactly one osaurus"
AVAIL="$(./.venv/bin/python -c 'import psutil; print(int(psutil.virtual_memory().available/1024**3))')"
if [ "$AVAIL" -lt 20 ]; then
  ./.venv/bin/python -c "
import psutil
rows=sorted(((p.info['memory_info'].rss, p.info['name']) for p in
             psutil.process_iter(['name','memory_info'])), reverse=True)[:5]
print('top RSS:', [(n, round(r/1024**3,1)) for r,n in rows])
print('NOTE: RSS misses a compressed leak -- check: top -l 1 -o mem -stats pid,command,mem,cmprs')
"
  die "only ${AVAIL}GB available; refusing to measure under memory pressure"
fi
ok "machine clean: ${AVAIL}GB available"

# 3. Full 24 tasks, not just the 7 it missed. Greedy decoding makes a re-run of the
#    other 17 nearly a no-op, but "nearly" is doing work in that sentence -- GPU
#    batching is still non-deterministic -- and a single complete run is the only
#    provenance that cannot be questioned when this decides best_models.
info "running $MODEL, ceiling ${CEILING}s, log: $LOG"
START=$(date +%s)
set +e
timeout "$CEILING" ./.venv/bin/python -m eval --model "$MODEL" >"$LOG" 2>&1
CODE=$?
set -e
ELAPSED=$(( $(date +%s) - START ))

DONE_COUNT=$(grep -aoE '^[[:space:]]*[·⚠✗][[:space:]]+[a-z_]+: [0-9]+%' "$LOG" \
             | sed -E 's/.*[[:space:]]([a-z_]+): .*/\1/' | sort -u | wc -l | tr -d ' ')

if [ "$CODE" -eq 124 ]; then
  warn "$MODEL TRUNCATED AGAIN at ${CEILING}s after $DONE_COUNT task(s) — raise RERUN_TIMEOUT"
elif [ "$CODE" -ne 0 ]; then
  warn "$MODEL exited $CODE after ${ELAPSED}s, $DONE_COUNT task(s)"
else
  ok "$MODEL complete: $DONE_COUNT task(s) in ${ELAPSED}s"
fi
printf '%s\t%s\t%ss\ttasks=%s\texit=%s\n' \
  "$([ "$CODE" -eq 0 ] && echo DONE || echo INCOMPLETE)" "$MODEL" "$ELAPSED" "$DONE_COUNT" "$CODE" \
  >> "$ROOT/.sweep_status"
