#!/usr/bin/env bash
# Re-run models the sweep TRUNCATED, with a raised per-model ceiling.
#
# WHY. The sweep's ceiling is 14400s (4h). Under the corrected 32000-token budgets the
# reasoning models blow past it: nemotron averages 838s/task because it emits up to 77k
# characters of reasoning before answering, and ornith-9b is the same shape. Both were
# cut off mid-sweep -- nemotron at 17 of 24, ornith-9b at 13 of 24 -- and a model
# measured on a subset cannot be ranked against models measured on the full set.
# Worse, the tasks they miss are the LAST ones, which are the largest prompts, so the
# subset is biased toward the easy end. (Task counts in this comment are historical --
# the suite has grown since; read the count from the run, never from here.)
#
# The per-TASK timeout USED to be described here as "never the constraint" because it
# derives from measured rates and sat around 7055s. That was wrong, and the correction
# matters: the derivation reads the machine as it is, so a contended box measures slow
# rates, slow rates INFLATE the timeout, and the inflated timeout permits a longer
# stall. qwen3.8-27b-mxfp8 earned a 2-hour per-task ceiling from a decode rate of
# 0.1158 tok/s and then sat wedged for 83 minutes having completed nothing.
#
# Two things changed as a result, and this script relies on both: _derived_timeout now
# uses CLEAN samples only (falling back to the 900s floor rather than trusting a
# reading the sampler already distrusted), and eval/watchdog.py abandons a model that
# completes no task inside MODEL_STALL_SECONDS regardless of what any timeout believes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
# shellcheck source=/dev/null
[ -f tui/lib.sh ] && . tui/lib.sh || { info(){ echo "→ $*"; }; ok(){ echo "✓ $*"; }
                                       warn(){ echo "⚠ $*"; }; die(){ echo "✗ $*" >&2; exit 1; }; }

CEILING="${RERUN_TIMEOUT:-36000}"
MIN_FREE_GB="${RERUN_MIN_FREE_GB:-6}"   # exhaustion floor only; pressure is gated separately
LOGDIR="${TMPDIR:-/tmp}/ztools-sweep"; mkdir -p "$LOGDIR"

# Models to redo. Default: whatever .sweep_status recorded as not DONE.
if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  # LAST state per model, not every line. This script APPENDS its own result, so a
  # model fixed by an earlier invocation still has its old TRUNCATED line on disk --
  # `sort -u` would dutifully queue it again, forever.
  MODELS=()
  while IFS=$'\t' read -r state model; do
    [ "$state" = "DONE" ] || MODELS+=("$model")
  done < <(awk -F'\t' 'NF>1 {last[$2]=$1} END {for (m in last) print last[m] "\t" m}' \
             "$ROOT/.sweep_status" 2>/dev/null || true)
fi
[ "${#MODELS[@]}" -gt 0 ] || { ok "nothing truncated; nothing to redo"; exit 0; }
info "to redo: ${MODELS[*]}"

while pgrep -f "sweep_models.sh" >/dev/null 2>&1; do sleep 60; done

for MODEL in "${MODELS[@]}"; do
  # Restart the server FIRST, then measure free memory.
  #
  # Order matters and getting it wrong is what broke the first version of this script:
  # after a sweep osaurus still holds the last model resident (14GB+), so "available
  # memory" is low BY DESIGN and a naive threshold refuses to run on a perfectly clean
  # machine. Restarting frees it, which is also the state the sweep itself starts from,
  # so the check then measures what it is actually about -- whether something ELSE is
  # eating the box.
  ./tools/osaurus_one.sh --restart >/dev/null 2>&1 || die "could not restart osaurus"

  # Gate on PRESSURE (swap + compressor), not on headroom.
  #
  # "Available memory" is the wrong instrument on this machine and the first version of
  # this guard was wrong because of it. After a sweep, wired GPU memory and the
  # file-backed page cache holding model weights legitimately push `available` down to
  # ~12GB on an idle, healthy box -- so a headroom threshold refuses to run on a machine
  # that had just completed a 24-task sweep at full speed.
  #
  # Swap and compressor separate the two states cleanly, which headroom never did:
  #
  #     during the 31GB leak    swap 12.88 GB   compressor 29.3 GB   (thrashing)
  #     healthy after a sweep   swap  1.43 GB   compressor  5.1 GB   (fine)
  #
  # A floor on free+available is kept only to catch outright exhaustion.
  read -r SWAP_GB CMPR_GB AVAIL <<EOF
$(./.venv/bin/python - <<'PY'
import re, subprocess, psutil
vm = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
page = 16384
def pages(label):
    m = re.search(rf'{label}:\s+(\d+)', vm)
    return (int(m.group(1)) * page / 1024**3) if m else 0.0
print(f"{psutil.swap_memory().used/1024**3:.1f} {pages('Pages occupied by compressor'):.1f} "
      f"{int(psutil.virtual_memory().available/1024**3)}")
PY
)
EOF
  if awk "BEGIN{exit !($SWAP_GB > ${RERUN_MAX_SWAP_GB:-8} || $CMPR_GB > ${RERUN_MAX_CMPR_GB:-15})}"; then
    ./.venv/bin/python - <<'PY'
import psutil
rows = []
for p in psutil.process_iter(['name', 'memory_info']):
    mi = p.info.get('memory_info')          # None for processes we cannot inspect
    if mi is not None:
        rows.append((mi.rss, p.info.get('name') or '?'))
rows.sort(reverse=True)
print('  top RSS:', [(n, f'{r/1024**3:.1f}G') for r, n in rows[:5]])
print('  RSS HIDES A COMPRESSED LEAK -- the 31GB one showed 0.55GB of RSS.')
print('  Find it with:  top -l 1 -o mem -n 10 -stats pid,command,mem,cmprs')
PY
    die "machine is thrashing (swap ${SWAP_GB}GB, compressor ${CMPR_GB}GB); refusing to measure"
  fi
  [ "$AVAIL" -ge "$MIN_FREE_GB" ] || die "only ${AVAIL}GB available; machine is exhausted"
  ok "$MODEL: no pressure (swap ${SWAP_GB}GB, compressor ${CMPR_GB}GB, ${AVAIL}GB available)"

  LOG="$LOGDIR/rerun-$MODEL.log"
  info "running $MODEL, ceiling ${CEILING}s, log: $LOG"
  START=$(date +%s)
  set +e
  timeout "$CEILING" ./.venv/bin/python -m eval --model "$MODEL" >"$LOG" 2>&1
  CODE=$?
  set -e
  ELAPSED=$(( $(date +%s) - START ))

  # Count DISTINCT task names that reported a score. Counting lines over-counts retries;
  # counting only the ok marker under-counts, since a warn or a fail is still a score.
  DONE_COUNT=$(grep -aoE '[·⚠✗][[:space:]]+[a-z_]+: [0-9]+%' "$LOG" 2>/dev/null \
               | sed -E 's/.*[[:space:]]([a-z_]+): .*/\1/' | sort -u | wc -l | tr -d ' ')
  DONE_COUNT=${DONE_COUNT:-0}

  if [ "$CODE" -eq 124 ]; then
    warn "$MODEL TRUNCATED AGAIN at ${CEILING}s after $DONE_COUNT task(s) — raise RERUN_TIMEOUT"
    STATE=TRUNCATED
  elif [ "$CODE" -ne 0 ]; then
    warn "$MODEL exited $CODE after ${ELAPSED}s, $DONE_COUNT task(s)"
    STATE=FAILED
  else
    ok "$MODEL complete: $DONE_COUNT task(s) in ${ELAPSED}s"
    STATE=DONE
  fi
  printf '%s\t%s\t%ss\ttasks=%s\texit=%s\n' \
    "$STATE" "$MODEL" "$ELAPSED" "$DONE_COUNT" "$CODE" >> "$ROOT/.sweep_status"
done
