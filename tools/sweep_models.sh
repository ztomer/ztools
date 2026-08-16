#!/usr/bin/env bash
# sweep_models.sh — run the full eval task set against every installed model,
# one model at a time, resumably, without ever recording a truncated run as complete.
#
#   ./tools/sweep_models.sh            sweep every installed model
#   ./tools/sweep_models.sh --resume   skip models already recorded DONE
#   ./tools/sweep_models.sh --status   print the status file and exit
#   ./tools/sweep_models.sh --model X  just one model
#
# WHY THIS IS A SHIPPED TOOL AND NOT A SCRATCH SCRIPT. The previous version lived in
# a scratchpad and wrote its DONE marker regardless of exit code, so a model killed at
# the `timeout` boundary -- ornith at 16/23 tasks, bonsai at 11/23 -- was recorded as
# complete and SKIPPED on the next run, silently losing every task it never reached.
# Worse, it invalidated a comparison nobody knew was invalid: bonsai's mean of 99
# against ornith's 70 was two different task subsets. A truncated run must LOOK
# truncated, which is the one thing that script got wrong and the reason this one
# records the exit code and the task count rather than a bare marker.
#
# Serial by construction: the GPU is one shared resource, models are 4-27GB resident,
# and two concurrent runs measure contention rather than either model.

set -uo pipefail

# Re-exec from an immutable copy of this script.
#
# bash reads a script LAZILY, by byte offset, so editing the file while it runs can
# make the running shell resume mid-token and execute garbage. A sweep runs for
# hours, which is exactly the window in which someone -- me, during this session --
# edits the harness to fix something. That edit silently did not take effect (the
# loop body was already parsed) and could just as easily have corrupted the run.
#
# Copying to a temp file and re-execing makes a sweep immune to edits of its own
# source, and means an edited harness applies to the NEXT run rather than half of
# this one, which is also the only way its results stay comparable.
if [ -z "${SWEEP_REEXEC:-}" ]; then
  # ROOT must be resolved from the ORIGINAL location and carried across. After the
  # re-exec BASH_SOURCE points at the snapshot in $TMPDIR, and deriving the repo root
  # from it sends every relative path -- tui/lib.sh, tools/osaurus_one.sh, .venv --
  # into the temp directory. Caught by running the guard rather than by reading it.
  SWEEP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  _snapshot="$(mktemp -t sweep_models)"
  cat "${BASH_SOURCE[0]}" > "$_snapshot"
  SWEEP_REEXEC=1 SWEEP_ROOT="$SWEEP_ROOT" exec bash "$_snapshot" "$@"
fi

ROOT="${SWEEP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1091
source "$ROOT/tui/lib.sh"

STATUS="${SWEEP_STATUS:-$ROOT/.sweep_status}"
# Per-RUN log directory, with a `latest` symlink.
#
# A shared directory outlives its run, and stale logs are indistinguishable from
# current ones to anything reading them afterwards -- a monitor watching for failure
# signatures reported seven HTTP 499s and eight INFRA failures from a previous sweep
# while the live run was perfectly clean. Same class as writing DONE regardless of
# exit code: the artifact stops describing the run that produced it.
LOGROOT="${SWEEP_LOGDIR:-${TMPDIR:-/tmp}/ztools-sweep}"
LOGDIR="$LOGROOT/run-$(date +%Y%m%d-%H%M%S)"
PER_MODEL_TIMEOUT="${SWEEP_MODEL_TIMEOUT:-14400}"   # 4h; a slow model is not a failure
RESUME=0
ONLY_MODEL=""
# Entries the server lists that are not chat models, and so cannot be ranked:
#   ^potion-   model2vec embeddings; the server answers HTTP 500 "Unsupported model
#              type: model2vec". `ev` skips these by name too, but doing it here keeps
#              the sweep's model list honest rather than relying on a downstream skip.
#   -mtp       speculative-decoding DRAFTER weights (Qwen3.8-27B-MTP-4bit), listed as
#              a peer of the model they accelerate. Evaluating one measures nothing.
# Anything else you want out (a model known too slow to finish) goes in --skip, so the
# reason is stated at the call site instead of buried here where it would rot.
SKIP_RE="${SWEEP_SKIP:-^potion-|-mtp}"

while [ $# -gt 0 ]; do
  case "$1" in
    --resume) RESUME=1; shift ;;
    --status) [ -f "$STATUS" ] && cat "$STATUS" || echo "(no status file at $STATUS)"; exit 0 ;;
    --model)  ONLY_MODEL="$2"; shift 2 ;;
    --skip)   SKIP_RE="$SKIP_RE|$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

# --resume continues an existing run, so it keeps that run's directory rather than
# opening a new one -- otherwise half a run's logs would sit in one directory and half
# in another, which is the confusion this is meant to remove.
if [ "$RESUME" -eq 1 ] && [ -L "$LOGROOT/latest" ]; then
  LOGDIR="$(cd "$LOGROOT/latest" 2>/dev/null && pwd)"
fi
mkdir -p "$LOGDIR"
ln -sfn "$LOGDIR" "$LOGROOT/latest"
touch "$STATUS"

# One server, or the numbers are worthless. See tools/osaurus_one.sh.
"$ROOT/tools/osaurus_one.sh" >/dev/null || die "could not establish a single osaurus server"

if [ -n "$ONLY_MODEL" ]; then
  MODELS="$ONLY_MODEL"
else
  MODELS="$(osaurus list 2>/dev/null | grep -vE "$SKIP_RE" | grep -v '^$')"
fi
[ -n "$MODELS" ] || die "no models to sweep (skip pattern: $SKIP_RE)"
info "skipping: $SKIP_RE"

TOTAL="$(printf '%s\n' "$MODELS" | grep -c .)"
section "Sweeping $TOTAL model(s)"
info "status: $STATUS"
info "logs:   $LOGDIR"

i=0
for MODEL in $MODELS; do
  i=$((i + 1))
  if [ "$RESUME" -eq 1 ] && grep -q "^DONE	$MODEL	" "$STATUS" 2>/dev/null; then
    info "[$i/$TOTAL] $MODEL — already DONE, skipping"
    continue
  fi

  LOG="$LOGDIR/$MODEL.log"
  info "[$i/$TOTAL] $MODEL — running (log: $LOG)"
  START=$(date +%s)

  timeout "$PER_MODEL_TIMEOUT" "$ROOT/.venv/bin/python" -m eval --model "$MODEL" \
    > "$LOG" 2>&1
  CODE=$?

  ELAPSED=$(( $(date +%s) - START ))
  # Let the child's buffered output land before counting. `timeout` returning means
  # the process is gone, not that everything it wrote has reached the file: counting
  # immediately reported 22 of 23 for a model that had in fact scored all 23, with
  # the last line appearing a moment later.
  sleep 1
  # Count tasks that actually reported a score, so a partial run is visible as a
  # number rather than inferred from an exit code alone.
  #
  # ALL THREE result markers, not just the good one. `ev` prints a scored task as
  # `· name: 91%` when it passed, `⚠ name: 55%` when it scored poorly and `✗ name: 30%`
  # when it failed -- all three ARE scores. Counting only `·` reported 20 of 23 for
  # every model and made a model that scored badly look like a model that ran fewer
  # tasks, which is the truncated-looks-complete confusion this script exists to
  # prevent, inverted.
  # DISTINCT task names, because a retried task logs a second score line and a raw
  # line count then exceeds the number of tasks that exist -- 30 of 23, which is not
  # a progress number, it is a bug wearing one.
  # `wc -l`, not `grep -c ... || echo 0`: grep -c prints 0 AND exits non-zero when
  # nothing matches, so the fallback fired too and TASKS_DONE became "0\n0" -- which
  # then split the status line in two. wc -l succeeds on empty input.
  TASKS_DONE=$(grep -ohE '^[[:space:]]+(·|⚠|✗)[[:space:]]+[a-z_]+:' "$LOG" 2>/dev/null \
    | tr -d ' ·⚠✗:' | sort -u | wc -l | tr -d ' ')

  # Remove any prior line for this model so --resume sees one record per model.
  if [ -s "$STATUS" ]; then
    grep -v "	$MODEL	" "$STATUS" > "$STATUS.tmp" 2>/dev/null || true
    mv "$STATUS.tmp" "$STATUS"
  fi

  if [ "$CODE" -eq 0 ]; then
    printf 'DONE\t%s\t%ss\ttasks=%s\texit=0\n' "$MODEL" "$ELAPSED" "$TASKS_DONE" >> "$STATUS"
    ok "[$i/$TOTAL] $MODEL — done in ${ELAPSED}s, $TASKS_DONE task(s) scored"
  elif [ "$CODE" -eq 124 ]; then
    printf 'TRUNCATED\t%s\t%ss\ttasks=%s\texit=124(timeout)\n' "$MODEL" "$ELAPSED" "$TASKS_DONE" >> "$STATUS"
    warn "[$i/$TOTAL] $MODEL — TRUNCATED at ${PER_MODEL_TIMEOUT}s after $TASKS_DONE task(s);" \
         "its scores cover a SUBSET and are not comparable with a complete run"
  else
    printf 'FAILED\t%s\t%ss\ttasks=%s\texit=%s\n' "$MODEL" "$ELAPSED" "$TASKS_DONE" "$CODE" >> "$STATUS"
    err "[$i/$TOTAL] $MODEL — FAILED (exit $CODE) after $TASKS_DONE task(s); see $LOG"
  fi
done

section "Sweep summary"
cat "$STATUS"
INCOMPLETE="$(grep -cE '^(TRUNCATED|FAILED)' "$STATUS" 2>/dev/null || echo 0)"
if [ "$INCOMPLETE" -gt 0 ]; then
  warn "$INCOMPLETE model(s) did not finish — do NOT rank those against complete runs"
  exit 1
fi
ok "every model completed"
