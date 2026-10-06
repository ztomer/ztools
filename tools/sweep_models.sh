#!/usr/bin/env bash
# sweep_models.sh — run the full eval task set against every installed model,
# one model at a time, resumably, without ever recording a truncated run as complete.
#
#   ./tools/sweep_models.sh            sweep every installed model
#   ./tools/sweep_models.sh --resume   skip models already recorded DONE (a REFUSED,
#                                      TRUNCATED or FAILED model is re-run)
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
  # from it sends every relative path -- tui/lib.sh, tools/osaurus_one.sh --
  # into the temp directory. Caught by running the guard rather than by reading it.
  SWEEP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  _snapshot="$(mktemp -t sweep_models)"
  cat "${BASH_SOURCE[0]}" > "$_snapshot"
  SWEEP_REEXEC=1 SWEEP_ROOT="$SWEEP_ROOT" exec bash "$_snapshot" "$@"
fi

ROOT="${SWEEP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GOH="${GOH_DIR:-$HOME/Projects/gates_of_heck}"

# The output helpers, defined BEFORE the source so a missing tui degrades instead of
# dying. Two separate failures, one cause: `source` on an absent file prints
# "tui/lib.sh: No such file or directory" and, under `set -e`, exits with no
# explanation -- and had `die` then been undefined, the bare "die: command not found"
# is not an abort at all but a CONTINUE. tools/gpu_lock.sh documents the damage that
# caused: gpu_lock_acquire could never time out, it spun forever, and the test suite
# hung at 24%. A guard that only returns non-zero is not enough either -- `return` from
# a sourced file returns from the SOURCE and the caller carries on -- so the helpers
# are DEFINED here, not merely checked for.
for _sweep_helper in info ok warn err; do
  declare -F "$_sweep_helper" >/dev/null 2>&1 || eval "
    $_sweep_helper() { printf '%s\n' \"\$*\" >&2; }"
done
declare -F die >/dev/null 2>&1 || die() { err "$*"; exit "${2:-1}"; }
if [ -f "$GOH/tui/lib.sh" ]; then
  # shellcheck source=/dev/null
  source "$GOH/tui/lib.sh"
else
  section() { printf '\n-- %s --\n' "$*"; }
  warn "no tui/lib.sh under $GOH — output is plain text, no icons or colour."
  warn "set GOH_DIR to your gates_of_heck checkout (or run its install.sh) for the house TUI style."
fi
unset _sweep_helper

# The server guard, as a PATH-INDEPENDENT seam. This runs unattended for hours and its
# results are worthless without exactly one server, so the guarantee is this file
# succeeding -- and a test must be able to prove the sweep REFUSES without one without
# loading 27GB of weights. A bare absolute path can be stubbed by neither PATH nor
# $ROOT, so production keeps the default and a test overrides the whole path.
OSAURUS_ONE="${ZTOOLS_OSAURUS_ONE:-$ROOT/tools/osaurus_one.sh}"

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
#   ornith-1.0-9b-mxfp8
#              Cannot produce a score, so it cannot be ranked -- the same category as
#              the two above, reached by a different route. Its reasoning EXPANDS to
#              fill whatever budget it is handed (72,005 chars at 32,000 tokens,
#              144,441 at 64,000, cut by the stream guard at both) so it answers
#              nothing on the hard tasks at any budget. Historical mean 25% over 55
#              runs; it has never finished a sweep in three attempts (13/24, 11/30,
#              5/24) and it wedges the server on the way down. Full reasoning in
#              conf/config.toml's EXCLUDED block and docs/MODEL_QUIRKS.md.
#
# A one-off "not this time" still goes in --skip, where the reason is stated at the
# call site. A PERMANENT exclusion belongs here instead: --skip is only applied by an
# operator who remembers to pass it, and a bare `./tools/sweep_models.sh` re-running a
# model already known unrunnable is how 10 GPU-hours went into ornith-9b on 2026-08-23.
SKIP_RE="${SWEEP_SKIP:-^potion-|-mtp|^ornith-1\.0-9b-mxfp8$}"

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
  # `pwd -P`, not `pwd`. Bash reports the LOGICAL path by default, so resolving the
  # `latest` symlink returned ".../latest" rather than the run directory it points
  # at -- and the `ln -sfn` below then aimed the symlink at itself. Every model in a
  # resumed sweep failed instantly with "Too many levels of symbolic links".
  LOGDIR="$(cd "$LOGROOT/latest" 2>/dev/null && pwd -P)"
fi
mkdir -p "$LOGDIR"
ln -sfn "$LOGDIR" "$LOGROOT/latest"
touch "$STATUS"

# One server, or the numbers are worthless. See tools/osaurus_one.sh.
"$OSAURUS_ONE" >/dev/null || die "could not establish a single osaurus server"

if [ -n "$ONLY_MODEL" ]; then
  MODELS="$ONLY_MODEL"
else
  MODELS="$(osaurus list 2>/dev/null | grep -vE "$SKIP_RE" | grep -v '^$')"
fi
[ -n "$MODELS" ] || die "no models to sweep (skip pattern: $SKIP_RE)"
info "skipping: $SKIP_RE"

# Count the model list with awk, not `grep -c .`. Same reason as the two counts below
# and for the same class: `grep -c` prints its count on stdout and THEN exits non-zero
# when nothing matched, so a value built from it is only correct by accident, and a
# `|| echo 0` fallback turns "0" into the two-line string "0\n0".
TOTAL="$(printf '%s\n' "$MODELS" | awk 'NF { n++ } END { printf "%d\n", n + 0 }')"
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

  # The native eval (the Python `-m eval` was retired 2026-09-13). One model, the
  # full roster + taxes snapshots, markdown table on stdout.
  timeout "$PER_MODEL_TIMEOUT" "${ZTOOLS_BIN:-/opt/homebrew/bin/ztools}" model-eval \
    --model "$MODEL" --suite full > "$LOG" 2>&1
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
# DISTINCT task NAMES, not distinct rows. The old `tr -d ' ·⚠✗:'` normalised the
  # matched row but kept the SCORE in it, so `sort -u` deduped (name, score) pairs: a
  # retry that came back with a different score counted twice, and the progress number
  # exceeded the number of tasks that exist -- 30 of 23, which is not a progress
  # number, it is a bug wearing one. The name is extracted FIRST and the dedup runs on
  # the name alone, which is what rerun_truncated.sh already does.
  # `wc -l`, not `grep -c ... || echo 0`: grep -c prints 0 AND exits non-zero when
  # nothing matches, so the fallback fired too and TASKS_DONE became "0\n0" -- which
  # then split the status line in two. wc -l succeeds on empty input.
  # The `|| true` on the grep stage is the same rule for the sibling failure mode: the
  # matcher exits non-zero when it matched NOTHING, which is the ordinary case for a
  # model that refused to score. This script has no `set -e` so that exit is harmless
  # here, but it reaches the assignment's status, and rerun_truncated.sh -- which does
  # have `set -euo pipefail` -- aborted on exactly this before the fix. `|| true`
  # INSIDE the braces keeps the count a value and not an exit status.
  # A scored task is one row of the results table `| task | score | status | ... |`
  # (the old Python log used `  · task:` lines).
  TASKS_DONE="$( { grep -ohE '^\| [a-z_0-9]+ \| [0-9]+ \| ' "$LOG" 2>/dev/null || true; } \
    | sed -E 's/^\| ([a-z_0-9]+) .*/\1/' | sort -u | wc -l | tr -d ' ')"
  [[ "$TASKS_DONE" =~ ^[0-9]+$ ]] || TASKS_DONE=0

  # Remove any prior line for this model so --resume sees one record per model.
  if [ -s "$STATUS" ]; then
    grep -v "	$MODEL	" "$STATUS" > "$STATUS.tmp" 2>/dev/null || true
    mv "$STATUS.tmp" "$STATUS"
  fi

  if [ "$CODE" -eq 0 ] && [ "$TASKS_DONE" -eq 0 ]; then
    # Exit 0 with nothing scored is the eval REFUSING (oversize, paging, held
    # lock) -- a correct refusal, but not a measurement. Recorded as DONE it
    # was skipped by --resume forever: 2026-09-19 filed four models this way
    # in one sweep while a 17GB browser held the box, and a re-run on the
    # quiet box would have walked past all four.
    REASON=$(grep -m1 -oE 'Skipping [^:]+: .*' "$LOG" 2>/dev/null | cut -c1-120 || true)
    printf 'REFUSED\t%s\t%ss\ttasks=0\texit=0\t%s\n' "$MODEL" "$ELAPSED" "${REASON:-no reason logged}" >> "$STATUS"
    warn "[$i/$TOTAL] $MODEL — REFUSED, nothing scored: ${REASON:-see $LOG}"
  elif [ "$CODE" -eq 0 ]; then
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
# How many models did not finish. This is the verdict an operator reads, so it is
# counted by a tool that ALWAYS exits 0 and ALWAYS prints a number, with the
# unreadable-status-file case named rather than silently counted as zero.
#
# `grep -cE '^(TRUNCATED|FAILED)' "$STATUS" 2>/dev/null || echo 0` was here and was
# wrong twice: `grep -c` prints 0 AND exits 1 when nothing matches, so the fallback
# fired as well and INCOMPLETE became the two-line string "0\n0", which `[ -gt 0 ]`
# rejected with "integer expression expected" on every all-green sweep. It did not
# fail OPEN (a real TRUNCATED/FAILED row makes grep -c exit 0 with a valid count) --
# it printed an error instead of a verdict. It was also a re-introduction of the very
# construct this file had already fixed at TASKS_DONE, which is why the fix is now a
# RULE rather than a patched line: never build a number from a command whose EXIT
# STATUS is part of its output contract. Every count in this file ends in wc -l or
# awk, both of which succeed on empty input.
INCOMPLETE="$(awk '/^(TRUNCATED|FAILED)/ { n++ } END { printf "%d\n", n + 0 }' "$STATUS")" \
  || die "could not read the status file at $STATUS — not claiming the sweep completed"
if [ "$INCOMPLETE" -gt 0 ]; then
  warn "$INCOMPLETE model(s) did not finish — do NOT rank those against complete runs"
  exit 1
fi
ok "every model completed"
