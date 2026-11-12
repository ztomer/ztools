#!/usr/bin/env bash
# bench.sh -- osaurus bench speed rows (prefill/decode rate), serially, under the GPU lock.
#
#   tools/bench.sh --model qwen3.8-27b-jang_6d-crack [--model gemma-4-12b-it-mxfp8 ...]
#                  [--prompt-tokens 1024,8192] [--runs 3] [--out DIR]
#
# Writes DIR/bench.<model>.json and DIR/bench.<model>.log per model (default DIR: ./bench-out).
set -uo pipefail
GOH="${GOH_DIR:-$HOME/Projects/gates_of_heck}"
for _h in info ok warn err; do
  declare -F "$_h" >/dev/null 2>&1 || eval "$_h() { printf '%s\n' \"\$*\" >&2; }"
done
declare -F die >/dev/null 2>&1 || die() { err "$*"; exit "${2:-1}"; }
[ -f "$GOH/tui/lib.sh" ] && source "$GOH/tui/lib.sh"
unset _h
source "$GOH/lib/gpu_lock/gpu_lock.sh"

MODELS=(); PROMPT_TOKENS="1024,8192"; RUNS=3; OUT="./bench-out"
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODELS+=("$2"); shift 2 ;;
    --prompt-tokens) PROMPT_TOKENS="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[ "${#MODELS[@]}" -gt 0 ] || die "usage: tools/bench.sh --model NAME [--model NAME ...]"
mkdir -p "$OUT"

GPU_LOCK_TIMEOUT="${GPU_LOCK_TIMEOUT:-900}"
# Seconds a child gets after SIGTERM before it is sent SIGKILL.
KILL_GRACE="${BENCH_KILL_GRACE:-10}"
gpu_lock_acquire "bench.sh: osaurus bench speed rows" || exit 3

# THE CHILD IS REAPED BEFORE THE LOCK IS RELEASED. The lock exists so that no model run
# happens outside it; a SIGTERM that released the lock while `osaurus bench` kept running
# put a 12B bench next to an embedding run (ZT.TOOLFIX), and both measured contention.
# So the bench runs in the BACKGROUND and the script `wait`s on it -- bash defers a trap
# until a foreground child exits, but `wait` is interrupted by it -- and every exit path
# (clean, failed, INT/TERM/HUP) goes through reap_child, which stops the child and its
# descendants, escalates to SIGKILL after KILL_GRACE, and only then lets the EXIT trap
# release the lock.
CHILD=""
descendants() { # pid -> that pid's whole subtree, children first
  local c
  for c in $(pgrep -P "$1" 2>/dev/null); do descendants "$c"; printf '%s\n' "$c"; done
}
reap_child() {
  [ -n "$CHILD" ] || return 0
  local pids p i
  pids="$(descendants "$CHILD") $CHILD"
  # shellcheck disable=SC2086
  kill -TERM $pids 2>/dev/null || true
  for ((i = 0; i < KILL_GRACE * 20; i++)); do
    local live=0
    for p in $pids; do kill -0 "$p" 2>/dev/null && live=1; done
    [ "$live" = 0 ] && break
    sleep 0.05
  done
  # shellcheck disable=SC2086
  kill -KILL $pids 2>/dev/null || true
  wait "$CHILD" 2>/dev/null || true
  CHILD=""
}
finish() { reap_child; gpu_lock_release; }
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

FAILED=0
for m in "${MODELS[@]}"; do
  info "bench $m"
  osaurus bench --model "$m" --prompt-tokens "$PROMPT_TOKENS" --runs "$RUNS" \
    --json "$OUT/bench.$m.json" > "$OUT/bench.$m.log" 2>&1 &
  CHILD=$!
  rc=0
  wait "$CHILD" || rc=$?
  CHILD=""
  if [ "$rc" -ne 0 ]; then warn "$m: osaurus bench exited $rc (log: $OUT/bench.$m.log)"; FAILED=1; fi
done
exit "$FAILED"
