#!/usr/bin/env bash
# osaurus_one.sh — enforce EXACTLY ONE osaurus server, then hand back its port.
#
# Why this exists. Every model this repo evaluates is 4-35GB resident. A second
# server does not queue behind the first, it loads its OWN copy of whatever model
# it is asked for, and on a machine sized for one that means eviction, swapping,
# and requests the server cancels itself with
#
#     HTTP 499 {"error":{"message":"...Swift.CancellationError...",
#                        "type":"request_cancelled"}}
#
# Which is indistinguishable, from the client, from a model being slow. It is not
# a hypothetical: a stray second server turned qwen3.8-27b's decode rate into
# 0.1 tok/s and its cold start into 423s, and those numbers were on their way into
# conf/eval_signals.json as that model's permanent capability record -- permanent
# because the recorders keep the SLOWEST observation, so a contaminated reading
# can never be displaced by a correct one.
#
# So: never start a server by hand before a measurement. Call this.
#
#   ./tools/osaurus_one.sh              ensure one server on the default port
#   ./tools/osaurus_one.sh --port 1337  ensure one on a specific port
#   ./tools/osaurus_one.sh --check      report only, exit 1 if not exactly one
#   ./tools/osaurus_one.sh --restart    stop whatever is there, start one clean
#
# Deterministic: same end state whether it started with zero, one or several
# servers, and it prints which of those it found.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/tui/lib.sh"

PORT="${OSAURUS_PORT:-1337}"
MODE="ensure"
STARTUP_TIMEOUT="${OSAURUS_STARTUP_TIMEOUT:-90}"
LOG="${OSAURUS_LOG:-${TMPDIR:-/tmp}/osaurus-$PORT.log}"

while [ $# -gt 0 ]; do
  case "$1" in
    --port)    PORT="$2"; shift 2 ;;
    --check)   MODE="check"; shift ;;
    --restart) MODE="restart"; shift ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)         die "unknown argument: $1" ;;
  esac
done

require_commands osaurus lsof pgrep

# Every osaurus PROCESS, whether or not it holds a socket.
#
# An earlier version of this counted only PIDs listening on $PORT, reasoning that a
# process holding no socket cannot serve a request. True, and beside the point: the
# harm is not that a stray server answers, it is that it OCCUPIES memory and GPU.
# `osaurus stop` ends the serving and leaves the process alive, so the machine sat
# with two osaurus processes -- one holding 1337 and loading a 27B model, one idle
# but resident -- while this script reported "exactly one server" and the
# measurement it was guarding got corrupted anyway.
#
# Matched on the app binary path so this does not catch the `osaurus` CLI wrapper
# or a grep of its own command line.
OSAURUS_BIN="/Applications/osaurus.app/Contents/MacOS/osaurus"

server_pids() {
  pgrep -f "$OSAURUS_BIN" 2>/dev/null | sort -u || true
}

# The subset that is actually serving, which is a different question.
listener_pids() {
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | sort -u || true
}

describe() {
  local pid="$1" rss_kb
  rss_kb="$(/bin/ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || echo 0)"
  printf 'pid %s (%.1f GB resident)' "$pid" "$(echo "$rss_kb" | awk '{print $1/1048576}')"
}

wait_until_serving() {
  local waited=0
  while [ "$waited" -lt "$STARTUP_TIMEOUT" ]; do
    if curl -fsS -m 5 -o /dev/null "http://127.0.0.1:$PORT/v1/models" 2>/dev/null; then
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  return 1
}

stop_all() {
  local pids="$1"
  osaurus stop >/dev/null 2>&1 || true
  sleep 3
  # `osaurus stop` releases the socket but leaves the process resident, so the loop
  # below is the part that actually frees the memory -- and it is why this function
  # is given a PROCESS list rather than a listener list.
  for pid in $pids; do
    if kill -0 "$pid" 2>/dev/null; then
      warn "$(describe "$pid") survived 'osaurus stop'; terminating"
      kill "$pid" 2>/dev/null || true
    fi
  done
  sleep 2
  for pid in $pids; do
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
  done
  sleep 1
}

# Newline-separated string, not a bash array: `mapfile` is a bash 4 builtin and
# macOS ships bash 3.2, so a script that needs it only runs where someone already
# installed a newer bash. Word-splitting a list of PIDs is safe -- they are digits.
FOUND="$(server_pids)"
COUNT="$(printf '%s' "$FOUND" | grep -c . || true)"
FIRST="$(printf '%s\n' "$FOUND" | head -1)"

if [ "$MODE" = "check" ]; then
  case "$COUNT" in
    0) err "no osaurus process running"; exit 1 ;;
    1) if [ -z "$(listener_pids)" ]; then
         err "one osaurus process ($(describe "$FIRST")) but nothing listening on $PORT"
         exit 1
       fi
       ok "exactly one osaurus process, serving $PORT — $(describe "$FIRST")"; exit 0 ;;
    *) err "$COUNT osaurus processes — measurements taken now are not trustworthy"
       for pid in $FOUND; do err "  $(describe "$pid")"; done
       exit 1 ;;
  esac
fi

if [ "$MODE" = "restart" ] && [ "$COUNT" -gt 0 ]; then
  info "restart requested; stopping $COUNT server(s) on port $PORT"
  stop_all "$FOUND"
  COUNT=0
fi

if [ "$COUNT" -gt 1 ]; then
  warn "$COUNT osaurus processes — they compete for the same GPU and RAM"
  for pid in $FOUND; do warn "  $(describe "$pid")"; done
  info "stopping all, then starting one"
  stop_all "$FOUND"
  COUNT=0
elif [ "$COUNT" -eq 1 ]; then
  # Being the only process is not the same as answering. A server wedged
  # mid-eviction keeps its listener and 499s every request, and one that was
  # `osaurus stop`ped holds no listener at all while still occupying memory.
  if [ -n "$(listener_pids)" ] \
     && curl -fsS -m 10 -o /dev/null "http://127.0.0.1:$PORT/v1/models" 2>/dev/null; then
    ok "one osaurus server, answering — $(describe "$FIRST")"
    echo "$PORT"
    exit 0
  fi
  warn "one server holds port $PORT but does not answer /v1/models; restarting it"
  stop_all "$FOUND"
  COUNT=0
fi

info "starting osaurus on port $PORT (log: $LOG)"
nohup osaurus serve --port "$PORT" >"$LOG" 2>&1 &
disown || true

if ! wait_until_serving; then
  die "osaurus did not answer on port $PORT within ${STARTUP_TIMEOUT}s — see $LOG"
fi

NOW="$(server_pids)"
NOW_COUNT="$(printf '%s' "$NOW" | grep -c . || true)"
if [ "$NOW_COUNT" -ne 1 ]; then
  die "expected exactly one osaurus process, found $NOW_COUNT — see $LOG"
fi

ok "one osaurus server, answering — $(describe "$(printf '%s\n' "$NOW" | head -1)")"
echo "$PORT"
