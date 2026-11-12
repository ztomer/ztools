# shellcheck shell=bash
# osaurus_door.sh -- THE ONLY PLACE a ztools shell tool runs the osaurus CLI (ZT.OSAURUSSHELL).
#
# Sourced, never run. The osaurus CLI probes the server for 0.5 s and, when it does not
# answer, has LaunchServices open a SECOND Osaurus app (--launched-by-cli; ZT.OSAURUSDUP,
# where the Rust crate got the same door in refcheck/door.rs). So:
#
#   osaurus_door ARGS...      runs `osaurus ARGS...` only once $OSAURUS_URL/health answered
#                             within OSAURUS_DOOR_WAIT seconds (ten times the CLI's probe);
#                             otherwise it refuses with the reason, exit 69, and never starts
#                             the CLI. A health-checked door only NARROWS the race (under
#                             load /health answered and the CLI's own probe still failed), so
#                             reach for HTTP first wherever the server serves the question.
#   osaurus_door_serve ARGS... the deliberate start, `osaurus serve ARGS...`: refused (69)
#                             when /health ALREADY answers, since that would be a second server.
#   osaurus_door_models       the model ids over HTTP (GET /v1/models), one per line; no CLI.
#
# tools/check_osaurus_door.py fails the gate on any bare `osaurus` command outside this file.
OSAURUS_URL="${OSAURUS_URL:-http://127.0.0.1:1337}"
OSAURUS_DOOR_WAIT="${OSAURUS_DOOR_WAIT:-5}"

osaurus_door_health() {
  curl -fsS -m "$OSAURUS_DOOR_WAIT" -o /dev/null "$OSAURUS_URL/health" 2>/dev/null
}

osaurus_door() {
  if ! osaurus_door_health; then
    printf 'osaurus_door: refused `osaurus %s`: %s/health did not answer within %ss (the CLI would start a second app)\n' \
      "${1:-}" "$OSAURUS_URL" "$OSAURUS_DOOR_WAIT" >&2
    return 69
  fi
  command osaurus "$@"
}

osaurus_door_serve() {
  if osaurus_door_health; then
    printf 'osaurus_door: refused `osaurus serve`: %s/health already answers (that would be a second server)\n' \
      "$OSAURUS_URL" >&2
    return 69
  fi
  command osaurus serve "$@"
}

osaurus_door_models() {
  curl -fsS -m 10 "$OSAURUS_URL/v1/models" 2>/dev/null \
    | python3 -c 'import json,sys
for m in json.load(sys.stdin).get("data", []):
    print(m["id"])'
}
