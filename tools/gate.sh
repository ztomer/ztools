#!/usr/bin/env bash
# Per-repo gate entry point. Declares which toolchains this repo contains and
# delegates; it holds no gate logic of its own.
#   (none)   : the whole declared gate, identical to `make ci` and the pre-push hook
#   --full   : the same, spelled out
#   --staged : pre-commit scope (fast) -- structural checks only; the one cheap scope,
#              and it must be asked for by name
#
# THE FULL GATE IS ONE LIST, DECLARED ONCE, IN .gatesrc (GOH_CI_STEPS).
#
# THE DEFAULT IS THE FULL GATE (ZT.GATEHYG, 2026-11-12). An unscoped call used to fall through to
# structural.sh, so the entry point the docs name ran no clippy, no Rust suite, no coverage and no
# audit, and an agent that ran it got a green that skipped the tests. tools/tests/test_gate_entry.py
# pins the routing.
#
# Before 2026-09-02 there was no list: this file ran the structural layer with
# every language layer commented out, and nothing else existed to pick them up.
# The Rust half was therefore entirely ungated -- no fmt, no clippy, no tests,
# no coverage floor -- and had drifted to 223 formatting violations while every
# gate run reported all-green, because none of them had ever claimed to look.
set -euo pipefail
GOH="${GOH_DIR:-${GOH:-$HOME/Projects/gates_of_heck}}"

case "${1:-}" in
  --staged)
    exec "$GOH/gates/structural.sh" "$@"
    ;;
  ""|--full)
    exec "$GOH/gates/local_ci.sh" .
    ;;
  *)
    echo "tools/gate.sh: unknown scope '$1' (use --staged, --full, or none for the full gate)" >&2
    exit 2
    ;;
esac
