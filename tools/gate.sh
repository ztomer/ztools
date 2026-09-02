#!/usr/bin/env bash
# Per-repo gate entry point. Declares which toolchains this repo contains and
# delegates; it holds no gate logic of its own.
#   --staged : pre-commit scope (fast) -- structural checks only
#   --full   : pre-push scope -- the whole declared gate, identical to `make ci`
#
# THE FULL GATE IS ONE LIST, DECLARED ONCE, IN .gatesrc (GOH_CI_STEPS).
#
# Before 2026-09-02 there was no list: this file ran the structural layer with
# every language layer commented out, and nothing else existed to pick them up.
# The Rust half was therefore entirely ungated -- no fmt, no clippy, no tests,
# no coverage floor -- and had drifted to 223 formatting violations while every
# gate run reported all-green, because none of them had ever claimed to look.
set -euo pipefail
GOH="${GOH_DIR:-${GOH:-$HOME/Projects/gates_of_heck}}"

case "${1:-}" in
  --full)
    exec "$GOH/gates/local_ci.sh" .
    ;;
  *)
    exec "$GOH/gates/structural.sh" "$@"
    ;;
esac
