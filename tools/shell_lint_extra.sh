#!/usr/bin/env bash
# shell_lint_extra.sh — bash -n + shellcheck over the shell that the house
# structural checker cannot see.
#
# WHY THIS IS A REPO-LOCAL STEP. The gate's shell selection is `*.sh` plus
# `hooks/*`, and it lives in `gates_of_heck/structural.sh`, so two classes of
# file in THIS repo are outside it by construction:
#
#   .githooks/pre-commit   the last line of defence before a commit lands
#   .githooks/pre-push     the last line of defence before anything leaves this
#                          machine, and there is no CI behind it
#   bin/ab_test            extensionless, and therefore matched by neither glob
#
# The first two are the hooks that guard every other gate in this file's own
# peer list, so a syntax error or an unbound-variable bug in them is a gate that
# does not gate. They are also the files a reader is least likely to run
# casually, because running them means running the whole gate. `bin/ab_test` is
# lint-only here and is never edited by this step.
#
# WHY `--severity=error` AND NOT `--severity=style`. shellcheck's style and info
# levels are suggestions about a script's shape; `error` is the level where the
# script's own semantics are wrong (unquoted expansion, a redirect into a
# read-only assignment, a `local` outside a function, a command that will fail
# under `set -euo pipefail`). Those are what make a hook silently do nothing.
# Demanding style on shell that other people own would make this step red for
# opinions and teach everyone to ignore it.
#
# FAILING CLOSED, and naming what it checked. If shellcheck is not installed the
# step must NOT pass: a gate that skips itself when its tool is missing is a gate
# that reports success it never earned, and the missing tool is exactly the case
# nobody notices. It says what it would have checked and exits non-zero.
#
# Usage:
#     tools/shell_lint_extra.sh            # every path below
#     tools/shell_lint_extra.sh FILE ...   # only these paths
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
GOH="${GOH_DIR:-$HOME/Projects/gates_of_heck}"

# The output helpers, defined BEFORE the source so a missing tui degrades instead
# of dying. Sourcing an absent file prints "No such file or directory" and, under
# `set -e`, exits with nothing named -- fail-closed but unnamed, which is the
# worst of both. Same rule `tools/gpu_lock.sh` and `tools/release.sh` state: a
# guard that only returns non-zero is not enough, because `return` from a sourced
# file returns from the SOURCE and the caller carries on regardless. So the
# helpers are DEFINED here, not merely checked for.
for _lint_helper in info ok warn err; do
    declare -F "$_lint_helper" >/dev/null 2>&1 || eval "
        $_lint_helper() { printf '%s\n' \"\$*\" >&2; }"
done
declare -F die >/dev/null 2>&1 || die() { err "$*"; exit "${2:-1}"; }
if [ -f "$GOH/tui/lib.sh" ]; then
    # shellcheck source=/dev/null
    source "$GOH/tui/lib.sh"
else
    warn "no tui/lib.sh under $GOH — output is plain text, no icons or colour."
    warn "set GOH_DIR to your gates_of_heck checkout (or run its install.sh) for the house TUI style."
fi
unset _lint_helper

# WHAT IS CHECKED, and the one thing that is explicitly not: `bin/ab_test` is
# listed because it is shell and outside the house globs, and it is READ-ONLY
# here on purpose — it is under active development elsewhere, so a lint failure
# in it is reported to whoever owns it rather than fixed by a step that does not
# own it. The hooks are the opposite: this repo owns them, so a finding in them
# is fixed here.
TARGETS=(
    .githooks/pre-commit
    .githooks/pre-push
    bin/ab_test
)

if [ "$#" -gt 0 ]; then
    TARGETS=("$@")
fi

info "shell lint — ${#TARGETS[@]} path(s): ${TARGETS[*]}"

command -v shellcheck >/dev/null 2>&1 || die \
    "shellcheck is not installed, so this step cannot check: ${TARGETS[*]}
  → brew install shellcheck
  Skipping is NOT passing: the files listed above are the git hooks, and an
  unlinted hook is an ungated one."

# Checked for existence HERE rather than inside the loop, so a missing file is one
# loud failure naming the file instead of a per-file `ok` that reads like a pass.
for t in "${TARGETS[@]}"; do
    [ -f "$t" ] || die "expected shell to lint is missing: $t"
    # `bash -n` first: it is the parser, it needs no external tool, and a
    # SYNTAX error makes shellcheck's diagnostics unreadable -- the noise would
    # bury the one finding that matters.
    bash -n "$t" || die "bash -n: syntax error in $t"
done

failed=0
for t in "${TARGETS[@]}"; do
    # `-x` follows `source`d files, which is what makes this check the hooks:
    # without it SC1091 ("not following tui/lib.sh") is emitted for every house
    # script that sources the TUI, and a step that cries wolf on correct code is
    # a step people learn to skip.
    if shellcheck -x --severity=error --format=gcc "$t"; then
        ok "shellcheck --severity=error: $t"
    else
        err "shellcheck --severity=error: $t"
        failed=$((failed + 1))
    fi
done

if [ "$failed" -gt 0 ]; then
    die "$failed of ${#TARGETS[@]} path(s) failed shellcheck at --severity=error"
fi

# WHAT THIS STEP DOES NOT CATCH, counted rather than asserted, because the honest
# limit of a gate is part of its output. MEASURED on shellcheck 0.11: the defect
# people most expect a shell lint to catch -- an unquoted expansion that
# word-splits and globs (SC2086) -- is classified INFO, so it is below this
# step's floor and is NOT a failure here. Raising the floor to `--severity=style`
# is a one-word change and today all three files are clean at every level, but
# `bin/ab_test` is owned elsewhere and in progress: a floor that goes red on
# another agent's style nit gets the whole step ignored, so the floor stays at
# `error` and the rest is reported instead.
for t in "${TARGETS[@]}"; do
    nits=$(shellcheck -x --severity=style --format=gcc "$t" 2>/dev/null | grep -c . || true)
    [ "$nits" -gt 0 ] && warn "$t: $nits finding(s) below the failing floor (style/warning/info) — not failures"
done
unset nits

ok "shell lint clean — bash -n + shellcheck --severity=error over ${#TARGETS[@]} path(s)"
