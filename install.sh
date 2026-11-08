#!/usr/bin/env bash
# install.sh — Build and install ztools release binaries into ~/.local/bin
#
# The Homebrew tap (maintained by tools/release.sh) is the public install
# path; this is the LOCAL door: build the Rust release here and install it
# into "$HOME/.local/bin", no network, no tap.
#
# ONE OWNER PER PATH (ZT.BREWLINK): "$(brew --prefix)/bin" belongs to the tap
# formula, which `brew link`s its entries there. This script REFUSES that
# directory: a real file it left there made the next `brew reinstall` fail at
# `brew link` until someone ran `brew link --overwrite`.
#
# ENVIRONMENT
#   ZTOOLS_INSTALL_DIR         target dir; default "$HOME/.local/bin"
#   ZTOOLS_OS / ZTOOLS_ARCH    OVERRIDE the platform, which otherwise comes from
#                              `uname -s` / `uname -m` (see the gate below)
#   ZTOOLS_PLATFORM_CHECK_ONLY=1   run the platform gate, print it, exit 0 BEFORE
#                                   any build or install
#
# The last two are the platform gate's SEAM. They exist so the gate is testable
# in BOTH directions — accept and reject — on whatever machine you happen to be
# on (tools/tests/test_install_platform_gate.py drives them), and they are the
# documented escape hatch for a developer who needs to check the gate's verdict
# without a matching host. They are deliberately NOT a way to make an
# unsupported platform installable: they change what the gate BELIEVES the
# machine is, and every other step still runs for real.
set -euo pipefail

GOH="${GOH_DIR:-$HOME/Projects/gates_of_heck}"

# The output helpers, defined BEFORE the source so a missing tui degrades instead
# of dying. Sourcing an absent file prints "No such file or directory" and, under
# `set -e`, exits with nothing named — fail-closed but unnamed, which is the worst
# of both. Same rule the house `gpu_lock.sh` and tools/release.sh state: a guard that
# only returns non-zero is not enough, because `return` from a sourced file returns
# from the SOURCE and the caller carries on — so the helpers are DEFINED here, not
# merely checked for. `require_commands` is in the list for the same reason and is
# called below: with nothing sourced it is undefined, and "command not found" at
# 127 names neither the cause nor the remedy.
for _install_helper in info ok warn err; do
    declare -F "$_install_helper" >/dev/null 2>&1 || eval "
        $_install_helper() { printf '%s\n' \"\$*\" >&2; }"
done
declare -F die >/dev/null 2>&1 || die() { err "$*"; exit "${2:-1}"; }
# The real check, not a stub. A fallback that merely printed would take the OTHER
# failure: this script would carry on with cargo or jq absent and install whatever
# happened to be in the bin dir.
declare -F require_commands >/dev/null 2>&1 || require_commands() {
    local missing=() _cmd
    for _cmd in "$@"; do
        command -v "$_cmd" >/dev/null 2>&1 || missing+=("$_cmd")
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        die "missing required command(s): ${missing[*]}"
    fi
}
if [ -f "$GOH/tui/lib.sh" ]; then
    # shellcheck source=/dev/null
    source "$GOH/tui/lib.sh"
else
    warn "no tui/lib.sh under $GOH — output is plain text, no icons or colour."
    warn "set GOH_DIR to your gates_of_heck checkout (or run its install.sh) for the house TUI style."
fi
unset _install_helper

# ── Platform gate ──────────────────────────────────────────────────────────────
# House policy: 64-bit only; macOS is Apple silicon ONLY (Intel Macs are dropped);
# Linux keeps x86_64 and aarch64. Every other combination is a HARD FAILURE, never
# a warn-and-build-anyway `case` fallback.
#
# THE DECISION IS THE SHARED LIBRARY'S (LIB.1), not this file's. house-lib carries
# the policy once: in Rust (`house_lib::platform::check`, compiled into its
# build.rs too, so the library refuses to build for a target outside the set) and
# in shell (`platform.sh` beside its Cargo.toml, held to the Rust form over every
# spelling by the library's own tests). This script used to carry a third copy,
# whose spellings had drifted from the other two. It is found through cargo's
# dependency graph, so the shell gate is always the one from the tag
# rust/Cargo.toml names, never a sibling checkout. This repo keeps only the seam.
#
# `uname`'s own spellings, so the failure names what was found. ${VAR:-$(...)}
# evaluates the substitution only when VAR is unset, so the seam costs no uname call.
UNAME_OS="${ZTOOLS_OS:-$(uname -s)}"
UNAME_ARCH="${ZTOOLS_ARCH:-$(uname -m)}"

# cargo and jq are named BEFORE their first use, which is now the gate itself:
# `cargo metadata` READS the graph (it fetches house-lib, builds nothing). A host
# that cannot read the graph does not install unchecked: it stops here, named.
require_commands cargo jq
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if ! HOUSE_LIB_MANIFEST="$(cargo metadata --manifest-path "$ROOT/rust/Cargo.toml" \
        --format-version 1 \
        | jq -er '.packages[] | select(.name == "house-lib") | .manifest_path')"; then
    die "cannot find house-lib's platform gate: cargo metadata over $ROOT/rust/Cargo.toml failed, or names no house-lib package."
fi
HOUSE_PLATFORM_SH="${HOUSE_LIB_MANIFEST%/*}/platform.sh"
[ -f "$HOUSE_PLATFORM_SH" ] \
    || die "house-lib's platform.sh is not beside its manifest ($HOUSE_LIB_MANIFEST)."
# shellcheck source=/dev/null
. "$HOUSE_PLATFORM_SH"
house_platform_check "$UNAME_OS" "$UNAME_ARCH" || die "refusing to build or install ztools here."

# Test seam: stop HERE, after the gate and before any build, install or Homebrew
# requirement -- so both directions are testable on one machine (reject for every
# unsupported pair, accept for each supported one) and no test can ever install a
# binary as a side effect of checking the gate.
if [ "${ZTOOLS_PLATFORM_CHECK_ONLY:-}" = "1" ]; then
    ok "gate ok: ${UNAME_OS}/${UNAME_ARCH}"
    exit 0
fi

# ── Install target ─────────────────────────────────────────────────────────────
# Decided and checked BEFORE the build, so a refusal costs nothing and writes
# nothing. Homebrew is not required: it is consulted only to name the directory
# this script must never write into.
BIN_DIR="${ZTOOLS_INSTALL_DIR:-$HOME/.local/bin}"
if command -v brew >/dev/null 2>&1; then
    BREW_BIN="$(brew --prefix)/bin"
    _resolved="$(cd "$BIN_DIR" 2>/dev/null && pwd -P || printf '%s' "$BIN_DIR")"
    _brew_resolved="$(cd "$BREW_BIN" 2>/dev/null && pwd -P || printf '%s' "$BREW_BIN")"
    if [ "$_resolved" = "$_brew_resolved" ]; then
        die "refusing to install into the Homebrew bin ($BREW_BIN): the tap formula owns it.
  Use \`brew install ztomer/tap/ztools\` for that path, or leave ZTOOLS_INSTALL_DIR unset (~/.local/bin)."
    fi
    unset _resolved _brew_resolved
fi

info "Building ztools native Rust release binary..."
(cd "$ROOT/rust" && cargo build --release)

# Resolve the cargo target directory — ASKED OF CARGO, never assumed.
#
# This used to end in `|| echo "$HOME/.cache/cargo-target"`, which failed twice
# over. `set -e` cannot see a failure inside `$( ... || ... )`, so the fallback was
# the only reachable outcome on error; and the path it named is one cargo has not
# written to in months, because ~/.cargo/config.toml sets a machine-wide
# `build-dir` namespaced per workspace by path hash. So the failure pointed at a
# directory that cannot hold the artifact and the next check reported "binary not
# found" for a build that had never been asked for. Loud and named beats pointing
# at the wrong place: `jq -e` also fails on a missing/null key, which `-r` alone
# would have passed straight through as the string "null".
if ! TARGET_DIR="$(cargo metadata --manifest-path "$ROOT/rust/Cargo.toml" \
        --format-version 1 --no-deps 2>/dev/null | jq -er '.target_directory')"; then
    die "cannot resolve the cargo target directory for $ROOT/rust/Cargo.toml (cargo metadata failed, or jq could not read .target_directory).
  Refusing to guess: ~/.cargo/config.toml sets a machine-wide build-dir namespaced
  per workspace by path hash, so no fixed path is right."
fi

RELEASE_BIN="$TARGET_DIR/release/ztools"

if [ ! -f "$RELEASE_BIN" ]; then
    die "release binary not found at $RELEASE_BIN (cargo reported that as the target directory)"
fi

info "Installing binaries to $BIN_DIR ..."
mkdir -p "$BIN_DIR"

# Copy main binary. `cp` onto an existing path writes THROUGH a symlink (the
# Homebrew prefix entry points into a Cellar dir), so stage + `mv` in the same
# directory — mv replaces the link itself, atomic on the same filesystem, and
# independent of whether the Cellar is writable.
TMPBIN="$BIN_DIR/.ztools.$$"
cp "$RELEASE_BIN" "$TMPBIN"
mv "$TMPBIN" "$BIN_DIR/ztools"
chmod +x "$BIN_DIR/ztools"

# Create symlinks for all subcommands
for cmd in twitter \
           twitter-summarize \
           weekend \
           weekend-plan \
           rename_images \
           image-renamer \
           oeval \
           model-eval; do
    ln -sf ztools "$BIN_DIR/$cmd"
done

# An earlier install.sh copied bin/ab_test here too. It is a dev harness that
# needs the checkout (rust/, tests/fixtures/) and was broken at this path;
# remove the stale copy if one is present.
[ -f "$BIN_DIR/ab_test" ] && rm -f "$BIN_DIR/ab_test"

ok "ztools binaries successfully installed to $BIN_DIR"