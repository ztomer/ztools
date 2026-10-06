#!/usr/bin/env bash
# install.sh — Build and install ztools release binaries into the Homebrew bin
#
# The Homebrew tap (maintained by tools/release.sh) is the public install
# path; this is the LOCAL door: build the Rust release here and install it
# straight into "$(brew --prefix)/bin" (Apple silicon: /opt/homebrew/bin),
# no network, no tap. Installing over the prefix entry that the tap formula
# also populates is intentional until the next brew upgrade.
#
# ENVIRONMENT
#   ZTOOLS_INSTALL_DIR         target dir; default "$(brew --prefix)/bin"
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
# of both. Same rule tools/gpu_lock.sh and tools/release.sh state: a guard that
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
# a warn-and-build-anyway `case` fallback — that silently ships a binary nobody
# tested on this platform, which is exactly the class this gate exists to stop.
# "Drop Intel" is macOS-specific: the Linux x86_64 path is still supported, so no
# x86_64 code path may be deleted without checking that first.
SUPPORTED_PLATFORMS="macOS/arm64, Linux/x86_64, Linux/aarch64"

# `uname`'s own spellings, so the failure names what was found rather than what we
# normalised it to. ${VAR:-$(...)} evaluates the substitution only when VAR is
# unset, so the seam costs no uname call.
UNAME_OS="${ZTOOLS_OS:-$(uname -s)}"
UNAME_ARCH="${ZTOOLS_ARCH:-$(uname -m)}"

# Normalise to ONE spelling per platform, so the accept set below is exactly three
# pairs rather than a growing list of aliases: macOS reports the silicon as arm64
# and Linux as aarch64; both call x86-64 x86_64, but Linux may say amd64.
case "$UNAME_OS" in
    Darwin) OS="macOS" ;;
    Linux)  OS="Linux" ;;
    *)      OS="$UNAME_OS" ;;
esac
case "$UNAME_ARCH" in
    x86_64 | amd64) ARCH="x86_64" ;;
    arm64 | aarch64)
        # Same silicon, two names. Keep the macOS spelling on macOS so the pair
        # printed by the gate is the pair an operator would recognise.
        if [ "$OS" = "macOS" ]; then ARCH="arm64"; else ARCH="aarch64"; fi
        ;;
    *) ARCH="$UNAME_ARCH" ;;
esac

# 64-bit first, on every OS: an arch that is not one of the three 64-bit names is
# never a warning, whatever the OS says. A 32-bit build has never been compiled,
# run or tested here.
case "$ARCH" in
    arm64 | x86_64 | aarch64) ;;
    *)
        die "unsupported platform: ${OS}/${ARCH} (uname reported ${UNAME_OS}/${UNAME_ARCH}).
  64-bit only — arm64, aarch64 or x86_64; 32-bit is never supported.
  Supported: ${SUPPORTED_PLATFORMS}."
        ;;
esac

case "$OS" in
    macOS)
        if [ "$ARCH" != "arm64" ]; then
            die "unsupported platform: ${OS}/${ARCH} (uname reported ${UNAME_OS}/${UNAME_ARCH}).
  macOS is Apple silicon ONLY — Intel Macs are dropped.
  Linux x86_64 and Linux aarch64 remain supported.
  Supported: ${SUPPORTED_PLATFORMS}."
        fi
        ;;
    Linux) ;;
    *)
        die "unsupported platform: ${OS}/${ARCH} (uname reported ${UNAME_OS}/${UNAME_ARCH}).
  Unsupported OS — macOS (arm64) or Linux (x86_64/aarch64) only.
  Supported: ${SUPPORTED_PLATFORMS}."
        ;;
esac

# Test seam: stop HERE, after the gate and before any build, install or command
# requirement — so both directions are testable on one machine (reject for every
# unsupported pair, accept for each supported one) and no test can ever install a
# binary as a side effect of checking the gate.
if [ "${ZTOOLS_PLATFORM_CHECK_ONLY:-}" = "1" ]; then
    ok "gate ok: ${OS}/${ARCH}"
    exit 0
fi

# ── Dependencies ───────────────────────────────────────────────────────────────
# Named, checked, and BEFORE the first use — including before "$(brew --prefix)",
# which used to be called at the top of the file: on Linux with no Homebrew the
# script died with "brew: command not found", naming a missing package rather than
# the unsupported platform it was a symptom of.
require_commands cargo jq
if [ "$OS" = "macOS" ]; then
    require_commands brew
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# brew --prefix is only reached once brew is known to exist, on the platform whose
# gate accepts the Homebrew layout at all.
BIN_DIR="${ZTOOLS_INSTALL_DIR:-$(brew --prefix)/bin}"

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