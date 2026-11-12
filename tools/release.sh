#!/usr/bin/env bash
# Release ztools through the house release kit. This file holds only what is
# specific to this repo -- the ONE version source, the archive build, the
# install-and-exercise check, and the tap -- and none of the sequencing (gate,
# changelog stanza, tag, push, GitHub release, tap bump), which lives once in
# gates_of_heck/tools/release-kit/release.sh for every repo.
#
#   tools/release.sh 3.1.0
#   tools/release.sh 3.1.0 --dry-run
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
GOH="${GOH_DIR:-$HOME/Projects/gates_of_heck}"
# The output helpers, defined BEFORE the source so a missing tui degrades instead of
# dying. Sourcing an absent file prints "No such file or directory" and, under
# `set -e`, exits with nothing named -- fail-closed but unnamed, which is the worst of
# both: the operator is told the release failed and not why. Same rule the house `gpu_lock.sh`
# states -- a guard that only returns non-zero is not enough, because `return` from a
# sourced file returns from the SOURCE and the caller carries on -- so the helpers are
# DEFINED here, not merely checked for.
for _release_helper in info ok warn err; do
    declare -F "$_release_helper" >/dev/null 2>&1 || eval "
        $_release_helper() { printf '%s\n' \"\$*\" >&2; }"
done
declare -F die >/dev/null 2>&1 || die() { err "$*"; exit "${2:-1}"; }
if [ -f "$GOH/tui/lib.sh" ]; then
    # shellcheck source=/dev/null
    source "$GOH/tui/lib.sh"
else
    warn "no tui/lib.sh under $GOH — output is plain text, no icons or colour."
    warn "set GOH_DIR to your gates_of_heck checkout (or run its install.sh) for the house TUI style."
fi
unset _release_helper

[ $# -ge 1 ] || die "usage: tools/release.sh X.Y.Z [--dry-run]"
V="${1#v}"; shift

# rust/Cargo.toml is the ONE version source (a pyproject once disagreed with
# it and the binary shipped reporting the previous version). Sync it before
# the kit runs the gate, so the gate and the archive build see the release.
#
# --dry-run is the kit's contract ("performs NOTHING"), and this bump runs BEFORE the
# kit is exec'd, so it must honour the flag itself: a dry run once left a real
# "chore: version" commit behind while reporting a plan (tools/tests/test_release_dry_run.py).
DRY_RUN=0
for _arg in "$@"; do [ "$_arg" = "--dry-run" ] && DRY_RUN=1; done
unset _arg
CURRENT_V=$(grep -m1 '^version = ' rust/Cargo.toml | cut -d'"' -f2)
if [ "$CURRENT_V" != "$V" ] && [ "$DRY_RUN" = 1 ]; then
  info "[dry-run] would bump rust/Cargo.toml version: $CURRENT_V → $V and commit 'chore: version $V'"
  # The kit's first step asks that HEAD already DECLARE the version it tags, and a dry run
  # applies no bump -- so a preview of any real release failed there, every time
  # (ZT.TOOLFIX). The preview therefore runs the kit from a throwaway detached worktree that
  # carries the bump as a commit; this repo's HEAD, index, manifest and refs are untouched,
  # and the worktree is removed on every exit path. The commit is disposable and never
  # tagged, so it skips the hooks (the real bump below does not: U4).
  PREVIEW_WT="$(mktemp -d "${TMPDIR:-/tmp}/ztools-release-preview.XXXXXX")"
  REAL_TOP="$PWD"
  preview_cleanup() {
    cd "$REAL_TOP" 2>/dev/null || true
    git worktree remove --force "$PREVIEW_WT" >/dev/null 2>&1 || rm -rf "$PREVIEW_WT"
    git worktree prune >/dev/null 2>&1 || true
  }
  trap preview_cleanup EXIT
  trap 'exit 130' INT TERM
  { git worktree add --detach -q "$PREVIEW_WT" HEAD \
      && cd "$PREVIEW_WT" \
      && /usr/bin/sed -i '' "s|^version = \".*\"|version = \"$V\"|" rust/Cargo.toml \
      && git add -- rust/Cargo.toml \
      && git commit -q --no-verify -m "chore: version $V"; } \
    || die "could not prepare the dry-run preview worktree for $V"
elif [ "$CURRENT_V" != "$V" ]; then
  # A bump that fails part-way must leave the manifest, the lock and the index as it found
  # them (ZT.RESIDFLAKE): a pre-commit hook once refused the commit and the bump stayed STAGED
  # for the operator to discard by hand. Restoring both files to HEAD is only safe when they
  # carry nothing else, so local edits to either are refused before anything is touched.
  BUMPED=(rust/Cargo.toml rust/Cargo.lock)
  git diff --quiet HEAD -- "${BUMPED[@]}" \
    || die "rust/Cargo.toml or rust/Cargo.lock has uncommitted changes; commit or drop them before a version bump"
  restore_bump() { git restore --source=HEAD --staged --worktree -- "${BUMPED[@]}"; }
  trap 'restore_bump; exit 130' INT TERM
  info "rust/Cargo.toml version: $CURRENT_V → $V"
  # No --no-verify (U4). This is the commit every release tags, and it is the one place a
  # staged-scope check is exact: the pre-commit gate reads the index, so it sees the version
  # bump as staged. Bypassing it meant the tag could point at a commit no hook ever looked at.
  { /usr/bin/sed -i '' "s|^version = \".*\"|version = \"$V\"|" rust/Cargo.toml \
      && cargo update --manifest-path rust/Cargo.toml -p ztools --quiet \
      && git add -- "${BUMPED[@]}" \
      && git commit -q -m "chore: version $V"; } \
    || { restore_bump; die "the version bump to $V was not committed (above); rust/Cargo.toml, rust/Cargo.lock and the index are as they were"; }
  trap - INT TERM
fi

# Install, then prove the binary on PATH is this version and that every
# subcommand answers. bin/ab_test is the smoke harness over the INSTALLED
# binary (it also runs the Rust suite and the collect-parity comparator). Neither step's
# output is discarded (ZTL.RELQUIET): install.sh writes a per-run log whose path is printed,
# and ab_test prints its own suite log's path and names any failing test on stderr.
# (Single-quoted on purpose: the kit runs it, exporting GOH_RELEASE_VERSION.)
# shellcheck disable=SC2016
# The binary install.sh just wrote, by path: install.sh never touches the brew bin
# (ZT.BREWLINK), so a bare `ztools` could resolve to the tap's older copy. The temp
# dir is on no PATH, so --dev-copy shadows nothing (ZT.SHADOW).
VERIFY='ZT_DIR="$(mktemp -d)"; LOG="$(mktemp "${TMPDIR:-/tmp}/ztools-release-install.XXXXXX")"
  echo "install log: $LOG"
  ZTOOLS_INSTALL_DIR="$ZT_DIR" ./install.sh --dev-copy >"$LOG" 2>&1 || { echo "install.sh --dev-copy failed (full log: $LOG); its last 40 lines:" >&2; tail -n 40 "$LOG" >&2; rm -rf "$ZT_DIR"; exit 1; }
  [ "$("$ZT_DIR/ztools" --version | awk "{print \$NF}")" = "$GOH_RELEASE_VERSION" ] || { echo "installed ztools is not $GOH_RELEASE_VERSION" >&2; rm -rf "$ZT_DIR"; exit 1; }
  ZTOOLS_BIN="$ZT_DIR/ztools" bash bin/ab_test; rc=$?; rm -rf "$ZT_DIR"; exit $rc'

"$GOH/tools/release-kit/release.sh" --version "$V" \
  --gate "make ci" \
  --archive-build "cargo build --release --quiet --manifest-path rust/Cargo.toml" \
  --verify "$VERIFY" \
  --tap ztomer/homebrew-tap --formula ztools \
  "$@"

# ZT.SHADOW: the release is done only when a bare `ztools` runs it. A stale
# ~/.local/bin copy once came first in PATH after the tap shipped; brew only warned.
if [ "$DRY_RUN" = 1 ]; then
  info "[dry-run] would brew upgrade ztomer/tap/ztools and check the first ztools on PATH is $V"
  exit 0
fi
brew update --quiet
brew upgrade ztomer/tap/ztools || brew install ztomer/tap/ztools
tools/path_first_check.sh "$V"
