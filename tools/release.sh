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
# both: the operator is told the release failed and not why. Same rule tools/gpu_lock.sh
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
CURRENT_V=$(grep -m1 '^version = ' rust/Cargo.toml | cut -d'"' -f2)
if [ "$CURRENT_V" != "$V" ]; then
  info "rust/Cargo.toml version: $CURRENT_V → $V"
  /usr/bin/sed -i '' "s|^version = \".*\"|version = \"$V\"|" rust/Cargo.toml
  cargo update --manifest-path rust/Cargo.toml -p ztools --quiet
  git add rust/Cargo.toml rust/Cargo.lock
  git commit -q -m "chore: version $V" --no-verify
fi

# Install, then prove the binary on PATH is this version and that every
# subcommand answers. bin/ab_test is the smoke harness over the INSTALLED
# binary (it also runs the Rust suite and the collect-parity comparator).
# (Single-quoted on purpose: the kit runs it, exporting GOH_RELEASE_VERSION.)
# shellcheck disable=SC2016
VERIFY='./install.sh >/dev/null
  [ "$(ztools --version | awk "{print \$NF}")" = "$GOH_RELEASE_VERSION" ] || { echo "installed ztools is not $GOH_RELEASE_VERSION" >&2; exit 1; }
  bash bin/ab_test >/dev/null'

exec "$GOH/tools/release-kit/release.sh" --version "$V" \
  --gate "make ci" \
  --archive-build "cargo build --release --quiet --manifest-path rust/Cargo.toml" \
  --verify "$VERIFY" \
  --tap ztomer/homebrew-tap --formula ztools \
  "$@"
