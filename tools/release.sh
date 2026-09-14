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
# shellcheck source=/Users/ztomer/Projects/gates_of_heck/tui/lib.sh
source "$GOH/tui/lib.sh"

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
