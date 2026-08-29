#!/usr/bin/env bash
# install.sh — Build and install ztools release binaries into the Homebrew bin
#
# The Homebrew tap (maintained by tools/release.sh) is the public install
# path; this is the LOCAL door: build the Rust release here and install it
# straight into "$(brew --prefix)/bin" (Apple silicon: /opt/homebrew/bin),
# no network, no tap. Installing over the prefix entry that the tap formula
# also populates is intentional until the next brew upgrade. ZTOOLS_INSTALL_DIR
# overrides the target dir.
set -euo pipefail

GOH="${GOH_DIR:-$HOME/Projects/gates_of_heck}"
# shellcheck source=/Users/ztomer/Projects/gates_of_heck/tui/lib.sh
source "$GOH/tui/lib.sh"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BIN_DIR="${ZTOOLS_INSTALL_DIR:-$(brew --prefix)/bin}"

info "Building ztools native Rust release binary..."
(cd "$ROOT/rust" && cargo build --release)

# Resolve the cargo target directory
TARGET_DIR="$(cargo metadata --manifest-path "$ROOT/rust/Cargo.toml" --format-version 1 --no-deps 2>/dev/null \
    | jq -r '.target_directory' 2>/dev/null || echo "$HOME/.cache/cargo-target")"

RELEASE_BIN="$TARGET_DIR/release/ztools"

if [ ! -f "$RELEASE_BIN" ]; then
    err "Release binary not found at $RELEASE_BIN"
    exit 1
fi

info "Installing binaries to $BIN_DIR ..."
mkdir -p "$BIN_DIR"

# Copy main binary
cp "$RELEASE_BIN" "$BIN_DIR/ztools"
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

# Copy ab_test helper if present
if [ -f "$ROOT/bin/ab_test" ]; then
    cp "$ROOT/bin/ab_test" "$BIN_DIR/ab_test"
    chmod +x "$BIN_DIR/ab_test"
fi

ok "ztools binaries successfully installed to $BIN_DIR"
