#!/usr/bin/env bash
set -euo pipefail

# Print helper functions
print_info() { echo "[ ==> ] $1"; }
print_ok()   { echo "[ Ok  ] $1"; }

print_info "Building the ztools native Rust crate..."
(cd "$HOME/Projects/ztools/rust" && cargo build --release)

print_info "Creating binary launchers in ~/Projects/ztools/bin ..."
mkdir -p "$HOME/Projects/ztools/bin"

# Shared launcher body. Resolution order: a real `ztools` binary on PATH first
# (so this works on any machine), then the crate's build output. A PATH entry
# that is itself a launcher script — this very shim, which is what `bin/` puts
# on PATH — is skipped, or every invocation would recurse into itself. The
# target dir is resolved via cargo metadata rather than hardcoded: the
# machine-wide CARGO_TARGET_DIR is config-dependent, and a stale literal path
# is exactly how a fixed binary keeps not being the one that runs.
LAUNCHER='#!/usr/bin/env bash
# Thin shim onto the native Rust `ztools` binary. Resolves the crate build
# output via cargo metadata first (so local workspace changes take effect
# immediately), then falls back to a real `ztools` binary on PATH.
set -euo pipefail

TARGET_DIR="$(cargo metadata --manifest-path "$HOME/Projects/ztools/rust/Cargo.toml" --format-version 1 --no-deps 2>/dev/null | jq -r '"'"'.target_directory'"'"' 2>/dev/null)"
for variant in release debug; do
  BIN="$TARGET_DIR/$variant/ztools"
  if [ -x "$BIN" ]; then
    exec "$BIN" %SUB_OR_ARGS%
  fi
done
FOUND="$(command -v ztools 2>/dev/null || true)"
if [ -n "$FOUND" ] && [ "$(head -c 2 "$FOUND" 2>/dev/null)" != '"'"'#!'"'"' ]; then
  exec ztools %SUB_OR_ARGS%
fi
echo "✗ ztools binary not found in target dir $TARGET_DIR or on PATH" >&2
exit 1
'

# The `ztools` shim forwards arguments straight through; the others fix the
# subcommand and forward the rest. Format is `name:args`, where args is the
# literal argument list written into the launcher body.
for entry in 'ztools:"$@"' \
             'twitter-summarize:"twitter-summarize" "$@"' \
             'twitter:"twitter-summarize" "$@"' \
             'weekend-plan:"weekend-plan" "$@"' \
             'weekend:"weekend-plan" "$@"' \
             'image-renamer:"image-renamer" "$@"' \
             'rename_images:"image-renamer" "$@"' \
             'model-eval:"model-eval" "$@"' \
             'oeval:"model-eval" "$@"'; do
  IFS=':' read -r name args <<< "$entry"
  body="${LAUNCHER//%SUB_OR_ARGS%/$args}"
  printf '%s' "$body" > "$HOME/Projects/ztools/bin/$name"
  chmod +x "$HOME/Projects/ztools/bin/$name"
done

print_ok "ztools build complete! All native Rust binaries ready in ~/Projects/ztools/bin."