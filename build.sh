#!/usr/bin/env bash
set -euo pipefail

# Print helper functions
print_info() { echo "[ ==> ] $1"; }
print_ok()   { echo "[ Ok  ] $1"; }

print_info "Building ztools native Rust suite (via routines workspace)..."
(cd "$HOME/Projects/routines" && ./build.sh)

print_info "Creating binary launchers in ~/Projects/ztools/bin ..."
mkdir -p "$HOME/Projects/ztools/bin"

cat << 'EOF' > "$HOME/Projects/ztools/bin/twitter-summarize"
#!/usr/bin/env bash
exec "$HOME/Projects/routines/target/release/routines" twitter-summarize "$@"
EOF
chmod +x "$HOME/Projects/ztools/bin/twitter-summarize"

cat << 'EOF' > "$HOME/Projects/ztools/bin/weekend-plan"
#!/usr/bin/env bash
exec "$HOME/Projects/routines/target/release/routines" weekend-plan "$@"
EOF
chmod +x "$HOME/Projects/ztools/bin/weekend-plan"

cat << 'EOF' > "$HOME/Projects/ztools/bin/image-renamer"
#!/usr/bin/env bash
exec "$HOME/Projects/routines/target/release/routines" image-renamer "$@"
EOF
chmod +x "$HOME/Projects/ztools/bin/image-renamer"

cat << 'EOF' > "$HOME/Projects/ztools/bin/model-eval"
#!/usr/bin/env bash
exec "$HOME/Projects/routines/target/release/routines" model-eval "$@"
EOF
chmod +x "$HOME/Projects/ztools/bin/model-eval"

print_ok "ztools build complete! All native Rust binaries ready in ~/Projects/ztools/bin."
