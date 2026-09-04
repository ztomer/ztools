#!/usr/bin/env bash
# Release ztools: sync version, tag, push, update Homebrew tap via gh.
# Usage:  tools/release.sh            # bump patch from the latest tag
#         tools/release.sh 1.0.0      # explicit version (minor/major bump)
# Requires: gh (authenticated), git, curl, shasum
set -euo pipefail

SELF="${BASH_SOURCE[0]:-$0}"
REPO_ROOT="$(cd "$(dirname "$SELF")/.." && pwd -P)"
cd "$REPO_ROOT"

ORG="ztomer"
TAP_REPO="${ORG}/homebrew-tap"
REMOTE="origin"

info()  { echo "→ $*"; }
ok()    { echo "✓ $*"; }
err()   { echo "✗ $*" >&2; exit 1; }

# ── 1. Pick the new version ─────────────────────────────────────────
# Default: bump the patch of the latest tag. Override with an explicit version
# for a minor/major bump:  tools/release.sh 1.0.0
LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
if [ $# -ge 1 ]; then
  V="${1#v}"
  case "$V" in
    [0-9]*.[0-9]*.[0-9]*) ;;
    *) err "Version must look like MAJOR.MINOR.PATCH (got '$1')" ;;
  esac
  NEW_TAG="v$V"
  if git rev-parse "$NEW_TAG" >/dev/null 2>&1; then
    err "Tag $NEW_TAG already exists"
  fi
else
  VERSION="${LAST_TAG#v}"
  MAJOR="${VERSION%%.*}"; REST="${VERSION#*.}"
  MINOR="${REST%%.*}";    PATCH="${REST#*.}"
  NEW_TAG="v${MAJOR}.${MINOR}.$((PATCH + 1))"
  V="${NEW_TAG#v}"
fi

info "Last tag: $LAST_TAG → new tag: $NEW_TAG"

# ── 2. Check for unpushed commits ──────────────────────────────────
UNPUSHED=$(git log --oneline "$REMOTE/main..HEAD" 2>/dev/null | wc -l | tr -d ' ')
if [ "$UNPUSHED" -eq 0 ]; then
  err "Nothing to release — no unpushed commits"
fi
if ! git diff --quiet; then
  err "Working tree is dirty — commit or stash before releasing (the tag would
    point at HEAD, silently shipping something different from what you tested)"
fi

# ── 2b. Run the gate ───────────────────────────────────────────────
# This script pushes with --no-verify, which SKIPS the pre-push hook -- so
# without this step nothing checked a release at all. The comment above about
# "silently shipping something different from what you tested" described
# exactly what happened next: the tag went out ungated.
#
# `make ci` is the gate of record and reads its step list from .gatesrc, the
# same list `tools/gate.sh --full` runs, so this cannot drift from the hook it
# stands in for.
info "Running the full gate (make ci) before tagging …"
make ci >/dev/null || err "gate failed — nothing tagged, nothing pushed"
ok "Gate green"

# ── 3. Sync BOTH manifests to $V, then tag ────────────────────────
# pyproject.toml drives the sdist version; rust/Cargo.toml drives the binary
# and Cargo.lock. At v2.1.5 only pyproject moved and the binary shipped
# reporting 2.1.4 -- two sources of truth that were allowed to disagree.
SYNCED=0
for manifest in pyproject.toml rust/Cargo.toml; do
  CURRENT_V=$(grep -m1 '^version = ' "$manifest" | cut -d'"' -f2)
  if [ "$CURRENT_V" != "$V" ]; then
    info "Syncing $manifest version: $CURRENT_V → $V"
    sed -i '' "s|^version = \".*\"|version = \"$V\"|" "$manifest"
    git add "$manifest"
    SYNCED=1
  else
    info "$manifest already at $V"
  fi
done
if [ "$SYNCED" -eq 1 ]; then
  # Cargo.lock carries the crate's own version too.
  if [ -f rust/Cargo.lock ] && grep -q 'name = "ztools"' rust/Cargo.lock; then
    git add rust/Cargo.lock || true
  fi
  git commit -m "chore: version $V" --no-verify
fi

info "Tagging HEAD as $NEW_TAG ..."
git tag -a "$NEW_TAG" -m "Release $NEW_TAG" --no-sign

# ── 4. Push commits + tag ──────────────────────────────────────────
info "Pushing commits and tag (--no-verify) …"
git push --no-verify "$REMOTE" main
git push --no-verify "$REMOTE" "$NEW_TAG"
ok "Pushed $NEW_TAG"

# ── 5. Compute SHA256 ──────────────────────────────────────────────
TARBALL_URL="https://github.com/${ORG}/ztools/archive/refs/tags/$NEW_TAG.tar.gz"
info "Fetching tarball to compute SHA256 …"
SHA=$(curl -sL "$TARBALL_URL" | shasum -a 256 | cut -d' ' -f1)
info "SHA256: $SHA"

# ── 6. Update Homebrew tap via gh ──────────────────────────────────
info "Cloning $TAP_REPO via gh …"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

gh repo clone "$TAP_REPO" "$TMPDIR/tap" 2>/dev/null

FORMULA="$TMPDIR/tap/Formula/ztools.rb"
if [ ! -f "$FORMULA" ]; then
  err "Formula not found at $FORMULA"
fi

# Update version in URL and SHA
sed -i '' \
  -e "s|/v[0-9.]*\.tar\.gz|/$NEW_TAG.tar.gz|" \
  -e "s|sha256 \".*\"|sha256 \"$SHA\"|" \
  "$FORMULA"

(cd "$TMPDIR/tap" && git add -A && git commit -m "ztools $NEW_TAG" && git push)
ok "Homebrew tap updated: $TAP_REPO"

ok "Release $NEW_TAG complete"
