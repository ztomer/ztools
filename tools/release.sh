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

GOH="${GOH_DIR:-$HOME/Projects/gates_of_heck}"
# shellcheck source=/Users/ztomer/Projects/gates_of_heck/tui/lib.sh
source "$GOH/tui/lib.sh"

# ── 1. Pick the new version ─────────────────────────────────────────
# Default: bump the patch of the latest tag. Override with an explicit version
# for a minor/major bump:  tools/release.sh 1.0.0
LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
if [ $# -ge 1 ]; then
  V="${1#v}"
  case "$V" in
    [0-9]*.[0-9]*.[0-9]*) ;;
    *) die "Version must look like MAJOR.MINOR.PATCH (got '$1')" ;;
  esac
  NEW_TAG="v$V"
  if git rev-parse "$NEW_TAG" >/dev/null 2>&1; then
    die "Tag $NEW_TAG already exists"
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
  die "Nothing to release — no unpushed commits"
fi
if ! git diff --quiet; then
  die "Working tree is dirty — commit or stash before releasing (the tag would
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
make ci >/dev/null || die "gate failed — nothing tagged, nothing pushed"
ok "Gate green"

# ── 3. Sync the manifest to $V, then tag ──────────────────────────
# rust/Cargo.toml is the ONE version source. There used to be a pyproject.toml
# too (the sdist); at v2.1.5 only that one moved and the binary shipped
# reporting 2.1.4 -- two sources of truth allowed to disagree. If a second
# manifest ever appears, it syncs here or it does not exist.
SYNCED=0
manifest=rust/Cargo.toml
CURRENT_V=$(grep -m1 '^version = ' "$manifest" | cut -d'"' -f2)
if [ "$CURRENT_V" != "$V" ]; then
  info "Syncing $manifest version: $CURRENT_V → $V"
  /usr/bin/sed -i '' "s|^version = \".*\"|version = \"$V\"|" "$manifest"
  git add "$manifest"
  SYNCED=1
else
  info "$manifest already at $V"
fi
if [ "$SYNCED" -eq 1 ]; then
  # Cargo.lock carries the crate's own version too.
  if [ -f rust/Cargo.lock ] && grep -q 'name = "ztools"' rust/Cargo.lock; then
    git add rust/Cargo.lock || true
  fi
  git commit -m "chore: version $V" --no-verify
fi

# ── 3b. Prove the tarball builds, then install and exercise it ─────
# The tap builds `cargo install --path rust` from the GitHub tarball, i.e.
# from TRACKED files only. v2.x tagged a tree whose vendored path dependency
# was gitignored: green locally, unbuildable from the archive. So build from
# `git archive HEAD` — the same bytes the tarball will carry — and refuse to
# tag if that fails. Then install the local build and check the binary on
# PATH answers with the version being tagged: a tag is a claim that this
# exact surface was run, not that it compiled.
ARCHIVE_DIR=$(mktemp -d)
trap 'rm -rf "$ARCHIVE_DIR"' EXIT
info "Building from git archive HEAD (what the tap will see) …"
git archive --format=tar HEAD | tar -x -C "$ARCHIVE_DIR"
(cd "$ARCHIVE_DIR" && cargo build --release --quiet --manifest-path rust/Cargo.toml) \
  || die "the archived tree does not build — a tracked file is missing (vendor/? conf/?); nothing tagged"
ok "Archive builds"

info "Installing the release build locally …"
./install.sh >/dev/null || die "install.sh failed — nothing tagged"
INSTALLED="$(command -v ztools || true)"
[ -n "$INSTALLED" ] || die "ztools is not on PATH after install"
REPORTED="$("$INSTALLED" --version 2>/dev/null | awk '{print $NF}')"
[ "$REPORTED" = "$V" ] || die "installed ztools reports '$REPORTED', expected $V ($INSTALLED)"
for cmd in twitter weekend rename_images oeval; do
  "$cmd" --help >/dev/null 2>&1 || die "$cmd --help failed after install"
done
ok "Installed $INSTALLED reports $V; every subcommand answers"

# ── 3c. The tag carries the CHANGELOG entry as its record ──────────
NOTES="$(awk -v v="## v$V" 'index($0, v)==1 {p=1; next} /^## v/ && p {exit} p' CHANGELOG.md)"
[ -n "$NOTES" ] || die "CHANGELOG.md has no '## v$V' section — write the record before tagging"

info "Tagging HEAD as $NEW_TAG ..."
git tag -a "$NEW_TAG" -F - --no-sign <<EOF_TAG
ztools $NEW_TAG
$NOTES
EOF_TAG

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
trap 'rm -rf "$TMPDIR" "$ARCHIVE_DIR"' EXIT

gh repo clone "$TAP_REPO" "$TMPDIR/tap" 2>/dev/null

FORMULA="$TMPDIR/tap/Formula/ztools.rb"
if [ ! -f "$FORMULA" ]; then
  die "Formula not found at $FORMULA"
fi

# Update version in URL and SHA
/usr/bin/sed -i '' \
  -e "s|/v[0-9.]*\.tar\.gz|/$NEW_TAG.tar.gz|" \
  -e "s|sha256 \".*\"|sha256 \"$SHA\"|" \
  "$FORMULA"

(cd "$TMPDIR/tap" && git add -A && git commit -m "ztools $NEW_TAG" && git push)
ok "Homebrew tap updated: $TAP_REPO"

ok "Release $NEW_TAG complete"
