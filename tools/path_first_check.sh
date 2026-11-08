#!/usr/bin/env bash
# path_first_check.sh VERSION -- prove a bare `ztools` runs the release (ZT.SHADOW).
#
# Fails unless exactly ONE ztools is on PATH and it reports VERSION. Two installs
# (install.sh's ~/.local/bin and the tap's brew bin) shadow each other silently:
# whichever comes first in PATH wins, and brew only warns. Every copy found is
# named with its version, so the failure says which one to delete.
set -euo pipefail
GOH="${GOH_DIR:-${HOME:-}/Projects/gates_of_heck}"
for _h in info ok warn err; do
    declare -F "$_h" >/dev/null 2>&1 || eval "$_h() { printf '%s\n' \"\$*\" >&2; }"
done
declare -F die >/dev/null 2>&1 || die() { err "$*"; exit "${2:-1}"; }
# shellcheck source=/dev/null
[ -f "$GOH/tui/lib.sh" ] && source "$GOH/tui/lib.sh"
unset _h

[ $# -eq 1 ] || die "usage: tools/path_first_check.sh X.Y.Z"
WANT="${1#v}"

found=()
seen=":"
IFS=: read -r -a _dirs <<<"$PATH"
for d in "${_dirs[@]}"; do
    [ -n "$d" ] && [ -x "$d/ztools" ] && [ ! -d "$d/ztools" ] || continue
    real="$(cd "$d" && pwd -P)/ztools"
    case "$seen" in *":$real:"*) continue ;; esac
    seen="$seen$real:"
    found+=("$d/ztools")
done
[ "${#found[@]}" -gt 0 ] || die "no ztools on PATH; expected $WANT."

listing=""
for b in "${found[@]}"; do
    v="$("$b" --version 2>/dev/null | awk '{print $NF}')" || v="?"
    listing="$listing
  $b  ${v:-?}"
done
first_v="$("${found[0]}" --version 2>/dev/null | awk '{print $NF}')" || first_v="?"
if [ "$first_v" != "$WANT" ] || [ "${#found[@]}" -gt 1 ]; then
    die "a bare \`ztools\` does not run exactly the release $WANT; on PATH, first wins:$listing
  Delete every copy but the tap's (brew upgrade ztools)."
fi
ok "the only ztools on PATH is ${found[0]} ($WANT)"
