"""CHANGELOG.md's heading contract, as a gate.

The release kit finds a version's notes by its `## v<version>` heading. A stanza
whose heading lost its `## v...` prefix (v3.12.0 carried " Non-interactive
Camoufox streaming fetch _(2026-11-07)_" as a bare line) is invisible to it: the
text is folded into the stanza above and no version owns it. The kit only warns
about tags with no stanza; it cannot see a heading that does not parse at all.

Rules, each run against the real file AND a synthetic violation it must reject:
  1. a dated title line (ending `_(YYYY-MM-DD)_`) is a heading: `## v<semver> — `
     for a release, or `### ` for a dated section inside one;
  2. every `## ` heading is a release heading of that shape;
  3. release versions are unique and strictly descending.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHANGELOG = ROOT / "CHANGELOG.md"

DATED = re.compile(r"_\(\d{4}-\d{2}-\d{2}\)_\s*$")
RELEASE = re.compile(r"^## v(\d+)\.(\d+)(?:\.(\d+))? — \S.* _\(\d{4}-\d{2}-\d{2}\)_$")


def problems(text: str) -> list[str]:
    out: list[str] = []
    versions: list[tuple[int, int, int]] = []
    fence = False
    for n, line in enumerate(text.splitlines(), 1):
        if line.startswith("```"):
            fence = not fence
        if fence:
            continue
        m = RELEASE.match(line)
        if m:
            versions.append((int(m[1]), int(m[2]), int(m[3] or 0)))
        elif line.startswith("## "):
            out.append(
                f"line {n}: '## ' heading is not '## v<version> — <title> _(date)_': {line!r}"
            )
        elif DATED.search(line) and not line.startswith("### "):
            out.append(
                f"line {n}: dated title is not a heading (missing '## v<version> — '?): {line!r}"
            )
    if not versions:
        out.append("no release headings found")
    for a, b in zip(versions, versions[1:], strict=False):
        if not a > b:
            out.append(f"versions not strictly descending: {a} then {b}")
    return out


def test_real_changelog_headings_are_well_formed() -> None:
    assert problems(CHANGELOG.read_text(encoding="utf-8")) == []


GOOD = (
    "# Changelog\n\n## v1.1.0 — b _(2026-01-02)_\n\n"
    "### Part _(2026-01-02)_\n\n## v1.0.0 — a _(2026-01-01)_\n"
)


def test_well_formed_synthetic_passes() -> None:
    assert problems(GOOD) == []


def test_bare_dated_title_is_rejected() -> None:
    bad = GOOD.replace("### Part", " Part")
    assert any("dated title is not a heading" in p for p in problems(bad))


def test_non_release_h2_is_rejected() -> None:
    bad = GOOD.replace("## v1.0.0 — a", "## 1.0.0 a")
    assert any("'## ' heading" in p for p in problems(bad))


def test_out_of_order_versions_are_rejected() -> None:
    bad = GOOD.replace("v1.0.0", "v1.2.0")
    assert any("descending" in p for p in problems(bad))


def test_empty_file_is_rejected() -> None:
    assert problems("# Changelog\n") == ["no release headings found"]
