"""CHANGELOG.md's heading contract, as a gate.

The release kit finds a version's notes by its `## v<version>` heading. A stanza
whose heading lost its `## v...` prefix (v3.12.0 carried " Non-interactive
Camoufox streaming fetch _(2026-11-07)_" as a bare line) is invisible to it: the
text is folded into the stanza above and no version owns it. The kit only warns
about tags with no stanza; it cannot see a heading that does not parse at all.

Rules, each run against the real file AND a synthetic violation it must reject:
  1. a dated title line (ending `_(YYYY-MM-DD)_`) is a heading: `## v<semver> — `
     for a release, or `### ` for a dated section inside one;
  2. every `## ` heading is a release heading `## v<semver>`, bare (the shape the house
     release kit's changelog_cut.py writes when it cuts the stanza) or followed by
     ` — <title>` / ` -- <title>`, or the ONE `## Unreleased` / `## [Unreleased]`
     section, above every release: entries accumulate there and the kit cuts the
     release stanza FROM it (goh GOH.CUTAUTO), as in gates_of_heck's own CHANGELOG;
  3. release versions are unique and strictly descending.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHANGELOG = ROOT / "CHANGELOG.md"

DATED = re.compile(r"_\(\d{4}-\d{2}-\d{2}\)_\s*$")
RELEASE = re.compile(r"^## v(\d+)\.(\d+)(?:\.(\d+))?(?:$| (?:—|--) \S.*$)")
UNRELEASED = re.compile(r"^## \[?Unreleased\]?$", re.IGNORECASE)


def problems(text: str) -> list[str]:
    out: list[str] = []
    versions: list[tuple[int, int, int]] = []
    unreleased = False
    fence = False
    for n, line in enumerate(text.splitlines(), 1):
        if line.startswith("```"):
            fence = not fence
        if fence:
            continue
        m = RELEASE.match(line)
        if UNRELEASED.match(line):
            if versions or unreleased:
                out.append(f"line {n}: Unreleased must be the one section above every release")
            unreleased = True
        elif m:
            versions.append((int(m[1]), int(m[2]), int(m[3] or 0)))
        elif line.startswith("## "):
            out.append(f"line {n}: '## ' heading is not '## v<version>[ — <title>]': {line!r}")
        elif DATED.search(line) and not line.startswith("### "):
            out.append(
                f"line {n}: dated title is not a heading (missing '## v<version> — '?): {line!r}"
            )
    if not versions:
        out.append("no release headings found")
    for a, b in itertools.pairwise(versions):
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


def test_an_unreleased_section_above_the_releases_passes() -> None:
    assert problems(GOOD.replace("## v1.1.0", "## Unreleased\n\n* new\n\n## v1.1.0")) == []
    assert problems(GOOD.replace("## v1.1.0", "## [Unreleased]\n\n## v1.1.0")) == []


def test_the_bare_heading_the_release_kit_cuts_passes() -> None:
    assert problems(GOOD.replace("## v1.1.0 — b _(2026-01-02)_", "## v1.1.0")) == []


def test_unreleased_below_a_release_or_twice_is_rejected() -> None:
    below = GOOD.replace("## v1.0.0", "## Unreleased\n\n## v1.0.0")
    assert any("Unreleased must be" in p for p in problems(below))
    twice = GOOD.replace("## v1.1.0", "## Unreleased\n\n## Unreleased\n\n## v1.1.0")
    assert any("Unreleased must be" in p for p in problems(twice))
