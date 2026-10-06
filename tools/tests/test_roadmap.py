"""The roadmap's contract, as a gate.

`docs/ROADMAP.md` states four rules at its top. Before this file they were house
rule #13 and good intentions, and the roadmap grew to 575 lines: twenty lines of
open work wrapped in "Closed 2026-..." paragraphs, a resume point naming agent
session IDs, three numbering schemes (1-3, A-C, E-J) with one item number
reused across two of them, and an inbound reference (`MODEL_QUIRKS.md`: "ROADMAP
item 1 re-derives them") pointing at an item that had become something else.

Each rule is a pure function over text, and each is run TWICE: against the real
file, and against a synthetic violation it must reject. The second run is what
keeps a rule from passing by inspecting nothing -- a parser that finds zero
items would otherwise report every item complete.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROADMAP = ROOT / "docs" / "ROADMAP.md"

ITEM = re.compile(r"^### ([A-Z]\d+) — .+$", re.M)
FIELDS = ("**Class:**", "**Why now:**", "**Done when:**", "**Blocked by:**")
# A done item is deleted, never annotated. Matched in item bodies only: the
# contract section quotes the forbidden form in order to forbid it.
ANNOTATED_DONE = re.compile(r"(?i)\b(closed|shipped|landed|done|fixed)\s+(on\s+)?20\d\d-\d\d-\d\d")
# Repo paths worth checking: rooted at a directory this repo has, or a root file.
PATH_ROOTS = ("rust/", "tools/", "docs/", "conf/", "bin/", "vendor/", ".githooks/", ".cargo/")
ROOT_FILES = re.compile(r"^[\w.-]+\.(md|toml|sh|jsonc?|lock)$|^\.gatesrc$")
INBOUND = re.compile(r"ROADMAP(?:\.md)?`?\s+([A-Z]\d+)\b")
STALE_INBOUND = re.compile(r"ROADMAP(?:\.md)?`?\s+(?:item|Phase)\s+\w")


def items(text: str) -> dict[str, str]:
    """Item ID -> body, up to the next item or section heading."""
    out: dict[str, str] = {}
    marks = list(ITEM.finditer(text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end() : end]
        section = re.search(r"^## ", body, re.M)
        if section:
            body = body[: section.start()]
        if m.group(1) in out:
            raise AssertionError(f"item ID {m.group(1)} is defined twice")
        out[m.group(1)] = body
    return out


def missing_fields(text: str) -> list[str]:
    return [
        f"{iid}: no {field}"
        for iid, body in items(text).items()
        for field in FIELDS
        if field not in body
    ]


def done_annotations(text: str) -> list[str]:
    return [
        f"{iid}: '{m.group(0)}' -- delete a done item, its record is CHANGELOG.md"
        for iid, body in items(text).items()
        for m in ANNOTATED_DONE.finditer(body)
    ]


def dangling_paths(text: str, root: Path) -> list[str]:
    bad = []
    for m in re.finditer(r"`([^`\n]+)`(\s*\(new\))?", text):
        token, new = m.group(1).split()[0] if m.group(1).split() else "", m.group(2)
        token = re.sub(r":\d+(:\d+)?$", "", token)
        if new or any(c in token for c in ("*", "<", "$", "...", "…")):
            continue
        if not (token.startswith(PATH_ROOTS) or ROOT_FILES.match(token)):
            continue
        if not (root / token).exists():
            bad.append(token)
    return bad


def unresolved_inbound(defined: set[str], sources: dict[str, str]) -> list[str]:
    bad = []
    for name, text in sources.items():
        for m in INBOUND.finditer(text):
            if m.group(1) not in defined:
                bad.append(f"{name}: ROADMAP {m.group(1)} is not an item")
        for m in STALE_INBOUND.finditer(text):
            bad.append(f"{name}: '{m.group(0)}' -- refer to a roadmap item by its ID")
    return bad


def inbound_sources() -> dict[str, str]:
    globs = [
        "*.md",
        "docs/*.md",
        "rust/src/**/*.rs",
        "tools/**/*.py",
        "tools/**/*.sh",
        "conf/**/*.toml",
        "tools/**/*.jsonc",
    ]
    skip = {ROADMAP, ROOT / "CHANGELOG.md", Path(__file__).resolve()}
    out = {}
    for g in globs:
        for p in ROOT.glob(g):
            if p.resolve() not in skip and ".claude" not in p.parts:
                out[str(p.relative_to(ROOT))] = p.read_text(errors="replace")
    return out


# --- the real file -------------------------------------------------------------


@pytest.fixture(scope="module")
def roadmap() -> str:
    return ROADMAP.read_text()


def test_the_roadmap_has_items_to_check(roadmap: str) -> None:
    # Guard against the vacuous pass: every other rule iterates over items().
    assert len(items(roadmap)) >= 5, "the item parser found almost nothing"


def test_every_item_carries_its_four_fields(roadmap: str) -> None:
    assert missing_fields(roadmap) == []


def test_no_item_is_annotated_done_instead_of_deleted(roadmap: str) -> None:
    assert done_annotations(roadmap) == []


def misplaced_items(text: str) -> list[str]:
    """Items whose ID letter is not their phase's letter.

    A section cut that drops a `## Phase X` heading moves every item under it into the
    phase above, silently: the 2026-10-06 edit that deleted `L1` took `## Phase G` with
    it, and the G items read as Phase L until a human noticed.
    """
    bad, phase = [], None
    for line in text.splitlines():
        if line.startswith("## "):
            m = re.match(r"## Phase ([A-Z]) — ", line)
            phase = m.group(1) if m else None
        m = ITEM.match(line)
        if m and m.group(1)[0] != phase:
            bad.append(f"{m.group(1)} sits under Phase {phase}")
    return bad


def test_every_item_sits_under_its_own_phase(roadmap: str) -> None:
    assert misplaced_items(roadmap) == []


def test_an_item_under_the_wrong_phase_is_named() -> None:
    text = GOOD + "\n### Y1 — stray\n"
    assert misplaced_items(text) == ["Y1 sits under Phase Z"]


def test_every_backticked_repo_path_exists(roadmap: str) -> None:
    assert dangling_paths(roadmap, ROOT) == []


def test_every_reference_into_the_roadmap_resolves(roadmap: str) -> None:
    assert unresolved_inbound(set(items(roadmap)), inbound_sources()) == []


# --- each rule rejects the thing it exists to reject ---------------------------

GOOD = """## Phase Z — fixture
### Z1 — an item
- **Class:** c
- **Why now:** w
- **Done when:** d
- **Blocked by:** nothing
"""


def test_the_rules_accept_a_conforming_item() -> None:
    assert (missing_fields(GOOD), done_annotations(GOOD)) == ([], [])


def test_a_missing_field_is_named() -> None:
    assert missing_fields(GOOD.replace("- **Blocked by:** nothing\n", "")) == [
        "Z1: no **Blocked by:**"
    ]


def test_a_closed_annotation_is_rejected() -> None:
    assert done_annotations(GOOD + "\n**Closed 2026-10-05.** it works now\n")


def test_a_reused_id_is_rejected() -> None:
    with pytest.raises(AssertionError, match="defined twice"):
        items(GOOD + GOOD)


def test_a_dangling_path_is_named_and_a_new_one_is_allowed(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "real.md").write_text("x")
    text = "`docs/real.md` `docs/gone.md` `docs/coming.md` (new) `tools/x.sh --flag` `rust/src/...`"
    assert dangling_paths(text, tmp_path) == ["docs/gone.md", "tools/x.sh"]


def test_an_unknown_or_old_style_inbound_reference_is_rejected() -> None:
    sources = {
        "a.md": "see `docs/ROADMAP.md` M9",
        "b.md": "see ROADMAP item 1",
        "c.md": "ROADMAP Z1",
    }
    assert unresolved_inbound({"Z1"}, sources) == [
        "a.md: ROADMAP M9 is not an item",
        "b.md: 'ROADMAP item 1' -- refer to a roadmap item by its ID",
    ]
