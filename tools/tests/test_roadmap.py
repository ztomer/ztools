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
# The same reference written the other way round, which is how prose usually puts it:
# "that mechanism is `X6` in `docs/ROADMAP.md`". Both word orders name the same
# promise, so a check that only knows one of them cannot see half the references
# it is here to resolve (P4: rust-toolchain.toml's X6, which has never existed).
INBOUND_REVERSED = re.compile(r"`?([A-Z]\d+)`?\s+(?:in|see|per)\s+`?(?:docs/)?ROADMAP(?:\.md)?`?")
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
        for m in INBOUND_REVERSED.finditer(text):
            if m.group(1) not in defined:
                bad.append(f"{name}: `{m.group(1)}` in ...ROADMAP.md is not an item")
        for m in STALE_INBOUND.finditer(text):
            bad.append(f"{name}: '{m.group(0)}' -- refer to a roadmap item by its ID")
    return bad


def inbound_sources() -> dict[str, str]:
    globs = [
        "*.md",
        # Root *.toml: rust-toolchain.toml carries a roadmap pointer in a comment, and
        # a check that only globs docs/ cannot see a file the compiler reads (P4).
        "*.toml",
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


def independent_item_count(text: str) -> int:
    """Count item headings WITHOUT using `ITEM`, so the two can disagree.

    `items()` keys on `^### ([A-Z]\\d+) — .+$`. A parser that silently stops
    matching -- a renamed heading level, a lost em dash, a non-breaking space --
    returns {} and every rule below it reports "satisfied". This counts the
    headings a reader would call items by their shape alone, which is the only
    way a broken parser can fail rather than pass.

    A `### ` belongs to the `## ` section that contains it: a section runs from
    its heading to the next `## ` of any kind.
    """
    sections = [(m.start(), m.group(0)) for m in re.finditer(r"^## .+$", text, re.M)]
    sections.append((len(text), ""))
    n = 0
    for h in re.finditer(r"^### (.+)$", text, re.M):
        owner = ""
        for i, (start, title) in enumerate(sections):
            if start <= h.start() < sections[i + 1][0]:
                owner = title
                break
        if h.group(1).strip() and owner.startswith("## Phase"):
            n += 1
    return n


def item_count_disagreement(text: str) -> str | None:
    """None when `items()` and an independent heading count agree; why they differ otherwise."""
    found = len(items(text))
    independent = independent_item_count(text)
    if found != independent:
        return f"the item parser found {found} but {independent} `### ` headings sit under a phase"
    return None


def test_the_roadmap_has_items_to_check(roadmap: str) -> None:
    # `>= 0` is true of every possible input, so this asserted nothing (P3). An empty
    # roadmap passes honestly; a parser that has stopped matching the file's own
    # heading shape fails, which is the failure it was pretending to catch.
    assert len(items(roadmap)) > 0, (
        "the roadmap has no items at all; if that is genuinely true, delete this test deliberately"
    )
    assert item_count_disagreement(roadmap) is None


def test_the_count_check_fails_on_a_heading_the_parser_cannot_read() -> None:
    """A hyphen where the em dash belongs: `ITEM` misses B2, a reader still sees an item."""
    fixture = "## Phase Z — anything\n\n### A1 — real\n\n### B2 - not an em dash\n"
    assert "A1" in items(fixture) and "B2" not in items(fixture)
    assert item_count_disagreement(fixture) is not None, (
        "a heading the parser cannot read must make this check red -- it is the only thing P3 added"
    )


def test_an_empty_roadmap_passes_honestly() -> None:
    """Zero items is not a parser failure. It must not read as one, or the rule would force a
    placeholder item into the file forever."""
    assert item_count_disagreement("# Roadmap\n\nNothing open.\n") is None


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


def test_the_reversed_word_order_is_seen_too() -> None:
    """`X6` in `docs/ROADMAP.md` is the same promise as `ROADMAP X6`, and it is how prose
    writes it. rust-toolchain.toml's pointer survived for months because only one order was
    matched -- and because root `*.toml` was not even globbed, so the file was never read."""
    sources = {"rust-toolchain.toml": "that mechanism is `X6` in `docs/ROADMAP.md`"}
    assert unresolved_inbound({"P4"}, sources) == [
        "rust-toolchain.toml: `X6` in ...ROADMAP.md is not an item"
    ]
    assert unresolved_inbound({"X6"}, sources) == []


# --- no line-number citations into a config file (D3) ---------------------------
# `CLAUDE.md` cited `.gatesrc:26` for the coverage floor while `GOH_CI_STEPS` sits at
# `.gatesrc:113`. The number was never checked and nobody noticed for months: a doc
# citation is a claim about a FILE's interior, so it goes stale on any edit above it.
# The rule is to name the variable, never a line.

LINE_CITATION = re.compile(r"\.gatesrc:(\d+)")
# The roadmap quotes the forbidden form in order to forbid it, same as ANNOTATED_DONE.
_SKIP = {ROADMAP}


def line_citations(sources: dict[str, str]) -> list[str]:
    out = []
    for name, text in sorted(sources.items()):
        if Path(name).resolve() in _SKIP:
            continue
        for m in LINE_CITATION.finditer(text):
            out.append(f"{name}: `.gatesrc:{m.group(1)}` -- name the variable, not a line")
    return out


def test_no_doc_cites_a_line_number_in_gatesrc(roadmap: str) -> None:
    assert line_citations(inbound_sources()) == []


def test_a_line_citation_is_named_when_it_appears() -> None:
    found = line_citations({"docs/real.md": "the floor is at `.gatesrc:26` here"})
    assert found == ["docs/real.md: `.gatesrc:26` -- name the variable, not a line"]


# --- a historical doc must SAY it is historical (D2) ---------------------------
# Five docs under docs/ are measurement records from the Python era. They cite
# `twitter/summarize.py`, `eval/validate.py`, `lib/osaurus_lib.py` and friends --
# all deleted at the Rust cutover. The records are worth keeping; what is not
# worth keeping is a reader taking them for a description of the current tree.
# A banner is the whole fix, and a check is what stops the sixth one arriving
# unbannered.

# The RETIRED product tree only. `tools/` is deliberately absent: it is live
# tooling, and a tools script deleted on purpose (U1 retired upgrade_tap.py) is
# not a stale pointer to anything. Demanding a "Python-era" banner for it would
# make this check wrong in the same way it was wrong before.
PY_PATH = re.compile(
    r"(?<![\w/])((?:twitter|weekend|eval|lib|report_class_cases|"
    r"test_report_class_fixes|twitter_summarizer)[A-Za-z0-9_/]*\.py)\b"
)
# A doc declares itself in its own words; TESTING.md says "The Python-era version
# of this file ..." rather than the phrase this check was written around, and a
# check that only accepts its own phrasing is the same blindness in a new coat.
BANNER = re.compile(r"python-era", re.I)
# PORT_PARITY.md is the ledger whose SUBJECT is retired files: naming
# `twitter/summarize.py` is the job, not a stale pointer.
EXEMPT = {"docs/PORT_PARITY.md"}


def py_era_docs_without_a_banner(root: Path) -> list[str]:
    out = []
    for doc in sorted((root / "docs").glob("*.md")):
        rel = str(doc.relative_to(root))
        if rel in EXEMPT:
            continue
        text = doc.read_text(errors="replace")
        # Only DELETED paths are the defect. `tools/vendor_lint_ratchet.py` is cited
        # by ROADMAP.md and is still on disk; flagging it would have made this check
        # cry wolf on live files, which is how a check gets switched off.
        dead = sorted(
            {m.group(1) for m in PY_PATH.finditer(text) if not (root / m.group(1)).exists()}
        )
        if dead and not BANNER.search(text):
            out.append(f"{rel}: cites deleted Python-era paths with no banner ({len(dead)})")
    return out


def test_a_python_era_doc_declares_itself_one() -> None:
    assert py_era_docs_without_a_banner(ROOT) == []


def test_the_banner_check_would_catch_the_next_one(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "live.py").write_text("# still here\n")
    (tmp_path / "docs" / "ok.md").write_text("> Python-era record\n\nsee twitter/summarize.py\n")
    (tmp_path / "docs" / "new.md").write_text("# A fresh doc\n\nit calls twitter/summarize.py\n")
    (tmp_path / "docs" / "live_ref.md").write_text("see `tools/live.py`, which exists\n")
    assert py_era_docs_without_a_banner(tmp_path) == [
        "docs/new.md: cites deleted Python-era paths with no banner (1)"
    ]
