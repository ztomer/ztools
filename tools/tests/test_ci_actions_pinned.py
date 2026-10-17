"""Every GitHub Action this repo runs is pinned to a release SHA, with its tag in a comment.

`uses: actions/checkout@v7` is a supply-chain dependency on whatever the maintainer pushes to
that tag next: the same workflow file runs different code tomorrow, and a review of today's diff
says nothing about it. `uses: actions/checkout@<40-hex sha> # v7` is the same action, twice
readable -- machine-checkable, and still naming the release.

The runner-image half of this area -- the runner image (`macos-15`) being two majors behind
the host -- is a separate risk with a separate fix (it can break a build; this cannot) and is
tracked on its own. This test is only the pin.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

#: `- uses: owner/repo@ref`, with or without a leading step dash. The ref stops at whitespace or
#: a `#`, so a trailing tag comment is not read as part of it.
USES = re.compile(r"^\s*(?:-\s+)?uses:\s*['\"]?([^'\"\s#]+)")
#: The trailing `# v7` a pinned line carries.
COMMENT = re.compile(r"#\s*(\S.*?)\s*$")
SHA = re.compile(r"^[0-9a-f]{40}$")


def _workflow_files() -> list[Path]:
    return sorted(list(WORKFLOWS.glob("*.yml")) + list(WORKFLOWS.glob("*.yaml")))


def _uses_lines(text: str):
    """(lineno, line, ref) for every `uses:` in one workflow file."""
    for n, line in enumerate(text.splitlines(), 1):
        m = USES.match(line)
        if m:
            yield n, line, m.group(1)


def _offenders_in(name: str, text: str) -> list[str]:
    """Every `uses:` in one workflow that is not a SHA pin naming its tag. Local and docker refs
    are exempt: neither is a supply-chain dependency on someone else's moving tag."""
    out = []
    for n, line, ref in _uses_lines(text):
        if ref.startswith("./") or ref.startswith("docker://"):
            continue
        if "@" not in ref:
            out.append(f"{name}:{n}: `{ref}` names no ref at all")
            continue
        _, _, version = ref.rpartition("@")
        if not SHA.match(version):
            out.append(f"{name}:{n}: `{ref}` is not pinned to a 40-hex SHA")
        elif not COMMENT.search(line):
            out.append(f"{name}:{n}: `{ref}` pins a SHA but names no tag in a comment")
    return out


def _offenders() -> list[str]:
    return [o for wf in _workflow_files() for o in _offenders_in(wf.name, wf.read_text())]


def test_the_reader_finds_each_uses_shape():
    """The parser is the gate, so it is proved on every shape a `uses:` takes here -- a reader
    that found nothing would pass every workflow."""
    text = (
        "      - uses: actions/checkout@v7\n"
        "      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2\n"
        "      - uses: ./local/action\n"
        "      - uses: docker://alpine:3.20\n"
    )
    assert [ref for _n, _line, ref in _uses_lines(text)] == [
        "actions/checkout@v7",
        "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683",
        "./local/action",
        "docker://alpine:3.20",
    ]


def test_each_way_a_pin_is_missing_is_an_offender():
    """Three ways to be wrong, one way to be right -- so a rule that only caught one of them
    would still pass this suite's real tree, which has four of the first."""
    good = "a" * 40
    text = (
        "      - uses: actions/checkout@v7\n"  # moving tag
        "      - uses: actions/setup-python\n"  # no ref
        f"      - uses: actions/cache@{good}\n"  # pinned, but unreadable
        f"      - uses: actions/checkout@{good} # v4.2.2\n"  # good
        "      - uses: ./local/action\n"  # exempt
    )
    found = _offenders_in("w.yml", text)
    assert len(found) == 3, found
    assert any("not pinned to a 40-hex SHA" in f for f in found)
    assert any("names no ref at all" in f for f in found)
    assert any("names no tag in a comment" in f for f in found)


def test_every_action_is_pinned_to_a_release_sha_naming_its_tag():
    offenders = _offenders()
    assert not offenders, "pin every action and name its tag:\n  " + "\n  ".join(offenders)
