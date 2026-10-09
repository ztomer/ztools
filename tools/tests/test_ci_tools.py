"""`tools/ci_tools.tsv` is a pin, and this file is what makes it one.

The list it holds is the tools only THIS REPO'S OWN steps need: `cargo-audit` and
`cargo-deny`, for the `cargo audit` and `cargo deny` steps `.gatesrc` added
(2026-10-05). It is deliberately NOT a copy of gates_of_heck's manifest -- that
manifest names the tools ITS gates refuse without, ci.yml consumes it directly
(`"$GOH_DIR/gates/required_tools.py" --repo . --install`), and a consumed list
cannot drift. The repo-local half is the half nothing keeps honest, and it is the
half that rotted: run 37855122727 died at step 1/12 with `cargo-machete is not
installed, and this gate does not pass without the step it runs`, because a
hand-copied tool list had not heard that a step had grown.

THE INVARIANT: every name in the file is a step in `.gatesrc`'s `GOH_CI_STEPS`.
Both directions are checked, because either one alone passes by inspecting
nothing:

  * a name no step needs goes red. That is the direction rot runs in, and it is
    why the check is a rule rather than a golden list -- a golden list fails on
    every legitimate change, and a list that fails on every change gets
    regenerated without being read.
  * the file names something, and blank lines do not become names, so the rule
    above is not vacuously true. An empty name would satisfy it: every step
    starts with `""`. That is not a theory about a parser, it is a hole the rule
    has, and it is asserted shut below.

RESIDUAL, stated rather than papered over: the rule cannot see a step that needs
a tool the file forgot -- there is nothing here to be missing. That half is
caught by the run itself, loudly and late (`cargo audit: command not found` in
the step that needs it), which is why ci.yml's read of this file FAILS on an empty
list rather than installing nothing and passing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CI_TOOLS = ROOT / "tools" / "ci_tools.tsv"
GATESRC = ROOT / ".gatesrc"

#: The `.gatesrc` key that declares what the gate runs. Read from the file, never
#: restated here: the pin has to move when the gate moves, or it pins the past.
KEY = "GOH_CI_STEPS"


def _unquote(raw: str) -> str:
    """The literal value bash assigns to `KEY=<raw>`, inline comment removed.

    `.gatesrc` comments a key on its own line (`GOH_MAX_LINES=500   # cap`), so
    an unquoted whitespace-`#` starts a comment and is not part of the value.
    """
    raw = re.split(r"\s#", raw, maxsplit=1)[0].strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        if raw[0] == "'":
            # Bash strips the quotes and does NOT re-parse the interior, so the
            # double quotes inside a `'...'` value survive into the value itself.
            return raw[1:-1]
        return re.sub(r"\\([$`\"\\])", r"\1", raw[1:-1])
    return raw


def _gatesrc_steps(text: str) -> list[str]:
    """`GOH_CI_STEPS` the way the gate's own runner splits it: on `:`, empties dropped.

    THE QUOTING IS LOAD-BEARING, and it is why the value is unquoted rather than
    tokenised. `.gatesrc` writes the value in single quotes so a step can carry
    `$GOH_DIR` and `$PWD` that only resolve when the step RUNS, and bash assigns
    the inner text verbatim -- the double quotes are in the value. local_ci.sh
    then splits on `:` and hands each piece to a shell, so
    `"$GOH_DIR/gates/rust_gate.sh" . rust` is ONE step and its quotes are read at
    execution time. A tokenising parse (`shlex.split`) would cut that step at its
    spaces and lose it, and the first step of the gate is not a step to lose.
    """
    match = re.search(rf"^\s*(?:export\s+)?{KEY}=(.*)$", text, re.M)
    assert match, f".gatesrc declares no {KEY}, so no step needs any tool"
    return [step for step in _unquote(match.group(1)).split(":") if step]


def _tools(text: str) -> list[str]:
    """The names ci.yml's install step reads out of the file.

    The same parse, not a second opinion: a `#` starts a comment anywhere in the
    line, and a line left empty by that is not a tool.
    """
    names = []
    for line in text.splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            names.append(name)
    return names


def _step_needing(tool: str, steps: list[str]) -> str | None:
    """The step that runs this tool, or None.

    The two spellings differ on purpose: the name is a package (`cargo-audit`)
    and the step is a command line (`cargo audit --file rust/Cargo.lock`), so
    either the name or its space spelling is accepted as the step's prefix.
    """
    for step in steps:
        if step.startswith(tool) or step.startswith(tool.replace("-", " ")):
            return step
    return None


def _unneeded(tools_text: str, steps_text: str) -> list[str]:
    """The names in `tools_text` that no step in `steps_text` runs."""
    steps = _gatesrc_steps(steps_text)
    return [
        f"{tool!r} -- no step in {KEY} runs it; add the step, or drop the line"
        for tool in _tools(tools_text)
        if _step_needing(tool, steps) is None
    ]


def _gatesrc_with(steps: str) -> str:
    """A `.gatesrc`-shaped text carrying `steps` as its GOH_CI_STEPS.

    Shaped like the real file rather than handed to the splitter directly, so
    the synthetic cases exercise the same parse the real file goes through --
    including the quoting, which is the part most easily got wrong.
    """
    return f"export GOH_CI_STEPS='{steps}'\n"


def _duplicates(tools_text: str) -> list[str]:
    names = _tools(tools_text)
    return sorted({name for name in names if names.count(name) > 1})


# ── the real file ─────────────────────────────────────────────────────────────


def test_every_tool_is_a_step_the_gate_runs():
    """The invariant itself: a name in the file, a step in `.gatesrc` that runs it.

    This is the direction rot runs in. A tool added to the file for a step that
    was since dropped from GOH_CI_STEPS installs on every run for nothing, and a
    tool dropped from GOH_CI_STEPS while it stayed here is a step the runner
    cannot run -- either way the runner is preparing for a gate that is no longer
    the gate.
    """
    bad = _unneeded(CI_TOOLS.read_text(), GATESRC.read_text())
    assert not bad, "\n".join(bad)


def test_the_file_names_something():
    """A rule checked over an empty file passes by inspecting nothing.

    The workflow's own read refuses an empty list for the same reason; this is
    the same refusal on the side that can see it.
    """
    names = _tools(CI_TOOLS.read_text())
    assert names, f"{CI_TOOLS} names no tools, so the rule above inspected nothing"


def test_the_step_parse_finds_the_steps_this_file_exists_for():
    """Anti-vacuity for the parse itself.

    Every other test here is only as good as `_gatesrc_steps`. A parse that
    returned [] -- a renamed key, a reworded assignment -- would make the main
    rule vacuously true, so the parse is pinned to the two steps that are the
    file's whole subject: if `.gatesrc` renames them, this says so rather than
    letting the file quietly check nothing.
    """
    steps = _gatesrc_steps(GATESRC.read_text())
    assert len(steps) > 2, f"parsed {len(steps)} step(s), which is too few to be the gate"
    for command in ("cargo audit", "cargo deny"):
        assert any(step.startswith(command) for step in steps), (
            f"no {KEY} step runs `{command}`, which ci_tools.tsv exists to serve"
        )


def test_no_name_appears_twice():
    """One tool, one line: a duplicate is a reader that installs it twice."""
    assert not _duplicates(CI_TOOLS.read_text())


# ── the rule, run against what it must reject ─────────────────────────────────


@pytest.mark.parametrize(
    "steps,tools_text,offender",
    [
        (
            "cargo audit --file rust/Cargo.lock:cargo deny --manifest-path rust/Cargo.toml check\n",
            "cargo-audit\ncargo-deny\ncargo-nextest\n",
            "cargo-nextest",
        ),
        (
            "cargo deny --manifest-path rust/Cargo.toml check\n",
            "cargo-audit\ncargo-deny\n",
            "cargo-audit",
        ),
        (
            "cargo audit --file rust/Cargo.lock:cargo deny --manifest-path rust/Cargo.toml check\n",
            "cargo-audit\ncargo-deny-tools\n",
            "cargo-deny-tools",
        ),
    ],
)
def test_a_name_no_step_runs_is_red(steps, tools_text, offender):
    """The direction that catches the rot, proven on inputs built to fail.

    The rule must name the offender in its message, or a reader of a red run has
    to diff the file by hand to find the line. (That the same rule passes over
    the real file is `test_every_tool_is_a_step_the_gate_runs` above.)
    """
    bad = _unneeded(tools_text, _gatesrc_with(steps))
    assert bad, "the rule accepted a tool no step runs, so it inspects nothing"
    assert any(offender in entry for entry in bad), f"{offender} is not named in {bad}"


def test_a_duplicate_is_named():
    assert _duplicates("cargo-audit\ncargo-deny\ncargo-audit\n") == ["cargo-audit"]


# ── the holes the rule would otherwise have ───────────────────────────────────


def test_blank_lines_do_not_become_names():
    """An empty name passes every other check in this file.

    `"".startswith` is true of every step, and `len(names) == len(set(names))`
    cannot see it, so a reader that appended unconditionally would report an
    all-whitespace file as compliant. Asserted on the parse, in the shapes the
    file can actually hold them.
    """
    names = _tools("# a header\n\ncargo-audit\n   \n\t\ncargo-deny\n# trailing\n")
    assert names == ["cargo-audit", "cargo-deny"]


def test_a_comment_only_file_names_nothing():
    """...which is also why the emptiness check and this rule are both needed."""
    assert _tools("# nothing here\n\n   # still nothing\n") == []


@pytest.mark.parametrize("spelling", ["cargo-audit", "cargo audit"])
def test_either_spelling_of_a_name_finds_its_step(spelling):
    """The matcher accepts both spellings the rule allows, on one real step."""
    steps = _gatesrc_steps(GATESRC.read_text())
    assert _step_needing(spelling, steps), f"no step matches the spelling {spelling!r}"


def test_a_name_with_no_step_finds_nothing():
    """And rejects the third case, so the two above are not both trivially true."""
    steps = _gatesrc_steps(GATESRC.read_text())
    assert _step_needing("cargo-nextest", steps) is None
