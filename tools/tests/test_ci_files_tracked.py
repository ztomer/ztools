"""Every file a workflow reads from the checkout is a file the repo tracks.

CI checks out what git tracks, nothing more. A workflow input that names a file
only the developer's tree has -- ignored, or never added -- passes every local
gate and stops the gate of record before it starts: `python-version-file:
.python-version` did exactly that from 22d8009 on (CI red since 2026-10-10),
because `.gitignore` carried the stock `# pyenv` stanza and the file had never
been added.

So each `*-file:` input, each `hashFiles(...)` pattern and each `path:` input in
`.github/workflows/*.yml` must resolve to at least one `git ls-files` path.
"""

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# `python-version-file: .python-version`, `cache-dependency-path` is not a
# `-file` key and is read as a path below.
FILE_INPUT = re.compile(r"^\s*(?:-\s+)?[\w-]+-file:\s*['\"]?([^'\"#\s]+)")
PATH_INPUT = re.compile(r"^\s*(?:-\s+)?(?:[\w-]+-)?path:\s*['\"]?([^'\"#\s|>]+)")
HASH_FILES = re.compile(r"hashFiles\(([^)]*)\)")
QUOTED = re.compile(r"'([^']+)'|\"([^\"]+)\"")


def workflow_inputs(text: str) -> list[str]:
    """The checkout-relative paths and globs a workflow file reads."""
    found = []
    for line in text.splitlines():
        for pattern in (FILE_INPUT, PATH_INPUT):
            m = pattern.match(line)
            if m and "${{" not in m.group(1):
                found.append(m.group(1))
        for call in HASH_FILES.findall(line):
            found.extend(a or b for a, b in QUOTED.findall(call))
    return found


def tracked(pattern: str) -> list[str]:
    """What `git ls-files` holds for one path or glob (a dir counts by its files)."""
    out = subprocess.run(
        ["git", "ls-files", "--", f":(glob){pattern}", pattern.rstrip("/") + "/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
    ).stdout
    return out.split()


def test_the_reader_finds_each_input_shape():
    """The parser is the gate, so it is proved on every shape it claims to read --
    an extractor that found nothing would pass every workflow."""
    text = (
        "      - uses: actions/setup-python@v7\n"
        "        with:\n"
        "          python-version-file: .python-version\n"
        "          cache-dependency-path: 'requirements.txt'\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n"
        "          path: target/\n"
        "          key: ${{ hashFiles('rust/Cargo.lock', \"**/x.toml\") }}\n"
        "          path: ${{ runner.temp }}/goh\n"
    )
    assert workflow_inputs(text) == [
        ".python-version",
        "requirements.txt",
        "target/",
        "rust/Cargo.lock",
        "**/x.toml",
    ]


def test_an_untracked_input_is_not_tracked():
    """`tracked` must say no: an ignored file and a missing one both read empty."""
    assert tracked("no/such/file.txt") == []
    assert tracked("rust/Cargo.lock") == ["rust/Cargo.lock"]


def test_every_workflow_input_is_tracked():
    workflows = sorted(WORKFLOWS.glob("*.yml"))
    assert [w.name for w in workflows] == ["ci.yml"]
    inputs = {
        (w.name, p) for w in workflows for p in workflow_inputs(w.read_text(encoding="utf-8"))
    }
    assert ("ci.yml", ".python-version") in inputs
    missing = sorted(f"{name}: {p}" for name, p in inputs if not tracked(p))
    assert missing == [], (
        "a workflow reads a file CI's checkout will not have -- track it, or stop "
        f"reading it: {missing}"
    )
