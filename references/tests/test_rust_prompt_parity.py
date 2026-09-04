"""Drift gate: Rust prompt constants must be byte-identical to the Python source.

`rust/src/ztools/eval/prompts/` is GENERATED from `references/eval/tasks_prompts.py`
by `tools/gen_rust_prompts.py`. This test re-derives the Python values at test time
and diffs them against what the generated Rust source actually declares, so a
hand-edit on either side — or a regeneration from a changed Python constant —
fails the standard pytest gate instead of silently measuring models against a
different task than the reference.

This is the same anti-drift contract as
`rust/src/ztools/config.rs::test_twitter_prompt_matches_shared_conf`, applied to
the whole eval prompt set.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROMPTS_DIR = REPO_ROOT / "rust" / "src" / "ztools" / "eval" / "prompts"

sys_path = str(REPO_ROOT / "references")
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from eval import tasks_prompts as tp  # noqa: E402

# pub const NAME: &str = r##"..."##;   (raw string, 1-4 hashes)
_STR_RE = re.compile(r'pub const (\w+): &str = r(#{1,4})"(.*?)"\2;', re.DOTALL)
# pub const NAME: &[&str] = &["a", "b"];
_LIST_RE = re.compile(r"pub const (\w+): &\[&str\] = &\[(.*?)\];", re.DOTALL)
_LIST_ITEM_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _parse_rust_constants():
    """Extract every generated const from the prompts/ directory."""
    found_strs, found_lists = {}, {}
    for path in sorted(PROMPTS_DIR.glob("*.rs")):
        if path.name == "mod.rs":
            continue
        src = path.read_text()
        for name, _, value in _STR_RE.findall(src):
            found_strs[name] = (value, path.name)
        for name, body in _LIST_RE.findall(src):
            items = [
                m.replace('\\"', '"').replace("\\\\", "\\")
                for m in _LIST_ITEM_RE.findall(body)
            ]
            found_lists[name] = (items, path.name)
    return found_strs, found_lists


def _generated_names():
    strs, lists = _parse_rust_constants()
    return set(strs) | set(lists)


def test_every_exported_python_prompt_has_a_rust_counterpart():
    """A constant added to tasks_prompts.py must be added to the generator layout.

    A missing counterpart means the Rust eval would silently lack a prompt the
    Python reference measures with -- the two suites would diverge without any
    signal.
    """
    exported = [n for n in dir(tp) if n.isupper() and not n.startswith("_")]
    # Constants intentionally excluded: derived intermediates whose final form
    # IS exported (e.g. WEEKEND_USR_* feed the *_MIXED variants), and noise
    # blocks that exist only inside the mixed variants.
    excluded_prefixes = ("WEEKEND_FABRICATION",)
    missing = [
        n for n in exported
        if n not in _generated_names() and not n.startswith(excluded_prefixes)
    ]
    assert not missing, (
        f"tasks_prompts.py exports {missing} with no generated Rust constant; "
        "add them to FILE_LAYOUT in tools/gen_rust_prompts.py and regenerate"
    )


@pytest.mark.parametrize(
    "name",
    sorted(n for n in dir(tp) if n.isupper() and not n.startswith("_")),
)
def test_string_constant_matches_python(name):
    strs, lists = _parse_rust_constants()
    if name not in strs:
        pytest.skip("not a string constant (or covered by the list test)")
    rust_value, source_file = strs[name]
    assert rust_value == getattr(tp, name), (
        f"{name} in rust/src/ztools/eval/prompts/{source_file} drifted from "
        "references/eval/tasks_prompts.py; regenerate with "
        "PYTHONPATH=references python3 tools/gen_rust_prompts.py"
    )


@pytest.mark.parametrize(
    "name",
    sorted(n for n in dir(tp) if n.isupper() and not n.startswith("_")),
)
def test_list_constant_matches_python(name):
    strs, lists = _parse_rust_constants()
    if name not in lists:
        pytest.skip("not a list constant (or covered by the string test)")
    rust_value, source_file = lists[name]
    assert rust_value == list(getattr(tp, name)), (
        f"{name} in rust/src/ztools/eval/prompts/{source_file} drifted from "
        "references/eval/tasks_prompts.py; regenerate with "
        "PYTHONPATH=references python3 tools/gen_rust_prompts.py"
    )
