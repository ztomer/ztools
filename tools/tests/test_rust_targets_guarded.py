"""Every Rust integration-test target refuses a stale build of itself.

Cargo rebuilds a test when its file's mtime moves; a mutant restored with
`mv x.bak x` keeps the old mtime, so cargo runs the mutant (ZT.RESIDFLAKE).
`ztools::built_from_disk!();` adds a test comparing the compiled bytes with the
file on disk. A target without it can run stale silently (ZT.MTIMECLASS).
"""

from pathlib import Path

TESTS = Path(__file__).resolve().parents[2] / "rust" / "tests"
GUARD = "\nztools::built_from_disk!();\n"


def unguarded(tests_dir: Path) -> list[str]:
    """The top-level target files under `tests_dir` lacking the guard."""
    return sorted(f.name for f in tests_dir.glob("*.rs") if GUARD not in f.read_text())


def test_every_rust_test_target_calls_the_guard() -> None:
    targets = list(TESTS.glob("*.rs"))
    assert len(targets) >= 25, f"found only {len(targets)} targets under {TESTS}"
    missing = unguarded(TESTS)
    assert missing == [], f"add `ztools::built_from_disk!();` to: {missing}"


def test_the_check_sees_an_unguarded_target(tmp_path: Path) -> None:
    (tmp_path / "ok.rs").write_text("//! x\n\nztools::built_from_disk!();\n")
    (tmp_path / "bad.rs").write_text("//! x\n// ztools::built_from_disk!();\n")
    assert unguarded(tmp_path) == ["bad.rs"]
