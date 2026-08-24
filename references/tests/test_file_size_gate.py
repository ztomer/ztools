"""The 500-line file size gate. No exemptions, for tests or for any directory.

The gate existed only for Python until 2026-08-23: `.githooks/pre-commit` filtered
`f.endswith(".py")`, so every Rust file was outside the cap while the hook still
reported success. json_validator.rs reached 1126 lines next to a 485-line Python
file the same gate was about to block.

A test-code exemption existed briefly after that (Python files under
`references/tests/`; Rust `*_tests.rs` siblings and inline `#[cfg(test)]` blocks
subtracted from the count) and was removed 2026-08-24: a 1505-line test file is
exactly as unreadable and undiffable as a 1505-line production file, and an
exemption by directory or filename is one more place the rule silently narrows
when a layout doesn't match the carve-out's shape.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from check_file_size import MAX_LINES, check, line_count  # noqa: E402


class TestLineCounting:
    def test_python_counts_every_line(self):
        src = "def a():\n    pass\n\n\ndef b():\n    pass\n"
        assert line_count(src) == 6

    def test_rust_counts_every_line_including_cfg_test(self):
        src = (
            "pub fn a() -> u8 {\n    1\n}\n"
            "\n"
            "#[cfg(test)]\nmod tests {\n    #[test]\n    fn t() {\n        assert!(true);\n"
            "    }\n}\n"
        )
        assert line_count(src) == 11

    def test_a_file_ending_without_a_trailing_newline_still_counts_its_last_line(self):
        assert line_count("a\nb\nc") == 3

    def test_an_empty_file_is_zero_lines(self):
        assert line_count("") == 0


class TestNoExemptions:
    """The carve-out these pin the absence of: no path or filename escapes the cap."""

    def test_a_test_directory_file_over_the_limit_is_reported(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_big.py").write_text("x = 1\n" * (MAX_LINES + 1))
        errors = check(["tests/test_big.py"], tmp_path)
        assert len(errors) == 1
        assert "test_big.py" in errors[0]

    def test_a_rust_underscore_tests_file_over_the_limit_is_reported(self, tmp_path):
        (tmp_path / "weekend_enforce_tests.rs").write_text("fn t() {}\n" * (MAX_LINES + 1))
        errors = check(["weekend_enforce_tests.rs"], tmp_path)
        assert len(errors) == 1

    def test_a_conftest_file_over_the_limit_is_reported(self, tmp_path):
        (tmp_path / "conftest.py").write_text("x = 1\n" * (MAX_LINES + 1))
        errors = check(["conftest.py"], tmp_path)
        assert len(errors) == 1

    def test_an_inline_cfg_test_module_is_no_longer_subtracted(self, tmp_path):
        """A file over the limit purely because of its `#[cfg(test)]` body must
        still be reported -- that subtraction was the exemption, and it is gone."""
        body = "fn f() {}\n" * 400
        tests = "#[cfg(test)]\nmod tests {\n" + "    fn t() {}\n" * 400 + "}\n"
        (tmp_path / "m.rs").write_text(body + tests)
        errors = check(["m.rs"], tmp_path)
        assert len(errors) == 1


class TestTheGateReports:
    def test_it_names_an_over_limit_file(self, tmp_path):
        (tmp_path / "big.rs").write_text("fn f() {}\n" * (MAX_LINES + 1))
        errors = check(["big.rs"], tmp_path)
        assert len(errors) == 1
        assert "big.rs" in errors[0] and str(MAX_LINES) in errors[0]

    def test_it_stays_quiet_at_exactly_the_limit(self, tmp_path):
        (tmp_path / "ok.rs").write_text("fn f() {}\n" * MAX_LINES)
        assert check(["ok.rs"], tmp_path) == []

    def test_non_source_files_are_ignored(self, tmp_path):
        (tmp_path / "d.md").write_text("x\n" * (MAX_LINES + 1))
        assert check(["d.md"], tmp_path) == []

    def test_a_missing_staged_file_is_skipped_not_errored(self, tmp_path):
        """A staged rename/delete lists a path that may no longer exist on disk."""
        assert check(["gone.py"], tmp_path) == []


class TestTheRepoIsCleanUnderTheGate:
    """The regression guard. Runs the gate over every tracked source file,
    test files included -- there is no directory this skips."""

    def test_no_tracked_source_file_is_over_the_limit(self):
        tracked = subprocess.run(
            ["git", "ls-files", "*.py", "*.rs"],
            capture_output=True, text=True, cwd=str(ROOT),
        ).stdout.split()
        assert tracked, "git ls-files returned nothing -- the check would vacuously pass"
        errors = check(tracked, ROOT)
        assert errors == [], "\n".join(errors)


class TestTheHookUsesThisChecker:
    """Structural, per CLAUDE.md rule #3: the gate and the rule cannot drift.

    The hook previously inlined its own size check, which is how the `.py`-only
    filter survived a Rust port, and later how a test exemption could slip in
    without touching the rule's own wording. Sharing one implementation means
    fixing the rule here fixes it everywhere it is enforced.
    """

    def test_the_pre_commit_hook_calls_check_file_size(self):
        hook = (ROOT / ".githooks" / "pre-commit").read_text()
        assert "check_file_size" in hook

    def test_the_hook_no_longer_filters_the_size_gate_to_python(self):
        hook = (ROOT / ".githooks" / "pre-commit").read_text()
        assert 'not f.endswith(".py")' not in hook, (
            "the size gate is filtering to Python again -- Rust would go ungated"
        )

    def test_the_checker_module_defines_no_exemption_function(self):
        """A reviewer grepping for `is_exempt` should find nothing to re-wire."""
        checker_src = (ROOT / "tools" / "check_file_size.py").read_text()
        assert "is_exempt" not in checker_src
        assert "/tests/" not in checker_src
