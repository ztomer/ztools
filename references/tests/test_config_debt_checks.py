"""`_is_test_file` decides which files the config-debt gate skips.

Regression coverage for a real gap: it checked only the immediate parent
directory's name, so `references/tests/conftest_fixtures/legacy.py` -- test
fixture data one level below `references/tests/` -- was NOT recognized as test
code, and splitting conftest.py's fixtures into that subdirectory made the
gate flag verbatim moved lines (a hardcoded sample year, a mock model name) as
new config debt.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from config_debt_checks import _is_test_file  # noqa: E402


class TestIsTestFile:
    def test_a_file_directly_under_tests_is_test_code(self):
        assert _is_test_file(Path("references/tests/conftest.py"))

    def test_a_file_one_level_below_tests_is_test_code(self):
        assert _is_test_file(Path("references/tests/conftest_fixtures/legacy.py"))

    def test_a_file_named_test_prefixed_is_test_code_anywhere(self):
        assert _is_test_file(Path("references/lib/test_helpers.py"))

    def test_production_code_is_not_test_code(self):
        assert not _is_test_file(Path("references/eval/run_transport.py"))

    def test_a_file_under_an_unrelated_directory_named_tests_like_is_not_test_code(self):
        assert not _is_test_file(Path("references/contests/scoring.py"))
