"""Tests for lib.quality_runner - query_model and run_suite."""

from unittest.mock import patch


class TestTestCases:
    def test_filename_cases_count(self, mock_llm):
        from lib.eval_data import FILENAME_CASES

        # 5 cases covering keyword, no-match, format variations
        assert len(FILENAME_CASES) == 5
        for case in FILENAME_CASES:
            assert case.task == "filename"
            assert case.input_text
            assert case.reference

    def test_summarize_cases_count(self, mock_llm):
        from lib.eval_data import SUMMARIZE_CASES

        # 2 summarize cases
        assert len(SUMMARIZE_CASES) == 2
        for case in SUMMARIZE_CASES:
            assert case.task == "summarize"

    def test_file_summary_cases_count(self, mock_llm):
        from lib.eval_data import FILE_SUMMARY_CASES

        # 1 file_summary case
        assert len(FILE_SUMMARY_CASES) == 1
        for case in FILE_SUMMARY_CASES:
            assert case.task == "file_summary"

    def test_all_test_cases_combined(self, mock_llm):
        from lib.eval_data import (
            ALL_TEST_CASES,
            FILE_SUMMARY_CASES,
            FILENAME_CASES,
            SUMMARIZE_CASES,
            WEEKEND_FIXED_CASES,
            WEEKEND_TRANSIENT_CASES,
        )

        expected = (
            len(FILENAME_CASES)
            + len(SUMMARIZE_CASES)
            + len(FILE_SUMMARY_CASES)
            + len(WEEKEND_TRANSIENT_CASES)
            + len(WEEKEND_FIXED_CASES)
        )
        assert len(ALL_TEST_CASES) == expected


class TestQueryModel:
    def test_query_model_success(self, mock_llm):
        from lib import quality_runner as qr

        result = qr.query_model("test-model", "Hello {text}", "world", "think")
        # Mock returns "mock content for think" for unknown task
        assert result == "mock content for think"

    def test_query_model_exception(self, mock_llm):
        from lib import osaurus_lib
        from lib import quality_runner as qr

        with patch.object(osaurus_lib, "call", side_effect=Exception("boom")):
            result = qr.query_model("test-model", "Hi {text}", "world", "think")
        assert result is None

    def test_query_model_no_content(self, mock_llm):
        from lib import osaurus_lib
        from lib import quality_runner as qr

        mock_result = {"content": None}
        with patch.object(osaurus_lib, "call", return_value=mock_result):
            result = qr.query_model("test-model", "Hi {text}", "world", "think")
        assert result == ""


class TestRunSuite:
    def test_run_suite_with_cases(self, mock_llm, capsys):
        from lib import quality_runner as qr
        from lib.quality_models import TestCase

        case = TestCase(
            task="filename",
            input_text="Screenshot showing login error",
            reference="login_error",
            description="test case",
        )
        results = qr.run_suite(["mock-model"], [case], verbose=True)
        assert len(results) == 1
        sc = results[0]
        assert sc.model == "mock-model"

    def test_run_suite_default_cases(self, mock_llm, capsys):
        from lib import quality_runner as qr

        results = qr.run_suite(["mock-model"], None, verbose=False)
        # Default cases - all_test_cases
        assert len(results) > 0

    def test_run_suite_no_prompt(self, mock_llm, capsys):
        """When model has no prompt, case is skipped."""
        from lib import quality_runner as qr
        from lib.quality_models import TestCase

        case = TestCase(
            task="filename",
            input_text="test",
            reference="test",
            description="test case",
        )
        with patch.object(qr, "get_model_prompt", return_value=None):
            results = qr.run_suite(["mock-model"], [case], verbose=True)
        assert len(results) == 0
        captured = capsys.readouterr()
        assert "skip" in captured.out

    def test_run_suite_no_output(self, mock_llm, capsys):
        """When query returns None, get a no-dimensions ScoreCard."""
        from lib import quality_runner as qr
        from lib.quality_models import TestCase

        case = TestCase(
            task="filename",
            input_text="test",
            reference="test",
            description="test case",
        )
        with patch.object(qr, "query_model", return_value=None):
            results = qr.run_suite(["mock-model"], [case], verbose=True)
        assert len(results) == 1
        assert results[0].dimensions == []

    def test_run_suite_with_output_dimensions(self, mock_llm, capsys):
        """When output exists, dimensions are populated."""
        from lib import quality_runner as qr
        from lib.quality_models import TestCase

        case = TestCase(
            task="filename",
            input_text="Screenshot showing login",
            reference="login_screenshot",
            description="test",
        )
        with patch.object(qr, "query_model", return_value="login_screenshot.txt"):
            results = qr.run_suite(["mock-model"], [case], verbose=True)
        assert len(results) == 1
        sc = results[0]
        assert sc.elapsed >= 0
        assert sc.task == "filename"

    def _case(self):
        from lib.quality_models import TestCase

        return TestCase(
            task="filename",
            input_text="Screenshot showing login page",
            reference="login_screenshot",
            description="t",
        )

    def _run(self, output, capsys, verbose=True):
        """Drive run_suite with only the LLM boundary mocked.

        These tests used to patch `score_output` itself and assert that the
        number they injected appeared in the output, so the real scorer never
        ran — against the repo rule to use the real scorer and mock only the
        LLM layer. Now the marks come from real scores.
        """
        from lib import quality_runner as qr

        with patch.object(qr, "query_model", return_value=output):
            results = qr.run_suite(["mock-model"], [self._case()], verbose=verbose)
        return results, capsys.readouterr().out

    def test_run_suite_worst_dim_low(self, mock_llm, capsys):
        """A leaked instruction really does score low, and is marked as failing."""
        from lib.tui import FAIL

        results, out = self._run("Here is the filename: img.txt", capsys)
        assert round(results[0].composite) == 16
        assert min(d.score for d in results[0].dimensions) < 40
        assert FAIL in out

    def test_run_suite_worst_dim_mid(self, mock_llm, capsys):
        """A weak dimension above the failing composite gets the WARN mark.

        This is the branch that pins the 60 threshold: composite 73.75 with a
        worst dimension of 50 must warn, not pass.
        """
        from lib.tui import WARN

        results, out = self._run("login.txt", capsys)
        composite = results[0].composite
        worst = min(d.score for d in results[0].dimensions)
        assert composite == 73.75, composite
        assert worst == 50, worst
        assert WARN in out

    def test_run_suite_good_output_gets_the_step_mark(self, mock_llm, capsys):
        """The passing branch, from a genuinely good filename."""
        from lib.tui import STEP

        results, out = self._run("login_screenshot.txt", capsys)
        assert results[0].composite == 100
        assert min(d.score for d in results[0].dimensions) >= 60
        assert STEP in out

    def test_run_suite_verbose_false(self, mock_llm, capsys):
        _, out = self._run("login_screenshot.txt", capsys, verbose=False)
        assert out == ""

    def test_run_suite_empty_dimensions(self, mock_llm, capsys):
        """Empty model output really does produce a dimensionless 0.0% card."""
        results, out = self._run("", capsys)
        assert results[0].dimensions == []
        assert results[0].composite == 0
        assert "0.0%" in out

    def test_run_suite_multiple_models(self, mock_llm, capsys):
        from lib import quality_runner as qr
        from lib.quality_models import TestCase

        case = TestCase(
            task="filename",
            input_text="Screenshot",
            reference="screenshot",
            description="t",
        )
        with patch.object(qr, "query_model", return_value="ok.txt"):
            results = qr.run_suite(["model-a", "model-b"], [case], verbose=False)
        assert len(results) == 2
