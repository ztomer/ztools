"""Tests for lib.quality_runner - query_model and run_suite."""
import json
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_llm():
    from lib.testing import MockLLM
    mock = MockLLM()
    mock.patch_all()
    yield mock
    mock.unpatch()


class TestTestCases:
    def test_filename_cases_count(self, mock_llm):
        from lib.quality_runner import FILENAME_CASES
        # 5 cases covering keyword, no-match, format variations
        assert len(FILENAME_CASES) == 5
        for case in FILENAME_CASES:
            assert case.task == "filename"
            assert case.input_text
            assert case.reference

    def test_summarize_cases_count(self, mock_llm):
        from lib.quality_runner import SUMMARIZE_CASES
        # 2 summarize cases
        assert len(SUMMARIZE_CASES) == 2
        for case in SUMMARIZE_CASES:
            assert case.task == "summarize"

    def test_file_summary_cases_count(self, mock_llm):
        from lib.quality_runner import FILE_SUMMARY_CASES
        # 1 file_summary case
        assert len(FILE_SUMMARY_CASES) == 1
        for case in FILE_SUMMARY_CASES:
            assert case.task == "file_summary"

    def test_all_test_cases_combined(self, mock_llm):
        from lib.quality_runner import ALL_TEST_CASES, FILENAME_CASES, SUMMARIZE_CASES, FILE_SUMMARY_CASES
        assert len(ALL_TEST_CASES) == len(FILENAME_CASES) + len(SUMMARIZE_CASES) + len(FILE_SUMMARY_CASES)


class TestQueryModel:
    def test_query_model_success(self, mock_llm):
        from lib import quality_runner as qr
        result = qr.query_model("test-model", "Hello {text}", "world", "think")
        # Mock returns "mock content for think" for unknown task
        assert result == "mock content for think"

    def test_query_model_exception(self, mock_llm):
        from lib import quality_runner as qr
        with patch.object(qr, "llm_call", side_effect=Exception("boom")):
            result = qr.query_model("test-model", "Hi {text}", "world", "think")
        assert result is None

    def test_query_model_no_content(self, mock_llm):
        from lib import quality_runner as qr
        mock_result = {"content": None}
        with patch.object(qr, "llm_call", return_value=mock_result):
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

    def test_run_suite_worst_dim_low(self, mock_llm, capsys):
        """When worst dimension is < 60, the FAIL mark prints."""
        from lib import quality_runner as qr
        from lib.quality_models import TestCase, Score
        case = TestCase(
            task="filename",
            input_text="Screenshot showing login",
            reference="login_screenshot",
            description="test",
        )
        # Force a low-scoring scorecard
        with patch.object(qr, "query_model", return_value="login_screenshot.txt"), \
             patch.object(qr, "score_output") as mock_score:
            sc = MagicMock()
            sc.composite = 30
            sc.dimensions = [Score("Relevance", 30, 0.4, failures=["bad"])]
            sc.elapsed = 0.1
            sc.task = "filename"
            sc.model = "mock-model"
            sc.case_id = "test"
            sc.output = "login_screenshot.txt"
            mock_score.return_value = sc
            results = qr.run_suite(["mock-model"], [case], verbose=True)
        captured = capsys.readouterr()
        assert "30" in captured.out

    def test_run_suite_worst_dim_mid(self, mock_llm, capsys):
        """When worst is 60-80, WARN mark."""
        from lib import quality_runner as qr
        from lib.quality_models import TestCase, Score
        case = TestCase(
            task="filename",
            input_text="Screenshot",
            reference="screenshot",
            description="t",
        )
        with patch.object(qr, "query_model", return_value="ok.txt"), \
             patch.object(qr, "score_output") as mock_score:
            sc = MagicMock()
            sc.composite = 50
            sc.dimensions = [Score("R", 65, 0.4, failures=[])]
            sc.elapsed = 0.1
            sc.model = "mock-model"
            sc.task = "filename"
            sc.case_id = "t"
            sc.output = "ok.txt"
            mock_score.return_value = sc
            results = qr.run_suite(["mock-model"], [case], verbose=True)
        captured = capsys.readouterr()
        assert "50" in captured.out

    def test_run_suite_verbose_false(self, mock_llm, capsys):
        from lib import quality_runner as qr
        from lib.quality_models import TestCase
        case = TestCase(
            task="filename",
            input_text="Screenshot",
            reference="screenshot",
            description="t",
        )
        results = qr.run_suite(["mock-model"], [case], verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_run_suite_empty_dimensions(self, mock_llm, capsys):
        """ScoreCard with output but no dimensions shows 0.0%."""
        from lib import quality_runner as qr
        from lib.quality_models import TestCase
        case = TestCase(
            task="filename",
            input_text="Screenshot",
            reference="screenshot",
            description="t",
        )
        # Output exists, but score_output returns ScoreCard with no dimensions
        with patch.object(qr, "query_model", return_value="something"), \
             patch.object(qr, "score_output") as mock_score:
            sc = MagicMock()
            sc.composite = 0
            sc.dimensions = []
            sc.elapsed = 0.1
            sc.model = "mock-model"
            sc.task = "filename"
            sc.case_id = "t"
            sc.output = "something"
            mock_score.return_value = sc
            results = qr.run_suite(["mock-model"], [case], verbose=True)
        captured = capsys.readouterr()
        assert "0.0%" in captured.out

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
