"""Tests for lib.quality_report - generate_report, save/load/compare baseline."""

import json
from unittest.mock import patch

import pytest
from lib.quality_models import Score, ScoreCard


@pytest.fixture
def mock_llm():
    from lib.testing import MockLLM

    mock = MockLLM()
    mock.patch_all()
    yield mock
    mock.unpatch()


def make_scorecard(model, task, case_id, dimensions, output="", elapsed=0.5):
    return ScoreCard(
        model=model,
        task=task,
        case_id=case_id,
        dimensions=dimensions,
        output=output,
        elapsed=elapsed,
    )


class TestGenerateReport:
    def test_empty(self, mock_llm):
        from lib.quality_report import generate_report

        result = generate_report([])
        assert "Model" in result
        assert "Filename" in result

    def test_single_model_single_task(self, mock_llm):
        from lib.quality_report import generate_report

        sc = make_scorecard(
            "model-a",
            "filename",
            "test",
            [Score("Relevance", 80, 0.4), Score("Format", 90, 0.35)],
        )
        result = generate_report([sc])
        assert "model-a" in result
        assert "filename" in result
        assert "Relevance" in result
        assert "Format" in result
        assert "Composite" in result

    def test_multiple_models(self, mock_llm):
        from lib.quality_report import generate_report

        sc1 = make_scorecard("a", "filename", "t1", [Score("R", 80, 0.4)])
        sc2 = make_scorecard("b", "filename", "t1", [Score("R", 60, 0.4)])
        result = generate_report([sc1, sc2])
        assert "a" in result
        assert "b" in result

    def test_multiple_tasks(self, mock_llm):
        from lib.quality_report import generate_report

        sc1 = make_scorecard("a", "filename", "t1", [Score("R", 80, 0.4)])
        sc2 = make_scorecard("a", "summarize", "t2", [Score("C", 70, 0.3)])
        sc3 = make_scorecard("a", "file_summary", "t3", [Score("C", 90, 0.4)])
        result = generate_report([sc1, sc2, sc3])
        assert "filename" in result
        assert "summarize" in result
        assert "file_summary" in result

    def test_failures_counted(self, mock_llm):
        from lib.quality_report import generate_report

        sc = make_scorecard(
            "a",
            "filename",
            "t1",
            [Score("R", 50, 0.4, failures=["bad"]), Score("F", 90, 0.35)],
        )
        result = generate_report([sc])
        # failures > 0
        assert "1" in result  # at least 1 failure

    def test_avg_time(self, mock_llm):
        from lib.quality_report import generate_report

        sc = make_scorecard("a", "filename", "t1", [Score("R", 80, 0.4)], elapsed=2.5)
        result = generate_report([sc])
        assert "2.5s" in result or "Avg time" in result

    def test_avg_time_no_cases(self, mock_llm):
        """When times list is empty, avg_time = 0."""
        from lib.quality_report import generate_report

        sc = make_scorecard("a", "filename", "t1", [Score("R", 80, 0.4)], elapsed=0)
        result = generate_report([sc])
        assert "0.0s" in result

    def test_speed_calculation(self, mock_llm):
        from lib.quality_report import generate_report

        sc1 = make_scorecard("a", "filename", "t1", [Score("R", 80, 0.4)], elapsed=1.0)
        sc2 = make_scorecard("a", "summarize", "t2", [Score("C", 70, 0.3)], elapsed=3.0)
        result = generate_report([sc1, sc2])
        # Speed = avg of all times = 2.0
        assert "Speed" in result

    def test_missing_task_no_avg(self, mock_llm):
        """When a model has no cards for a task, it shows 0%."""
        from lib.quality_report import generate_report

        sc = make_scorecard("a", "filename", "t1", [Score("R", 80, 0.4)])
        result = generate_report([sc])
        # Summarize and FileSum columns should be 0
        assert "  0.0%" in result

    def test_no_tasks_for_task_section(self, mock_llm):
        """If no cards for a task, the task section is skipped."""
        from lib.quality_report import generate_report

        sc = make_scorecard("a", "summarize", "t1", [Score("C", 70, 0.3)])
        result = generate_report([sc])
        # No filename section in middle
        # The summary table still has all columns
        assert "summarize" in result
        # The other task sections should not appear
        assert "FILENAME" not in result
        assert "FILE_SUMMARY" not in result


class TestBaseline:
    def test_save_baseline(self, mock_llm, tmp_path):
        from lib import quality_report

        sc = make_scorecard("a", "filename", "t1", [Score("R", 80, 0.4)])
        with patch.object(quality_report, "BASELINE_PATH", tmp_path / "bl.json"):
            result = quality_report.save_baseline([sc])
        assert "a::filename::t1" in result
        assert result["a::filename::t1"]["composite"] == 80 * 0.4

    def test_load_baseline_missing(self, mock_llm, tmp_path):
        from lib import quality_report

        with patch.object(quality_report, "BASELINE_PATH", tmp_path / "missing.json"):
            assert quality_report.load_baseline() == {}

    def test_load_baseline_exists(self, mock_llm, tmp_path):
        from lib import quality_report

        bl_path = tmp_path / "bl.json"
        bl_path.write_text('{"a": 1}')
        with patch.object(quality_report, "BASELINE_PATH", bl_path):
            assert quality_report.load_baseline() == {"a": 1}

    def test_compare_to_baseline_no_baseline(self, mock_llm, tmp_path):
        from lib import quality_report

        sc = make_scorecard("a", "filename", "t1", [Score("R", 80, 0.4)])
        with patch.object(quality_report, "BASELINE_PATH", tmp_path / "missing.json"):
            warnings = quality_report.compare_to_baseline([sc])
        assert "No baseline" in warnings[0]

    def test_compare_to_baseline_regression(self, mock_llm, tmp_path):
        from lib import quality_report

        # Current score 50, baseline 80 -> regression
        bl_path = tmp_path / "bl.json"
        bl_path.write_text(
            json.dumps(
                {
                    "a::filename::t1": {
                        "composite": 80.0,
                        "dimensions": {"Relevance": 80, "Format": 80},
                        "elapsed": 0.5,
                    }
                }
            )
        )
        sc = make_scorecard(
            "a",
            "filename",
            "t1",
            [Score("Relevance", 30, 0.4), Score("Format", 30, 0.35)],
        )
        with patch.object(quality_report, "BASELINE_PATH", bl_path):
            warnings = quality_report.compare_to_baseline([sc])
        assert any("REGRESSION" in w for w in warnings)

    def test_compare_to_baseline_improvement(self, mock_llm, tmp_path):
        from lib import quality_report

        bl_path = tmp_path / "bl.json"
        # Baseline composite 10, current composite 50 (e.g. two dims at 50, 50 with weights .5, .5)
        bl_path.write_text(
            json.dumps(
                {
                    "a::filename::t1": {
                        "composite": 10.0,
                        "dimensions": {},
                        "elapsed": 0.5,
                    }
                }
            )
        )
        sc = make_scorecard(
            "a",
            "filename",
            "t1",
            [Score("R", 50, 0.5), Score("F", 50, 0.5)],  # composite = 50
        )
        with patch.object(quality_report, "BASELINE_PATH", bl_path):
            warnings = quality_report.compare_to_baseline([sc])
        # delta = 50 - 10 = 40 > 10 -> improvement
        assert any("IMPROVEMENT" in w for w in warnings)

    def test_compare_to_baseline_unchanged(self, mock_llm, tmp_path):
        from lib import quality_report

        bl_path = tmp_path / "bl.json"
        bl_path.write_text(
            json.dumps(
                {
                    "a::filename::t1": {
                        "composite": 40.0,  # composite 40 = score 80 * weight 0.5
                        "dimensions": {},
                        "elapsed": 0.5,
                    }
                }
            )
        )
        sc = make_scorecard(
            "a",
            "filename",
            "t1",
            [Score("R", 80, 0.5)],  # composite = 40
        )
        with patch.object(quality_report, "BASELINE_PATH", bl_path):
            warnings = quality_report.compare_to_baseline([sc])
        # delta = 40 - 40 = 0, neither
        assert warnings == []

    def test_compare_skips_unknown_key(self, mock_llm, tmp_path):
        from lib import quality_report

        bl_path = tmp_path / "bl.json"
        bl_path.write_text(json.dumps({"other::filename::t1": {"composite": 80, "dimensions": {}}}))
        sc = make_scorecard("a", "filename", "t1", [Score("R", 80, 0.4)])
        with patch.object(quality_report, "BASELINE_PATH", bl_path):
            warnings = quality_report.compare_to_baseline([sc])
        # key not in baseline -> skipped
        assert warnings == []

    def test_compare_regression_with_dim_details(self, mock_llm, tmp_path):
        from lib import quality_report

        bl_path = tmp_path / "bl.json"
        bl_path.write_text(
            json.dumps(
                {
                    "a::filename::t1": {
                        "composite": 80.0,
                        "dimensions": {"Relevance": 80, "Format": 80},
                        "elapsed": 0.5,
                    }
                }
            )
        )
        sc = make_scorecard(
            "a",
            "filename",
            "t1",
            [Score("Relevance", 30, 0.4, failures=["bad"]), Score("Format", 90, 0.35)],
        )
        with patch.object(quality_report, "BASELINE_PATH", bl_path):
            warnings = quality_report.compare_to_baseline([sc])
        # dim_deltas should be populated for Relevance (-50)
        assert any("Relevance" in w for w in warnings)
