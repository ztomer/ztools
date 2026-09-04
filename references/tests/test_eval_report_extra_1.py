"""Tests for eval_report print functions and other uncovered code paths."""

import json
from io import StringIO
from unittest.mock import patch

from rich.console import Console


def _console_buffer():
    """A console that writes to a buffer, handed to the callee as `out=`.

    The print functions take their console as a parameter, so no test needs to
    rebind a module global — which is exactly what broke silently every time a
    function moved to another module during the split.
    """
    buf = StringIO()
    return buf, Console(file=buf, force_terminal=True, force_interactive=True, width=120)


class TestPrintCrossModelComparison:
    def test_empty_results(self):
        from eval import report as eval_report

        buf, out = _console_buffer()
        eval_report.print_cross_model_comparison([], out=out)
        assert "Cross-Model" not in buf.getvalue()

    def test_no_models(self):
        from eval import report as eval_report

        buf, out = _console_buffer()
        eval_report.print_cross_model_comparison([{"model": "m1", "results": []}], out=out)
        # Header printed but no rows (first_results is empty)
        rendered = buf.getvalue()
        assert "Cross-Model" in rendered
        # No table rows because first_results is empty
        assert "model_a" not in rendered

    def test_no_first_results(self):
        from eval import report as eval_report

        buf, out = _console_buffer()
        results = [
            {"model": "m1", "results": []},
            {"model": "m2", "results": []},
        ]
        eval_report.print_cross_model_comparison(results, out=out)
        rendered = buf.getvalue()
        # Header printed, no task rows (but model names appear in header)
        assert "Cross-Model" in rendered
        assert "Task" in rendered
        assert "m1" in rendered  # model name appears in header
        assert "m2" in rendered

    def test_full_table(self):
        from eval import report as eval_report

        buf, out = _console_buffer()
        results = [
            {
                "model": "model_a",
                "results": [
                    {"task": "t1", "quality_score": 80},
                    {"task": "t2", "quality_score": 90},
                ],
            },
            {
                "model": "model_b",
                "results": [
                    {"task": "t1", "quality_score": 95},
                    {"task": "t2", "quality_score": 70},
                ],
            },
        ]
        eval_report.print_cross_model_comparison(results, out=out)
        rendered = buf.getvalue()
        # Header printed
        assert "Cross-Model" in rendered
        # Both models in the table
        assert "model_a" in rendered
        assert "model_b" in rendered
        # Tasks rendered
        assert "t1" in rendered
        assert "t2" in rendered
        # Best score marker (*)
        assert "*" in rendered


class TestPrintScoreStats:
    def test_empty_stats(self):
        from eval import report as eval_report

        buf, out = _console_buffer()
        eval_report.print_score_stats({}, out=out)
        # Empty stats → no header printed
        assert "Mean" not in buf.getvalue()

    def test_full_stats(self):
        from eval import report as eval_report

        buf, out = _console_buffer()
        stats = {
            "m1": {"mean": 85.0, "median": 85.0, "stdev": 5.0, "min": 80, "max": 90},
            "m2": {"mean": 70.0, "median": 70.0, "stdev": 0.0, "min": 70, "max": 70},
        }
        eval_report.print_score_stats(stats, out=out)
        rendered = buf.getvalue()
        # Both models printed
        assert "m1" in rendered
        assert "m2" in rendered
        # Mean values
        assert "85.0" in rendered
        assert "70.0" in rendered
        # Header columns
        assert "Mean" in rendered
        assert "Stdev" in rendered


class TestPrintFailureSummary:
    def test_empty_categories(self):
        from eval import report as eval_report

        buf, out = _console_buffer()
        eval_report.print_failure_summary({}, out=out)
        # Empty → nothing printed
        assert buf.getvalue() == ""

    def test_with_categories(self):
        from eval import report as eval_report

        buf, out = _console_buffer()
        categories = {
            "FORMAT": {"count": 5, "models": ["m1", "m2"], "tasks": ["t1"]},
            "INFRA": {"count": 3, "models": ["m3"], "tasks": ["t2"]},
        }
        eval_report.print_failure_summary(categories, out=out)
        rendered = buf.getvalue()
        # Both categories printed
        assert "FORMAT" in rendered
        assert "INFRA" in rendered
        # Counts
        assert "5" in rendered
        assert "3" in rendered
        # Models mentioned
        assert "m1" in rendered or "m2" in out


class TestHistoricalFunctions:
    def test_save_and_load_history(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        # Redirect the config dir to tmp_path
        _EVAL_DIR = tmp_path
        results = [
            {
                "model": "m1",
                "results": [
                    {"task": "t1", "quality_score": 80, "time": 1.5},
                    {"task": "t2", "quality_score": 90, "time": 2.0},
                ],
            },
            {"model": "m2", "results": []},
        ]
        stats = {"m1": {"mean": 85, "median": 85, "stdev": 5, "min": 80, "max": 90, "count": 2}}
        categories = {"FORMAT": {"count": 1, "models": ["m1"], "tasks": ["t1"]}}
        eval_report.save_historical_results(results, stats, categories, eval_dir=_EVAL_DIR)
        # File should now exist
        hist_file = tmp_path / "eval_history.json"
        assert hist_file.exists()
        # Loading should work
        loaded = eval_report.load_historical_stats(eval_dir=_EVAL_DIR)
        assert "m1" in loaded
        # Check specific fields
        assert loaded["m1"]["mean"] == 85.0
        assert loaded["m1"]["min"] == 80
        assert loaded["m1"]["max"] == 90

    def test_save_truncates_to_100(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        _EVAL_DIR = tmp_path
        # Generate 110 results for one model
        results = [
            {
                "model": "m1",
                "results": [
                    {"task": f"t{i}", "quality_score": 50, "time": 1.0} for i in range(110)
                ],
            }
        ]
        eval_report.save_historical_results(results, {}, {}, eval_dir=_EVAL_DIR)
        hist_file = tmp_path / "eval_history.json"
        with open(hist_file) as f:
            data = json.load(f)
        assert len(data["m1"]) == 100  # truncated

    def test_save_with_invalid_existing_history(self, tmp_path, monkeypatch):
        """Lines 146-150: existing history file has invalid JSON, exception caught."""
        from eval import report as eval_report

        _EVAL_DIR = tmp_path
        hist_file = tmp_path / "eval_history.json"
        hist_file.write_text("{not valid json")
        results = [
            {
                "model": "m1",
                "results": [
                    {"task": "t1", "quality_score": 80, "time": 1.0},
                ],
            }
        ]
        # Should not raise — just silently replaces with new data
        eval_report.save_historical_results(results, {}, {}, eval_dir=_EVAL_DIR)
        # New data should be saved
        with open(hist_file) as f:
            data = json.load(f)
        assert "m1" in data

    def test_load_history_no_file(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        _EVAL_DIR = tmp_path
        result = eval_report.load_historical_stats(eval_dir=_EVAL_DIR)
        assert result == {}

    def test_load_history_invalid_json(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        _EVAL_DIR = tmp_path
        hist_file = tmp_path / "eval_history.json"
        hist_file.write_text("{invalid json")
        result = eval_report.load_historical_stats(eval_dir=_EVAL_DIR)
        assert result == {}

    def test_load_history_empty(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        _EVAL_DIR = tmp_path
        hist_file = tmp_path / "eval_history.json"
        hist_file.write_text("{}")
        result = eval_report.load_historical_stats(eval_dir=_EVAL_DIR)
        assert result == {}

    def test_load_history_no_scores(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        _EVAL_DIR = tmp_path
        hist_file = tmp_path / "eval_history.json"
        hist_file.write_text(json.dumps({"m1": [{"date": "2024-01-01"}]}))  # no score
        result = eval_report.load_historical_stats(eval_dir=_EVAL_DIR)
        assert "m1" not in result

    def test_check_model_history_no_file(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        _EVAL_DIR = tmp_path
        result = eval_report.check_model_history("m1", eval_dir=_EVAL_DIR)
        assert result == {}

    def test_check_model_history_invalid_json(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        _EVAL_DIR = tmp_path
        hist_file = tmp_path / "eval_history.json"
        hist_file.write_text("{not json")
        result = eval_report.check_model_history("m1", eval_dir=_EVAL_DIR)
        assert result == {}

    def test_check_model_history_found(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        _EVAL_DIR = tmp_path
        hist_file = tmp_path / "eval_history.json"
        hist_file.write_text(
            json.dumps(
                {
                    "m1": [{"score": 80}, {"score": 90}],
                    "m2": [{"score": 50}],
                }
            )
        )
        result = eval_report.check_model_history("m1", eval_dir=_EVAL_DIR)
        assert len(result) == 2


class TestPrintHistoricalTrends:
    def test_no_stats(self):
        from eval import report as eval_report

        buf, out = _console_buffer()
        # print_historical_trends calls load_historical_stats from its own
        # module, so the shim's re-exported name is not the seam.
        with patch("eval.report_history.load_historical_stats", return_value={}):
            eval_report.print_historical_trends(out=out)
        # No stats → no output
        assert "Historical" not in buf.getvalue() or buf.getvalue() == ""

    def test_with_stats(self):
        from eval import report as eval_report

        buf, out = _console_buffer()
        fake_stats = {
            "m1": {"mean": 85, "stdev": 3, "runs": 10},  # stable
            "m2": {"mean": 70, "stdev": 10, "runs": 5},  # variable
            "m3": {"mean": 60, "stdev": 20, "runs": 8},  # unstable
            "m4": {"mean": 90, "stdev": 0, "runs": 1},  # new (runs < 3)
        }
        with patch("eval.report_history.load_historical_stats", return_value=fake_stats):
            eval_report.print_historical_trends(out=out)
        rendered = buf.getvalue()
        # At least the section header is printed
        assert "Historical" in rendered or "Trend" in rendered


class TestPrintVerbosity:
    def test_empty_verbosity(self):
        from eval import report as eval_report

        buf, out = _console_buffer()
        eval_report.print_verbosity({}, out=out)
        # Empty → no output
        assert buf.getvalue() == ""

    def test_with_verbosity(self):
        from eval import report as eval_report

        buf, out = _console_buffer()
        verbosity = {
            "m1": {"t1": 100, "t2": 200},
            "m2": {"t1": 50, "t2": 75},
        }
        eval_report.print_verbosity(verbosity, out=out)
        rendered = buf.getvalue()
        # Both models printed
        assert "m1" in rendered
        assert "m2" in rendered


class TestPrintErrorRates:
    def test_empty_rates(self):
        from eval import report as eval_report

        buf, out = _console_buffer()
        eval_report.print_error_rates({}, out=out)
        # Empty → no output
        assert buf.getvalue() == ""

    def test_with_rates(self):
        from eval import report as eval_report

        buf, out = _console_buffer()
        rates = {
            "m1": {
                "infra": 1,
                "quality": 2,
                "success": 5,
                "infra_rate": 0.125,
                "quality_rate": 0.25,
                "success_rate": 0.625,
            },
            "m2": {
                "infra": 0,
                "quality": 0,
                "success": 10,
                "infra_rate": 0,
                "quality_rate": 0,
                "success_rate": 1.0,
            },
        }
        eval_report.print_error_rates(rates, out=out)
        rendered = buf.getvalue()
        # Both models printed
        assert "m1" in rendered
        assert "m2" in rendered
        # At least one rate is visible
        assert "62" in rendered or "100" in out or "%" in out
