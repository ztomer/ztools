"""Tests for eval_report print functions and other uncovered code paths."""

import json
from io import StringIO
from unittest.mock import patch

from rich.console import Console


def _capture_rich_console():
    """Replace eval_report.console with one that writes to a StringIO buffer."""
    from eval import report as eval_report

    buf = StringIO()
    new_console = Console(file=buf, force_terminal=True, force_interactive=True, width=120)
    return eval_report.console, new_console, buf


class TestPrintCrossModelComparison:
    def test_empty_results(self):
        from eval import report as eval_report

        old, new, buf = _capture_rich_console()
        try:
            eval_report.console = new
            eval_report.print_cross_model_comparison([])
        finally:
            eval_report.console = old
        assert "Cross-Model" not in buf.getvalue()

    def test_no_models(self):
        from eval import report as eval_report

        old, new, buf = _capture_rich_console()
        try:
            eval_report.console = new
            eval_report.print_cross_model_comparison([{"model": "m1", "results": []}])
        finally:
            eval_report.console = old
        # Header printed but no rows (first_results is empty)
        out = buf.getvalue()
        assert "Cross-Model" in out
        # No table rows because first_results is empty
        assert "model_a" not in out

    def test_no_first_results(self):
        from eval import report as eval_report

        old, new, buf = _capture_rich_console()
        try:
            eval_report.console = new
            results = [
                {"model": "m1", "results": []},
                {"model": "m2", "results": []},
            ]
            eval_report.print_cross_model_comparison(results)
        finally:
            eval_report.console = old
        out = buf.getvalue()
        # Header printed, no task rows (but model names appear in header)
        assert "Cross-Model" in out
        assert "Task" in out
        assert "m1" in out  # model name appears in header
        assert "m2" in out

    def test_full_table(self):
        from eval import report as eval_report

        old, new, buf = _capture_rich_console()
        try:
            eval_report.console = new
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
            eval_report.print_cross_model_comparison(results)
        finally:
            eval_report.console = old
        out = buf.getvalue()
        # Header printed
        assert "Cross-Model" in out
        # Both models in the table
        assert "model_a" in out
        assert "model_b" in out
        # Tasks rendered
        assert "t1" in out
        assert "t2" in out
        # Best score marker (*)
        assert "*" in out


class TestPrintScoreStats:
    def test_empty_stats(self):
        from eval import report as eval_report

        old, new, buf = _capture_rich_console()
        try:
            eval_report.console = new
            eval_report.print_score_stats({})
        finally:
            eval_report.console = old
        # Empty stats → no header printed
        assert "Mean" not in buf.getvalue()

    def test_full_stats(self):
        from eval import report as eval_report

        old, new, buf = _capture_rich_console()
        try:
            eval_report.console = new
            stats = {
                "m1": {"mean": 85.0, "median": 85.0, "stdev": 5.0, "min": 80, "max": 90},
                "m2": {"mean": 70.0, "median": 70.0, "stdev": 0.0, "min": 70, "max": 70},
            }
            eval_report.print_score_stats(stats)
        finally:
            eval_report.console = old
        out = buf.getvalue()
        # Both models printed
        assert "m1" in out
        assert "m2" in out
        # Mean values
        assert "85.0" in out
        assert "70.0" in out
        # Header columns
        assert "Mean" in out
        assert "Stdev" in out


class TestPrintFailureSummary:
    def test_empty_categories(self):
        from eval import report as eval_report

        old, new, buf = _capture_rich_console()
        try:
            eval_report.console = new
            eval_report.print_failure_summary({})
        finally:
            eval_report.console = old
        # Empty → nothing printed
        assert buf.getvalue() == ""

    def test_with_categories(self):
        from eval import report as eval_report

        old, new, buf = _capture_rich_console()
        try:
            eval_report.console = new
            categories = {
                "FORMAT": {"count": 5, "models": ["m1", "m2"], "tasks": ["t1"]},
                "INFRA": {"count": 3, "models": ["m3"], "tasks": ["t2"]},
            }
            eval_report.print_failure_summary(categories)
        finally:
            eval_report.console = old
        out = buf.getvalue()
        # Both categories printed
        assert "FORMAT" in out
        assert "INFRA" in out
        # Counts
        assert "5" in out
        assert "3" in out
        # Models mentioned
        assert "m1" in out or "m2" in out


class TestHistoricalFunctions:
    def test_save_and_load_history(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        # Redirect the config dir to tmp_path
        monkeypatch.setattr("eval.report._get_eval_dir", lambda: tmp_path)
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
        eval_report.save_historical_results(results, stats, categories)
        # File should now exist
        hist_file = tmp_path / "eval_history.json"
        assert hist_file.exists()
        # Loading should work
        loaded = eval_report.load_historical_stats()
        assert "m1" in loaded
        # Check specific fields
        assert loaded["m1"]["mean"] == 85.0
        assert loaded["m1"]["min"] == 80
        assert loaded["m1"]["max"] == 90

    def test_save_truncates_to_100(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        monkeypatch.setattr("eval.report._get_eval_dir", lambda: tmp_path)
        # Generate 110 results for one model
        results = [
            {
                "model": "m1",
                "results": [
                    {"task": f"t{i}", "quality_score": 50, "time": 1.0} for i in range(110)
                ],
            }
        ]
        eval_report.save_historical_results(results, {}, {})
        hist_file = tmp_path / "eval_history.json"
        with open(hist_file) as f:
            data = json.load(f)
        assert len(data["m1"]) == 100  # truncated

    def test_save_with_invalid_existing_history(self, tmp_path, monkeypatch):
        """Lines 146-150: existing history file has invalid JSON, exception caught."""
        from eval import report as eval_report

        monkeypatch.setattr("eval.report._get_eval_dir", lambda: tmp_path)
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
        eval_report.save_historical_results(results, {}, {})
        # New data should be saved
        with open(hist_file) as f:
            data = json.load(f)
        assert "m1" in data

    def test_load_history_no_file(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        monkeypatch.setattr("eval.report._get_eval_dir", lambda: tmp_path)
        result = eval_report.load_historical_stats()
        assert result == {}

    def test_load_history_invalid_json(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        monkeypatch.setattr("eval.report._get_eval_dir", lambda: tmp_path)
        hist_file = tmp_path / "eval_history.json"
        hist_file.write_text("{invalid json")
        result = eval_report.load_historical_stats()
        assert result == {}

    def test_load_history_empty(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        monkeypatch.setattr("eval.report._get_eval_dir", lambda: tmp_path)
        hist_file = tmp_path / "eval_history.json"
        hist_file.write_text("{}")
        result = eval_report.load_historical_stats()
        assert result == {}

    def test_load_history_no_scores(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        monkeypatch.setattr("eval.report._get_eval_dir", lambda: tmp_path)
        hist_file = tmp_path / "eval_history.json"
        hist_file.write_text(json.dumps({"m1": [{"date": "2024-01-01"}]}))  # no score
        result = eval_report.load_historical_stats()
        assert "m1" not in result

    def test_check_model_history_no_file(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        monkeypatch.setattr("eval.report._get_eval_dir", lambda: tmp_path)
        result = eval_report.check_model_history("m1")
        assert result == {}

    def test_check_model_history_invalid_json(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        monkeypatch.setattr("eval.report._get_eval_dir", lambda: tmp_path)
        hist_file = tmp_path / "eval_history.json"
        hist_file.write_text("{not json")
        result = eval_report.check_model_history("m1")
        assert result == {}

    def test_check_model_history_found(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        monkeypatch.setattr("eval.report._get_eval_dir", lambda: tmp_path)
        hist_file = tmp_path / "eval_history.json"
        hist_file.write_text(
            json.dumps(
                {
                    "m1": [{"score": 80}, {"score": 90}],
                    "m2": [{"score": 50}],
                }
            )
        )
        result = eval_report.check_model_history("m1")
        assert len(result) == 2


class TestPrintHistoricalTrends:
    def test_no_stats(self):
        from eval import report as eval_report

        old, new, buf = _capture_rich_console()
        try:
            eval_report.console = new
            with patch.object(eval_report, "load_historical_stats", return_value={}):
                eval_report.print_historical_trends()
        finally:
            eval_report.console = old
        # No stats → no output
        assert "Historical" not in buf.getvalue() or buf.getvalue() == ""

    def test_with_stats(self):
        from eval import report as eval_report

        old, new, buf = _capture_rich_console()
        try:
            eval_report.console = new
            fake_stats = {
                "m1": {"mean": 85, "stdev": 3, "runs": 10},  # stable
                "m2": {"mean": 70, "stdev": 10, "runs": 5},  # variable
                "m3": {"mean": 60, "stdev": 20, "runs": 8},  # unstable
                "m4": {"mean": 90, "stdev": 0, "runs": 1},  # new (runs < 3)
            }
            with patch.object(eval_report, "load_historical_stats", return_value=fake_stats):
                eval_report.print_historical_trends()
        finally:
            eval_report.console = old
        out = buf.getvalue()
        # At least the section header is printed
        assert "Historical" in out or "Trend" in out or len(out) > 0


class TestPrintVerbosity:
    def test_empty_verbosity(self):
        from eval import report as eval_report

        old, new, buf = _capture_rich_console()
        try:
            eval_report.console = new
            eval_report.print_verbosity({})
        finally:
            eval_report.console = old
        # Empty → no output
        assert buf.getvalue() == ""

    def test_with_verbosity(self):
        from eval import report as eval_report

        old, new, buf = _capture_rich_console()
        try:
            eval_report.console = new
            verbosity = {
                "m1": {"t1": 100, "t2": 200},
                "m2": {"t1": 50, "t2": 75},
            }
            eval_report.print_verbosity(verbosity)
        finally:
            eval_report.console = old
        out = buf.getvalue()
        # Both models printed
        assert "m1" in out
        assert "m2" in out


class TestPrintErrorRates:
    def test_empty_rates(self):
        from eval import report as eval_report

        old, new, buf = _capture_rich_console()
        try:
            eval_report.console = new
            eval_report.print_error_rates({})
        finally:
            eval_report.console = old
        # Empty → no output
        assert buf.getvalue() == ""

    def test_with_rates(self):
        from eval import report as eval_report

        old, new, buf = _capture_rich_console()
        try:
            eval_report.console = new
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
            eval_report.print_error_rates(rates)
        finally:
            eval_report.console = old
        out = buf.getvalue()
        # Both models printed
        assert "m1" in out
        assert "m2" in out
        # At least one rate is visible
        assert "62" in out or "100" in out or "%" in out


class TestDiffFromLastRun:
    def test_no_prev_file(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        monkeypatch.setattr("eval.report._get_eval_dir", lambda: tmp_path)
        result = eval_report.diff_from_last_run([])
        assert result == {}

    def test_invalid_prev_json(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        monkeypatch.setattr("eval.report._get_eval_dir", lambda: tmp_path)
        prev = tmp_path / "eval_results.json"
        prev.write_text("{invalid")
        result = eval_report.diff_from_last_run([])
        assert result == {}

    def test_no_models_in_prev(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        monkeypatch.setattr("eval.report._get_eval_dir", lambda: tmp_path)
        prev = tmp_path / "eval_results.json"
        prev.write_text(json.dumps({}))  # no models key
        result = eval_report.diff_from_last_run([])
        assert result == {}

    def test_no_matching_model(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        monkeypatch.setattr("eval.report._get_eval_dir", lambda: tmp_path)
        prev = tmp_path / "eval_results.json"
        prev.write_text(json.dumps({"models": [{"model": "other", "results": []}]}))
        result = eval_report.diff_from_last_run([{"model": "m1", "results": []}])
        assert result == {}

    def test_with_diffs(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        monkeypatch.setattr("eval.report._get_eval_dir", lambda: tmp_path)
        prev = tmp_path / "eval_results.json"
        prev.write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "model": "m1",
                            "results": [
                                {"task": "t1", "quality_score": 80},
                                {"task": "t2", "quality_score": 90},
                            ],
                        }
                    ]
                }
            )
        )
        result = eval_report.diff_from_last_run(
            [
                {
                    "model": "m1",
                    "results": [
                        {"task": "t1", "quality_score": 85},  # +5
                        {"task": "t2", "quality_score": 90},  # 0 (not in diffs)
                    ],
                }
            ]
        )
        assert "m1" in result
        assert "t1" in result["m1"]
        assert result["m1"]["t1"]["diff"] == 5


class TestPrintDiff:
    def test_empty_diffs(self):
        from eval import report as eval_report

        old, new, buf = _capture_rich_console()
        try:
            eval_report.console = new
            eval_report.print_diff({})
        finally:
            eval_report.console = old
        # Empty → no output
        assert buf.getvalue() == ""

    def test_no_changes(self):
        from eval import report as eval_report

        old, new, buf = _capture_rich_console()
        try:
            eval_report.console = new
            diffs = {"m1": {"t1": {"current": 80, "prev": 80, "diff": 0}}}
            eval_report.print_diff(diffs)
        finally:
            eval_report.console = old
        out = buf.getvalue()
        # No changes → no table printed (only blank line at top)
        assert "Model" not in out
        assert "Diff" not in out

    def test_with_changes(self):
        from eval import report as eval_report

        old, new, buf = _capture_rich_console()
        try:
            eval_report.console = new
            diffs = {
                "m1": {
                    "t1": {"current": 85, "prev": 80, "diff": 5},
                    "t2": {"current": 70, "prev": 80, "diff": -10},
                }
            }
            eval_report.print_diff(diffs)
        finally:
            eval_report.console = old
        out = buf.getvalue()
        # Both tasks printed with their diffs
        assert "m1" in out
        assert "t1" in out
        assert "t2" in out
        # Arrow characters present (up for +5, down for -10)
        assert "\u2191" in out
        assert "\u2193" in out
        # Diff magnitudes shown
        assert "5" in out
        assert "10" in out


class TestExportToCsv:
    def test_default_path(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        monkeypatch.setattr("eval.report._get_eval_dir", lambda: tmp_path)
        old, new, buf = _capture_rich_console()
        try:
            eval_report.console = new
            results = [
                {
                    "model": "m1",
                    "results": [
                        {
                            "task": "t1",
                            "quality_score": 95,
                            "time": 1.5,
                            "failure_reason": "",
                            "failure_category": "",
                        },
                        {
                            "task": "t2",
                            "quality_score": 60,
                            "time": 2.0,
                            "failure_reason": "x",
                            "failure_category": "FORMAT",
                        },
                        {
                            "task": "t3",
                            "quality_score": 30,
                            "time": 3.0,
                            "failure_reason": "y",
                            "failure_category": "INFRA",
                        },
                    ],
                },
            ]
            eval_report.export_to_csv(results)
            csv_file = tmp_path / "eval_results.csv"
            assert csv_file.exists()
            content = csv_file.read_text()
            assert "Model" in content
            assert "PASS" in content  # score >= 90
            assert "WARN" in content  # score >= 50
            assert "FAIL" in content  # score < 50
        finally:
            eval_report.console = old

    def test_custom_path(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        monkeypatch.setattr("eval.report._get_eval_dir", lambda: tmp_path)
        old, new, buf = _capture_rich_console()
        try:
            eval_report.console = new
            output = tmp_path / "custom.csv"
            results = [
                {"model": "m1", "results": [{"task": "t1", "quality_score": 80, "time": 1.0}]}
            ]
            eval_report.export_to_csv(results, str(output))
            assert output.exists()
            # Verify file was written to custom path, not default
            content = output.read_text()
            assert "m1" in content
            assert "t1" in content
        finally:
            eval_report.console = old
