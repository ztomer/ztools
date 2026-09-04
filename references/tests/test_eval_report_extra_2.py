"""Tests for eval_report print functions and other uncovered code paths."""

import json
from io import StringIO

from rich.console import Console


def _console_buffer():
    """A console that writes to a buffer, handed to the callee as `out=`.

    The print functions take their console as a parameter, so no test needs to
    rebind a module global — which is exactly what broke silently every time a
    function moved to another module during the split.
    """
    buf = StringIO()
    return buf, Console(file=buf, force_terminal=True, force_interactive=True, width=120)


class TestDiffFromLastRun:
    def test_no_prev_file(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        _EVAL_DIR = tmp_path
        result = eval_report.diff_from_last_run([], eval_dir=_EVAL_DIR)
        assert result == {}

    def test_invalid_prev_json(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        _EVAL_DIR = tmp_path
        prev = tmp_path / "eval_results.json"
        prev.write_text("{invalid")
        result = eval_report.diff_from_last_run([], eval_dir=_EVAL_DIR)
        assert result == {}

    def test_no_models_in_prev(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        _EVAL_DIR = tmp_path
        prev = tmp_path / "eval_results.json"
        prev.write_text(json.dumps({}))  # no models key
        result = eval_report.diff_from_last_run([], eval_dir=_EVAL_DIR)
        assert result == {}

    def test_no_matching_model(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        _EVAL_DIR = tmp_path
        prev = tmp_path / "eval_results.json"
        prev.write_text(json.dumps({"models": [{"model": "other", "results": []}]}))
        result = eval_report.diff_from_last_run([{"model": "m1", "results": []}], eval_dir=_EVAL_DIR)
        assert result == {}

    def test_with_diffs(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        _EVAL_DIR = tmp_path
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
            ],
            eval_dir=_EVAL_DIR,
        )
        assert "m1" in result
        assert "t1" in result["m1"]
        assert result["m1"]["t1"]["diff"] == 5


class TestPrintDiff:
    def test_empty_diffs(self):
        from eval import report as eval_report

        buf, out = _console_buffer()
        eval_report.print_diff({}, out=out)
        # Empty → no output
        assert buf.getvalue() == ""

    def test_no_changes(self):
        from eval import report as eval_report

        buf, out = _console_buffer()
        diffs = {"m1": {"t1": {"current": 80, "prev": 80, "diff": 0}}}
        eval_report.print_diff(diffs, out=out)
        rendered = buf.getvalue()
        # No changes → no table printed (only blank line at top)
        assert "Model" not in rendered
        assert "Diff" not in rendered

    def test_with_changes(self):
        from eval import report as eval_report

        buf, out = _console_buffer()
        diffs = {
            "m1": {
                "t1": {"current": 85, "prev": 80, "diff": 5},
                "t2": {"current": 70, "prev": 80, "diff": -10},
            }
        }
        eval_report.print_diff(diffs, out=out)
        rendered = buf.getvalue()
        # Both tasks printed with their diffs
        assert "m1" in rendered
        assert "t1" in rendered
        assert "t2" in rendered
        # Arrow characters present (up for +5, down for -10)
        assert "\u2191" in rendered
        assert "\u2193" in rendered
        # Diff magnitudes shown
        assert "5" in rendered
        assert "10" in rendered


class TestExportToCsv:
    def test_default_path(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        _EVAL_DIR = tmp_path
        buf, out = _console_buffer()
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
        eval_report.export_to_csv(results, out=out, eval_dir=_EVAL_DIR)
        csv_file = tmp_path / "eval_results.csv"
        assert csv_file.exists()
        content = csv_file.read_text()
        assert "Model" in content
        assert "PASS" in content  # score >= 90
        assert "WARN" in content  # score >= 50
        assert "FAIL" in content  # score < 50

    def test_custom_path(self, tmp_path, monkeypatch):
        from eval import report as eval_report

        _EVAL_DIR = tmp_path
        buf, out = _console_buffer()
        output = tmp_path / "custom.csv"
        results = [
            {"model": "m1", "results": [{"task": "t1", "quality_score": 80, "time": 1.0}]}
        ]
        eval_report.export_to_csv(results, str(output), out=out, eval_dir=_EVAL_DIR)
        assert output.exists()
        # Verify file was written to custom path, not default
        content = output.read_text()
        assert "m1" in content
        assert "t1" in content
