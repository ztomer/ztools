"""Tests for benchmark_output.py functions."""
import pytest
from unittest.mock import patch
import io


class TestPrintHeader:
    def test_print_header(self, capsys):
        from eval.benchmark_output import print_header
        all_cases = [("case1", "desc", ["a", "b", "c"]), ("case2", "desc2", ["d"])]
        print_header(["model-a", "model-b"], all_cases)
        captured = capsys.readouterr()
        assert "2 models" in captured.out
        assert "4 cases" in captured.out


class TestPrintModelHeader:
    def test_print_model_header(self, capsys):
        from eval.benchmark_output import print_model_header
        print_model_header("qwen3.6")
        captured = capsys.readouterr()
        assert "qwen3.6" in captured.out


class TestPrintCaseResult:
    def test_high_score_uses_step(self, capsys):
        from eval.benchmark_output import print_case_result
        print_case_result(80, 85, 1.5, "test case", "output", [])
        captured = capsys.readouterr()
        assert "H: 80" in captured.out
        assert "test case" in captured.out

    def test_medium_score_uses_warn(self, capsys):
        from eval.benchmark_output import print_case_result
        print_case_result(50, 60, 2.0, "medium case", "out", ["issue1"])
        captured = capsys.readouterr()
        assert "H: 50" in captured.out
        assert "issue1" in captured.out

    def test_low_score_uses_fail(self, capsys):
        from eval.benchmark_output import print_case_result
        print_case_result(20, 30, 0.5, "fail case", "bad out", ["a", "b"])
        captured = capsys.readouterr()
        assert "H: 20" in captured.out
        assert "issues" in captured.out


class TestPrintModelSummary:
    def test_print_model_summary(self, capsys):
        from eval.benchmark_output import print_model_summary
        print_model_summary("model-x", 75.5, 80.3, 5)
        captured = capsys.readouterr()
        assert "model-x" in captured.out
        assert "76/100" in captured.out
        assert "80/100" in captured.out
        assert "Gap" in captured.out


class TestPrintCrossModelComparison:
    def test_no_print_when_single_model(self, capsys):
        from eval.benchmark_output import print_cross_model_comparison
        print_cross_model_comparison({"m1": {"avg_human": 80, "avg_auto": 75, "gap": -5}})
        captured = capsys.readouterr()
        assert "CROSS-MODEL" not in captured.out

    def test_prints_table_when_multiple_models(self, capsys):
        from eval.benchmark_output import print_cross_model_comparison
        results = {
            "model-a": {"avg_human": 90, "avg_auto": 85, "gap": -5},
            "model-b": {"avg_human": 70, "avg_auto": 80, "gap": 10},
        }
        print_cross_model_comparison(results)
        captured = capsys.readouterr()
        assert "CROSS-MODEL COMPARISON" in captured.out
        assert "model-a" in captured.out
        assert "model-b" in captured.out
