"""Tests for lib.quality_entry - main CLI."""
import contextlib
import json
import sys
import pytest
from unittest.mock import patch


@pytest.fixture
def mock_llm():
    from lib.testing import MockLLM
    mock = MockLLM()
    mock.patch_all()
    yield mock
    mock.unpatch()


@contextlib.contextmanager
def _patch_argv(args):
    old = sys.argv
    sys.argv = args
    try:
        yield
    finally:
        sys.argv = old


class TestQualityEntryMain:
    def test_main_basic(self, mock_llm, capsys, tmp_path):
        from lib import quality_entry, quality_report
        with _patch_argv(["quality_entry.py"]), \
             patch.object(quality_entry, "BASELINE_PATH", tmp_path / "bl.json"), \
             patch.object(quality_report, "BASELINE_PATH", tmp_path / "bl.json"):
            quality_entry.main()
        captured = capsys.readouterr()
        assert "Quality Suite" in captured.out

    def test_main_with_models(self, mock_llm, capsys, tmp_path):
        from lib import quality_entry, quality_report
        with _patch_argv(["quality_entry.py", "--models", "model-a", "model-b"]), \
             patch.object(quality_entry, "BASELINE_PATH", tmp_path / "bl.json"), \
             patch.object(quality_report, "BASELINE_PATH", tmp_path / "bl.json"):
            quality_entry.main()
        captured = capsys.readouterr()
        assert "2 models" in captured.out

    def test_main_with_tasks(self, mock_llm, capsys, tmp_path):
        from lib import quality_entry, quality_report
        with _patch_argv(["quality_entry.py", "--tasks", "filename"]), \
             patch.object(quality_entry, "BASELINE_PATH", tmp_path / "bl.json"), \
             patch.object(quality_report, "BASELINE_PATH", tmp_path / "bl.json"):
            quality_entry.main()
        captured = capsys.readouterr()
        # Only filename cases
        assert "Quality Suite" in captured.out

    def test_main_save_baseline(self, mock_llm, capsys, tmp_path):
        from lib import quality_entry, quality_report
        bl_path = tmp_path / "bl.json"
        with _patch_argv(["quality_entry.py", "--models", "model-a",
                          "--tasks", "filename", "--save-baseline"]), \
             patch.object(quality_entry, "BASELINE_PATH", bl_path), \
             patch.object(quality_report, "BASELINE_PATH", bl_path):
            quality_entry.main()
        captured = capsys.readouterr()
        assert "Baseline saved" in captured.out
        assert bl_path.exists()

    def test_main_quiet(self, mock_llm, capsys, tmp_path):
        from lib import quality_entry, quality_report
        with _patch_argv(["quality_entry.py", "--models", "model-a",
                          "--tasks", "filename", "--quiet"]), \
             patch.object(quality_entry, "BASELINE_PATH", tmp_path / "bl.json"), \
             patch.object(quality_report, "BASELINE_PATH", tmp_path / "bl.json"):
            quality_entry.main()
        captured = capsys.readouterr()
        # Quiet mode suppresses per-case prints
        assert "Quality Suite" in captured.out

    def test_main_regression_only_no_baseline(self, mock_llm, capsys, tmp_path):
        from lib import quality_entry, quality_report
        with _patch_argv(["quality_entry.py", "--regression-only"]), \
             patch.object(quality_entry, "BASELINE_PATH", tmp_path / "missing.json"), \
             patch.object(quality_report, "BASELINE_PATH", tmp_path / "missing.json"):
            quality_entry.main()
        captured = capsys.readouterr()
        assert "No baseline found" in captured.out

    def test_main_regression_only_with_baseline(self, mock_llm, capsys, tmp_path):
        from lib import quality_entry, quality_report
        from lib.quality_models import Score, ScoreCard
        bl_path = tmp_path / "bl.json"
        # Create baseline with a known score
        sc = ScoreCard(
            model="model-a", task="filename", case_id="t1",
            dimensions=[Score("Relevance", 50, 0.4)], output="", elapsed=0.5
        )
        from lib.quality_report import save_baseline
        with patch.object(quality_entry, "BASELINE_PATH", bl_path), \
             patch.object(quality_report, "BASELINE_PATH", bl_path):
            save_baseline([sc])
        with _patch_argv(["quality_entry.py", "--regression-only"]), \
             patch.object(quality_entry, "BASELINE_PATH", bl_path), \
             patch.object(quality_report, "BASELINE_PATH", bl_path):
            quality_entry.main()
        captured = capsys.readouterr()
        assert "Loaded baseline" in captured.out

    def test_main_regression_only_bad_key(self, mock_llm, capsys, tmp_path):
        from lib import quality_entry, quality_report
        bl_path = tmp_path / "bl.json"
        bl_path.write_text(json.dumps({"badkey_no_double_colon": {"composite": 50, "dimensions": {}}}))
        with _patch_argv(["quality_entry.py", "--regression-only"]), \
             patch.object(quality_entry, "BASELINE_PATH", bl_path), \
             patch.object(quality_report, "BASELINE_PATH", bl_path):
            quality_entry.main()
        captured = capsys.readouterr()
        # Bad key skipped
        assert "Loaded baseline" in captured.out

    def test_main_regression_only_with_warning(self, mock_llm, capsys, tmp_path):
        from lib import quality_entry, quality_report
        from lib.quality_models import Score, ScoreCard
        bl_path = tmp_path / "bl.json"
        # Baseline with high score; after rebuild, score will be 0 (weight=0)
        sc = ScoreCard(
            model="model-a", task="filename", case_id="t1",
            dimensions=[Score("Relevance", 80, 0.4)], output="", elapsed=0.5
        )
        from lib.quality_report import save_baseline
        with patch.object(quality_entry, "BASELINE_PATH", bl_path), \
             patch.object(quality_report, "BASELINE_PATH", bl_path):
            save_baseline([sc])
        with _patch_argv(["quality_entry.py", "--regression-only"]), \
             patch.object(quality_entry, "BASELINE_PATH", bl_path), \
             patch.object(quality_report, "BASELINE_PATH", bl_path):
            quality_entry.main()
        captured = capsys.readouterr()
        # regression_only reconstructs scorecards with weight=0 -> composite=0
        # baseline composite = 32 (80*0.4), current = 0 -> delta = -32 -> REGRESSION
        assert "REGRESSION" in captured.out

    def test_main_no_regressions(self, mock_llm, capsys, tmp_path):
        """When no regressions are detected, prints 'No regressions detected.'"""
        from lib import quality_entry, quality_report
        from lib.quality_models import Score, ScoreCard
        bl_path = tmp_path / "bl.json"
        # Create a baseline that matches the current results (no regression)
        # Composite is score*weight. We need the baseline to be similar to what
        # the model will produce. Use a relevance score of 25 (which the mock produces).
        sc = ScoreCard(
            model="model-a", task="filename", case_id="Login error screenshot",
            dimensions=[
                Score("Relevance", 25, 0.4),
                Score("Format", 100, 0.35),
                Score("Conciseness", 100, 0.25),
            ],
            output="", elapsed=0.0
        )
        from lib.quality_report import save_baseline
        with patch.object(quality_entry, "BASELINE_PATH", bl_path), \
             patch.object(quality_report, "BASELINE_PATH", bl_path):
            save_baseline([sc])
        with _patch_argv(["quality_entry.py", "--models", "model-a",
                          "--tasks", "filename"]), \
             patch.object(quality_entry, "BASELINE_PATH", bl_path), \
             patch.object(quality_report, "BASELINE_PATH", bl_path):
            quality_entry.main()
        captured = capsys.readouterr()
        # The mock produces the same scores as baseline -> no regression
        assert "No regressions detected" in captured.out

    def test_dunder_main_block(self, mock_llm, capsys, tmp_path, monkeypatch):
        """Test that __name__ == '__main__' block invokes main()."""
        import runpy
        from lib import quality_report, quality_entry
        bl_path = tmp_path / "bl.json"
        monkeypatch.setattr(quality_report, "BASELINE_PATH", bl_path)
        monkeypatch.setattr(quality_entry, "BASELINE_PATH", bl_path)
        with _patch_argv(["quality_entry.py", "--models", "model-a", "--tasks", "filename"]):
            # Run the module as __main__ to execute the if __name__ == "__main__" block
            runpy.run_module("lib.quality_entry", run_name="__main__")
        captured = capsys.readouterr()
        assert "Quality Suite" in captured.out
