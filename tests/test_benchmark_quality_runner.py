"""Tests for benchmark_quality: query_model, run_benchmark, __main__."""
import pytest
from unittest.mock import patch, MagicMock


class TestQueryModel:
    def test_query_model_success(self, mock_llm):
        from eval import benchmark_quality as bq
        with patch.object(bq, "llm_call", return_value={"content": "result.txt"}):
            result = bq.query_model("model-x", "system prompt", "input", "filename")
        assert result == "result.txt"

    def test_query_model_exception(self, mock_llm):
        from eval import benchmark_quality as bq
        with patch.object(bq, "llm_call", side_effect=Exception("API error")):
            result = bq.query_model("model-x", "system", "input", "filename")
        assert result is None


class TestRunBenchmark:
    def test_run_benchmark_with_models(self, mock_llm, capsys):
        """run_benchmark calls query_model for each model x task x case combo."""
        from eval import benchmark_quality as bq
        with patch.object(bq, "get_model_prompt", return_value="test prompt"), \
             patch.object(bq, "query_model", return_value="login_error_invalid.png") as mock_q, \
             patch.object(bq, "print_header") as mock_header, \
             patch.object(bq, "print_model_header"), \
             patch.object(bq, "print_case_result"), \
             patch.object(bq, "print_model_summary") as mock_summary, \
             patch.object(bq, "print_cross_model_comparison"):
            bq.run_benchmark(["model-a", "model-b"], verbose=True)
        # print_header is called once with the models list and ALL_CASES
        mock_header.assert_called_once()
        args, _ = mock_header.call_args
        assert args[0] == ["model-a", "model-b"]
        assert len(args[1]) == 3  # 3 task groups
        # query_model is called once per (model x task x case) — 2 models x 7 cases
        # (5 filename + 1 summarize + 1 file_summary) = 14 calls
        assert mock_q.call_count == 2 * (5 + 1 + 1)
        models_called = [c.args[0] for c in mock_q.call_args_list]
        assert "model-a" in models_called
        assert "model-b" in models_called
        # summary printed once per model
        assert mock_summary.call_count == 2

    def test_run_benchmark_default_models(self, mock_llm, capsys):
        """run_benchmark() with no args uses get_filename_models() from config."""
        from eval import benchmark_quality as bq
        from lib import config as lib_config
        dummy_models = ["model-a", "model-b"]
        with patch.object(bq, "get_model_prompt", return_value=None), \
             patch.object(lib_config, "get_filename_models", return_value=dummy_models), \
             patch.object(bq, "print_header") as mock_header, \
             patch.object(bq, "print_model_header"), \
             patch.object(bq, "print_model_summary") as mock_summary, \
             patch.object(bq, "print_cross_model_comparison"):
            bq.run_benchmark()
        args, _ = mock_header.call_args
        assert args[0] == dummy_models
        mock_summary.assert_not_called()

    def test_run_benchmark_no_prompt_skips_task(self, mock_llm, capsys):
        """When get_model_prompt returns None, that task is skipped."""
        from eval import benchmark_quality as bq
        with patch.object(bq, "get_model_prompt", return_value=None), \
             patch.object(bq, "query_model") as mock_q, \
             patch.object(bq, "print_header"), \
             patch.object(bq, "print_model_header"), \
             patch.object(bq, "print_model_summary"), \
             patch.object(bq, "print_cross_model_comparison"):
            bq.run_benchmark(["m1"], verbose=True)
        # All tasks skipped → query_model never invoked
        mock_q.assert_not_called()

    def test_run_benchmark_none_output_skips_case(self, mock_llm, capsys):
        """When query_model returns None, the case is skipped (no scoring)."""
        from eval import benchmark_quality as bq
        with patch.object(bq, "get_model_prompt", return_value="prompt"), \
             patch.object(bq, "query_model", return_value=None), \
             patch.object(bq, "print_header"), \
             patch.object(bq, "print_model_header"), \
             patch.object(bq, "print_case_result") as mock_case, \
             patch.object(bq, "print_model_summary"), \
             patch.object(bq, "print_cross_model_comparison"):
            bq.run_benchmark(["m1"], verbose=True)
        # None output → continue before print_case_result
        mock_case.assert_not_called()

    def test_run_benchmark_quiet_mode(self, mock_llm, capsys):
        """verbose=False skips per-case printing."""
        from eval import benchmark_quality as bq
        with patch.object(bq, "get_model_prompt", return_value="prompt"), \
             patch.object(bq, "query_model", return_value="login_error.png"), \
             patch.object(bq, "print_header"), \
             patch.object(bq, "print_model_header"), \
             patch.object(bq, "print_case_result") as mock_case, \
             patch.object(bq, "print_model_summary") as mock_summary, \
             patch.object(bq, "print_cross_model_comparison"):
            bq.run_benchmark(["m1"], verbose=False)
        # verbose=False → no per-case print, but summary still printed
        mock_case.assert_not_called()
        assert mock_summary.call_count == 1

    def test_run_benchmark_scoring_correctness(self, mock_llm, capsys):
        """run_benchmark computes correct avg_human/avg_auto/gap from scorer output.

        Uses the REAL score_filename on each FILENAME_CASE so we exercise the
        actual scoring code. Returns good outputs for each case and verifies
        the aggregated summary numbers.
        """
        from eval import benchmark_quality as bq
        # Build a query_model that returns a perfect output for each filename case
        # by referencing the real FILENAME_CASES
        from eval.benchmark_quality import FILENAME_CASES, SUMMARIZE_CASES, FILE_SUMMARY_CASES
        from lib.config import Task
        # Perfect filename outputs: each contains all expected keywords, lowercase,
        # no spaces, no invalid chars, < 60 chars
        perfect_outputs = [
            "login_error_invalid_credentials.png",  # case 0: login, error, invalid, credential
            "summer_festival_2024_central_park.txt",  # case 1: summer, festival, 2024, park
            "scott_adams_failure_ambition_quotes.txt",  # case 2: scott, adam, failure, ambition
            "screen_shot_2024_03_15.png",  # case 3: screen, shot
            "meeting_notes_project_alpha_q1_review.txt",  # case 4: meeting, note, alpha, review
        ]
        # Summarize & file_summary: just return enough text to score
        summarize_out = ("This is a summary of the document. " * 10)[:500]
        file_summary_out = ("This is a file summary. " * 10)[:500]

        outputs = []
        # 5 filename + 1 summarize + 1 file_summary
        for _ in FILENAME_CASES:
            outputs.append(perfect_outputs[len(outputs) if len(outputs) < 5 else 0])
        outputs.append(summarize_out)
        outputs.append(file_summary_out)
        # Need to repeat for each model — but we use 1 model below

        # Return a single output that always passes the real scorer for filename
        def qm_side_effect(model, prompt, input_text, task):
            if task == "filename":
                # Match by input to get the right perfect output
                for i, case in enumerate(FILENAME_CASES):
                    if case["input"] == input_text:
                        return perfect_outputs[i]
                return perfect_outputs[0]
            return summarize_out if task == "summarize" else file_summary_out

        # get_model_prompt only returns a prompt for the FILENAME task
        def prompt_filter(model, task):
            return "p" if task == Task.FILENAME else None

        with patch.object(bq, "get_model_prompt", side_effect=prompt_filter), \
             patch.object(bq, "query_model", side_effect=qm_side_effect) as mock_q, \
             patch.object(bq, "print_header"), \
             patch.object(bq, "print_model_header"), \
             patch.object(bq, "print_case_result"), \
             patch.object(bq, "print_model_summary") as mock_summary, \
             patch.object(bq, "print_cross_model_comparison"):
            bq.run_benchmark(["m1"], verbose=True)
        # All 5 filename cases ran, no summarize/file_summary (no prompt)
        assert mock_q.call_count == 5
        # Summary printed once with correct aggregated numbers
        assert mock_summary.call_count == 1
        model_arg, avg_human, avg_auto, count = mock_summary.call_args.args
        assert model_arg == "m1"
        assert count == 5
        # All perfect outputs should score 100 on both human and auto
        assert avg_human == 100.0
        assert avg_auto == 100.0


class TestMainBlock:
    """Tests for the if __name__ == '__main__' block in benchmark_quality.py.

    We extract the actual __main__ block source from the file and exec it
    with mocked run_benchmark. This ensures we test the REAL __main__ block,
    not a re-implementation.
    """

    @staticmethod
    def _get_main_block_source():
        """Read and return the __main__ block body source from eval/benchmark_quality.py.

        Returns the body with common leading whitespace removed so it can be
        exec'd at module level.
        """
        import re
        import textwrap
        with open("eval/benchmark_quality.py") as f:
            source = f.read()
        # Match: if __name__ == "__main__":\n    <body lines>
        match = re.search(
            r'if __name__ == "__main__":\n((?:    .*\n)+)',
            source,
        )
        if not match:
            raise RuntimeError("Could not find __main__ block in benchmark_quality.py")
        return textwrap.dedent(match.group(1))

    def _exec_main_block(self, monkeypatch, argv):
        """Exec the actual __main__ block with mocked run_benchmark."""
        import sys
        from unittest.mock import MagicMock
        from eval import benchmark_quality as bq

        monkeypatch.setattr(sys, "argv", argv)
        mock_run = MagicMock()
        saved = bq.run_benchmark
        bq.run_benchmark = mock_run
        try:
            main_source = self._get_main_block_source()
            # Exec the actual __main__ block, using benchmark_quality's globals
            # so that run_benchmark (which we just replaced) is visible
            code = compile(main_source, "<benchmark_quality __main__>", "exec")
            exec(code, bq.__dict__)
        finally:
            bq.run_benchmark = saved
        return mock_run

    def test_main_block_default(self, monkeypatch):
        """The __main__ block parses argv and calls run_benchmark."""
        mock_run = self._exec_main_block(monkeypatch, ["benchmark_quality"])
        mock_run.assert_called_once_with(None, verbose=True)

    def test_main_block_with_models_and_quiet(self, monkeypatch):
        """The __main__ block filters --quiet from models list."""
        mock_run = self._exec_main_block(
            monkeypatch, ["benchmark_quality", "model-a", "model-b", "--quiet"]
        )
        # --quiet filtered from models
        mock_run.assert_called_once_with(["model-a", "model-b"], verbose=False)

    def test_main_block_only_quiet(self, monkeypatch):
        """When only --quiet is passed, models should be None."""
        mock_run = self._exec_main_block(monkeypatch, ["benchmark_quality", "--quiet"])
        # No models left after filtering
        mock_run.assert_called_once_with(None, verbose=False)

    def test_main_block_argv_parsing(self, monkeypatch):
        """Test the argv parsing logic of the __main__ block."""
        mock_run = self._exec_main_block(monkeypatch, ["benchmark_quality", "m1"])
        mock_run.assert_called_once_with(["m1"], verbose=True)
