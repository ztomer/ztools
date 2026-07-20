"""Tests for the mlx-vlm fallback path in twitter.summarize (server+retry, then mlx-vlm)."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch


def _make_tweets():
    return [
        {
            "screen_name": "openai",
            "created_at": datetime(2026, 1, 1, 10, 0),
            "text": "We announced GPT-5 today.",
            "favorite_count": 1200,
            "retweet_count": 300,
        },
        {
            "screen_name": "apple",
            "created_at": datetime(2026, 1, 1, 11, 0),
            "text": "Vision Pro 2 enters production.",
            "favorite_count": 800,
            "retweet_count": 150,
        },
    ]


class TestMlxVlmFallback:
    def test_mlx_vlm_fallback_success(self, mock_llm):
        """After server retries fail, mlx-vlm-loadable model is used as last resort."""
        import twitter.summarize as twit_summarize

        with (
            patch.object(twit_summarize, "get_available_models", return_value=["m1", "m2"]),
            patch.object(twit_summarize, "call_llm_api", return_value={"error": "fail"}),
            patch.object(twit_summarize, "restart_server", return_value=True),
            patch.object(twit_summarize, "find_mlx_model", return_value=Path("/tmp/mlx-vlm-test")),
            patch.object(twit_summarize, "find_best_mlx_vlm_model", return_value=None),
            patch.object(twit_summarize, "find_any_working_mlx_vlm_model", return_value=None),
            patch.object(twit_summarize, "probe_mlx_vlm_loadable", return_value=(True, "ok")),
            patch.object(
                twit_summarize, "call_mlx_vlm", return_value="## VLM\n- a fact\n- b fact\n- c fact"
            ),
            patch.object(twit_summarize, "process_mlx_content", side_effect=lambda x: x),
            patch.object(twit_summarize, "get_mlx_context_length", return_value=8192),
            patch.object(twit_summarize, "get_model_prompt", return_value="Summarize: {}"),
        ):
            result = twit_summarize.summarize_with_llm(
                _make_tweets(), "http://localhost:1337", "m1"
            )
        assert "VLM" in result

    def test_mlx_vlm_runs_when_server_down(self, mock_llm):
        """When the server is down (no models at all), mlx-vlm runs directly."""
        import twitter.summarize as twit_summarize

        with (
            patch.object(twit_summarize, "get_available_models", return_value=[]),
            patch.object(twit_summarize, "ensure_server"),
            patch.object(twit_summarize, "restart_server", return_value=False),
            patch.object(twit_summarize, "select_best_model", return_value=None),
            patch.object(twit_summarize, "call_llm_api", return_value={"error": "down"}),
            patch.object(twit_summarize, "find_mlx_model", return_value=Path("/tmp/mlx-vlm-test")),
            patch.object(twit_summarize, "find_best_mlx_vlm_model", return_value=None),
            patch.object(twit_summarize, "find_any_working_mlx_vlm_model", return_value=None),
            patch.object(twit_summarize, "probe_mlx_vlm_loadable", return_value=(True, "ok")),
            patch.object(
                twit_summarize, "call_mlx_vlm", return_value="## VLM\n- a fact\n- b fact\n- c fact"
            ),
            patch.object(twit_summarize, "process_mlx_content", side_effect=lambda x: x),
            patch.object(twit_summarize, "get_mlx_context_length", return_value=8192),
            patch.object(twit_summarize, "get_model_prompt", return_value="Summarize: {}"),
        ):
            result = twit_summarize.summarize_with_llm(
                _make_tweets(), "http://localhost:1337", "m1"
            )
        assert "VLM" in result

    def test_mlx_fallback_success_with_warnings(self, mock_llm):
        """All server models fail, MLX returns summary with quality warnings (non-critical)."""
        import twitter.summarize as twit_summarize

        with (
            patch.object(twit_summarize, "get_available_models", return_value=["m1", "m2"]),
            patch.object(twit_summarize, "call_llm_api", return_value={"error": "fail"}),
            patch.object(twit_summarize, "restart_server", return_value=True),
            patch.object(twit_summarize, "select_best_model", return_value="m1"),
            patch.object(twit_summarize, "find_mlx_model", return_value=Path("/tmp/mlx-test")),
            patch.object(twit_summarize, "find_best_mlx_model", return_value=None),
            patch.object(twit_summarize, "find_any_working_mlx_model", return_value=None),
            # no vlm-loadable models discovered -> stock mlx_lm stage is used
            patch.object(twit_summarize, "find_best_mlx_vlm_model", return_value=None),
            patch.object(twit_summarize, "find_any_working_mlx_vlm_model", return_value=None),
            patch.object(twit_summarize, "probe_mlx_vlm_loadable", return_value=(False, "no")),
            patch.object(twit_summarize, "_check_mlx_model_compatible", return_value=True),
            patch.object(twit_summarize, "call_mlx", return_value="## Short\n- fact a\n- fact b"),
            patch.object(twit_summarize, "process_mlx_content", side_effect=lambda x: x),
            patch.object(twit_summarize, "get_mlx_context_length", return_value=8192),
            patch.object(twit_summarize, "get_model_prompt", return_value="Summarize: {}"),
        ):
            result = twit_summarize.summarize_with_llm(
                _make_tweets(), "http://localhost:1337", "m1"
            )
        assert "Short" in result

    def test_mlx_fallback_error_start(self, mock_llm):
        """MLX returns string starting with [LLM error, tries next, falls through."""
        import twitter.summarize as twit_summarize

        with (
            patch.object(twit_summarize, "get_available_models", return_value=["m1", "m2"]),
            patch.object(twit_summarize, "call_llm_api", return_value={"error": "fail"}),
            patch.object(twit_summarize, "restart_server", return_value=True),
            patch.object(twit_summarize, "select_best_model", return_value="m1"),
            patch.object(twit_summarize, "find_mlx_model", return_value=Path("/tmp/mlx-test")),
            patch.object(twit_summarize, "find_best_mlx_model", return_value=None),
            patch.object(twit_summarize, "find_any_working_mlx_model", return_value=None),
            patch.object(twit_summarize, "find_best_mlx_vlm_model", return_value=None),
            patch.object(twit_summarize, "find_any_working_mlx_vlm_model", return_value=None),
            patch.object(twit_summarize, "probe_mlx_vlm_loadable", return_value=(False, "no")),
            patch.object(twit_summarize, "call_mlx", return_value="[LLM error: oops]"),
            patch.object(twit_summarize, "process_mlx_content", side_effect=lambda x: x),
            patch.object(twit_summarize, "get_mlx_context_length", return_value=8192),
            patch.object(twit_summarize, "get_model_prompt", return_value="Summarize: {}"),
        ):
            result = twit_summarize.summarize_with_llm(
                _make_tweets(), "http://localhost:1337", "m1"
            )
        assert "LLM error" in result

    def test_mlx_fallback_critical(self, mock_llm):
        """MLX summary fails quality check, falls through to error."""
        import twitter.summarize as twit_summarize

        with (
            patch.object(twit_summarize, "get_available_models", return_value=["m1", "m2"]),
            patch.object(twit_summarize, "call_llm_api", return_value={"error": "fail"}),
            patch.object(twit_summarize, "restart_server", return_value=True),
            patch.object(twit_summarize, "select_best_model", return_value="m1"),
            patch.object(twit_summarize, "find_mlx_model", return_value=Path("/tmp/mlx-test")),
            patch.object(twit_summarize, "find_best_mlx_model", return_value=None),
            patch.object(twit_summarize, "find_any_working_mlx_model", return_value=None),
            patch.object(twit_summarize, "find_best_mlx_vlm_model", return_value=None),
            patch.object(twit_summarize, "find_any_working_mlx_vlm_model", return_value=None),
            patch.object(twit_summarize, "probe_mlx_vlm_loadable", return_value=(False, "no")),
            patch.object(twit_summarize, "call_mlx", return_value="plain text no structure"),
            patch.object(twit_summarize, "process_mlx_content", side_effect=lambda x: x),
            patch.object(twit_summarize, "get_mlx_context_length", return_value=8192),
            patch.object(twit_summarize, "get_model_prompt", return_value="Summarize: {}"),
        ):
            result = twit_summarize.summarize_with_llm(
                _make_tweets(), "http://localhost:1337", "m1"
            )
        assert "LLM error" in result
