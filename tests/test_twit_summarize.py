"""Tests for twit_summarize.py."""
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestCheckSummaryQuality:
    def test_empty_summary(self, mock_llm):
        import twitter.summarize as twit_summarize
        warnings, critical = twit_summarize._check_summary_quality("")
        assert "empty" in warnings[0]
        assert critical is True

    def test_good_summary(self, mock_llm):
        import twitter.summarize as twit_summarize
        summary = """## Topic 1
- fact 1 about this topic
- fact 2 about this topic
- fact 3 about this topic
- fact 4 about this topic
- fact 5 about this topic
- fact 6 about this topic

## Topic 2
- fact 7 about this topic
- fact 8 about this topic
- fact 9 about this topic
- fact 10 about this topic
- fact 11 about this topic
- fact 12 about this topic
"""
        warnings, critical = twit_summarize._check_summary_quality(summary)
        assert critical is False
        assert warnings == []

    def test_no_headers(self, mock_llm):
        import twitter.summarize as twit_summarize
        summary = "Just some text without structure"
        warnings, critical = twit_summarize._check_summary_quality(summary)
        assert any("headers" in w for w in warnings)

    def test_few_bullets(self, mock_llm):
        import twitter.summarize as twit_summarize
        summary = """## Topic
- only one bullet
"""
        warnings, critical = twit_summarize._check_summary_quality(summary)
        assert any("bullet" in w for w in warnings)

    def test_very_short(self, mock_llm):
        import twitter.summarize as twit_summarize
        summary = "## Short\n- one"
        warnings, critical = twit_summarize._check_summary_quality(summary)
        assert any("short" in w for w in warnings)

    def test_critical_both_missing(self, mock_llm):
        import twitter.summarize as twit_summarize
        summary = "Just text with no structure and short"
        warnings, critical = twit_summarize._check_summary_quality(summary)
        assert critical is True


class TestBuildPrompt:
    def test_basic(self, mock_llm):
        import twitter.summarize as twit_summarize
        tweets = [
            {"screen_name": "user1", "text": "Hello world", "created_at": datetime(2024, 1, 1, 12, 0)},
            {"screen_name": "user2", "text": "Another tweet", "created_at": datetime(2024, 1, 1, 13, 0)},
        ]
        with patch.object(twit_summarize, "get_model_prompt", return_value="Summarize: {}"):
            prompt, n = twit_summarize._build_prompt(tweets, max_chars=10000, model="m1")
        assert n == 2
        assert "user1" in prompt
        assert "Hello world" in prompt

    def test_prompt_without_placeholder(self, mock_llm):
        import twitter.summarize as twit_summarize
        tweets = [{"screen_name": "u", "text": "t", "created_at": datetime(2024, 1, 1)}]
        with patch.object(twit_summarize, "get_model_prompt", return_value="Just summarize"):
            prompt, n = twit_summarize._build_prompt(tweets, max_chars=10000)
        assert prompt == "Just summarize"

    def test_no_prompt_falls_back(self, mock_llm):
        import twitter.summarize as twit_summarize
        tweets = [{"screen_name": "u", "text": "t", "created_at": datetime(2024, 1, 1)}]
        with patch.object(twit_summarize, "get_model_prompt", return_value=None):
            prompt, n = twit_summarize._build_prompt(tweets, max_chars=10000)
        assert "Summarize this timeline" in prompt
        assert "{}" not in prompt  # placeholder is filled

    def test_respects_budget(self, mock_llm):
        import twitter.summarize as twit_summarize
        tweets = [
            {"screen_name": "u", "text": "x" * 200, "created_at": datetime(2024, 1, 1)},
            {"screen_name": "u", "text": "x" * 200, "created_at": datetime(2024, 1, 1)},
        ]
        with patch.object(twit_summarize, "get_model_prompt", return_value="{}"):
            prompt, n = twit_summarize._build_prompt(tweets, max_chars=300)
        # 200 chars each + overhead > 300 budget; none fit
        assert n == 0
        assert prompt == ""


class TestSummarizeWithLlm:
    def _make_tweets(self):
        return [
            {"screen_name": "u1", "text": "hello", "created_at": datetime(2024, 1, 1, 12, 0)},
        ]

    def test_success_first_model(self, mock_llm):
        import twitter.summarize as twit_summarize
        with patch.object(twit_summarize, "get_available_models", return_value=["m1"]), \
             patch.object(twit_summarize, "call_llm_api", return_value={"content": "## Topic\n- fact 1\n- fact 2\n- fact 3"}), \
             patch.object(twit_summarize, "extract_thinking", return_value=(None, "## Topic\n- fact 1\n- fact 2\n- fact 3")), \
             patch.object(twit_summarize, "strip_thinking", side_effect=lambda x: x):
            result = twit_summarize.summarize_with_llm(self._make_tweets(), "http://localhost:1337", "m1")
        assert "Topic" in result

    def test_success_with_thinking(self, mock_llm):
        import twitter.summarize as twit_summarize
        with patch.object(twit_summarize, "get_available_models", return_value=["m1"]), \
             patch.object(twit_summarize, "call_llm_api", return_value={"content": "<thinking>x</thinking>## Topic\n- a\n- b\n- c"}), \
             patch.object(twit_summarize, "extract_thinking", return_value=("<thinking>x</thinking>", "## Topic\n- a\n- b\n- c")), \
             patch.object(twit_summarize, "merge_thinking_with_summary", return_value="merged"):
            result = twit_summarize.summarize_with_llm(self._make_tweets(), "http://localhost:1337", "m1")
        assert result == "merged"

    def test_thinking_critical_skips(self, mock_llm):
        """When thinking is present but summary is critical, skip to next model."""
        import twitter.summarize as twit_summarize
        with patch.object(twit_summarize, "get_available_models", return_value=["m1", "qwen3.6-35b-a3b-mxfp4"]), \
             patch.object(twit_summarize, "call_llm_api", side_effect=[
                 {"content": "<thinking>x</thinking>bad"},
                 {"content": "## Good\n- a\n- b\n- c"},
             ]), \
             patch.object(twit_summarize, "extract_thinking", side_effect=[
                 ("<thinking>x</thinking>", "bad"),
                 (None, "## Good\n- a\n- b\n- c"),
             ]), \
             patch.object(twit_summarize, "strip_thinking", side_effect=lambda x: x):
            result = twit_summarize.summarize_with_llm(self._make_tweets(), "http://localhost:1337", "m1")
        assert "Good" in result

    def test_no_content_error(self, mock_llm):
        """Result has 'error' key — falls through to next model."""
        import twitter.summarize as twit_summarize
        with patch.object(twit_summarize, "get_available_models", return_value=["m1", "qwen3.6-35b-a3b-mxfp4"]), \
             patch.object(twit_summarize, "call_llm_api", side_effect=[
                 {"error": "model not found"},
                 {"content": "## Good\n- a\n- b\n- c"},
             ]), \
             patch.object(twit_summarize, "extract_thinking", return_value=(None, "## Good\n- a\n- b\n- c")), \
             patch.object(twit_summarize, "strip_thinking", side_effect=lambda x: x):
            result = twit_summarize.summarize_with_llm(self._make_tweets(), "http://localhost:1337", "m1")
        assert "Good" in result

    def test_exception_in_call(self, mock_llm):
        """Exception during call_llm_api is caught, falls through to next model."""
        import twitter.summarize as twit_summarize
        with patch.object(twit_summarize, "get_available_models", return_value=["m1", "qwen3.6-35b-a3b-mxfp4"]), \
             patch.object(twit_summarize, "call_llm_api", side_effect=[
                 Exception("API error"),
                 {"content": "## Good\n- a\n- b\n- c"},
             ]), \
             patch.object(twit_summarize, "extract_thinking", return_value=(None, "## Good\n- a\n- b\n- c")), \
             patch.object(twit_summarize, "strip_thinking", side_effect=lambda x: x):
            result = twit_summarize.summarize_with_llm(self._make_tweets(), "http://localhost:1337", "m1")
        assert "Good" in result

    def test_target_model_not_in_models(self, mock_llm):
        """When target model not in available models, select_best_model picks one."""
        import twitter.summarize as twit_summarize
        with patch.object(twit_summarize, "get_available_models", return_value=["m2"]), \
             patch.object(twit_summarize, "get_best_model", return_value="m1"), \
             patch.object(twit_summarize, "select_best_model", return_value="m2"), \
             patch.object(twit_summarize, "call_llm_api", return_value={"content": "## Topic\n- a\n- b\n- c"}), \
             patch.object(twit_summarize, "extract_thinking", return_value=(None, "## Topic\n- a\n- b\n- c")), \
             patch.object(twit_summarize, "strip_thinking", side_effect=lambda x: x):
            result = twit_summarize.summarize_with_llm(self._make_tweets(), "http://localhost:1337", "m1")
        assert "Topic" in result

    def test_all_models_fail(self, mock_llm):
        """When all server models fail, return error string."""
        import twitter.summarize as twit_summarize
        with patch.object(twit_summarize, "get_available_models", return_value=["m1"]), \
             patch.object(twit_summarize, "call_llm_api", return_value={"error": "fail"}):
            result = twit_summarize.summarize_with_llm(self._make_tweets(), "http://localhost:1337", "m1")
        assert "LLM error" in result

    def test_target_model_with_known_critical_skips(self, mock_llm):
        """When target model returns critical summary, try next model."""
        import twitter.summarize as twit_summarize
        with patch.object(twit_summarize, "get_available_models", return_value=["m1", "qwen3.6-35b-a3b-mxfp4"]), \
             patch.object(twit_summarize, "call_llm_api", side_effect=[
                 {"content": "no structure here at all"},
                 {"content": "## Good\n- a\n- b\n- c"},
             ]), \
             patch.object(twit_summarize, "extract_thinking", side_effect=[
                 (None, "no structure here at all"),
                 (None, "## Good\n- a\n- b\n- c"),
             ]), \
             patch.object(twit_summarize, "strip_thinking", side_effect=lambda x: x):
            result = twit_summarize.summarize_with_llm(self._make_tweets(), "http://localhost:1337", "m1")
        assert "Good" in result

    def test_server_not_responding_recovers(self, mock_llm):
        """When models empty, ensure_server is called, then success."""
        import twitter.summarize as twit_summarize
        with patch.object(twit_summarize, "get_available_models", side_effect=[[], ["m1"]]), \
             patch.object(twit_summarize, "ensure_server"), \
             patch.object(twit_summarize, "call_llm_api", return_value={"content": "## Topic\n- a\n- b\n- c"}), \
             patch.object(twit_summarize, "extract_thinking", return_value=(None, "## Topic\n- a\n- b\n- c")), \
             patch.object(twit_summarize, "strip_thinking", side_effect=lambda x: x):
            result = twit_summarize.summarize_with_llm(self._make_tweets(), "http://localhost:1337", "m1")
        assert "Topic" in result

    def test_mlx_fallback_success_with_warnings(self, mock_llm):
        """All server models fail, MLX returns summary with quality warnings (non-critical)."""
        import twitter.summarize as twit_summarize
        with patch.object(twit_summarize, "get_available_models", return_value=["m1", "m2"]), \
             patch.object(twit_summarize, "call_llm_api", return_value={"error": "fail"}), \
             patch.object(twit_summarize, "find_mlx_model", return_value=Path("/tmp/mlx-test")), \
             patch.object(twit_summarize, "find_best_mlx_model", return_value=None), \
             patch.object(twit_summarize, "find_any_working_mlx_model", return_value=None), \
             patch.object(twit_summarize, "call_mlx", return_value="## Short\n- fact a\n- fact b"), \
             patch.object(twit_summarize, "process_mlx_content", side_effect=lambda x: x), \
             patch.object(twit_summarize, "get_mlx_context_length", return_value=8192), \
             patch.object(twit_summarize, "get_model_prompt", return_value="Summarize: {}"):
            result = twit_summarize.summarize_with_llm(self._make_tweets(), "http://localhost:1337", "m1")
        assert "Short" in result

    def test_mlx_fallback_error_start(self, mock_llm):
        """MLX returns string starting with [LLM error, tries next, falls through."""
        import twitter.summarize as twit_summarize
        with patch.object(twit_summarize, "get_available_models", return_value=["m1", "m2"]), \
             patch.object(twit_summarize, "call_llm_api", return_value={"error": "fail"}), \
             patch.object(twit_summarize, "find_mlx_model", return_value=Path("/tmp/mlx-test")), \
             patch.object(twit_summarize, "find_best_mlx_model", return_value=None), \
             patch.object(twit_summarize, "find_any_working_mlx_model", return_value=None), \
             patch.object(twit_summarize, "call_mlx", return_value="[LLM error: oops]"), \
             patch.object(twit_summarize, "process_mlx_content", side_effect=lambda x: x), \
             patch.object(twit_summarize, "get_mlx_context_length", return_value=8192), \
             patch.object(twit_summarize, "get_model_prompt", return_value="Summarize: {}"):
            result = twit_summarize.summarize_with_llm(self._make_tweets(), "http://localhost:1337", "m1")
        assert "LLM error" in result

    def test_mlx_fallback_critical(self, mock_llm):
        """MLX summary fails quality check, falls through to error."""
        import twitter.summarize as twit_summarize
        with patch.object(twit_summarize, "get_available_models", return_value=["m1", "m2"]), \
             patch.object(twit_summarize, "call_llm_api", return_value={"error": "fail"}), \
             patch.object(twit_summarize, "find_mlx_model", return_value=Path("/tmp/mlx-test")), \
             patch.object(twit_summarize, "find_best_mlx_model", return_value=None), \
             patch.object(twit_summarize, "find_any_working_mlx_model", return_value=None), \
             patch.object(twit_summarize, "call_mlx", return_value="plain text no structure"), \
             patch.object(twit_summarize, "process_mlx_content", side_effect=lambda x: x), \
             patch.object(twit_summarize, "get_mlx_context_length", return_value=8192), \
             patch.object(twit_summarize, "get_model_prompt", return_value="Summarize: {}"):
            result = twit_summarize.summarize_with_llm(self._make_tweets(), "http://localhost:1337", "m1")
        assert "LLM error" in result
