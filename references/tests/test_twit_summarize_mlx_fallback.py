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
            patch.object(twit_summarize, "get_mlx_context_length", return_value=8192),
            patch.object(twit_summarize, "get_model_prompt", return_value="Summarize: {}"),
        ):
            result = twit_summarize.summarize_with_llm(
                _make_tweets(), "http://localhost:1337", "m1"
            )
        assert "LLM error" in result


# --------------------------------------------------------------------------
# Provenance across the fallback chain (class C9). Each tier must be
# distinguishable in the artifact -- a summary that needed a server restart is
# not the same as one the primary answered first time.
# --------------------------------------------------------------------------


def test_a_summary_produced_after_a_server_restart_is_marked_degraded():
    """The restart tier was the one path with no provenance test: a run that only
    succeeded because Osaurus was bounced looked identical to a clean one."""
    from unittest.mock import patch

    from twitter import summarize as ts

    tweets = [{"screen_name": "a", "created_at": __import__("datetime").datetime(2026, 8, 1, 9, 0),
               "text": "hello"}]

    with (
        patch.object(ts, "get_available_models", return_value=["m"]),
        patch.object(ts, "select_best_model", return_value="m"),
        patch.object(ts, "_summarize_with_model", return_value=None),
        patch.object(ts, "_restart_and_retry_server", return_value="## T\n- a\n- b\n- c"),
        patch.object(ts, "_direct_mlx_fallback", return_value=None),
    ):
        result = ts.summarize_with_llm(tweets, "http://localhost:1337", "m")

    prov = result.provenance
    assert prov.degraded, "a restart-recovered run must not read as primary"
    assert "restart" in prov.backend or any("restart" in r for r in prov.reasons)


def test_a_summary_from_the_local_mlx_tier_is_marked_last_resort():
    from unittest.mock import patch

    from twitter import summarize as ts
    from twitter.provenance import LAST_RESORT

    tweets = [{"screen_name": "a", "created_at": __import__("datetime").datetime(2026, 8, 1, 9, 0),
               "text": "hello"}]

    with (
        patch.object(ts, "get_available_models", return_value=["m"]),
        patch.object(ts, "select_best_model", return_value="m"),
        patch.object(ts, "_summarize_with_model", return_value=None),
        patch.object(ts, "_restart_and_retry_server", return_value=None),
        patch.object(ts, "_direct_mlx_fallback", return_value="## T\n- a\n- b\n- c"),
    ):
        result = ts.summarize_with_llm(tweets, "http://localhost:1337", "m")

    assert result.provenance.tier == LAST_RESORT
    assert result.provenance.degraded


def test_total_failure_is_marked_failed_not_merely_empty():
    from unittest.mock import patch

    from twitter import summarize as ts
    from twitter.provenance import FAILED

    tweets = [{"screen_name": "a", "created_at": __import__("datetime").datetime(2026, 8, 1, 9, 0),
               "text": "hello"}]

    with (
        patch.object(ts, "get_available_models", return_value=["m"]),
        patch.object(ts, "select_best_model", return_value="m"),
        patch.object(ts, "_summarize_with_model", return_value=None),
        patch.object(ts, "_restart_and_retry_server", return_value=None),
        patch.object(ts, "_direct_mlx_fallback", return_value=None),
    ):
        result = ts.summarize_with_llm(tweets, "http://localhost:1337", "m")

    assert result.provenance.tier == FAILED
    assert "LLM error" in result


def test_a_reply_is_attributed_in_the_prompt():
    """The reply-to marker is how the model can tell a conversation from a
    monologue; it was the one prompt-building branch with no test."""
    import datetime
    from unittest.mock import patch

    from twitter import summarize as ts

    tweets = [{
        "screen_name": "a",
        "created_at": datetime.datetime(2026, 8, 1, 9, 0),
        "text": "replying now",
        "in_reply_to_screen_name": "b",
    }]
    with patch.object(ts, "get_model_prompt", return_value=""):
        prompt, _ = ts._build_prompt(tweets, max_chars=4000)
    assert "@b" in prompt
