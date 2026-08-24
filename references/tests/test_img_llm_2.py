"""Tests for img_llm LLM integration functions."""

from pathlib import Path
from unittest.mock import patch


class TestQueryMlxForFilename:
    def test_first_model_succeeds(self):
        from rename.llm import query_mlx_for_filename

        with (
            patch("rename.llm.find_mlx_model", return_value=Path("/tmp/test.mlx")),
            patch("rename.llm.find_any_working_mlx_model", return_value=None),
            patch("rename.llm.call_mlx", return_value="my_cool_file"),
            patch("rename.llm.default_filename_prompt", return_value="Text: {text}"),
            patch("rename.llm.filename_models", return_value=["test-model"]),
            patch("rename.llm.MLX_MODELS_DIR", Path("/tmp")),
        ):
            result = query_mlx_for_filename("some text")
        assert result == "my_cool_file"

    def test_model_not_found_skips(self):
        from rename.llm import query_mlx_for_filename

        with (
            patch("rename.llm.find_mlx_model", return_value=None),
            patch("rename.llm.find_any_working_mlx_model", return_value=None),
            patch("rename.llm.default_filename_prompt", return_value="Text: {text}"),
            patch("rename.llm.filename_models", return_value=["test-model"]),
            patch("rename.llm.MLX_MODELS_DIR", Path("/tmp")),
        ):
            result = query_mlx_for_filename("some text")
        assert result is None

    def test_call_mlx_returns_none(self):
        from rename.llm import query_mlx_for_filename

        with (
            patch("rename.llm.find_mlx_model", return_value=Path("/tmp/test.mlx")),
            patch("rename.llm.find_any_working_mlx_model", return_value=None),
            patch("rename.llm.call_mlx", return_value=None),
            patch("rename.llm.default_filename_prompt", return_value="Text: {text}"),
            patch("rename.llm.filename_models", return_value=["test-model"]),
            patch("rename.llm.MLX_MODELS_DIR", Path("/tmp")),
        ):
            result = query_mlx_for_filename("some text")
        assert result is None

    def test_short_content_skipped(self):
        from rename.llm import query_mlx_for_filename

        with (
            patch("rename.llm.find_mlx_model", return_value=Path("/tmp/test.mlx")),
            patch("rename.llm.find_any_working_mlx_model", return_value=None),
            patch("rename.llm.call_mlx", return_value="x"),
            patch("rename.llm.default_filename_prompt", return_value="Text: {text}"),
            patch("rename.llm.filename_models", return_value=["test-model"]),
            patch("rename.llm.MLX_MODELS_DIR", Path("/tmp")),
        ):
            result = query_mlx_for_filename("some text")
        assert result is None

    def test_fallback_succeeds(self):
        from rename.llm import query_mlx_for_filename

        with (
            patch("rename.llm.find_mlx_model", return_value=None),
            patch("rename.llm.find_any_working_mlx_model", return_value=Path("/tmp/fallback.mlx")),
            patch("rename.llm.call_mlx", return_value="fallback_name"),
            patch("rename.llm.default_filename_prompt", return_value="Text: {text}"),
            patch("rename.llm.filename_models", return_value=["test-model"]),
            patch("rename.llm.MLX_MODELS_DIR", Path("/tmp")),
        ):
            result = query_mlx_for_filename("some text")
        assert result == "fallback_name"

    def test_a_model_path_is_never_tried_twice(self):
        """The dedup guard, actually exercised.

        The previous version of this test let the first model succeed, so the
        `model_path in tried` guard and the fallback branch it protects were
        never reached — deleting the guard could not fail it. Here two configured
        models and the fallback all resolve to the SAME path, so a missing guard
        shows up as extra call_mlx invocations.
        """
        from rename.llm import query_mlx_for_filename

        mlx_path = Path("/tmp/my.mlx")
        with (
            patch("rename.llm.find_mlx_model", return_value=mlx_path),
            patch("rename.llm.find_any_working_mlx_model", return_value=mlx_path),
            patch("rename.llm.call_mlx", return_value=None) as call_mlx,
            patch("rename.llm.filename_models", return_value=["model-a", "model-b"]),
            patch("rename.llm.MLX_MODELS_DIR", Path("/tmp")),
        ):
            result = query_mlx_for_filename("some text")

        assert result is None
        assert call_mlx.call_count == 1, "the same model path must not be tried twice"

    def test_fallback_call_mlx_returns_none(self):
        from rename.llm import query_mlx_for_filename

        with (
            patch("rename.llm.find_mlx_model", return_value=None),
            patch("rename.llm.find_any_working_mlx_model", return_value=Path("/tmp/fallback.mlx")),
            patch("rename.llm.call_mlx", return_value=None),
            patch("rename.llm.default_filename_prompt", return_value="Text: {text}"),
            patch("rename.llm.filename_models", return_value=["test-model"]),
            patch("rename.llm.MLX_MODELS_DIR", Path("/tmp")),
        ):
            result = query_mlx_for_filename("some text")
        assert result is None


class TestRealFilenameTemplate:
    """The production templates, unpatched.

    Every other test in this file replaces default_filename_prompt() with a
    `{text}` stand-in, so none of them could see that `.format()` raised
    IndexError on foundation's positional `{}` template and silently disabled
    the whole LLM naming path.
    """

    def test_every_configured_model_renders_its_real_template(self):
        import rename.llm as rl

        assert rl.filename_models(), "config must supply a filename fallback chain"
        for model in rl.filename_models():
            prompt = rl._filename_prompt(model, "quarterly revenue report")
            assert "quarterly revenue report" in prompt
            assert "{}" not in prompt and "{text}" not in prompt

    def test_query_llm_sends_the_rendered_real_template(self):
        import rename.llm as rl

        sent = {}

        def _capture(model, messages, host, timeout, api_key=""):
            sent["messages"] = messages
            return {"content": "revenue_report", "error": ""}

        with patch("rename.llm._shared_call", _capture):
            result = rl.query_llm_for_filename("quarterly revenue report")

        assert result == "revenue_report"
        assert "quarterly revenue report" in sent["messages"][0]["content"]
