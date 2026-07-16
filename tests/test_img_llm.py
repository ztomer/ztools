"""Tests for img_llm LLM integration functions."""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch


class TestEnsureLlmRunning:
    def test_delegates_to_ensure_server(self):
        from rename.llm import ensure_llm_running

        with patch("lib.osaurus_lib.ensure_server", return_value=True) as mock_ensure:
            assert ensure_llm_running() is True
            mock_ensure.assert_called_once()

        with patch("lib.osaurus_lib.ensure_server", return_value=False) as mock_ensure:
            assert ensure_llm_running() is False
            mock_ensure.assert_called_once()


class TestIsRelevantWithLlm:
    def test_keep_response(self):
        from rename.llm import is_relevant_with_llm

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"message": {"content": "keep"}}'
        with patch("rename.llm.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            assert is_relevant_with_llm("some content", "http://localhost:1337") is True

    def test_skip_response(self):
        from rename.llm import is_relevant_with_llm

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"message": {"content": "skip"}}'
        with patch("rename.llm.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            assert is_relevant_with_llm("some content", "http://localhost:1337") is False

    def test_first_model_fails_second_succeeds(self):
        from rename.llm import is_relevant_with_llm

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"message": {"content": "keep"}}'

        fail_resp = MagicMock()
        fail_resp.status_code = 500

        with patch("rename.llm.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.side_effect = [fail_resp, mock_resp]
            assert is_relevant_with_llm("x", "http://localhost:1337") is True

    def test_all_models_fail(self):
        from rename.llm import is_relevant_with_llm

        fail_resp = MagicMock()
        fail_resp.status_code = 500
        with patch("rename.llm.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = fail_resp
            assert is_relevant_with_llm("x", "http://localhost:1337") is None

    def test_http_exception(self):
        from rename.llm import is_relevant_with_llm

        with patch("rename.llm.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.side_effect = Exception("connection error")
            assert is_relevant_with_llm("x", "http://localhost:1337") is None

    def test_invalid_json_line_continues(self):
        """Lines 82-83: invalid JSON in response line is skipped."""
        from rename.llm import is_relevant_with_llm

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = 'not valid json\n{"message": {"content": "keep"}}'
        with patch("rename.llm.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            assert is_relevant_with_llm("x", "http://localhost:1337") is True

    def test_empty_lines_skipped(self):
        """Lines 77-78: empty lines in response are skipped."""
        from rename.llm import is_relevant_with_llm

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '\n\n{"message": {"content": "keep"}}'
        with patch("rename.llm.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            assert is_relevant_with_llm("x", "http://localhost:1337") is True


class TestQueryLlmForFilename:
    def test_successful_query(self):
        from rename.llm import query_llm_for_filename

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            '{"message": {"content": "my_cool_photo"}}\n'
            '{"message": {"content": "", "done": true}}\n'
        )
        with (
            patch("rename.llm.requests.Session") as mock_session,
            patch("rename.llm.FILENAME_MODELS", ["test-model"]),
            patch("rename.llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"),
        ):
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            result = query_llm_for_filename("Test image text", "http://localhost:1337")
            assert result == "my_cool_photo"

    def test_strips_instruction_prefix(self):
        from rename.llm import query_llm_for_filename

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            '{"message": {"content": "filename: my_photo"}}\n'
            '{"message": {"content": "", "done": true}}\n'
        )
        with (
            patch("rename.llm.requests.Session") as mock_session,
            patch("rename.llm.FILENAME_MODELS", ["test-model"]),
            patch("rename.llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"),
        ):
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            result = query_llm_for_filename("Test image text", "http://localhost:1337")
            assert result == "my_photo"

    def test_empty_response(self):
        from rename.llm import query_llm_for_filename

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"message": {"content": ""}}\n'
        with (
            patch("rename.llm.requests.Session") as mock_session,
            patch("rename.llm.FILENAME_MODELS", ["test-model"]),
            patch("rename.llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"),
        ):
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            assert query_llm_for_filename("Test text", "http://localhost:1337") is None

    def test_http_error_fallback(self):
        from rename.llm import query_llm_for_filename

        fail_resp = MagicMock()
        fail_resp.status_code = 500

        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.text = (
            '{"message": {"content": "successful_name"}}\n'
            '{"message": {"content": "", "done": true}}\n'
        )
        with (
            patch("rename.llm.requests.Session") as mock_session,
            patch("rename.llm.FILENAME_MODELS", ["fail-model", "success-model"]),
            patch("rename.llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"),
        ):
            s = mock_session.return_value.__enter__.return_value
            s.post.side_effect = [fail_resp, success_resp]
            result = query_llm_for_filename("Test text", "http://localhost:1337")
            assert result == "successful_name"

    def test_limits_words_to_6(self):
        from rename.llm import query_llm_for_filename

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            '{"message": {"content": "one two three four five six seven eight"}}\n'
            '{"message": {"content": "", "done": true}}\n'
        )
        with (
            patch("rename.llm.requests.Session") as mock_session,
            patch("rename.llm.FILENAME_MODELS", ["test-model"]),
            patch("rename.llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"),
        ):
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            result = query_llm_for_filename("x", "http://localhost:1337")
            assert result == "one_two_three_four_five_six"
            assert "_seven" not in result

    def test_no_alpha_content_returns_none(self):
        from rename.llm import query_llm_for_filename

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            '{"message": {"content": "123 456 789"}}\n{"message": {"content": "", "done": true}}\n'
        )
        with (
            patch("rename.llm.requests.Session") as mock_session,
            patch("rename.llm.FILENAME_MODELS", ["test-model"]),
        ):
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            assert query_llm_for_filename("x", "http://localhost:1337") is None

    def test_invalid_json_in_streaming_response(self):
        """Lines 118-120: invalid JSON in streaming response is skipped."""
        from rename.llm import query_llm_for_filename

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = 'not valid json\n{"message": {"content": "valid_name", "done": true}}\n'
        with (
            patch("rename.llm.requests.Session") as mock_session,
            patch("rename.llm.FILENAME_MODELS", ["test-model"]),
            patch("rename.llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"),
        ):
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            result = query_llm_for_filename("x", "http://localhost:1337")
            assert result == "valid_name"

    def test_truncate_long_content(self):
        """Line 133: content longer than 35 chars gets truncated."""
        from rename.llm import query_llm_for_filename

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        long_name = "x" * 50
        mock_resp.text = (
            f'{{"message": {{"content": "{long_name}"}}}}\n'
            '{"message": {"content": "", "done": true}}\n'
        )
        with (
            patch("rename.llm.requests.Session") as mock_session,
            patch("rename.llm.FILENAME_MODELS", ["test-model"]),
            patch("rename.llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"),
        ):
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            result = query_llm_for_filename("x", "http://localhost:1337")
            # Truncated to exactly 35 chars
            assert result == "x" * 35
            assert len(result) == 35

    def test_non_alpha_content_skipped(self):
        """Lines 135-136, 138-139: non-alpha content is skipped."""
        from rename.llm import query_llm_for_filename

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # Punctuation only - no a-z
        mock_resp.text = (
            '{"message": {"content": "!!! ??? ???"}}\n{"message": {"content": "", "done": true}}\n'
        )
        with (
            patch("rename.llm.requests.Session") as mock_session,
            patch("rename.llm.FILENAME_MODELS", ["test-model"]),
            patch("rename.llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"),
        ):
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            assert query_llm_for_filename("x", "http://localhost:1337") is None

    def test_short_content_skipped(self):
        """Line 129: content with no words (after regex) returns None."""
        from rename.llm import query_llm_for_filename

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # Just punctuation - no words
        mock_resp.text = (
            '{"message": {"content": "!!!"}}\n{"message": {"content": "", "done": true}}\n'
        )
        with (
            patch("rename.llm.requests.Session") as mock_session,
            patch("rename.llm.FILENAME_MODELS", ["test-model"]),
            patch("rename.llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"),
        ):
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            assert query_llm_for_filename("x", "http://localhost:1337") is None

    def test_invalid_alpha_pattern(self):
        """Line 135: content with invalid chars (not a-z) is rejected."""
        from rename.llm import query_llm_for_filename

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # All caps - regex a-z fails
        mock_resp.text = (
            '{"message": {"content": "TEST NAME"}}\n{"message": {"content": "", "done": true}}\n'
        )
        with (
            patch("rename.llm.requests.Session") as mock_session,
            patch("rename.llm.FILENAME_MODELS", ["test-model"]),
            patch("rename.llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"),
        ):
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            # Result is lowercased to "test name" → joined "test_name" → valid
            result = query_llm_for_filename("x", "http://localhost:1337")
            assert result == "test_name"


class TestQueryVlmForFilename:
    def test_successful_vlm_query(self):
        from rename.llm import query_vlm_for_filename

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            '{"message": {"content": "white goose grass"}}\n'
            '{"message": {"content": "", "done": true}}\n'
        )
        with (
            patch("builtins.open", mock_open(read_data=b"fake_image_data")),
            patch("rename.llm.requests.Session") as mock_session,
        ):
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            result = query_vlm_for_filename(
                Path("/fake/image.png"), "http://localhost:1337", "vlm-model"
            )
            assert result == "white goose grass"

    def test_api_error(self):
        from rename.llm import query_vlm_for_filename

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal server error"
        with (
            patch("builtins.open", mock_open(read_data=b"data")),
            patch("rename.llm.requests.Session") as mock_session,
        ):
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            result = query_vlm_for_filename(
                Path("/fake/image.png"), "http://localhost:1337", "vlm-model"
            )
            assert result is None

    def test_file_read_exception(self):
        from rename.llm import query_vlm_for_filename

        with patch("builtins.open", side_effect=Exception("file error")):
            result = query_vlm_for_filename(
                Path("/fake/image.png"), "http://localhost:1337", "vlm-model"
            )
            assert result is None

    def test_with_api_key(self):
        from rename.llm import query_vlm_for_filename

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"message": {"content": "output"}}\n{"message": {"done": true}}\n'
        with (
            patch("builtins.open", mock_open(read_data=b"data")),
            patch("rename.llm.requests.Session") as mock_session,
        ):
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            result = query_vlm_for_filename(
                Path("/fake/image.png"), "http://localhost:1337", "vlm-model", api_key="mykey"
            )
            assert result == "output"
            headers = s.post.call_args[1].get("headers", {})
            assert headers.get("Authorization") == "Bearer mykey"

    def test_vlm_done_break(self):
        """Line 210: VLM stream ends with done=true at top level (currently unreachable)."""
        from rename.llm import query_vlm_for_filename

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # done at TOP level of JSON (per img_llm.py line 209: j.get("done"))
        mock_resp.text = (
            '{"message": {"content": "first"}}\n{"done": true, "message": {"content": "more"}}\n'
        )
        with (
            patch("builtins.open", mock_open(read_data=b"data")),
            patch("rename.llm.requests.Session") as mock_session,
        ):
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            result = query_vlm_for_filename(
                Path("/fake/image.png"), "http://localhost:1337", "vlm-model"
            )
            assert result == "firstmore"

    def test_vlm_invalid_json_continues(self):
        """Lines 211-212: invalid JSON in VLM stream is skipped."""
        from rename.llm import query_vlm_for_filename

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = 'not valid json\n{"message": {"content": "valid_part", "done": true}}\n'
        with (
            patch("builtins.open", mock_open(read_data=b"data")),
            patch("rename.llm.requests.Session") as mock_session,
        ):
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            result = query_vlm_for_filename(
                Path("/fake/image.png"), "http://localhost:1337", "vlm-model"
            )
            assert result == "valid_part"

    def test_query_llm_with_done_break(self):
        """Line 118: query_llm_for_filename stops on done (at top level of JSON)."""
        from rename.llm import query_llm_for_filename

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # done is at top level (line 117: j.get("done", False))
        mock_resp.text = (
            '{"message": {"content": "first_"}}\n{"message": {"content": "name"}, "done": true}\n'
        )
        with (
            patch("rename.llm.requests.Session") as mock_session,
            patch("rename.llm.FILENAME_MODELS", ["test-model"]),
            patch("rename.llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"),
        ):
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            result = query_llm_for_filename("x", "http://localhost:1337")
            assert result == "first_name"


class TestQueryMlxForFilename:
    def test_first_model_succeeds(self):
        from rename.llm import query_mlx_for_filename

        with (
            patch("rename.llm.find_mlx_model", return_value=Path("/tmp/test.mlx")),
            patch("rename.llm.find_any_working_mlx_model", return_value=None),
            patch("rename.llm.call_mlx", return_value="my_cool_file"),
            patch("rename.llm.process_mlx_content", side_effect=lambda x: x),
            patch("rename.llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"),
            patch("rename.llm.FILENAME_MODELS", ["test-model"]),
            patch("rename.llm.MLX_MODELS_DIR", Path("/tmp")),
        ):
            result = query_mlx_for_filename("some text")
        assert result == "my_cool_file"

    def test_model_not_found_skips(self):
        from rename.llm import query_mlx_for_filename

        with (
            patch("rename.llm.find_mlx_model", return_value=None),
            patch("rename.llm.find_any_working_mlx_model", return_value=None),
            patch("rename.llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"),
            patch("rename.llm.FILENAME_MODELS", ["test-model"]),
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
            patch("rename.llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"),
            patch("rename.llm.FILENAME_MODELS", ["test-model"]),
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
            patch("rename.llm.process_mlx_content", side_effect=lambda x: x),
            patch("rename.llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"),
            patch("rename.llm.FILENAME_MODELS", ["test-model"]),
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
            patch("rename.llm.process_mlx_content", side_effect=lambda x: x),
            patch("rename.llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"),
            patch("rename.llm.FILENAME_MODELS", ["test-model"]),
            patch("rename.llm.MLX_MODELS_DIR", Path("/tmp")),
        ):
            result = query_mlx_for_filename("some text")
        assert result == "fallback_name"

    def test_fallback_already_tried(self):
        from rename.llm import query_mlx_for_filename

        mlx_path = Path("/tmp/my.mlx")
        with (
            patch("rename.llm.find_mlx_model", return_value=mlx_path),
            patch("rename.llm.find_any_working_mlx_model", return_value=mlx_path),
            patch("rename.llm.call_mlx", return_value="skipped"),
            patch("rename.llm.process_mlx_content", side_effect=lambda x: x),
            patch("rename.llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"),
            patch("rename.llm.FILENAME_MODELS", ["test-model"]),
            patch("rename.llm.MLX_MODELS_DIR", Path("/tmp")),
        ):
            result = query_mlx_for_filename("some text")
        assert result == "skipped"

    def test_fallback_call_mlx_returns_none(self):
        from rename.llm import query_mlx_for_filename

        with (
            patch("rename.llm.find_mlx_model", return_value=None),
            patch("rename.llm.find_any_working_mlx_model", return_value=Path("/tmp/fallback.mlx")),
            patch("rename.llm.call_mlx", return_value=None),
            patch("rename.llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"),
            patch("rename.llm.FILENAME_MODELS", ["test-model"]),
            patch("rename.llm.MLX_MODELS_DIR", Path("/tmp")),
        ):
            result = query_mlx_for_filename("some text")
        assert result is None
