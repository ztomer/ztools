"""Tests for img_llm LLM integration functions."""

from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path


class TestEnsureLlmRunning:
    def test_returns_true_when_running(self):
        from img_llm import ensure_llm_running
        with patch("img_llm.check_llm_availability", return_value=True):
            assert ensure_llm_running() is True

    def test_restarts_and_succeeds(self):
        from img_llm import ensure_llm_running
        with patch("img_llm.check_llm_availability", side_effect=[False, True]), \
             patch("img_llm.subprocess.run") as mock_run, \
             patch("img_llm.subprocess.Popen") as mock_popen, \
             patch("img_llm.time.sleep"):
            assert ensure_llm_running() is True
        mock_run.assert_called_once()
        mock_popen.assert_called_once()

    def test_restart_fails(self):
        from img_llm import ensure_llm_running
        with patch("img_llm.check_llm_availability", return_value=False), \
             patch("img_llm.subprocess.run"), \
             patch("img_llm.subprocess.Popen"), \
             patch("img_llm.time.sleep"):
            assert ensure_llm_running() is False

    def test_popen_exception(self):
        from img_llm import ensure_llm_running
        with patch("img_llm.check_llm_availability", side_effect=[False, False]), \
             patch("img_llm.subprocess.run"), \
             patch("img_llm.subprocess.Popen", side_effect=Exception("fail")), \
             patch("img_llm.time.sleep"):
            assert ensure_llm_running() is False

    def test_pkill_exception_swallowed(self):
        """Line 47-48: pkill exception is swallowed."""
        from img_llm import ensure_llm_running
        with patch("img_llm.check_llm_availability", side_effect=[False, True]), \
             patch("img_llm.subprocess.run", side_effect=Exception("pkill fail")), \
             patch("img_llm.subprocess.Popen"), \
             patch("img_llm.time.sleep"):
            assert ensure_llm_running() is True


class TestIsRelevantWithLlm:
    def test_keep_response(self):
        from img_llm import is_relevant_with_llm
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"message": {"content": "keep"}}'
        with patch("img_llm.requests.post", return_value=mock_resp):
            assert is_relevant_with_llm("some content", "http://localhost:1337") is True

    def test_skip_response(self):
        from img_llm import is_relevant_with_llm
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"message": {"content": "skip"}}'
        with patch("img_llm.requests.post", return_value=mock_resp):
            assert is_relevant_with_llm("some content", "http://localhost:1337") is False

    def test_first_model_fails_second_succeeds(self):
        from img_llm import is_relevant_with_llm
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"message": {"content": "keep"}}'

        fail_resp = MagicMock()
        fail_resp.status_code = 500

        with patch("img_llm.requests.post", side_effect=[fail_resp, mock_resp]):
            assert is_relevant_with_llm("x", "http://localhost:1337") is True

    def test_all_models_fail(self):
        from img_llm import is_relevant_with_llm
        fail_resp = MagicMock()
        fail_resp.status_code = 500
        with patch("img_llm.requests.post", return_value=fail_resp):
            assert is_relevant_with_llm("x", "http://localhost:1337") is None

    def test_http_exception(self):
        from img_llm import is_relevant_with_llm
        with patch("img_llm.requests.post", side_effect=Exception("connection error")):
            assert is_relevant_with_llm("x", "http://localhost:1337") is None

    def test_invalid_json_line_continues(self):
        """Lines 82-83: invalid JSON in response line is skipped."""
        from img_llm import is_relevant_with_llm
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            'not valid json\n'
            '{"message": {"content": "keep"}}'
        )
        with patch("img_llm.requests.post", return_value=mock_resp):
            assert is_relevant_with_llm("x", "http://localhost:1337") is True

    def test_empty_lines_skipped(self):
        """Lines 77-78: empty lines in response are skipped."""
        from img_llm import is_relevant_with_llm
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            '\n\n'
            '{"message": {"content": "keep"}}'
        )
        with patch("img_llm.requests.post", return_value=mock_resp):
            assert is_relevant_with_llm("x", "http://localhost:1337") is True


class TestQueryLlmForFilename:
    def test_successful_query(self):
        from img_llm import query_llm_for_filename
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            '{"message": {"content": "my_cool_photo"}}\n'
            '{"message": {"content": "", "done": true}}\n'
        )
        with patch("img_llm.requests.post", return_value=mock_resp), \
             patch("img_llm.FILENAME_MODELS", ["test-model"]), \
             patch("img_llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"):
            result = query_llm_for_filename("Test image text", "http://localhost:1337")
            assert result == "my_cool_photo"

    def test_strips_instruction_prefix(self):
        from img_llm import query_llm_for_filename
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            '{"message": {"content": "filename: my_photo"}}\n'
            '{"message": {"content": "", "done": true}}\n'
        )
        with patch("img_llm.requests.post", return_value=mock_resp), \
             patch("img_llm.FILENAME_MODELS", ["test-model"]), \
             patch("img_llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"):
            result = query_llm_for_filename("Test image text", "http://localhost:1337")
            assert result == "my_photo"

    def test_empty_response(self):
        from img_llm import query_llm_for_filename
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"message": {"content": ""}}\n'
        with patch("img_llm.requests.post", return_value=mock_resp), \
             patch("img_llm.FILENAME_MODELS", ["test-model"]), \
             patch("img_llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"):
            assert query_llm_for_filename("Test text", "http://localhost:1337") is None

    def test_http_error_fallback(self):
        from img_llm import query_llm_for_filename
        fail_resp = MagicMock()
        fail_resp.status_code = 500

        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.text = (
            '{"message": {"content": "successful_name"}}\n'
            '{"message": {"content": "", "done": true}}\n'
        )
        with patch("img_llm.requests.post", side_effect=[fail_resp, success_resp]), \
             patch("img_llm.FILENAME_MODELS", ["fail-model", "success-model"]), \
             patch("img_llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"):
            result = query_llm_for_filename("Test text", "http://localhost:1337")
            assert result == "successful_name"

    def test_limits_words_to_6(self):
        from img_llm import query_llm_for_filename
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            '{"message": {"content": "one two three four five six seven eight"}}\n'
            '{"message": {"content": "", "done": true}}\n'
        )
        with patch("img_llm.requests.post", return_value=mock_resp), \
             patch("img_llm.FILENAME_MODELS", ["test-model"]), \
             patch("img_llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"):
            result = query_llm_for_filename("x", "http://localhost:1337")
            assert result == "one_two_three_four_five_six"
            assert "_seven" not in result

    def test_no_alpha_content_returns_none(self):
        from img_llm import query_llm_for_filename
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            '{"message": {"content": "123 456 789"}}\n'
            '{"message": {"content": "", "done": true}}\n'
        )
        with patch("img_llm.requests.post", return_value=mock_resp), \
             patch("img_llm.FILENAME_MODELS", ["test-model"]):
            assert query_llm_for_filename("x", "http://localhost:1337") is None

    def test_invalid_json_in_streaming_response(self):
        """Lines 118-120: invalid JSON in streaming response is skipped."""
        from img_llm import query_llm_for_filename
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            'not valid json\n'
            '{"message": {"content": "valid_name", "done": true}}\n'
        )
        with patch("img_llm.requests.post", return_value=mock_resp), \
             patch("img_llm.FILENAME_MODELS", ["test-model"]), \
             patch("img_llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"):
            result = query_llm_for_filename("x", "http://localhost:1337")
            assert result == "valid_name"

    def test_truncate_long_content(self):
        """Line 133: content longer than 35 chars gets truncated."""
        from img_llm import query_llm_for_filename
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        long_name = "x" * 50
        mock_resp.text = (
            f'{{"message": {{"content": "{long_name}"}}}}\n'
            '{"message": {"content": "", "done": true}}\n'
        )
        with patch("img_llm.requests.post", return_value=mock_resp), \
             patch("img_llm.FILENAME_MODELS", ["test-model"]), \
             patch("img_llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"):
            result = query_llm_for_filename("x", "http://localhost:1337")
            # Truncated to exactly 35 chars
            assert result == "x" * 35
            assert len(result) == 35

    def test_non_alpha_content_skipped(self):
        """Lines 135-136, 138-139: non-alpha content is skipped."""
        from img_llm import query_llm_for_filename
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # Punctuation only - no a-z
        mock_resp.text = (
            '{"message": {"content": "!!! ??? ???"}}\n'
            '{"message": {"content": "", "done": true}}\n'
        )
        with patch("img_llm.requests.post", return_value=mock_resp), \
             patch("img_llm.FILENAME_MODELS", ["test-model"]), \
             patch("img_llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"):
            assert query_llm_for_filename("x", "http://localhost:1337") is None

    def test_short_content_skipped(self):
        """Line 129: content with no words (after regex) returns None."""
        from img_llm import query_llm_for_filename
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # Just punctuation - no words
        mock_resp.text = (
            '{"message": {"content": "!!!"}}\n'
            '{"message": {"content": "", "done": true}}\n'
        )
        with patch("img_llm.requests.post", return_value=mock_resp), \
             patch("img_llm.FILENAME_MODELS", ["test-model"]), \
             patch("img_llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"):
            assert query_llm_for_filename("x", "http://localhost:1337") is None

    def test_invalid_alpha_pattern(self):
        """Line 135: content with invalid chars (not a-z) is rejected."""
        from img_llm import query_llm_for_filename
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # All caps - regex a-z fails
        mock_resp.text = (
            '{"message": {"content": "TEST NAME"}}\n'
            '{"message": {"content": "", "done": true}}\n'
        )
        with patch("img_llm.requests.post", return_value=mock_resp), \
             patch("img_llm.FILENAME_MODELS", ["test-model"]), \
             patch("img_llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"):
            # Result is lowercased to "test name" → joined "test_name" → valid
            result = query_llm_for_filename("x", "http://localhost:1337")
            assert result == "test_name"


class TestQueryMlxForFilename:
    def test_successful_query(self):
        from img_llm import query_mlx_for_filename
        mock_model = MagicMock()
        mock_model.name = "test-mlx-model"
        with patch("img_llm.find_mlx_model", return_value=mock_model), \
             patch("img_llm.call_mlx", return_value="mlx_output_name"), \
             patch("img_llm.process_mlx_content", return_value="mlx_output_name"), \
             patch("img_llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"):
            result = query_mlx_for_filename("test text")
            assert result == "mlx_output_name"

    def test_no_model_found(self):
        from img_llm import query_mlx_for_filename
        with patch("img_llm.find_mlx_model", return_value=None), \
             patch("img_llm.FILENAME_MODELS", ["test-model"]), \
             patch("img_llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"):
            assert query_mlx_for_filename("test text") is None

    def test_empty_output(self):
        from img_llm import query_mlx_for_filename
        mock_model = MagicMock()
        with patch("img_llm.find_mlx_model", return_value=mock_model), \
             patch("img_llm.call_mlx", return_value=""), \
             patch("img_llm.FILENAME_MODELS", ["test-model"]), \
             patch("img_llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"):
            assert query_mlx_for_filename("test text") is None

    def test_call_mlx_exception(self):
        from img_llm import query_mlx_for_filename
        mock_model = MagicMock()
        with patch("img_llm.find_mlx_model", return_value=mock_model), \
             patch("img_llm.call_mlx", side_effect=Exception("mlx error")), \
             patch("img_llm.FILENAME_MODELS", ["test-model"]), \
             patch("img_llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"):
            assert query_mlx_for_filename("test text") is None


class TestQueryVlmForFilename:
    def test_successful_vlm_query(self):
        from img_llm import query_vlm_for_filename
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            '{"message": {"content": "white goose grass"}}\n'
            '{"message": {"content": "", "done": true}}\n'
        )
        with patch("builtins.open", mock_open(read_data=b"fake_image_data")), \
             patch("img_llm.requests.post", return_value=mock_resp):
            result = query_vlm_for_filename(
                Path("/fake/image.png"), "http://localhost:1337", "vlm-model"
            )
            assert result == "white goose grass"

    def test_api_error(self):
        from img_llm import query_vlm_for_filename
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal server error"
        with patch("builtins.open", mock_open(read_data=b"data")), \
             patch("img_llm.requests.post", return_value=mock_resp):
            result = query_vlm_for_filename(
                Path("/fake/image.png"), "http://localhost:1337", "vlm-model"
            )
            assert result is None

    def test_file_read_exception(self):
        from img_llm import query_vlm_for_filename
        with patch("builtins.open", side_effect=Exception("file error")):
            result = query_vlm_for_filename(
                Path("/fake/image.png"), "http://localhost:1337", "vlm-model"
            )
            assert result is None

    def test_with_api_key(self):
        from img_llm import query_vlm_for_filename
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"message": {"content": "output"}}\n{"message": {"done": true}}\n'
        with patch("builtins.open", mock_open(read_data=b"data")), \
             patch("img_llm.requests.post") as mock_post:
            mock_post.return_value = mock_resp
            result = query_vlm_for_filename(
                Path("/fake/image.png"), "http://localhost:1337", "vlm-model", api_key="mykey"
            )
            assert result == "output"
            headers = mock_post.call_args[1].get("headers", {})
            assert headers.get("Authorization") == "Bearer mykey"

    def test_vlm_done_break(self):
        """Line 210: VLM stream ends with done=true at top level (currently unreachable)."""
        from img_llm import query_vlm_for_filename
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # done at TOP level of JSON (per img_llm.py line 209: j.get("done"))
        mock_resp.text = (
            '{"message": {"content": "first"}}\n'
            '{"done": true, "message": {"content": "more"}}\n'
        )
        with patch("builtins.open", mock_open(read_data=b"data")), \
             patch("img_llm.requests.post", return_value=mock_resp):
            result = query_vlm_for_filename(
                Path("/fake/image.png"), "http://localhost:1337", "vlm-model"
            )
            assert result == "firstmore"

    def test_vlm_invalid_json_continues(self):
        """Lines 211-212: invalid JSON in VLM stream is skipped."""
        from img_llm import query_vlm_for_filename
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            'not valid json\n'
            '{"message": {"content": "valid_part", "done": true}}\n'
        )
        with patch("builtins.open", mock_open(read_data=b"data")), \
             patch("img_llm.requests.post", return_value=mock_resp):
            result = query_vlm_for_filename(
                Path("/fake/image.png"), "http://localhost:1337", "vlm-model"
            )
            assert result == "valid_part"

    def test_query_llm_with_done_break(self):
        """Line 118: query_llm_for_filename stops on done (at top level of JSON)."""
        from img_llm import query_llm_for_filename
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # done is at top level (line 117: j.get("done", False))
        mock_resp.text = (
            '{"message": {"content": "first_"}}\n'
            '{"message": {"content": "name"}, "done": true}\n'
        )
        with patch("img_llm.requests.post", return_value=mock_resp), \
             patch("img_llm.FILENAME_MODELS", ["test-model"]), \
             patch("img_llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"):
            result = query_llm_for_filename("x", "http://localhost:1337")
            assert result == "first_name"
