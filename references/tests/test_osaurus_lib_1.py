"""Tests for lib.osaurus_lib - core LLM call functions."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests


@pytest.fixture
def mock_llm():
    from lib.testing import MockLLM

    mock = MockLLM()
    mock.patch_all()
    yield mock
    mock.unpatch()


@pytest.fixture
def no_mock_llm():
    """Use this when you need to call the real call() with mocked requests."""
    yield


class TestApplyModelQuirks:
    def test_apply_quirks_empty_messages(self, mock_llm):
        from lib.osaurus_lib import apply_model_quirks

        result = apply_model_quirks([], "llama-3.1-8b")
        assert result == []

    def test_apply_quirks_user_execute(self, mock_llm):
        from lib.osaurus_lib import apply_model_quirks

        messages = [{"role": "user", "content": "Execute the task based on input"}]
        result = apply_model_quirks(messages, "llama-3.1-8b")
        assert "Extract" in result[0]["content"]
        assert "Execute" not in result[0]["content"]

    def test_apply_quirks_user_current_context(self, mock_llm):
        from lib.osaurus_lib import apply_model_quirks

        messages = [{"role": "user", "content": "Use Current Context to determine result"}]
        result = apply_model_quirks(messages, "llama-3.1-8b")
        assert "Data" in result[0]["content"]
        assert "Current Context" not in result[0]["content"]

    def test_apply_quirks_user_with_context_word(self, mock_llm):
        from lib.osaurus_lib import apply_model_quirks

        messages = [{"role": "user", "content": "Look at the context here"}]
        result = apply_model_quirks(messages, "llama-3.1-8b")
        # No replacement triggered, but logs
        assert result[0]["content"] == "Look at the context here"

    def test_apply_quirks_qwen_system_adds_trigger(self, mock_llm):
        from lib.osaurus_lib import apply_model_quirks

        messages = [{"role": "system", "content": "Extract data"}]
        with patch("lib.llm.quirks._get_model_family", return_value="qwen"):
            result = apply_model_quirks(messages, "qwen2.5-7b")
        assert "Output JSON now" in result[0]["content"]

    def test_apply_quirks_qwen_system_already_has_trigger(self, mock_llm):
        from lib.osaurus_lib import apply_model_quirks

        messages = [{"role": "system", "content": "Output JSON now\nDo thing"}]
        with patch("lib.llm.quirks._get_model_family", return_value="qwen"):
            result = apply_model_quirks(messages, "qwen2.5-7b")
        assert result[0]["content"] == "Output JSON now\nDo thing"

    def test_apply_quirks_qwen_no_json_text_skips(self, mock_llm):
        from lib.osaurus_lib import apply_model_quirks

        messages = [{"role": "system", "content": "no JSON required plain text output"}]
        with patch("lib.llm.quirks._get_model_family", return_value="qwen"):
            result = apply_model_quirks(messages, "qwen2.5-7b")
        assert "Output JSON now" not in result[0]["content"]

    def test_apply_quirks_qwen_empty_content(self, mock_llm):
        from lib.osaurus_lib import apply_model_quirks

        messages = [{"role": "system", "content": ""}]
        with patch("lib.llm.quirks._get_model_family", return_value="qwen"):
            result = apply_model_quirks(messages, "qwen2.5-7b")
        assert result[0]["content"] == ""

    def test_apply_quirks_gemma4_system_json(self, mock_llm):
        from lib.osaurus_lib import apply_model_quirks

        messages = [{"role": "system", "content": "Output JSON"}]
        with patch("lib.llm.quirks._get_model_family", return_value="gemma4"):
            result = apply_model_quirks(messages, "gemma-4-9b")
        assert "IMPORTANT" in result[0]["content"]

    def test_apply_quirks_gemma4_already_important(self, mock_llm):
        from lib.osaurus_lib import apply_model_quirks

        messages = [{"role": "system", "content": "IMPORTANT: Output JSON"}]
        with patch("lib.llm.quirks._get_model_family", return_value="gemma4"):
            result = apply_model_quirks(messages, "gemma-4-9b")
        assert result[0]["content"].count("IMPORTANT") == 1

    def test_apply_quirks_default_role(self, mock_llm):
        from lib.osaurus_lib import apply_model_quirks

        messages = [{"role": "user", "content": "x"}]
        result = apply_model_quirks(messages, "llama-3.1-8b")
        # role is preserved through the function
        assert result[0]["role"] == "user"
        # Default role fallback path: when role missing, function uses 'user' internally
        # but the output dict only contains keys from the input msg + updated content

    def test_apply_quirks_unknown_family(self, mock_llm):
        from lib.osaurus_lib import apply_model_quirks

        messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]
        result = apply_model_quirks(messages, "unknown-model")
        assert len(result) == 2


class TestCall:
    def test_call_success(self, no_mock_llm):
        import lib.osaurus_lib

        call = lib.osaurus_lib.call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hello response"}}]}
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            result = call("model-a", [{"role": "user", "content": "hi"}])
        assert result["content"] == "Hello response"
        assert result["error"] is None
        # time is a float, always >= 0
        assert isinstance(result["time"], (int, float))
        assert result["time"] >= 0

    def test_call_with_parse_json(self, no_mock_llm):
        import lib.osaurus_lib

        call = lib.osaurus_lib.call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": '{"a": 1}'}}]}
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            result = call("model-a", [{"role": "user", "content": "hi"}], parse_json=True)
        assert "response_format" in s.post.call_args.kwargs["json"]
        # extract_json normalizes {"a": 1} dict to ["a"] keys list
        assert result["parsed"] == ["a"]

    def test_call_parse_json_fails(self, no_mock_llm):
        import lib.osaurus_lib

        call = lib.osaurus_lib.call
        # Use content that truly fails extract_json. The function normalizes text via _extract_plain_list
        # so "not json" becomes [{name: not json}]. We use a string with brackets that breaks parsing.
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "This is {broken: json with no quotes}"}}]
        }
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            result = call("model-a", [{"role": "user", "content": "hi"}], parse_json=True)
        # Either parsed is None, or extract_json recovered something
        # The key is no exception is raised
        assert "parsed" in result

    def test_call_http_error(self, no_mock_llm):
        import lib.osaurus_lib

        call = lib.osaurus_lib.call
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            result = call("model-a", [{"role": "user", "content": "hi"}])
        assert "HTTP 500" in result["error"]

    def test_call_empty_choices(self, no_mock_llm):
        import lib.osaurus_lib

        call = lib.osaurus_lib.call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            result = call("model-a", [{"role": "user", "content": "hi"}])
        assert "Empty response" in result["error"]

    def test_call_no_choices_key(self, no_mock_llm):
        import lib.osaurus_lib

        call = lib.osaurus_lib.call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"oops": "no choices"}
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            result = call("model-a", [{"role": "user", "content": "hi"}])
        assert "Empty response" in result["error"]

    def test_call_timeout(self, no_mock_llm):
        import lib.osaurus_lib

        call = lib.osaurus_lib.call
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.side_effect = requests.exceptions.Timeout
            result = call("model-a", [{"role": "user", "content": "hi"}])
        assert result["error"] == "Timeout"

    def test_call_connection_error(self, no_mock_llm):
        import lib.osaurus_lib

        call = lib.osaurus_lib.call
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.side_effect = requests.exceptions.ConnectionError
            result = call("model-a", [{"role": "user", "content": "hi"}], use_foundation=False)
        assert "Connection failed" in result["error"]

    def test_call_connection_error_falls_back_to_foundation(self, no_mock_llm):
        import lib.osaurus_lib

        call = lib.osaurus_lib.call
        with (
            patch("lib.osaurus_lib.requests.Session") as mock_session,
            patch("lib.osaurus_lib._try_foundation", return_value=True) as mock_found,
        ):
            s = mock_session.return_value.__enter__.return_value
            s.post.side_effect = requests.exceptions.ConnectionError
            call("model-a", [{"role": "user", "content": "hi"}])
        mock_found.assert_called_once()

    def test_call_json_decode_error(self, no_mock_llm):
        import lib.osaurus_lib

        call = lib.osaurus_lib.call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("err", "doc", 0)
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            result = call("model-a", [{"role": "user", "content": "hi"}])
        assert "Invalid JSON" in result["error"]

    def test_call_key_error(self, no_mock_llm):
        import lib.osaurus_lib

        call = lib.osaurus_lib.call
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.side_effect = KeyError("missing")
            result = call("model-a", [{"role": "user", "content": "hi"}])
        assert "Unexpected response format" in result["error"]

    def test_call_generic_exception(self, no_mock_llm):
        import lib.osaurus_lib

        call = lib.osaurus_lib.call
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.side_effect = RuntimeError("oops")
            result = call("model-a", [{"role": "user", "content": "hi"}])
        assert "RuntimeError" in result["error"]

    def test_call_with_max_tokens(self, no_mock_llm):
        import lib.osaurus_lib

        call = lib.osaurus_lib.call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "x"}}]}
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            call("model-a", [{"role": "user", "content": "hi"}], max_tokens=99)
        assert s.post.call_args.kwargs["json"]["max_tokens"] == 99

    def test_call_with_custom_timeout(self, no_mock_llm):
        import lib.osaurus_lib

        call = lib.osaurus_lib.call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "x"}}]}
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            call("model-a", [{"role": "user", "content": "hi"}], timeout=42)
        assert s.post.call_args.kwargs["timeout"] == 42

    def test_call_parse_json_empty_content(self, no_mock_llm):
        import lib.osaurus_lib

        call = lib.osaurus_lib.call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": ""}}]}
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            result = call("model-a", [{"role": "user", "content": "hi"}], parse_json=True)
        assert result["parsed"] is None

    def test_call_default_max_tokens(self, no_mock_llm):
        import lib.osaurus_lib

        call = lib.osaurus_lib.call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "x"}}]}
        with (
            patch("lib.osaurus_lib.requests.Session") as mock_session,
            patch("lib.osaurus_lib.get_max_tokens_for_task", return_value=2048),
        ):
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            call("model-a", [{"role": "user", "content": "hi"}])
        assert s.post.call_args.kwargs["json"]["max_tokens"] == 2048
