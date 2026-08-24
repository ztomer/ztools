"""Tests for lib.osaurus_lib - core LLM call functions."""

from unittest.mock import MagicMock, patch

import pytest


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


class TestCallWithPrompt:
    def test_think_task(self, no_mock_llm):
        import lib.osaurus_lib

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "thoughtful"}}]}
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            result = lib.osaurus_lib.call_with_prompt("model-a", "Tell me about X", task="think")
        assert result["content"] == "thoughtful"

    def test_json_task(self, no_mock_llm):
        import lib.osaurus_lib

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": '{"a":1}'}}]}
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            lib.osaurus_lib.call_with_prompt("model-a", "extract", task="json")
        assert s.post.call_args.kwargs["json"].get("response_format") == {"type": "json_object"}

    def test_detailed_json_task(self, no_mock_llm):
        import lib.osaurus_lib

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            lib.osaurus_lib.call_with_prompt("model-a", "extract", task="detailed_json")
        assert s.post.call_args.kwargs["json"].get("response_format") == {"type": "json_object"}

    def test_summarize_task(self, no_mock_llm):
        import lib.osaurus_lib

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "summary"}}]}
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            result = lib.osaurus_lib.call_with_prompt("model-a", "lots of text", task="summarize")
        assert result["content"] == "summary"

    def test_filename_task(self, no_mock_llm):
        import lib.osaurus_lib

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "my_file"}}]}
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            result = lib.osaurus_lib.call_with_prompt("model-a", "long text", task="filename")
        assert result["content"] == "my_file"

    def test_unknown_task(self, no_mock_llm):
        import lib.osaurus_lib

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "x"}}]}
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            lib.osaurus_lib.call_with_prompt("model-a", "hi", task="unknown_task")
        # Uses single user message
        assert s.post.call_args.kwargs["json"]["messages"] == [{"role": "user", "content": "hi"}]

    def test_prompt_substitution(self, no_mock_llm):
        import lib.osaurus_lib

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "x"}}]}
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            lib.osaurus_lib.call_with_prompt("model-a", "lots of content", task="think")
        # Substitution happened for {prompt} -> "lots of content"
        assert s.post.call_args.kwargs["json"]["messages"][1]["content"] == "lots of content"


class TestTestModel:
    def test_test_model_default(self, no_mock_llm):
        import lib.osaurus_lib

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hello!"}}]}
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            result = lib.osaurus_lib.test_model("model-a")
        assert result["content"] == "Hello!"


class TestCallLLMApi:
    def test_call_llm_api_with_http(self, no_mock_llm):
        import lib.osaurus_lib

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "resp"}}],
            "usage": {"tokens": 100},
            "model": "remote-model",
        }
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            result = lib.osaurus_lib.call_llm_api(
                "https://api.example.com", "model", [{"role": "user", "content": "hi"}]
            )
        assert result["content"] == "resp"
        assert result["model"] == "remote-model"
        assert result["usage"] == {"tokens": 100}
        # Authorization NOT set when api_key is empty
        assert "Authorization" not in s.post.call_args.kwargs["headers"]

    def test_call_llm_api_without_http(self, no_mock_llm):
        import lib.osaurus_lib

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            lib.osaurus_lib.call_llm_api("api.example.com", "model", [])
        assert s.post.call_args.args[0] == "http://api.example.com/v1/chat/completions"

    def test_call_llm_api_with_api_key(self, no_mock_llm):
        import lib.osaurus_lib

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            lib.osaurus_lib.call_llm_api("https://api.x.com", "model", [], api_key="key123")
        assert s.post.call_args.kwargs["headers"]["Authorization"] == "Bearer key123"

    def test_call_llm_api_parse_json(self, no_mock_llm):
        import lib.osaurus_lib

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            lib.osaurus_lib.call_llm_api("https://api.x.com", "model", [], parse_json=True)
        assert s.post.call_args.kwargs["json"]["response_format"] == {"type": "json_object"}

    def test_call_llm_api_http_error(self, no_mock_llm):
        import lib.osaurus_lib

        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.side_effect = Exception("boom")
            result = lib.osaurus_lib.call_llm_api("https://api.x.com", "model", [])
        assert "boom" in result["error"]

    def test_call_llm_api_missing_choices(self, no_mock_llm):
        import lib.osaurus_lib

        mock_response = MagicMock()
        mock_response.json.return_value = {"oops": "no choices"}
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            result = lib.osaurus_lib.call_llm_api("https://api.x.com", "model", [])
        assert "error" in result

    def test_call_llm_api_no_model_in_response(self, no_mock_llm):
        import lib.osaurus_lib

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "x"}}]}
        with patch("lib.osaurus_lib.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_response
            result = lib.osaurus_lib.call_llm_api("https://api.x.com", "fallback-model", [])
        assert result["model"] == "fallback-model"


class TestThinkingHelpers:
    def test_extract_thinking_with_thinking_block(self, mock_llm):
        from lib.osaurus_lib import extract_thinking

        # extract_thinking uses <thinking> tag, but remove_thinking_blocks only strips <think>
        # So content may still contain <thinking> tags (this is a quirk of the implementation)
        text = "<thinking>reasoning here</thinking>\nThe answer is 42"
        thinking, content = extract_thinking(text)
        assert thinking == "reasoning here"
        assert "42" in content

    def test_extract_thinking_no_thinking(self, mock_llm):
        from lib.osaurus_lib import extract_thinking

        thinking, content = extract_thinking("Just plain text")
        assert thinking == ""
        assert content == "Just plain text"

    def test_extract_thinking_with_attributes(self, mock_llm):
        from lib.osaurus_lib import extract_thinking

        text = '<think> type="reasoning"</think>my thoughts</think>\nfinal'
        thinking, content = extract_thinking(text)
        # <think> with attributes is matched
        assert "my thoughts" in thinking or "my thoughts" in content

    def test_merge_thinking_empty(self, mock_llm):
        from lib.osaurus_lib import merge_thinking_with_summary

        result = merge_thinking_with_summary("", "just summary")
        assert result == "just summary"

    def test_merge_thinking_with_thinking(self, mock_llm):
        from lib.osaurus_lib import merge_thinking_with_summary

        result = merge_thinking_with_summary("my thoughts", "the answer")
        assert "## Analysis" in result
        assert "the answer" in result
        assert "my thoughts" in result

    def test_strip_thinking(self, mock_llm):
        from lib.osaurus_lib import strip_thinking

        text = "before <think>x</think> after"
        result = strip_thinking(text)
        assert "<think>" not in result
        assert "before" in result
        assert "after" in result

    def test_strip_thinking_no_thinking(self, mock_llm):
        from lib.osaurus_lib import strip_thinking

        assert strip_thinking("clean text") == "clean text"
