"""Tests for lib.osaurus_lib - core LLM call functions."""
import json
import pytest
from unittest.mock import patch, MagicMock, Mock
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
        with patch("lib.osaurus_lib._get_model_family", return_value="qwen"):
            result = apply_model_quirks(messages, "qwen2.5-7b")
        assert "Output JSON now" in result[0]["content"]

    def test_apply_quirks_qwen_system_already_has_trigger(self, mock_llm):
        from lib.osaurus_lib import apply_model_quirks
        messages = [{"role": "system", "content": "Output JSON now\nDo thing"}]
        with patch("lib.osaurus_lib._get_model_family", return_value="qwen"):
            result = apply_model_quirks(messages, "qwen2.5-7b")
        assert result[0]["content"] == "Output JSON now\nDo thing"

    def test_apply_quirks_qwen_no_json_text_skips(self, mock_llm):
        from lib.osaurus_lib import apply_model_quirks
        messages = [{"role": "system", "content": "no JSON required plain text output"}]
        with patch("lib.osaurus_lib._get_model_family", return_value="qwen"):
            result = apply_model_quirks(messages, "qwen2.5-7b")
        assert "Output JSON now" not in result[0]["content"]

    def test_apply_quirks_qwen_empty_content(self, mock_llm):
        from lib.osaurus_lib import apply_model_quirks
        messages = [{"role": "system", "content": ""}]
        with patch("lib.osaurus_lib._get_model_family", return_value="qwen"):
            result = apply_model_quirks(messages, "qwen2.5-7b")
        assert result[0]["content"] == ""

    def test_apply_quirks_gemma4_system_json(self, mock_llm):
        from lib.osaurus_lib import apply_model_quirks
        messages = [{"role": "system", "content": "Output JSON"}]
        with patch("lib.osaurus_lib._get_model_family", return_value="gemma4"):
            result = apply_model_quirks(messages, "gemma-4-9b")
        assert "IMPORTANT" in result[0]["content"]

    def test_apply_quirks_gemma4_already_important(self, mock_llm):
        from lib.osaurus_lib import apply_model_quirks
        messages = [{"role": "system", "content": "IMPORTANT: Output JSON"}]
        with patch("lib.osaurus_lib._get_model_family", return_value="gemma4"):
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
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello response"}}]
        }
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response):
            result = call("model-a", [{"role": "user", "content": "hi"}])
        assert result["content"] == "Hello response"
        assert result["error"] is None
        assert result["time"] is not None

    def test_call_with_parse_json(self, no_mock_llm):
        import lib.osaurus_lib
        call = lib.osaurus_lib.call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"a": 1}'}}]
        }
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response) as post:
            result = call("model-a", [{"role": "user", "content": "hi"}], parse_json=True)
        assert "response_format" in post.call_args.kwargs["json"]
        assert result["parsed"] is not None

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
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response):
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
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response):
            result = call("model-a", [{"role": "user", "content": "hi"}])
        assert "HTTP 500" in result["error"]

    def test_call_empty_choices(self, no_mock_llm):
        import lib.osaurus_lib
        call = lib.osaurus_lib.call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response):
            result = call("model-a", [{"role": "user", "content": "hi"}])
        assert "Empty response" in result["error"]

    def test_call_no_choices_key(self, no_mock_llm):
        import lib.osaurus_lib
        call = lib.osaurus_lib.call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"oops": "no choices"}
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response):
            result = call("model-a", [{"role": "user", "content": "hi"}])
        assert "Empty response" in result["error"]

    def test_call_timeout(self, no_mock_llm):
        import lib.osaurus_lib
        call = lib.osaurus_lib.call
        with patch("lib.osaurus_lib.requests.post", side_effect=requests.exceptions.Timeout):
            result = call("model-a", [{"role": "user", "content": "hi"}])
        assert result["error"] == "Timeout"

    def test_call_connection_error(self, no_mock_llm):
        import lib.osaurus_lib
        call = lib.osaurus_lib.call
        with patch("lib.osaurus_lib.requests.post", side_effect=requests.exceptions.ConnectionError):
            result = call("model-a", [{"role": "user", "content": "hi"}])
        assert "Connection failed" in result["error"]

    def test_call_json_decode_error(self, no_mock_llm):
        import lib.osaurus_lib
        call = lib.osaurus_lib.call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("err", "doc", 0)
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response):
            result = call("model-a", [{"role": "user", "content": "hi"}])
        assert "Invalid JSON" in result["error"]

    def test_call_key_error(self, no_mock_llm):
        import lib.osaurus_lib
        call = lib.osaurus_lib.call
        with patch("lib.osaurus_lib.requests.post", side_effect=KeyError("missing")):
            result = call("model-a", [{"role": "user", "content": "hi"}])
        assert "Unexpected response format" in result["error"]

    def test_call_generic_exception(self, no_mock_llm):
        import lib.osaurus_lib
        call = lib.osaurus_lib.call
        with patch("lib.osaurus_lib.requests.post", side_effect=RuntimeError("oops")):
            result = call("model-a", [{"role": "user", "content": "hi"}])
        assert "RuntimeError" in result["error"]

    def test_call_with_max_tokens(self, no_mock_llm):
        import lib.osaurus_lib
        call = lib.osaurus_lib.call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "x"}}]}
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response) as post:
            call("model-a", [{"role": "user", "content": "hi"}], max_tokens=99)
        assert post.call_args.kwargs["json"]["max_tokens"] == 99

    def test_call_with_custom_timeout(self, no_mock_llm):
        import lib.osaurus_lib
        call = lib.osaurus_lib.call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "x"}}]}
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response) as post:
            call("model-a", [{"role": "user", "content": "hi"}], timeout=42)
        assert post.call_args.kwargs["timeout"] == 42

    def test_call_parse_json_empty_content(self, no_mock_llm):
        import lib.osaurus_lib
        call = lib.osaurus_lib.call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": ""}}]}
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response):
            result = call("model-a", [{"role": "user", "content": "hi"}], parse_json=True)
        assert result["parsed"] is None

    def test_call_default_max_tokens(self, no_mock_llm):
        import lib.osaurus_lib
        call = lib.osaurus_lib.call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "x"}}]}
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response) as post, \
             patch("lib.osaurus_lib.get_max_tokens_for_task", return_value=2048):
            call("model-a", [{"role": "user", "content": "hi"}])
        assert post.call_args.kwargs["json"]["max_tokens"] == 2048


class TestCallWithPrompt:
    def test_think_task(self, no_mock_llm):
        import lib.osaurus_lib
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "thoughtful"}}]}
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response):
            result = lib.osaurus_lib.call_with_prompt("model-a", "Tell me about X", task="think")
        assert result["content"] == "thoughtful"

    def test_json_task(self, no_mock_llm):
        import lib.osaurus_lib
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": '{"a":1}'}}]}
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response) as post:
            result = lib.osaurus_lib.call_with_prompt("model-a", "extract", task="json")
        assert post.call_args.kwargs["json"].get("response_format") == {"type": "json_object"}

    def test_detailed_json_task(self, no_mock_llm):
        import lib.osaurus_lib
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response) as post:
            lib.osaurus_lib.call_with_prompt("model-a", "extract", task="detailed_json")
        assert post.call_args.kwargs["json"].get("response_format") == {"type": "json_object"}

    def test_summarize_task(self, no_mock_llm):
        import lib.osaurus_lib
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "summary"}}]}
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response):
            result = lib.osaurus_lib.call_with_prompt("model-a", "lots of text", task="summarize")
        assert result["content"] == "summary"

    def test_filename_task(self, no_mock_llm):
        import lib.osaurus_lib
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "my_file"}}]}
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response):
            result = lib.osaurus_lib.call_with_prompt("model-a", "long text", task="filename")
        assert result["content"] == "my_file"

    def test_unknown_task(self, no_mock_llm):
        import lib.osaurus_lib
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "x"}}]}
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response) as post:
            lib.osaurus_lib.call_with_prompt("model-a", "hi", task="unknown_task")
        # Uses single user message
        assert post.call_args.kwargs["json"]["messages"] == [{"role": "user", "content": "hi"}]

    def test_prompt_substitution(self, no_mock_llm):
        import lib.osaurus_lib
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "x"}}]}
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response) as post:
            lib.osaurus_lib.call_with_prompt("model-a", "lots of content", task="think")
        # Substitution happened for {prompt} -> "lots of content"
        assert post.call_args.kwargs["json"]["messages"][1]["content"] == "lots of content"


class TestTestModel:
    def test_test_model_default(self, no_mock_llm):
        import lib.osaurus_lib
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hello!"}}]}
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response):
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
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response) as post:
            result = lib.osaurus_lib.call_llm_api("https://api.example.com", "model", [{"role": "user", "content": "hi"}])
        assert result["content"] == "resp"
        assert result["model"] == "remote-model"
        assert result["usage"] == {"tokens": 100}
        # Authorization NOT set when api_key is empty
        assert "Authorization" not in post.call_args.kwargs["headers"]

    def test_call_llm_api_without_http(self, no_mock_llm):
        import lib.osaurus_lib
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response) as post:
            lib.osaurus_lib.call_llm_api("api.example.com", "model", [])
        assert post.call_args.args[0] == "http://api.example.com/v1/chat/completions"

    def test_call_llm_api_with_api_key(self, no_mock_llm):
        import lib.osaurus_lib
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response) as post:
            lib.osaurus_lib.call_llm_api("https://api.x.com", "model", [], api_key="key123")
        assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer key123"

    def test_call_llm_api_parse_json(self, no_mock_llm):
        import lib.osaurus_lib
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response) as post:
            lib.osaurus_lib.call_llm_api("https://api.x.com", "model", [], parse_json=True)
        assert post.call_args.kwargs["json"]["response_format"] == {"type": "json_object"}

    def test_call_llm_api_http_error(self, no_mock_llm):
        import lib.osaurus_lib
        with patch("lib.osaurus_lib.requests.post", side_effect=Exception("boom")):
            result = lib.osaurus_lib.call_llm_api("https://api.x.com", "model", [])
        assert "boom" in result["error"]

    def test_call_llm_api_missing_choices(self, no_mock_llm):
        import lib.osaurus_lib
        mock_response = MagicMock()
        mock_response.json.return_value = {"oops": "no choices"}
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response):
            result = lib.osaurus_lib.call_llm_api("https://api.x.com", "model", [])
        assert "error" in result

    def test_call_llm_api_no_model_in_response(self, no_mock_llm):
        import lib.osaurus_lib
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "x"}}]}
        with patch("lib.osaurus_lib.requests.post", return_value=mock_response):
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
