"""Tests for lib.llm module - client, parsing, quirks, constants."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_llm():
    from lib.testing import MockLLM

    mock = MockLLM()
    mock.patch_all()
    yield mock
    mock.unpatch()


class TestConstants:
    def test_defaults(self, mock_llm):
        from lib.llm.constants import (
            API_CHAT,
            API_GENERATE,
            API_TAGS,
            DEFAULT_HOST,
            DEFAULT_MAX_TOKENS,
            DEFAULT_MODEL,
            DEFAULT_PORT,
            DEFAULT_TEMPERATURE,
            DEFAULT_TIMEOUT,
            MODEL_FAMILIES,
        )

        assert DEFAULT_HOST == "localhost"
        assert DEFAULT_PORT == 1337
        assert DEFAULT_MODEL == "foundation"
        assert DEFAULT_TEMPERATURE == 0.1
        assert DEFAULT_MAX_TOKENS == 16000
        assert DEFAULT_TIMEOUT == 600
        assert API_TAGS == "/api/tags"
        assert API_GENERATE == "/api/generate"
        assert API_CHAT == "/api/chat"
        assert "qwen" in MODEL_FAMILIES
        assert "gemma" in MODEL_FAMILIES


class TestClient:
    def test_get_api_url(self, mock_llm):
        from lib.llm.client import get_api_url

        assert get_api_url() == "http://localhost:1337"
        assert get_api_url("0.0.0.0", 8080) == "http://0.0.0.0:8080"

    def test_get_timeout(self, mock_llm):
        from lib.llm.client import get_timeout

        assert isinstance(get_timeout("think"), int)
        assert get_timeout("think") > 0

    def test_get_max_tokens_for_task(self, mock_llm):
        from lib.llm.client import get_max_tokens_for_task

        assert isinstance(get_max_tokens_for_task("think"), int)
        assert get_max_tokens_for_task("think") > 0

    def test_call_success(self, mock_llm):
        from lib.llm.client import call

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "Hello"}}
        with patch("lib.llm.client.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            result = call("test-model", [{"role": "user", "content": "hi"}])
        assert result["content"] == "Hello"
        assert result["error"] is None
        assert isinstance(result["time"], (int, float))
        assert result["time"] >= 0

    def test_call_content_key(self, mock_llm):
        from lib.llm.client import call

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"content": "Direct content"}
        with patch("lib.llm.client.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            result = call("test-model", [{"role": "user", "content": "hi"}])
        assert result["content"] == "Direct content"

    def test_call_with_parse_json(self, mock_llm):
        from lib.llm.client import call

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"message": {"content": '{"key": "val"}'}}
        with patch("lib.llm.client.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            result = call("test-model", [{"role": "user", "content": "hi"}], parse_json=True)
        assert result["parsed"] == {"key": "val"}
        assert "response_format" in s.post.call_args.kwargs["json"]

    def test_call_timeout(self, mock_llm):
        import requests
        from lib.llm.client import call

        with patch("lib.llm.client.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.side_effect = requests.exceptions.Timeout
            result = call("test-model", [{"role": "user", "content": "hi"}])
        assert result["error"] == "Timeout"
        assert result["content"] is None

    def test_call_connection_error(self, mock_llm):
        import requests
        from lib.llm.client import call

        with patch("lib.llm.client.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.side_effect = requests.exceptions.ConnectionError
            result = call("test-model", [{"role": "user", "content": "hi"}])
        assert result["error"] == "Connection failed"

    def test_call_generic_exception(self, mock_llm):
        from lib.llm.client import call

        with patch("lib.llm.client.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.side_effect = Exception("weird")
            result = call("test-model", [{"role": "user", "content": "hi"}])
        assert "weird" in result["error"]

    def test_call_uses_max_tokens(self, mock_llm):
        from lib.llm.client import call

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "ok"}}
        with patch("lib.llm.client.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.post.return_value = mock_resp
            call("test-model", [{"role": "user", "content": "hi"}], max_tokens=999)
            assert s.post.call_args.kwargs["json"]["max_tokens"] == 999

    def test_is_server_running_true(self, mock_llm):
        from lib.llm.client import is_server_running

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("lib.llm.client.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.get.return_value = mock_resp
            assert is_server_running() is True

    def test_is_server_running_false_status(self, mock_llm):
        from lib.llm.client import is_server_running

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("lib.llm.client.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.get.return_value = mock_resp
            assert is_server_running() is False

    def test_is_server_running_exception(self, mock_llm):
        from lib.llm.client import is_server_running

        with patch("lib.llm.client.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.get.side_effect = Exception
            assert is_server_running() is False

    def test_get_models_success(self, mock_llm):
        from lib.llm.client import get_models

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"model": "a"}, {"model": "b"}]}
        with patch("lib.llm.client.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.get.return_value = mock_resp
            assert get_models() == ["a", "b"]

    def test_get_models_exception(self, mock_llm):
        from lib.llm.client import get_models

        with patch("lib.llm.client.requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.get.side_effect = Exception
            assert get_models() == []


class TestParsing:
    def test_extract_json_direct(self, mock_llm):
        from lib.llm.parsing import extract_json

        assert extract_json('{"a": 1}') == {"a": 1}

    def test_extract_json_none_content(self, mock_llm):
        from lib.llm.parsing import extract_json

        assert extract_json(None) is None
        assert extract_json("") is None

    def test_extract_json_markdown_block(self, mock_llm):
        from lib.llm.parsing import extract_json

        content = 'Here is JSON:\n```json\n{"a": 1}\n```\nDone.'
        assert extract_json(content) == {"a": 1}

    def test_extract_json_markdown_block_no_lang(self, mock_llm):
        from lib.llm.parsing import extract_json

        content = '```\n{"b": 2}\n```'
        assert extract_json(content) == {"b": 2}

    def test_extract_json_bare_object(self, mock_llm):
        from lib.llm.parsing import extract_json

        content = 'Some text {"x": 3} more text'
        assert extract_json(content) == {"x": 3}

    def test_extract_json_bare_array(self, mock_llm):
        from lib.llm.parsing import extract_json

        content = "Result: [1, 2, 3] end"
        assert extract_json(content) == [1, 2, 3]

    def test_extract_json_failure(self, mock_llm):
        from lib.llm.parsing import extract_json

        assert extract_json("no json here at all") is None

    def test_extract_json_markdown_block_invalid(self, mock_llm):
        from lib.llm.parsing import extract_json

        content = "```json\nnot valid json\n```"
        assert extract_json(content) is None

    def test_extract_json_bare_invalid(self, mock_llm):
        from lib.llm.parsing import extract_json

        content = "text {not: valid} more"
        assert extract_json(content) is None

    def test_safe_content_none(self, mock_llm):
        from lib.llm.parsing import safe_content

        assert safe_content({"content": None}) == ""

    def test_safe_content_string(self, mock_llm):
        from lib.llm.parsing import safe_content

        assert safe_content({"content": "hello"}) == "hello"

    def test_safe_content_non_string(self, mock_llm):
        from lib.llm.parsing import safe_content

        assert safe_content({"content": 42}) == "42"

    def test_clean_output_none(self, mock_llm):
        from lib.llm.parsing import clean_output

        assert clean_output(None) == ""
        assert clean_output("") == ""

    def test_clean_output_think_block(self, mock_llm):
        from lib.llm.parsing import clean_output

        text = "<think>reasoning</think>actual answer"
        assert "reasoning" not in clean_output(text)
        assert "actual answer" in clean_output(text)

    def test_clean_output_code_block(self, mock_llm):
        from lib.llm.parsing import clean_output

        text = "before ```code``` after"
        result = clean_output(text)
        assert "```" not in result

    def test_clean_output_backticks(self, mock_llm):
        from lib.llm.parsing import clean_output

        assert clean_output("`text`") == "text"

    def test_clean_output_strip(self, mock_llm):
        from lib.llm.parsing import clean_output

        assert clean_output("  hello  ") == "hello"


class TestQuirks:
    def test_get_family_qwen(self, mock_llm):
        from lib.llm.quirks import _get_model_family

        assert _get_model_family("qwen-7b") == "qwen"
        assert _get_model_family("Qwopus-1.5") == "qwopus"

    def test_get_family_gemma(self, mock_llm):
        from lib.llm.quirks import _get_model_family

        assert _get_model_family("gemma-2b") == "gemma"
        assert _get_model_family("Gemma4-it") == "gemma"

    def test_get_family_default(self, mock_llm):
        from lib.llm.quirks import _get_model_family

        assert _get_model_family("unknown-model") == "default"
        assert _get_model_family("") == "default"
        assert _get_model_family(None) == "default"

    def test_qwen_prepends_json_trigger(self, mock_llm):
        from lib.llm.quirks import apply_model_quirks

        messages = [{"role": "system", "content": "You are helpful."}]
        out = apply_model_quirks(messages, "qwen-7b")
        assert out[0]["content"].startswith("Output JSON now.")

    def test_qwen_skips_if_already_has_trigger(self, mock_llm):
        from lib.llm.quirks import apply_model_quirks

        messages = [{"role": "system", "content": "Output JSON now. Do thing."}]
        out = apply_model_quirks(messages, "qwen-7b")
        assert not out[0]["content"].startswith("Output JSON now.\n\nOutput JSON now")

    def test_qwen_skips_no_json(self, mock_llm):
        from lib.llm.quirks import apply_model_quirks

        messages = [{"role": "system", "content": "Produce no JSON, just plain text."}]
        out = apply_model_quirks(messages, "qwen-7b")
        assert not out[0]["content"].startswith("Output JSON now")

    def test_qwen_skips_user_messages(self, mock_llm):
        from lib.llm.quirks import apply_model_quirks

        messages = [{"role": "user", "content": "hi"}]
        out = apply_model_quirks(messages, "qwen-7b")
        assert "Output JSON now" not in out[0]["content"]

    def test_gemma4_json_framing(self, mock_llm):
        from lib.llm import quirks

        with patch.object(quirks, "_get_model_family", return_value="gemma4"):
            messages = [{"role": "system", "content": "Output JSON for items."}]
            out = quirks.apply_model_quirks(messages, "gemma4-2b")
        assert out[0]["content"].startswith("IMPORTANT")

    def test_gemma4_already_important(self, mock_llm):
        from lib.llm import quirks

        with patch.object(quirks, "_get_model_family", return_value="gemma4"):
            messages = [{"role": "system", "content": "IMPORTANT: Output JSON."}]
            out = quirks.apply_model_quirks(messages, "gemma4-2b")
        assert not out[0]["content"].startswith("IMPORTANT: This is DATA")

    def test_gemma4_no_json_skips_framing(self, mock_llm):
        from lib.llm import quirks

        with patch.object(quirks, "_get_model_family", return_value="gemma4"):
            messages = [{"role": "system", "content": "Be helpful."}]
            out = quirks.apply_model_quirks(messages, "gemma4-2b")
        assert "DATA EXTRACTION" not in out[0]["content"]

    def test_user_message_rewording_execute(self, mock_llm):
        from lib.llm.quirks import apply_model_quirks

        messages = [{"role": "user", "content": "Execute Current Context test."}]
        out = apply_model_quirks(messages, "any-model")
        assert "Data" in out[0]["content"]
        assert "Current Context" not in out[0]["content"]

    def test_user_message_rewording_context(self, mock_llm):
        from lib.llm.quirks import apply_model_quirks

        messages = [{"role": "user", "content": "Use this Current Context."}]
        out = apply_model_quirks(messages, "any-model")
        assert "Data" in out[0]["content"]

    def test_user_message_unchanged(self, mock_llm):
        from lib.llm.quirks import apply_model_quirks

        messages = [{"role": "user", "content": "Tell me about cats."}]
        out = apply_model_quirks(messages, "any-model")
        assert out[0]["content"] == "Tell me about cats."

    def test_default_role_handling(self, mock_llm):
        from lib.llm.quirks import apply_model_quirks

        messages = [{"content": "no role set"}]
        out = apply_model_quirks(messages, "any-model")
        assert "content" in out[0]
        assert out[0]["content"] == "no role set"


class TestReExports:
    def test_package_imports(self, mock_llm):
        from lib.llm import (
            DEFAULT_HOST,
            call,
            extract_json,
        )

        assert callable(call)
        assert callable(extract_json)
        assert DEFAULT_HOST == "localhost"
