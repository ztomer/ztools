"""Tests for lib.osaurus_models - URL builders, server checks, model selection."""
import os
import pytest
import requests
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_llm():
    from lib.testing import MockLLM
    mock = MockLLM()
    mock.patch_all()
    yield mock
    mock.unpatch()


class TestUrlBuilders:
    def test_get_api_url(self, mock_llm):
        from lib.osaurus_models import get_api_url
        assert get_api_url() == "http://localhost:1337/v1/chat/completions"

    def test_get_api_url_custom(self, mock_llm):
        from lib.osaurus_models import get_api_url
        assert get_api_url("0.0.0.0", 8080) == "http://0.0.0.0:8080/v1/chat/completions"

    def test_get_base_url(self, mock_llm):
        from lib.osaurus_models import get_base_url
        assert get_base_url() == "http://localhost:1337"

    def test_get_base_url_custom(self, mock_llm):
        from lib.osaurus_models import get_base_url
        assert get_base_url("host.com", 443) == "http://host.com:443"


class TestGetModels:
    def test_get_models_success(self, mock_llm):
        from lib.osaurus_models import get_models
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "a"}, {"id": "b"}]}
        with patch("lib.osaurus_models.requests.get", return_value=mock_resp):
            assert get_models() == ["a", "b"]

    def test_get_models_with_api_key(self, mock_llm):
        from lib.osaurus_models import get_models
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "a"}]}
        with patch("lib.osaurus_models.requests.get", return_value=mock_resp) as get:
            get_models("localhost", 1337, api_key="secret")
        assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer secret"

    def test_get_models_http_prefix(self, mock_llm):
        from lib.osaurus_models import get_models
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        with patch("lib.osaurus_models.requests.get", return_value=mock_resp) as get:
            get_models("http://example.com", 80)
        assert "http://example.com/v1/models" in get.call_args.args[0]

    def test_get_models_non_200(self, mock_llm):
        from lib.osaurus_models import get_models
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("lib.osaurus_models.requests.get", return_value=mock_resp):
            assert get_models() == []

    def test_get_models_timeout(self, mock_llm):
        from lib.osaurus_models import get_models
        with patch("lib.osaurus_models.requests.get",
                   side_effect=requests.exceptions.Timeout):
            assert get_models() == []

    def test_get_models_connection_error(self, mock_llm):
        from lib.osaurus_models import get_models
        with patch("lib.osaurus_models.requests.get",
                   side_effect=requests.exceptions.ConnectionError):
            assert get_models() == []

    def test_get_models_exception(self, mock_llm, capsys):
        from lib.osaurus_models import get_models
        with patch("lib.osaurus_models.requests.get", side_effect=Exception("boom")):
            assert get_models() == []
        captured = capsys.readouterr()
        assert "Warning" in captured.out


class TestServerCheck:
    def test_is_server_running_200(self, mock_llm):
        from lib.osaurus_models import is_server_running
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("lib.osaurus_models.requests.get", return_value=mock_resp):
            assert is_server_running() is True

    def test_is_server_running_404(self, mock_llm):
        from lib.osaurus_models import is_server_running
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("lib.osaurus_models.requests.get", return_value=mock_resp):
            assert is_server_running() is True

    def test_is_server_running_500(self, mock_llm):
        from lib.osaurus_models import is_server_running
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("lib.osaurus_models.requests.get", return_value=mock_resp):
            assert is_server_running() is False

    def test_is_server_running_timeout(self, mock_llm):
        from lib.osaurus_models import is_server_running
        with patch("lib.osaurus_models.requests.get",
                   side_effect=requests.exceptions.Timeout):
            assert is_server_running() is False

    def test_is_server_running_connection(self, mock_llm):
        from lib.osaurus_models import is_server_running
        with patch("lib.osaurus_models.requests.get",
                   side_effect=requests.exceptions.ConnectionError):
            assert is_server_running() is False

    def test_is_server_running_exception(self, mock_llm, capsys):
        from lib.osaurus_models import is_server_running
        with patch("lib.osaurus_models.requests.get", side_effect=Exception("boom")):
            assert is_server_running() is False
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    def test_is_server_running_http_prefix(self, mock_llm):
        from lib.osaurus_models import is_server_running
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("lib.osaurus_models.requests.get", return_value=mock_resp) as get:
            is_server_running("http://example.com")
        assert "http://example.com/v1/models" in get.call_args.args[0]


class TestCheckAvailability:
    def test_check_llm_availability_true(self, mock_llm):
        from lib.osaurus_models import check_llm_availability
        with patch("lib.osaurus_models.is_server_running", return_value=True):
            assert check_llm_availability() is True

    def test_check_llm_availability_false(self, mock_llm):
        from lib.osaurus_models import check_llm_availability
        with patch("lib.osaurus_models.is_server_running", return_value=False):
            assert check_llm_availability() is False


class TestGetBestModel:
    def test_get_best_model_with_task(self, mock_llm):
        from lib.osaurus_models import get_best_model
        with patch.dict(os.environ, {"OLLAMA_MODEL": "env-model"}, clear=False):
            with patch("lib.config.get_best_model", return_value="cfg-model"):
                result = get_best_model("think")
        # env_var takes precedence
        assert result == "env-model"

    def test_get_best_model_with_task_no_env(self, mock_llm):
        from lib.osaurus_models import get_best_model
        with patch.dict(os.environ, {}, clear=True):
            with patch("lib.config.get_best_model", return_value="cfg-model") as cfg:
                result = get_best_model("think")
        assert result == "cfg-model"
        cfg.assert_called_once_with("think")

    def test_get_best_model_no_task(self, mock_llm):
        from lib.osaurus_models import get_best_model
        with patch.dict(os.environ, {}, clear=True):
            assert get_best_model() == "foundation"

    def test_get_best_model_no_task_with_env(self, mock_llm):
        from lib.osaurus_models import get_best_model
        with patch.dict(os.environ, {"OLLAMA_MODEL": "env"}, clear=False):
            assert get_best_model() == "env"


class TestSelectBestVlm:
    def test_select_vlm_found(self, mock_llm):
        from lib.osaurus_models import select_best_vlm_model
        assert select_best_vlm_model(["llama-7b", "qwen-vl-7b"]) == "qwen-vl-7b"

    def test_select_vlm_vision_keyword(self, mock_llm):
        from lib.osaurus_models import select_best_vlm_model
        assert select_best_vlm_model(["foo", "vision-model"]) == "vision-model"

    def test_select_vlm_llamavl(self, mock_llm):
        from lib.osaurus_models import select_best_vlm_model
        assert select_best_vlm_model(["text-model", "llamavl-7b"]) == "llamavl-7b"

    def test_select_vlm_none(self, mock_llm):
        from lib.osaurus_models import select_best_vlm_model
        assert select_best_vlm_model(["text-model"]) is None

    def test_select_vlm_empty(self, mock_llm):
        from lib.osaurus_models import select_best_vlm_model
        assert select_best_vlm_model([]) is None


class TestSelectBestModel:
    def test_select_empty(self, mock_llm):
        from lib.osaurus_models import select_best_model
        assert select_best_model([]) is None

    def test_select_no_preferred(self, mock_llm):
        from lib.osaurus_models import select_best_model
        # Default preferred: ["foundation", "qwen", "gemma"]
        result = select_best_model(["foo", "qwen-7b", "gemma-2b"])
        assert result == "qwen-7b"

    def test_select_with_preferred(self, mock_llm):
        from lib.osaurus_models import select_best_model
        result = select_best_model(["foo", "qwen-7b", "gemma-2b"], preferred=["gemma"])
        assert result == "gemma-2b"

    def test_select_fallback(self, mock_llm):
        from lib.osaurus_models import select_best_model
        result = select_best_model(["foo", "bar"])
        # No match in preferred defaults, returns first
        assert result == "foo"

    def test_select_substring(self, mock_llm):
        from lib.osaurus_models import select_best_model
        result = select_best_model(["text-model", "my-foundation-model"])
        # "foundation" is in "my-foundation-model"
        assert result == "my-foundation-model"


class TestModuleConstants:
    def test_default_host(self, mock_llm):
        from lib.osaurus_models import DEFAULT_HOST, DEFAULT_PORT
        assert DEFAULT_HOST == "localhost"
        assert DEFAULT_PORT == 1337

    def test_get_available_models_alias(self, mock_llm):
        from lib.osaurus_models import get_available_models, get_models
        assert get_available_models is get_models
