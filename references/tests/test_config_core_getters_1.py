"""Tests for lib.config_core and lib.config_getters - state and lookup functions."""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def reset_config():
    import lib.config_core as cc

    cc._config_loaded = False
    cc._config = {}
    cc._model_configs_cache = {}
    yield
    cc._config_loaded = False
    cc._config = {}
    cc._model_configs_cache = {}


@pytest.fixture
def mock_llm():
    from lib.testing import MockLLM

    mock = MockLLM()
    mock.patch_all()
    yield mock
    mock.unpatch()


class TestTaskEnum:
    def test_task_values(self):
        from lib.config_core import Task

        assert Task.WEEKEND_FIXED.value == "weekend_fixed"
        assert Task.WEEKEND_TRANSIENT.value == "weekend_transient"
        assert Task.SUMMARIZE.value == "summarize"
        assert Task.FILENAME.value == "filename"
        assert Task.FILE_SUMMARY.value == "file_summary"
        assert Task.JSON.value == "json"
        assert Task.DETAILED_JSON.value == "detailed_json"

    def test_task_keys_alias(self):
        from lib.config_core import Task, TaskKeys

        assert TaskKeys is Task


class TestAutoLoad:
    def test_auto_load_with_no_config(self, capsys, tmp_path, monkeypatch):
        import lib.config_core as cc

        # A real, existing conf dir that simply has no config.toml in it.
        monkeypatch.setenv("ZTOOLS_CONF", str(tmp_path))
        cc._auto_load()
        assert cc._config_loaded is True
        assert cc._config == {}
        # Should print fallback message
        out = capsys.readouterr().out
        assert "not found" in out

    def test_auto_load_with_config(self, tmp_path, monkeypatch):
        import lib.config_core as cc

        (tmp_path / "config.toml").write_text("[timeouts]\njson = 100\n")
        monkeypatch.setenv("ZTOOLS_CONF", str(tmp_path))
        cc._auto_load()
        assert cc._config_loaded is True
        assert cc._config.get("timeouts", {}).get("json") == 100

    def test_auto_load_skips_if_loaded(self):
        import lib.config_core as cc

        cc._config_loaded = True
        cc._config = {"already": "loaded"}
        with patch("lib.config_core.conf_path") as mock_conf_path:
            cc._auto_load()
        # Should not have been called
        mock_conf_path.assert_not_called()
        assert cc._config == {"already": "loaded"}

    def test_auto_load_non_dict_config(self, tmp_path, monkeypatch):
        import lib.config_core as cc

        (tmp_path / "config.toml").write_text("[timeouts]\njson = 100\n")
        monkeypatch.setenv("ZTOOLS_CONF", str(tmp_path))
        with patch("lib.config_core.load_config", return_value=["not", "a", "dict"]):
            cc._auto_load()
        assert cc._config == {}
        assert cc._config_loaded is True


class TestInitConfig:
    def test_init_config_default_path_not_found(self, tmp_path, monkeypatch):
        import lib.config_core as cc

        # Real, existing conf dir with no config.toml: the crash `wk`/`ev` hit.
        monkeypatch.setenv("ZTOOLS_CONF", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="config.toml"):
            cc.init_config()

    def test_init_config_default_path_found(self):
        import lib.config_core as cc

        # No override: the shipped conf/ must be reachable in this layout.
        assert cc.init_config() is True
        assert cc._config

    def test_init_config_explicit_path(self, tmp_path):
        import lib.config_core as cc

        toml_file = tmp_path / "test_config.toml"
        toml_file.write_text("[timeouts]\njson = 200\n")
        cc.init_config(str(toml_file))
        assert cc._config.get("timeouts", {}).get("json") == 200

    def test_init_config_yaml_null(self, tmp_path):
        import lib.config_core as cc

        toml_file = tmp_path / "null.toml"
        toml_file.write_text("")
        cc.init_config(str(toml_file))
        assert cc._config == {}

    def test_init_config_yaml_not_dict(self, tmp_path):
        import lib.config_core as cc

        toml_file = tmp_path / "list.toml"
        toml_file.write_text("invalid toml content here")
        with pytest.raises(cc.ConfigurationError):
            cc.init_config(str(toml_file))

    def test_init_config_replaces_existing(self, tmp_path):
        import lib.config_core as cc

        cc._config = {"old": "data"}
        cc._config_loaded = True
        toml_file = tmp_path / "new.toml"
        toml_file.write_text('new = "data"\n')
        cc.init_config(str(toml_file))
        assert "old" not in cc._config
        assert cc._config["new"] == "data"


class TestResetConfig:
    def test_reset(self):
        import lib.config_core as cc

        cc._config = {"x": 1}
        cc._config_loaded = True
        cc._model_configs_cache = {"a": "b"}
        cc.reset_config()
        assert cc._config == {}
        assert cc._config_loaded is False
        assert cc._model_configs_cache == {}


class TestGetConfig:
    def test_get_config(self):
        import lib.config_core as cc

        cc._config = {"x": 1}
        cc._config_loaded = True
        result = cc.get_config()
        assert result == {"x": 1}
        # Should be a copy
        result["y"] = 2
        assert "y" not in cc._config


class TestIsConfigLoaded:
    def test_is_loaded_triggers_autoload(self, tmp_path, monkeypatch):
        import lib.config_core as cc

        monkeypatch.setenv("ZTOOLS_CONF", str(tmp_path))
        cc._config_loaded = False
        assert cc.is_config_loaded() is True
        assert cc._config == {}

    def test_is_loaded_when_already(self):
        import lib.config_core as cc

        cc._config_loaded = True
        assert cc.is_config_loaded() is True


class TestGetTimeouts:
    def test_get_timeouts_empty(self):
        import lib.config_core as cc
        import lib.config_getters as cg

        cc._config = {}
        cc._config_loaded = True
        assert cg.get_timeouts() == {}

    def test_get_timeouts_with_values(self):
        import lib.config_core as cc
        import lib.config_getters as cg

        cc._config = {"timeouts": {"json": 100}}
        cc._config_loaded = True
        assert cg.get_timeouts() == {"json": 100}


class TestGetMaxTokens:
    def test_get_max_tokens_empty(self):
        import lib.config_core as cc
        import lib.config_getters as cg

        cc._config = {}
        cc._config_loaded = True
        assert cg.get_max_tokens() == {}

    def test_get_max_tokens_with_values(self):
        import lib.config_core as cc
        import lib.config_getters as cg

        cc._config = {"max_tokens": {"json": 2000}}
        cc._config_loaded = True
        assert cg.get_max_tokens() == {"json": 2000}


class TestGetBestModels:
    def test_get_best_models_empty(self):
        import lib.config_core as cc
        import lib.config_getters as cg

        cc._config = {}
        cc._config_loaded = True
        assert cg.get_best_models() == {}

    def test_get_best_models_with_values(self):
        import lib.config_core as cc
        import lib.config_getters as cg

        cc._config = {"best_models": {"json": "gpt-4"}}
        cc._config_loaded = True
        assert cg.get_best_models() == {"json": "gpt-4"}


class TestGetBestModel:
    def test_get_best_model_with_task_enum(self):
        from lib.config_core import Task
        from lib.config_getters import get_best_model

        with patch("lib.config_getters.get_best_models", return_value={"json": "best-model"}):
            assert get_best_model(Task.JSON) == "best-model"

    def test_get_best_model_fallback_to_default(self):
        from lib.config_getters import get_best_model

        with (
            patch("lib.config_getters.get_best_models", return_value={}),
            patch("lib.config_core._config", {"default_model": "default-m"}),
            patch("lib.config_core._config_loaded", True),
        ):
            assert get_best_model("unknown") == "default-m"

    def test_get_best_model_no_default(self):
        from lib.config_getters import get_best_model

        with (
            patch("lib.config_getters.get_best_models", return_value={}),
            patch("lib.config_core._config", {}),
            patch("lib.config_core._config_loaded", True),
        ):
            assert get_best_model("unknown") == "foundation"

    def test_get_best_model_string_task(self):
        from lib.config_getters import get_best_model

        with patch("lib.config_getters.get_best_models", return_value={"json": "gpt-4"}):
            assert get_best_model("json") == "gpt-4"


class TestGetTimeout:
    def test_get_timeout_for_task(self):
        from lib.config_getters import get_timeout

        with patch("lib.config_getters.get_timeouts", return_value={"json": 120}):
            assert get_timeout("json") == 120

    def test_get_timeout_fallback(self):
        from lib.config_getters import get_timeout

        with patch("lib.config_getters.get_timeouts", return_value={}):
            assert get_timeout("anything") == 600


class TestGetMaxTokensForTask:
    def test_get_max_tokens_for_task(self):
        from lib.config_getters import get_max_tokens_for_task

        with patch("lib.config_getters.get_max_tokens", return_value={"json": 2000}):
            assert get_max_tokens_for_task("json") == 2000

    def test_get_max_tokens_fallback(self):
        from lib.config_getters import get_max_tokens_for_task

        with patch("lib.config_getters.get_max_tokens", return_value={}):
            from lib.llm.constants import DEFAULT_MAX_TOKENS

            assert get_max_tokens_for_task("anything") == DEFAULT_MAX_TOKENS


class TestGetModelFamily:
    def test_empty_model(self):
        from lib.config_getters import get_model_family

        assert get_model_family("") == "default"

    def test_qwopus(self):
        from lib.config_getters import get_model_family

        assert get_model_family("QwOpus-7B") == "qwopus"

    def test_qwen(self):
        from lib.config_getters import get_model_family

        assert get_model_family("qwen2.5-7b") == "qwen"

    def test_gemma(self):
        from lib.config_getters import get_model_family

        assert get_model_family("gemma-3-4b") == "gemma"

    def test_nemotron(self):
        from lib.config_getters import get_model_family

        assert get_model_family("nemotron-9b") == "nemotron"

    def test_laguna(self):
        from lib.config_getters import get_model_family

        assert get_model_family("laguna-7b") == "laguna"

    def test_foundation(self):
        from lib.config_getters import get_model_family

        assert get_model_family("foundation-1b") == "foundation"

    def test_default(self):
        from lib.config_getters import get_model_family

        assert get_model_family("unknown-7b") == "default"


class TestClearModelConfigCache:
    def test_clear(self):
        import lib.config_core as cc
        import lib.config_getters as cg

        cc._model_configs_cache = {"x": 1}
        cg.clear_model_config_cache()
        assert cc._model_configs_cache == {}
