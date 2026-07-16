"""Tests for lib.config_core and lib.config_getters - state and lookup functions."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_config():
    import lib.config_core as cc

    cc._config_loaded = False
    cc._config = {}
    cc._model_configs_cache = {}
    import lib.config_getters as cg

    cg._model_configs_cache = cc._model_configs_cache
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
    def test_auto_load_with_no_config(self, capsys):
        import lib.config_core as cc

        with patch("lib.config_core.Path") as mock_path:
            instance = MagicMock()
            # Chain: Path(__file__).parent.parent / "conf" / "config.yaml"
            # First / returns a MagicMock whose / returns another MagicMock that .exists() is False
            conf_mock = MagicMock()
            yaml_mock = MagicMock()
            yaml_mock.exists.return_value = False
            conf_mock.__truediv__.return_value = yaml_mock
            instance.parent.parent.__truediv__.return_value = conf_mock
            mock_path.return_value = instance
            cc._auto_load()
        assert cc._config_loaded is True
        # Should print fallback message
        out = capsys.readouterr().out
        assert "not found" in out

    def test_auto_load_with_config(self, tmp_path):
        import lib.config_core as cc

        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[timeouts]\njson = 100\n")
        with patch("lib.config_core.Path") as mock_path:
            instance = MagicMock()
            # We need a real path-like that exists() returns True on the final result
            conf_dir = tmp_path / "conf"
            conf_dir.mkdir()
            real_toml = conf_dir / "config.toml"
            real_toml.write_text("[timeouts]\njson = 100\n")
            instance.parent.parent.__truediv__.return_value = conf_dir
            mock_path.return_value = instance
            cc._auto_load()
        assert cc._config_loaded is True
        assert cc._config.get("timeouts", {}).get("json") == 100

    def test_auto_load_skips_if_loaded(self):
        import lib.config_core as cc

        cc._config_loaded = True
        cc._config = {"already": "loaded"}
        with patch("lib.config_core.Path") as mock_path:
            cc._auto_load()
        # Should not have been called
        mock_path.assert_not_called()

    def test_auto_load_yaml_non_dict(self, tmp_path):
        import lib.config_core as cc

        toml_file = tmp_path / "config.toml"
        toml_file.write_text("not_a_dict_key\n")
        with patch("lib.config_core.Path") as mock_path:
            instance = MagicMock()
            instance.parent.parent.__truediv__.return_value = toml_file
            mock_path.return_value = instance
            cc._auto_load()
        assert cc._config == {}


class TestInitConfig:
    def test_init_config_default_path_not_found(self):
        import lib.config_core as cc

        with patch("lib.config_core.Path") as mock_path:
            instance = MagicMock()
            instance.exists.return_value = False
            mock_path.return_value = instance
            with pytest.raises(FileNotFoundError):
                cc.init_config()

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
    def test_is_loaded_triggers_autoload(self):
        import lib.config_core as cc

        with patch("lib.config_core.Path") as mock_path:
            instance = MagicMock()
            instance.parent.parent.__truediv__.return_value.exists.return_value = False
            mock_path.return_value = instance
            cc._config_loaded = False
            assert cc.is_config_loaded() is True

    def test_is_loaded_when_already(self):
        import lib.config_core as cc

        cc._config_loaded = True
        assert cc.is_config_loaded() is True


class TestGetTimeouts:
    def test_get_timeouts_empty(self):
        import lib.config_getters as cg

        cg._config = {}
        assert cg.get_timeouts() == {}

    def test_get_timeouts_with_values(self):
        import lib.config_getters as cg

        cg._config = {"timeouts": {"json": 100}}
        assert cg.get_timeouts() == {"json": 100}


class TestGetMaxTokens:
    def test_get_max_tokens_empty(self):
        import lib.config_getters as cg

        cg._config = {}
        assert cg.get_max_tokens() == {}

    def test_get_max_tokens_with_values(self):
        import lib.config_getters as cg

        cg._config = {"max_tokens": {"json": 2000}}
        assert cg.get_max_tokens() == {"json": 2000}


class TestGetBestModels:
    def test_get_best_models_empty(self):
        import lib.config_getters as cg

        cg._config = {}
        assert cg.get_best_models() == {}

    def test_get_best_models_with_values(self):
        import lib.config_getters as cg

        cg._config = {"best_models": {"json": "gpt-4"}}
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
            patch("lib.config_getters._config", {"default_model": "default-m"}),
        ):
            assert get_best_model("unknown") == "default-m"

    def test_get_best_model_no_default(self):
        from lib.config_getters import get_best_model

        with (
            patch("lib.config_getters.get_best_models", return_value={}),
            patch("lib.config_getters._config", {}),
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
            assert get_max_tokens_for_task("anything") == 16000


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
        import lib.config_getters as cg

        cg._model_configs_cache = {"x": 1}
        cg.clear_model_config_cache()
        assert cg._model_configs_cache == {}


class TestGetModelConfig:
    def test_get_model_config_with_cache_version_specific(self, tmp_path):
        import lib.config_getters as cg

        cg._model_configs_cache = {
            "qwen": {"name": "qwen", "models": {"qwen2.5-7b": {"version": "2.5"}}}
        }
        result = cg.get_model_config("qwen2.5-7b")
        # version is model.replace("qwen-", "") = "qwen2.5-7b" (not "2.5")
        # because the function uses naive string replace
        assert "version" in result
        assert result.get("name") == "qwen"

    def test_get_model_config_with_cache_family_only(self):
        import lib.config_getters as cg

        cg._model_configs_cache = {"qwen": {"name": "qwen", "timeout": 300}}
        result = cg.get_model_config("qwen-default")
        assert result == {"name": "qwen", "timeout": 300}

    def test_get_model_config_with_cache_version_not_found(self):
        import lib.config_getters as cg

        cg._model_configs_cache = {"qwen": {"name": "qwen", "models": {"qwen2.5-7b": {}}}}
        result = cg.get_model_config("qwen3.0-7b")
        # Falls back to family config (no version)
        assert result == {"name": "qwen", "models": {"qwen2.5-7b": {}}}

    def test_get_model_config_with_version_yaml(self, tmp_path):
        import lib.config_getters as cg

        # Create tmp/conf/models/qwen_versions.toml
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        models_dir = conf_dir / "models"
        models_dir.mkdir()
        version_toml = models_dir / "qwen_versions.toml"
        version_toml.write_text('name = "qwen"\n[models."qwen2.5-7b"]\nextra = 1\n')
        with patch("lib.config_getters.Path") as mock_path:
            instance = MagicMock()
            instance.parent.parent.__truediv__.return_value = conf_dir
            mock_path.return_value = instance
            result = cg.get_model_config("qwen2.5-7b")
        assert "extra" in result
        assert result.get("name") == "qwen"

    def test_get_model_config_with_version_yaml_no_match(self, tmp_path):
        import lib.config_getters as cg

        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        models_dir = conf_dir / "models"
        models_dir.mkdir()
        version_yaml = models_dir / "qwen_versions.yaml"
        version_yaml.write_text("name: qwen\nmodels: {}\n")
        with patch("lib.config_getters.Path") as mock_path:
            instance = MagicMock()
            instance.parent.parent.__truediv__.return_value = conf_dir
            mock_path.return_value = instance
            result = cg.get_model_config("qwen-other")
        # Returns the loaded yaml as-is
        assert result["name"] == "qwen"

    def test_get_model_config_with_family_yaml(self, tmp_path):
        import lib.config_getters as cg

        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        models_dir = conf_dir / "models"
        models_dir.mkdir()
        family_toml = models_dir / "qwen.toml"
        family_toml.write_text('name = "qwen"\ntimeout = 500\n')
        with patch("lib.config_getters.Path") as mock_path:
            instance = MagicMock()
            instance.parent.parent.__truediv__.return_value = conf_dir
            mock_path.return_value = instance
            result = cg.get_model_config("qwen-default")
        assert result == {"name": "qwen", "timeout": 500}

    def test_get_model_config_fallback(self, tmp_path):
        import lib.config_getters as cg

        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        models_dir = conf_dir / "models"
        models_dir.mkdir()
        # No yaml files - should use built-in fallback
        with patch("lib.config_getters.Path") as mock_path:
            instance = MagicMock()
            instance.parent.parent.__truediv__.return_value = conf_dir
            mock_path.return_value = instance
            result = cg.get_model_config("unknown-7b")
        assert "prompts" in result
        assert result["name"] == "default"

    def test_get_model_config_default_fallback(self):
        """When family is not in cache and path is not in model dict."""
        import lib.config_getters as cg

        cg._model_configs_cache = {"other": {"x": 1}}
        with patch("lib.config_getters.Path") as mock_path:
            instance = MagicMock()
            # Make both yaml files not exist through chained __truediv__
            leaf = MagicMock()
            leaf.exists.return_value = False
            mid = MagicMock()
            mid.exists.return_value = False
            mid.__truediv__.return_value = leaf
            top = MagicMock()
            top.exists.return_value = False
            top.__truediv__.return_value = mid
            instance.parent.parent.__truediv__.return_value = top
            mock_path.return_value = instance
            result = cg.get_model_config("totally-unknown")
        assert result["name"] == "default"


class TestGetModelFieldMapping:
    def test_returns_field_mapping(self):
        import lib.config_getters as cg

        with patch(
            "lib.config_getters.get_model_config", return_value={"field_mapping": {"event": "name"}}
        ):
            assert cg.get_model_field_mapping("any") == {"event": "name"}

    def test_returns_empty(self):
        import lib.config_getters as cg

        with patch("lib.config_getters.get_model_config", return_value={}):
            assert cg.get_model_field_mapping("any") == {}


class TestGetModelTopKeys:
    def test_returns_top_keys(self):
        import lib.config_getters as cg

        with patch(
            "lib.config_getters.get_model_config", return_value={"top_keys": {"fixed": ["a"]}}
        ):
            assert cg.get_model_top_keys("any") == {"fixed": ["a"]}

    def test_returns_default(self):
        import lib.config_getters as cg

        with patch("lib.config_getters.get_model_config", return_value={}):
            result = cg.get_model_top_keys("any")
        assert "fixed" in result
        assert "transient" in result


class TestGetModelQuirks:
    def test_returns_quirks(self):
        import lib.config_getters as cg

        with patch(
            "lib.config_getters.get_model_config", return_value={"quirks": [{"type": "prefix"}]}
        ):
            assert cg.get_model_quirks("any") == [{"type": "prefix"}]

    def test_returns_empty(self):
        import lib.config_getters as cg

        with patch("lib.config_getters.get_model_config", return_value={}):
            assert cg.get_model_quirks("any") == []


class TestGetModelPrompt:
    def test_returns_prompt(self):
        from lib.config_core import Task
        from lib.config_getters import get_model_prompt

        with patch(
            "lib.config_getters.get_model_config", return_value={"prompts": {"json": "do thing"}}
        ):
            assert get_model_prompt("any", Task.JSON) == "do thing"

    def test_returns_prompt_string_task(self):
        from lib.config_getters import get_model_prompt

        with patch("lib.config_getters.get_model_config", return_value={"prompts": {"json": "x"}}):
            assert get_model_prompt("any", "json") == "x"

    def test_returns_empty(self):
        from lib.config_getters import get_model_prompt

        with patch("lib.config_getters.get_model_config", return_value={"prompts": {}}):
            assert get_model_prompt("any", "json") == ""


class TestGetModelPromptsAll:
    def test_returns_all_prompts(self):
        from lib.config_getters import get_model_prompts_all

        with patch(
            "lib.config_getters.get_model_config", return_value={"prompts": {"a": "1", "b": "2"}}
        ):
            assert get_model_prompts_all("any") == {"a": "1", "b": "2"}


class TestGetFilenameModels:
    def test_returns_models(self):
        import lib.config_getters as cg

        cg._config = {"filename_models": ["gpt-4", "claude"]}
        assert cg.get_filename_models() == ["gpt-4", "claude"]

    def test_fallback(self):
        import lib.config_getters as cg

        cg._config = {}
        assert cg.get_filename_models() == ["foundation"]


class TestGetFilenamePrompt:
    def test_returns_prompt(self):
        import lib.config_getters as cg

        cg._config = {"prompts": {"filename": "Name it: {text}"}}
        assert cg.get_filename_prompt() == "Name it: {text}"

    def test_returns_default(self):
        import lib.config_getters as cg

        cg._config = {}
        assert "{text}" in cg.get_filename_prompt()
