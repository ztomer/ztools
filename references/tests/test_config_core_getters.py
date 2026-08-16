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
        import lib.config_core as cc
        import lib.config_getters as cg

        cc._model_configs_cache = {"x": 1}
        cg.clear_model_config_cache()
        assert cc._model_configs_cache == {}


class TestGetModelConfig:
    def test_get_model_config_with_cache_version_specific(self, tmp_path):
        import lib.config_core as cc
        import lib.config_getters as cg

        cc._model_configs_cache = {
            "qwen": {"name": "qwen", "models": {"qwen2.5-7b": {"version": "2.5"}}}
        }
        result = cg.get_model_config("qwen2.5-7b")
        # version is model.replace("qwen-", "") = "qwen2.5-7b" (not "2.5")
        # because the function uses naive string replace
        assert "version" in result
        assert result.get("name") == "qwen"

    def test_get_model_config_with_cache_family_only(self):
        import lib.config_core as cc
        import lib.config_getters as cg

        cc._model_configs_cache = {"qwen": {"name": "qwen", "timeout": 300}}
        result = cg.get_model_config("qwen-default")
        assert result == {"name": "qwen", "timeout": 300}

    def test_get_model_config_with_cache_version_not_found(self):
        import lib.config_core as cc
        import lib.config_getters as cg

        cc._model_configs_cache = {"qwen": {"name": "qwen", "models": {"qwen2.5-7b": {}}}}
        result = cg.get_model_config("qwen3.0-7b")
        # Falls back to family config (no version)
        assert result == {"name": "qwen", "models": {"qwen2.5-7b": {}}}

    def test_get_model_config_with_version_toml(self, tmp_path, monkeypatch):
        import lib.config_getters as cg

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        version_toml = models_dir / "qwen_versions.toml"
        version_toml.write_text('name = "qwen"\n[models."qwen2.5-7b"]\nextra = 1\n')
        monkeypatch.setenv("ZTOOLS_CONF", str(tmp_path))
        result = cg.get_model_config("qwen2.5-7b")
        assert "extra" in result
        assert result.get("name") == "qwen"

    def test_get_model_config_with_version_toml_no_match(self, tmp_path, monkeypatch):
        import lib.config_getters as cg

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        version_toml = models_dir / "qwen_versions.toml"
        version_toml.write_text('name = "qwen"\n[models]\n')
        monkeypatch.setenv("ZTOOLS_CONF", str(tmp_path))
        result = cg.get_model_config("qwen-other")
        # Returns the loaded toml as-is
        assert result["name"] == "qwen"

    def test_get_model_config_with_family_toml(self, tmp_path, monkeypatch):
        import lib.config_getters as cg

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        family_toml = models_dir / "qwen.toml"
        family_toml.write_text('name = "qwen"\ntimeout = 500\n')
        monkeypatch.setenv("ZTOOLS_CONF", str(tmp_path))
        result = cg.get_model_config("qwen-default")
        assert result == {"name": "qwen", "timeout": 500}

    def test_get_model_config_fallback(self, tmp_path, monkeypatch):
        import lib.config_getters as cg

        (tmp_path / "models").mkdir()
        # No toml files - should use built-in fallback
        monkeypatch.setenv("ZTOOLS_CONF", str(tmp_path))
        result = cg.get_model_config("unknown-7b")
        assert "prompts" in result
        assert result["name"] == "default"

    def test_get_model_config_default_fallback(self, tmp_path, monkeypatch):
        """When family is not in cache and no config file exists for it."""
        import lib.config_core as cc
        import lib.config_getters as cg

        cc._model_configs_cache = {"other": {"x": 1}}
        # An empty conf dir: neither <family>_versions.toml nor <family>.toml exists.
        monkeypatch.setenv("ZTOOLS_CONF", str(tmp_path))
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
        import lib.config_core as cc
        import lib.config_getters as cg

        cc._config = {"filename_models": ["gpt-4", "claude"]}
        cc._config_loaded = True
        assert cg.get_filename_models() == ["gpt-4", "claude"]

    def test_fallback(self):
        import lib.config_core as cc
        import lib.config_getters as cg

        cc._config = {}
        cc._config_loaded = True
        assert cg.get_filename_models() == ["foundation"]


class TestGetFilenamePrompt:
    def test_returns_prompt(self):
        import lib.config_core as cc
        import lib.config_getters as cg

        cc._config = {"prompts": {"filename": "Name it: {text}"}}
        cc._config_loaded = True
        assert cg.get_filename_prompt() == "Name it: {text}"

    def test_returns_default(self):
        import lib.config_core as cc
        import lib.config_getters as cg

        cc._config = {}
        cc._config_loaded = True
        assert "{text}" in cg.get_filename_prompt()


class TestFallbackPromptSchemas:
    """Every embedded "Schema:" in a prompt must be parseable JSON.

    The fallback weekend_transient prompt shipped `"day": "str"]}` — a missing
    brace — so any model family without a conf/models/*.toml was told to use an
    EXACT schema that is not valid JSON.
    """

    def _schema_segments(self, prompt: str):
        import re

        for match in re.finditer(r"[Ss]chema:?\s*(\{.*?\}\]?\})", prompt, re.DOTALL):
            yield match.group(1)

    def test_fallback_prompt_schemas_parse(self):
        import json

        import lib.config_getters as cg

        prompts = cg.get_model_config("totally-unknown-family")["prompts"]
        checked = 0
        for name, prompt in prompts.items():
            for segment in self._schema_segments(prompt):
                json.loads(segment)  # raises if the schema is malformed
                checked += 1
        assert checked >= 2, f"expected embedded schemas to check, saw {checked}"


class TestTheGettersReadThroughTheModule:
    """Rebinding `lib.config_core._config` must not desynchronise the getters.

    THE CLASS: `from x import <mutable state>` binds the OBJECT, once, at import.
    `_auto_load` mutates whatever `config_core._config` currently names, so an
    importer that holds its own alias agrees only until something rebinds the source
    -- after which it reads a dict that will never be updated again. Nothing raises.
    The two halves simply return different answers about the same config.

    It shipped exactly that way: the TUI's config audit reported no drift (it imports
    `_config` inside the function body, so it saw the truth) while the model dropdowns
    on the same screen silently substituted two slots (they went through
    `get_best_models`, which saw an empty dict). Three separate workarounds for this
    had accumulated in the test suite -- a manual re-alias in a fixture, a comment
    explaining which module to patch, and a cache reached for through the importer --
    before anyone fixed the binding itself.

    These pin the invariant rather than any one symptom: after a rebind, every reader
    of the config must agree.
    """

    def _rebind(self, mapping):
        import lib.config_core as cc

        cc._config = mapping
        cc._config_loaded = True

    def test_a_rebind_is_visible_to_the_getters(self):
        import lib.config_getters as cg

        self._rebind({"best_models": {"json": "rebound-model"}})
        assert cg.get_best_models() == {"json": "rebound-model"}

    def test_get_config_and_get_best_models_cannot_disagree(self):
        """The production symptom, stated directly."""
        import lib.config_core as cc
        import lib.config_getters as cg

        self._rebind({"best_models": {"json": "m"}, "default_model": "d"})
        assert cc.get_config()["best_models"] == cg.get_best_models()

    def test_every_config_getter_sees_the_same_rebind(self):
        """One getter reading through would not help if the others still aliased."""
        import lib.config_getters as cg

        self._rebind(
            {
                "timeouts": {"json": 7},
                "max_tokens": {"json": 8},
                "best_models": {"json": "m"},
                "filename_models": ["fm"],
                "prompts": {"filename": "p {text}"},
                "model_fallback_chain": ["c"],
            }
        )
        assert cg.get_timeouts() == {"json": 7}
        assert cg.get_max_tokens() == {"json": 8}
        assert cg.get_best_models() == {"json": "m"}
        assert cg.get_filename_models() == ["fm"]
        assert cg.get_model_fallback_chain() == ["c"]

    def test_a_second_rebind_is_also_visible(self):
        """A single rebind could be survived by luck of ordering; two cannot."""
        import lib.config_getters as cg

        self._rebind({"best_models": {"json": "first"}})
        assert cg.get_best_models() == {"json": "first"}
        self._rebind({"best_models": {"json": "second"}})
        assert cg.get_best_models() == {"json": "second"}

    def test_a_rebind_of_the_model_config_cache_is_visible(self):
        import lib.config_core as cc
        import lib.config_getters as cg

        cc._model_configs_cache = {"fam": {"name": "fam"}}
        assert cg._model_caches() == {"fam": {"name": "fam"}}

    def test_the_getters_hold_no_module_level_alias(self):
        """Structural, so a future edit cannot quietly reintroduce the binding. The
        three workarounds this replaced all existed because the alias was reachable."""
        import lib.config_getters as cg

        assert not hasattr(cg, "_config")
        assert not hasattr(cg, "_model_configs_cache")
