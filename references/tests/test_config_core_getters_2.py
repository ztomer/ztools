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
