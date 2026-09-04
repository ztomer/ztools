"""Tests for lib.config_tasks - task builder from model config."""

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_llm():
    from lib.testing import MockLLM

    mock = MockLLM()
    mock.patch_all()
    yield mock
    mock.unpatch()


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset the module-level cache between tests."""
    import lib.config_tasks as ct

    ct._eval_inputs_cache = {}
    yield
    ct._eval_inputs_cache = {}


class TestLoadEvalInputs:
    def test_load_eval_inputs_via_fixture(self, tmp_path, monkeypatch, mock_llm):
        """A real eval_inputs.toml in an overridden conf dir is read verbatim."""
        import lib.config_tasks as ct

        toml_content = '[test_inputs]\ntask_a = "input1"\ntask_b = "input2"\n'
        (tmp_path / "eval_inputs.toml").write_text(toml_content)
        monkeypatch.setenv("ZTOOLS_CONF", str(tmp_path))
        inputs = ct._load_eval_inputs()
        assert inputs == {"task_a": "input1", "task_b": "input2"}

    def test_load_eval_inputs_caches(self, tmp_path, monkeypatch, mock_llm):
        import lib.config_tasks as ct

        (tmp_path / "eval_inputs.toml").write_text('[test_inputs]\nx = "y"\n')
        monkeypatch.setenv("ZTOOLS_CONF", str(tmp_path))
        first = ct._load_eval_inputs()
        # Second call must not touch the filesystem again.
        with patch("lib.config_tasks.conf_path") as mock_conf_path:
            second = ct._load_eval_inputs()
        mock_conf_path.assert_not_called()
        assert first == second == {"x": "y"}

    def test_load_eval_inputs_missing_file(self, tmp_path, monkeypatch, mock_llm):
        import lib.config_tasks as ct

        monkeypatch.setenv("ZTOOLS_CONF", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="eval_inputs.toml"):
            ct._load_eval_inputs()

    def test_load_eval_inputs_empty(self, tmp_path, monkeypatch, mock_llm):
        import lib.config_tasks as ct

        (tmp_path / "eval_inputs.toml").write_text("[test_inputs]\n")
        monkeypatch.setenv("ZTOOLS_CONF", str(tmp_path))
        with pytest.raises(ValueError, match="Empty test_inputs"):
            ct._load_eval_inputs()

    def test_load_eval_inputs_null_yaml(self, tmp_path, monkeypatch, mock_llm):
        import lib.config_tasks as ct

        (tmp_path / "eval_inputs.toml").write_text("")
        monkeypatch.setenv("ZTOOLS_CONF", str(tmp_path))
        with pytest.raises(ValueError, match="Empty test_inputs"):
            ct._load_eval_inputs()


class TestGetEvalInput:
    def test_get_eval_input_found(self, mock_llm):
        import lib.config_tasks as ct

        ct._eval_inputs_cache = {"task_a": "input_data"}
        assert ct.get_eval_input("task_a") == "input_data"

    def test_get_eval_input_missing(self, mock_llm):
        import lib.config_tasks as ct

        ct._eval_inputs_cache = {"task_a": "x"}
        with pytest.raises(KeyError):
            ct.get_eval_input("unknown_task")


class TestSafeFormatPrompt:
    def test_format_with_brace_placeholder(self, mock_llm):
        from lib.config_tasks import _safe_format_prompt

        result = _safe_format_prompt("Hello {}", "world")
        assert result == "Hello world"

    def test_format_with_brace_placeholder_exception(self, mock_llm):
        from lib.config_tasks import _safe_format_prompt

        # A template with extra braces that breaks .format() but works via replace
        # Use a template that causes .format() to fail (it raises ValueError for invalid syntax)
        result = _safe_format_prompt("Hello {}", "world")
        assert "world" in result

    def test_mixed_positional_template_raises_instead_of_half_rendering(self, mock_llm):
        """Previously this asserted the swallow: `{0}` survived into the prompt.

        A template mixing `{0}` and `{}` cannot be rendered coherently, so it is a
        loud failure now rather than a prompt shipped with `{0}` still in it.
        """
        from lib.config_tasks import _safe_format_prompt
        from lib.prompt_render import PromptRenderError

        with pytest.raises(PromptRenderError, match=r"unrendered placeholder.*'0'"):
            _safe_format_prompt("Hello {0} {} world", "test")

    def test_format_with_location(self, mock_llm):
        from lib.config_tasks import _safe_format_prompt

        test_input = json.dumps([{"location": "Toronto", "target_ages": "5-12"}])
        result = _safe_format_prompt("Events in {location} for {age_range}", test_input)
        assert "Toronto" in result
        assert "5-12" in result

    def test_format_with_only_location(self, mock_llm):
        from lib.config_tasks import _safe_format_prompt

        test_input = json.dumps([{"location": "NYC", "target_ages": ""}])
        result = _safe_format_prompt("{location}", test_input)
        assert result == "NYC"

    def test_format_with_only_target_ages(self, mock_llm):
        from lib.config_tasks import _safe_format_prompt

        test_input = json.dumps([{"location": "", "target_ages": "3-5"}])
        result = _safe_format_prompt("{age_range}", test_input)
        assert result == "3-5"

    def test_format_no_braces(self, mock_llm):
        from lib.config_tasks import _safe_format_prompt

        result = _safe_format_prompt("plain text prompt", "irrelevant input")
        assert result == "plain text prompt"

    # The three cases below previously asserted "no input -> no replacement",
    # i.e. they pinned shipping a literal `{age_range}` to a model as CORRECT.
    # That is class C1/C12 encoded as a test. The contract is now the opposite:
    # a placeholder is always resolved, and an unresolvable one raises.

    def test_format_empty_test_input_still_resolves_the_placeholder(self, mock_llm):
        from lib.config_tasks import _safe_format_prompt

        result = _safe_format_prompt("with {age_range}", "")
        assert "{age_range}" not in result
        assert result == "with 6-13"

    def test_format_invalid_json_still_resolves_the_placeholder(self, mock_llm):
        from lib.config_tasks import _safe_format_prompt

        result = _safe_format_prompt("with {age_range}", "not json")
        assert "{age_range}" not in result

    def test_format_empty_json_array_still_resolves_the_placeholder(self, mock_llm):
        from lib.config_tasks import _safe_format_prompt

        result = _safe_format_prompt("with {age_range}", "[]")
        assert "{age_range}" not in result

    def test_unknown_placeholder_raises_instead_of_reaching_a_model(self, mock_llm):
        from lib.config_tasks import _safe_format_prompt
        from lib.prompt_render import PromptRenderError

        with pytest.raises(PromptRenderError, match="unrendered placeholder"):
            _safe_format_prompt("needs {some_unknown_field}", "[]")

    def test_json_schema_braces_survive_rendering(self, mock_llm):
        """The bug that started C1: a literal JSON brace must not be a format field."""
        from lib.config_tasks import _safe_format_prompt

        template = '{"transient_events": [{"name": "str"}]} for {age_range}'
        result = _safe_format_prompt(template, "[]")
        assert '{"transient_events": [{"name": "str"}]}' in result
        assert "{age_range}" not in result


class TestBuildTasksFromModel:
    def test_build_tasks_no_prompts(self, mock_llm):
        from lib.config_tasks import build_tasks_from_model

        with patch("lib.config_tasks.get_model_prompts_all", return_value={}):
            assert build_tasks_from_model("model-x") == {}

    def test_build_tasks_all_tasks(self, mock_llm, tmp_path):
        from lib.config_core import Task
        from lib.config_tasks import build_tasks_from_model

        prompts = {
            Task.WEEKEND_FIXED.value: "List events {}",
            Task.WEEKEND_TRANSIENT.value: "Find events {}",
            Task.FILENAME.value: "Name it {}",
            Task.SUMMARIZE.value: "Summarize {}",
            Task.FILE_SUMMARY.value: "File summary {}",
        }
        with (
            patch("lib.config_tasks.get_model_prompts_all", return_value=prompts),
            patch("lib.config_tasks.get_eval_input", side_effect=lambda x: f"input-for-{x}"),
        ):
            tasks = build_tasks_from_model("model-x")
        assert "detailed_json" in tasks
        assert "json" in tasks
        assert "filename" in tasks
        assert "summarize" in tasks
        assert "file_summary" in tasks
        assert tasks["filename"]["parse_json"] is False
        assert tasks["summarize"]["parse_json"] is False

    def test_build_tasks_with_validate_file_summary(self, mock_llm, tmp_path):
        from lib.config_core import Task
        from lib.config_tasks import build_tasks_from_model

        prompts = {Task.FILE_SUMMARY.value: "Summarize {}"}
        with (
            patch("lib.config_tasks.get_model_prompts_all", return_value=prompts),
            patch("lib.config_tasks.get_eval_input", return_value="x"),
            patch.dict(
                "sys.modules", {"eval.validate": MagicMock(validate_file_summary=lambda d, s="": d)}
            ),
        ):
            tasks = build_tasks_from_model("model-x")
        assert "file_summary" in tasks

    def test_build_tasks_file_summary_import_fails(self, mock_llm):
        from lib.config_core import Task
        from lib.config_tasks import build_tasks_from_model

        prompts = {Task.FILE_SUMMARY.value: "Summarize {}"}
        import builtins

        real_import = builtins.__import__

        def my_import(name, *a, **kw):
            if name == "eval.validate":
                raise ImportError("test")
            return real_import(name, *a, **kw)

        with (
            patch("lib.config_tasks.get_model_prompts_all", return_value=prompts),
            patch("lib.config_tasks.get_eval_input", return_value="x"),
            patch.object(builtins, "__import__", side_effect=my_import),
        ):
            with pytest.raises(ImportError):
                build_tasks_from_model("model-x")

    def test_build_tasks_only_filename(self, mock_llm):
        from lib.config_core import Task
        from lib.config_tasks import build_tasks_from_model

        prompts = {Task.FILENAME.value: "Name it {}"}
        with (
            patch("lib.config_tasks.get_model_prompts_all", return_value=prompts),
            patch("lib.config_tasks.get_eval_input", return_value="x"),
        ):
            tasks = build_tasks_from_model("model-x")
        assert "filename" in tasks
        assert "json" not in tasks


class TestPerModelTransientTaskCarriesItsSource:
    """`ev --config-tasks` must grade the shipped prompts against real source.

    The transient templates use named placeholders only, so the eval input never
    reached the model: the prompt ordered "Copy every value from the source
    text. NEVER invent one" with no source text, and with no "source" key
    validate_detailed_json skipped both the grounding score and the
    MAX_SCORE_NO_SOURCE cap — pure hallucination could score 100.
    """

    def test_source_is_delivered_and_recorded(self, mock_llm):
        from lib.config_tasks import build_tasks_from_model

        tasks = build_tasks_from_model("qwen3.6-35b-a3b-mxfp8-mtp")
        json_task = tasks["json"]
        assert json_task.get("source"), "no source recorded for grounding/cap"
        combined = " ".join(m["content"] for m in json_task["messages"])
        assert json_task["source"][:40] in combined

    def test_templates_that_embed_the_source_are_not_duplicated(self, mock_llm):
        from lib.config_tasks import _embeds_source

        assert _embeds_source("Events:\n{}") is True
        assert _embeds_source("Events:\n{text}") is True
        assert _embeds_source("Plan for {location} ages {age_range}") is False
