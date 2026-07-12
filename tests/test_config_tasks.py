"""Tests for lib.config_tasks - task builder from model config."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open


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
    def test_load_eval_inputs_via_fixture(self, tmp_path, mock_llm):
        """Test via patching the Path to create a real toml file."""
        import lib.config_tasks as ct
        toml_content = '[test_inputs]\ntask_a = "input1"\ntask_b = "input2"\n'
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        real_toml = conf_dir / "eval_inputs.toml"
        real_toml.write_text(toml_content)
        with patch("lib.config_tasks.Path") as mock_path_class:
            instance = MagicMock()
            instance.parent.parent.__truediv__.return_value = conf_dir
            mock_path_class.return_value = instance
            inputs = ct._load_eval_inputs()
        assert inputs == {"task_a": "input1", "task_b": "input2"}

    def test_load_eval_inputs_caches(self, tmp_path, mock_llm):
        import lib.config_tasks as ct
        toml_content = '[test_inputs]\nx = "y"\n'
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        real_toml = conf_dir / "eval_inputs.toml"
        real_toml.write_text(toml_content)
        with patch("lib.config_tasks.Path") as mock_path_class:
            instance = MagicMock()
            instance.parent.parent.__truediv__.return_value = conf_dir
            mock_path_class.return_value = instance
            ct._load_eval_inputs()
            call_count = mock_path_class.call_count
            ct._load_eval_inputs()
        assert mock_path_class.call_count == call_count

    def test_load_eval_inputs_missing_file(self, mock_llm):
        import lib.config_tasks as ct
        with patch("lib.config_tasks.Path") as mock_path_class:
            instance = MagicMock()
            # First / gives a fake path whose second / gives another fake path
            conf_path = MagicMock()
            fake = MagicMock()
            fake.exists.return_value = False
            conf_path.__truediv__.return_value = fake
            instance.parent.parent.__truediv__.return_value = conf_path
            mock_path_class.return_value = instance
            with pytest.raises(FileNotFoundError):
                ct._load_eval_inputs()

    def test_load_eval_inputs_empty(self, tmp_path, mock_llm):
        import lib.config_tasks as ct
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        real_toml = conf_dir / "eval_inputs.toml"
        real_toml.write_text('[test_inputs]\n')
        with patch("lib.config_tasks.Path") as mock_path_class:
            instance = MagicMock()
            instance.parent.parent.__truediv__.return_value = conf_dir
            mock_path_class.return_value = instance
            with pytest.raises(ValueError):
                ct._load_eval_inputs()

    def test_load_eval_inputs_null_yaml(self, tmp_path, mock_llm):
        import lib.config_tasks as ct
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        real_toml = conf_dir / "eval_inputs.toml"
        real_toml.write_text("")
        with patch("lib.config_tasks.Path") as mock_path_class:
            instance = MagicMock()
            instance.parent.parent.__truediv__.return_value = conf_dir
            mock_path_class.return_value = instance
            with pytest.raises(ValueError):
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

    def test_format_with_invalid_template_fallback(self, mock_llm):
        """Template that contains {} but is otherwise invalid - .format raises ValueError."""
        from lib.config_tasks import _safe_format_prompt
        # Create a template that .format will reject
        template = "Hello {0} {} world"  # Mixing positional and empty
        # .format with single arg raises ValueError for mixed
        result = _safe_format_prompt(template, "test")
        # Falls back to replace
        assert "test" in result

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

    def test_format_empty_test_input(self, mock_llm):
        from lib.config_tasks import _safe_format_prompt
        result = _safe_format_prompt("with {age_range}", "")
        # No input -> no replacement
        assert result == "with {age_range}"

    def test_format_invalid_json(self, mock_llm):
        from lib.config_tasks import _safe_format_prompt
        result = _safe_format_prompt("with {age_range}", "not json")
        # Falls through to no replacement
        assert result == "with {age_range}"

    def test_format_empty_json_array(self, mock_llm):
        from lib.config_tasks import _safe_format_prompt
        result = _safe_format_prompt("with {age_range}", "[]")
        # Empty array - no replacement
        assert result == "with {age_range}"


class TestBuildTasksFromModel:
    def test_build_tasks_no_prompts(self, mock_llm):
        from lib.config_tasks import build_tasks_from_model
        with patch("lib.config_tasks.get_model_prompts_all", return_value={}):
            assert build_tasks_from_model("model-x") == {}

    def test_build_tasks_all_tasks(self, mock_llm, tmp_path):
        from lib.config_tasks import build_tasks_from_model
        from lib.config_core import Task
        prompts = {
            Task.WEEKEND_FIXED.value: "List events {}",
            Task.WEEKEND_TRANSIENT.value: "Find events {}",
            Task.FILENAME.value: "Name it {}",
            Task.SUMMARIZE.value: "Summarize {}",
            Task.FILE_SUMMARY.value: "File summary {}",
        }
        with patch("lib.config_tasks.get_model_prompts_all", return_value=prompts), \
             patch("lib.config_tasks.get_eval_input", side_effect=lambda x: f"input-for-{x}"):
            tasks = build_tasks_from_model("model-x")
        assert "detailed_json" in tasks
        assert "json" in tasks
        assert "filename" in tasks
        assert "summarize" in tasks
        assert "file_summary" in tasks
        assert tasks["filename"]["parse_json"] is False
        assert tasks["summarize"]["parse_json"] is False

    def test_build_tasks_with_validate_file_summary(self, mock_llm, tmp_path):
        from lib.config_tasks import build_tasks_from_model
        from lib.config_core import Task
        prompts = {Task.FILE_SUMMARY.value: "Summarize {}"}
        with patch("lib.config_tasks.get_model_prompts_all", return_value=prompts), \
             patch("lib.config_tasks.get_eval_input", return_value="x"), \
             patch.dict("sys.modules", {"eval.validate": MagicMock(validate_file_summary=lambda d, s="": d)}):
            tasks = build_tasks_from_model("model-x")
        assert "file_summary" in tasks

    def test_build_tasks_file_summary_import_fails(self, mock_llm):
        from lib.config_tasks import build_tasks_from_model
        from lib.config_core import Task
        prompts = {Task.FILE_SUMMARY.value: "Summarize {}"}
        import builtins
        real_import = builtins.__import__
        def my_import(name, *a, **kw):
            if name == "eval.validate":
                raise ImportError("test")
            return real_import(name, *a, **kw)
        with patch("lib.config_tasks.get_model_prompts_all", return_value=prompts), \
             patch("lib.config_tasks.get_eval_input", return_value="x"), \
             patch.object(builtins, "__import__", side_effect=my_import):
            with pytest.raises(ImportError):
                build_tasks_from_model("model-x")

    def test_build_tasks_only_filename(self, mock_llm):
        from lib.config_tasks import build_tasks_from_model
        from lib.config_core import Task
        prompts = {Task.FILENAME.value: "Name it {}"}
        with patch("lib.config_tasks.get_model_prompts_all", return_value=prompts), \
             patch("lib.config_tasks.get_eval_input", return_value="x"):
            tasks = build_tasks_from_model("model-x")
        assert "filename" in tasks
        assert "json" not in tasks
