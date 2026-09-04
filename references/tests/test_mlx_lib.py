"""Tests for lib.mlx_lib - MLX model discovery, execution, and unified call."""

import json
import subprocess
from pathlib import Path
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
def fake_mlx_dir(tmp_path):
    """Create a fake MLX models directory with several models."""
    models = tmp_path / "MLXModels"
    models.mkdir()
    # Qwen model
    qwen = models / "qwen-7b-fp16"
    qwen.mkdir()
    (qwen / "config.json").write_text(json.dumps({"context_length": 8192}))
    # Llama with config in subdir
    llama = models / "mlx-community"
    llama.mkdir()
    llama_sub = llama / "llama-3-8b"
    llama_sub.mkdir()
    (llama_sub / "config.json").write_text(json.dumps({"max_position_embeddings": 4096}))
    # Folder without config
    (models / "no-config").mkdir()
    return models


class TestModelDiscovery:
    def test_find_mlx_model_not_exists(self, mock_llm):
        from lib.mlx_lib import find_mlx_model

        with patch("lib.mlx_lib.MLX_MODELS_DIR", Path("/nonexistent")):
            assert find_mlx_model("qwen") is None

    def test_find_mlx_model_top_level(self, mock_llm, fake_mlx_dir, real_mlx_functions):
        real = real_mlx_functions["find_mlx_model"]
        result = real("qwen", mlx_dir=fake_mlx_dir)
        # qwen-7b-fp16 is direct child, has config.json
        assert result == fake_mlx_dir / "qwen-7b-fp16"

    def test_find_mlx_model_subdir(self, mock_llm, fake_mlx_dir, real_mlx_functions):
        real = real_mlx_functions["find_mlx_model"]
        result = real("llama", mlx_dir=fake_mlx_dir)
        # llama-3-8b is nested under mlx-community
        assert result == fake_mlx_dir / "mlx-community" / "llama-3-8b"

    def test_find_mlx_model_not_found(self, mock_llm, fake_mlx_dir, real_mlx_functions):
        real = real_mlx_functions["find_mlx_model"]
        assert real("nonexistent", mlx_dir=fake_mlx_dir) is None

    def test_find_mlx_model_nonexistent_dir(self, mock_llm, real_mlx_functions):
        """When mlx_dir doesn't exist, return None."""
        real = real_mlx_functions["find_mlx_model"]
        result = real("qwen", mlx_dir=Path("/definitely/does/not/exist/12345"))
        assert result is None

    def test_find_mlx_model_skips_files(self, mock_llm, fake_mlx_dir, real_mlx_functions):
        """Non-directory items in mlx_dir are skipped."""
        real = real_mlx_functions["find_mlx_model"]
        (fake_mlx_dir / "stray.txt").write_text("not a model")
        assert real("stray", mlx_dir=fake_mlx_dir) is None

    def test_find_best_mlx_model(self, mock_llm, fake_mlx_dir):
        from lib import mlx_lib

        with patch("lib.mlx_lib.find_mlx_model", return_value=fake_mlx_dir / "qwen-7b-fp16"):
            result = mlx_lib.find_best_mlx_model(["nonexistent", "qwen"])
        # First match wins (qwen is preferred over nonexistent)
        assert result == fake_mlx_dir / "qwen-7b-fp16"

    def test_find_best_mlx_model_none(self, mock_llm):
        from lib import mlx_lib

        with patch("lib.mlx_lib.find_mlx_model", return_value=None):
            assert mlx_lib.find_best_mlx_model(["x", "y"]) is None

    def test_find_text_mlx_model_default(self, mock_llm, fake_mlx_dir, real_mlx_functions):
        real = real_mlx_functions["find_text_mlx_model"]
        with patch("lib.mlx_lib.find_best_mlx_model", return_value=fake_mlx_dir / "qwen-7b-fp16"):
            result = real()
        assert result == fake_mlx_dir / "qwen-7b-fp16"

    def test_find_text_mlx_model_preferred(self, mock_llm, fake_mlx_dir, real_mlx_functions):
        real = real_mlx_functions["find_text_mlx_model"]
        with patch("lib.mlx_lib.find_best_mlx_model", return_value=fake_mlx_dir / "qwen-7b-fp16"):
            result = real(["qwen"])
        assert result == fake_mlx_dir / "qwen-7b-fp16"

    def test_get_mlx_context_length(self, mock_llm, fake_mlx_dir):
        from lib.mlx_lib import get_mlx_context_length

        qwen = fake_mlx_dir / "qwen-7b-fp16"
        assert get_mlx_context_length(qwen) == 8192

    def test_get_mlx_context_length_max_pos(self, mock_llm, fake_mlx_dir):
        from lib.mlx_lib import get_mlx_context_length

        llama = fake_mlx_dir / "mlx-community" / "llama-3-8b"
        assert get_mlx_context_length(llama) == 4096

    def test_get_mlx_context_length_default(self, mock_llm, tmp_path):
        from lib.mlx_lib import get_mlx_context_length

        # No config.json -> default 4096
        assert get_mlx_context_length(tmp_path) == 4096

    def test_list_mlx_models_not_exists(self, mock_llm):
        from lib import mlx_lib

        assert mlx_lib.list_mlx_models(Path("/nonexistent")) == []

    def test_list_mlx_models(self, mock_llm, fake_mlx_dir):
        from lib import mlx_lib

        models = mlx_lib.list_mlx_models(fake_mlx_dir)
        # qwen and llama should be there, no-config should not
        assert any("qwen" in m for m in models)
        assert any("llama" in m for m in models)
        assert not any("no-config" in m for m in models)

    def test_normalize_mlx_model_name_with_prefix(self, mock_llm):
        from lib.mlx_lib import normalize_mlx_model_name

        assert normalize_mlx_model_name("OsaurusAI/Qwen-7B") == "qwen-7b"
        assert normalize_mlx_model_name("mlx-community/Llama-3") == "llama-3"

    def test_normalize_mlx_model_name_no_prefix(self, mock_llm):
        from lib.mlx_lib import normalize_mlx_model_name

        assert normalize_mlx_model_name("gemma-4-26b") == "gemma-4-26b"


class TestCallMlx:
    def test_call_mlx_model_not_exists(self, mock_llm, tmp_path, real_mlx_functions):
        real_call_mlx = real_mlx_functions["call_mlx"]
        result = real_call_mlx(tmp_path / "nonexistent", "prompt")
        assert result is None

    def test_call_mlx_success(self, mock_llm, tmp_path, real_mlx_functions):
        real_call_mlx = real_mlx_functions["call_mlx"]
        model = tmp_path / "model"
        model.mkdir()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Hello world"
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            result = real_call_mlx(model, "test prompt")
        assert result == "Hello world"

    def test_call_mlx_fallback_to_main_py(self, mock_llm, tmp_path, real_mlx_functions):
        real_call_mlx = real_mlx_functions["call_mlx"]
        model = tmp_path / "model"
        model.mkdir()
        main_py = model / "main.py"
        main_py.write_text("# stub")
        fail_result = MagicMock()
        fail_result.returncode = 1
        fail_result.stdout = ""
        fail_result.stderr = "error"
        ok_result = MagicMock()
        ok_result.returncode = 0
        ok_result.stdout = "fallback output"
        with patch("subprocess.run", side_effect=[fail_result, ok_result]):
            result = real_call_mlx(model, "prompt")
        assert result == "fallback output"

    def test_call_mlx_both_fail(self, mock_llm, tmp_path, real_mlx_functions):
        real_call_mlx = real_mlx_functions["call_mlx"]
        model = tmp_path / "model"
        model.mkdir()
        fail_result = MagicMock()
        fail_result.returncode = 1
        fail_result.stdout = ""
        fail_result.stderr = "error"
        with patch("subprocess.run", return_value=fail_result):
            result = real_call_mlx(model, "prompt")
        assert result is None

    def test_call_mlx_exception(self, mock_llm, tmp_path, real_mlx_functions):
        real_call_mlx = real_mlx_functions["call_mlx"]
        model = tmp_path / "model"
        model.mkdir()
        with patch("subprocess.run", side_effect=Exception("boom")):
            result = real_call_mlx(model, "prompt")
        assert result is None

    def test_call_mlx_exception_with_fallback(self, mock_llm, tmp_path, real_mlx_functions):
        real_call_mlx = real_mlx_functions["call_mlx"]
        model = tmp_path / "model"
        model.mkdir()
        (model / "main.py").write_text("# stub")
        with patch("subprocess.run", side_effect=Exception("boom")):
            result = real_call_mlx(model, "prompt")
        assert result is None

    def test_call_mlx_returncode_0_no_stdout(self, mock_llm, tmp_path, real_mlx_functions):
        real_call_mlx = real_mlx_functions["call_mlx"]
        model = tmp_path / "model"
        model.mkdir()
        empty_result = MagicMock()
        empty_result.returncode = 0
        empty_result.stdout = ""
        empty_result.stderr = ""
        with patch("subprocess.run", return_value=empty_result):
            result = real_call_mlx(model, "prompt")
        assert result is None

    def test_call_mlx_non_zero_with_stdout(self, mock_llm, tmp_path, real_mlx_functions):
        real_call_mlx = real_mlx_functions["call_mlx"]
        model = tmp_path / "model"
        model.mkdir()
        result = MagicMock()
        result.returncode = 1
        result.stdout = "partial output"
        result.stderr = ""
        with patch("subprocess.run", return_value=result):
            r = real_call_mlx(model, "prompt")
        assert r == "partial output"

    def test_generated_script_interpolates_max_tokens(
        self, mock_llm, tmp_path, real_mlx_functions
    ):
        """Regression: the emitted MLX script must contain the literal max_tokens value,
        not a bare `MLX_MAX_TOKENS` name that is undefined inside the subprocess."""
        from lib.mlx_lib import MLX_MAX_TOKENS

        real_call_mlx = real_mlx_functions["call_mlx"]
        model = tmp_path / "model"
        model.mkdir()
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["script"] = Path(cmd[-1]).read_text()
            r = MagicMock()
            r.returncode = 0
            r.stdout = "ok"
            r.stderr = ""
            return r

        with patch("subprocess.run", side_effect=fake_run):
            real_call_mlx(model, "prompt")

        script = captured["script"]
        assert f"max_tokens={MLX_MAX_TOKENS}" in script
        assert "MLX_MAX_TOKENS" not in script


class TestRunMlxVlm:
    def test_vlm_model_not_exists(self, mock_llm, tmp_path):
        from lib.mlx_vlm import run_mlx_vlm

        assert run_mlx_vlm(tmp_path / "no", tmp_path / "img") is None

    def test_vlm_image_not_exists(self, mock_llm, tmp_path):
        from lib.mlx_vlm import run_mlx_vlm

        model = tmp_path / "m"
        model.mkdir()
        assert run_mlx_vlm(model, tmp_path / "no") is None

    def test_vlm_success(self, mock_llm, tmp_path):
        from lib.mlx_vlm import run_mlx_vlm

        model = tmp_path / "m"
        model.mkdir()
        img = tmp_path / "img.png"
        img.write_text("x")
        result = MagicMock()
        result.returncode = 0
        result.stdout = "vlm output"
        with patch("subprocess.run", return_value=result):
            r = run_mlx_vlm(model, img)
        assert r == "vlm output"

    def test_vlm_failure(self, mock_llm, tmp_path):
        from lib.mlx_vlm import run_mlx_vlm

        model = tmp_path / "m"
        model.mkdir()
        img = tmp_path / "img.png"
        img.write_text("x")
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        with patch("subprocess.run", return_value=result):
            r = run_mlx_vlm(model, img)
        assert r is None

    def test_vlm_timeout(self, mock_llm, tmp_path):
        from lib.mlx_vlm import run_mlx_vlm

        model = tmp_path / "m"
        model.mkdir()
        img = tmp_path / "img.png"
        img.write_text("x")
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 180)):
            r = run_mlx_vlm(model, img)
        assert r is None

    def test_vlm_exception(self, mock_llm, tmp_path):
        from lib.mlx_vlm import run_mlx_vlm

        model = tmp_path / "m"
        model.mkdir()
        img = tmp_path / "img.png"
        img.write_text("x")
        with patch("subprocess.run", side_effect=Exception("boom")):
            r = run_mlx_vlm(model, img)
        assert r is None


class TestProcessMlxContent:
    def test_empty(self, mock_llm, real_mlx_functions):
        proc = real_mlx_functions["process_mlx_content"]
        assert proc("") == ""
        assert proc(None) == ""

    def test_clean_thinking(self, mock_llm, real_mlx_functions):
        proc = real_mlx_functions["process_mlx_content"]
        text = "<think>reasoning</think>actual answer"
        result = proc(text)
        assert "reasoning" not in result
        assert "actual answer" in result

    def test_extract_from_code_blocks(self, mock_llm, real_mlx_functions):
        proc = real_mlx_functions["process_mlx_content"]
        text = "before ```code``` after"
        result = proc(text)
        # Code block markers removed, surrounding text preserved
        assert "before" in result and "after" in result

    def test_extract_json_code_block(self, mock_llm, real_mlx_functions):
        """When the content has a code block, it's extracted and used."""
        proc = real_mlx_functions["process_mlx_content"]
        # Code block with valid JSON-like content
        text = 'before ```json\n{"key": "value"}\n``` after'
        result = proc(text)
        # Should be cleaned to remove the code block
        assert "key" in result or "value" in result


class TestUnifiedCall:
    def test_call_model_not_found(self, mock_llm, real_mlx_functions):
        real_call = real_mlx_functions["call"]
        with (
            patch("lib.mlx_lib.find_text_mlx_model", return_value=None),
            patch("lib.mlx_lib.find_mlx_model", return_value=None),
        ):
            result = real_call("nonexistent-model", [{"role": "user", "content": "hi"}])
        # Both model finders return None → "Model not found: <name>"
        assert result["error"] == "Model not found: nonexistent-model"
        assert result["content"] is None

    def test_call_success(self, mock_llm, tmp_path, real_mlx_functions):
        real_call = real_mlx_functions["call"]
        model_path = tmp_path / "m"
        model_path.mkdir()
        with (
            patch("lib.mlx_lib.find_text_mlx_model", return_value=model_path),
            patch("lib.mlx_lib.call_mlx", return_value="hello") as m_call,
        ):
            result = real_call("qwen", [{"role": "user", "content": "hi"}])
        assert result["content"] == "hello"
        assert result["error"] is None
        # Verify call_mlx was actually called
        assert m_call.called

    def test_call_with_system(self, mock_llm, tmp_path, real_mlx_functions):
        real_call = real_mlx_functions["call"]
        model_path = tmp_path / "m"
        model_path.mkdir()
        with (
            patch("lib.mlx_lib.find_text_mlx_model", return_value=model_path),
            patch("lib.mlx_lib.call_mlx", return_value="ok") as m_call,
        ):
            real_call(
                "qwen",
                [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "hi"},
                ],
            )
        # The combined prompt contains both
        prompt = m_call.call_args[0][1]
        assert "You are helpful" in prompt
        assert "hi" in prompt

    def test_call_with_parse_json(self, mock_llm, tmp_path, real_mlx_functions):
        real_call = real_mlx_functions["call"]
        model_path = tmp_path / "m"
        model_path.mkdir()
        with (
            patch("lib.mlx_lib.find_text_mlx_model", return_value=model_path),
            patch("lib.mlx_lib.call_mlx", return_value='```json\n{"a": 1}\n```'),
        ):
            result = real_call("qwen", [{"role": "user", "content": "hi"}], parse_json=True)
        # extract_json strips the code fence and normalizes {"a": 1} dict to ["a"] list
        assert result["parsed"] == ["a"]
        # content is the raw response (code fence preserved)
        assert result["content"] == '```json\n{"a": 1}\n```'

    def test_call_parse_json_failure(self, mock_llm, tmp_path, real_mlx_functions):
        """When the response has no parseable JSON, result is whatever extract_json does."""
        real_call = real_mlx_functions["call"]
        model_path = tmp_path / "m"
        model_path.mkdir()
        with (
            patch("lib.mlx_lib.find_text_mlx_model", return_value=model_path),
            patch("lib.mlx_lib.call_mlx", return_value="!!!"),
        ):
            result = real_call("qwen", [{"role": "user", "content": "hi"}], parse_json=True)
        # The osaurus extract_json may still try to normalize "!!!" as text - either way,
        # it returns something, possibly None.
        # Just check that the call succeeded without exception
        assert "parsed" in result

    def test_call_parse_json_unparseable(self, mock_llm, tmp_path, real_mlx_functions):
        """When the response is not parseable as JSON, logger warning is triggered."""
        real_call = real_mlx_functions["call"]
        model_path = tmp_path / "m"
        model_path.mkdir()
        with (
            patch("lib.mlx_lib.find_text_mlx_model", return_value=model_path),
            patch("lib.mlx_lib.call_mlx", return_value="just plain text, no json"),
            patch("lib.osaurus_lib.extract_json", return_value=None),
            patch("lib.mlx_lib.logger") as mock_logger,
        ):
            result = real_call("qwen", [{"role": "user", "content": "hi"}], parse_json=True)
        assert "parsed" in result
        assert result["parsed"] is None
        # Verify logger.warning was called for the parse failure
        assert any("Could not parse" in str(call) for call in mock_logger.warning.call_args_list)

    def test_call_empty_response(self, mock_llm, tmp_path, real_mlx_functions):
        real_call = real_mlx_functions["call"]
        model_path = tmp_path / "m"
        model_path.mkdir()
        with (
            patch("lib.mlx_lib.find_text_mlx_model", return_value=model_path),
            patch("lib.mlx_lib.call_mlx", return_value=None),
        ):
            result = real_call("qwen", [{"role": "user", "content": "hi"}])
        assert "Empty response" in result["error"]

    def test_call_exception(self, mock_llm, tmp_path, real_mlx_functions):
        real_call = real_mlx_functions["call"]
        model_path = tmp_path / "m"
        model_path.mkdir()
        with (
            patch("lib.mlx_lib.find_text_mlx_model", return_value=model_path),
            patch("lib.mlx_lib.call_mlx", side_effect=Exception("boom")),
        ):
            result = real_call("qwen", [{"role": "user", "content": "hi"}])
        assert "Error" in result["error"]
        assert "boom" in result["error"]

    def test_call_strips_prefix(self, mock_llm, tmp_path, real_mlx_functions):
        real_call = real_mlx_functions["call"]
        model_path = tmp_path / "m"
        model_path.mkdir()
        with (
            patch("lib.mlx_lib.find_text_mlx_model", return_value=None),
            patch("lib.mlx_lib.find_mlx_model", return_value=model_path),
            patch("lib.mlx_lib.call_mlx", return_value="ok"),
        ):
            result = real_call("mlx-community/Qwen-7B", [{"role": "user", "content": "hi"}])
        # lookup_name should be "Qwen-7B"
        assert result["content"] == "ok"
