"""Tests for lib.mlx_vlm (mlx-vlm git-main model loading/generation)."""

from unittest.mock import MagicMock, patch


class TestProbeMlxVlmLoadable:
    def test_not_a_model_dir(self, mock_llm, tmp_path, real_mlx_functions):
        probe = real_mlx_functions["probe_mlx_vlm_loadable"]
        ok, reason = probe(tmp_path / "no-config")
        assert ok is False
        assert "config.json" in reason

    def test_load_ok_parsed(self, mock_llm, tmp_path, real_mlx_functions):
        probe = real_mlx_functions["probe_mlx_vlm_loadable"]
        model = tmp_path / "m"
        model.mkdir()
        (model / "config.json").write_text("{}")
        result = MagicMock()
        result.returncode = 0
        result.stdout = "VLM_LOAD_OK"
        result.stderr = ""
        with patch("subprocess.run", return_value=result):
            ok, reason = probe(model)
        assert ok is True
        assert reason == "ok"

    def test_load_fail_parsed(self, mock_llm, tmp_path, real_mlx_functions):
        probe = real_mlx_functions["probe_mlx_vlm_loadable"]
        model = tmp_path / "m"
        model.mkdir()
        (model / "config.json").write_text("{}")
        result = MagicMock()
        result.returncode = 1
        result.stdout = "VLM_LOAD_FAIL ValueError: Received 126 parameters not in model"
        result.stderr = ""
        with patch("subprocess.run", return_value=result):
            ok, reason = probe(model)
        assert ok is False
        assert "126 parameters" in reason

    def test_cache_reused(self, mock_llm, tmp_path, real_mlx_functions):
        probe = real_mlx_functions["probe_mlx_vlm_loadable"]
        model = tmp_path / "m"
        model.mkdir()
        (model / "config.json").write_text("{}")
        result = MagicMock()
        result.returncode = 0
        result.stdout = "VLM_LOAD_OK"
        result.stderr = ""
        with patch("subprocess.run", return_value=result) as run:
            probe(model)
            probe(model)
        # The second call must hit the cache, not spawn another subprocess.
        assert run.call_count == 1


class TestCallMlxVlm:
    def test_call_mlx_vlm_model_not_exists(self, mock_llm, tmp_path, real_mlx_functions):
        real = real_mlx_functions["call_mlx_vlm"]
        assert real(tmp_path / "no", "prompt") is None

    def test_call_mlx_vlm_success(self, mock_llm, tmp_path, real_mlx_functions):
        real = real_mlx_functions["call_mlx_vlm"]
        model = tmp_path / "m"
        model.mkdir()
        (model / "config.json").write_text("{}")
        result = MagicMock()
        result.returncode = 0
        result.stdout = "## VLM\n- a fact"
        result.stderr = ""
        with patch("subprocess.run", return_value=result):
            assert real(model, "prompt") == "## VLM\n- a fact"

    def test_call_mlx_vlm_load_error(self, mock_llm, tmp_path, real_mlx_functions):
        from lib.mlx_lib import last_mlx_error

        real = real_mlx_functions["call_mlx_vlm"]
        model = tmp_path / "m"
        model.mkdir()
        (model / "config.json").write_text("{}")
        result = MagicMock()
        result.returncode = 1
        result.stdout = "[VLM LOAD ERROR] ValueError: boom"
        result.stderr = ""
        with patch("subprocess.run", return_value=result):
            assert real(model, "prompt") is None
        assert last_mlx_error() is not None
        assert "boom" in last_mlx_error()
