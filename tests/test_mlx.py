import pytest


@pytest.mark.skip(reason="Integration test requiring MLX model installed")
class TestMlxModel:
    def test_find_mlx_model(self):
        from lib.mlx_lib import find_text_mlx_model, call_mlx
        mlx_model = find_text_mlx_model(["qwen"])
        assert mlx_model is not None
        sys_prompt = "Output JSON now."
        usr_prompt = "Extract popular Vaughan venues."
        raw = call_mlx(mlx_model, f"System: {sys_prompt}\n\nUser: {usr_prompt}")
        assert raw is not None
        assert len(raw) > 0
