import pytest


@pytest.mark.skip(reason="Integration test requiring MLX model installed")
class TestMlxModel:
    def test_find_mlx_model(self):
        from lib.mlx_lib import call_mlx, find_text_mlx_model

        mlx_model = find_text_mlx_model(["qwen"])
        # Should be a path-like with "qwen" in the name
        assert mlx_model is not None
        assert "qwen" in str(mlx_model).lower()
        sys_prompt = "Output JSON now."
        usr_prompt = "Extract popular Vaughan venues."
        raw = call_mlx(mlx_model, f"System: {sys_prompt}\n\nUser: {usr_prompt}")
        # Raw is a non-empty string response
        assert isinstance(raw, str)
        assert len(raw) > 0
