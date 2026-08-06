# Tests for config module
import pytest
from lib.config import (
    clear_model_config_cache,
    get_model_field_mapping,
    get_model_quirks,
    get_model_top_keys,
)


def test_get_model_field_mapping_qwen():
    """Test qwen field mapping from toml config."""
    clear_model_config_cache()
    mapping = get_model_field_mapping("qwen3.6-35b-a3b-mxfp4")
    assert mapping.get("category") == "target_ages"
    assert mapping.get("context_highlight") == "price"
    assert mapping.get("type") == "weather"


def test_get_model_field_mapping_gemma():
    """Test gemma field mapping from toml config."""
    clear_model_config_cache()
    mapping = get_model_field_mapping("gemma-4-31b-it-jang_4m")
    assert mapping.get("activity") == "name"
    assert mapping.get("venue") == "location"


def test_get_model_field_mapping_unknown():
    """Test unknown model returns empty mapping."""
    clear_model_config_cache()
    mapping = get_model_field_mapping("unknown-model")
    assert mapping == {}


def test_get_model_top_keys_qwen():
    """Test qwen top keys from toml config."""
    clear_model_config_cache()
    keys = get_model_top_keys("qwen3.6-35b-a3b-mxfp4")
    assert "fixed" in keys
    assert "transient" in keys
    assert "fixed_activities" in keys["fixed"]
    assert "transient_events" in keys["transient"]


def test_get_model_top_keys_gemma():
    """Test gemma top keys from toml config."""
    clear_model_config_cache()
    keys = get_model_top_keys("gemma-4-31b-it-jang_4m")
    assert "transient_events" in keys["transient"]


def test_get_model_top_keys_default():
    """Test default top keys when model not found."""
    clear_model_config_cache()
    keys = get_model_top_keys("unknown-model")
    assert "fixed_activities" in keys["fixed"]
    assert "transient_events" in keys["transient"]


def test_get_model_quirks_qwen():
    """Test qwen quirks from toml config."""
    clear_model_config_cache()
    quirks = get_model_quirks("qwen3.6-35b-a3b-mxfp4")
    assert len(quirks) > 0
    assert any(q.get("type") == "prefix" for q in quirks)


def test_get_model_quirks_gemma():
    """Test gemma quirks from toml config."""
    clear_model_config_cache()
    quirks = get_model_quirks("gemma-4-31b-it-jang_4m")
    assert len(quirks) > 0


def test_osaurus_port():
    """Test osaurus server port is 1337 (not 8000)."""
    import shutil
    import subprocess

    if not shutil.which("osaurus"):
        pytest.skip("osaurus binary not found on PATH")

    result = subprocess.run(["osaurus", "status"], capture_output=True, text=True)
    assert result.returncode in (0, 1)
    assert "1337" in result.stdout or "stopped" in result.stdout



def test_init_config_missing_file():
    """Test init_config raises FileNotFoundError for missing file."""
    from lib.config import init_config

    with pytest.raises(FileNotFoundError):
        init_config("nonexistent_config_file_xyz.toml")


def test_init_config_invalid_toml(tmp_path):
    """Test init_config raises ConfigurationError for corrupt TOML file."""
    from lib.config import ConfigurationError, init_config

    invalid_file = tmp_path / "invalid_config.toml"
    invalid_file.write_text("[[[")  # Invalid TOML syntax

    with pytest.raises(ConfigurationError) as excinfo:
        init_config(str(invalid_file))
    assert "Failed to read configuration file" in str(excinfo.value)
