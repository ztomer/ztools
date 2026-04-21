# Tests for config module
import os
from lib.config import get_model_field_mapping


def test_get_model_field_mapping_qwen():
    """Test qwen field mapping from yaml config."""
    mapping = get_model_field_mapping("qwen3.6-35b-a3b-mxfp4")
    assert mapping.get("category") == "target_ages"
    assert mapping.get("context_highlight") == "price"
    assert mapping.get("type") == "weather"


def test_get_model_field_mapping_unknown():
    """Test unknown model returns empty mapping."""
    mapping = get_model_field_mapping("unknown-model")
    assert mapping == {}


def test_osaurus_port():
    """Test osaurus server port is 1337 (not 8000)."""
    import subprocess
    result = subprocess.run(["osaurus", "status"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "1337" in result.stdout
