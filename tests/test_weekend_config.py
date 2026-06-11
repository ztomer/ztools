"""Tests for weekend_config module."""
import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestCacheFunctions:
    def test_load_events_cache_exists(self, tmp_path, monkeypatch):
        from weekend.config import load_events_cache
        cache_file = tmp_path / ".weekend_events_debug_cache.json"
        cache_file.write_text('["event1", "event2"]')
        monkeypatch.setattr("weekend.config.DEBUG_EVENTS_FILE", cache_file)
        result = load_events_cache()
        assert result == '["event1", "event2"]'

    def test_load_events_cache_missing(self, tmp_path, monkeypatch):
        from weekend.config import load_events_cache
        cache_file = tmp_path / "missing.json"
        monkeypatch.setattr("weekend.config.DEBUG_EVENTS_FILE", cache_file)
        assert load_events_cache() is None

    def test_save_events_cache(self, tmp_path, monkeypatch):
        from weekend.config import save_events_cache
        cache_file = tmp_path / ".weekend_events_debug_cache.json"
        monkeypatch.setattr("weekend.config.DEBUG_EVENTS_FILE", cache_file)
        save_events_cache('["saved"]')
        assert cache_file.read_text() == '["saved"]'

    def test_load_venues_cache_exists(self, tmp_path, monkeypatch):
        from weekend.config import load_venues_cache
        cache_file = tmp_path / ".weekend_venues_debug_cache.json"
        cache_file.write_text('["venue1"]')
        monkeypatch.setattr("weekend.config.DEBUG_VENUES_FILE", cache_file)
        result = load_venues_cache()
        assert result == '["venue1"]'

    def test_load_venues_cache_missing(self, tmp_path, monkeypatch):
        from weekend.config import load_venues_cache
        cache_file = tmp_path / "missing.json"
        monkeypatch.setattr("weekend.config.DEBUG_VENUES_FILE", cache_file)
        assert load_venues_cache() is None

    def test_save_venues_cache(self, tmp_path, monkeypatch):
        from weekend.config import save_venues_cache
        cache_file = tmp_path / ".weekend_venues_debug_cache.json"
        monkeypatch.setattr("weekend.config.DEBUG_VENUES_FILE", cache_file)
        save_venues_cache('["saved"]')
        assert cache_file.read_text() == '["saved"]'


class TestLoadWeekendConfig:
    def test_load_existing(self):
        from weekend.config import load_weekend_config
        # Just verify it works on the real config (returns dict)
        result = load_weekend_config()
        assert isinstance(result, dict)

    def test_load_exception(self, capsys):
        from weekend.config import load_weekend_config
        import builtins
        real_open = builtins.open
        def fake_open(*args, **kwargs):
            # Only fail for the weekend.yaml file
            if "weekend.yaml" in str(args[0]):
                raise Exception("file missing")
            return real_open(*args, **kwargs)
        with patch("builtins.open", side_effect=fake_open):
            result = load_weekend_config()
        assert result == {}
        out = capsys.readouterr()
        assert "Failed to load weekend.yaml" in out.out


class TestServerHelpers:
    def test_is_server_running_ours(self):
        from weekend.config import is_server_running_ours
        with patch("weekend.config.is_server_running", return_value=True):
            assert is_server_running_ours() is True

    def test_is_server_running_ours_false(self):
        from weekend.config import is_server_running_ours
        with patch("weekend.config.is_server_running", return_value=False):
            assert is_server_running_ours() is False

    def test_restart_osaurus(self):
        from weekend.config import restart_osaurus
        with patch("weekend.config.restart_server", return_value=True) as mock_restart:
            result = restart_osaurus(wait=10)
        assert result is True
        mock_restart.assert_called_once_with(app_path="/Applications/osaurus.app", wait=10)

    def test_ensure_server(self):
        from weekend.config import ensure_server
        with patch("weekend.config._osaurus_ensure_server", return_value=True) as mock_ensure:
            result = ensure_server(max_retries=5, wait=30)
        assert result is True
        mock_ensure.assert_called_once_with(max_retries=5, wait=30)


class TestConstants:
    def test_constants_defined(self):
        from weekend.config import (
            DEBUG_EVENTS_FILE,
            DEBUG_VENUES_FILE,
            MODEL_CONFIG,
            MODEL_NAME,
            OSAURUS_BASE_URL,
            OSAURUS_APP,
            DATES_STR,
        )
        assert isinstance(DEBUG_EVENTS_FILE, Path)
        assert isinstance(DEBUG_VENUES_FILE, Path)
        # MODEL_CONFIG is the path to the config JSON file
        assert isinstance(MODEL_CONFIG, str)
        assert MODEL_CONFIG.endswith(".json")
        assert isinstance(MODEL_NAME, str)
        assert len(MODEL_NAME) > 0
        assert OSAURUS_BASE_URL.startswith("http")
        assert OSAURUS_APP.endswith(".app")
        assert DATES_STR  # Non-empty
