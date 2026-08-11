"""Tests for weekend_config module."""

from pathlib import Path
from unittest.mock import patch


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

    def test_save_events_cache_creates_dir(self, tmp_path, monkeypatch):
        """Cache save must create parent dir if missing (regression)."""
        from weekend.config import save_events_cache

        cache_file = tmp_path / "nonexistent" / "subdir" / "events.json"
        assert not cache_file.parent.exists()
        monkeypatch.setattr("weekend.config.DEBUG_EVENTS_FILE", cache_file)
        save_events_cache('["data"]')
        assert cache_file.read_text() == '["data"]'

    def test_save_venues_cache_creates_dir(self, tmp_path, monkeypatch):
        """Venues cache save must create parent dir if missing (regression)."""
        from weekend.config import save_venues_cache

        cache_file = tmp_path / "nonexistent" / "subdir" / "venues.json"
        assert not cache_file.parent.exists()
        monkeypatch.setattr("weekend.config.DEBUG_VENUES_FILE", cache_file)
        save_venues_cache('["data"]')
        assert cache_file.read_text() == '["data"]'

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
        import builtins

        from weekend.config import load_weekend_config

        real_open = builtins.open

        def fake_open(*args, **kwargs):
            if "weekend" in str(args[0]):
                raise FileNotFoundError("file missing")
            return real_open(*args, **kwargs)

        with patch("builtins.open", side_effect=fake_open):
            result = load_weekend_config()
        assert result == {}


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
            DATES_STR,
            DEBUG_EVENTS_FILE,
            DEBUG_VENUES_FILE,
            MODEL_CONFIG,
            MODEL_NAME,
            OSAURUS_APP,
            OSAURUS_BASE_URL,
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


class TestFetchDataIntegration:
    def test_fetch_data_creates_cache_dirs(self, tmp_path, monkeypatch):
        """_fetch_data creates cache dirs and doesn't crash when all mocked."""
        with monkeypatch.context() as m:
            m.setattr("weekend.cli.ensure_server", lambda: None)
            m.setattr("weekend.cli.get_weekend_dates_string", lambda *a: "Jul 17-19, 2026")
            m.setattr("weekend.cli.fetch_weather", lambda *a: "Daily Forecast: 24C Clear")
            m.setattr("weekend.cli.fetch_transient_events", lambda *a: "Mock events data")
            m.setattr("weekend.cli.fetch_fixed_venues", lambda *a: "Mock venues data")

            cache_dir = tmp_path / ".cache" / "weekend"
            m.setattr("weekend.config.DEBUG_EVENTS_FILE", cache_dir / "events_debug_cache.json")
            m.setattr("weekend.config.DEBUG_VENUES_FILE", cache_dir / "venues_debug_cache.json")

            import datetime

            from weekend.cli import _fetch_data

            fri = datetime.date(2026, 7, 17)
            sun = datetime.date(2026, 7, 19)
            weather, events, venues, dates = _fetch_data(fri, sun, "2026", "July", use_cache=False)

        assert cache_dir.exists()
        assert weather == "Daily Forecast: 24C Clear"
        assert events == "Mock events data"
        assert venues == "Mock venues data"
        assert dates == "Jul 17-19, 2026"

    def test_fetch_data_with_cache_uses_cached(self, tmp_path, monkeypatch):
        """With use_cache=True, cached data is returned without re-fetching."""
        with monkeypatch.context() as m:
            m.setattr("weekend.cli.ensure_server", lambda: None)
            m.setattr("weekend.cli.get_weekend_dates_string", lambda *a: "Jul 17-19, 2026")
            m.setattr("weekend.cli.fetch_weather", lambda *a: "Daily Forecast: 24C Clear")
            m.setattr("weekend.cli.fetch_transient_events", lambda *a: "Mock events data")
            m.setattr("weekend.cli.fetch_fixed_venues", lambda *a: "Mock venues data")

            cache_dir = tmp_path / ".cache" / "weekend"
            # Pre-populate cache
            cache_dir.mkdir(parents=True)
            (cache_dir / "events_debug_cache.json").write_text("Cached events")
            (cache_dir / "venues_debug_cache.json").write_text("Cached venues")
            m.setattr("weekend.config.DEBUG_EVENTS_FILE", cache_dir / "events_debug_cache.json")
            m.setattr("weekend.config.DEBUG_VENUES_FILE", cache_dir / "venues_debug_cache.json")

            import datetime

            from weekend.cli import _fetch_data

            fri = datetime.date(2026, 7, 17)
            sun = datetime.date(2026, 7, 19)
            weather, events, venues, dates = _fetch_data(fri, sun, "2026", "July", use_cache=True)

        assert events == "Cached events"
        assert venues == "Cached venues"


class TestMainIntegration:
    def test_main_runs_with_mocked_deps(self, tmp_path, monkeypatch):
        """Full main() flow doesn't crash when external deps are mocked."""
        import datetime

        mock_events = [
            {
                "name": "Kids Coding Workshop",
                "location": "Vaughan",
                "target_ages": "8-14",
                "price": "$25",
                "weather": "indoor",
                "day": "Saturday",
            },
            {
                "name": "Nature Walk",
                "location": "Toronto",
                "target_ages": "All",
                "price": "Free",
                "weather": "outdoor",
                "day": "Sunday",
            },
            {
                "name": "Swimming",
                "location": "Richmond Hill",
                "target_ages": "5-12",
                "price": "$10",
                "weather": "indoor",
                "day": "Saturday",
            },
            {
                "name": "Art Class",
                "location": "Markham",
                "target_ages": "6-10",
                "price": "$15",
                "weather": "indoor",
                "day": "Sunday",
            },
            {
                "name": "Bike Trail",
                "location": "Thornhill",
                "target_ages": "All",
                "price": "Free",
                "weather": "outdoor",
                "day": "Saturday",
            },
        ]
        mock_fixed = [
            {
                "name": "Canada's Wonderland",
                "location": "Vaughan",
                "target_ages": "All",
                "price": "$$$",
                "weather": "outdoor",
            },
            {
                "name": "Science Centre",
                "location": "Toronto",
                "target_ages": "All",
                "price": "$25",
                "weather": "indoor",
            },
            {
                "name": "Adventure Park",
                "location": "Vaughan",
                "target_ages": "6-14",
                "price": "$30",
                "weather": "outdoor",
            },
            {
                "name": "Indoor Playground",
                "location": "Richmond Hill",
                "target_ages": "2-10",
                "price": "$12",
                "weather": "indoor",
            },
            {
                "name": "Library Story Time",
                "location": "Thornhill",
                "target_ages": "3-7",
                "price": "Free",
                "weather": "indoor",
            },
        ]

        with monkeypatch.context() as m:
            m.setattr("weekend.cli.setup_signals", lambda: None)
            m.setattr("weekend.cli.init_config", lambda: None)
            m.setattr("weekend.cli.ensure_server", lambda: None)
            m.setattr("weekend.cli.check_server_or_die", lambda *a: None)
            m.setattr("weekend.cli.get_best_model", lambda *a: "mock-model")
            m.setattr(
                "weekend.cli.get_weekend_date_objects",
                lambda: (datetime.date(2026, 7, 17), datetime.date(2026, 7, 19)),
            )
            # The corpus must contain the events the mocked model "found":
            # provenance enforcement drops rows that trace to nothing fetched,
            # and a fixture whose model invents everything is not a pipeline
            # this test can meaningfully exercise.
            mock_corpus = "\n".join(
                f"- {e['name']} in {e['location']}" for e in (*mock_events, *mock_fixed)
            )
            m.setattr(
                "weekend.cli._fetch_data",
                lambda *a: ("24C Clear", mock_corpus, mock_corpus, "Jul 17-19"),
            )
            m.setattr(
                "weekend.cli.get_model_field_mapping",
                lambda *a: {"name": "name", "location": "location"},
            )
            m.setattr(
                "weekend.cli.generate_weekend_plan",
                lambda *a, **kw: (
                    {"transient_events": mock_events},
                    {"fixed_activities": mock_fixed},
                ),
            )
            m.setattr("weekend.config.DEBUG_EVENTS_FILE", tmp_path / "events_debug_cache.json")
            m.setattr("weekend.config.DEBUG_VENUES_FILE", tmp_path / "venues_debug_cache.json")
            m.setattr("weekend.cli.OUTPUT_DIR_PATH", str(tmp_path))

            m.setenv("OLLAMA_MODEL", "mock-model")

            import argparse

            from weekend.cli import main

            ns = argparse.Namespace(
                use_cache=False, model="mock-model", skip_web=False, debug=False
            )
            main(ns)

        # Output file was written
        out_files = list(tmp_path.glob("weekend_plan_*.md"))
        assert len(out_files) == 1
        content = out_files[0].read_text()
        assert "Kids Coding Workshop" in content
        # C8: "Canada's Wonderland" is in the user's conf/weekend.toml
        # exclude_places, so the pipeline must DROP it. This assertion used to
        # require its presence -- i.e. it pinned the unenforced-exclusion defect.
        assert "Canada's Wonderland" not in content
        assert "Science Centre" in content
