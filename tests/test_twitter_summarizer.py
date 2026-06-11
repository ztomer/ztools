import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from twitter import resolve_since_time, main, parse_args


class TestResolveSinceTime:
    def test_relative_24h(self):
        result = resolve_since_time("24h", {})
        now = datetime.now(timezone.utc)
        diff = now - result
        assert timedelta(hours=23, minutes=55) <= diff <= timedelta(hours=24, minutes=5)

    def test_relative_6h(self):
        result = resolve_since_time("6h", {})
        diff = datetime.now(timezone.utc) - result
        assert timedelta(hours=5, minutes=55) <= diff <= timedelta(hours=6, minutes=5)

    def test_iso_format(self):
        result = resolve_since_time("2026-01-15T10:30:00+00:00", {})
        expected = datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc)
        assert result == expected

    def test_iso_without_tz_gets_utc(self):
        result = resolve_since_time("2026-01-15T10:30:00", {})
        assert result.tzinfo == timezone.utc

    def test_fallback_to_state_last_run(self):
        state = {"last_run": "2026-05-01T12:00:00+00:00"}
        result = resolve_since_time(None, state)
        expected = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_fallback_to_no_state_24h(self):
        result = resolve_since_time(None, {})
        now = datetime.now(timezone.utc)
        diff = now - result
        assert timedelta(hours=23, minutes=55) <= diff <= timedelta(hours=24, minutes=5)

    def test_invalid_format_falls_back(self):
        result = resolve_since_time("invalid", {})
        now = datetime.now(timezone.utc)
        diff = now - result
        assert timedelta(hours=23, minutes=55) <= diff <= timedelta(hours=24, minutes=5)


class TestParseArgs:
    def test_defaults(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["twitter"])
        args = parse_args()
        assert args.since is None
        assert args.debug is False
        assert args.clean is False
        assert args.use_cache is False

    def test_with_flags(self, monkeypatch):
        monkeypatch.setattr("sys.argv", [
            "twitter",
            "--since", "24h",
            "--debug",
            "--clean",
            "--use-cache",
            "--model", "m1",
            "--base-url", "http://x",
            "--api-key", "k",
        ])
        args = parse_args()
        assert args.since == "24h"
        assert args.debug is True
        assert args.clean is True
        assert args.use_cache is True
        assert args.model == "m1"
        assert args.base_url == "http://x"
        assert args.api_key == "k"


class TestMain:
    def test_clean_exits(self, monkeypatch, tmp_path, mock_llm):
        """--clean calls clean_folder and continues (clean_folder sys.exit(0)s)."""
        monkeypatch.setattr("sys.argv", ["twitter", "--clean", "--output", str(tmp_path)])
        with patch("twitter.cli.clean_folder", side_effect=SystemExit(0)) as mock_clean:
            with pytest.raises(SystemExit):
                main()
            mock_clean.assert_called_once()

    def test_use_cache_no_tweets(self, monkeypatch, mock_llm):
        """--use-cache with no cached tweets → sys.exit(1)."""
        monkeypatch.setattr("sys.argv", ["twitter", "--use-cache"])
        with patch("twitter.cli.load_state", return_value={}), \
             patch("twitter.cli.load_debug_cache", return_value=[]), \
             patch("twitter.cli.sys") as mock_sys:
            mock_sys.exit.side_effect = SystemExit(1)
            with pytest.raises(SystemExit):
                main()
            mock_sys.exit.assert_called_with(1)

    def test_use_cache_with_tweets(self, monkeypatch, mock_llm, tmp_path):
        """--use-cache with cached tweets → summarize and write."""
        monkeypatch.setattr("sys.argv", ["twitter", "--use-cache", "--output", str(tmp_path)])
        tweets = [{"screen_name": "u", "text": "t", "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc)}]
        with patch("twitter.cli.load_state", return_value={}), \
             patch("twitter.cli.load_debug_cache", return_value=tweets), \
             patch("twitter.cli.summarize_with_llm", return_value="## Good\n- a\n- b\n- c") as mock_sum, \
             patch("twitter.cli.write_markdown", return_value=(tmp_path / "out.md", "md")) as mock_write:
            main()
        # Verify summarize was called with our cached tweets
        mock_sum.assert_called_once()
        args, kwargs = mock_sum.call_args
        assert args[0] == tweets
        # Verify write was called
        mock_write.assert_called_once()

    def test_no_tweets_exits(self, monkeypatch, mock_llm):
        """When no tweets collected → sys.exit(0)."""
        monkeypatch.setattr("sys.argv", ["twitter"])
        with patch("twitter.cli.load_state", return_value={}), \
             patch("twitter.cli.collect_tweets_via_browser", return_value=[]), \
             patch("twitter.cli.sys") as mock_sys:
            mock_sys.exit.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()
            mock_sys.exit.assert_called_with(0)

    def test_llm_error_exits(self, monkeypatch, mock_llm, tmp_path):
        """When summarize_with_llm returns error → sys.exit(1)."""
        monkeypatch.setattr("sys.argv", ["twitter", "--output", str(tmp_path)])
        tweets = [{"screen_name": "u", "text": "t", "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc)}]
        with patch("twitter.cli.load_state", return_value={}), \
             patch("twitter.cli.collect_tweets_via_browser", return_value=tweets), \
             patch("twitter.cli.summarize_with_llm", return_value="[LLM error: oom]"), \
             patch("twitter.cli.sys") as mock_sys:
            mock_sys.exit.side_effect = SystemExit(1)
            with pytest.raises(SystemExit):
                main()
            mock_sys.exit.assert_called_with(1)

    def test_full_success(self, monkeypatch, mock_llm, tmp_path, capsys):
        """Full happy path: collect, summarize, write, save state."""
        monkeypatch.setattr("sys.argv", ["twitter", "--output", str(tmp_path)])
        tweets = [{"screen_name": "u", "text": "t", "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc)}]
        with patch("twitter.cli.load_state", return_value={}), \
             patch("twitter.cli.collect_tweets_via_browser", return_value=tweets), \
             patch("twitter.cli.summarize_with_llm", return_value="## Good\n- a\n- b\n- c"), \
             patch("twitter.cli.write_markdown", return_value=(tmp_path / "out.md", "md content")), \
             patch("twitter.cli.save_state") as mock_save:
            main()
            mock_save.assert_called_once()
            # Verify the save state was called with a last_run key
            call_args = mock_save.call_args[0][0]
            assert "last_run" in call_args
            # Verify output was written
            captured = capsys.readouterr()
            assert "Summary written" in captured.out or "out.md" in captured.out

    def test_main_with_iso_since(self, monkeypatch, mock_llm, tmp_path):
        """--since with ISO format is used to compute since_time."""
        monkeypatch.setattr("sys.argv", ["twitter", "--since", "2026-05-01T10:00:00+00:00", "--output", str(tmp_path)])
        tweets = [{"screen_name": "u", "text": "t", "created_at": datetime(2026, 5, 1, 11, 0, tzinfo=timezone.utc)}]
        with patch("twitter.cli.load_state", return_value={}), \
             patch("twitter.cli.collect_tweets_via_browser", return_value=tweets) as mock_collect, \
             patch("twitter.cli.summarize_with_llm", return_value="## Good\n- a\n- b\n- c"), \
             patch("twitter.cli.write_markdown", return_value=(tmp_path / "out.md", "md")):
            main()
            # Verify since_time was passed to collect_tweets_via_browser
            call_args = mock_collect.call_args
            since_arg = call_args[0][0]
            assert since_arg.year == 2026
            assert since_arg.month == 5
            assert since_arg.day == 1
            assert since_arg.hour == 10

    def test_playwright_import_fallback(self, monkeypatch):
        """When playwright fails to import, sync_playwright/PWTimeout are None."""
        import sys
        monkeypatch.setitem(sys.modules, "playwright", None)
        monkeypatch.setitem(sys.modules, "playwright.sync_api", None)

        if "twitter.browser" in sys.modules:
            del sys.modules["twitter.browser"]
        import twitter.browser

        assert twitter.browser.sync_playwright is None
        assert twitter.browser.PWTimeout is None
