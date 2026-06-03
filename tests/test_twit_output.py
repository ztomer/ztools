"""Tests for twit_output.py."""
import pytest
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open


class TestLoadSaveState:
    def test_load_state_no_file(self, tmp_path, monkeypatch):
        import twit_output
        # Redirect Path.home() to tmp_path
        monkeypatch.setattr(twit_output, "STATE_FILE", tmp_path / "state.json")
        result = twit_output.load_state()
        assert result == {}

    def test_load_state_valid(self, tmp_path, monkeypatch):
        import twit_output
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({"key": "value"}))
        monkeypatch.setattr(twit_output, "STATE_FILE", state_file)
        result = twit_output.load_state()
        assert result == {"key": "value"}

    def test_load_state_invalid_json(self, tmp_path, monkeypatch):
        import twit_output
        state_file = tmp_path / "state.json"
        state_file.write_text("not json")
        monkeypatch.setattr(twit_output, "STATE_FILE", state_file)
        result = twit_output.load_state()
        assert result == {}

    def test_save_state(self, tmp_path, monkeypatch):
        import twit_output
        state_file = tmp_path / "state.json"
        monkeypatch.setattr(twit_output, "STATE_FILE", state_file)
        twit_output.save_state({"key": "value"})
        assert json.loads(state_file.read_text()) == {"key": "value"}


class TestLoadSaveDebugCache:
    def test_load_debug_cache_no_file(self, tmp_path, monkeypatch):
        import twit_output
        monkeypatch.setattr(twit_output, "DEBUG_CACHE_FILE", tmp_path / "cache.json")
        result = twit_output.load_debug_cache()
        assert result == []

    def test_load_debug_cache_invalid(self, tmp_path, monkeypatch):
        import twit_output
        cache_file = tmp_path / "cache.json"
        cache_file.write_text("not json")
        monkeypatch.setattr(twit_output, "DEBUG_CACHE_FILE", cache_file)
        result = twit_output.load_debug_cache()
        assert result == []

    def test_load_debug_cache_with_dates(self, tmp_path, monkeypatch):
        import twit_output
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps([
            {"id": 1, "created_at": "2024-01-01T12:00:00"},
            {"id": 2, "created_at": "2024-01-02T13:00:00"},
        ]))
        monkeypatch.setattr(twit_output, "DEBUG_CACHE_FILE", cache_file)
        result = twit_output.load_debug_cache()
        assert len(result) == 2
        assert isinstance(result[0]["created_at"], datetime)

    def test_load_debug_cache_no_dates(self, tmp_path, monkeypatch):
        import twit_output
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps([{"id": 1, "screen_name": "x"}]))
        monkeypatch.setattr(twit_output, "DEBUG_CACHE_FILE", cache_file)
        result = twit_output.load_debug_cache()
        assert result == [{"id": 1, "screen_name": "x"}]

    def test_save_debug_cache(self, tmp_path, monkeypatch):
        import twit_output
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(twit_output, "DEBUG_CACHE_FILE", cache_file)
        dt = datetime(2024, 1, 1, 12, 0, 0)
        twit_output.save_debug_cache([{"id": 1, "created_at": dt}])
        loaded = json.loads(cache_file.read_text())
        assert loaded[0]["created_at"] == "2024-01-01T12:00:00"

    def test_save_debug_cache_non_serializable_raises(self, tmp_path, monkeypatch):
        """Line 50: TypeError for non-datetime non-serializable objects."""
        import twit_output
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(twit_output, "DEBUG_CACHE_FILE", cache_file)
        # set is not JSON serializable
        with pytest.raises(TypeError) as exc_info:
            twit_output.save_debug_cache([{"id": 1, "data": {1, 2, 3}}])
        assert "set" in str(exc_info.value) or "not JSON serializable" in str(exc_info.value)


class TestPrintToStdout:
    def test_no_bat_uses_print(self, mock_llm, capsys):
        import twit_output
        with patch("shutil.which", return_value=None):
            twit_output.print_to_stdout("# Hello")
        captured = capsys.readouterr()
        assert "Hello" in captured.out

    def test_with_bat_success(self, mock_llm):
        import twit_output
        with patch("shutil.which", return_value="/usr/bin/bat"), \
             patch("subprocess.run") as mock_run:
            twit_output.print_to_stdout("# Hello")
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["input"] == "# Hello"

    def test_with_bat_failure_falls_back_to_print(self, mock_llm, capsys):
        import twit_output
        with patch("shutil.which", return_value="/usr/bin/bat"), \
             patch("subprocess.run", side_effect=Exception("bat failed")):
            twit_output.print_to_stdout("# Hello")
        captured = capsys.readouterr()
        assert "Hello" in captured.out


class TestWriteMarkdown:
    def test_basic(self, tmp_path):
        from datetime import datetime, timezone
        import twit_output
        tweets = [
            {"screen_name": "alice", "text": "Hello"},
            {"screen_name": "bob", "text": "World"},
        ]
        since = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        until = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        out_path, content = twit_output.write_markdown(
            tweets, "My summary", since, until, tmp_path
        )
        assert out_path.exists()
        assert "My summary" in content
        assert "2024-01-01" in content
        assert "2 from 2 accounts" in content

    def test_unique_authors(self, tmp_path):
        from datetime import datetime, timezone
        import twit_output
        tweets = [
            {"screen_name": "alice", "text": "1"},
            {"screen_name": "alice", "text": "2"},
            {"screen_name": "bob", "text": "3"},
        ]
        since = datetime(2024, 1, 1, tzinfo=timezone.utc)
        until = datetime(2024, 1, 2, tzinfo=timezone.utc)
        out_path, content = twit_output.write_markdown(
            tweets, "S", since, until, tmp_path
        )
        assert "3 from 2 accounts" in content


class TestCleanFolder:
    def test_folder_not_exists(self, tmp_path, monkeypatch):
        import twit_output
        from io import StringIO
        fake_dir = tmp_path / "nonexistent"
        # twit_output does `import sys` and calls sys.exit(0)
        # patch with a real SystemExit-raising function
        with patch.object(twit_output, "sys") as mock_sys, \
             patch("builtins.print") as mock_print:
            mock_sys.exit.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                twit_output.clean_folder(fake_dir)
        mock_sys.exit.assert_called_once_with(0)
        mock_print.assert_called()
        assert "does not exist" in str(mock_print.call_args)

    def test_folder_exists_with_files(self, tmp_path, monkeypatch):
        import twit_output
        out_dir = tmp_path / "summaries"
        out_dir.mkdir()
        (out_dir / "old1.md").write_text("old1")
        (out_dir / "old2.md").write_text("old2")
        (out_dir / "other.txt").write_text("other")  # not .md, should be kept
        with patch.object(twit_output, "sys") as mock_sys:
            mock_sys.exit.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                twit_output.clean_folder(out_dir)
        assert not (out_dir / "old1.md").exists()
        assert not (out_dir / "old2.md").exists()
        assert (out_dir / "other.txt").exists()
        mock_sys.exit.assert_called_once_with(0)

    def test_delete_failure_warning(self, tmp_path, monkeypatch):
        import twit_output
        out_dir = tmp_path / "summaries"
        out_dir.mkdir()
        md = out_dir / "locked.md"
        md.write_text("locked")
        with patch.object(Path, "unlink", side_effect=OSError("locked")):
            with patch.object(twit_output, "sys") as mock_sys:
                mock_sys.exit.side_effect = SystemExit(0)
                with patch("builtins.print") as mock_print:
                    with pytest.raises(SystemExit):
                        twit_output.clean_folder(out_dir)
        mock_sys.exit.assert_called_once_with(0)
        # Find the warning call (one for each failed delete)
        warn_calls = [c for c in mock_print.call_args_list if "Failed" in str(c)]
        assert len(warn_calls) == 1
        # Verify it mentions the file that failed
        assert "locked.md" in str(warn_calls[0])
