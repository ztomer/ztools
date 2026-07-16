"""Tests for rename.cli rename_image function."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_path(name="test_image.png"):
    """Create a Path that exists (avoids filesystem checks)."""
    p = MagicMock(spec=Path)
    p.name = name
    p.suffix = ".png"
    p.exists.return_value = True
    p.with_name.side_effect = lambda n: Path(f"/fake/{n}.png")
    return p


class TestRenameImageFileNotExists:
    def test_returns_false_when_missing(self):
        from rename.cli import rename_image

        p = MagicMock(spec=Path)
        p.name = "missing.png"
        p.exists.return_value = False
        success, msg = rename_image(
            p,
            dry_run=True,
            force=False,
            llm_host="",
            llm_model="",
            vlm_model="",
            api_key="",
        )
        assert success is False
        assert "File not found" in msg


class TestRenameImageNoText:
    def test_falls_back_to_vlm_when_no_text(self):
        from rename.cli import rename_image

        p = _make_path()
        with (
            patch("rename.cli.extract_full_text", return_value=None),
            patch("rename.cli.extract_first_line", return_value=None),
            patch("rename.cli.query_vlm_for_filename", return_value="white_goose") as mock_vlm,
            patch("rename.cli.clean_filename", side_effect=lambda x: x),
        ):
            success, msg = rename_image(
                p,
                dry_run=True,
                force=False,
                llm_host="http://localhost:1337",
                llm_model="test-model",
                vlm_model="vlm-model",
                api_key="",
            )
            assert success is True
            mock_vlm.assert_called_once()

    def test_skipped_when_no_text_and_no_vlm(self):
        from rename.cli import rename_image

        p = _make_path()
        with (
            patch("rename.cli.extract_full_text", return_value=None),
            patch("rename.cli.extract_first_line", return_value=None),
        ):
            success, msg = rename_image(
                p,
                dry_run=True,
                force=False,
                llm_host="http://localhost:1337",
                llm_model="test-model",
                vlm_model="",
                api_key="",
            )
            assert success is False
            assert "No text & no VLM fallback" in msg


class TestRenameImageVlmFallback:
    def test_vlm_used_when_text_is_non_human_readable(self):
        from rename.cli import rename_image

        p = _make_path()
        with (
            patch("rename.cli.extract_full_text", return_value="HFSd9k2Lm"),
            patch("rename.cli.is_non_human_readable", return_value=True),
            patch("rename.cli.query_vlm_for_filename", return_value="vlm_name") as mock_vlm,
            patch("rename.cli.clean_filename", side_effect=lambda x: x),
        ):
            success, msg = rename_image(
                p,
                dry_run=True,
                force=False,
                llm_host="http://localhost:1337",
                llm_model="test",
                vlm_model="vlm",
                api_key="",
            )
            assert success is True
            mock_vlm.assert_called_once()

    def test_vlm_used_when_text_not_meaningful(self):
        from rename.cli import rename_image

        p = _make_path()
        with (
            patch("rename.cli.extract_full_text", return_value="ab cd"),
            patch("rename.cli.is_non_human_readable", return_value=False),
            patch("rename.cli.is_meaningful_text", return_value=False),
            patch("rename.cli.query_vlm_for_filename", return_value="vlm_result"),
            patch("rename.cli.clean_filename", side_effect=lambda x: x),
        ):
            success, msg = rename_image(
                p,
                dry_run=True,
                force=False,
                llm_host="http://localhost:1337",
                llm_model="test",
                vlm_model="vlm",
                api_key="",
            )
            assert success is True

    def test_vlm_generic_name_rejected(self):
        from rename.cli import rename_image

        p = _make_path()
        with (
            patch("rename.cli.extract_full_text", return_value=None),
            patch("rename.cli.extract_first_line", return_value=None),
            patch("rename.cli.query_vlm_for_filename", return_value="unnamed"),
            patch("rename.cli.clean_filename", side_effect=lambda x: x),
        ):
            success, msg = rename_image(
                p,
                dry_run=True,
                force=False,
                llm_host="http://localhost:1337",
                llm_model="test",
                vlm_model="vlm",
                api_key="",
            )
            assert success is False
            assert "Could not generate name" in msg


class TestRenameImageLlmPath:
    def test_force_path_checks_relevance(self):
        from rename.cli import rename_image

        p = _make_path()
        with (
            patch("rename.cli.extract_full_text", return_value="some useful text"),
            patch("rename.cli.is_non_human_readable", return_value=False),
            patch("rename.cli.is_meaningful_text", return_value=True),
            patch("rename.cli.is_relevant_with_llm", return_value=True),
            patch("rename.cli.query_llm_for_filename", return_value="relevant_name"),
            patch("rename.cli.clean_filename", side_effect=lambda x: x),
        ):
            success, msg = rename_image(
                p,
                dry_run=True,
                force=True,
                llm_host="http://localhost:1337",
                llm_model="test",
                vlm_model="",
                api_key="",
            )
            assert success is True

    def test_force_path_skips_not_relevant(self):
        from rename.cli import rename_image

        p = _make_path()
        with (
            patch("rename.cli.extract_full_text", return_value="junk text"),
            patch("rename.cli.is_non_human_readable", return_value=False),
            patch("rename.cli.is_meaningful_text", return_value=True),
            patch("rename.cli.is_relevant_with_llm", return_value=False),
        ):
            success, msg = rename_image(
                p,
                dry_run=True,
                force=True,
                llm_host="http://localhost:1337",
                llm_model="test",
                vlm_model="",
                api_key="",
            )
            assert success is False
            assert "Not relevant" in msg

    def test_llm_successful_rename(self):
        from rename.cli import rename_image

        p = _make_path()
        with (
            patch("rename.cli.extract_full_text", return_value="some text"),
            patch("rename.cli.is_non_human_readable", return_value=False),
            patch("rename.cli.is_meaningful_text", return_value=True),
            patch("rename.cli.query_llm_for_filename", return_value="llm_filename"),
            patch("rename.cli.clean_filename", side_effect=lambda x: x),
        ):
            success, msg = rename_image(
                p,
                dry_run=True,
                force=False,
                llm_host="http://localhost:1337",
                llm_model="test",
                vlm_model="",
                api_key="",
            )
            assert success is True

    def test_llm_generic_name_rejected(self):
        from rename.cli import rename_image

        p = _make_path()
        with (
            patch("rename.cli.extract_full_text", return_value="some text"),
            patch("rename.cli.is_non_human_readable", return_value=False),
            patch("rename.cli.is_meaningful_text", return_value=True),
            patch("rename.cli.query_llm_for_filename", return_value="unnamed"),
            patch("rename.cli.clean_filename", side_effect=lambda x: x),
        ):
            success, msg = rename_image(
                p,
                dry_run=True,
                force=False,
                llm_host="http://localhost:1337",
                llm_model="test",
                vlm_model="",
                api_key="",
            )
            assert success is True  # falls back to clean_filename
            assert msg == ""

    def test_llm_too_short_name_rejected(self):
        from rename.cli import rename_image

        p = _make_path()
        with (
            patch("rename.cli.extract_full_text", return_value="some text"),
            patch("rename.cli.is_non_human_readable", return_value=False),
            patch("rename.cli.is_meaningful_text", return_value=True),
            patch("rename.cli.query_llm_for_filename", return_value="ab"),
            patch("rename.cli.clean_filename", side_effect=lambda x: x),
        ):
            success, msg = rename_image(
                p,
                dry_run=True,
                force=False,
                llm_host="http://localhost:1337",
                llm_model="test",
                vlm_model="",
                api_key="",
            )
            assert success is True  # falls back to clean_filename of original text

    def test_llm_returns_none_falls_back_to_text(self):
        from rename.cli import rename_image

        p = _make_path()
        with (
            patch("rename.cli.extract_full_text", return_value="useful text content"),
            patch("rename.cli.is_non_human_readable", return_value=False),
            patch("rename.cli.is_meaningful_text", return_value=True),
            patch("rename.cli.query_llm_for_filename", return_value=None),
            patch("rename.cli.clean_filename", side_effect=lambda x: x),
        ):
            success, msg = rename_image(
                p,
                dry_run=True,
                force=False,
                llm_host="http://localhost:1337",
                llm_model="test",
                vlm_model="",
                api_key="",
            )
            assert success is True

    def test_no_llm_host_and_no_model_returns_error(self):
        from rename.cli import rename_image

        p = _make_path()
        with (
            patch("rename.cli.extract_full_text", return_value="useful text"),
            patch("rename.cli.is_non_human_readable", return_value=False),
            patch("rename.cli.is_meaningful_text", return_value=True),
        ):
            success, msg = rename_image(
                p, dry_run=True, force=False, llm_host="", llm_model="", vlm_model="", api_key=""
            )
            assert success is False
            assert "Could not generate name" in msg


class TestRenameImageFileOperations:
    def test_dry_run_does_not_rename(self):
        from rename.cli import rename_image

        p = MagicMock(spec=Path)
        p.name = "test.png"
        p.suffix = ".png"
        p.exists.return_value = True
        p.with_name.side_effect = lambda n: Path(f"/fake/{n}.png")
        with (
            patch("rename.cli.extract_full_text", return_value="filename"),
            patch("rename.cli.is_non_human_readable", return_value=False),
            patch("rename.cli.is_meaningful_text", return_value=True),
            patch("rename.cli.query_llm_for_filename", return_value="new_name"),
            patch("rename.cli.clean_filename", side_effect=lambda x: x),
        ):
            p.rename.assert_not_called()  # not called yet
            success, msg = rename_image(
                p,
                dry_run=True,
                force=False,
                llm_host="http://localhost:1337",
                llm_model="test",
                vlm_model="",
                api_key="",
            )
            assert success is True
            p.rename.assert_not_called()

    def test_live_run_calls_rename(self):
        from rename.cli import rename_image

        p = MagicMock(spec=Path)
        p.name = "test.png"
        p.suffix = ".png"
        p.exists.return_value = True
        p.with_name.side_effect = lambda n: Path(f"/fake/{n}.png")
        with (
            patch("rename.cli.extract_full_text", return_value="filename"),
            patch("rename.cli.is_non_human_readable", return_value=False),
            patch("rename.cli.is_meaningful_text", return_value=True),
            patch("rename.cli.query_llm_for_filename", return_value="new_name"),
            patch("rename.cli.clean_filename", side_effect=lambda x: x),
        ):
            success, msg = rename_image(
                p,
                dry_run=False,
                force=False,
                llm_host="http://localhost:1337",
                llm_model="test",
                vlm_model="",
                api_key="",
            )
            assert success is True
            p.rename.assert_called_once()

    def test_rename_exception_reported(self):
        from rename.cli import rename_image

        p = MagicMock(spec=Path)
        p.name = "test.png"
        p.suffix = ".png"
        p.exists.return_value = True
        p.with_name.side_effect = lambda n: Path(f"/fake/{n}.png")
        p.rename.side_effect = PermissionError("permission denied")
        with (
            patch("rename.cli.extract_full_text", return_value="filename"),
            patch("rename.cli.is_non_human_readable", return_value=False),
            patch("rename.cli.is_meaningful_text", return_value=True),
            patch("rename.cli.query_llm_for_filename", return_value="new_name"),
            patch("rename.cli.clean_filename", side_effect=lambda x: x),
        ):
            success, msg = rename_image(
                p,
                dry_run=False,
                force=False,
                llm_host="http://localhost:1337",
                llm_model="test",
                vlm_model="",
                api_key="",
            )
            assert success is False
            assert "Error renaming" in msg

    def test_generic_final_name_rejected(self):
        from rename.cli import rename_image

        p = _make_path()
        with (
            patch("rename.cli.extract_full_text", return_value="file txt"),
            patch("rename.cli.is_non_human_readable", return_value=False),
            patch("rename.cli.is_meaningful_text", return_value=True),
            patch("rename.cli.query_llm_for_filename", return_value="file_txt"),
        ):
            success, msg = rename_image(
                p,
                dry_run=True,
                force=False,
                llm_host="http://localhost:1337",
                llm_model="test",
                vlm_model="",
                api_key="",
            )
            assert success is False
            assert "Generic name" in msg


class TestRenameImageLlmException:
    def test_llm_exception_falls_back_to_text(self):
        from rename.cli import rename_image

        p = _make_path()
        with (
            patch("rename.cli.extract_full_text", return_value="fallback text content"),
            patch("rename.cli.is_non_human_readable", return_value=False),
            patch("rename.cli.is_meaningful_text", return_value=True),
            patch("rename.cli.query_llm_for_filename", side_effect=Exception("LLM down")),
            patch("rename.cli.clean_filename", side_effect=lambda x: x),
        ):
            success, msg = rename_image(
                p,
                dry_run=True,
                force=False,
                llm_host="http://localhost:1337",
                llm_model="test",
                vlm_model="",
                api_key="",
            )
            assert success is True


class TestRenameImageDuplicates:
    def test_too_many_duplicates_returns_error(self):
        from rename.cli import rename_image

        p = MagicMock(spec=Path)
        p.name = "test.png"
        p.suffix = ".png"
        p.exists.return_value = True

        # Make with_name return paths that always "exist"
        counter = [0]

        def make_new_path(name):
            counter[0] += 1
            new_p = MagicMock(spec=Path)
            new_p.name = name
            new_p.exists.return_value = True
            return new_p

        p.with_name.side_effect = make_new_path

        with (
            patch("rename.cli.extract_full_text", return_value="filename"),
            patch("rename.cli.is_non_human_readable", return_value=False),
            patch("rename.cli.is_meaningful_text", return_value=True),
            patch("rename.cli.query_llm_for_filename", return_value="new_name"),
            patch("rename.cli.clean_filename", side_effect=lambda x: x),
        ):
            success, msg = rename_image(
                p,
                dry_run=True,
                force=False,
                llm_host="http://localhost:1337",
                llm_model="test",
                vlm_model="",
                api_key="",
            )
            assert success is False
            assert "Too many duplicates" in msg


class TestRenameImageMain:
    def _setup_args(self, **overrides):
        args = MagicMock()
        args.directory = "."
        args.dry_run = False
        args.force = False
        args.pattern = "*"
        args.max_length = 50
        args.llm_host = "http://localhost:1337"
        args.llm_model = "test-model"
        args.vlm_model = ""
        args.api_key = ""
        args.test = False
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def test_main_invalid_directory(self, capsys):
        from rename.cli import main

        with patch("sys.argv", ["rename", "/nonexistent/dir/xyz"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
        out = capsys.readouterr()
        assert "Invalid directory" in out.out or "Invalid directory" in out.err

    def test_main_no_images_found(self, tmp_path, capsys):
        from rename.cli import main

        with (
            patch("sys.argv", ["rename", str(tmp_path)]),
            patch("rename.cli.check_llm_availability", return_value=True),
            patch("rename.cli.rename_image", return_value=(True, "")),
        ):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0
        out = capsys.readouterr()
        assert "No images found" in out.out or "No images found" in out.err

    def test_main_renames_images(self, tmp_path, capsys):
        from rename.cli import main

        # Create fake image files
        (tmp_path / "a.png").write_text("fake")
        (tmp_path / "b.png").write_text("fake")
        with (
            patch("sys.argv", ["rename", str(tmp_path), "--dry-run"]),
            patch("rename.cli.check_llm_availability", return_value=True),
            patch("rename.cli.rename_image", return_value=(True, "")),
        ):
            main()
        out = capsys.readouterr()
        assert "2 images" in out.out

    def test_main_llm_server_unavailable(self, tmp_path, capsys):
        from rename.cli import main

        with (
            patch("sys.argv", ["rename", str(tmp_path)]),
            patch("rename.cli.check_llm_availability", return_value=False),
            patch("lib.osaurus_server.is_server_running", return_value=False),
        ):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
        out = capsys.readouterr()
        assert "not reachable" in out.out or "not reachable" in out.err

    def test_main_skipped_messages_counted(self, tmp_path, capsys):
        from rename.cli import main

        (tmp_path / "a.png").write_text("fake")
        with (
            patch("sys.argv", ["rename", str(tmp_path), "--dry-run"]),
            patch("rename.cli.check_llm_availability", return_value=True),
            patch("rename.cli.rename_image", return_value=(False, "Skipped (X)")),
        ):
            main()
        out = capsys.readouterr()
        assert "1 skipped" in out.out

    def test_main_error_messages_counted(self, tmp_path, capsys):
        from rename.cli import main

        (tmp_path / "a.png").write_text("fake")
        with (
            patch("sys.argv", ["rename", str(tmp_path), "--dry-run"]),
            patch("rename.cli.check_llm_availability", return_value=True),
            patch("rename.cli.rename_image", return_value=(False, "Failed")),
        ):
            main()
        out = capsys.readouterr()
        assert "1 errors" in out.out

    def test_main_pattern_specific(self, tmp_path, capsys):
        from rename.cli import main

        (tmp_path / "a.png").write_text("fake")
        (tmp_path / "b.jpg").write_text("fake")
        with (
            patch("sys.argv", ["rename", str(tmp_path), "--pattern", "*.png", "--dry-run"]),
            patch("rename.cli.check_llm_availability", return_value=True),
            patch("rename.cli.rename_image", return_value=(True, "")),
        ):
            main()
        out = capsys.readouterr()
        # Only PNG file should be found
        assert "1 images" in out.out

    def test_main_vlm_default_to_model(self, tmp_path, capsys):
        from rename.cli import main

        (tmp_path / "a.png").write_text("fake")
        with (
            patch("sys.argv", ["rename", str(tmp_path), "--dry-run", "--model", "my-test-model"]),
            patch("rename.cli.check_llm_availability", return_value=True),
            patch("rename.cli.rename_image", return_value=(True, "")) as mock_rename,
        ):
            main()
        # vlm_model defaults to active_model when not specified
        call_args = mock_rename.call_args
        assert call_args.kwargs.get("vlm_model") == "my-test-model"

    def test_main_dry_run_renamed_message(self, tmp_path, capsys):
        from rename.cli import main

        (tmp_path / "a.png").write_text("fake")
        with (
            patch("sys.argv", ["rename", str(tmp_path), "--dry-run"]),
            patch("rename.cli.check_llm_availability", return_value=True),
            patch("rename.cli.rename_image", return_value=(True, "")),
        ):
            main()
        out = capsys.readouterr()
        # Dry run message
        assert "Dry run" in out.out

    def test_main_dry_run_invites_actual_run(self, tmp_path, capsys):
        from rename.cli import main

        (tmp_path / "a.png").write_text("fake")
        with (
            patch("sys.argv", ["rename", str(tmp_path), "--dry-run"]),
            patch("rename.cli.check_llm_availability", return_value=True),
            patch("rename.cli.rename_image", return_value=(True, "")),
        ):
            main()
        out = capsys.readouterr()
        assert "Pass --apply to rename" in out.out

    def test_main_print_message_on_skip(self, tmp_path, capsys):
        from rename.cli import main

        (tmp_path / "a.png").write_text("fake")
        with (
            patch("sys.argv", ["rename", str(tmp_path), "--dry-run"]),
            patch("rename.cli.check_llm_availability", return_value=True),
            patch("rename.cli.rename_image", return_value=(False, "Skipped (bad)")),
        ):
            main()
        out = capsys.readouterr()
        # Skipped messages are printed
        assert "Skipped (bad)" in out.out

    def test_main_print_message_on_error(self, tmp_path, capsys):
        from rename.cli import main

        (tmp_path / "a.png").write_text("fake")
        with (
            patch("sys.argv", ["rename", str(tmp_path), "--dry-run"]),
            patch("rename.cli.check_llm_availability", return_value=True),
            patch("rename.cli.rename_image", return_value=(False, "Some error")),
        ):
            main()
        out = capsys.readouterr()
        # Error messages are printed
        assert "Some error" in out.out

    def test_main_test_flag(self, tmp_path, capsys):
        from rename.cli import main

        (tmp_path / "a.png").write_text("fake")
        with (
            patch("sys.argv", ["rename", str(tmp_path), "--test", "--model", "my-test-model"]),
            patch("rename.cli.check_llm_availability", return_value=True),
            patch("rename.cli.rename_image", return_value=(True, "")),
        ):
            main()
        out = capsys.readouterr()
        # Test flag is parsed
        assert "my-test-model" in out.out

    def test_main_vlm_model_explicit(self, tmp_path, capsys):
        from rename.cli import main

        (tmp_path / "a.png").write_text("fake")
        with (
            patch("sys.argv", ["rename", str(tmp_path), "--vlm-model", "my-vlm", "--dry-run"]),
            patch("rename.cli.check_llm_availability", return_value=True),
            patch("rename.cli.rename_image", return_value=(True, "")) as mock_rename,
        ):
            main()
        call_args = mock_rename.call_args
        assert call_args.kwargs.get("vlm_model") == "my-vlm"

    def test_main_runs_as_script(self, tmp_path, capsys):
        import runpy

        (tmp_path / "a.png").write_text("fake")
        with (
            patch("sys.argv", ["rename", str(tmp_path), "--dry-run"]),
            patch("rename.cli.check_llm_availability", return_value=True),
            patch("rename.cli.rename_image", return_value=(True, "")),
        ):
            # Run the __main__ block
            runpy.run_module("rename", run_name="__main__")
        out = capsys.readouterr()
        assert "1 renamed" in out.out or "1 renamed" in out.err
