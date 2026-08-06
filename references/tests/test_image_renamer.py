"""Tests for rename.cli rename_image function."""

from pathlib import Path
from unittest.mock import MagicMock, patch


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
