"""Tests for the rename.cli main() entry point (argparse, model resolution, stats).

Split out of test_image_renamer.py to keep both files under the 500-line limit;
rename_image() unit tests stay there.
"""

from unittest.mock import MagicMock, patch

import pytest


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

    def test_main_vlm_defaults_to_configured_vlm(self, tmp_path, capsys):
        """Without --vlm-model, the configured best_models["vlm"] wins over --model.

        --model names the *text* model; it is not necessarily a vision model, so it
        must not override the configured VLM (rename/cli.py resolution order).
        """
        from rename.cli import main

        (tmp_path / "a.png").write_text("fake")
        with (
            patch("sys.argv", ["rename", str(tmp_path), "--dry-run", "--model", "my-test-model"]),
            patch("rename.cli.check_llm_availability", return_value=True),
            patch("rename.cli.get_best_models", return_value={"vlm": "configured-vlm"}),
            patch("rename.cli.rename_image", return_value=(True, "")) as mock_rename,
        ):
            main()
        call_args = mock_rename.call_args
        assert call_args.kwargs.get("vlm_model") == "configured-vlm"
        assert call_args.kwargs.get("llm_model") == "my-test-model"

    def test_main_vlm_falls_back_to_model_when_unconfigured(self, tmp_path, capsys):
        """With no --vlm-model and no configured VLM, vlm_model falls back to --model."""
        from rename.cli import main

        (tmp_path / "a.png").write_text("fake")
        with (
            patch("sys.argv", ["rename", str(tmp_path), "--dry-run", "--model", "my-test-model"]),
            patch("rename.cli.check_llm_availability", return_value=True),
            patch("rename.cli.get_best_models", return_value={}),
            patch("rename.cli.rename_image", return_value=(True, "")) as mock_rename,
        ):
            main()
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
            patch("rename.cli.get_best_models", return_value={"vlm": "configured-vlm"}),
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
