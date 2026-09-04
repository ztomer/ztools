"""
Tests for the CLI UX improvements:
  - friendly server-down handling (check_server_or_die)
  - rename_images dry-run-by-default / --apply
Validated against a mock Osaurus server (no real LLM required).
"""

import sys
from unittest.mock import patch

from lib.osaurus_server import check_server_or_die


def monkeypatch_argv(argv):
    sys.argv = ["rename_images", *argv]


class _Exit(Exception):
    pass


def _run_server_check(url, capsys):
    with patch.object(sys, "exit", side_effect=_Exit) as exit_mock:
        try:
            check_server_or_die(url)
        except _Exit:
            pass
    out = capsys.readouterr().out
    return exit_mock, out


def test_server_up_passes(mock_osaurus_server, capsys):
    exit_mock, out = _run_server_check(mock_osaurus_server["up"], capsys)
    assert not exit_mock.called
    assert "not reachable" not in out


def test_server_down_dies_with_guidance(mock_osaurus_server, capsys):
    exit_mock, out = _run_server_check(mock_osaurus_server["down"], capsys)
    assert exit_mock.called
    assert "Osaurus server not reachable" in out
    assert "brew install --cask osaurus" in out
    # Points at osaurus_one.sh, NOT at `osaurus serve &`, which this assertion used
    # to pin. A hand-started server checks for no existing one and takes no GPU
    # lock, so following that advice while another agent session is mid-eval
    # produces two servers on a machine sized for one -- contention the sample
    # guard cannot see, since it reads swap and compressor and not the GPU.
    assert "./tools/osaurus_one.sh" in out
    assert "never start one by hand" in out


def test_rename_dry_run_is_default(mock_osaurus_server, capsys, tmp_path):
    """Without --apply, rename_images must preview only (no files renamed)."""
    import rename.cli as rc

    img = tmp_path / "screenshot_001.png"
    img.write_text("fake image")

    with (
        patch("lib.osaurus_server.check_server_or_die"),
        patch.object(rc, "check_llm_availability", return_value=True),
        patch.object(rc, "extract_full_text", return_value=""),
        patch.object(rc, "extract_first_line", return_value=""),
        patch.object(rc, "query_vlm_for_filename", return_value="holiday_photo"),
        patch.object(rc, "get_best_models", return_value={"vlm": "foundation"}),
    ):
        monkeypatch_argv([str(tmp_path)])
        rc.main()

    assert img.exists()
    assert not (tmp_path / "holiday_photo.png").exists()


def test_rename_apply_renames(mock_osaurus_server, capsys, tmp_path):
    import rename.cli as rc

    img = tmp_path / "screenshot_001.png"
    img.write_text("fake image")

    with (
        patch("lib.osaurus_server.check_server_or_die"),
        patch.object(rc, "check_llm_availability", return_value=True),
        patch.object(rc, "extract_full_text", return_value=""),
        patch.object(rc, "extract_first_line", return_value=""),
        patch.object(rc, "query_vlm_for_filename", return_value="holiday_photo"),
        patch.object(rc, "get_best_models", return_value={"vlm": "foundation"}),
    ):
        monkeypatch_argv([str(tmp_path), "--apply"])
        rc.main()

    assert not img.exists()
    assert (tmp_path / "holiday_photo.png").exists()


def test_rename_host_alias(mock_osaurus_server):
    import rename.cli as rc

    monkeypatch_argv(["--host", mock_osaurus_server["up"]])
    args = rc.parse_args()
    assert args.host == mock_osaurus_server["up"]
