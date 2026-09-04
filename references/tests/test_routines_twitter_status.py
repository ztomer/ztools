"""The Twitter status adapter and its dashboard tab read back one summary file.

It has a drift regression baked in: summaries were written once as
`{since}_to_{until}.md` and today as `{when}_HHMM_summary.md`, but discovery
used to glob `tw_*.md` — which matches NEITHER — so the status answered
"no summary found" on a directory full of them. Discovery sorts every `*.md`
by mtime instead of pinning today's prefix.

The tab (`show_newest_twitter.py`) must stay READ-ONLY AND CHEAP: it cats one
file. Anything that imports the `twitter` package (browser/playwright chain)
fails that contract and is not a tab command.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import routines_twitter_status as rts

REAL_NAMES = [
    "chrissugar_to_mewdeus.md",  # the old {since}_to_{until}.md generation
    "2026-08-29_0851_summary.md",  # the current {when}_HHMM_summary.md generation
]
NEWEST = "2026-08-29_0851_summary.md"
HELPERS = REPO_ROOT / "references"


def write(directory: Path, name: str, body: str = "**Tweets:** 3 high-signal") -> Path:
    p = directory / name
    p.write_text(body, encoding="utf-8")
    return p


def made_newer(newest: Path, older: Path) -> None:
    os.utime(newest, (older.stat().st_mtime + 5,) * 2)


def test_old_glob_matches_nothing_on_real_names(tmp_path):
    for name in REAL_NAMES:
        write(tmp_path, name)
    # Sanity that the fixture names really are the ones the old glob missed.
    assert len(list(tmp_path.glob("tw_*.md"))) == 0
    assert len(list(tmp_path.glob("*.md"))) == 2


def test_newest_summary_picks_newest_by_mtime(tmp_path):
    older = write(tmp_path, REAL_NAMES[0])
    newest = write(tmp_path, NEWEST)
    made_newer(newest, older)
    assert rts.newest_summary(tmp_path) == newest


def test_newest_summary_none_when_empty_or_non_markdown(tmp_path):
    assert rts.newest_summary(tmp_path) is None
    write(tmp_path, "2026-08-29_0851_summary.txt")
    assert rts.newest_summary(tmp_path) is None


def test_status_ok_and_reports_the_newest_file(tmp_path, monkeypatch):
    write(tmp_path, REAL_NAMES[0])
    newest = write(tmp_path, NEWEST, "**Tweets:** 3 high-signal")
    made_newer(newest, newest.parent / REAL_NAMES[0])
    monkeypatch.setenv("TWITTER_OUTPUT_DIR", str(tmp_path))
    status = rts.build_status()
    assert status["state"] == "ok"
    assert NEWEST.removesuffix(".md") in status["summary"]
    assert "3 high-signal" in status["summary"]


def test_show_newest_twitter_tab_prints_the_summary(tmp_path):
    older = write(tmp_path, REAL_NAMES[0])
    newest = write(tmp_path, NEWEST, "**Tweets:** 3 high-signal")
    made_newer(newest, older)
    proc = subprocess.run(
        [sys.executable, str(HELPERS / "show_newest_twitter.py")],
        capture_output=True,
        text=True,
        env={**os.environ, "TWITTER_OUTPUT_DIR": str(tmp_path)},
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "**Tweets:** 3 high-signal"


def test_show_newest_twitter_tab_fails_statedly_when_no_summary(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(HELPERS / "show_newest_twitter.py")],
        capture_output=True,
        text=True,
        env={**os.environ, "TWITTER_OUTPUT_DIR": str(tmp_path)},
        cwd=tmp_path,
    )
    assert proc.returncode == 1
    assert "No twitter summary" in proc.stderr
    assert proc.stdout.strip() == ""


def test_show_newest_plan_tab_prints_the_newest_plan(tmp_path):
    older = tmp_path / "weekend_plan_August_08.md"
    older.write_text("# Weekend (old)", encoding="utf-8")
    newest = tmp_path / "weekend_plan_August_15.md"
    newest.write_text("# Weekend (new)", encoding="utf-8")
    made_newer(newest, older)
    proc = subprocess.run(
        [sys.executable, str(HELPERS / "show_newest_plan.py")],
        capture_output=True,
        text=True,
        env={**os.environ, "WEEKEND_OUTPUT_DIR": str(tmp_path)},
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "# Weekend (new)"


def test_show_newest_plan_tab_fails_statedly_when_no_plan(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(HELPERS / "show_newest_plan.py")],
        capture_output=True,
        text=True,
        env={**os.environ, "WEEKEND_OUTPUT_DIR": str(tmp_path)},
        cwd=tmp_path,
    )
    assert proc.returncode == 1
    assert "No weekend plan" in proc.stderr