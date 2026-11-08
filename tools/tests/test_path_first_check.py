"""The release ends by proving the FIRST `ztools` on PATH is the one just released (ZT.SHADOW).

A stale ~/.local/bin/ztools 3.12.0 came before /opt/homebrew/bin in PATH and ran for
every bare `ztools` after 3.12.1 shipped via brew. tools/path_first_check.sh is what
release.sh runs after `brew upgrade`; it must name every ztools on PATH, with its version.
"""

import subprocess
from pathlib import Path

CHECK = Path(__file__).resolve().parent.parent / "path_first_check.sh"


def _fake(d: Path, version: str) -> Path:
    d.mkdir(parents=True)
    b = d / "ztools"
    b.write_text(f"#!/bin/sh\necho 'ztools {version}'\n")
    b.chmod(0o755)
    return d


def _run(path, version):
    return subprocess.run(
        ["/bin/bash", str(CHECK), version],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        stdin=subprocess.DEVNULL,
        env={"PATH": f"{path}:/usr/bin:/bin", "NO_COLOR": "1"},
    )


def test_stale_copy_first_on_path_fails_naming_both(tmp_path):
    local = _fake(tmp_path / "local", "3.12.0")
    brew = _fake(tmp_path / "brew", "3.12.1")
    r = _run(f"{local}:{brew}", "3.12.1")
    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    for s in (str(local / "ztools"), "3.12.0", str(brew / "ztools"), "3.12.1"):
        assert s in out


def test_second_copy_on_path_fails_even_when_first_is_right(tmp_path):
    brew = _fake(tmp_path / "brew", "3.12.1")
    local = _fake(tmp_path / "local", "3.12.0")
    r = _run(f"{brew}:{local}", "3.12.1")
    assert r.returncode != 0
    assert str(local / "ztools") in r.stdout + r.stderr


def test_only_the_released_copy_passes(tmp_path):
    brew = _fake(tmp_path / "brew", "3.12.1")
    r = _run(str(brew), "3.12.1")
    assert r.returncode == 0, r.stdout + r.stderr


def test_no_ztools_on_path_fails(tmp_path):
    (tmp_path / "empty").mkdir()
    r = _run(str(tmp_path / "empty"), "3.12.1")
    assert r.returncode != 0
