"""Cargo.lock and the dependency graph must hold exactly one version of miniz_oxide.

Owner ruling OW.30: the dependency graph must hold exactly ONE miniz_oxide.
`png` previously pinned an older miniz_oxide (0.8.x) while `flate2` used 0.9.x,
resolving two versions into `rust/Cargo.lock`. Upstream `image-rs/image-png`
removed the unused `miniz_oxide` dependency in commit b2f10c6a7e, unified via
`[patch.crates-io]`.

This gate fails if a second `miniz_oxide` version enters `rust/Cargo.lock` or if
`cargo tree -i miniz_oxide --workspace` becomes ambiguous.
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CARGO_LOCK = REPO_ROOT / "rust" / "Cargo.lock"


def miniz_oxide_versions_in_lock(lock_text: str) -> list[str]:
    data = tomllib.loads(lock_text)
    return [pkg["version"] for pkg in data.get("package", []) if pkg.get("name") == "miniz_oxide"]


def test_cargo_lock_holds_exactly_one_miniz_oxide() -> None:
    versions = miniz_oxide_versions_in_lock(CARGO_LOCK.read_text(encoding="utf-8"))
    assert len(versions) == 1, (
        f"Owner ruling OW.30: {CARGO_LOCK.relative_to(REPO_ROOT)} must hold exactly "
        f"ONE miniz_oxide, found {len(versions)}: {versions}"
    )


def test_extra_miniz_oxide_version_fails_calibration() -> None:
    """The check must fail if a second version is introduced."""
    sample_lock = (
        '[[package]]\nname = "miniz_oxide"\nversion = "0.8.9"\n\n'
        '[[package]]\nname = "miniz_oxide"\nversion = "0.9.1"\n'
    )
    versions = miniz_oxide_versions_in_lock(sample_lock)
    assert len(versions) == 2


def test_cargo_tree_miniz_oxide_single_version() -> None:
    """`cargo tree -i miniz_oxide --workspace` succeeds only when miniz_oxide is unambiguous."""
    proc = subprocess.run(
        ["cargo", "tree", "-i", "miniz_oxide", "--workspace"],
        cwd=REPO_ROOT / "rust",
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 0, (
        f"cargo tree -i miniz_oxide --workspace failed (exit {proc.returncode}):\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "miniz_oxide v" in proc.stdout
