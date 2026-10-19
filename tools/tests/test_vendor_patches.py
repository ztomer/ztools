"""Local divergences from vendored upstream code survive a re-vendor, or the gate goes red.

`vendor/camoufox-rs` is third-party code carried in-tree and re-vendored
WHOLESALE when upstream moves (one sync rewrote 29 files). An edit made inside it
is therefore deleted by the next sync, silently: the build stays green, and the
defect the edit fixed comes back. The first such edit (2026-10-19) fixed a mock
test that ordered two events with a 10 ms sleep and flaked on busy CI runners.

So every local divergence lives as a patch under `vendor/patches/<crate>/`,
OUTSIDE the tree a sync replaces, and this test asserts each one is applied:
`git apply --reverse --check` succeeds only when the patch's result is what is on
disk. A re-vendor that forgets to re-apply goes red here and names the patch,
rather than reintroducing the flake weeks later as an unexplained CI failure.

Calibrated both ways below: a patch that is NOT applied must be reported.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATCHES = ROOT / "vendor" / "patches"


def patch_files() -> list[Path]:
    return sorted(PATCHES.glob("*/*.patch"))


def is_applied(patch: Path, cwd: Path) -> bool:
    """True when the tree at `cwd` already carries `patch`'s result."""
    proc = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def test_there_is_something_to_check() -> None:
    # Empty-scope guard: a glob that matched nothing would pass every patch.
    assert patch_files(), f"no patches found under {PATCHES}"


@pytest.mark.parametrize("patch", patch_files(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_vendor_patch_is_applied(patch: Path) -> None:
    assert is_applied(patch, ROOT), (
        f"{patch.relative_to(ROOT)} is not applied to the vendored tree. A re-vendor "
        f"replaced the crate without re-applying its local patches; run "
        f"`git apply {patch.relative_to(ROOT)}` (see {patch.parent.relative_to(ROOT)}/README.md)."
    )


@pytest.mark.parametrize("patch", patch_files(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_patch_dir_explains_itself(patch: Path) -> None:
    readme = patch.parent / "README.md"
    assert readme.is_file(), f"{readme.relative_to(ROOT)} is missing"
    assert patch.name in readme.read_text(encoding="utf-8"), (
        f"{readme.relative_to(ROOT)} does not name {patch.name}: every divergence carries its why"
    )


def test_an_unapplied_patch_is_reported(tmp_path: Path) -> None:
    """Calibration: the instrument can say NO. A fresh repo holding the patch's
    PRE-image (the upstream file) must read as not-applied."""
    patch = patch_files()[0]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, stdin=subprocess.DEVNULL)
    # Reconstruct the upstream (pre-patch) file by reverse-applying into a copy.
    lines = patch.read_text().splitlines()
    for target in {line[6:].strip() for line in lines if line.startswith("+++ b/")}:
        dest = tmp_path / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / target).read_bytes())
    subprocess.run(
        ["git", "apply", "--reverse", str(patch)],
        cwd=tmp_path,
        check=True,
        stdin=subprocess.DEVNULL,
    )
    assert not is_applied(patch, tmp_path)
    # ...and the same copy with the patch re-applied reads as applied.
    subprocess.run(["git", "apply", str(patch)], cwd=tmp_path, check=True, stdin=subprocess.DEVNULL)
    assert is_applied(patch, tmp_path)
