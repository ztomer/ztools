"""tools/release.sh --dry-run changes nothing.

The house release kit's contract is that --dry-run "prints the planned actions and
performs NOTHING". This wrapper syncs rust/Cargo.toml to the release version and
COMMITS that bump before it execs the kit -- and it did so on a dry run too, so
`tools/release.sh 3.8.1 --dry-run` (2026-10-18) left a local "chore: version 3.8.1"
commit behind while reporting itself a plan. A dry run that mutates the repo is not a
dry run: the operator previewing a release has, without being told, started one.

Both directions, on a throwaway repo holding a copy of the REAL script: the dry run
leaves the manifest and the history alone and says what it would bump; the real run
still bumps and commits. The kit itself is a stub that records its argv, so nothing
here can tag, push or release.
"""

import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent  # tools/tests/ -> repo root
RELEASE = REPO / "tools" / "release.sh"

#: The wrapper is a few git and sed calls. Bounded, so a regression that blocks is a
#: failure with a name rather than a hung suite.
TIMEOUT = 60

MANIFEST = '[package]\nname = "ztools"\nversion = "1.0.0"\n'


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
        env=_env(repo.parent),
        timeout=TIMEOUT,
    ).stdout.strip()


def _env(root):
    return {
        "PATH": f"{root / 'bin'}:/usr/bin:/bin",
        "HOME": str(root),
        "NO_COLOR": "1",
        "GOH_DIR": str(root / "goh"),
        # Isolated from the operator's git config: no global hooksPath, no signing.
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }


def _sandbox(tmp_path):
    """A repo with the real wrapper, a stub kit that logs argv, and a no-op cargo."""
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    (repo / "rust").mkdir()
    shutil.copy2(RELEASE, repo / "tools" / "release.sh")
    shutil.copy2(RELEASE.parent / "path_first_check.sh", repo / "tools" / "path_first_check.sh")
    (repo / "rust" / "Cargo.toml").write_text(MANIFEST)
    (repo / "rust" / "Cargo.lock").write_text("# lock\n")

    kit = tmp_path / "goh" / "tools" / "release-kit"
    kit.mkdir(parents=True)
    (kit / "release.sh").write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > {tmp_path / "kit.argv"}\n')
    (kit / "release.sh").chmod(0o755)

    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "cargo").write_text("#!/bin/bash\nexit 0\n")
    (bindir / "cargo").chmod(0o755)
    # After the kit, the wrapper upgrades the tap and checks the FIRST ztools on PATH
    # (ZT.SHADOW). brew logs its argv; the "installed" ztools is the release.
    (bindir / "brew").write_text(f'#!/bin/bash\nprintf "%s\\n" "$*" >> {tmp_path / "brew.argv"}\n')
    (bindir / "brew").chmod(0o755)
    (bindir / "ztools").write_text("#!/bin/sh\necho 'ztools 2.0.0'\n")
    (bindir / "ztools").chmod(0o755)

    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _release(repo, *args):
    return subprocess.run(
        ["/bin/bash", str(repo / "tools" / "release.sh"), *args],
        # Not an exception path: the exit code is asserted with the output beside it.
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=_env(repo.parent),
        cwd=str(repo),
        timeout=TIMEOUT,
    )


def test_dry_run_leaves_manifest_and_history_alone(tmp_path):
    repo = _sandbox(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")

    result = _release(repo, "2.0.0", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert (repo / "rust" / "Cargo.toml").read_text() == MANIFEST
    assert _git(repo, "rev-parse", "HEAD") == head, "the dry run committed"
    assert _git(repo, "status", "--porcelain") == "", "the dry run left the tree dirty"
    # It still says what the real run would do, and still hands the kit --dry-run.
    assert "1.0.0 → 2.0.0" in result.stderr
    assert "dry-run" in result.stderr
    argv = (tmp_path / "kit.argv").read_text().split("\n")
    assert "--dry-run" in argv
    assert not (tmp_path / "brew.argv").exists(), "the dry run touched brew"
    assert argv[argv.index("--version") + 1] == "2.0.0"


def test_real_run_still_bumps_and_commits(tmp_path):
    repo = _sandbox(tmp_path)

    result = _release(repo, "2.0.0")

    assert result.returncode == 0, result.stderr
    assert 'version = "2.0.0"' in (repo / "rust" / "Cargo.toml").read_text()
    assert _git(repo, "log", "-1", "--format=%s") == "chore: version 2.0.0"
    assert "--dry-run" not in (tmp_path / "kit.argv").read_text().split("\n")
    assert "upgrade ztomer/tap/ztools" in (tmp_path / "brew.argv").read_text()


def test_real_run_fails_when_a_stale_copy_shadows_the_tap(tmp_path):
    repo = _sandbox(tmp_path)
    (tmp_path / "bin" / "ztools").write_text("#!/bin/sh\necho 'ztools 1.0.0'\n")

    result = _release(repo, "2.0.0")

    assert result.returncode != 0
    assert "1.0.0" in result.stderr


def test_a_refused_bump_commit_leaves_the_index_and_tree_as_found(tmp_path):
    # ZT.RESIDFLAKE: a pre-commit hook refused the "chore: version" commit (the red_run
    # trailer check), release.sh exited 1, and the bump stayed STAGED for the operator to
    # discard by hand. A failed bump must leave the manifest, the lock and the index as
    # they were, and name what happened.
    repo = _sandbox(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\necho 'refused by the test hook' >&2\nexit 1\n")
    hook.chmod(0o755)

    result = _release(repo, "2.0.0")

    assert result.returncode != 0, result.stderr
    assert _git(repo, "rev-parse", "HEAD") == head
    assert _git(repo, "status", "--porcelain") == "", "the failed bump left changes behind"
    assert (repo / "rust" / "Cargo.toml").read_text() == MANIFEST
    assert "version bump" in result.stderr
    assert not (tmp_path / "kit.argv").exists(), "the kit ran after a failed bump"


def test_a_bump_refuses_a_manifest_with_uncommitted_changes(tmp_path):
    # Restoring the manifest on failure is only safe when nothing else lives in it, so a
    # bump onto local manifest edits is refused before anything is touched.
    repo = _sandbox(tmp_path)
    (repo / "rust" / "Cargo.lock").write_text("# lock, edited\n")

    result = _release(repo, "2.0.0")

    assert result.returncode != 0
    assert (repo / "rust" / "Cargo.lock").read_text() == "# lock, edited\n"
    assert (repo / "rust" / "Cargo.toml").read_text() == MANIFEST
    assert "uncommitted" in result.stderr
