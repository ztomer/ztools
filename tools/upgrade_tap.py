#!/usr/bin/env python3
"""
Automate updating the Homebrew formula for ztools in the homebrew-tap repository.

THE TOKEN. `--token`, or `HOMEBREW_TAP_TOKEN` in the environment. Prefer the
environment variable: a command-line argument is visible to every process
listing on the machine for as long as the command runs. The token is never put
in a URL or in argv -- the clone uses a credential-free URL plus a GIT_ASKPASS
helper that reads the token out of the environment at call time, so it cannot
land in this process's argv or in the temp clone's `.git/config` either.

WHICH RELEASE PATH THIS IS NOT. There is one tap path in this repo and it is
not this script: `tools/release.sh` hands the tap bump to the house release kit
(`gates_of_heck/tools/release-kit/release.sh`, step "tap"), which rewrites the
formula's url+sha256 in place and refuses to create a formula that is missing.
This script is the manual/local door for that same edit. Verified 2026-10-04:
nothing in the repo invokes it.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FORMULA_NAME = "ztools.rb"
TAP_REPO = "ztomer/homebrew-tap"


def print_info(message):
    print(f"[ ==> ] {message}")


def print_err(message):
    print(f"[ Err ] {message}")


def print_ok(message):
    print(f"[ Ok  ] {message}")


def update_formula_content(file_path: Path, version: str, sha256: str) -> bool:
    if not file_path.exists():
        print_err(f"Formula file not found: {file_path}")
        return False

    content = file_path.read_text()

    # The URL must name the artifact the checksum was computed over. The live tap
    # path rewrites url+sha256 together, to GitHub's auto-generated tag archive:
    # the house release kit (gates_of_heck/tools/release-kit/release.sh, step
    # "tap") derives the archive URL from the tag it just pushed and hashes that
    # same download, so it writes the pair this function writes and the two cannot
    # fight over which file is canonical. There is no second writer: this repo has
    # no .github/ directory and no workflow (verified 2026-10-04), so an earlier
    # version of this comment citing `.github/workflows/release.yml` named a path
    # that does not exist.
    release_url = f"https://github.com/ztomer/ztools/archive/refs/tags/v{version}.tar.gz"
    url_pattern = r'(url\s+)"https://github.com/ztomer/ztools/[^"]+"'
    sha_pattern = r'(sha256\s+)"[0-9a-fA-F]{64}"'

    new_url = f'\\1"{release_url}"'
    new_sha = f'\\1"{sha256}"'

    updated_content, url_count = re.subn(url_pattern, new_url, content)
    updated_content, sha_count = re.subn(sha_pattern, new_sha, updated_content)

    if url_count == 0 or sha_count == 0:
        print_info("Standard URL/SHA256 patterns not matched. Retrying with generic patterns...")
        updated_content = re.sub(r'url\s+"[^"]+"', f'url "{release_url}"', content)
        updated_content = re.sub(r'sha256\s+"[^"]+"', f'sha256 "{sha256}"', updated_content)

    file_path.write_text(updated_content)
    print_ok(f"Updated {file_path.name} with version v{version} and sha256 {sha256}")
    return True


def run_cmd(args, cwd=None, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


#: The subcommand aliases the binary answers to, mirroring the symlink list in
#: install.sh -- one source of truth for what a `ztools` install puts on PATH.
SUBCOMMAND_ALIASES = (
    "twitter",
    "twitter-summarize",
    "weekend",
    "weekend-plan",
    "rename_images",
    "image-renamer",
    "oeval",
    "model-eval",
)

# A formula that builds THIS repo: the archive is a source tarball, the product
# is a Rust binary, and nothing on the runtime path runs an interpreter. The
# previous template declared `depends_on "python@3.12"` and left `def install`
# empty, which is not valid Ruby and would install nothing even if it parsed.
# No `license` line either: the repo carries no LICENSE file to name.
FORMULA_TEMPLATE = """class Ztools < Formula
  desc "Local LLM tools for Osaurus"
  homepage "https://github.com/ztomer/ztools"
  url "https://github.com/ztomer/ztools/archive/refs/tags/v{version}.tar.gz"
  sha256 "{sha256}"

  depends_on "rust" => :build

  def install
    system "cargo", "build", "--release", "--manifest-path", "rust/Cargo.toml"
    bin.install "target/release/ztools"
    SYMLINKED_SUBCOMMANDS.each do |subcommand|
      bin.install_symlink subcommand => "ztools"
    end
  end

  test do
    assert_match(/\\d+\\.\\d+\\.\\d+/, shell_output("#{bin}/ztools --version"))
  end
end
"""

# Ruby needs the list once; Python needs it above. They are the same list.
FORMULA_TEMPLATE = FORMULA_TEMPLATE.replace(
    "SYMLINKED_SUBCOMMANDS", "[" + ", ".join(f'"{name}"' for name in SUBCOMMAND_ALIASES) + "]"
)


def render_formula(version: str, sha256: str) -> str:
    """The tap formula for a release, as valid Ruby.

    `.replace`, not `.format`: the template is Ruby and is full of `#{}`
    interpolation (`shell_output("#{bin}/ztools --version")`), which `str.format`
    reads as its own replacement fields and chokes on. Substituting the two
    placeholders literally leaves the Ruby exactly as written.
    """
    return FORMULA_TEMPLATE.replace("{version}", version).replace("{sha256}", sha256)


def write_askpass_helper(directory: Path) -> Path:
    """A git credential helper that HOLDS NO SECRET.

    The token used to be interpolated into the clone URL
    (`https://x-access-token:<token>@github.com/...`), which put it in this
    process's argv -- readable by `ps` and by anything else listing processes --
    and, because git records the remote it cloned, in the temp clone's
    `.git/config` too. A credential-free URL plus GIT_ASKPASS keeps the token in
    one place: this process's environment, read by the helper at call time. Git
    invokes the helper with the prompt as $1 and reads the answer from its
    stdout.
    """
    path = directory / "git-askpass.sh"
    path.write_text(
        "#!/usr/bin/env bash\n"
        "# Written by tools/upgrade_tap.py. Contains no credential: the token is\n"
        "# read from the environment when git asks, never stored here.\n"
        'case "$1" in\n'
        '  *[Uu]sername*) printf "%s\\n" "$ZTOOLS_TAP_GIT_USERNAME" ;;\n'
        '  *) printf "%s\\n" "$HOMEBREW_TAP_TOKEN" ;;\n'
        "esac\n"
    )
    path.chmod(0o700)
    return path


def git_credentials_env(token: str, askpass: Path) -> dict:
    """The environment git authenticates with: no token in argv, no prompts."""
    return {
        **os.environ,
        "GIT_ASKPASS": str(askpass),
        # Without this, a helper that failed to answer falls back to a PROMPT on
        # the terminal -- in a release script, an interactive hang.
        "GIT_TERMINAL_PROMPT": "0",
        "ZTOOLS_TAP_GIT_USERNAME": "x-access-token",
        "HOMEBREW_TAP_TOKEN": token,
    }


def upgrade_remote(version: str, sha256: str, token: str):
    print_info(f"Cloning {TAP_REPO}...")
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # A credential-FREE url plus an askpass helper (see write_askpass_helper):
        # the token travels in the environment, never in argv and never in the
        # clone's .git/config.
        repo_dir = temp_dir / "homebrew-tap"
        credentials = git_credentials_env(token, write_askpass_helper(temp_dir))
        run_cmd(
            ["git", "clone", f"https://github.com/{TAP_REPO}.git", str(repo_dir)],
            env=credentials,
        )

        # Locate formula
        formula_path = repo_dir / "Formula" / FORMULA_NAME
        if not formula_path.exists():
            formula_path = repo_dir / FORMULA_NAME

        if not formula_path.exists():
            # If ztools.rb does not exist anywhere, create Formula/ztools.rb
            formula_path = repo_dir / "Formula" / FORMULA_NAME
            formula_path.parent.mkdir(exist_ok=True, parents=True)
            formula_path.write_text(render_formula(version, sha256))
            print_info(f"Created new formula at {formula_path}")
        else:
            update_formula_content(formula_path, version, sha256)

        # Commit and push. The push needs the same credentials as the clone, and
        # `gh` is not involved -- git authenticates the remote URL itself.
        run_cmd(["git", "config", "user.name", "github-actions[bot]"], cwd=repo_dir)
        bot_email = "github-actions[bot]@users.noreply.github.com"
        run_cmd(["git", "config", "user.email", bot_email], cwd=repo_dir)
        run_cmd(["git", "add", "."], cwd=repo_dir)

        # Check if anything changed
        status = run_cmd(["git", "status", "--porcelain"], cwd=repo_dir).stdout.strip()
        if not status:
            print_ok("No changes detected in Homebrew formula. Tap is already up-to-date.")
            return

        run_cmd(["git", "commit", "-m", f"Update ztools to v{version}"], cwd=repo_dir)
        run_cmd(["git", "push"], cwd=repo_dir, env=credentials)
        print_ok(f"Successfully pushed formula update to {TAP_REPO}")
    finally:
        shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(description="Upgrade Homebrew Tap formula for ztools")
    parser.add_argument("--version", required=True, help="New version (e.g. 0.9.7)")
    parser.add_argument(
        "--sha256", required=True, help="SHA256 checksum of the release source tarball"
    )
    parser.add_argument(
        "--tap-dir", help="Path to local homebrew-tap repository clone (if updating locally)"
    )
    parser.add_argument(
        "--token",
        help=(
            "GitHub token for the remote upgrade. Prefer the HOMEBREW_TAP_TOKEN "
            "environment variable: an argument is visible to every process "
            "listing on the machine while the command runs"
        ),
    )

    args = parser.parse_args()

    # Clean version string (remove leading 'v' if present)
    version = args.version.lstrip("v")

    if args.tap_dir:
        tap_dir = Path(args.tap_dir).resolve()
        formula_path = tap_dir / "Formula" / FORMULA_NAME
        if not formula_path.exists():
            formula_path = tap_dir / FORMULA_NAME
        if update_formula_content(formula_path, version, args.sha256):
            print_ok("Local Homebrew formula updated successfully.")
        else:
            sys.exit(1)
    else:
        # Remote upgrade using token
        token = args.token or os.environ.get("HOMEBREW_TAP_TOKEN")
        if not token:
            msg_err = (
                "GitHub token required for remote upgrade. "
                "Set HOMEBREW_TAP_TOKEN (preferred) or pass --token."
            )
            print_err(msg_err)
            sys.exit(1)
        try:
            upgrade_remote(version, args.sha256, token)
        # These two and ONLY these two, named rather than caught blind. Every
        # failure this script can be expected to have is one of them: a git
        # command that exited non-zero (run_cmd is check=True, so the remote is
        # where the failure almost always is -- a bad token, a diverged tap), or
        # the filesystem under it (mkdtemp, the clone, the formula, rmtree).
        # Catching `Exception` also caught the mistakes -- an AttributeError from
        # a typo'd argument printed "Failed to upgrade remote Homebrew tap:
        # 'str' object has no attribute ..." and exited 1, which reads like a
        # network problem and sends the reader to the wrong place. Anything else
        # propagates with its own traceback, which is the correct outcome.
        except (subprocess.CalledProcessError, OSError) as e:
            print_err(f"Failed to upgrade remote Homebrew tap: {e}")
            stderr = getattr(e, "stderr", None)
            if stderr:
                print_err(f"Command error output: {stderr}")
            sys.exit(1)


if __name__ == "__main__":
    main()
