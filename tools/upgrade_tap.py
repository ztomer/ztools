#!/usr/bin/env python3
"""
Automate updating the Homebrew formula for ztools in the homebrew-tap repository.
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

    # The URL must name the artifact the checksum was computed over. Both
    # release paths (this one, driven by .github/workflows/release.yml, and the
    # manual tools/release.sh) shasum GitHub's auto-generated tag archive, so
    # they write the same formula and cannot fight over which file is canonical.
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


def upgrade_remote(version: str, sha256: str, token: str):
    print_info(f"Cloning {TAP_REPO}...")
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Clone using token authentication
        clone_url = f"https://x-access-token:{token}@github.com/{TAP_REPO}.git"
        repo_dir = temp_dir / "homebrew-tap"
        run_cmd(["git", "clone", clone_url, str(repo_dir)])

        # Locate formula
        formula_path = repo_dir / "Formula" / FORMULA_NAME
        if not formula_path.exists():
            formula_path = repo_dir / FORMULA_NAME

        if not formula_path.exists():
            # If ztools.rb does not exist anywhere, create Formula/ztools.rb
            formula_path = repo_dir / "Formula" / FORMULA_NAME
            formula_path.parent.mkdir(exist_ok=True, parents=True)
            formula_path.write_text(f"""class Ztools < Formula
  desc "Local LLM tools for Osaurus"
  homepage "https://github.com/ztomer/ztools"
  url "https://github.com/ztomer/ztools/archive/refs/tags/v{version}.tar.gz"
  sha256 "{sha256}"

  depends_on "python@3.12"

  def install
    # Installation logic
  end
end
""")
            print_info(f"Created new formula at {formula_path}")
        else:
            update_formula_content(formula_path, version, sha256)

        # Commit and push
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
        run_cmd(["git", "push"], cwd=repo_dir)
        print_ok(f"Successfully pushed formula update to {TAP_REPO}")
    finally:
        shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(description="Upgrade Homebrew Tap formula for ztools")
    parser.add_argument(
        "--version", required=True, help="New version (e.g. 0.9.7)"
    )
    parser.add_argument(
        "--sha256", required=True, help="SHA256 checksum of the release source tarball"
    )
    parser.add_argument(
        "--tap-dir", help="Path to local homebrew-tap repository clone (if updating locally)"
    )
    parser.add_argument("--token", help="GitHub Personal Access Token for remote upgrade")

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
                "Specify --token or set HOMEBREW_TAP_TOKEN."
            )
            print_err(msg_err)
            sys.exit(1)
        try:
            upgrade_remote(version, args.sha256, token)
        except Exception as e:
            print_err(f"Failed to upgrade remote Homebrew tap: {e}")
            if hasattr(e, "stderr") and e.stderr:
                print_err(f"Command error output: {e.stderr}")
            sys.exit(1)


if __name__ == "__main__":
    main()
