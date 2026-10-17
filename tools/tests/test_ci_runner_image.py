"""The CI runner image is the newest GitHub-hosted macOS arm64 image, not a moving tag.

`.github/workflows/ci.yml` used to run on `macos-15` while the host is
macOS 27. The runner image must be named from GitHub's runner list (measured, not
assumed); as of 2026-10-16 the newest GA macOS arm64 image is `macos-26`. A moving
tag such as `macos-latest` is deliberately not used: "latest means a named release,
never a moving tag".
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

#: The newest GA GitHub-hosted macOS arm64 runner image, measured from the runner list.
NEWEST_MACOS_ARM64 = "macos-26"


def test_the_reader_finds_runs_on():
    text = WORKFLOW.read_text(encoding="utf-8")
    m = re.search(r"^\s*runs-on:\s*['\"]?(\S+?)['\"]?\s*$", text, re.MULTILINE)
    assert m is not None, "ci.yml has no runs-on line"
    assert m.group(1) == NEWEST_MACOS_ARM64, (
        f"ci.yml runs on {m.group(1)!r}; expected the newest macOS arm64 image "
        f"{NEWEST_MACOS_ARM64!r}"
    )
