"""`tools/gate.sh` is the one gate entry, and its default is the FULL gate.

The documented gate was the structural layer alone: `tools/gate.sh` with no scope
fell through to `structural.sh`, while the Rust suite, clippy, coverage and audit
ran only under `--full` / `make ci`. An agent that ran the entry point it was
told about got a green that skipped the suite (ZT.GATEHYG, seen by ZT-4,
2026-11-12). The wrong state is now unrepresentable: an unscoped call IS the
push gate, and the cheap scope must be asked for by name (`--staged`, which the
pre-commit hook passes).

The test runs the real script against a stand-in GOH_DIR whose two runners only
record that they were called, so it proves routing, not the gates themselves.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parents[1] / "gate.sh"


def _fake_goh(root: Path) -> Path:
    gates = root / "gates"
    gates.mkdir(parents=True)
    for name in ("local_ci.sh", "structural.sh"):
        script = gates / name
        script.write_text(f'#!/usr/bin/env bash\necho "RAN {name} $*"\n')
        script.chmod(0o755)
    return root


def _run(tmp_path: Path, *args: str) -> str:
    env = {**os.environ, "GOH_DIR": str(_fake_goh(tmp_path / "goh"))}
    env.pop("GOH", None)
    done = subprocess.run(
        ["bash", str(GATE), *args],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=True,
    )
    return done.stdout


@pytest.mark.parametrize("args", [(), ("--full",)])
def test_the_unscoped_and_full_calls_run_the_push_gate(tmp_path: Path, args: tuple[str, ...]):
    out = _run(tmp_path, *args)
    assert "RAN local_ci.sh" in out
    assert "structural.sh" not in out


def test_staged_is_the_named_cheap_scope(tmp_path: Path):
    out = _run(tmp_path, "--staged")
    assert "RAN structural.sh --staged" in out
    assert "local_ci.sh" not in out
