"""`cargo run` sees the operator's real HOME; only test binaries see the sandbox (ZT.CARGORUNHOME).

The test sandbox (`HOME=rust/target/test-home`) used to be a `[env]` entry in
`.cargo/config.toml`. Cargo applies `[env]` to every process it starts, `cargo run` included, so
`cargo run -- models scout` read a machine with no installed models and no free disk (found
2026-11-12, LM.SCOUT). The sandbox now lives in a cargo runner, `tools/cargo_test_runner.sh`,
which sets it only for binaries cargo keeps in `deps/` (test, bench and integration-test
executables). `rust/tests/test_home_is_sandboxed.rs` is the other half: a test process that does
not see the sandbox fails.

The end-to-end check drives REAL cargo against a scratch crate that carries a copy of the repo's
own config and runner, so it measures the mechanism without compiling ztools.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / ".cargo" / "config.toml"
RUNNER = REPO_ROOT / "tools" / "cargo_test_runner.sh"

PROBE_MAIN = """\
fn main() {
    std::fs::write(std::env::var("PROBE_OUT").unwrap(), std::env::var("HOME").unwrap()).unwrap();
}

#[test]
fn probe() {
    std::fs::write(std::env::var("PROBE_OUT").unwrap(), std::env::var("HOME").unwrap()).unwrap();
}
"""


def scratch_crate(tmp: Path, config_text: str) -> Path:
    """A one-file crate under `tmp` that carries `config_text` and the repo's runner."""
    (tmp / ".cargo").mkdir()
    (tmp / ".cargo" / "config.toml").write_text(config_text, encoding="utf-8")
    (tmp / "tools").mkdir()
    shutil.copy(RUNNER, tmp / "tools" / RUNNER.name)
    (tmp / "src").mkdir()
    (tmp / "src" / "main.rs").write_text(PROBE_MAIN, encoding="utf-8")
    (tmp / "Cargo.toml").write_text(
        '[package]\nname = "probe"\nversion = "0.0.0"\nedition = "2024"\n', encoding="utf-8"
    )
    return tmp


def probe(tmp: Path, *cargo_args: str) -> str:
    """The HOME the probe saw under `cargo <cargo_args>`."""
    out = tmp / "probe_out"
    env = {
        **os.environ,
        "PROBE_OUT": str(out),
        "CARGO_TARGET_DIR": str(tmp / "target"),
    }
    subprocess.run(
        ["cargo", *cargo_args, "-q"],
        cwd=tmp,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    return out.read_text(encoding="utf-8")


def test_config_holds_no_home_entry_and_names_the_runner() -> None:
    cfg = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    assert "HOME" not in cfg.get("env", {}), "[env] HOME reaches `cargo run` (ZT.CARGORUNHOME)"
    runners = [t.get("runner") for t in cfg.get("target", {}).values()]
    assert any(r and "tools/cargo_test_runner.sh" in str(r) for r in runners), runners


def test_cargo_run_sees_the_real_home(tmp_path: Path) -> None:
    crate = scratch_crate(tmp_path, CONFIG.read_text(encoding="utf-8"))
    assert probe(crate, "run") == os.environ["HOME"]


def test_cargo_test_sees_the_sandbox_home(tmp_path: Path) -> None:
    crate = scratch_crate(tmp_path, CONFIG.read_text(encoding="utf-8"))
    home = probe(crate, "test")
    assert home == str(crate / "rust" / "target" / "test-home"), home
    assert home != os.environ["HOME"]


def test_the_old_env_entry_fails_the_run_check(tmp_path: Path) -> None:
    """Calibration: the pre-fix config makes `cargo run` see the sandbox (the check can go red)."""
    old = '[env]\nHOME = { value = "rust/target/test-home", relative = true, force = true }\n'
    crate = scratch_crate(tmp_path, old)
    assert probe(crate, "run") != os.environ["HOME"]


def test_runner_sandboxes_deps_and_out_directories_only(tmp_path: Path) -> None:
    """The runner's rule, per directory shape, without cargo (both layouts cargo has used)."""
    (tmp_path / "tools").mkdir()
    shutil.copy(RUNNER, tmp_path / "tools" / RUNNER.name)
    probe_sh = '#!/bin/sh\nprintf %s "$HOME"\n'
    seen = {}
    for sub in (
        "target/debug/deps",
        "b/debug/build/p/h/out",
        "target/debug",
        "target/debug/examples",
    ):
        exe = tmp_path / sub / "probe"
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_text(probe_sh, encoding="utf-8")
        exe.chmod(0o755)
        seen[sub] = subprocess.run(
            [str(tmp_path / "tools" / RUNNER.name), str(exe)],
            capture_output=True,
            text=True,
            check=True,
            stdin=subprocess.DEVNULL,
        ).stdout
    sandbox = str(tmp_path / "rust" / "target" / "test-home")
    assert seen == {
        "target/debug/deps": sandbox,
        "b/debug/build/p/h/out": sandbox,
        "target/debug": os.environ["HOME"],
        "target/debug/examples": os.environ["HOME"],
    }
