"""bin/ab_test's Rust-suite step keeps a per-run log and names the failing test.

WHY (ZTL.RELQUIET). A v3.14.1 release attempt stopped at its install-and-test step
with one line, "Rust ztools tests failed": the suite ran as `cargo test --quiet`
and tools/release.sh sent ab_test's stdout to /dev/null, so the failing test's
name was gone and five reruns did not reproduce it. A flake that leaves no name
cannot be hunted. So the step must (1) write the suite's full output to a log of
its own run, (2) print that log's path whatever the outcome, and (3) on failure
print the failing tests' names on stderr, which a caller hiding stdout still shows.

`cargo` and the installed binary are stubbed on PATH: the property under test is
the harness's reporting, not the suite. The stub prints the shape libtest prints
for a failing run (a `... FAILED` line and the `failures:` list).
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
AB_TEST = REPO / "bin" / "ab_test"
TIMEOUT = 60

#: ab_test is the script that runs the real binary and suite; conftest proves the
#: stubbed run below never reaches the machine.
pytestmark = pytest.mark.sandboxed_server_script

FAILING_RUN = """\
running 3 tests
test store::tests::opens_db ... ok
test sched::tests::fires_once ... FAILED
test sched::tests::ticks ... ok

failures:

---- sched::tests::fires_once stdout ----
thread 'sched::tests::fires_once' panicked at src/sched.rs:10:5:
UNIQUE-MARKER-FROM-THE-PANIC

failures:
    sched::tests::fires_once

test result: FAILED. 2 passed; 1 failed; 0 ignored
error: test failed, to rerun pass `--lib`
"""


ZTOOLS_STUB = 'if [ "$1" = --version ]; then echo "ztools 0.0.0"; else echo usage; fi\n'


def _stub(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


def run_ab_test(tmp_path: Path, cargo_body: str) -> tuple[subprocess.CompletedProcess, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _stub(bin_dir, "cargo", cargo_body)
    _stub(bin_dir, "ztools", ZTOOLS_STUB)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["ZTOOLS_BIN"] = str(bin_dir / "ztools")
    env["ZTOOLS_AB_LOG_DIR"] = str(log_dir)
    result = subprocess.run(
        ["bash", str(AB_TEST)],
        capture_output=True,
        text=True,
        env=env,
        timeout=TIMEOUT,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    return result, log_dir


def logged_path(stream: str) -> Path:
    found = re.findall(r"(/\S+ztools-rust-suite\.[A-Za-z0-9]+)", stream)
    assert found, f"no suite log path printed:\n{stream}"
    return Path(found[-1])


def test_failure_names_the_test_on_stderr_and_keeps_the_full_log(tmp_path):
    body = f"cat <<'EOF'\n{FAILING_RUN}EOF\nexit 101\n"
    result, log_dir = run_ab_test(tmp_path, body)
    assert result.returncode != 0
    # stderr alone: release.sh may still hide stdout.
    assert "sched::tests::fires_once" in result.stderr, result.stderr
    assert "sched::tests::ticks" not in result.stderr
    log = logged_path(result.stderr)
    assert log.parent == log_dir
    assert "UNIQUE-MARKER-FROM-THE-PANIC" in log.read_text(encoding="utf-8")


def test_failure_with_no_test_name_shows_the_logs_tail(tmp_path):
    # A compile error or a crashed test binary names no test: the tail is the clue.
    result, _ = run_ab_test(tmp_path, "echo 'error[E0425]: COMPILE-MARKER'; exit 101\n")
    assert result.returncode != 0
    assert "COMPILE-MARKER" in result.stderr, result.stderr
    logged_path(result.stderr)


def test_success_prints_the_log_path_and_keeps_the_log(tmp_path):
    result, log_dir = run_ab_test(tmp_path, "echo 'test result: ok. 3 passed'; exit 0\n")
    assert result.returncode == 0, result.stderr
    log = logged_path(result.stdout + result.stderr)
    assert log.parent == log_dir
    assert "3 passed" in log.read_text(encoding="utf-8")
