"""rerun_truncated.sh — which rows it re-queues, the pressure gate, and the ordering.

WHY THIS FILE EXISTS. This script is what a sweep runs when a model was cut off at the
4h ceiling, so three things have to hold and none of them were tested: it re-queues only
the rows MORE TIME can fix (an EXCLUDED model cost 10 GPU-hours on 2026-08-23), it
refuses to measure a thrashing box (a leaked compressor reads as 0.55GB of RSS, so headroom
is the wrong instrument), and it restarts the server BEFORE measuring free memory, because
after a sweep osaurus still holds the last model resident and "available" is low BY DESIGN.

EVERY TEST DRIVES THE REAL SCRIPT, and the harness is closed. Nothing here can start a
server, load a model, signal a process, or touch the machine's real /tmp or the repo's own
.sweep_status:

  - `sysctl`, `vm_stat`, `pgrep`, `kill`, `top`, `sleep` and `timeout` are scripted into a
    stub binroot PREPENDED to PATH, so the real ones are unreachable -- `kill` in
    particular, because this script's sibling can SIGTERM and SIGKILL process lists;
  - the osaurus_one.sh seam is a stub, so nothing is started, stopped or quit and no gpu
    lock is taken;
  - SWEEP_STATUS, RERUN_LOGDIR and TMPDIR point inside tmp_path, and a test asserts the
    repo's own .sweep_status is untouched;
  - the pressure readings are CANNED, not this machine's: a test that read the live box
    would be a test of whoever happened to be running, and would go red during every eval.

The stub `sleep` is a no-op so the "wait for a running sweep" loop can be tested without
spending 60s per poll — the wait itself is asserted by the ORDER of the calls, not by the
clock.

tui/lib.sh is a formatting library in another checkout: the real one is used when present
and a minimal stand-in otherwise, because a state-machine test must not fail because a
second checkout is missing. Assertions are substring matches, so icons do not matter.
"""

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "tools" / "rerun_truncated.sh"
REPO_STATUS = REPO / ".sweep_status"

RUN_TIMEOUT = 60


def _stub(root, name, body):
    path = Path(root) / name
    path.write_text("#!/usr/bin/env bash\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _stand_in_tui(goh):
    tui = Path(goh) / "tui"
    tui.mkdir(parents=True, exist_ok=True)
    (tui / "lib.sh").write_text(
        "#!/usr/bin/env bash\n"
        "info() { printf '%s\\n' \"$*\" >&2; }\n"
        "ok() { printf '%s\\n' \"$*\" >&2; }\n"
        "warn() { printf '%s\\n' \"$*\" >&2; }\n"
        "err() { printf '%s\\n' \"$*\" >&2; }\n"
        'die() { err "$1"; exit "${2:-1}"; }\n'
        "section() { printf '\\n-- %s --\\n' \"$*\" >&2; }\n"
    )
    return goh


def _tui_dir(tmp_path):
    real = Path(os.environ.get("GOH_DIR", Path.home() / "Projects" / "gates_of_heck"))
    if (real / "tui" / "lib.sh").is_file():
        return str(real)
    return str(_stand_in_tui(tmp_path / "gates_of_heck"))


#: 16 KiB pages on Apple silicon — the page size the script's arithmetic assumes.
PAGE = 16384
#: Canned swap line. The script reads the `used = <N>M` field and divides by 1024.
SWAP_QUIET = "total = 16384.00M  used = 1024.00M  free = 15360.00M  (encrypted)"
SWAP_THRASHING = "total = 65536.00M  used = 20480.00M  free = 45056.00M  (encrypted)"


def _vm_stat(free=300_000, inactive=300_000, compressor=1_000, page=PAGE):
    """A vm_stat page in the shape the script parses: digits off line 1 for the page
    size, then lines STARTING with each label for the page counts."""
    return (
        f"Mach Virtual Memory Statistics: (page size of {page} bytes)\n"
        f"Pages free:                               {free}.\n"
        f"Pages active:                            {free + inactive + compressor}.\n"
        f"Pages inactive:                          {inactive}.\n"
        f"Pages occupied by compressor:            {compressor}.\n"
    )


def _row(state, model, tasks=17, exit_code=124):
    return f"{state}\t{model}\t14400s\ttasks={tasks}\texit={exit_code}\n"


class Rerun:
    """One hermetic rerun_truncated.sh run over canned pressure and canned evals."""

    def __init__(self, tmp_path):
        # realpath: macOS hands out /var/folders/... behind a symlink.
        self.root = Path(os.path.realpath(str(tmp_path)))
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.evals = self.root / "evals"
        self.evals.mkdir()
        self.logs = self.root / "logs"
        self.status = self.root / "sweep_status"
        self.marker = self.root / "marker"
        self.tmp = self.root / "tmp"
        self.tmp.mkdir()
        self.swap = SWAP_QUIET
        self.vm = _vm_stat()
        self.server_rc = 0
        self.pgrep_says_sweep = 0  # how many polls report a sweep running
        self.polls = self.root / "polls"

        # `timeout <ceiling> <cmd> model-eval --model M --suite full`.
        _stub(
            self.bin,
            "timeout",
            "shift\n"
            'model=""\n'
            "while [ $# -gt 0 ]; do\n"
            '  if [ "$1" = "--model" ] && [ $# -ge 2 ]; then model="$2"; shift 2; continue; fi\n'
            "  shift\n"
            "done\n"
            'printf "eval %s\\n" "$model" >> "$FAKE_MARKER"\n'
            'cat "$FAKE_EVAL_DIR/$model.out" 2>/dev/null\n'
            "rc=0\n"
            'if [ -f "$FAKE_EVAL_DIR/$model.rc" ]; then rc=$(cat "$FAKE_EVAL_DIR/$model.rc"); fi\n'
            'exit "$rc"',
        )
        _stub(
            self.bin,
            "sysctl",
            'printf "pressure\\n" >> "$FAKE_MARKER"\nprintf "%s\\n" "$FAKE_SWAPUSAGE"\n',
        )
        _stub(self.bin, "vm_stat", 'cat "$FAKE_VMSTAT"\n')
        _stub(self.bin, "top", "echo 'stub top: no RSS here'\n")
        _stub(self.bin, "sleep", "exit 0\n")
        # Insurance rather than a reachable path: no test may ever SIGNAL a real
        # process. osaurus_one.sh, the sibling that does `kill`, is stubbed whole.
        _stub(self.bin, "kill", "exit 1\n")
        # A sweep is "running" for the first FAKE_PGREP_BUSY polls, then not — so the
        # wait loop can be asserted by the ORDER of calls, without 60s per poll.
        _stub(
            self.bin,
            "pgrep",
            'printf "pgrep\\n" >> "$FAKE_MARKER"\n'
            "n=0\n"
            '[ -f "$FAKE_POLLS" ] && n=$(cat "$FAKE_POLLS")\n'
            "n=$((n + 1))\n"
            'printf "%s\\n" "$n" > "$FAKE_POLLS"\n'
            '[ "$n" -le "$FAKE_PGREP_BUSY" ] && exit 0\n'
            "exit 1\n",
        )
        self.server = _stub(
            self.bin,
            "osaurus_one.sh",
            'printf "server %s\\n" "$*" >> "$FAKE_MARKER"\nexit "${FAKE_SERVER_RC:-0}"',
        )

    def plan(self, model, rows=23, rc=0):
        table = [f"| task_{n} | {90 - n} | ok | note |" for n in range(rows)]
        (self.evals / f"{model}.out").write_text(
            f"## model-eval {model}\n" + "\n".join(table) + "\n"
        )
        (self.evals / f"{model}.rc").write_text(str(rc))

    def install(self, rows):
        self.status.write_text("".join(rows))

    def run(self, *args, extra_env=None):
        (self.root / "vm_stat.txt").write_text(self.vm)
        env = {
            **os.environ,
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "TMPDIR": str(self.tmp),
            "NO_COLOR": "1",
            "GOH_DIR": _tui_dir(self.root),
            "SWEEP_STATUS": str(self.status),
            "RERUN_LOGDIR": str(self.logs),
            "ZTOOLS_OSAURUS_ONE": str(self.server),
            "ZTOOLS_BIN": str(self.bin / "ztools"),
            "FAKE_MARKER": str(self.marker),
            "FAKE_EVAL_DIR": str(self.evals),
            "FAKE_SWAPUSAGE": self.swap,
            "FAKE_VMSTAT": str(self.root / "vm_stat.txt"),
            "FAKE_SERVER_RC": str(self.server_rc),
            "FAKE_PGREP_BUSY": str(self.pgrep_says_sweep),
            "FAKE_POLLS": str(self.polls),
        }
        env.pop("ZTOOLS_GPU_LOCK_OWNER", None)
        env.update(extra_env or {})
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO),
            timeout=RUN_TIMEOUT,
            check=False,
        )

    def all_output(self, result):
        return result.stdout + result.stderr

    def calls(self):
        """What the run reached, in order: pgrep / pressure / server --restart / eval M."""
        if not self.marker.exists():
            return []
        return [line for line in self.marker.read_text().splitlines()]

    def evaluated(self):
        return [c.split(" ", 1)[1] for c in self.calls() if c.startswith("eval ")]

    def rows(self):
        """Every status row, in file order — including the ones the test installed,
        because what this run APPENDED is the last one for that model."""
        parsed = []
        for line in self.status.read_text().splitlines():
            if line.strip():
                parts = line.split("\t")
                parsed.append({"state": parts[0], "model": parts[1], "raw": parts})
        return parsed

    def last(self, model):
        """The row this run appended for MODEL, which is what it learned."""
        return next(r for r in reversed(self.rows()) if r["model"] == model)

    def tasks_of(self, model):
        row = self.last(model)
        return int(next(p for p in row["raw"] if p.startswith("tasks=")).split("=")[1])


@pytest.fixture
def rerun(tmp_path):
    return Rerun(tmp_path)


class TestWhatGetsRequeued:
    def test_it_queues_truncated_and_failed_rows_only(self, rerun):
        rerun.install(
            [
                _row("DONE", "bonsai"),
                _row("TRUNCATED", "nemotron"),
                _row("FAILED", "shiba"),
                _row("REFUSED", "puppy"),
            ]
        )
        for model in ("nemotron", "shiba"):
            rerun.plan(model, rows=23)
        r = rerun.run()
        assert r.returncode == 0, rerun.all_output(r)
        assert sorted(rerun.evaluated()) == ["nemotron", "shiba"], (
            "DONE and REFUSED rows must not be re-measured"
        )

    def test_a_model_whose_last_state_is_done_is_not_re_queued_forever(self, rerun):
        """This script APPENDS its own result, so a model fixed by an earlier
        invocation still has its old TRUNCATED line on disk. LAST state per model."""
        rerun.install([_row("TRUNCATED", "nemotron"), _row("DONE", "nemotron")])
        r = rerun.run()
        assert rerun.evaluated() == [], rerun.all_output(r)
        assert "nothing truncated" in rerun.all_output(r)

    def test_an_unrecognised_state_is_left_alone(self, rerun):
        """A state this script does not understand is not a state it should assume it
        can fix. Queueing EXCLUDED is what spent 10 GPU-hours on 2026-08-23."""
        rerun.install([_row("EXCLUDED", "ornith-1.0-9b-mxfp8")])
        rerun.run()
        assert rerun.evaluated() == []

    def test_no_rows_at_all_is_a_clean_no_op(self, rerun):
        r = rerun.run()
        assert r.returncode == 0, rerun.all_output(r)
        assert rerun.calls() == [], "it polled, restarted or evaluated with nothing to do"

    def test_explicit_arguments_choose_the_models(self, rerun):
        """An operator re-running one model by name overrides the selection, including
        a model the status file calls DONE."""
        rerun.install([_row("DONE", "bonsai")])
        rerun.plan("bonsai", rows=23)
        r = rerun.run("bonsai")
        assert rerun.evaluated() == ["bonsai"], rerun.all_output(r)


class TestThePressureGate:
    def test_it_refuses_a_thrashing_swap(self, rerun):
        rerun.install([_row("TRUNCATED", "nemotron")])
        rerun.plan("nemotron")
        rerun.swap = SWAP_THRASHING
        r = rerun.run()
        assert r.returncode != 0
        out = rerun.all_output(r)
        assert "thrashing" in out and "swap 20.0GB" in out
        assert rerun.evaluated() == [], "it measured a box it had just called thrashing"

    def test_it_refuses_a_leaked_compressor_with_swap_still_low(self, rerun):
        """The 31GB leak: swap 12.88GB but compressor 29.3GB. Whichever field moved,
        the verdict must be refuse — that is the whole reason this is a pressure gate
        and not a headroom gate."""
        rerun.install([_row("TRUNCATED", "nemotron")])
        rerun.plan("nemotron")
        rerun.vm = _vm_stat(compressor=2_000_000)  # ~30.5GB compressed
        r = rerun.run()
        assert r.returncode != 0
        out = rerun.all_output(r)
        assert "thrashing" in out and "compressor 30.5GB" in out
        assert rerun.evaluated() == []

    def test_it_proceeds_on_a_quiet_box_and_says_what_it_read(self, rerun):
        """Pins the arithmetic the gate decides on: the page size, the divisor and both
        readings. 1.0GB swap, ~0.0GB compressed, (free+inactive) = 9GB available."""
        rerun.install([_row("TRUNCATED", "nemotron")])
        rerun.plan("nemotron", rows=23)
        r = rerun.run()
        assert r.returncode == 0, rerun.all_output(r)
        out = rerun.all_output(r)
        assert "no pressure (swap 1.0GB, compressor 0.0GB, 9GB available)" in out
        assert rerun.evaluated() == ["nemotron"]

    def test_it_refuses_an_exhausted_box(self, rerun):
        """Swap and compressor both fine, 4GB free against a 6GB floor: nothing here
        is thrashing, but there is not enough room to load a model."""
        rerun.install([_row("TRUNCATED", "nemotron")])
        rerun.plan("nemotron")
        rerun.vm = _vm_stat(free=100_000, inactive=200_000)  # 4GB available
        r = rerun.run()
        assert r.returncode != 0
        out = rerun.all_output(r)
        assert "exhausted" in out and "only 4GB available" in out
        assert rerun.evaluated() == []

    def test_the_ceilings_are_overridable(self, rerun):
        """The thresholds are env knobs, so they are live knobs — pinned here rather
        than assumed."""
        rerun.install([_row("TRUNCATED", "nemotron")])
        rerun.plan("nemotron")
        r = rerun.run(extra_env={"RERUN_MAX_SWAP_GB": "0.5"})
        assert r.returncode != 0
        assert "thrashing" in rerun.all_output(r)
        assert rerun.evaluated() == []


class TestOrdering:
    def test_the_server_is_restarted_before_the_probe_and_the_eval(self, rerun):
        """The order is the fix. After a sweep osaurus still holds the last model
        resident (14GB+), so measuring free memory BEFORE restarting it reads low BY
        DESIGN and a headroom threshold refuses on a perfectly clean machine."""
        rerun.install([_row("TRUNCATED", "nemotron")])
        rerun.plan("nemotron", rows=23)
        r = rerun.run()
        assert r.returncode == 0, rerun.all_output(r)
        calls = rerun.calls()
        assert calls.index("server --restart") < calls.index("pressure")
        assert calls.index("pressure") < calls.index("eval nemotron")
        assert calls.count("server --restart") == 1, "one restart per model, not one per loop"

    def test_it_refuses_to_measure_when_the_server_cannot_be_restarted(self, rerun):
        """Measuring against whatever state the server was left in is exactly what
        this script exists to avoid, so a failed restart stops it before the probe."""
        rerun.install([_row("TRUNCATED", "nemotron")])
        rerun.plan("nemotron")
        rerun.server_rc = 1
        r = rerun.run()
        assert r.returncode != 0
        assert "could not restart osaurus" in rerun.all_output(r)
        assert "pressure" not in rerun.calls()
        assert rerun.evaluated() == []

    def test_it_waits_while_a_sweep_is_running(self, rerun):
        """Two models are queued; the sweep is still running for the first poll, so
        the eval cannot start until it has gone. Asserted by the ORDER of the calls,
        with `sleep` stubbed to a no-op."""
        rerun.install([_row("TRUNCATED", "nemotron"), _row("TRUNCATED", "shiba")])
        rerun.plan("nemotron")
        rerun.plan("shiba")
        rerun.pgrep_says_sweep = 1
        r = rerun.run()
        assert r.returncode == 0, rerun.all_output(r)
        calls = rerun.calls()
        assert calls.count("pgrep") >= 2, calls
        assert calls.index("pgrep") < calls.index("eval nemotron")
        assert sorted(rerun.evaluated()) == ["nemotron", "shiba"]


class TestWhatItRecords:
    def test_a_timeout_records_truncated_with_its_task_count(self, rerun):
        rerun.install([_row("TRUNCATED", "nemotron")])
        rerun.plan("nemotron", rows=17, rc=124)
        r = rerun.run()
        assert r.returncode == 0, rerun.all_output(r)
        assert rerun.last("nemotron")["state"] == "TRUNCATED"
        assert rerun.tasks_of("nemotron") == 17
        assert "TRUNCATED AGAIN" in rerun.all_output(r)

    def test_a_nonzero_exit_records_failed(self, rerun):
        rerun.install([_row("TRUNCATED", "nemotron")])
        rerun.plan("nemotron", rows=4, rc=101)
        rerun.run()
        assert rerun.last("nemotron")["state"] == "FAILED"
        assert rerun.tasks_of("nemotron") == 4

    def test_a_zero_task_run_still_records_its_state_and_the_next_model_runs(self, rerun):
        """The case this whole script exists for: a model that wedges before scoring
        anything produces an EMPTY log. `grep -o` exits 1 when it matched nothing and
        this script runs `set -euo pipefail`, so the count assignment failed and the
        script ABORTED — recording nothing, and never reaching the second model."""
        rerun.install([_row("TRUNCATED", "nemotron"), _row("FAILED", "shiba")])
        rerun.plan("nemotron", rows=0, rc=124)
        rerun.plan("shiba", rows=23)
        r = rerun.run()
        assert r.returncode == 0, rerun.all_output(r)
        # The queue order comes out of an awk hash, so it is a SET, not a sequence.
        assert sorted(rerun.evaluated()) == ["nemotron", "shiba"], rerun.all_output(r)
        assert rerun.last("nemotron")["state"] == "TRUNCATED"
        assert rerun.tasks_of("nemotron") == 0
        assert rerun.last("shiba")["state"] == "DONE"

    def test_a_retried_task_is_counted_once(self, rerun):
        rerun.install([_row("TRUNCATED", "nemotron")])
        (rerun.evals / "nemotron.out").write_text(
            "| task_a | 91 | ok | note |\n"
            "| task_a | 88 | ok | note |\n"
            "| task_b | 70 | warn | note |\n"
        )
        (rerun.evals / "nemotron.rc").write_text("0")
        rerun.run()
        assert rerun.tasks_of("nemotron") == 2


class TestTheSandboxHolds:
    def test_the_repos_own_status_file_is_never_written(self, rerun):
        """The seam, asserted: this script appends to whatever .sweep_status names, and
        a test run must not append to the repo's real one."""
        before = REPO_STATUS.stat().st_mtime_ns if REPO_STATUS.exists() else None
        rerun.install([_row("TRUNCATED", "nemotron")])
        rerun.plan("nemotron", rows=23)
        rerun.run()
        after = REPO_STATUS.stat().st_mtime_ns if REPO_STATUS.exists() else None
        assert after == before
        assert (rerun.logs / "rerun-nemotron.log").is_file()

    def test_a_missing_tui_is_a_named_warning_not_a_bare_exit(self, rerun):
        rerun.install([_row("TRUNCATED", "nemotron")])
        rerun.plan("nemotron", rows=23)
        r = rerun.run(extra_env={"GOH_DIR": str(rerun.root / "absent")})
        assert r.returncode == 0, rerun.all_output(r)
        assert "GOH_DIR" in rerun.all_output(r)
        assert rerun.evaluated() == ["nemotron"]

    def test_a_missing_tui_still_makes_die_a_real_abort(self, rerun):
        rerun.install([_row("TRUNCATED", "nemotron")])
        rerun.plan("nemotron")
        rerun.swap = SWAP_THRASHING
        r = rerun.run(extra_env={"GOH_DIR": str(rerun.root / "absent")})
        assert r.returncode == 1, rerun.all_output(r)
        assert "thrashing" in rerun.all_output(r)
        assert "command not found" not in rerun.all_output(r)
