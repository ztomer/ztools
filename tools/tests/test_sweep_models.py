"""sweep_models.sh — the status-file state machine, the summary verdict and --resume.

WHY THIS FILE EXISTS. tools/sweep_models.sh is what a multi-hour unattended sweep's
correctness rests on, and nothing tested it. Its own header says what it is for: the
scratch script it replaced wrote DONE regardless of exit code, so a model killed at the
timeout boundary was skipped forever and its subset scores were ranked against complete
runs. Every test below drives the REAL script — the state transitions and the summary
verdict, not a re-implementation of them.

THE HARNESS IS CLOSED. Nothing here can start a server, load a model, signal a process,
or touch the machine-wide GPU lock or the repo's own .sweep_status:

  - `osaurus`, `timeout` (the process that would load 27GB of weights), `sysctl`,
    `vm_stat`, `pgrep`, `kill` and `sleep` are scripted into a stub binroot that is
    PREPENDED to PATH, so the real ones are unreachable;
  - the osaurus_one.sh seam is a stub, so no server is started, stopped or quit, and
    no gpu lock is taken (--check does not take one either, and the mutating modes are
    never entered);
  - SWEEP_STATUS, SWEEP_LOGDIR and TMPDIR all point inside tmp_path;
  - the eval is answered from canned files, one per model, in the same directory.

The stub `sleep` is a no-op because the sweep's only sleep is a one-second wait for a
REAL child's buffered output to land; a stub child writes its log and exits, so there is
nothing to wait for and a 1s tax per test buys nothing.

tui/lib.sh is a formatting library in another checkout. These tests use the real one
when it is present (so the real output path is exercised) and a minimal stand-in when it
is not, because a state-machine test must not fail because a second checkout is absent —
the same reason tools/gpu_lock.sh defines its own helpers. Assertions are substring
matches, so icons and colour are irrelevant either way.
"""

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "tools" / "sweep_models.sh"

#: Every sweep driven here settles well inside a minute.
RUN_TIMEOUT = 90


def _stub(root, name, body):
    path = Path(root) / name
    path.write_text("#!/usr/bin/env bash\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _stand_in_tui(goh):
    """A tui/ that exists, so the script's missing-tui path is not what gets tested."""
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


class Sweep:
    """One hermetic sweep_models.sh run over canned per-model eval results."""

    def __init__(self, tmp_path):
        # realpath: macOS hands out /var/folders/..., which is a symlink to
        # /private/var/folders/..., and the script echoes the path it built while
        # the test resolves the same directory through `latest`.
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
        self.server_rc = 0
        self.models = ""

        # `timeout <duration> <cmd> model-eval --model M --suite full`. Replays a
        # canned stdout and exit code, and records that the model was asked for.
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
        _stub(self.bin, "osaurus", '[ "$1" = "list" ] || exit 1\ncat "$FAKE_MODELS"')
        _stub(self.bin, "sleep", "exit 0")
        # Not on this script's path today. They are here so that a test can never read
        # the live machine even if the sweep grows a pressure probe, and so that no test
        # can signal a real process.
        _stub(self.bin, "sysctl", 'printf "%s\\n" "$FAKE_SWAPUSAGE"\n')
        _stub(self.bin, "vm_stat", 'cat "$FAKE_VMSTAT"\n')
        _stub(self.bin, "pgrep", "exit 1")
        _stub(self.bin, "kill", "exit 1")
        self.server = _stub(
            self.bin,
            "osaurus_one.sh",
            'printf "server %s\\n" "$*" >> "$FAKE_MARKER"\nexit "${FAKE_SERVER_RC:-0}"',
        )

    def plan(self, model, rows=2, rc=0):
        """Canned eval result for MODEL: ROWS scored rows of the results table, exit RC."""
        table = [f"| task_{n} | {90 - n} | ok | note |" for n in range(rows)]
        (self.evals / f"{model}.out").write_text(
            "## model-eval " + model + "\n" + "\n".join(table) + "\n"
        )
        (self.evals / f"{model}.rc").write_text(str(rc))

    def install(self, rows):
        (self.status).write_text("".join(rows))

    def run(self, *args, extra_env=None):
        (self.root / "vm_stat.txt").write_text(_vm_stat(16384, 300_000, 300_000, 1_000))
        env = {
            **os.environ,
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "TMPDIR": str(self.tmp),
            "NO_COLOR": "1",
            "GOH_DIR": _tui_dir(self.root),
            "SWEEP_STATUS": str(self.status),
            "SWEEP_LOGDIR": str(self.logs),
            "ZTOOLS_OSAURUS_ONE": str(self.server),
            "ZTOOLS_BIN": str(self.bin / "ztools"),
            "FAKE_MARKER": str(self.marker),
            "FAKE_EVAL_DIR": str(self.evals),
            "FAKE_MODELS": str(self.root / "models"),
            "FAKE_SERVER_RC": str(self.server_rc),
            "FAKE_SWAPUSAGE": "total = 16384.00M  used = 1024.00M  free = 15360.00M  (encrypted)",
            "FAKE_VMSTAT": str(self.root / "vm_stat.txt"),
        }
        env.pop("ZTOOLS_GPU_LOCK_OWNER", None)
        env.update(extra_env or {})
        (self.root / "models").write_text(self.models)
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

    def rows(self):
        """The status file parsed: one dict per row, in file order."""
        parsed = []
        for line in self.status.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            assert len(parts) >= 2, f"malformed status row: {line!r}"
            parsed.append({"state": parts[0], "model": parts[1], "raw": parts})
        return parsed

    def states(self, model):
        return [row["state"] for row in self.rows() if row["model"] == model]

    def ran(self):
        if not self.marker.exists():
            return []
        return [
            line.split(" ", 1)[1]
            for line in self.marker.read_text().splitlines()
            if line.startswith("eval ")
        ]


def _vm_stat(page_size, free, inactive, compressor):
    """A vm_stat page, in the shape rerun_truncated.sh and these tests parse."""
    return (
        f"Mach Virtual Memory Statistics: (page size of {page_size} bytes)\n"
        f"Pages free:                               {free}.\n"
        f"Pages active:                            {free + inactive + compressor}.\n"
        f"Pages inactive:                          {inactive}.\n"
        f"Pages occupied by compressor:            {compressor}.\n"
    )


def _row(state, model, tasks=23, exit_code=0):
    return f"{state}\t{model}\t42s\ttasks={tasks}\texit={exit_code}\n"


def _score_of(row):
    """The tasks=N field, as an int — so a test can tell 0 from a missing value."""
    return int(next(part for part in row["raw"] if part.startswith("tasks=")).split("=")[1])


def _scored_rows(log):
    """How many scored rows a per-model log holds."""
    return sum(1 for line in log.read_text().splitlines() if line.startswith("| task_"))


@pytest.fixture
def sweep(tmp_path):
    return Sweep(tmp_path)


class TestTheSweepVerdict:
    """The summary line is the thing an operator reads to decide whether to rank."""

    def test_a_sweep_whose_models_all_scored_is_complete(self, sweep):
        """The all-green path, and the one that used to print
        `integer expression expected`: `grep -c` printed 0 AND exited 1, so
        `|| echo 0` appended a second 0 and the verdict was the two-line "0\n0"."""
        sweep.models = "bonsai\n"
        sweep.plan("bonsai", rows=23)
        r = sweep.run("--model", "bonsai")
        assert r.returncode == 0, sweep.all_output(r)
        assert "every model completed" in sweep.all_output(r)
        assert "integer expression expected" not in sweep.all_output(r)
        assert sweep.states("bonsai") == ["DONE"]

    def test_a_truncated_row_makes_the_sweep_incomplete(self, sweep):
        """Exit 124 is `timeout`'s own code: the model was cut off, so its scores
        cover a SUBSET and must not be ranked against complete runs."""
        sweep.models = "nemotron\n"
        sweep.plan("nemotron", rows=17, rc=124)
        r = sweep.run("--model", "nemotron")
        assert r.returncode == 1, sweep.all_output(r)
        assert "1 model(s) did not finish" in sweep.all_output(r)
        assert sweep.states("nemotron") == ["TRUNCATED"]
        assert _score_of(sweep.rows()[0]) == 17

    def test_a_failed_row_makes_the_sweep_incomplete(self, sweep):
        sweep.models = "bonsai\n"
        sweep.plan("bonsai", rows=3, rc=101)
        r = sweep.run("--model", "bonsai")
        assert r.returncode == 1, sweep.all_output(r)
        assert "1 model(s) did not finish" in sweep.all_output(r)
        assert sweep.states("bonsai") == ["FAILED"]

    def test_every_unfinished_model_is_counted_not_just_the_first(self, sweep):
        """The count, not a yes/no: `^DONE` instead of `^(TRUNCATED|FAILED)`, or a
        grep that stopped at the first hit, would report one of two."""
        sweep.models = "nemotron\nbonsai\n"
        sweep.plan("nemotron", rows=17, rc=124)
        sweep.plan("bonsai", rows=0, rc=7)
        r = sweep.run()
        assert r.returncode == 1, sweep.all_output(r)
        assert "2 model(s) did not finish" in sweep.all_output(r)
        assert sweep.states("nemotron") == ["TRUNCATED"]
        assert sweep.states("bonsai") == ["FAILED"]

    def test_a_refused_row_is_not_counted_as_unfinished(self, sweep):
        """REFUSED is the eval declining — oversize, paging, a held lock — and it
        scored nothing, so there are no subset scores to mistakenly rank. It is
        deliberately outside `^(TRUNCATED|FAILED)`."""
        sweep.models = "bonsai\n"
        sweep.plan("bonsai", rows=0, rc=0)
        r = sweep.run("--model", "bonsai")
        assert sweep.states("bonsai") == ["REFUSED"]
        assert r.returncode == 0, sweep.all_output(r)
        assert "every model completed" in sweep.all_output(r)

    def test_a_refusal_records_the_reason_the_eval_gave(self, sweep):
        """Exit 0 with zero scored rows is a REFUSAL, not a measurement, and the
        reason is worth keeping: the four models filed this way on 2026-09-19 were
        all the eval declining while a 17GB browser held the box. `no reason logged`
        would have left the operator with a verdict and no cause."""
        sweep.models = "bonsai\n"
        sweep.plan("bonsai", rows=0, rc=0)
        (sweep.evals / "bonsai.out").write_text(
            "## model-eval bonsai\nSkipping budget: request exceeds the context window\n"
        )
        sweep.run("--model", "bonsai")
        row = sweep.rows()[0]
        assert row["state"] == "REFUSED", row["raw"]
        assert _score_of(row) == 0
        assert "exceeds the context window" in row["raw"][-1]

    def test_a_poor_score_is_still_a_score(self, sweep):
        """All three result markers count. Counting only the ok marker reported 20 of
        23 for every model and made a bad score look like a short run."""
        sweep.models = "weakling\n"
        (sweep.evals / "weakling.out").write_text(
            "| task_a | 91 | ok | note |\n"
            "| task_b | 55 | warn | note |\n"
            "| task_c | 30 | fail | note |\n"
        )
        (sweep.evals / "weakling.rc").write_text("0")
        r = sweep.run("--model", "weakling")
        assert r.returncode == 0, sweep.all_output(r)
        assert _score_of(sweep.rows()[0]) == 3

    def test_a_retried_task_is_counted_once(self, sweep):
        """DISTINCT names: a retry logs a second row, and a raw line count then
        exceeds the number of tasks that exist — 30 of 23, which is not a progress
        number, it is a bug wearing one."""
        sweep.models = "bonsai\n"
        (sweep.evals / "bonsai.out").write_text(
            "| task_a | 91 | ok | note |\n"
            "| task_a | 90 | ok | note |\n"
            "| task_b | 80 | ok | note |\n"
        )
        (sweep.evals / "bonsai.rc").write_text("0")
        sweep.run("--model", "bonsai")
        assert _score_of(sweep.rows()[0]) == 2


class TestResume:
    def test_resume_skips_a_model_already_done(self, sweep):
        sweep.models = "bonsai\nornith\n"
        sweep.install([_row("DONE", "bonsai"), _row("TRUNCATED", "ornith")])
        sweep.plan("ornith", rows=23)
        r = sweep.run("--resume")
        assert r.returncode == 0, sweep.all_output(r)
        assert sweep.ran() == ["ornith"], "a DONE model must not be re-measured"
        assert sweep.states("bonsai") == ["DONE"]
        assert sweep.states("ornith") == ["DONE"]

    def test_resume_picks_up_a_truncated_model(self, sweep):
        sweep.models = "nemotron\n"
        sweep.install([_row("TRUNCATED", "nemotron", tasks=17, exit_code=124)])
        sweep.plan("nemotron", rows=23)
        r = sweep.run("--resume")
        assert r.returncode == 0, sweep.all_output(r)
        assert sweep.ran() == ["nemotron"]
        rows = [row for row in sweep.rows() if row["model"] == "nemotron"]
        assert [row["state"] for row in rows] == ["DONE"], (
            "one record per model, or --resume reads a stale state"
        )

    def test_resume_reruns_a_refused_model(self, sweep):
        """REFUSED is re-queued, deliberately: it was never measured, and the whole
        point of the 2026-09-19 fix was that a re-run must walk past models a
        refused eval never scored. It is only the SUMMARY that excludes it."""
        sweep.models = "bonsai\n"
        sweep.install([_row("REFUSED", "bonsai", tasks=0)])
        sweep.plan("bonsai", rows=23)
        sweep.run("--resume")
        assert sweep.ran() == ["bonsai"]
        assert sweep.states("bonsai") == ["DONE"]

    def test_without_resume_every_model_runs(self, sweep):
        sweep.models = "bonsai\n"
        sweep.install([_row("DONE", "bonsai")])
        sweep.plan("bonsai", rows=23)
        sweep.run()
        assert sweep.ran() == ["bonsai"]

    def test_the_unrunnable_default_skip_list_is_honoured(self, sweep):
        """10 GPU-hours went into ornith-1.0-9b on 2026-08-23 because a bare
        `./tools/sweep_models.sh` re-ran a model already known unrunnable, so the
        permanent exclusions are in the script rather than left to an operator's
        --skip."""
        sweep.models = "bonsai\nqwen-mtp-4bit\npotion-embed\n"
        sweep.plan("bonsai", rows=23)
        r = sweep.run()
        assert sweep.ran() == ["bonsai"], sweep.all_output(r)
        assert "Sweeping 1 model(s)" in sweep.all_output(r)


class TestTheSweepRefusesToMeasureWrong:
    def test_it_refuses_without_a_single_server(self, sweep):
        """One server, or the numbers are worthless: a second server does not queue,
        it loads its own copy and the client cannot tell that from a slow model."""
        sweep.models = "bonsai\n"
        sweep.plan("bonsai", rows=23)
        sweep.server_rc = 1
        r = sweep.run("--model", "bonsai")
        assert r.returncode != 0
        assert "could not establish a single osaurus server" in sweep.all_output(r)
        assert sweep.rows() == [], "a refused sweep recorded a verdict anyway"
        assert sweep.ran() == [], "an eval ran without a server"

    def test_it_refuses_when_every_model_is_excluded(self, sweep):
        sweep.models = "potion-embed\n"
        r = sweep.run()
        assert r.returncode != 0
        assert "no models to sweep" in sweep.all_output(r)

    def test_an_unknown_argument_is_named(self, sweep):
        r = sweep.run("--nope")
        assert r.returncode != 0
        assert "unknown argument: --nope" in sweep.all_output(r)


class TestTheSweepSurvivesItsEnvironment:
    def test_status_prints_the_file(self, sweep):
        sweep.install([_row("DONE", "bonsai")])
        r = sweep.run("--status")
        assert r.returncode == 0, sweep.all_output(r)
        assert "DONE\tbonsai" in sweep.all_output(r)

    def test_a_missing_tui_is_a_named_warning_not_a_bare_exit(self, sweep):
        """Fail-closed is right, unnamed is not: the operator has to learn that
        GOH_DIR points nowhere and that the fix is that checkout's install.sh."""
        sweep.models = "bonsai\n"
        sweep.plan("bonsai", rows=23)
        r = sweep.run("--model", "bonsai", extra_env={"GOH_DIR": str(sweep.root / "absent")})
        assert r.returncode == 0, sweep.all_output(r)
        assert "GOH_DIR" in sweep.all_output(r)
        assert "every model completed" in sweep.all_output(r)

    def test_a_missing_tui_still_makes_die_a_real_abort(self, sweep):
        """`die: command not found` is not an abort but a CONTINUE, which is how
        gpu_lock_acquire came to spin forever and hang the suite at 24%. So: no tui
        at all, the server guard fails, and the script must still STOP — non-zero,
        and with the message, not exit 127 from the shell."""
        sweep.models = "bonsai\n"
        sweep.plan("bonsai", rows=23)
        sweep.server_rc = 1
        r = sweep.run("--model", "bonsai", extra_env={"GOH_DIR": str(sweep.root / "absent")})
        assert r.returncode == 1, sweep.all_output(r)
        assert "could not establish a single osaurus server" in sweep.all_output(r)
        assert "command not found" not in sweep.all_output(r)

    def test_logs_land_in_a_per_run_directory_with_a_latest_symlink(self, sweep):
        """A shared directory outlives its run, and stale logs read as current ones:
        a monitor reported seven 499s from a previous sweep while the live one was
        clean."""
        sweep.models = "bonsai\n"
        sweep.plan("bonsai", rows=23)
        r = sweep.run("--model", "bonsai")
        latest = sweep.logs / "latest"
        assert latest.is_symlink()
        rundir = latest.resolve()
        assert rundir != latest, "latest resolves to itself"
        assert (rundir / "bonsai.log").is_file()
        assert "logs:   " + str(rundir) in sweep.all_output(r)

    def test_resume_reuses_the_previous_runs_log_directory(self, sweep):
        """Half a resumed sweep's logs in one directory and half in another is the
        confusion the per-run directory exists to remove. Proven by CONTENT rather
        than by the path: the re-run overwrote the first run's log in place, so the
        truncated 2-row log is gone from the directory `--resume` pointed at. (A
        path comparison would be vacuous — both runs can land in the same
        second-resolution directory name.)"""
        sweep.models = "bonsai\n"
        sweep.install([_row("TRUNCATED", "bonsai", tasks=2, exit_code=124)])
        sweep.plan("bonsai", rows=2, rc=124)
        sweep.run("--resume")
        rundir = (sweep.logs / "latest").resolve()
        assert _scored_rows(rundir / "bonsai.log") == 2

        sweep.plan("bonsai", rows=23)  # the re-run scores everything
        sweep.run("--resume")
        assert sweep.states("bonsai") == ["DONE"]
        assert _scored_rows(rundir / "bonsai.log") == 23, (
            "the resumed run wrote its log somewhere other than the run it resumed"
        )
