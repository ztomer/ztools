"""The two osascript quit call sites must not quit a server a peer is measuring on.

There are exactly two places in this repo that run `quit app "osaurus"`:
lib/osaurus_server.py::_kill_osaurus (reached from the twitter summariser, the
weekend planner and check_server_or_die) and eval/cli_runtime.py::
flush_between_models. Both were written for a machine where the only client was
this process. Several agent sessions now run here at once, and quitting the
server one of them is measuring against evicts its model mid-measurement: the peer
sees HTTP 499 request_cancelled and records a rate an order of magnitude low -- as
a CLEAN sample, because `machine_is_uncontended()` reads swap and compressor and
cannot see the GPU at all.

They REFUSE rather than queue. These callers are trying to recover a server for
their own request, not to reserve the GPU, and blocking a tweet summary for the
hours an eval runs helps nobody. Refusing degrades with a stated reason and leaves
the peer's measurement intact.
"""

import os
import re
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from lib import gpu_lock

_REAL_RUN = subprocess.run
_REAL_POPEN = subprocess.Popen


def _spy_popen(launches):
    """Same shape as `_spy_run`, and needed for the same reason.

    `subprocess.run` is implemented on top of `Popen`, so a blanket Popen mock
    silently breaks the lock's `ps` probe even when `run` itself was left alone.
    That is not theoretical -- it turned this file's sharpest assertion red for a
    reason unrelated to the code under test.
    """
    def popen(cmd, *args, **kwargs):
        parts = cmd if isinstance(cmd, (list, tuple)) else [cmd]
        if parts and str(parts[0]) == "ps":
            return _REAL_POPEN(cmd, *args, **kwargs)
        launches.append(" ".join(str(part) for part in parts))
        return MagicMock(pid=4242)
    return popen


def _spy_run(calls):
    """A `subprocess.run` stand-in that records commands but lets `ps` through.

    Patching `subprocess.run` wholesale reaches every caller in the process, and
    the lock's own liveness probe is one of them: it shells out to `ps` to compare
    a recorded start time against the live one. A blanket mock breaks that probe
    and makes the lock report every owner as an impostor -- so the test would pass
    or fail for a reason that has nothing to do with the call site under test.
    """
    def run(cmd, *args, **kwargs):
        parts = cmd if isinstance(cmd, (list, tuple)) else [cmd]
        if parts and str(parts[0]) == "ps":
            return _REAL_RUN(cmd, *args, **kwargs)
        calls.append(" ".join(str(part) for part in parts))
        return MagicMock(returncode=0, stdout="")
    return run


@pytest.fixture(autouse=True)
def _clean_gpu_lock():
    """Clear the MACHINE-WIDE gpu lock around every test in this file.

    _peer_holds() writes a real owner file into gpu_lock.lock_dir() -- a shared
    path under /tmp, not a tmp_path fixture -- and nothing removed it. It records
    os.getppid() as the owner, a process that outlives the whole suite, so the
    state leaked into every later test AND into whatever else on this machine
    consults that lock, which is the entire point of a machine-wide lock.

    Clearing BEFORE as well as after matters: a run killed mid-test leaves the
    file behind, and every later run would start under a phantom holder.
    """
    owner = os.path.join(gpu_lock.lock_dir(), "owner")

    def _clear():
        try:
            os.remove(owner)
        except FileNotFoundError:
            pass

    _clear()
    yield
    _clear()


def _peer_holds(label="peer eval"):
    """Make a LIVE process that is not us the recorded owner.

    os.getppid() is the shape of a peer agent session: a real PID, really alive,
    really not this process.
    """
    d = gpu_lock.lock_dir()
    os.makedirs(d, exist_ok=True)
    pid = os.getppid()
    with open(os.path.join(d, "owner"), "w") as fh:
        fh.write(f"{pid}\n{gpu_lock._start_time(pid)}\n{label}\n")


class TestKillOsaurusRefusesUnderAPeer:
    def test_it_does_not_run_osascript_when_a_peer_holds_the_gpu(self):
        import lib.osaurus_server as srv

        _peer_holds()
        calls = []
        with patch.object(srv.subprocess, "run", _spy_run(calls)):
            assert srv._kill_osaurus() is False
        assert calls == []

    def test_it_quits_normally_when_the_gpu_is_free(self, tmp_path):
        """The refusal must be narrow. A lock that blocked recovery on an idle
        machine would be worse than no lock: every wedged server would stay
        wedged."""
        import lib.osaurus_server as srv

        calls = []
        with patch.object(srv.subprocess, "run", _spy_run(calls)), \
             patch.object(srv, "PID_FILE", Path(tmp_path) / "absent.pid"):
            assert srv._kill_osaurus() is True
        assert any("osascript" in call for call in calls)

    def test_our_own_hold_does_not_block_our_own_restart(self, tmp_path):
        """An eval that holds the GPU is entitled to restart the server it owns.
        If its own hold read as foreign it could never flush between models --
        the lock would break the one workflow it exists to protect."""
        import lib.osaurus_server as srv

        with gpu_lock.gpu_lock("my own eval"):
            calls = []
            with patch.object(srv.subprocess, "run", _spy_run(calls)), \
                 patch.object(srv, "PID_FILE", Path(tmp_path) / "absent.pid"):
                assert srv._kill_osaurus() is True
            assert any("osascript" in call for call in calls)

    def test_the_refusal_says_who_and_why(self, caplog, capsys):
        import lib.osaurus_server as srv

        _peer_holds("eval qwen3.8-27b (pid 4321)")
        srv._kill_osaurus()
        captured = capsys.readouterr()
        log_text = f"{caplog.text} {captured.err} {captured.out}"
        assert "qwen3.8-27b" in log_text


class TestRestartServerRefusesUnderAPeer:
    def test_a_refused_quit_does_not_fall_through_to_a_relaunch(self):
        """THE SHARPEST EDGE. Launching while a peer still holds a live server is
        how a SECOND osaurus appears, and two on a machine sized for one is the
        original contamination -- each loads its own copy of the weights, both
        swap, and both sets of numbers are ruined. A refusal that skipped only the
        quit would make this recovery path CAUSE the failure it recovers from."""
        import lib.osaurus_server as srv

        _peer_holds()
        launches = []
        with patch.object(srv.subprocess, "Popen", _spy_popen(launches)):
            assert srv.restart_server() is False
        assert launches == [], f"a refused quit still launched a server: {launches}"

    def test_it_restarts_normally_when_the_gpu_is_free(self, tmp_path):
        import lib.osaurus_server as srv

        launches = []
        with patch.object(srv.subprocess, "run", _spy_run([])), \
             patch.object(srv.subprocess, "Popen", _spy_popen(launches)), \
             patch.object(srv, "_wait_until_down", return_value=True), \
             patch.object(srv, "is_server_running", return_value=True), \
             patch.object(srv, "PID_FILE", Path(tmp_path) / "absent.pid"), \
             patch.object(srv.time, "sleep"):
            assert srv.restart_server() is True
        assert len(launches) == 1


class TestFlushBetweenModelsRefusesUnderAPeer:
    def test_it_neither_quits_nor_launches_when_a_peer_holds_the_gpu(self):
        """`open -n -a osaurus` is unconditional: it launches a SECOND instance
        whether or not the first is still there. Skipping only the quit would
        leave the more destructive half of this path armed."""
        import eval.cli_runtime as rt

        _peer_holds("peer eval")
        out = MagicMock()
        calls = []
        with patch.object(rt, "call", return_value={"error": "boom"}), \
             patch("subprocess.run", _spy_run(calls)), \
             patch.object(rt.time, "sleep"):
            rt.flush_between_models("m1", "m2", out=out)
        assert calls == [], f"quit/launch ran under a peer's measurement: {calls}"

    def test_it_says_who_holds_the_gpu(self):
        import eval.cli_runtime as rt

        _peer_holds("eval nemotron (pid 77)")
        out = MagicMock()
        with patch.object(rt, "call", return_value={"error": "boom"}), \
             patch("subprocess.run", _spy_run([])), \
             patch.object(rt.time, "sleep"):
            rt.flush_between_models("m1", "m2", out=out)
        printed = " ".join(str(c) for c in out.print.call_args_list)
        assert "nemotron" in printed

    def test_it_restarts_normally_when_the_gpu_is_free(self):
        import eval.cli_runtime as rt

        out = MagicMock()
        calls = []
        with patch.object(rt, "call", return_value={"error": "boom"}), \
             patch("subprocess.run", _spy_run(calls)), \
             patch.object(rt.time, "sleep"), \
             patch("requests.Session") as session:
            session.return_value.__enter__.return_value.get.return_value = MagicMock(
                status_code=200)
            rt.flush_between_models("m1", "m2", out=out)
        commands = " ".join(calls)
        assert "osaurus_one.sh" in commands and "--restart" in commands, commands
        assert "open -n" not in commands, (
            f"`open -n` launches a SECOND osaurus unconditionally: {commands}"
        )


class TestTheEvalEntryPointHoldsTheGpu:
    def test_it_acquires_and_names_the_run(self):
        import eval.cli_runtime as rt

        rt.hold_gpu_for_eval("eval qwen3.8-27b", out=MagicMock())
        assert gpu_lock.holder() == "eval qwen3.8-27b (pid %d)" % os.getpid()

    def test_the_hold_is_released_on_a_signal(self):
        """Release path #1 of three. A Ctrl-C'd eval must not wedge every later
        eval on the machine, and the signal handler is the only path that runs
        on SIGTERM from a supervising script."""
        import eval.cli_runtime as rt
        import lib.signal_handling as sh

        sh.reset_signal_state()
        rt.hold_gpu_for_eval("eval m1", out=MagicMock())
        sh._run_cleanup()
        assert gpu_lock.holder() is None
        sh.reset_signal_state()

    def test_the_hold_is_released_at_process_exit(self):
        """Release path #2. Covers a normal return and an unhandled exception,
        neither of which reaches the signal handler."""
        import atexit

        import eval.cli_runtime as rt

        with patch.object(atexit, "register") as register:
            rt.hold_gpu_for_eval("eval m1", out=MagicMock())
        assert gpu_lock.release in [c.args[0] for c in register.call_args_list]

    def test_main_claims_the_gpu_before_it_reaches_the_server(self, monkeypatch):
        """The wiring, not just the helper. `main` is the only entry point both
        `python3 -m eval` and the `ev`/`oeval` console scripts go through, so a
        helper nobody calls would leave every eval unprotected while every test
        of the helper stayed green."""
        import sys

        import eval.cli as cli

        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        held = []
        with patch.object(cli, "hold_gpu_for_eval", lambda label, **kw: held.append(label)), \
             patch.object(cli, "init_config"), \
             patch.object(cli, "is_server_running", return_value=False), \
             patch.object(cli, "get_models", return_value=[]), \
             patch.object(cli, "build_tasks_from_model", return_value={}):
            with pytest.raises(SystemExit):
                cli.main()
        assert held, "main() reached the server without claiming the GPU"

    def test_it_refuses_to_measure_under_a_peer(self):
        """No degrade here, unlike the quit call sites. A run that cannot get the
        GPU has nothing useful to do: proceeding would produce a full set of
        results that look ordinary and are quietly wrong."""
        import eval.cli_runtime as rt

        _peer_holds("peer eval")
        with pytest.raises(gpu_lock.GpuBusy):
            rt.hold_gpu_for_eval("eval m1", out=MagicMock(), timeout=0)


class TestTheHeartbeatIsWiredIntoTheTaskLoop:
    def test_run_eval_beats_once_per_task(self):
        """The ceiling measures progress, so something has to report progress.
        An unbeaten lock expires under a healthy multi-hour run and gets handed
        to a peer that then restarts the server."""
        import eval.run as er
        from eval import run_attempt, run_loop, run_transport

        tasks = {
            "t1": {"messages": [{"role": "user", "content": "a"}]},
            "t2": {"messages": [{"role": "user", "content": "b"}]},
        }
        with patch.object(run_loop.gpu_lock, "heartbeat") as beat, \
             patch.object(run_transport, "call", return_value={"content": "{}", "elapsed": 1.0}), \
             patch.object(run_loop, "measure_prefill_rate", return_value=None), \
             patch.object(run_attempt, "save_output"), \
             patch.object(run_loop, "contended_server_warning", return_value=""):
            er.run_eval("m1", tasks=tasks, verbose=False, timeout=1)
        assert beat.call_count >= len(tasks)


class TestNoUnguardedServerMutationCanBeAdded:
    """The CLASS-level gate, not another instance.

    Fixing the two known quit sites leaves the failure mode alive: the next tool
    that reacts to a slow server by quitting it reintroduces exactly this bug, and
    every test above stays green because none of them knows the new site exists.
    So the gate is on the SET of places that can stop or start the server. Adding
    one is fine -- guard it and add it here, which is the moment to think about
    whether it should refuse under a peer.

    Deliberately a repo scan rather than a review checklist: rule #3, gates are
    structural. The commands are matched as literal text because that is what a
    new call site will be copy-pasted as.
    """

    ROOT = Path(__file__).resolve().parent.parent.parent

    # Every command that stops or starts the machine-wide server.
    #
    # Word-anchored, because the bare substring "osaurus serve" also matches the
    # prose "osaurus servers" and "osaurus serves from MODELS_DIR" -- and an
    # allowlist gate that fires on prose gets padded with innocent files until it
    # stops meaning anything.
    MUTATORS = (
        r'quit app "osaurus"',
        r"open -n -a",
        r"\bosaurus serve\b",
        r"\bosaurus stop\b",
    )

    # Files allowed to contain one, each with the guard that makes it safe.
    GUARDED = {
        "references/lib/osaurus_server.py": "gpu_lock.foreign_holder()",
        "references/eval/cli_runtime.py": "gpu_lock.foreign_holder()",
        "tools/osaurus_one.sh": "gpu_lock_acquire",
        "tools/gpu_lock.sh": None,          # documents them in its header
        "references/lib/gpu_lock.py": None,
    }

    @staticmethod
    def code_only(text):
        """Drop whole-line `#` comments before scanning.

        The gate is about call sites, and a comment cannot execute. Without this
        the scan flags any file that MENTIONS the commands -- including the
        comments this change added to explain why they are dangerous -- which
        forces innocent files onto the allowlist until the allowlist stops
        meaning "these are the places that mutate the server". Prose is not the
        hazard; a line that runs is.
        """
        return "\n".join(
            line for line in text.split("\n") if not line.lstrip().startswith("#")
        )

    @classmethod
    def _tracked(cls):
        """Git-tracked paths only.

        rglob walked `build/`, which is gitignored and held an 8-day-old COPY of
        eval/cli.py and friends from a stale wheel build. The gate failed naming
        three files nobody had touched, on a checkout that was correct. A gate
        about the source that SHIPS must look at what is tracked, or the next
        person's leftover build output fails it for them too.
        """
        import subprocess

        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=cls.ROOT, capture_output=True, text=True, check=False
        )
        return {p for p in out.stdout.split("\0") if p}

    def _sites(self):
        found = {}
        tracked = self._tracked()
        for path in self.ROOT.rglob("*"):
            if path.suffix not in (".py", ".sh") or not path.is_file():
                continue
            rel = path.relative_to(self.ROOT).as_posix()
            if rel not in tracked:
                continue
            if "/tests/" in rel or rel.startswith(".venv/") or "/.venv/" in rel:
                continue
            text = path.read_text(errors="ignore")
            if any(re.search(pattern, self.code_only(text)) for pattern in self.MUTATORS):
                found[rel] = text
        return found

    def test_every_place_that_stops_or_starts_the_server_is_known_and_guarded(self):
        sites = self._sites()
        unexpected = sorted(set(sites) - set(self.GUARDED))
        assert not unexpected, (
            "these files stop or start the machine-wide osaurus server but are not "
            f"in the guarded set: {unexpected}. Several agent sessions run on this "
            "Mac at once; an unguarded site quits a server a peer is measuring "
            "against, and the ruined timing is filed as a CLEAN sample because the "
            "sample guard reads swap and compressor, not the GPU. Consult "
            "gpu_lock.foreign_holder() (or gpu_lock_acquire in bash) and add the "
            "file here."
        )
        for rel, guard in self.GUARDED.items():
            if guard is None or rel not in sites:
                continue
            assert guard in sites[rel], f"{rel} lost its GPU-lock guard"

    def test_the_scan_would_notice_a_new_site(self):
        """Calibration. A scan that matched nothing would pass this file forever
        while proving nothing, which is the failure mode of every allowlist gate."""
        assert set(self._sites()) >= {
            "references/lib/osaurus_server.py",
            "references/eval/cli_runtime.py",
            "tools/osaurus_one.sh",
        }

    def test_comment_stripping_did_not_blunt_the_scan(self):
        """The other half of the calibration: prove the exemption is narrow.

        Stripping comments is the kind of change that quietly turns a gate into a
        no-op -- one over-broad rule and every real call site is exempt too. So
        assert both directions on the same command: executable, caught; commented
        out, ignored."""
        pattern = self.MUTATORS[0]
        executable = '    subprocess.run(["osascript", "-e", \'quit app "osaurus"\'])'
        commented = "    # " + executable.strip()
        assert re.search(pattern, self.code_only(executable))
        assert not re.search(pattern, self.code_only(commented))
