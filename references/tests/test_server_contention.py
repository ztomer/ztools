"""Two osaurus servers must announce themselves, per task, while measuring.

A second server does not queue behind the first -- it loads its own copy of the
weights. Two on a machine sized for one produce evictions, swapping, and requests the
server cancels itself with HTTP 499, which from the client is indistinguishable from a
slow model. That is how qwen3.8-27b was once recorded at 0.1 tok/s with a 423s cold
start, and those numbers are permanent because the recorders keep the SLOWEST reading.

`tools/osaurus_one.sh` already detects this correctly and exits 1. The gap was that
nothing called it during a run: it fires once before a model, then hours pass. A second
server did appear mid-run, sat idle burning CPU, and was found only because a human
happened to look. A correct check nobody invokes is not a gate.
"""

from unittest.mock import patch

import eval.signals as signals


class FakeProc:
    def __init__(self, name):
        self.info = {"name": name}


def procs(*names):
    return [FakeProc(n) for n in names]


class TestTheCounterCountsServersNotShellCommands:
    """Matched on process NAME, not a command-line substring.

    `pgrep -f /path/to/osaurus` also matches any shell whose command line quotes that
    path -- including the diagnostic commands you run while investigating. That is how
    a count of "3 servers" turned out to be one server and two greps.
    """

    def test_one_server_is_one(self):
        with patch.object(signals.psutil, "process_iter", return_value=procs("osaurus")):
            assert signals.count_osaurus_servers() == 1

    def test_two_servers_are_two(self):
        with patch.object(signals.psutil, "process_iter",
                          return_value=procs("osaurus", "osaurus")):
            assert signals.count_osaurus_servers() == 2

    def test_a_shell_quoting_the_binary_path_is_not_a_server(self):
        with patch.object(signals.psutil, "process_iter",
                          return_value=procs("osaurus", "zsh", "python3.14", "grep")):
            assert signals.count_osaurus_servers() == 1

    def test_no_server_is_zero(self):
        with patch.object(signals.psutil, "process_iter", return_value=procs("zsh")):
            assert signals.count_osaurus_servers() == 0


class TestTheWarningFiresOnlyWhenContended:
    def test_two_servers_warn(self):
        with patch.object(signals, "count_osaurus_servers", return_value=2):
            assert "2 osaurus servers" in signals.contended_server_warning("m", "t")

    def test_the_warning_names_the_model_and_task(self):
        """A bare 'two servers' cannot be traced back to which measurement it dirtied."""
        with patch.object(signals, "count_osaurus_servers", return_value=3):
            msg = signals.contended_server_warning("qwen3.8-27b-mxfp8", "taxes_synthesis")
        assert "qwen3.8-27b-mxfp8" in msg and "taxes_synthesis" in msg

    def test_the_warning_says_how_to_fix_it(self):
        with patch.object(signals, "count_osaurus_servers", return_value=2):
            assert "osaurus_one.sh --restart" in signals.contended_server_warning("m", "t")

    def test_one_server_is_silent(self):
        """Without this the warning could be firing on every task, which would train
        the reader to ignore it -- the failure mode of a gate that cries wolf."""
        with patch.object(signals, "count_osaurus_servers", return_value=1):
            assert signals.contended_server_warning("m", "t") == ""

    def test_an_undeterminable_count_is_silent(self):
        """-1 means psutil could not tell us. Warning on that would be a false alarm,
        and refusing to run would make the eval depend on introspection it may not have."""
        with patch.object(signals, "count_osaurus_servers", return_value=-1):
            assert signals.contended_server_warning("m", "t") == ""


class TestTheCheckIsWiredIntoTheTaskLoop:
    """Structural: the function existing is not the same as it being called. That
    distinction IS this bug -- osaurus_one.sh was correct the whole time."""

    def test_run_eval_calls_it(self):
        import inspect

        import eval.run_loop as run

        src = inspect.getsource(run)
        assert "contended_server_warning(" in src

    def test_it_is_called_inside_the_per_task_loop(self):
        """Once per MODEL would not catch a server that appears mid-run, which is
        exactly what happened."""
        import inspect

        import eval.run as run

        src = inspect.getsource(run.run_eval)
        loop = src.index("for task_name, task_cfg in tasks.items():")
        assert "contended_server_warning(" in src[loop:], (
            "the check must run per task, not once before the loop"
        )
