"""The gate that keeps tests off the live model server, and the class it caught.

`from lib.osaurus_lib import call` binds a COPY of the function. Patching the
definition never reaches the importer, so the mock silently misses and the test
talks to the real server while reporting green. Twenty tests were doing exactly
that, across four separate aliases nobody had thought to patch.

These tests pin the gate and the fix. They are deliberately about mechanism, not
about any one caller, because the next value-import will be somewhere new.
"""

import socket

import pytest
from lib.llm.constants import DEFAULT_PORT

# Captured at import time, before any fixture wraps it.
_REAL_SOCKET_CONNECT = socket.socket.connect


@pytest.mark.llm_gate_selftest
class TestTheGateBlocksTheRealServer:
    def test_a_connection_to_the_llm_port_is_refused(self):
        """The gate is active in this very test, so the attempt must not succeed."""
        sock = socket.socket()
        try:
            with pytest.raises(BaseException) as exc:
                sock.connect(("127.0.0.1", DEFAULT_PORT))
            assert "real LLM server" in str(exc.value)
        finally:
            sock.close()

    def test_the_error_names_the_offending_test(self):
        """A bare 'connection blocked' in a 2000-test run tells you nothing."""
        sock = socket.socket()
        try:
            with pytest.raises(BaseException) as exc:
                sock.connect(("127.0.0.1", DEFAULT_PORT))
            assert "test_the_error_names_the_offending_test" in str(exc.value)
        finally:
            sock.close()

    def test_other_ports_are_untouched(self):
        """The gate must not break tests that use local sockets for other things."""
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        client = socket.socket()
        try:
            client.connect(("127.0.0.1", port))  # must not raise
        finally:
            client.close()
            server.close()

    def test_the_block_cannot_be_swallowed_by_a_broad_except(self):
        """lib/ catches Exception everywhere and would turn a violation into a dict.

        Deriving from BaseException is what stops a gate violation from being
        laundered into a normal error return.
        """
        sock = socket.socket()
        try:
            try:
                sock.connect(("127.0.0.1", DEFAULT_PORT))
            except Exception:  # noqa: BLE001 - the point of the test
                pytest.fail("the gate raised something `except Exception` can swallow")
            except BaseException as exc:
                assert "real LLM server" in str(exc)
        finally:
            sock.close()


class TestTheMockReachesEveryAliasOfTheTransport:
    """Object identity, not a hand-maintained list of module names."""

    def test_no_module_still_holds_the_real_call(self, mock_llm):
        """The regression that started this: four modules held unpatched aliases."""
        import sys

        import lib.osaurus_lib as osaurus

        real_call = mock_llm._real_call_for_test
        leaked = []
        for module in list(sys.modules.values()):
            name = getattr(module, "__name__", "")
            if not name.startswith(("eval", "weekend", "twitter", "rename", "lib")):
                continue
            try:
                members = list(vars(module).items())
            except Exception:
                continue
            for attr, value in members:
                if value is real_call:
                    leaked.append(f"{name}.{attr}")

        assert not leaked, (
            "these names still point at the real transport while MockLLM is "
            f"active, so anything calling them reaches the live server: {leaked}"
        )
        assert osaurus.call is not real_call

    def test_renamed_imports_are_covered_too(self, mock_llm):
        """`from lib.osaurus_lib import call as llm_call` hides behind a new name."""
        import eval.benchmark_quality as bq

        # `==` not `is`: attribute access on a bound method makes a new object
        # each time, while equality compares (function, instance).
        assert bq.llm_call == mock_llm.call

    def test_the_probe_and_the_eval_loop_share_the_mock(self, mock_llm):
        import eval.prefill as prefill
        import eval.run as run

        assert run.call == mock_llm.call
        assert prefill.call == mock_llm.call


@pytest.mark.real_llm
def test_the_opt_out_marker_removes_the_guard():
    """Integration tests must get the real socket, not a guarded one.

    Asserted by identity rather than by connecting: a test that proves the
    opt-out by actually reaching the server would need a server running, which
    is precisely the dependency the rest of the suite is now free of.
    """
    assert socket.socket.connect is _REAL_SOCKET_CONNECT, (
        "the gate is still installed on a @pytest.mark.real_llm test"
    )


def test_the_guard_is_installed_without_the_marker():
    """The other half of the same claim -- otherwise the assertion above is vacuous."""
    assert socket.socket.connect is not _REAL_SOCKET_CONNECT
