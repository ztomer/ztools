"""Structural gates blocking real network/process reach: the live LLM server,
real browsers/cookies, and the real osaurus process.

Split out of conftest.py for the 500-line cap (no test exemption; see
CLAUDE.md). Imported by name into conftest.py so pytest's fixture discovery
finds them there.
"""

import pytest


class _RealLLMServerBlocked(BaseException):
    """Raised when a test reaches for the live model server.

    Derived from BaseException, not Exception, so the `except Exception` handlers
    throughout lib/ cannot quietly turn a gate violation into an error dict.
    """


@pytest.fixture(autouse=True)
def no_real_llm_server(request):
    """Structural gate: no test may reach the real model server.

    test_prefill_measurement patched `eval.run.call` after that function had
    moved to `eval.prefill`. patch.object happily bound a name nobody read, the
    probe called the live server, and the only trace was a stray "HTTP 404:
    Model 'some-model' is not installed" in the captured log. An unrelated
    assertion caught it; nothing in the suite would have.

    The pre-push gate sets OLLAMA_BASE_URL=http://127.0.0.1:1, which looks like
    it covers this and does not: lib/osaurus_lib.py builds its URL from its own
    DEFAULT_HOST/DEFAULT_PORT and never reads that variable.

    Blocking at the socket layer rather than at `requests` is deliberate — it
    catches urllib, httpx and anything else a future caller reaches for, and it
    cannot be defeated by patching the wrong module, which is the exact failure
    this gate exists to catch.

    Integration tests that genuinely need a server opt out with
    @pytest.mark.real_llm; they already skip themselves when none is running.
    Tests OF this gate use @pytest.mark.llm_gate_selftest, which keeps the guard
    active but accepts the deliberate attempt instead of failing on it.
    """
    if "real_llm" in request.keywords:
        yield
        return

    import socket
    from unittest.mock import patch

    from lib.llm.constants import DEFAULT_PORT

    real_connect = socket.socket.connect
    test_id = request.node.nodeid
    attempts = []

    def guarded_connect(sock, address, *args, **kwargs):
        port = address[1] if isinstance(address, tuple) and len(address) > 1 else None
        if port == DEFAULT_PORT:
            attempts.append(f"{address[0]}:{port}")
            raise _RealLLMServerBlocked(
                f"{test_id} tried to reach the real LLM server at "
                f"{address[0]}:{port}. A mock is missing or is patching the "
                f"wrong module — patch the module that OWNS the function "
                f"(e.g. eval.run_transport.call, not eval.run.call). Integration "
                f"tests mark themselves @pytest.mark.real_llm."
            )
        return real_connect(sock, address, *args, **kwargs)

    with patch.object(socket.socket, "connect", guarded_connect):
        yield

    # Raising is not enough on its own. lib/osaurus_lib.py::call catches broad
    # exceptions and returns {"error": ...}, and urllib3 wraps whatever connect
    # raises into a ConnectionError on the way out — so a blocked call is
    # indistinguishable from a server that happens to be down, and the test goes
    # green having proved nothing. Failing here, after the test body, is what
    # the caller cannot swallow.
    if attempts and "llm_gate_selftest" not in request.keywords:
        pytest.fail(
            f"{test_id} attempted {len(attempts)} connection(s) to the real LLM "
            f"server ({', '.join(sorted(set(attempts)))}). The call was blocked, "
            f"but the code under test swallowed the error and carried on — so "
            f"whatever this test asserted, it did not assert it against a mock."
        )


@pytest.fixture(autouse=True)
def no_real_browsers_or_cookies(request):
    """Structural gate: no test may launch a browser or read real cookies.

    test_twitter_browser_no_playwright patched only `sync_playwright`, so once
    camoufox became the preferred backend that test started launching a real
    Firefox and reading the developer's own x.com session. Pinning the backend
    and stubbing discovery here makes that impossible for every test, present
    and future, rather than relying on each one remembering.

    Tests that genuinely exercise cookie discovery opt out with
    @pytest.mark.real_cookie_discovery.

    Pinning the backend only decided WHICH browser would launch; the launch
    itself stayed reachable, so the guarantee rested on every test remembering
    to patch `sync_playwright`/`launch_camoufox` locally. Both are stubbed here
    to raise instead, which is what makes this a gate rather than a convention.
    """
    from unittest.mock import patch

    import twitter.browser as browser
    import twitter.browser_launch as launch

    def _blocked(*args, **kwargs):
        raise RuntimeError(
            "real browser launch blocked in tests — patch open_browser "
            "(or twitter.browser.sync_playwright) in the test itself"
        )

    patches = [
        patch.object(launch, "BROWSER_BACKEND", launch.BACKEND_CHROMIUM),
        patch.object(browser, "sync_playwright", _blocked),
        patch.object(launch, "launch_camoufox", _blocked),
        patch.object(launch, "launch_camoufox_persistent", _blocked),
        # twitter.browser imported launch_camoufox by value, so the module-level
        # name there needs blocking too.
        patch.object(browser, "launch_camoufox", _blocked),
    ]
    if "real_cookie_discovery" not in request.keywords:
        import twitter.cookies as ck
        import twitter.cookies_firefox as ckf

        patches += [
            patch.object(ck, "_get_chrome_keychain_key", return_value=b"k" * 16),
            patch.object(ck, "_read_profile_cookies", return_value=[]),
            patch.object(ckf, "firefox_profile_dbs", return_value=[]),
        ]
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in patches:
            p.stop()


@pytest.fixture(autouse=True)
def no_real_server_restart(request):
    """Structural gate: no test may restart or quit the real osaurus server.

    `flush_between_models` used to hand-roll its restart with `osascript` and
    `open -n -a osaurus`. When it was changed to delegate to
    `tools/osaurus_one.sh --restart`, two tests in TestFlushBetweenModels that had
    never patched `subprocess.run` began executing the REAL script: a unit run
    killed the developer's osaurus, waited out a relaunch, and took 89 seconds.
    Nothing failed loudly -- the tests just got slow and the server pid changed
    underneath whatever else was using it.

    `no_real_llm_server` blocks the SOCKET, so it could not see this: the damage
    was done by spawning a process, not by opening a connection. Blocking the
    spawn is the missing half.

    Tests that need to assert on the restart path patch `subprocess.run`
    themselves, which replaces this wrapper and is exactly the intended usage.
    """
    import subprocess
    from unittest.mock import patch

    # test_gpu_lock_shell.py drives the REAL osaurus_one.sh on purpose, against a
    # stubbed PATH and a tmp-dir lock, so it never reaches the developer's server.
    # This gate matched on the script NAME and blocked all eight of those tests.
    # A blanket ban that also bans the legitimate case gets weakened or deleted;
    # an explicit opt-out keeps the ban meaningful. Same shape as
    # @pytest.mark.real_cookie_discovery above.
    if "sandboxed_server_script" in request.keywords:
        yield
        return

    real_run = subprocess.run
    forbidden = ("osaurus_one.sh", "quit app \"osaurus\"", "-a osaurus")

    def guarded_run(*args, **kwargs):
        argv = args[0] if args else kwargs.get("args", [])
        rendered = " ".join(str(a) for a in argv) if isinstance(argv, (list, tuple)) else str(argv)
        if any(token in rendered for token in forbidden):
            raise RuntimeError(
                "real osaurus restart blocked in tests — this would kill the "
                f"developer's running server. Patch subprocess.run in the test. Got: {rendered}"
            )
        return real_run(*args, **kwargs)

    with patch("subprocess.run", guarded_run):
        yield
