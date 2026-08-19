"""
Pytest configuration: shared fixtures for all tests.
"""

import sys
from pathlib import Path

import pytest
from lib.testing import MockLLM

sys.path.insert(0, str(Path(__file__).parent.parent))


# Capture references to lib.mlx_lib functions at conftest load time.
# This MUST happen before any mock patches them.
import lib.mlx_lib  # noqa: E402
import lib.mlx_vlm  # noqa: E402

_REAL_MLX_FUNCTIONS = {
    "call": lib.mlx_lib.call,
    "call_mlx": lib.mlx_lib.call_mlx,
    "call_mlx_vlm": lib.mlx_vlm.call_mlx_vlm,
    "probe_mlx_vlm_loadable": lib.mlx_vlm.probe_mlx_vlm_loadable,
    "process_mlx_content": lib.mlx_lib.process_mlx_content,
    "find_mlx_model": lib.mlx_lib.find_mlx_model,
    "find_text_mlx_model": lib.mlx_lib.find_text_mlx_model,
}


@pytest.fixture(autouse=True)
def reset_global_sessions():
    """Reset persistent session variables in client and osaurus_lib before and after each test."""
    import lib.llm.client
    import lib.osaurus_lib

    lib.llm.client.reset_session()
    lib.osaurus_lib.reset_session()
    yield
    lib.llm.client.reset_session()
    lib.osaurus_lib.reset_session()


@pytest.fixture
def real_mlx_functions():
    """Return real (unmocked) lib.mlx_lib functions.

    Captures references at conftest load time, before any mock patches.
    """
    return _REAL_MLX_FUNCTIONS


@pytest.fixture
def mock_llm():
    """Fixture that patches all LLM functions with a MockLLM provider.

    Usage:
        def test_something(mock_llm):
            import eval.run as er
            from unittest.mock import patch
            with patch.object(er, "call", mock_llm.call):
                result = er.run_eval(...)
    """
    mock = MockLLM()
    mock.patch_all()
    yield mock
    mock.unpatch()


@pytest.fixture
def mock_llm_osaurus():
    """Same as mock_llm but only patches lib.osaurus_lib."""
    mock = MockLLM()
    mock.patch_osaurus()
    yield mock
    mock.unpatch()


# Legacy fixtures used by existing test files

import json  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402


class _MockOsaurusHandler(BaseHTTPRequestHandler):
    def log_message(self, *args, **kwargs):  # silence
        pass

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/v1/models"):
            self._send({"data": [{"id": "foundation"}]})
        else:
            self._send({"error": "not found"}, status=404)

    def do_POST(self):
        if self.path.endswith("/v1/chat/completions"):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                req = json.loads(raw)
            except Exception:
                req = {}
            content = json.dumps({"ok": True, "model": req.get("model", "foundation")})
            self._send(
                {
                    "choices": [{"message": {"content": content, "role": "assistant"}}],
                    "model": req.get("model", "foundation"),
                    "usage": {},
                }
            )
        else:
            self._send({"error": "not found"}, status=404)


@pytest.fixture
def mock_osaurus_server():
    """Yield base URLs for a running mock Osaurus server (and a down-port)."""
    server = HTTPServer(("127.0.0.1", 0), _MockOsaurusHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield {"up": f"http://127.0.0.1:{port}", "down": "http://127.0.0.1:1"}
    server.shutdown()


@pytest.fixture
def mock_llm_response():
    return {
        "json_with_activities": {
            "activities": [
                {"name": "Test Activity 1", "location": "Toronto", "target_ages": "6-12"},
                {"name": "Test Activity 2", "location": "Vaughan", "target_ages": "8-14"},
            ]
        },
        "json_with_fixed_activities": {
            "fixed_activities": [
                {
                    "name": "ROM",
                    "location": "Toronto",
                    "target_ages": "6-12",
                    "price": "$25",
                    "weather": "indoor",
                },
            ]
        },
        "json_with_transient_events": {
            "transient_events": [
                {"name": "Spring Festival", "location": "Vaughan", "day": "Saturday"},
            ]
        },
        "qwen_thinking_response": """Here's a thinking process:
1. Analyze the request
2. Formulate response

Output Generation.
{"activities": [{"name": "Test Event"}]}
stats:123""",
        "twitter_response": """Here's a thinking process:
Think about this carefully.

Output: ## Summary
- Main point
- Another point

stats:456""",
    }


@pytest.fixture
def sample_events_data():
    return """- Event 1 (Toronto): Details here
- Event 2 (Vaughan): More details"""


@pytest.fixture
def sample_venues_data():
    return """- Venue 1 (123 Main St): Great place
- Venue 2 (456 Oak Ave): Another great place"""


@pytest.fixture
def sample_tweets():
    return [
        {"screen_name": "user1", "text": "Test tweet 1", "created_at": "2026-04-21"},
        {"screen_name": "user2", "text": "Test tweet 2", "created_at": "2026-04-21"},
    ]


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
                f"(e.g. eval.prefill.call, not eval.run.call). Integration "
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
def _gpu_lock_never_touches_the_real_one(tmp_path, monkeypatch):
    """Structural gate: no test may read or write the machine-wide GPU lock.

    /tmp/mac-osaurus-gpu.lock is shared by every checkout, worktree and agent
    session on this Mac, and a real eval may be holding it right now. Without
    this redirect the suite would be coupled to that: tests of the quit-refusal
    path would pass or fail depending on whether a colleague session happened to
    be measuring, and a test that acquires would BLOCK a real eval -- the exact
    harm the lock exists to prevent, caused by the tests for it.

    Function-scoped so each test gets a clean, empty lock, and the module's
    `_held` flag is reset on both sides: it is process-global, so one test that
    acquires would otherwise leave every later test believing it holds the lock.
    """
    import lib.gpu_lock as gl

    monkeypatch.setenv(gl.DIR_ENV, str(tmp_path / "gpu.lock"))
    monkeypatch.delenv(gl.OWNER_ENV, raising=False)
    gl._held = False
    yield
    gl._held = False
    os.environ.pop(gl.OWNER_ENV, None)


@pytest.fixture(autouse=True, scope="session")
def _signals_files_stay_clean(tmp_path_factory):
    """Structural gate: `pytest` must not dirty tracked config.

    eval/signals.py and weekend/llm.py persist learned per-model timeouts into
    conf/eval_signals.json and conf/phase_signals.json. Both are tracked, so
    exercising those code paths rewrote them on every test run and left the
    working tree dirty. Redirect both at a tmp dir for the whole session.
    """
    from unittest.mock import patch

    tmp = tmp_path_factory.mktemp("signals")
    import eval.signals as eval_signals
    import weekend.llm as weekend_llm

    with patch.object(eval_signals, "EVAL_SIGNALS_PATH", tmp / "eval_signals.json"), \
         patch.object(weekend_llm, "PHASE_SIGNALS_PATH", tmp / "phase_signals.json"), \
         patch.object(weekend_llm, "EXTRACT_SIGNALS_PATH", tmp / "extract_signals.json"):
        yield


@pytest.fixture(autouse=True, scope="session")
def _eval_artefacts_stay_in_tmp(tmp_path_factory):
    """Structural gate: nothing may write eval artefacts into the real config dir.

    `default_eval_dir()` returns ~/.config/ztools, and its docstring says callers
    "take eval_dir as a parameter and fall back to this, so tests hand in a tmp dir".
    That is discipline, not a gate, and discipline failed: eval_history.json in the
    developer's own config directory accumulated `m1`, `m2` and `mock-model` entries
    from the suite. report_history HAS a test-model filter, but it matches the
    prefixes ("mock", "test-", "fake") and a fixture called `m1` matches none of them
    -- a name allowlist always lags behind fixture naming, which is why this is a path
    redirect instead of another name.

    Patched on every IMPORTER, not on eval.report_core. Each module does
    `from eval.report_core import default_eval_dir` at import time, so patching the
    source module rebinds a name nobody reads -- the same seam hazard this repo
    documents for patch.object across a module split.
    """
    from unittest.mock import patch

    import eval.cli_results
    import eval.outputs
    import eval.report_history
    import eval.report_metrics

    tmp = tmp_path_factory.mktemp("eval_artefacts")
    patches = [
        patch.object(mod, "default_eval_dir", lambda: tmp)
        for mod in (
            eval.report_history,
            eval.report_metrics,
            eval.cli_results,
            eval.outputs,
        )
    ]
    for p in patches:
        p.start()
    try:
        yield tmp
    finally:
        for p in patches:
            p.stop()


@pytest.fixture(autouse=True, scope="session")
def _saved_outputs_stay_in_tmp(tmp_path_factory):
    """Structural gate: saved eval outputs must not land in the real config dir.

    run_eval now writes each model's raw answer under ~/.config/ztools/outputs so
    a scorer can be questioned without re-running the model. Every test that
    calls run_eval with a fake model therefore wrote there too -- the suite left
    outputs/m, outputs/m1 and outputs/mock-model in the developer's own config
    directory within minutes of the feature landing.

    Redirected by environment variable because that is the seam production reads;
    patching a module attribute would miss any caller that imported the path by
    value. The existing tracked-config gate could not have caught this: it
    digests conf/ and docs/, and this escapes to $HOME.
    """
    tmp = tmp_path_factory.mktemp("eval_outputs")
    previous = os.environ.get("EVAL_OUTPUT_DIR")
    os.environ["EVAL_OUTPUT_DIR"] = str(tmp)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("EVAL_OUTPUT_DIR", None)
        else:
            os.environ["EVAL_OUTPUT_DIR"] = previous


# Files a LONG-RUNNING TOOL legitimately rewrites while the suite is running, and
# which a dedicated fixture already stops the tests themselves from touching.
#
# `_signals_files_stay_clean` redirects every one of these at a tmp dir for the
# whole session, so a change to the real file cannot have come from a test -- it
# came from an `ev` run in another terminal, which updates eval_signals.json after
# every task. Digesting them anyway made the pre-push hook fail for a reason no
# amount of reading the diff could fix, on a machine where a sweep can run for ten
# hours. Coverage is not lost: the redirect fixture is the stronger, more specific
# gate, and `test_the_suite_cannot_write_into_the_real_config_dir` is the pattern
# for proving such a redirect is in force.
_CONCURRENTLY_WRITTEN = {"eval_signals.json", "phase_signals.json", "extract_signals.json"}


def _tracked_config_digest() -> dict:
    """Hash every tracked file the tools can write back to."""
    import hashlib

    from lib.paths import conf_dir, repo_path

    digest = {}
    targets = sorted(conf_dir().rglob("*.toml")) + sorted(conf_dir().rglob("*.json"))
    baseline = repo_path("docs", "eval_baseline.json")
    if baseline is not None:
        targets.append(baseline)
    for path in targets:
        if path.is_file() and path.name not in _CONCURRENTLY_WRITTEN:
            digest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


@pytest.fixture(autouse=True, scope="session")
def _tracked_config_stays_clean():
    """Structural gate: no test may write to the real conf/ or docs/ baseline.

    Before the layout fix these writes silently landed in a nonexistent
    `references/conf`, so tests that exercised `update_config` or
    `save_baseline` looked harmless. With paths resolving correctly they hit
    the tracked files for real — `pytest` rewrote conf/config.toml on the
    developer's checkout. Tests must point the writers at tmp (ZTOOLS_CONF or
    a patched module attribute); this fails the run if one does not.
    """
    before = _tracked_config_digest()
    yield
    after = _tracked_config_digest()
    changed = sorted(
        set(before) ^ set(after) | {p for p in before.keys() & after.keys() if before[p] != after[p]}
    )
    # Attributed by pytest to whichever test ran last, which is not evidence
    # about that test: this is a session fixture. And the suite is not the only
    # thing that writes here -- an `ev` run in another terminal updates
    # conf/eval_signals.json after every task, which trips this gate with a
    # message accusing an unrelated test. Say so, rather than sending the reader
    # to audit code that did nothing.
    assert not changed, (
        "tracked config files changed during the test session:\n  "
        + "\n  ".join(changed)
        + "\n\nEither a test wrote to the real conf/ instead of tmp (point it at "
        "ZTOOLS_CONF or patch the module attribute), or another process wrote "
        "them while the suite ran -- a concurrent `ev` run updates "
        "conf/eval_signals.json. Check `git diff` on the listed files to tell "
        "which: eval writes latency and _capabilities records."
    )


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


@pytest.fixture(autouse=True)
def deterministic_machine_contention():
    """Structural gate: no test outcome may depend on the developer's memory pressure.

    `add_sample` tags every sample by calling `machine_is_uncontended()`, which
    reads live swap and compressor figures. That was harmless while the flag only
    steered a median, and stopped being harmless the moment `_derived_timeout`
    began consulting it: the timeout tests in test_prefill_measurement.py record
    rates and assert on the derived timeout, so they PASSED on a quiet machine and
    FAILED on a busy one -- three of them went red purely because the compressor
    was at 18GB while the suite ran.

    Pinned clean here. Tests that exercise contention patch `psutil`/`vm_stat`
    themselves (test_self_correcting_samples.py) or pass `clean=` to `add_sample`
    explicitly, both of which still work: those import the function by value or
    bypass it entirely.
    """
    from unittest.mock import patch

    with patch("eval.samples.machine_is_uncontended", return_value=True):
        yield
