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


@pytest.fixture(autouse=True, scope="session")
def _signals_files_stay_clean(tmp_path_factory):
    """Structural gate: `pytest` must not dirty tracked config.

    eval/run.py and weekend/llm.py persist learned per-model timeouts into
    conf/eval_signals.json and conf/phase_signals.json. Both are tracked, so
    exercising those code paths rewrote them on every test run and left the
    working tree dirty. Redirect both at a tmp dir for the whole session.
    """
    from unittest.mock import patch

    tmp = tmp_path_factory.mktemp("signals")
    import eval.run as eval_run
    import weekend.llm as weekend_llm

    with patch.object(eval_run, "EVAL_SIGNALS_PATH", tmp / "eval_signals.json"), \
         patch.object(weekend_llm, "PHASE_SIGNALS_PATH", tmp / "phase_signals.json"), \
         patch.object(weekend_llm, "EXTRACT_SIGNALS_PATH", tmp / "extract_signals.json"):
        yield


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
        if path.is_file():
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
    assert not changed, "tests modified tracked config files:\n  " + "\n  ".join(changed)
