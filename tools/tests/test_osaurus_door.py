"""Shell tools reach the osaurus CLI only through tools/osaurus_door.sh (ZT.OSAURUSSHELL).

The CLI's 0.5 s server probe opens a second Osaurus app when the server is slow. The door
requires /health first (`osaurus_door`), refuses a start while one answers
(`osaurus_door_serve`), and reads the model list over HTTP with no CLI at all. The gate
tools/check_osaurus_door.py refuses a bare call anywhere else; it is calibrated here on a
planted call. A stub `osaurus` that logs every call stands in for the real one.
"""

import http.server
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
DOOR = REPO / "tools" / "osaurus_door.sh"
CHECK = REPO / "tools" / "check_osaurus_door.py"


class _Server(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (the stdlib's name)
        if self.path == "/health":
            body = b'{"status":"ok"}'
        elif self.path == "/v1/models":
            body = b'{"data":[{"id":"alpha"},{"id":"beta"}]}'
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture
def live_url():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _dead_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def _door(tmp_path, url, snippet):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    log = tmp_path / "cli.log"
    stub = bindir / "osaurus"
    stub.write_text(f'#!/bin/bash\necho "$*" >> "{log}"\n')
    stub.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "OSAURUS_URL": url,
        "OSAURUS_DOOR_WAIT": "2",
    }
    r = subprocess.run(
        ["bash", "-c", f'source "{DOOR}"; {snippet}'],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    return r, (log.read_text() if log.exists() else "")


def test_the_door_refuses_with_a_reason_when_health_does_not_answer(tmp_path):
    r, calls = _door(tmp_path, _dead_url(), "osaurus_door bench --model m")
    assert r.returncode == 69
    assert "/health did not answer" in r.stderr
    assert calls == "", "the CLI ran with no server answering /health"


def test_the_door_runs_the_cli_once_health_answers(tmp_path, live_url):
    r, calls = _door(tmp_path, live_url, "osaurus_door bench --model m")
    assert r.returncode == 0, r.stderr
    assert calls == "bench --model m\n"


def test_serve_is_refused_while_a_server_answers(tmp_path, live_url):
    r, calls = _door(tmp_path, live_url, "osaurus_door_serve --port 1")
    assert r.returncode == 69 and "already answers" in r.stderr
    assert calls == ""


def test_serve_starts_when_nothing_answers(tmp_path):
    r, calls = _door(tmp_path, _dead_url(), "osaurus_door_serve --port 1")
    assert r.returncode == 0, r.stderr
    assert calls == "serve --port 1\n"


def test_the_model_list_comes_over_http_without_the_cli(tmp_path, live_url):
    r, calls = _door(tmp_path, live_url, "osaurus_door_models")
    assert r.stdout == "alpha\nbeta\n", r.stderr
    assert calls == ""


def _check(root):
    return subprocess.run(
        [sys.executable, str(CHECK), str(root)],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )


def test_no_tracked_script_calls_the_cli_outside_the_door():
    r = _check(REPO)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize(
    "line",
    [
        "osaurus list",
        'MODELS="$(osaurus list | grep x)"',
        "nohup osaurus serve --port 1 &",
        "true && osaurus stop",
        "if x; then osaurus bench; fi",
    ],
)
def test_a_planted_raw_call_fails_the_check(tmp_path, line):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, stdin=subprocess.DEVNULL)
    (tmp_path / "t.sh").write_text(f"#!/bin/bash\n# osaurus list in a comment is fine\n{line}\n")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "t.sh"], check=True, stdin=subprocess.DEVNULL
    )
    r = _check(tmp_path)
    assert r.returncode == 1 and "t.sh:3" in r.stderr, r.stderr


def test_a_message_naming_the_cli_is_not_a_call(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, stdin=subprocess.DEVNULL)
    (tmp_path / "t.sh").write_text(
        "#!/bin/bash\nwarn \"osaurus bench exited\"\necho 'osaurus stop'\n"
        "require_commands osaurus lsof\nosaurus_door bench\n"
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "t.sh"], check=True, stdin=subprocess.DEVNULL
    )
    r = _check(tmp_path)
    assert r.returncode == 0, r.stderr


def test_an_empty_tree_fails_the_check(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, stdin=subprocess.DEVNULL)
    assert _check(tmp_path).returncode == 1
