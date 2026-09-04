"""Legacy fixtures used by existing test files: a mock Osaurus HTTP server,
canned LLM responses, and small sample data blobs.

Split out of conftest.py for the 500-line cap (no test exemption; see
CLAUDE.md). Imported by name into conftest.py so pytest's fixture discovery
finds them there.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


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
