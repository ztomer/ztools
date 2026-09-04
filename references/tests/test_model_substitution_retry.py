"""The call layer retrying once against a servable model when the tag is gone.

lib/model_resolve.py decides WHAT to substitute; this covers WHEN the request path
acts on that decision. The failure being prevented: conf/config.toml named a deleted
model for four of seven tasks, so `wk` returned "HTTP 404" on every run and no output
said which config line was stale.

The negative cases matter more than the positive one. A retry that fires on any 404,
or on an unreachable roster, would convert an honest error into a silently different
model answering the user's question.
"""

from unittest.mock import MagicMock, patch

import pytest

MISSING_BODY = (
    '{"error":{"message":"Model \'qwen3.6-35b-a3b-mxfp8-mtp\' is not installed or '
    'registered with any provider.","type":"invalid_request_error"}}'
)

ROSTER = [
    {"model": "foundation", "details": {"family": "foundation", "parameter_size": ""}},
    {"model": "qwen3.8-27b-mxfp8", "details": {"family": "qwen3_5", "parameter_size": "27B"}},
]


def response(status_code, text="", payload=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = payload if payload is not None else {}
    return resp


def ok_payload(content="hello"):
    return {"choices": [{"message": {"content": content}}]}


@pytest.fixture
def posts(monkeypatch):
    """Capture every payload posted, and script the responses returned in order."""
    sent = []

    class Poster:
        responses = []

        def __call__(self, url, **kwargs):
            sent.append(kwargs.get("json", {}))
            return self.responses.pop(0)

    poster = Poster()
    with patch("lib.osaurus_lib.requests.Session") as mock_session:
        mock_session.return_value.__enter__.return_value.post = poster
        yield poster, sent


class TestRetryingOnAMissingModel:
    def test_a_missing_model_404_retries_against_the_substitute(self, posts, monkeypatch):
        import lib.osaurus_lib as ol

        poster, sent = posts
        poster.responses = [
            response(404, MISSING_BODY),
            response(200, payload=ok_payload("hello")),
        ]
        monkeypatch.setattr("lib.model_resolve.fetch_roster", lambda *a, **k: ROSTER)
        result = ol.call("qwen3.6-35b-a3b-mxfp8-mtp", [{"role": "user", "content": "hi"}])

        assert result["content"] == "hello"
        assert result["error"] is None
        assert result["model"] == "qwen3.8-27b-mxfp8"
        assert result["substituted_from"] == "qwen3.6-35b-a3b-mxfp8-mtp"
        assert "not installed" in result["substitution_reason"]
        assert [p["model"] for p in sent] == [
            "qwen3.6-35b-a3b-mxfp8-mtp",
            "qwen3.8-27b-mxfp8",
        ]

    def test_a_churning_roster_still_terminates_after_one_retry(self, posts, monkeypatch):
        """The recursion guard, exercised where it is actually load-bearing.

        A stable roster converges on its own: the substitute is picked FROM the roster,
        so the retry's model is installed by construction and a second lookup returns
        "no substitution needed". The guard only earns its keep when the roster changes
        between fetches — a server whose models are being loaded or evicted underneath
        us — where each 404 would otherwise name a fresh substitute forever.
        """
        import lib.osaurus_lib as ol

        poster, sent = posts
        poster.responses = [response(404, MISSING_BODY)] * 8
        rosters = iter(
            [
                [{"model": f"ghost-{i}", "details": {"parameter_size": "9B"}}]
                for i in range(8)
            ]
        )
        monkeypatch.setattr("lib.model_resolve.fetch_roster", lambda *a, **k: next(rosters))
        result = ol.call("qwen3.6-35b-a3b-mxfp8-mtp", [{"role": "user", "content": "hi"}])

        assert len(sent) == 2, f"expected one retry, got {len(sent)} posts"
        assert "HTTP 404" in result["error"]

    def test_quirks_are_re_derived_for_the_substitute_not_inherited(self, posts, monkeypatch):
        """The substitute is a different family, so it must not carry the dead model's
        prefix. Inheriting it also double-applies the substitute's own prefix."""
        import lib.osaurus_lib as ol

        poster, sent = posts
        poster.responses = [
            response(404, MISSING_BODY),
            response(200, payload=ok_payload()),
        ]
        monkeypatch.setattr("lib.model_resolve.fetch_roster", lambda *a, **k: ROSTER)
        ol.call(
            "qwen3.6-35b-a3b-mxfp8-mtp",
            [{"role": "system", "content": "Return JSON"}, {"role": "user", "content": "hi"}],
        )
        first_system = sent[0]["messages"][0]["content"]
        retry_system = sent[1]["messages"][0]["content"]
        assert retry_system.count("Output JSON now") <= 1
        assert retry_system == first_system or "Output JSON now" in first_system


class TestNotRetrying:
    def test_a_500_is_not_retried(self, posts, monkeypatch):
        import lib.osaurus_lib as ol

        poster, sent = posts
        poster.responses = [response(500, "Internal Server Error")]
        monkeypatch.setattr("lib.model_resolve.fetch_roster", lambda *a, **k: ROSTER)
        result = ol.call("qwen3.6-35b-a3b-mxfp8-mtp", [{"role": "user", "content": "hi"}])

        assert len(sent) == 1
        assert "HTTP 500" in result["error"]

    def test_a_404_from_a_wrong_path_is_not_retried(self, posts, monkeypatch):
        import lib.osaurus_lib as ol

        poster, sent = posts
        poster.responses = [response(404, "404 page not found")]
        monkeypatch.setattr("lib.model_resolve.fetch_roster", lambda *a, **k: ROSTER)
        result = ol.call("some-model", [{"role": "user", "content": "hi"}])

        assert len(sent) == 1
        assert "HTTP 404" in result["error"]

    def test_an_unreachable_roster_leaves_the_404_standing(self, posts, monkeypatch):
        """Server down mid-call: no evidence about the roster, so change nothing."""
        import lib.osaurus_lib as ol

        poster, sent = posts
        poster.responses = [response(404, MISSING_BODY)]
        monkeypatch.setattr("lib.model_resolve.fetch_roster", lambda *a, **k: [])
        result = ol.call("qwen3.6-35b-a3b-mxfp8-mtp", [{"role": "user", "content": "hi"}])

        assert len(sent) == 1
        assert "HTTP 404" in result["error"]
        assert "substituted_from" not in result

    def test_substitution_can_be_switched_off_by_the_caller(self, posts, monkeypatch):
        import lib.osaurus_lib as ol

        poster, sent = posts
        poster.responses = [response(404, MISSING_BODY)]
        monkeypatch.setattr("lib.model_resolve.fetch_roster", lambda *a, **k: ROSTER)
        result = ol.call(
            "qwen3.6-35b-a3b-mxfp8-mtp",
            [{"role": "user", "content": "hi"}],
            _allow_model_substitution=False,
        )

        assert len(sent) == 1
        assert "HTTP 404" in result["error"]
