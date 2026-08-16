"""Stopping only the runs that cannot finish, and leaving useful thinking alone.

The blunt alternative -- cap the token budget so the model is forced to stop early --
was tried and removed. It works, and it removes thinking from every request that was
already succeeding. These tests pin the distinction that replaced it: long reasoning
is allowed, runaway reasoning is not, and the line is "can the remaining budget still
hold an answer".
"""

import json

import pytest
from lib.llm import streaming


class FakeResponse:
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code
        self.closed = False

    def iter_lines(self, decode_unicode=False):
        for line in self._lines:
            if self.closed:
                return
            yield line

    def close(self):
        self.closed = True


class FakeSession:
    """Stands in for requests.Session. Records the payload it was asked to send."""

    def __init__(self, response):
        self.response = response
        self.payload = None

    def __call__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, headers=None, stream=None, timeout=None):
        self.payload = json
        return self.response


def sse(**delta):
    finish = delta.pop("finish_reason", None)
    return "data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]})


def run(lines, max_tokens=16000, status=200):
    session = FakeSession(FakeResponse(lines, status_code=status))
    result = streaming.stream_with_overrun_guard(
        "m", [{"role": "user", "content": "hi"}], max_tokens=max_tokens,
        session_factory=session,
    )
    return result, session


class TestLongThinkingIsAllowed:
    def test_a_model_that_thinks_hard_then_answers_is_not_touched(self):
        """The case a token cap would have destroyed: substantial reasoning that
        pays off in an answer."""
        lines = [sse(reasoning_content="think " * 800), sse(content="the answer"),
                 sse(finish_reason="stop")]
        result, _ = run(lines, max_tokens=16000)

        assert result["aborted"] is False
        assert result["content"] == "the answer"
        assert len(result["reasoning_content"]) > 1000

    def test_reasoning_below_the_line_is_left_running(self):
        """Half the budget spent thinking is still recoverable, so it continues."""
        half = "x" * int(16000 * 0.5 * streaming.CHARS_PER_TOKEN)
        result, _ = run([sse(reasoning_content=half), sse(content="ok"),
                         sse(finish_reason="stop")], max_tokens=16000)

        assert result["aborted"] is False
        assert result["content"] == "ok"

    def test_content_arriving_late_still_counts_as_answering(self):
        """Once content exists the model closed its think block; however long it
        thought, it must be left alone."""
        over = "x" * int(16000 * 0.9 * streaming.CHARS_PER_TOKEN)
        result, _ = run([sse(content="a"), sse(reasoning_content=over),
                         sse(finish_reason="stop")], max_tokens=16000)

        assert result["aborted"] is False


class TestRunawayThinkingIsStopped:
    def test_it_aborts_past_the_fraction_with_no_content(self):
        over = "x" * int(16000 * 0.8 * streaming.CHARS_PER_TOKEN)
        result, _ = run([sse(reasoning_content=over), sse(reasoning_content="more")],
                        max_tokens=16000)

        assert result["aborted"] is True
        assert result["finish_reason"] == "aborted_reasoning_overrun"

    def test_it_closes_the_connection_rather_than_draining_the_stream(self):
        """The point is to stop paying for a doomed run, not to detect it politely
        after the model has spent the whole budget."""
        over = "x" * int(16000 * 0.8 * streaming.CHARS_PER_TOKEN)
        session = FakeSession(FakeResponse([sse(reasoning_content=over)] + [
            sse(reasoning_content="more") for _ in range(50)
        ]))
        streaming.stream_with_overrun_guard(
            "m", [{"role": "user", "content": "hi"}], max_tokens=16000,
            session_factory=session,
        )
        assert session.response.closed is True

    def test_the_abort_reason_names_the_numbers(self):
        over = "x" * int(16000 * 0.8 * streaming.CHARS_PER_TOKEN)
        result, _ = run([sse(reasoning_content=over)], max_tokens=16000)

        assert "reasoning with no content" in result["abort_reason"]
        assert "16000-token budget" in result["abort_reason"]

    def test_the_line_scales_with_the_budget_not_a_fixed_token_count(self):
        """A small budget must abort sooner in absolute terms, or the guard would
        never fire on a short task."""
        small = "x" * int(1000 * 0.8 * streaming.CHARS_PER_TOKEN)
        aborted_small, _ = run([sse(reasoning_content=small)], max_tokens=1000)
        # The same volume of reasoning against a large budget is fine.
        allowed_large, _ = run([sse(reasoning_content=small), sse(content="ok")],
                               max_tokens=16000)

        assert aborted_small["aborted"] is True
        assert allowed_large["aborted"] is False


class TestTransportBehaviour:
    def test_it_requests_a_stream(self):
        _, session = run([sse(content="hi"), sse(finish_reason="stop")])
        assert session.payload["stream"] is True

    def test_a_non_200_is_reported_not_raised(self):
        result, _ = run([], status=503)
        assert result["error"] == "HTTP 503"
        assert result["aborted"] is False

    def test_a_transport_failure_is_reported_not_raised(self):
        """A failed request during a ten-hour sweep must not end the sweep."""
        class Boom(FakeSession):
            def post(self, *a, **k):
                raise OSError("connection reset")

        result = streaming.stream_with_overrun_guard(
            "m", [], max_tokens=100, session_factory=Boom(FakeResponse([])),
        )
        assert "OSError" in result["error"]

    @pytest.mark.parametrize(
        "line", ["", "event: ping", "data: ", "data: [DONE]", "data: not-json", "data: {}"]
    )
    def test_noise_lines_are_ignored(self, line):
        result, _ = run([line, sse(content="ok"), sse(finish_reason="stop")])
        assert result["content"] == "ok"
        assert result["error"] is None

    @pytest.mark.parametrize(
        "exc,expected",
        [
            ("Timeout", "Timeout"),
            ("ConnectionError", "Connection failed"),
        ],
    )
    def test_the_two_transport_failures_a_sweep_actually_hits(self, exc, expected):
        """A slow model and a restarted server, reported distinctly. Both are
        recoverable conditions a sweep should record and continue past."""
        import requests as rq

        class Boom(FakeSession):
            def post(self, *a, **k):
                raise getattr(rq.exceptions, exc)()

        result = streaming.stream_with_overrun_guard(
            "m", [], max_tokens=100, session_factory=Boom(FakeResponse([])),
        )
        assert result["error"] == expected
