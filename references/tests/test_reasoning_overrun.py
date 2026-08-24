"""A reasoning model that never stops is a distinct failure with a distinct remedy.

The qwen3_5 family streams its chain of thought into `reasoning_content` and leaves
`content` empty until it stops. On the harder eval tasks it never stops: it spends the
whole token budget thinking and returns finish_reason=length with nothing to score.

Before this, the harness recorded that as FORMAT / "Model returned empty response" --
which reads as a model that cannot follow instructions, so the response is to rewrite
the prompt. It is not a prompt problem, and the separate category is what stops the
reader going off to fix the wrong thing.

The REMEDY, however, was recorded backwards here for a while, and the correction is
worth stating because the wrong version was pinned by tests. This file used to claim
that a SMALLER budget forces the model to stop and answer, and that raising the budget
makes it strictly worse. That was generalised from one model (bonsai) and is false in
general. Measured on nemotron-3.5-lightning-30b, which returns EMPTY at 512, 2048,
8192 and 16000 alike -- guard on or off -- and ignores "detailed thinking off" and
"/no_think":

    trivial prompt ("what is 2+2")        ~400 chars reasoning, answers
    summarize_misattribution (8 lines)  19,009 chars reasoning, answers at 8192
    summarize (40 tweets)               77,208 chars reasoning, answers at 32000

Reasoning scales with the TASK, not with the budget it is handed. So the budget has to
cover it, and a retry after an overrun escalates rather than shrinks.

The cost of having had this backwards: nemotron placed last on the first full sweep
(51.3) on zeros that were all reasoning overruns, and ornith-9b second-last (49.5) the
same way. Excluding those tasks they score 90.8 and 81.4 -- first and fifth.

Five of eleven installed models are qwen3_5-family, so getting this wrong mislabels
almost half a sweep and burns a full-budget retry per task doing it.
"""

from unittest.mock import patch

import pytest
from eval.failures import FAIL_FORMAT, FAIL_REASONING, _classify_failure

JSON_TASK = {"parse_json": True}
TEXT_TASK = {"parse_json": False}


def overrun(reasoning_chars=4000, finish="length"):
    return {
        "content": "",
        "reasoning_content": "x" * reasoning_chars,
        "finish_reason": finish,
        "error": "",
        "parsed": None,
    }


def plain_empty():
    return {"content": "", "reasoning_content": "", "finish_reason": "stop",
            "error": "", "parsed": None}


class TestTheTransportCarriesWhatTheModelReturned:
    def test_call_surfaces_reasoning_content_and_finish_reason(self):
        """eval/outputs.py read result["reasoning_content"] and nothing wrote it,
        so every saved output for a reasoning model was blank."""
        from unittest.mock import MagicMock, patch

        import lib.osaurus_lib as ol

        resp = MagicMock()
        resp.status_code = 200
        resp.text = ""
        resp.json.return_value = {
            "choices": [{
                "message": {"content": "", "reasoning_content": "thinking..."},
                "finish_reason": "length",
            }]
        }
        with patch("lib.osaurus_lib.requests.Session") as sess:
            sess.return_value.__enter__.return_value.post.return_value = resp
            out = ol.call("m", [{"role": "user", "content": "hi"}])

        assert out["reasoning_content"] == "thinking..."
        assert out["finish_reason"] == "length"

    def test_a_normal_answer_reports_empty_reasoning_not_none(self):
        from unittest.mock import MagicMock, patch

        import lib.osaurus_lib as ol

        resp = MagicMock()
        resp.status_code = 200
        resp.text = ""
        resp.json.return_value = {
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]
        }
        with patch("lib.osaurus_lib.requests.Session") as sess:
            sess.return_value.__enter__.return_value.post.return_value = resp
            out = ol.call("m", [{"role": "user", "content": "hi"}])

        assert out["reasoning_content"] == ""
        assert out["content"] == "hello"


class TestClassification:
    @pytest.mark.parametrize("cfg", [JSON_TASK, TEXT_TASK], ids=["json", "text"])
    def test_an_overrun_is_not_a_format_failure(self, cfg):
        got = _classify_failure(overrun(), cfg, 0, "Empty content")
        assert got["category"] == FAIL_REASONING

    def test_the_json_path_is_covered_too(self):
        """The likelier case: the weekend tasks are the hard prompts that trigger it,
        and they are JSON tasks, where empty content otherwise reads as
        'no JSON brackets' -- the same mislabel one level down."""
        got = _classify_failure(overrun(), JSON_TASK, 0, "")
        assert got["category"] == FAIL_REASONING
        assert "JSON" not in got["evidence"]

    @pytest.mark.parametrize("cfg", [JSON_TASK, TEXT_TASK], ids=["json", "text"])
    def test_a_genuinely_empty_answer_is_still_a_format_failure(self, cfg):
        """Without reasoning_content there is no evidence of an overrun, and calling
        it one would be a different wrong diagnosis."""
        assert _classify_failure(plain_empty(), cfg, 0, "Empty content")["category"] == (
            FAIL_FORMAT
        )

    def test_the_evidence_points_at_the_budget_not_the_prompt(self):
        """The whole reason for a separate category: it must not send the reader off
        to rewrite a prompt that was never the problem."""
        evidence = _classify_failure(overrun(), JSON_TASK, 0, "")["evidence"]
        assert "raise max_tokens" in evidence
        assert "finish_reason=length" in evidence

    def test_content_that_arrived_is_never_an_overrun(self):
        """A model that answered is not a model that failed to stop, whatever else
        is wrong with the answer."""
        result = {"content": "some answer", "reasoning_content": "x" * 9000,
                  "finish_reason": "stop", "error": "", "parsed": None}
        assert _classify_failure(result, TEXT_TASK, 0, "bad")["category"] != FAIL_REASONING


class TestTheRetryUsesABiggerBudget:
    """A reasoning overrun means the model wanted MORE room, so the retry gives more.

    This class previously asserted the opposite -- that the retry budget was <= 1024
    and less than a quarter of the task budget -- on the theory that a tight budget
    forces a model to stop thinking and answer. That was measured on one model and is
    false for others. nemotron-3.5-lightning returns empty at 512, 2048, 8192 and
    16000 alike, guard on or off, and ignores "detailed thinking off" and "/no_think";
    it answers at 32000, after 77,208 characters of reasoning.

    Against that behaviour a 512-token retry does not force an answer, it guarantees a
    zero -- and that zero then averages into the model's quality mean as though it had
    answered badly. nemotron placed LAST on the first full sweep (51.3) almost entirely
    on such zeros; excluding them it placed first.

    The old assertions were not merely stale, they pinned the defect in place. Kept as
    a class so the direction cannot silently flip back.
    """

    def test_the_retry_asks_for_more_room_not_less(self):
        from eval.failures import REASONING_RETRY_MULTIPLIER

        assert REASONING_RETRY_MULTIPLIER > 1.0, (
            "a retry at or below the original budget cannot fix an overrun; the model "
            "already told us the budget was too small"
        )

    def test_the_escalated_budget_exceeds_the_task_budget(self):
        """The property that matters, computed the way run.py computes it."""
        from eval.failures import REASONING_RETRY_MAX_TOKENS, REASONING_RETRY_MULTIPLIER
        from lib.config import get_max_tokens_for_task

        base = get_max_tokens_for_task("json")
        escalated = min(int(base * REASONING_RETRY_MULTIPLIER), REASONING_RETRY_MAX_TOKENS)
        assert escalated > base

    def test_the_escalation_is_bounded(self):
        """Unbounded escalation would let one pathological task run until the timeout.
        The ceiling must still fit the installed models' context windows (131k-262k)."""
        from eval.failures import REASONING_RETRY_MAX_TOKENS

        assert 0 < REASONING_RETRY_MAX_TOKENS <= 131072

    def test_it_covers_the_budget_nemotron_actually_needed(self):
        """32000 is the measured requirement of the slowest installed reasoner on the
        largest task. A ceiling below it would reintroduce the zeros."""
        from eval.failures import REASONING_RETRY_MAX_TOKENS

        assert REASONING_RETRY_MAX_TOKENS >= 32000


class TestThePerModelBudgetCap:
    """conf/models/<family>.toml can narrow ONE model's output budget.

    No model is capped today, deliberately: a blanket cap removes thinking from every
    request that was already succeeding, and lib/llm/streaming.py now stops only the
    runs that cannot finish. The mechanism stays because a model MEASURED to be better
    capped should be cappable -- so these use a synthetic family config rather than
    whatever conf/ happens to contain, and keep testing after the last real cap is
    gone.
    """

    @pytest.fixture
    def capped_family(self, monkeypatch):
        """A synthetic family whose name appears in no model id.

        get_model_family is patched directly rather than the architecture probe and
        the architecture-to-file mapping behind it. Patching those two failed only
        under the full suite -- something ahead of it in the run leaves the resolution
        elsewhere -- and a fixture that works alone but not in the suite is exactly the
        kind of test this session has been deleting.
        """
        import lib.config_getters as getters

        monkeypatch.setattr(getters, "get_model_family", lambda model: "testfam")
        # There is now exactly one cache to inject into. This used to reach for
        # `getters._model_configs_cache`, because config_getters held its own
        # import-time alias and a rebind in test_config_core_edges.py left the two
        # modules reading different dicts. config_getters reads through the module
        # now (see `_model_caches`), so writing to config_core is writing to the
        # only cache there is.
        cache = getters._model_caches()
        cache.clear()
        cache["testfam"] = {
            "name": "testfam",
            "prompts": {"json": "family prompt"},
            "models": {"brandname-27b": {"max_tokens": 512}},
        }
        yield
        cache.clear()

    def test_a_model_whose_name_lacks_its_family_still_reaches_its_entry(
        self, capped_family
    ):
        """The bug this covers is subtle and was live.

        get_model_config gated the per-model lookup on `family in model`. That holds
        while families come from NAMES and stops holding the moment they come from the
        architecture: "brandname-27b" contains no "testfam", so the file loaded, the
        section existed, and the override was never read.
        """
        from lib.config import get_model_config

        # Twice, deliberately. The uncached path was never gated, so a single lookup
        # on a cold cache passes whether or not the bug is present. The gate lived in
        # the CACHED branch, which means the override worked on first read and
        # silently vanished afterwards -- the worst shape for a config bug.
        first = get_model_config("brandname-27b")
        second = get_model_config("brandname-27b")

        assert first.get("max_tokens") == 512, "per-model entry was not reached"
        assert second.get("max_tokens") == 512, (
            "override vanished on the cached lookup -- correct once is not correct"
        )

    def test_the_family_config_survives_the_overlay(self, capped_family):
        """An override adds to its family, it does not replace it."""
        from lib.config import get_model_config

        assert get_model_config("brandname-27b")["prompts"] == {"json": "family prompt"}

    def test_the_cap_narrows_the_task_budget(self, capped_family):
        from lib.config import get_max_tokens_for_task

        assert get_max_tokens_for_task("json", "brandname-27b") == 512

    def test_it_never_widens_a_tighter_task_budget(self, capped_family):
        """filename budgets 1000; a cap must narrow it or leave it, never raise it."""
        from lib.config import get_max_tokens_for_task

        assert get_max_tokens_for_task("filename", "brandname-27b") <= 1000

    def test_an_uncapped_model_of_the_same_family_is_untouched(self, capped_family):
        """Derived from the config, not written as a literal: the point is that an
        uncapped model gets the PLAIN task budget, whatever that budget happens to
        be. A hardcoded number turns a deliberate budget change into a test failure
        that says nothing about the cap this test exists for."""
        from lib.config import get_max_tokens, get_max_tokens_for_task

        assert get_max_tokens_for_task("json", "othermodel-9b") == get_max_tokens()["json"]

    def test_omitting_the_model_keeps_the_plain_task_budget(self):
        from lib.config import get_max_tokens, get_max_tokens_for_task

        assert get_max_tokens_for_task("json") == get_max_tokens()["json"]

    def test_no_model_is_capped_in_the_shipped_config(self):
        """A guard on the decision itself, not just the mechanism.

        A cap is a measurement result, never a default. If one appears here, it should
        arrive with numbers in its entry showing that model is better capped.
        """
        import tomllib

        from lib.paths import conf_path

        for family in ("qwen", "gemma", "nemotron", "foundation"):
            path = conf_path("models", f"{family}.toml")
            if not path.exists():
                continue
            data = tomllib.loads(path.read_text())
            for model, section in (data.get("models") or {}).items():
                assert "max_tokens" not in section, (
                    f"{model} is capped in {path.name}; if that is intended, record "
                    f"the measurement that justifies it and update this test"
                )


class TestTheStreamHasAWallClockDeadline:
    """A slowly-streaming model must not run forever.

    `requests` applies `timeout` to the connect and to the gap BETWEEN chunks, never
    to the total duration of a streamed response. So a model emitting one slow token
    at a time satisfies the read timeout on every chunk and `iter_lines()` never
    returns. The configured task timeout is silently not a bound at all.

    This hung a real sweep: muse-glimmer spent 97 minutes without completing 1 of 24
    tasks, against a configured 1800s task timeout, because each individual chunk
    arrived in time. Nothing in the harness could move on.

    The fixture streams chunks that are individually fast but collectively long, which
    is exactly the case the read timeout cannot see.
    """

    def _slow_stream(self, n_chunks, per_chunk_seconds, timeout):
        """A fake SSE stream whose chunks are fast but whose total exceeds `timeout`."""
        import time as _time
        from unittest.mock import MagicMock

        import lib.llm.streaming as streaming

        clock = {"t": 0.0}
        monkey_now = lambda: clock["t"]  # noqa: E731

        def lines(decode_unicode=True):
            for i in range(n_chunks):
                clock["t"] += per_chunk_seconds
                yield (
                    'data: {"choices":[{"delta":{"reasoning_content":"tick"},'
                    '"finish_reason":null}]}'
                )

        resp = MagicMock()
        resp.status_code = 200
        resp.iter_lines = lines
        session = MagicMock()
        session.__enter__.return_value = session
        session.post.return_value = resp

        real_monotonic = _time.monotonic
        streaming.time.monotonic = monkey_now
        try:
            return streaming.stream_with_overrun_guard(
                "m", [{"role": "user", "content": "hi"}], max_tokens=32000,
                timeout=timeout, session_factory=lambda: session,
            ), clock
        finally:
            streaming.time.monotonic = real_monotonic

    def test_a_slow_stream_is_cut_at_the_deadline(self):
        """500 chunks x 10s = 5000s total, but no single gap exceeds a 60s read
        timeout. Without a wall-clock deadline this returns only when the stream
        chooses to end."""
        result, clock = self._slow_stream(500, 10.0, timeout=60)
        assert "Timeout" in (result.get("error") or ""), result
        assert result.get("finish_reason") == "stream_deadline_exceeded"

    def test_it_stops_near_the_deadline_not_at_the_end_of_the_stream(self):
        """Proves the bound is the deadline and not merely the stream running out."""
        result, clock = self._slow_stream(500, 10.0, timeout=60)
        assert clock["t"] < 200, (
            f"consumed {clock['t']}s of stream for a 60s timeout; the deadline is not "
            "actually bounding the loop"
        )

    def test_a_stream_that_finishes_inside_the_deadline_is_untouched(self):
        """Without this the deadline could be firing on every stream, which would look
        like a fix while breaking every working run."""
        from unittest.mock import MagicMock

        import lib.llm.streaming as streaming

        def lines(decode_unicode=True):
            yield 'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}'

        resp = MagicMock()
        resp.status_code = 200
        resp.iter_lines = lines
        session = MagicMock()
        session.__enter__.return_value = session
        session.post.return_value = resp

        out = streaming.stream_with_overrun_guard(
            "m", [{"role": "user", "content": "hi"}], max_tokens=32000,
            timeout=600, session_factory=lambda: session,
        )
        assert out["content"] == "hello"
        assert not out.get("error")
        assert out.get("finish_reason") == "stop"


class TestEscalationStopsOnceProvenFutile:
    """Two model shapes produce an identical first attempt and need opposite remedies.

    nemotron-3.5-lightning wants MORE room and answers once it has it -- that is the
    measurement the escalation above was built on and it still holds.

    ornith-1.0-9b-mxfp8 is the other shape: its reasoning EXPANDS to fill whatever it
    is handed, so the guard cuts it at every budget and it never answers. Measured
    2026-08-23, same task, one escalation apart:

        32,000-token budget    72,005 chars of reasoning   aborted by the guard
        64,000-token budget   144,441 chars of reasoning   aborted by the guard

    Exactly 2.0x the reasoning for 2.0x the budget, both scored 0. Doubling bought a
    longer failure, which is what docs/MODEL_QUIRKS.md has said since 2026-08-11.

    One attempt cannot tell the two shapes apart, so the run escalates ONCE, reads the
    outcome, and stops paying once the escalated attempt is cut by the guard as well.
    Before this the harness re-bought that proof on every task -- a base attempt plus a
    double-budget retry, per task, for a model already proven unable to use it.
    """

    @staticmethod
    def _guard_aborted(budget):
        # Reasoning expands to fill whatever it is given: the guard cuts at 75% of the
        # budget, so a bigger budget only buys proportionally more wasted reasoning.
        spent_tokens = int((budget or 32000) * 0.75)
        return {
            "content": "",
            "reasoning_content": "x" * (spent_tokens * 3),
            "finish_reason": "aborted_reasoning_overrun",
            "aborted": True,
            "error": "",
            "parsed": None,
            "time": 1.0,
        }

    def _run(self, task_names):
        import eval.run as er
        from eval.tasks_core import TASKS

        budgets = []

        def fake_call(**kw):
            budgets.append(kw.get("max_tokens"))
            return self._guard_aborted(kw.get("max_tokens"))

        tasks = {name: TASKS[name] for name in task_names}
        with patch.object(er, "call", fake_call):
            er.run_eval("fake-expands-to-fill", tasks=tasks, verbose=False)
        return budgets

    def test_the_first_overrun_still_buys_the_escalation(self):
        # The nemotron case must keep working: one task, one escalated retry.
        budgets = self._run(["weekend_transient"])
        assert len([b for b in budgets if b]) == 1, (
            f"the first overrun must still escalate once, got {budgets}"
        )

    def test_later_tasks_do_not_re_buy_a_proven_futile_escalation(self):
        budgets = self._run(["weekend_transient", "image_rename", "summarize"])
        escalated = [b for b in budgets if b]
        assert len(escalated) == 1, (
            "escalation is proof-of-shape, bought once per model -- "
            f"got {len(escalated)} escalated calls: {budgets}"
        )

    def test_a_futile_retry_is_skipped_entirely_not_just_de_escalated(self):
        # Once the shape is known, repeating the base call cannot help either: the
        # first attempt already hit the guard at that exact budget.
        budgets = self._run(["weekend_transient", "image_rename", "summarize"])
        assert len(budgets) == 4, (
            "expected 2 calls proving the shape then 1 per later task, "
            f"got {len(budgets)}: {budgets}"
        )
