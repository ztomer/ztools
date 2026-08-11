"""The context budget must come from measurement, not from a chosen constant.

Before this, `PREFILL_CHARS_PER_SEC = 40` was a guess that ran 35-90x below the
real rate, and the first attempt to measure it derived chars/sec from ordinary
task calls — which conflates prefill with decode and produced a rate 17x too
LOW, worse than the guess. These tests pin both halves: that the probe isolates
prefill, and that what it records is what the budget is sized from.
"""

import json

import pytest
from eval import prefill as eval_prefill
from eval import signals as eval_signals
from lib import model_caps


@pytest.fixture
def signals_file(tmp_path, monkeypatch):
    """Point the eval's signal store at a tmp file for the duration of a test."""
    path = tmp_path / "eval_signals.json"
    monkeypatch.setattr(eval_signals, "EVAL_SIGNALS_PATH", path)
    monkeypatch.setattr(eval_signals, "EVAL_SIGNALS_DIR", tmp_path)
    return path


class TestProbeIsolatesPrefill:
    """max_tokens=1, because decode time is not ingestion time."""

    def test_probe_requests_a_single_output_token(self, monkeypatch):
        captured = {}

        def fake_call(model, messages, **kwargs):
            captured["max_tokens"] = kwargs.get("max_tokens")
            captured["chars"] = len(messages[0]["content"])
            return {"content": "x"}

        monkeypatch.setattr(eval_prefill, "call", fake_call)
        eval_prefill.measure_prefill_rate("some-model", "localhost", 1337)

        assert captured["max_tokens"] == 1, (
            "a probe that lets the model generate measures decode, not prefill"
        )
        assert captured["chars"] > 1000, "too small a probe is dominated by request overhead"

    def test_rate_is_probe_size_over_elapsed_time(self, monkeypatch):
        clock = iter([100.0, 102.0])  # 2 seconds elapsed
        monkeypatch.setattr(eval_prefill.time, "monotonic", lambda: next(clock))
        monkeypatch.setattr(eval_prefill, "PREFILL_PROBE_CHARS", 20000)
        monkeypatch.setattr(eval_prefill, "call", lambda *a, **k: {"content": "x"})
        monkeypatch.setattr(eval_prefill, "probe_context_window", lambda m: None)

        assert eval_prefill.measure_prefill_rate("m", "localhost", 1337) == 10000.0

    def test_probe_shrinks_to_fit_a_small_window(self, monkeypatch):
        """foundation's 4096-token window rejects the default 20K-char probe."""
        monkeypatch.setattr(eval_prefill, "probe_context_window", lambda m: 4096)
        size = eval_prefill._probe_size_for("foundation")

        assert size < eval_prefill.PREFILL_PROBE_CHARS
        assert size <= 4096 * eval_prefill.CHARS_PER_TOKEN, "probe must fit inside the window"

    def test_instant_response_is_rejected_as_implausible(self, monkeypatch):
        """A mock or a cache answers in microseconds; that is not throughput.

        Without this, any stubbed transport records a rate no hardware could
        produce and inflates every context budget derived from it.
        """
        monkeypatch.setattr(eval_prefill, "call", lambda *a, **k: {"content": "x"})
        monkeypatch.setattr(eval_prefill, "probe_context_window", lambda m: None)

        assert eval_prefill.measure_prefill_rate("m", "localhost", 1337) is None

    def test_failed_probe_reports_unknown_rather_than_a_number(self, monkeypatch):
        monkeypatch.setattr(eval_prefill, "call", lambda *a, **k: {"error": "connection refused"})
        assert eval_prefill.measure_prefill_rate("m", "localhost", 1337) is None


class TestRecordedRateSurvivesToTheBudget:
    """What zeval measures is what the tools size their prompts against."""

    def test_rate_is_stored_per_model_as_a_capability(self, signals_file):
        eval_prefill.record_prefill_rate("model-a", 1322.0)
        stored = json.loads(signals_file.read_text())

        assert stored["model-a"]["_capabilities"]["prefill_chars_per_sec"] == 1322.0
        assert stored["model-a"]["_capabilities"]["prefill_samples"] == 1

    def test_slowest_observation_wins(self, signals_file):
        """The budget has to hold on a bad run, not on a lucky one."""
        eval_prefill.record_prefill_rate("model-a", 3000.0)
        eval_prefill.record_prefill_rate("model-a", 900.0)
        eval_prefill.record_prefill_rate("model-a", 2500.0)

        caps = json.loads(signals_file.read_text())["model-a"]["_capabilities"]
        assert caps["prefill_chars_per_sec"] == 900.0
        assert caps["prefill_samples"] == 3

    def test_unmeasurable_rate_is_not_recorded(self, signals_file):
        eval_prefill.record_prefill_rate("model-a", None)
        eval_prefill.record_prefill_rate("model-b", 0)

        assert not signals_file.exists() or json.loads(signals_file.read_text()) == {}

    def test_measured_model_gets_a_larger_budget_than_the_floor(
        self, signals_file, monkeypatch
    ):
        monkeypatch.delenv("ZTOOLS_MAX_CONTEXT", raising=False)
        floor_cap = model_caps.practical_context_cap("never-measured")

        eval_prefill.record_prefill_rate("fast-model", model_caps.PREFILL_CHARS_PER_SEC_FLOOR * 3)
        measured_cap = model_caps.practical_context_cap("fast-model")

        assert measured_cap > floor_cap, (
            "a model measured faster than the floor must be allowed a bigger prompt"
        )
        assert measured_cap == pytest.approx(floor_cap * 3, rel=0.01)

    def test_slow_model_is_throttled_below_the_floor(self, signals_file, monkeypatch):
        """Measurement cuts both ways, or it is not measurement."""
        monkeypatch.delenv("ZTOOLS_MAX_CONTEXT", raising=False)
        floor_cap = model_caps.practical_context_cap("never-measured")

        eval_prefill.record_prefill_rate("slow-model", model_caps.PREFILL_CHARS_PER_SEC_FLOOR / 4)

        assert model_caps.practical_context_cap("slow-model") < floor_cap

    def test_time_budget_is_what_the_cap_actually_encodes(self, signals_file, monkeypatch):
        """cap = seconds x rate / chars-per-token, so prefill stays within budget."""
        monkeypatch.delenv("ZTOOLS_MAX_CONTEXT", raising=False)
        eval_prefill.record_prefill_rate("model-a", 1500.0)

        expected = int(model_caps.MAX_PREFILL_SECONDS * 1500.0) // model_caps.CHARS_PER_TOKEN
        assert model_caps.practical_context_cap("model-a") == expected

    def test_explicit_pin_overrides_measurement(self, signals_file, monkeypatch):
        eval_prefill.record_prefill_rate("model-a", 9999.0)
        monkeypatch.setenv("ZTOOLS_MAX_CONTEXT", "8192")

        assert model_caps.practical_context_cap("model-a") == 8192


class TestTimeoutUsesTheSameMeasurement:
    """The request timeout is budgeted from the same rate the prompt is sized by."""

    def test_faster_model_gets_a_shorter_timeout(self, signals_file):
        from twitter import budget

        prompt = "x" * 200_000
        eval_prefill.record_prefill_rate("slow-model", 1000.0)
        eval_prefill.record_prefill_rate("fast-model", 4000.0)

        assert budget._estimate_timeout(prompt, "fast-model") < budget._estimate_timeout(
            prompt, "slow-model"
        ), "a model measured 4x faster must not be budgeted the same ingestion time"

    def test_unmeasured_model_falls_back_to_the_constant(self, signals_file):
        from twitter import budget

        assert budget._prefill_rate_for_model("never-measured") == budget.PREFILL_CHARS_PER_SEC


class TestProbeDefeatsThePrefixCache:
    """A repeated prompt measures the cache, not the model."""

    def test_each_probe_sends_different_text(self, monkeypatch):
        """Identical filler read 130x faster than the same model measured honestly."""
        sent = []
        monkeypatch.setattr(
            eval_prefill, "call", lambda m, messages, **k: sent.append(messages[0]["content"])
        )
        monkeypatch.setattr(eval_prefill, "probe_context_window", lambda m: None)

        eval_prefill.measure_prefill_rate("m", "localhost", 1337)
        eval_prefill.measure_prefill_rate("m", "localhost", 1337)

        assert sent[0] != sent[1], "a repeated probe is served from the prefix cache"

    def test_the_difference_is_at_the_start_of_the_prompt(self, monkeypatch):
        """A prefix cache keys on the prefix, so a trailing nonce would not help."""
        sent = []
        monkeypatch.setattr(
            eval_prefill, "call", lambda m, messages, **k: sent.append(messages[0]["content"])
        )
        monkeypatch.setattr(eval_prefill, "probe_context_window", lambda m: None)

        eval_prefill.measure_prefill_rate("m", "localhost", 1337)
        eval_prefill.measure_prefill_rate("m", "localhost", 1337)

        assert sent[0][:40] != sent[1][:40], "the probes share a cacheable prefix"

    def test_probe_still_fits_its_size_budget(self, monkeypatch):
        monkeypatch.setattr(eval_prefill, "probe_context_window", lambda m: 4096)
        sent = []
        monkeypatch.setattr(
            eval_prefill, "call", lambda m, messages, **k: sent.append(messages[0]["content"])
        )

        eval_prefill.measure_prefill_rate("foundation", "localhost", 1337)

        assert len(sent[0]) == eval_prefill._probe_size_for("foundation"), (
            "the nonce must fit inside the probe, not extend past the window"
        )

    def test_cache_hit_speed_is_rejected(self, monkeypatch):
        """65,000-140,000 chars/sec is a cache hit, not a measurement."""
        clock = iter([100.0, 100.15])
        monkeypatch.setattr(eval_prefill.time, "monotonic", lambda: next(clock))
        monkeypatch.setattr(eval_prefill, "probe_context_window", lambda m: None)
        monkeypatch.setattr(eval_prefill, "call", lambda *a, **k: {"content": "x"})

        assert eval_prefill.measure_prefill_rate("m", "localhost", 1337) is None


class TestFullBudgetFitsTheTimeout:
    """The largest prompt the budget allows must still be affordable to send."""

    def test_max_prompt_for_a_measured_model_stays_under_max_timeout(
        self, signals_file, monkeypatch
    ):
        """Raising the context cap silently pushed every large prompt toward MAX_TIMEOUT.

        `_estimate_timeout` scales with prompt size, so a cap the budget is
        willing to fill must not produce a request the transport clamps.

        The window is forced to a real installed value (131072). Without it the
        model is unknown on disk, `usable_context_window` returns the plain 8192
        default, the practical cap never binds, and this test passes no matter
        how large the cap grows.
        """
        from twitter import budget

        monkeypatch.setattr(model_caps, "probe_context_window", lambda m: 131072)
        eval_prefill.record_prefill_rate("measured-model", 1100.0)
        max_chars = budget._ctx_chars_for_model("measured-model")
        assert max_chars > 8192 * model_caps.CHARS_PER_TOKEN, (
            "the practical cap is not binding; this test would measure nothing"
        )
        estimate = budget._estimate_timeout("x" * max_chars, "measured-model")

        assert estimate < budget.MAX_TIMEOUT, (
            f"a full {max_chars:,}-char prompt needs {estimate}s, at or past the "
            f"{budget.MAX_TIMEOUT}s ceiling -- it would be clamped and time out"
        )


class TestTheProbeUsesTheCallersTransport:
    """One mock seam must cover the probe as well as the task calls.

    `eval/prefill.py` imports `call` by value, so patching `eval.run.call` did
    not reach it. Eighteen mocked tests called the live server through that gap
    until the conftest connection gate started failing on it.
    """

    def test_run_eval_probes_through_its_own_call(self, signals_file):
        from unittest.mock import patch

        from eval import run as eval_run

        seen = []

        def fake_call(model, messages, **kwargs):
            seen.append(kwargs.get("max_tokens"))
            return {"content": "[]", "parsed": [], "time": 0.1}

        tasks = {"t": {"messages": [{"role": "user", "content": "hi"}], "validator": lambda *a, **k: 100}}
        with patch.object(eval_run, "call", fake_call):
            eval_run.run_eval("mock-model", tasks=tasks, verbose=False)

        assert 1 in seen, (
            "the prefill probe did not go through eval.run.call -- it is reaching "
            "the transport by some other name, which no mock covers"
        )

    def test_explicit_transport_is_preferred_over_the_module_import(self):
        sent = []
        eval_prefill.measure_prefill_rate(
            "m", "localhost", 1337, transport=lambda *a, **k: sent.append(1)
        )

        assert sent, "the injected transport was ignored"
