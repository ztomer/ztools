"""A reasoning model that never stops is a distinct failure with a distinct remedy.

The qwen3_5 family streams its chain of thought into `reasoning_content` and leaves
`content` empty until it stops. On the harder eval tasks it never stops: it spends the
whole token budget thinking and returns finish_reason=length with nothing to score.

Before this, the harness recorded that as FORMAT / "Model returned empty response" --
which reads as a model that cannot follow instructions, so the response is to rewrite
the prompt. The actual remedy is the opposite direction: a SMALLER token budget forces
it to stop and answer. Raising the budget makes it strictly worse, because the
reasoning expands to fill whatever it is given.

Five of eleven installed models are qwen3_5-family, so getting this wrong mislabels
almost half a sweep and burns a full-budget retry per task doing it.
"""

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
        assert "SMALLER max_tokens" in evidence
        assert "finish_reason=length" in evidence

    def test_content_that_arrived_is_never_an_overrun(self):
        """A model that answered is not a model that failed to stop, whatever else
        is wrong with the answer."""
        result = {"content": "some answer", "reasoning_content": "x" * 9000,
                  "finish_reason": "stop", "error": "", "parsed": None}
        assert _classify_failure(result, TEXT_TASK, 0, "bad")["category"] != FAIL_REASONING


class TestTheRetryUsesASmallerBudget:
    def test_the_retry_budget_is_small_enough_to_force_a_stop(self):
        """Measured, not chosen: the same model and prompt returns empty at 16000 and
        valid output at 512."""
        from eval.run import REASONING_RETRY_MAX_TOKENS

        assert 0 < REASONING_RETRY_MAX_TOKENS <= 1024, (
            "a retry budget near the original cannot force the model to stop"
        )

    def test_it_is_far_below_the_configured_task_budget(self):
        from eval.run import REASONING_RETRY_MAX_TOKENS
        from lib.config import get_max_tokens_for_task

        assert REASONING_RETRY_MAX_TOKENS < get_max_tokens_for_task("json") / 4


class TestThePerModelBudgetCap:
    """conf/models/<family>_versions.toml can narrow a model's output budget.

    The remedy for a reasoning overrun is a SMALLER budget, so this is the config
    surface for it. Two things have to hold: the entry must actually be reachable,
    and it must only ever narrow.
    """

    def _as_family(self, monkeypatch, architecture):
        from lib.config import clear_model_config_cache

        monkeypatch.setattr(
            "lib.model_caps.recorded_capability",
            lambda model, key: architecture if key == "family" else None,
        )
        clear_model_config_cache()

    def test_a_model_whose_name_lacks_its_family_still_reaches_its_entry(
        self, monkeypatch
    ):
        """The bug this covers is subtle and was live.

        get_model_config gated the per-model lookup on `family in model`. That holds
        while families come from NAMES, and stops holding the moment they come from
        the architecture: a qwen3_5 model named "bonsai-*" has no "qwen" in its id,
        so the file loaded, the section existed, and the override was never read.
        """
        from lib.config import get_model_config

        self._as_family(monkeypatch, "qwen3_5")

        # Twice, deliberately. The uncached path was never gated, so a single lookup
        # on a cold cache passes whether or not the bug is present. The gate lives in
        # the CACHED branch, which means the override worked on first read and
        # silently vanished on every later read in the same process -- the worst
        # shape for a config bug, because it is correct exactly once.
        first = get_model_config("bonsai-27b-ternary-jang")
        second = get_model_config("bonsai-27b-ternary-jang")

        assert first.get("name") == "qwen", "should resolve to the qwen family config"
        assert first.get("max_tokens") == 512, "per-model entry was not reached"
        assert second.get("max_tokens") == 512, (
            "override vanished on the cached lookup — correct once is not correct"
        )

    def test_the_cap_narrows_the_task_budget(self, monkeypatch):
        from lib.config import get_max_tokens_for_task

        self._as_family(monkeypatch, "qwen3_5")
        assert get_max_tokens_for_task("json", "bonsai-27b-ternary-jang") == 512

    def test_it_never_widens_a_tighter_task_budget(self, monkeypatch):
        """filename budgets 1000; a 512 cap must not raise it, and a hypothetical
        larger cap must not either."""
        from lib.config import get_max_tokens_for_task

        self._as_family(monkeypatch, "qwen3_5")
        assert get_max_tokens_for_task("filename", "bonsai-27b-ternary-jang") == 512

    def test_other_models_of_the_same_family_are_untouched(self, monkeypatch):
        from lib.config import get_max_tokens_for_task

        self._as_family(monkeypatch, "qwen3_5")
        assert get_max_tokens_for_task("json", "qwen3.8-27b-4bit") == 16000

    def test_omitting_the_model_keeps_the_plain_task_budget(self):
        from lib.config import get_max_tokens_for_task

        assert get_max_tokens_for_task("json") == 16000
