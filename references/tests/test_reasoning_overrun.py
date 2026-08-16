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
        from lib.config import get_max_tokens_for_task

        assert get_max_tokens_for_task("json", "othermodel-9b") == 16000

    def test_omitting_the_model_keeps_the_plain_task_budget(self):
        from lib.config import get_max_tokens_for_task

        assert get_max_tokens_for_task("json") == 16000

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
