"""Every eval task must carry the thing its validator judges.

THE CLASS: a task whose INPUT does not contain the property under test, graded by a
validator that then skips the test and scores the output on shape alone. It reads as
a working measurement -- models run, scores come back, a leaderboard forms -- and it
ranks nothing.

Three instances found so far:

  `filename`      sent the literal string "{text}" and scored 100 for summarising a
                  placeholder. Fixed earlier by filling the template and setting
                  `source`.
  `summarize`     carried no `source`, so validate_summary's misattribution cap --
                  `if source_text and total_bullets and faithful < total_bullets`,
                  the rule its own comment calls "disqualifying, not a deduction" --
                  could never fire. Ten of eleven models tied at 100 in the
                  2026-08-16 sweep. Fixed here.
  `image_rename`  sends its prompt as TEXT; no eval task in the suite feeds an actual
                  image, so ten models scoring 100 says nothing about vision and
                  `best_models.vlm` cannot be derived. Backlog item 9, still open.

These tests are structural on purpose. Checking that today's tasks happen to be wired
correctly is worth little; the point is that adding a sourceless task to a
source-gated validator has to fail loudly.
"""

import inspect

import pytest
from eval.tasks_core import TASKS


def source_gated_validators():
    """Validators that skip a check when `source_text` is empty.

    Derived by reading the validator, not hardcoded -- a hand-maintained list would
    silently stop covering a validator the day someone adds a new `if source_text`
    branch, which is exactly the failure mode this file exists for.
    """
    gated = set()
    for spec in TASKS.values():
        v = spec["validator"]
        try:
            src = inspect.getsource(v)
        except (OSError, TypeError):
            continue
        if "if source_text" in src or "source_text and" in src:
            gated.add(v)
    return gated


class TestTheDetectorWorks:
    """Calibration. If nothing were detected as source-gated, the rule below would
    pass for every task no matter how it was wired."""

    def test_at_least_one_validator_is_source_gated(self):
        assert source_gated_validators(), (
            "no validator was detected as source-gated; the detector is broken, "
            "not the tasks"
        )

    def test_validate_summary_is_detected_as_source_gated(self):
        from lib.validators.text_validator import validate_summary

        assert validate_summary in source_gated_validators()


class TestSourceGatedTasksCarryASource:
    def test_no_source_gated_task_is_missing_its_source(self):
        gated = source_gated_validators()
        offenders = [
            name
            for name, spec in TASKS.items()
            if spec["validator"] in gated and not spec.get("source")
        ]
        assert offenders == [], (
            f"{offenders} use a validator that skips checks without a source, but "
            "pass none. The task will score on shape alone and rank nothing."
        )

    def test_the_summarize_task_carries_a_source(self):
        """Named directly, because it is the instance that shipped."""
        assert TASKS["summarize"].get("source")

    def test_the_summarize_source_contains_attributable_claims(self):
        """A source is only useful here if the attribution parser can read it. A
        source that produces zero (handle, timestamp) pairs would satisfy the check
        above while leaving the cap just as unreachable."""
        from lib.validators.attribution import attribution_faithfulness

        source = TASKS["summarize"]["source"]
        summary = (
            "## Engineering\n"
            "- OpenAI announces GPT-5 with advanced reasoning capabilities "
            "(@TechCrunch | 08:00)\n"
        )
        faithful, total, _ = attribution_faithfulness(summary, source)
        assert total > 0, "the source yields no parseable attributions"
        assert faithful > 0, "a claim quoted verbatim from the source did not match"


class TestTheSummarizeTaskNowDiscriminates:
    """The property the fix exists for, exercised through the task's OWN wiring.

    Built from the task's real source rather than a synthetic one, so it cannot pass
    while the actual eval still fails to discriminate.
    """

    def _score(self, summary):
        spec = TASKS["summarize"]
        return spec["validator"](summary, source_text=spec.get("source", ""))[0]

    @pytest.fixture
    def claims(self):
        """Three real (handle, timestamp, claim) triples from the task's timeline."""
        return [
            ("@TechCrunch", "08:00", "OpenAI announces GPT-5 with advanced reasoning"),
            ("@TheVerge", "08:15", "Apple Vision Pro 2 enters mass production"),
            ("@TechCrunch", "08:30", "Google unveils Gemini 2.5 Pro with 1M context window"),
        ]

    def test_the_fixture_matches_the_real_timeline(self, claims):
        """Calibration: these have to be quotes from the task's own source, or the
        comparison below is between two summaries that are both simply wrong."""
        source = TASKS["summarize"]["source"]
        for handle, ts, claim in claims:
            assert handle in source and ts in source
            assert claim in source, f"{claim!r} is not in the task source"

    def test_a_misattributed_summary_scores_below_a_faithful_one(self, claims):
        faithful = "## News\n" + "".join(
            f"- {c} ({h} | {t})\n" for h, t, c in claims
        )
        # Same claims, same real handles and timestamps, rotated so every claim is
        # credited to someone who did not make it. Only the pairing is wrong.
        rotated = claims[1:] + claims[:1]
        misattributed = "## News\n" + "".join(
            f"- {c} ({h} | {t})\n"
            for (h, t, _), (_, _, c) in zip(rotated, claims)
        )
        assert self._score(misattributed) < self._score(faithful), (
            "the summarize task cannot tell a misattributed summary from a faithful "
            "one -- the cap is still unreachable"
        )

    def test_a_misattributed_summary_hits_the_cap(self, claims):
        from lib.validators.text_validator import MISATTRIBUTION_MAX_SCORE

        rotated = claims[1:] + claims[:1]
        misattributed = "## News\n" + "".join(
            f"- {c} ({h} | {t})\n"
            for (h, t, _), (_, _, c) in zip(rotated, claims)
        )
        assert self._score(misattributed) <= MISATTRIBUTION_MAX_SCORE


class TestTheAttributionTagMatcherIsNotMeasuringPunctuation:
    """A correctly attributed bullet must be RECOGNISED however it is punctuated.

    THE CLASS, again, one level down: an instrument that is blind to a surface
    variation reports confidently about the wrong dimension. `_BULLET_TAG_RE`
    anchored on `\\)\\s*$`, so any bullet ending with a full stop or wrapped in an
    extra bracket parsed as UNTAGGED. Downstream that is indistinguishable from a
    model that emitted no attributions: validate_summary gates its misattribution
    cap on `total_bullets`, so a model that punctuates its bullets was never
    attribution-checked at all, in the eval or in `tw`.

    Caught by running real models: gemma-4-12b scored 0 "no attributed bullets"
    while tagging every bullet correctly, and foundation scored 0 while genuinely
    failing a trap. Same reading for a right answer and a wrong one.
    """

    SOURCE = "[@Reuters | 07:10]: Vertex and Halcyon will complete their merger in Q3.\n"
    CLAIM = "- Vertex and Halcyon will complete their merger in Q3 "

    import pytest as _pytest

    @_pytest.mark.parametrize(
        "tag",
        [
            "(@Reuters | 07:10)",
            "(@Reuters | 07:10).",
            "(@Reuters | 07:10),",
            "(@Reuters | 07:10);",
            "(@Reuters | 07:10)!",
            "((@Reuters | 07:10))",
            "((@Reuters | 07:10)).",
            "(@Reuters | 07:10) ",
        ],
    )
    def test_each_punctuation_variant_is_recognised_and_faithful(self, tag):
        from lib.validators.attribution import attribution_faithfulness

        faithful, total, reasons = attribution_faithfulness(self.CLAIM + tag, self.SOURCE)
        assert total == 1, f"{tag!r} did not parse as a tagged bullet ({reasons})"
        assert faithful == 1, f"{tag!r} parsed but was judged unfaithful ({reasons})"

    def test_a_genuinely_untagged_bullet_still_counts_as_untagged(self):
        """The tolerance must not become 'everything is a tag'. Without this the
        parametrised test above could pass by matching indiscriminately."""
        from lib.validators.attribution import attribution_faithfulness

        assert attribution_faithfulness("- a claim with no attribution\n", self.SOURCE)[1] == 0

    def test_a_wrongly_attributed_punctuated_bullet_is_still_caught(self):
        """Tolerating the punctuation must not tolerate the error inside it."""
        from lib.validators.attribution import attribution_faithfulness

        faithful, total, _ = attribution_faithfulness(
            self.CLAIM + "(@Bloomberg | 07:10).", self.SOURCE
        )
        assert (faithful, total) == (0, 1)
