"""validate_summary's caps and thresholds, at the points where they decide.

Mutation testing left 8 survivors in this function. The worst was `min` becoming
`max` on the misattribution cap -- the cap whose own comment says:

    Misattribution is disqualifying, not a deduction. A summary that tells the user
    the wrong person said a thing is worse than one that omits it: the reader has no
    way to spot the error, and acting on it means repeating a false claim about a
    real person.

Nothing verified it. Under the surviving mutation a summary that credits every quote
to the wrong author would be RAISED to 45 rather than capped at it, and the strongest
stated rule in the scorer would have inverted silently.
"""

import pytest
from lib.validators.attribution import attribution_faithfulness
from lib.validators.text_validator import (
    MISATTRIBUTION_MAX_SCORE,
    STRUCT_BULLET_LONG_LEN,
    validate_summary,
)

SOURCE = (
    "[@alice | 08:15]: We shipped the new billing pipeline today after three weeks of work.\n"
    "[@bob | 09:02]: The migration finished cleanly and the dashboards look correct.\n"
    "[@carol | 10:30]: I am writing the postmortem for last week's outage now.\n"
)

FAITHFUL = (
    "## Engineering\n"
    "- shipped the new billing pipeline after three weeks of work (@alice | 08:15)\n"
    "- migration finished cleanly and dashboards look correct (@bob | 09:02)\n"
    "- writing the postmortem for last week's outage (@carol | 10:30)\n"
)

# The same three claims and the same three real (handle, timestamp) pairs, rotated so
# each claim is credited to someone who did not make it. Every token is present in the
# source; only the pairing is wrong. That is the failure this scorer exists to catch.
MISATTRIBUTED = (
    "## Engineering\n"
    "- shipped the new billing pipeline after three weeks of work (@carol | 10:30)\n"
    "- migration finished cleanly and dashboards look correct (@alice | 08:15)\n"
    "- writing the postmortem for last week's outage (@bob | 09:02)\n"
)


class TestTheFixtureIsWiredCorrectly:
    """Calibration first. These bullets only count if the parser recognises them, and
    an earlier draft of this file used `- @alice did X`, which it does not: the tag
    must trail the claim as `(@handle | timestamp)`. That fixture produced 0 of 0
    bullets, so the cap never ran and every assertion below would have been vacuous.
    """

    def test_the_faithful_summary_parses_as_fully_attributed(self):
        assert attribution_faithfulness(FAITHFUL, SOURCE)[:2] == (3, 3)

    def test_the_shuffled_summary_parses_as_wholly_misattributed(self):
        assert attribution_faithfulness(MISATTRIBUTED, SOURCE)[:2] == (0, 3)


class TestMisattributionIsCapped:
    def test_a_misattributed_summary_is_capped(self):
        score, _ = validate_summary(MISATTRIBUTED, source_text=SOURCE)
        assert score <= MISATTRIBUTION_MAX_SCORE

    def test_a_faithful_summary_scores_above_the_cap(self):
        """Without this the cap could be passing for the wrong reason -- if every
        summary scored below 45 the assertion above would hold trivially."""
        score, _ = validate_summary(FAITHFUL, source_text=SOURCE)
        assert score > MISATTRIBUTION_MAX_SCORE

    def test_misattribution_costs_more_than_omission(self):
        """The stated rule, as a test. A summary that drops a claim is merely
        incomplete; one that credits it to the wrong person is wrong in a way the
        reader cannot detect."""
        omits = "## Engineering\n- shipped the new billing pipeline (@alice | 08:15)\n"
        omitting, _ = validate_summary(omits, source_text=SOURCE)
        misattributing, _ = validate_summary(MISATTRIBUTED, source_text=SOURCE)
        assert misattributing < omitting

    def test_the_failure_names_the_person_it_got_wrong(self):
        """A score alone cannot be acted on; the reason has to identify the bullet."""
        _, msg = validate_summary(MISATTRIBUTED, source_text=SOURCE)
        assert "faithful attribution" in msg
        assert any(handle in msg for handle in ("@alice", "@bob", "@carol"))

    def test_a_partially_misattributed_summary_is_also_capped(self):
        """`faithful < total_bullets` -- one wrong bullet is enough. A summary that is
        mostly right still tells the reader something false."""
        mostly_right = (
            "## Engineering\n"
            "- shipped the new billing pipeline after three weeks of work (@alice | 08:15)\n"
            "- migration finished cleanly and dashboards look correct (@bob | 09:02)\n"
            "- writing the postmortem for last week's outage (@alice | 08:15)\n"
        )
        faithful, total, _ = attribution_faithfulness(mostly_right, SOURCE)
        assert 0 < faithful < total, "fixture must be partially, not wholly, wrong"
        assert validate_summary(mostly_right, source_text=SOURCE)[0] <= MISATTRIBUTION_MAX_SCORE


class TestStructureDetection:
    """`has_bullets = "•" in s or "* " in s or "- " in s` -- the `or` chain survived,
    so each marker is checked on its own."""

    @pytest.mark.parametrize("marker", ["•", "* ", "- "])
    def test_each_bullet_marker_is_recognised_alone(self, marker):
        body = "\n".join(f"{marker}a reasonably long claim about the work done today" for _ in range(6))
        assert len(body) >= STRUCT_BULLET_LONG_LEN
        scored, _ = validate_summary(body, source_text="")
        bare = validate_summary("just a paragraph of prose with no markers at all", source_text="")[0]
        assert scored > bare, f"{marker!r} was not recognised as a bullet marker"

    def test_prose_with_no_structure_is_reported(self):
        _, msg = validate_summary("short prose", source_text="")
        assert "no structure" in msg


class TestSpecificityReporting:
    """`if specificity_score == 0: failures.append("no timestamps or narrative words")`
    -- the `or` in that message survived, and so did the branch."""

    def test_output_with_neither_timestamps_nor_narrative_is_reported(self):
        body = "\n".join("- item" for _ in range(6))
        _, msg = validate_summary(body, source_text="")
        assert "no timestamps or narrative words" in msg

    def test_output_with_a_timestamp_is_not_reported_as_lacking_one(self):
        body = "## Update\n- the deploy finished at 14:30 and the dashboards recovered\n"
        _, msg = validate_summary(body, source_text="")
        assert "no timestamps or narrative words" not in msg


class TestPartialAttributionIsFlattenedByTheCap:
    """Why two surviving mutations here are EQUIVALENT, not test gaps.

    `validate_summary` grants specificity credit in tiers -- full at ratio >= 0.8,
    half at >= 0.5 -- and then caps the whole score at MISATTRIBUTION_MAX_SCORE
    whenever `faithful < total_bullets`. Any ratio below 1.0 trips that cap, so the
    tier is absorbed and never reaches the returned score:

        ratio 1.00 -> 65        ratio 0.60 -> 45
        ratio 0.80 -> 45        ratio 0.40 -> 45
                                ratio 0.00 -> 45

    The half-credit tier is therefore unreachable through this function's output. The
    `>=` mutations on those two lines survive every possible test written against
    validate_summary, because no input distinguishes them. Recorded here so the next
    person reading a mutation report does not spend an afternoon on them.

    This test pins the flattening itself, which IS observable and IS worth keeping:
    partial misattribution must not score better than total misattribution.
    """

    CLAIMS = [
        "shipped the billing pipeline after three weeks",
        "migration finished cleanly and dashboards recovered",
        "writing the postmortem for last week outage",
        "upgraded the search index to the new analyzer",
        "cut the nightly batch runtime by half",
    ]
    SRC = "".join(f"[@u{i + 1} | 0{i + 1}:00]: We {c} today.\n" for i, c in enumerate(CLAIMS))

    def _summary(self, n_faithful, total=5):
        out = ["## Update"]
        for i, c in enumerate(self.CLAIMS[:total]):
            author = i + 1 if i < n_faithful else ((i + 1) % total) + 1
            out.append(f"- {c} (@u{author} | 0{author}:00)")
        return "\n".join(out) + "\n"

    @pytest.mark.parametrize("n_faithful", [4, 3, 2, 0])
    def test_every_partial_ratio_lands_on_the_cap(self, n_faithful):
        summary = self._summary(n_faithful)
        faithful, total, _ = attribution_faithfulness(summary, self.SRC)
        assert faithful < total, "fixture must actually be partially misattributed"
        assert validate_summary(summary, source_text=self.SRC)[0] == MISATTRIBUTION_MAX_SCORE

    def test_only_complete_faithfulness_escapes_the_cap(self):
        summary = self._summary(5)
        assert validate_summary(summary, source_text=self.SRC)[0] > MISATTRIBUTION_MAX_SCORE


class TestFilenameRelevance:
    """The irrelevance cap, and the sentinel that decides whether it applies at all.

    `filename_relevance` returns -1.0 for "cannot assess" -- no source, or a source
    with no usable words -- and a real ratio otherwise. `if coverage >= 0.0` is what
    separates those two, and `>=` becoming `>` survived: under that mutation a
    filename with EXACTLY zero overlap would skip the relevance block entirely and
    keep its full score, which is the single case the check exists for.

    The cap itself, `min(FILENAME_IRRELEVANT_MAX_SCORE, score)`, also survived as
    `max` -- nothing verified it lowers anything.
    """

    SOURCE = "Scott Adams essays about failure ambition and navigating corporate life"

    def test_zero_overlap_is_assessed_not_skipped(self):
        """coverage == 0.0 exactly: the boundary between a real ratio and the
        sentinel. This is the case a `>` would let through at full marks."""
        from lib.validators.text_validator import filename_relevance

        assert filename_relevance("zzz_qqq_wwww", self.SOURCE) == 0.0

    def test_an_unrelated_filename_is_capped(self):
        from lib.validators.text_validator import (
            FILENAME_IRRELEVANT_MAX_SCORE,
            validate_filename,
        )

        score, msg = validate_filename("zzz_qqq_wwww", source_text=self.SOURCE)
        assert score <= FILENAME_IRRELEVANT_MAX_SCORE
        assert "unrelated to input" in msg

    def test_a_relevant_filename_scores_above_the_cap(self):
        """Otherwise the cap assertion above could hold for the wrong reason."""
        from lib.validators.text_validator import (
            FILENAME_IRRELEVANT_MAX_SCORE,
            validate_filename,
        )

        score, _ = validate_filename("scott_adams_essays", source_text=self.SOURCE)
        assert score > FILENAME_IRRELEVANT_MAX_SCORE

    def test_no_source_returns_the_sentinel_not_zero(self):
        """`if not source_text: return -1.0` -- the `not` survived. Returning 0.0 here
        would mark every filename irrelevant whenever the caller passed no source,
        which is precisely how `summary_request` once scored 100 for naming nothing:
        the assessment has to be SKIPPED, not failed."""
        from lib.validators.text_validator import filename_relevance

        assert filename_relevance("scott_adams_essays", "") == -1.0

    def test_a_source_with_no_usable_words_returns_the_sentinel(self):
        """`if not words: return -1.0`. Stopwords and sub-3-character tokens are
        stripped; a source left with nothing cannot judge relevance either way."""
        from lib.validators.text_validator import filename_relevance

        assert filename_relevance("scott_adams_essays", "a to the of") == -1.0

    def test_an_unassessable_source_does_not_cap_the_score(self):
        from lib.validators.text_validator import (
            FILENAME_IRRELEVANT_MAX_SCORE,
            validate_filename,
        )

        score, _ = validate_filename("scott_adams_essays", source_text="")
        assert score > FILENAME_IRRELEVANT_MAX_SCORE

    def test_partial_coverage_is_reported_without_capping(self):
        from lib.validators.text_validator import validate_filename

        score, msg = validate_filename("scott_adams_essays", source_text=self.SOURCE)
        assert "weak input coverage" in msg
        assert score > 40


class TestFileSummaryRejectsBoilerplate:
    """A description that says nothing must not score as a description.

    Found by mutation testing the thresholds around it. BOILERPLATE_RE --
    "not specified|n/a|unknown|not provided" -- has always existed and validate_summary
    has always checked it, but validate_file_summary did not: its specificity test used
    a separate local list of generic WORDS ("personal", "document", ...) which none of
    the boilerplate phrases contain.

    So a file summary whose every description read "not specified" counted every one as
    SPECIFIC and scored a full 100, indistinguishable from real work. "unknown" scored
    lower only by accident -- it is shorter than MIN_SPECIFIC_DESC_LEN, so the length
    test caught what the content test missed.
    """

    def _summary(self, descs):
        return [{"path": f"src/mod_{i}/handler.py", "desc": d} for i, d in enumerate(descs)]

    REAL = "parses inbound webhook payloads and validates signatures"

    def test_real_descriptions_score_full(self):
        from lib.validators.text_validator import validate_file_summary

        assert validate_file_summary(self._summary([self.REAL] * 4))[0] == 100

    @pytest.mark.parametrize("filler", ["not specified", "not provided", "unknown", "n/a"])
    def test_boilerplate_does_not_score_as_a_description(self, filler):
        from lib.validators.text_validator import validate_file_summary

        score, msg = validate_file_summary(self._summary([filler] * 4))
        assert score < 100, f"{filler!r} scored as a real description"
        assert "generic descriptions only" in msg

    def test_boilerplate_scores_the_same_as_no_description_at_all(self):
        """Because it conveys the same amount: nothing."""
        from lib.validators.text_validator import validate_file_summary

        boilerplate = validate_file_summary(self._summary(["not specified"] * 4))[0]
        empty = validate_file_summary(self._summary([""] * 4))[0]
        assert boilerplate == empty

    def test_a_short_but_real_description_still_counts(self):
        """The length floor must not be doing the boilerplate check's job -- a genuine
        short description is not boilerplate."""
        from lib.validators.text_validator import validate_file_summary

        assert validate_file_summary(self._summary(["parses webhooks"] * 4))[0] == 100

    def test_a_mix_still_earns_credit_for_the_real_ones(self):
        from lib.validators.text_validator import validate_file_summary

        mixed = self._summary([self.REAL, "not specified"] * 2)
        assert validate_file_summary(mixed)[0] > validate_file_summary(
            self._summary(["not specified"] * 4)
        )[0]


class TestFileSummaryThresholds:
    """The `>=` boundaries the mutation run flagged, at exactly their limits."""

    def _summary(self, n, real_paths=None, specific=None):
        real_paths = n if real_paths is None else real_paths
        specific = n if specific is None else specific
        return [
            {
                "path": f"src/mod_{i}/handler.py" if i < real_paths else f"item{i}",
                "desc": (
                    f"parses the {i}th inbound webhook payload and validates it"
                    if i < specific
                    else "not specified"
                ),
            }
            for i in range(n)
        ]

    def test_exactly_the_minimum_item_count_earns_the_count_credit(self):
        from lib.validators.text_validator import FILE_SUMMARY_MIN_ITEMS, validate_file_summary

        at = validate_file_summary(self._summary(FILE_SUMMARY_MIN_ITEMS))[0]
        below = validate_file_summary(self._summary(FILE_SUMMARY_MIN_ITEMS - 1))[0]
        assert at > below

    def test_too_few_items_is_reported(self):
        from lib.validators.text_validator import FILE_SUMMARY_MIN_ITEMS, validate_file_summary

        _, msg = validate_file_summary(self._summary(FILE_SUMMARY_MIN_ITEMS - 1))
        assert f"need {FILE_SUMMARY_MIN_ITEMS}+" in msg

    def test_unrealistic_paths_are_reported_and_cost_score(self):
        """A path with no dot and no slash is not a path."""
        from lib.validators.text_validator import validate_file_summary

        good = validate_file_summary(self._summary(6))[0]
        bad_score, bad_msg = validate_file_summary(self._summary(6, real_paths=0))
        assert bad_score < good
        assert "unrealistic paths" in bad_msg
