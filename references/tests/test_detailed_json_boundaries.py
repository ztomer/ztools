"""The thresholds in validate_detailed_json, tested AT their edges.

Found by mutation testing: 16 mutations inside this one function survived the whole
suite, and 9 of them were `>=` quietly becoming `>`. A threshold exercised only well
inside its range is a threshold whose boundary nobody has checked, and the boundary is
the one place it can be wrong.

Three of the survivors were worse than that. The quality caps -- generic locations,
constant columns, near-duplicate rows -- are applied as `min(score, CAP)`, and
`min` becoming `max` survived every existing test. Those caps ARE the scorer's
discrimination: they exist because output that passes a structure checklist can still
be worthless, and every one of these cases previously scored a full 100. Nothing
verified they actually reduce a score rather than raise it.

The three caps turned out to be correct; nothing had checked them. The item-count
threshold did NOT -- writing these found that its credit was being absorbed whole by
the score ceiling, so a 4-item weekend report scored the same as a 12-item one. That
is fixed, and pinned below.
"""

import pytest
from lib.validators.json_validator import (
    CONSTANT_COLUMN_LIMIT,
    CONSTANT_COLUMN_MAX_SCORE,
    GENERIC_LOCATION_LIMIT,
    GENERIC_LOCATION_MAX_SCORE,
    MIN_ITEMS_GOOD,
    MIN_ITEMS_OK,
    NEAR_DUPLICATE_LIMIT,
    NEAR_DUPLICATE_MAX_SCORE,
    constant_column_ratio,
    generic_location_ratio,
    near_duplicate_ratio,
    validate_detailed_json,
)


def varied(n, price=None, ages=None):
    """n distinct, well-formed rows. Fields vary unless pinned, so no cap fires."""
    return [
        {
            "name": f"Place {i}",
            "location": f"Venue {i}",
            "price": price or f"${i}",
            "target_ages": ages or f"{i}-{i + 5}",
            "weather": "indoor" if i % 2 else "outdoor",
        }
        for i in range(1, n + 1)
    ]


def source_for(items):
    return " ".join(f"{it['name']} at {it['location']}" for it in items)


def score(items, src=None):
    return validate_detailed_json(items, source_for(items) if src is None else src)[0]


class TestTheQualityCapsActuallyCap:
    """`min(score, CAP)` -- the mutation that turned these into `max` survived.

    Each of these fails if the cap raises a score instead of lowering it, which is
    what the surviving mutation would have done in production.
    """

    def test_generic_locations_cap_downward(self):
        """Every location a placeholder: capped, not merely penalised."""
        items = [dict(r, location="TBD") for r in varied(12)]
        assert generic_location_ratio(items) >= GENERIC_LOCATION_LIMIT
        assert score(items) <= GENERIC_LOCATION_MAX_SCORE

    def test_constant_columns_cap_downward(self):
        items = varied(12, price="$10", ages="5-10")
        ratio, _ = constant_column_ratio(items)
        assert ratio >= CONSTANT_COLUMN_LIMIT
        assert score(items) <= CONSTANT_COLUMN_MAX_SCORE

    def test_near_duplicate_rows_cap_downward(self):
        items = [
            {"name": "Same Place", "location": "Same Venue", "price": "$10",
             "target_ages": "5-10", "weather": "indoor"}
            for _ in range(12)
        ]
        assert near_duplicate_ratio(items) >= NEAR_DUPLICATE_LIMIT
        assert score(items, "Same Place at Same Venue") <= NEAR_DUPLICATE_MAX_SCORE

    def test_a_capped_report_scores_below_an_uncapped_one(self):
        """The property the caps exist for, stated directly: worthless-but-well-formed
        output must not out-score real output."""
        clean = varied(12)
        placeholder = [dict(r, location="TBD") for r in varied(12)]
        assert score(placeholder) < score(clean)


class TestCapThresholdsFireAtTheirBoundary:
    """`>=` becoming `>` survived on every one of these lines.

    Exactly-at-the-limit is the only input that distinguishes the two, so it is the
    only input that tests the threshold rather than the region around it.
    """

    def test_generic_locations_exactly_at_the_limit_are_capped(self):
        n = 12
        generic_count = int(n * GENERIC_LOCATION_LIMIT)
        items = varied(n)
        for i in range(generic_count):
            items[i]["location"] = "TBD"
        ratio = generic_location_ratio(items)
        if ratio != pytest.approx(GENERIC_LOCATION_LIMIT):
            pytest.skip(f"cannot land exactly on the limit: ratio={ratio}")
        assert score(items) <= GENERIC_LOCATION_MAX_SCORE

    def test_near_duplicates_exactly_at_the_limit_are_capped(self):
        n = 10
        dupes = int(n * NEAR_DUPLICATE_LIMIT)
        items = varied(n)
        for i in range(dupes):
            items[i] = dict(items[-1])
        ratio = near_duplicate_ratio(items)
        if ratio < NEAR_DUPLICATE_LIMIT:
            pytest.skip(f"cannot reach the limit: ratio={ratio}")
        assert score(items) <= NEAR_DUPLICATE_MAX_SCORE


class TestItemCountThresholds:
    """MIN_ITEMS_GOOD and MIN_ITEMS_OK, at exactly the counts that define them.

    These two failed when first written, and the failure was real. The count credit is
    additive, the weights sum to 120 against a ceiling of 100, and the 20-point
    overhang swallowed it whole -- a 4-item weekend report scored the same 100 as a
    12-item one, on tasks whose entire purpose is producing a list. Too-few-items is
    now a CAP, like the three quality caps below it.
    """

    def test_exactly_min_items_good_scores_above_one_short(self):
        at = score(varied(MIN_ITEMS_GOOD))
        below = score(varied(MIN_ITEMS_GOOD - 1))
        assert at > below, (
            f"{MIN_ITEMS_GOOD} items must out-score {MIN_ITEMS_GOOD - 1}; equal scores "
            "mean the count credit is being absorbed by the score ceiling again"
        )

    def test_exactly_min_items_ok_scores_above_one_short(self):
        at = score(varied(MIN_ITEMS_OK))
        below = score(varied(MIN_ITEMS_OK - 1))
        assert at > below

    def test_the_cap_reproduces_the_credit_that_was_not_earned(self):
        """The cap is derived from the weights, not chosen, so it stays correct if
        the weights are ever retuned."""
        from lib.validators.json_validator import (
            DETAILED_COUNT_GOOD,
            DETAILED_COUNT_OK,
            MAX_SCORE,
        )

        assert score(varied(MIN_ITEMS_OK - 1)) <= MAX_SCORE - DETAILED_COUNT_GOOD
        assert score(varied(MIN_ITEMS_GOOD - 1)) <= MAX_SCORE - (
            DETAILED_COUNT_GOOD - DETAILED_COUNT_OK
        )

    def test_a_full_length_report_is_not_capped_for_count(self):
        assert score(varied(MIN_ITEMS_GOOD)) == 100

    def test_below_the_ok_threshold_is_reported_as_too_few(self):
        _, msg = validate_detailed_json(varied(2), source_for(varied(2)))
        assert "only 2 items" in msg


class TestSourceGroundingThresholds:
    """Ungrounded output must not out-score grounded output, at any threshold."""

    def test_fully_grounded_beats_ungrounded(self):
        items = varied(12)
        grounded = validate_detailed_json(items, source_for(items))[0]
        invented = validate_detailed_json(items, "totally unrelated source text")[0]
        assert grounded > invented

    def test_ungrounded_output_says_it_was_hallucinated(self):
        """The `not removed` mutation on this failure message survived."""
        items = varied(12)
        _, msg = validate_detailed_json(items, "totally unrelated source text")
        assert "hallucinated" in msg

    def test_no_source_is_not_treated_as_perfect_grounding(self):
        """`if source_text and items` -- `and` becoming `or` survived. With no source
        the grounding cap cannot apply, so an absent source must not silently earn the
        credit a verified one would."""
        items = varied(12)
        with_source = validate_detailed_json(items, source_for(items))[0]
        without = validate_detailed_json(items, "")[0]
        assert without <= with_source


# Ten venues with pairwise-disjoint vocabulary. Built deliberately: an earlier
# fixture reused "Venue"/"Street" across every row, so every item matched the source
# through shared filler words regardless of what the source actually contained, and
# the rows were similar enough to trip the near-duplicate cap. A fixture that cannot
# produce the ratio you are testing tests nothing.
DISJOINT = [
    ("aquarium", "harbourfront"), ("planetarium", "riverside"), ("castle", "northgate"),
    ("observatory", "eastwood"), ("conservatory", "lakeshore"), ("bazaar", "clifftop"),
    ("orchard", "westmoor"), ("foundry", "brickworks"), ("lighthouse", "saltmarsh"),
    ("arboretum", "stonebridge"),
]


def distinct(n=10):
    return [
        {
            "name": f"{a.title()} {b.title()}",
            "location": f"{b.title()} Road",
            "price": f"${i + 3}",
            "target_ages": f"{i}-{i + 4}",
            "weather": "indoor" if i % 2 else "outdoor",
        }
        for i, (a, b) in enumerate(DISJOINT[:n])
    ]


def source_covering(items, k):
    """A source mentioning only the first k items, so the ratio is exactly k/len."""
    return " ".join(f"{it['name']} on {it['location']}" for it in items[:k])


class TestSourceGroundingAtExactThresholds:
    """SOURCE_THRESHOLD_HIGH / MED / LOW, hit exactly.

    Every one of these was a surviving `>=` -> `>` mutation. Only an input landing
    precisely on the threshold separates the two, and reaching one needs a fixture
    where the grounded fraction is controllable -- hence DISJOINT above.
    """

    def test_the_fixture_produces_the_ratios_these_tests_assume(self):
        """Calibration. If this fails, every assertion below is meaningless."""
        from lib.validators.json_validator import check_source_extraction

        items = distinct(10)
        for k, expected in ((10, 1.0), (8, 0.8), (5, 0.5), (3, 0.3)):
            assert check_source_extraction(items, source_covering(items, k)) == pytest.approx(
                expected
            ), f"{k}/10 did not produce a ratio of {expected}"

    def test_exactly_the_high_threshold_earns_the_high_cap(self):
        items = distinct(10)
        assert validate_detailed_json(items, source_covering(items, 8))[0] == 100

    def test_exactly_the_medium_threshold_earns_the_medium_cap(self):
        items = distinct(10)
        at_med = validate_detailed_json(items, source_covering(items, 5))[0]
        assert at_med == 60

    def test_exactly_the_low_threshold_earns_the_low_cap(self):
        items = distinct(10)
        assert validate_detailed_json(items, source_covering(items, 3))[0] == 30

    def test_below_the_low_threshold_falls_to_the_no_source_cap(self):
        items = distinct(10)
        assert validate_detailed_json(items, source_covering(items, 2))[0] == 15

    def test_the_caps_are_monotonic_in_grounding(self):
        """More grounding must never score worse."""
        items = distinct(10)
        scores = [validate_detailed_json(items, source_covering(items, k))[0]
                  for k in (2, 3, 5, 8, 10)]
        assert scores == sorted(scores), scores

    def test_an_unrelated_source_is_reported_as_hallucinated(self):
        """`failures.append("not from input (hallucinated)")` -- `not removed`
        survived. Distinct from an ABSENT source, which cannot be assessed at all."""
        items = distinct(10)
        score, msg = validate_detailed_json(items, "a paragraph about recycling schedules")
        assert score == 15
        assert "hallucinated" in msg

    def test_an_absent_source_is_not_treated_as_hallucination(self):
        """Without a source there is nothing to contradict, so the grounding caps do
        not apply -- documented in the function, and easy to mistake for a bug."""
        items = distinct(10)
        score, msg = validate_detailed_json(items, "")
        assert score > 15
        assert "hallucinated" not in msg


class TestDetailRatioThreshold:
    """`valid_with_details >= len(items) * 0.8` -- another surviving `>=`."""

    def _mixed(self, n_with_details):
        items = distinct(10)
        for i in range(n_with_details, 10):
            items[i] = {"name": items[i]["name"]}      # name only: no detail fields
        return items

    def test_exactly_eighty_percent_with_details_earns_the_partial_credit(self):
        items = self._mixed(8)
        full = distinct(10)
        assert validate_detailed_json(items, source_covering(full, 10))[0] == 100

    def test_just_below_eighty_percent_does_not(self):
        items = self._mixed(7)
        full = distinct(10)
        at_8 = validate_detailed_json(self._mixed(8), source_covering(full, 10))[0]
        at_7 = validate_detailed_json(items, source_covering(full, 10))[0]
        assert at_7 < at_8
