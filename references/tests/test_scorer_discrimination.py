"""A scorer that cannot tell good output from bad cannot rank models.

Two models scored identically on every shared task in the first sweep --
100,100,100,91,100,100 -- which said more about the scorer than the models. Fed
deliberately defective output, `validate_detailed_json` returned 100 for three
distinct real defects, and `validate_summary` returned 65 for a summary with
every quote attributed to the wrong person.

These tests are about DISCRIMINATION, not about any single score. Each pairs
good output against output with one defect and asserts the scorer separates
them, because `best_models` derived from a blind scorer is noise with a number
on it.
"""

import pytest
from lib.validators.attribution import attribution_faithfulness
from lib.validators.json_validator import validate_detailed_json
from lib.validators.report_defects import (
    constant_column_ratio,
    generic_location_ratio,
    near_duplicate_ratio,
)
from lib.validators.text_validator import validate_summary

VENUE_SOURCE = """
- Royal Ontario Museum (100 Queens Park, Toronto): dinosaur galleries, $26 adult
- High Park (1873 Bloor St W, Toronto): trails and playground, free
- Ontario Science Centre (770 Don Mills Rd, Toronto): hands-on exhibits, $22
- Vaughan Sports Arena (81 Zenway Blvd, Vaughan): skating, $20
- Kortright Centre (9550 Pine Valley Dr, Vaughan): nature walks, $10
- Black Creek Pioneer Village (1000 Murray Ross Pkwy, Toronto): living history, $18
"""


def row(name, location, price="$20", ages="All", weather="indoor"):
    return {
        "name": name,
        "location": location,
        "target_ages": ages,
        "price": price,
        "weather": weather,
    }


GOOD_ROWS = [
    row("Royal Ontario Museum", "100 Queens Park, Toronto", "$26", "6-12"),
    row("High Park", "1873 Bloor St W, Toronto", "Free", "All", "outdoor"),
    row("Ontario Science Centre", "770 Don Mills Rd, Toronto", "$22", "8-14"),
    row("Vaughan Sports Arena", "81 Zenway Blvd, Vaughan", "$20", "5-15"),
    row("Kortright Centre", "9550 Pine Valley Dr, Vaughan", "$10", "All", "outdoor"),
    row("Black Creek Pioneer Village", "1000 Murray Ross Pkwy, Toronto", "$18", "6-16"),
]


def score(rows):
    return validate_detailed_json(rows, source_text=VENUE_SOURCE)[0]


class TestGoodOutputIsStillRecognised:
    """Guard against the opposite failure: a scorer that just says no."""

    def test_correct_report_scores_well(self):
        assert score(GOOD_ROWS) >= 90

    def test_no_defect_is_reported_for_good_output(self):
        assert generic_location_ratio(GOOD_ROWS) == 0.0
        assert constant_column_ratio(GOOD_ROWS)[0] == 0.0
        assert near_duplicate_ratio(GOOD_ROWS) == 0.0


class TestGenericLocationsAreCaught:
    """The prompt says: NEVER output generic words like 'Indoor venue'."""

    def test_generic_locations_score_far_below_good_output(self):
        # Alternating forms, not one repeated value: an identical location in
        # every row also trips the constant-column rule, and then this test
        # passes even with the generic-location check deleted. A mutation run
        # caught exactly that.
        forms = ["Indoor venue", "Outdoor venue", "Various locations", "TBD", "N/A", "Unknown"]
        generic = [
            row(r["name"], forms[i % len(forms)], price=f"${i * 5}", ages=f"{i}-{i + 6}")
            for i, r in enumerate(GOOD_ROWS)
        ]

        assert score(generic) < score(GOOD_ROWS) - 30, (
            "a report whose every location restates the column header scored the "
            "same as one with real addresses"
        )

    @pytest.mark.parametrize(
        "value", ["Indoor venue", "Outdoor Venue", "various locations", "N/A", "TBD", "location"]
    )
    def test_each_placeholder_form_is_recognised(self, value):
        assert generic_location_ratio([row("X", value)]) == 1.0

    def test_a_real_address_is_not_flagged(self):
        assert generic_location_ratio([row("X", "100 Queens Park, Toronto")]) == 0.0

    def test_a_bare_city_name_is_not_flagged(self):
        """Thin, but it is an answer; the scorer must not punish brevity as fraud."""
        assert generic_location_ratio([row("X", "Vaughan")]) == 0.0


class TestConstantColumnsAreCaught:
    """Six different venues do not all cost exactly $20."""

    def test_a_constant_column_scores_below_good_output(self):
        constant = [row(r["name"], r["location"], price="$20", ages="All") for r in GOOD_ROWS]

        assert score(constant) < score(GOOD_ROWS) - 30

    def test_the_offending_columns_are_named(self):
        constant = [row(r["name"], r["location"], price="$20", ages="All") for r in GOOD_ROWS]
        _, names = constant_column_ratio(constant)

        assert "price" in names and "target_ages" in names

    def test_day_and_weather_may_legitimately_repeat(self):
        """A one-day, all-indoor report is a real answer, not a defect."""
        same_day = [
            dict(
                row(r["name"], r["location"], price=f"${i * 5}", ages=f"{i}-{i + 6}"),
                day="Saturday",
            )
            for i, r in enumerate(GOOD_ROWS)
        ]

        assert constant_column_ratio(same_day)[0] == 0.0

    def test_two_rows_are_not_enough_to_call_it_a_pattern(self):
        assert constant_column_ratio(GOOD_ROWS[:2])[0] == 0.0


class TestPaddingByRestatementIsCaught:
    """The old check compared names exactly, so spelling variants counted as venues."""

    def test_a_padded_report_scores_below_good_output(self):
        padded = GOOD_ROWS[:2] + [
            row("Royal Ontario Museum (ROM)", "100 Queens Park, Toronto"),
            row("The ROM", "100 Queens Park, Toronto"),
            row("Royal Ontario Museum Toronto", "100 Queens Park, Toronto"),
            row("ROM Museum", "100 Queens Park, Toronto"),
        ]

        assert score(padded) < score(GOOD_ROWS) - 30

    def test_spelling_variants_of_one_venue_collapse(self):
        variants = [
            row("Royal Ontario Museum", "x"),
            row("The ROM", "x"),
            row("ROM Museum", "x"),
        ]

        assert near_duplicate_ratio(variants) > 0.5

    def test_genuinely_different_venues_do_not_collapse(self):
        """Two Toronto venues share a city, not an identity."""
        assert near_duplicate_ratio(GOOD_ROWS) == 0.0


TWEET_SOURCE = """[@sama | 09:12]: We are releasing GPT-5 today, available to all Plus users.
[@tim_cook | 10:04]: Vision Pro 2 has entered production in Shenzhen.
[@sundarpichai | 11:30]: Gemini 2.5 Pro is live with a 2M token window.
[@jensenhuang | 12:15]: Blackwell shipments doubled quarter over quarter."""

FAITHFUL = """## Summary
- OpenAI released GPT-5, available to all Plus users (@sama | 09:12)
- Vision Pro 2 entered production in Shenzhen (@tim_cook | 10:04)
- Gemini 2.5 Pro is live with a 2M token window (@sundarpichai | 11:30)"""

SHUFFLED = """## Summary
- OpenAI released GPT-5, available to all Plus users (@tim_cook | 10:04)
- Vision Pro 2 entered production in Shenzhen (@sama | 09:12)
- Gemini 2.5 Pro is live with a 2M token window (@jensenhuang | 12:15)"""


class TestMisattributionIsCaught:
    """The failure mode tw exists to avoid, and the eval could not see it."""

    def test_a_shuffled_summary_scores_below_a_faithful_one(self):
        faithful = validate_summary(FAITHFUL, source_text=TWEET_SOURCE)[0]
        shuffled = validate_summary(SHUFFLED, source_text=TWEET_SOURCE)[0]

        assert shuffled < faithful, (
            "every quote is attributed to the wrong person and it scored the same"
        )

    def test_the_reason_names_the_misattributed_handles(self):
        _, reason = validate_summary(SHUFFLED, source_text=TWEET_SOURCE)

        assert "tim_cook" in reason or "sama" in reason

    def test_faithful_attribution_is_counted(self):
        assert attribution_faithfulness(FAITHFUL, TWEET_SOURCE)[:2] == (3, 3)

    def test_a_handle_wearing_another_tweets_timestamp_is_caught(self):
        borrowed = "- OpenAI released GPT-5, available to Plus users (@sama | 12:15)"
        faithful, total, reasons = attribution_faithfulness(borrowed, TWEET_SOURCE)

        assert (faithful, total) == (0, 1)
        assert "did not post at" in reasons[0]

    def test_an_invented_handle_is_caught(self):
        invented = "- OpenAI released GPT-5 (@nobody_real | 09:12)"
        faithful, total, reasons = attribution_faithfulness(invented, TWEET_SOURCE)

        assert (faithful, total) == (0, 1)
        assert "not in the source" in reasons[0]

    def test_untagged_bullets_are_not_counted_as_faithful(self):
        """Otherwise a summary with no attribution at all scores as perfect."""
        untagged = "## Summary\n- OpenAI released GPT-5\n- Vision Pro 2 in production"

        assert attribution_faithfulness(untagged, TWEET_SOURCE)[:2] == (0, 0)


class TestTheDetectorsDoNotPunishGoodOutput:
    """A false positive here marks a correct model as broken, which is worse
    than the blindness it replaced: at least blindness was even-handed."""

    def test_numbered_names_are_distinct_venues(self):
        """"Place 1".."Place 10" collapsed to one venue and capped a perfect
        report at 50, because the length filter dropped the only token that
        distinguished them. Real names do this too: Studio 54, Pier 39."""
        numbered = [row(f"Place {i}", f"{i} Main St, Toronto") for i in range(1, 11)]

        assert near_duplicate_ratio(numbered) == 0.0

    def test_venues_sharing_a_common_word_stay_distinct(self):
        shared = [
            row("Toronto Zoo", "2000 Meadowvale Rd"),
            row("Toronto Islands", "9 Queens Quay"),
            row("Toronto Botanical Garden", "777 Lawrence Ave E"),
        ]

        assert near_duplicate_ratio(shared) == 0.0

    def test_a_report_with_varied_columns_is_not_flagged_constant(self):
        assert constant_column_ratio(GOOD_ROWS)[0] == 0.0

    def test_a_faithful_summary_is_not_capped(self):
        from lib.validators.text_validator import MISATTRIBUTION_MAX_SCORE

        assert validate_summary(FAITHFUL, source_text=TWEET_SOURCE)[0] > MISATTRIBUTION_MAX_SCORE
