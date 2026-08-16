"""The fuzzy name matcher that decides whether output was grounded in its source.

`_names_match` feeds `check_source_extraction`, which produces `source_ratio`, which
drives the hallucination cap in `validate_detailed_json`. It is the function that
decides whether a model invented its answer -- one of the highest-stakes judgements
the scorer makes.

Mutation testing found 12 surviving mutations in `_names_match` alone and 15 more
across `_name_tokens`, `check_source_extraction` and `get_source_matching_details`.
Among them: `return False` becoming `return True` on both early exits, `max` becoming
`min` in the fallback, and every `>=` becoming `>`. A matcher whose every decision
path can be inverted without a test noticing is a matcher nothing is holding in place.

Each test below targets one decision path at the point where it flips.
"""

from lib.validators.json_validator import (
    STOPWORDS,
    _name_tokens,
    _names_match,
    check_source_extraction,
    get_source_matching_details,
)


class TestEmptyNamesNeverMatch:
    """`if not na or not nb: return False` -- both the `or` and the `False` survived.

    Matching on an empty name would mark any item as grounded, which turns the
    hallucination check off for exactly the output most likely to be hallucinated.
    """

    def test_an_empty_name_does_not_match(self):
        assert _names_match("", "Royal Ontario Museum") is False
        assert _names_match("Royal Ontario Museum", "") is False

    def test_both_empty_do_not_match(self):
        assert _names_match("", "") is False

    def test_a_name_of_only_punctuation_does_not_match(self):
        """It normalises to empty, so it must take the same path."""
        assert _names_match("!!!", "Royal Ontario Museum") is False

    def test_a_name_with_no_usable_tokens_does_not_match(self):
        """`if not ta or not tb: return False` -- the second early exit. All-stopword
        names survive normalisation but produce no tokens."""
        assert _name_tokens("the and for") == set()
        assert _names_match("the and for", "Royal Ontario Museum") is False


class TestContainmentMatches:
    def test_a_name_contained_in_the_other_matches(self):
        assert _names_match("Hockey Hall", "Hockey Hall of Fame") is True

    def test_containment_is_symmetric(self):
        assert _names_match("Hockey Hall of Fame", "Hockey Hall") is True


class TestSharedTokenThreshold:
    """`if len(ta & tb) >= 2` -- `>=` becoming `>` survived, so exactly two shared
    tokens is the only input that tests this line."""

    def test_exactly_two_shared_tokens_match(self):
        # royal + ontario shared; museum vs gallery differ; neither contains the other
        assert _names_match("Royal Ontario Museum", "Ontario Royal Gallery") is True

    def test_one_shared_short_token_does_not_match(self):
        """casa is shared but only 4 chars, so the >=5 fallback cannot rescue it."""
        assert _names_match("Casa Loma", "Casa Verde") is False


class TestTheLongestTokenFallback:
    """`max(ta, key=len)` -- `max` becoming `min` survived, as did the `>= 5`."""

    def test_a_long_distinctive_token_rescues_a_single_overlap(self):
        assert _names_match("Rogers Centre", "Rogers Arena") is True

    def test_the_fallback_uses_the_LONGEST_token_not_the_shortest(self):
        """The case that separates max from min.

        Shared token count is 1, so the fallback decides. 'aquarium' (8) is contained
        in the other name and matches; the shortest tokens are 'zoo' (3) and 'park'
        (4), both under the 5-char floor. Under `min` the fallback would consider only
        those and return False.
        """
        assert _names_match("zoo aquarium", "park aquarium") is True

    def test_a_short_shared_token_is_not_distinctive_enough(self):
        """'zoo' is 3 chars: shared, but below the fallback's 5-char floor."""
        assert _names_match("ABC Zoo", "XYZ Zoo") is False


class TestTokenExtraction:
    """`len(t) >= 3 and t not in STOPWORDS` -- the `>=`, the `and` and the `not` all
    survived, so all three are pinned here."""

    def test_three_character_tokens_are_kept(self):
        assert "abc" in _name_tokens("abc")

    def test_two_character_tokens_are_dropped(self):
        assert _name_tokens("ab") == set()

    def test_stopwords_are_dropped_even_when_long_enough(self):
        stopword = next(w for w in STOPWORDS if len(w) >= 3)
        assert stopword not in _name_tokens(f"{stopword} museum")

    def test_a_non_stopword_of_the_same_length_is_kept(self):
        """Distinguishes the stopword filter from the length filter."""
        assert "zoo" in _name_tokens("zoo")


class TestSourceExtractionRatio:
    """`if not items or not source_text` -- `or` becoming `and` survived on both
    check_source_extraction and get_source_matching_details."""

    def test_no_items_is_not_full_grounding(self):
        assert check_source_extraction([], "some source text") == 0.0

    def test_no_source_is_not_full_grounding(self):
        assert check_source_extraction([{"name": "Royal Ontario Museum"}], "") == 0.0

    def test_items_present_in_the_source_are_matched(self):
        items = [{"name": "Royal Ontario Museum"}, {"name": "Casa Loma"}]
        src = "Visit the Royal Ontario Museum and then Casa Loma for the afternoon"
        assert check_source_extraction(items, src) == 1.0

    def test_items_absent_from_the_source_are_not_matched(self):
        items = [{"name": "Invented Venue Alpha"}, {"name": "Fabricated Place Beta"}]
        assert check_source_extraction(items, "an entirely unrelated paragraph") == 0.0

    def test_the_details_view_agrees_with_the_ratio(self):
        """Two functions computing the same thing must not disagree -- that is how a
        report says 'matched 2/2' beside a score that assumed nothing matched."""
        items = [{"name": "Royal Ontario Museum"}, {"name": "Invented Venue Alpha"}]
        src = "Visit the Royal Ontario Museum today"
        ratio = check_source_extraction(items, src)
        details = get_source_matching_details(items, src)

        assert details["ratio"] == ratio
        assert len(details["matched"]) == 1
        assert len(details["unmatched"]) == 1

    def test_an_unnamed_item_is_reported_rather_than_dropped(self):
        """`item_name or "unnamed"` -- the `or` survived. An item with no name is
        still an item that failed to match, and silently dropping it would inflate
        the ratio."""
        details = get_source_matching_details([{"location": "somewhere"}], "text")
        assert details["unmatched"] == ["unnamed"]
