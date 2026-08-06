"""Follow-the-aggregator: read events OUT of directory pages (class C14 follow-up).

The assertion that matters here is about event COUNT AND QUALITY, not that the
fetch executed. A run that dutifully fetches three pages and still yields one
event has not fixed anything.
"""

from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    "title",
    [
        "Festivals in Toronto and GTA in August 2026",
        "Toronto Events in August 2026: Festivals, Concerts, CNE",
        "2026 August Fairs & Festivals | GTA and Surrounding Area",
        "Things to do in Vaughan with kids",
    ],
)
def test_directory_pages_are_recognised_as_worth_following(title):
    from weekend.followup import looks_like_aggregator

    assert looks_like_aggregator(title)


@pytest.mark.parametrize("title", ["Jurassic Quest", "TD Community Sunday at MOCA", "Playdium"])
def test_a_real_event_is_not_followed(title):
    """Following a specific event's own page is wasted budget, not a bug, but the
    signal must still distinguish the two."""
    from weekend.followup import looks_like_aggregator

    assert not looks_like_aggregator(title)


def test_the_checker_and_the_follower_share_one_marker_list():
    """C8b lesson: a private copy drifts and then agrees with the bug."""
    from eval import report_classes as rc
    from weekend.followup import AGGREGATOR_MARKERS

    assert rc._aggregator_markers() is AGGREGATOR_MARKERS


def test_only_in_region_directories_are_followed():
    from weekend.followup import follow_aggregators

    results = [
        {"title": "Things to do in Toronto", "href": "https://x/1", "body": "Toronto events"},
        {"title": "Things to do in Boston", "href": "https://x/2", "body": "Boston events"},
    ]
    with patch("weekend.followup.fetch_page_text", return_value="EVENT TEXT") as fetch:
        out = follow_aggregators(results)
    assert fetch.call_count == 1
    assert "Toronto" in out and "Boston" not in out


def test_follow_is_bounded():
    """A scheduled unattended run must not fetch an unbounded number of pages."""
    from weekend.followup import follow_aggregators

    results = [
        {"title": f"Things to do in Toronto {i}", "href": f"https://x/{i}", "body": "Toronto"}
        for i in range(10)
    ]
    with patch("weekend.followup.fetch_page_text", return_value="EVENT TEXT") as fetch:
        follow_aggregators(results, limit=3)
    assert fetch.call_count == 3


def test_a_failed_fetch_degrades_instead_of_aborting():
    """One slow site must cost events, never the whole run."""
    from weekend.followup import fetch_page_text, follow_aggregators

    with patch("requests.get", side_effect=OSError("boom")):
        assert fetch_page_text("https://x/1") == ""

    results = [{"title": "Things to do in Toronto", "href": "https://x/1", "body": "Toronto"}]
    with patch("weekend.followup.fetch_page_text", return_value=""):
        assert follow_aggregators(results) == ""


def test_followed_text_reaches_the_events_corpus():
    """The whole point: the page's text must end up where extract can read it."""
    from weekend import data

    results = [{"title": "Festivals in Toronto", "href": "https://x/1", "body": "Toronto"}]

    class FakeDDGS:
        def text(self, *args, **kwargs):
            return list(results)

    with (
        patch.object(data, "DDGS", FakeDDGS),
        patch("weekend.followup.fetch_page_text", return_value="TAIWANfest August 28-30"),
    ):
        corpus = data.fetch_transient_events("Aug 07 to Aug 09", 2026, "August")

    assert "TAIWANfest" in corpus, "the followed page's events never reached the corpus"
    assert "Festivals in Toronto" in corpus
