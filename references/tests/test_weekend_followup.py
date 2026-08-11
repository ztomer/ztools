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


class TestPageTextBecomesCandidates:
    """Followed pages must arrive as `- ` lines, one listing each.

    `extract_sources` keeps only lines starting with "- ", and this module used
    to return a single `- Events listed on 'X':` header followed by raw page
    text — so the header survived and every actual event was discarded. The
    whole follow-aggregators feature was dead downstream.
    """

    def test_each_page_line_becomes_its_own_candidate(self):
        from weekend.followup import as_candidate_lines

        page = (
            "Home\n"
            "Summer Fair at Vaughan Mills on Sat Aug 15, free admission\n"
            "tiny\n"
            "Family Movie Night, Woodbridge Library, Sun Aug 16 at 6pm\n"
        )
        out = as_candidate_lines(page, "Vaughan Events").splitlines()
        assert len(out) == 2, out
        assert all(line.startswith("- [Vaughan Events] ") for line in out)
        assert "Summer Fair" in out[0]

    def test_every_candidate_survives_the_extract_sources_filter(self):
        """The exact filter downstream applies — a `- ` prefix on every line."""
        from weekend.followup import as_candidate_lines

        page = "\n".join(f"Event number {i} happening at a venue somewhere" for i in range(5))
        out = as_candidate_lines(page, "Src")
        kept = [line for line in out.split("\n") if line.startswith("- ")]
        assert len(kept) == 5

    def test_short_nav_chrome_is_dropped_but_short_events_are_kept(self):
        from weekend.followup import as_candidate_lines

        out = as_candidate_lines("Home\nAbout\nTAIWANfest Aug 28-30\n", "Src")
        assert out.count("\n") == 0
        assert "TAIWANfest" in out

    def test_line_budget_bounds_a_huge_page(self):
        from weekend.followup import MAX_LINES_PER_PAGE, as_candidate_lines

        page = "\n".join(f"Some event listing number {i} at a place" for i in range(500))
        assert len(as_candidate_lines(page, "Src").splitlines()) == MAX_LINES_PER_PAGE

    def test_empty_page_yields_nothing(self):
        from weekend.followup import as_candidate_lines

        assert as_candidate_lines("", "Src") == ""
        assert as_candidate_lines("   \n\n", "Src") == ""


class TestFetchPageTextKeepsStructure:
    """One source line per listing, not one blob per page.

    `" ".join(text.split())` collapsed every page into a single 4000-char line,
    so a followed page produced exactly ONE candidate and the extractor had to
    find events inside a wall of text.
    """

    def _fake_response(self, html):
        class _Resp:
            text = html

            def raise_for_status(self):
                return None

        return _Resp()

    def test_line_structure_survives(self):
        import weekend.followup as fu

        html = (
            "<html><body><nav>Menu</nav>"
            "<div>Summer Fair at Vaughan Mills, Sat Aug 15</div>"
            "<div>Family Movie Night, Woodbridge Library, Sun Aug 16</div>"
            "<script>ignore()</script></body></html>"
        )
        # `requests` is imported inside the function, so patch the library.
        with patch("requests.get", return_value=self._fake_response(html)):
            text = fu.fetch_page_text("https://example.test/events")

        lines = text.splitlines()
        assert len(lines) >= 2, lines
        assert any("Summer Fair" in line for line in lines)
        assert any("Family Movie Night" in line for line in lines)
        # Stripped tags must not contribute content.
        assert "ignore()" not in text and "Menu" not in text

    def test_failure_returns_empty_rather_than_raising(self):
        import weekend.followup as fu

        with patch("requests.get", side_effect=RuntimeError("network down")):
            assert fu.fetch_page_text("https://example.test/x") == ""
