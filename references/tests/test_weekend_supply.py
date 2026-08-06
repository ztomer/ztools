"""Candidate prioritisation must never reduce supply.

The reverted regression this guards against: filtering candidates to the plan
window BEFORE the draft did not make the model return fewer events, it made it
INVENT them. A constraint a component cannot satisfy honestly will be satisfied
dishonestly. So this module is only allowed to REORDER and MARK.
"""

from __future__ import annotations

from datetime import date

from weekend.supply import (
    IN_WINDOW_MARK,
    in_window_count,
    mentions_window,
    prioritise_in_window,
)

FRI, SUN = date(2026, 8, 7), date(2026, 8, 9)

CORPUS = "\n".join(
    [
        "- August roundup: everything happening this month",
        "- Jerkfest: Toronto, August 7 to August 9",
        "- Late thing: August 22 only",
        "- Evergreen venue: open daily, no dates given",
        "- ISO style: runs 2026-08-08",
    ]
)


def lines(text: str) -> list[str]:
    return [line for line in text.split("\n") if line.strip()]


def test_nothing_is_ever_removed():
    """THE guard. Losing a candidate here is how the model got starved into
    fabricating events the last time this was attempted."""
    out = prioritise_in_window(CORPUS, FRI, SUN)
    assert len(lines(out)) == len(lines(CORPUS))
    for original in lines(CORPUS):
        assert original in out.replace(f"{IN_WINDOW_MARK} ", ""), original


def test_in_window_candidates_are_marked_and_float_to_the_top():
    out = lines(prioritise_in_window(CORPUS, FRI, SUN))
    assert out[0].startswith(IN_WINDOW_MARK)
    assert out[1].startswith(IN_WINDOW_MARK)
    assert "Jerkfest" in out[0]
    assert "2026-08-08" in out[1]
    # Everything else survives, unmarked, below.
    assert not any(line.startswith(IN_WINDOW_MARK) for line in out[2:])


def test_out_of_window_and_undated_candidates_are_kept_unmarked():
    out = prioritise_in_window(CORPUS, FRI, SUN)
    assert "- Late thing: August 22 only" in out
    assert "- Evergreen venue: open daily, no dates given" in out
    assert f"{IN_WINDOW_MARK} - Late thing" not in out


def test_ordering_is_stable_so_two_runs_agree():
    once = prioritise_in_window(CORPUS, FRI, SUN)
    assert prioritise_in_window(CORPUS, FRI, SUN) == once
    # Within each group the original order is preserved.
    assert lines(once)[2] == "- August roundup: everything happening this month"


def test_a_corpus_with_no_in_window_dates_is_returned_untouched():
    """Marking nothing is the honest outcome for a corpus of evergreen venue
    listings; inventing a marker would tell the model something untrue."""
    corpus = "- Evergreen venue: open daily\n- Another: no dates"
    assert prioritise_in_window(corpus, FRI, SUN) == corpus


def test_the_count_is_what_explains_a_thin_plan():
    """20 candidates of which 0 are in-window is a SUPPLY problem, and it looks
    exactly like a model problem unless someone counts."""
    assert in_window_count(CORPUS, FRI, SUN) == 2
    assert in_window_count("- nothing dated here", FRI, SUN) == 0


def test_window_edges_are_inclusive():
    assert mentions_window("on August 7", FRI, SUN)
    assert mentions_window("on August 9", FRI, SUN)
    assert not mentions_window("on August 6", FRI, SUN)
    assert not mentions_window("on August 10", FRI, SUN)


def test_it_shares_the_checkers_scanner():
    """The prioritiser and the report checkers must answer "is there a date in
    this text" identically -- otherwise the pipeline can float a candidate the
    checker later rejects on the same evidence."""
    from eval.report_classes import find_dates_in as checker_scanner
    from lib.dates import find_dates_in as shared_scanner

    assert checker_scanner is shared_scanner
