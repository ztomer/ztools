"""The `routines` status adapter reads back a plan; it never generates one.

It exists to keep two failure modes visible, both of which look fine from
outside: a HOLLOW plan (an empty plan passes every content check, because
there are no rows to be wrong) and a STALE one (last week's plan left in the
output directory is indistinguishable from this week's unless someone compares
its window to the calendar).
"""

from __future__ import annotations

from datetime import date

import pytest

import routines_status

PLAN = """# Weekend

## Fixed Activities

| Activity | Venue | Cost |
| --- | --- | --- |
| Swim | Pool | Free |
| Park | High Park | Free |

## Transient Events

| Event | Venue | Dates | Day |
| --- | --- | --- | --- |
| Jerkfest | Toronto | 2026-08-07 | Friday |
"""

HOLLOW = """# Weekend

## Fixed Activities

| Activity | Venue | Cost |
| --- | --- | --- |
| Swim | Pool | Free |

## Transient Events

| Event | Venue | Dates | Day |
| --- | --- | --- | --- |
"""

# 2026-08-03 is a Monday; the weekend it points at is Aug 7-9.
MONDAY = date(2026, 8, 3)


@pytest.fixture
def plans(tmp_path, monkeypatch):
    """Point the adapter at a directory we own, never the real output dir."""
    monkeypatch.setattr(routines_status, "OUTPUT_DIR_PATH", str(tmp_path))
    return tmp_path


def write_plan(directory, name: str, body: str = PLAN):
    p = directory / name
    p.write_text(body, encoding="utf-8")
    return p


def test_no_plan_directory_is_unknown_not_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(routines_status, "OUTPUT_DIR_PATH", str(tmp_path / "nope"))
    status = routines_status.build_status(MONDAY)
    assert status["state"] == "unknown"
    assert "no plan directory" in status["summary"]


def test_no_plan_ever_written_is_unknown_not_an_empty_plan(plans):
    """"0 events" from a missing file is indistinguishable from a genuinely
    empty plan, and from a tool that has simply never run."""
    status = routines_status.build_status(MONDAY)
    assert status["state"] == "unknown"
    assert "has ever been written" in status["summary"]


def test_a_plan_for_the_upcoming_weekend_with_events_is_ok(plans):
    write_plan(plans, "weekend_plan_August_07_to_August_09_2026.md")
    status = routines_status.build_status(MONDAY)
    assert status["state"] == "ok"
    assert status["details"]["covers_upcoming"] is True
    assert status["details"]["fixed_rows"] == 2
    assert status["details"]["transient_rows"] == 1
    assert status["items"] == []


def test_a_hollow_plan_is_attention_however_well_formed_it_looks(plans):
    """The open defect (PENDING 5.1). An empty transient table passes every
    content check; only counting the rows catches it."""
    write_plan(plans, "weekend_plan_August_07_to_August_09_2026.md", HOLLOW)
    status = routines_status.build_status(MONDAY)
    assert status["state"] == "attention"
    assert {
        "name": "transient events",
        "action": "none in the latest plan",
    } in status["items"]


def test_last_weeks_plan_is_stale_rather_than_current(plans):
    write_plan(plans, "weekend_plan_July_31_to_August_02_2026.md")
    status = routines_status.build_status(MONDAY)
    assert status["state"] == "attention"
    assert "not planned" in status["summary"]
    assert status["details"]["covers_upcoming"] is False
    assert status["items"][0]["action"] == "not generated yet"


def test_the_newest_plan_wins_not_the_alphabetically_last(plans):
    import os
    import time

    old = write_plan(plans, "weekend_plan_July_31_to_August_02_2026.md")
    new = write_plan(plans, "weekend_plan_August_07_to_August_09_2026.md")
    # Make the stale one alphabetically later but older on disk.
    os.utime(old, (time.time(), time.time()))
    os.utime(new, (time.time() - 100, time.time() - 100))
    assert routines_status.build_status(MONDAY)["details"]["plan"] == str(old)


@pytest.mark.parametrize(
    "today,friday",
    [
        (date(2026, 8, 3), date(2026, 8, 7)),   # Monday  -> this Friday
        (date(2026, 8, 6), date(2026, 8, 7)),   # Thursday-> tomorrow
        (date(2026, 8, 7), date(2026, 8, 7)),   # Friday  -> today
        (date(2026, 8, 8), date(2026, 8, 7)),   # Saturday-> the one we are in
        (date(2026, 8, 9), date(2026, 8, 7)),   # Sunday  -> the one we are in
        (date(2026, 8, 10), date(2026, 8, 14)),  # Monday -> next Friday
    ],
)
def test_during_a_weekend_the_current_one_counts_as_upcoming(today, friday):
    """A plan for the days you are living through is current, not stale."""
    assert routines_status.upcoming_weekend(today)[0] == friday


def test_an_unreadable_plan_becomes_a_stated_reason_not_a_traceback(plans, capsys):
    (plans / "weekend_plan_August_07_to_August_09_2026.md").write_bytes(b"\xff\xfe\x00bad")
    routines_status.main()
    out = capsys.readouterr().out
    assert '"state": "unknown"' in out
    assert "status unavailable" in out
