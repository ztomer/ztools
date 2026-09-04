"""The constant-column assertion (PENDING 5.3).

Three bugs shipped in one shape: `Duration` reading `2-3 hours` on every row,
`Estimated Price` reading `$18-35` on every row, and `Target Age(s)` reading the
configured family range on every row. Every one was a field the model was told
never to leave empty, answered from the instructions rather than from the event
-- and every one was caught by a human noticing, which is why all three reached
a report.

`flag_constant_columns` is the check that would have caught all three. These
tests use the actual shipped values as the cases.
"""

from weekend.enforce import PROMPT_CONSTANTS, flag_constant_columns

# The configured family range, as `weekend.config.AGE_RANGE` builds it.
AGE_RANGE = "6-13"
SUSPECTS = {**PROMPT_CONSTANTS, "Target Age(s)": [AGE_RANGE]}


def _rows(n, **fields):
    """n rows that differ in name but share whatever `fields` sets."""
    return [{"name": f"event {i}", **fields} for i in range(n)]


def test_the_shipped_target_age_constant_is_flagged():
    """5.2's actual failure: every row carrying the configured family range."""
    notes = flag_constant_columns(_rows(5, target_ages=AGE_RANGE), SUSPECTS)
    assert len(notes) == 1, notes
    assert "Target Age(s)" in notes[0]
    assert AGE_RANGE in notes[0]
    # The note must say the rows were kept, or a reader will assume a drop.
    assert "kept" in notes[0].lower(), notes[0]


def test_the_shipped_duration_and_price_constants_are_flagged():
    """The other two instances, which shipped before 5.2 and were also
    caught by eye."""
    notes = flag_constant_columns(
        _rows(4, duration="2-3 hours", price="$18-35"), SUSPECTS
    )
    joined = " ".join(notes)
    assert "Duration" in joined, notes
    assert "Estimated Price" in joined, notes


def test_an_alias_field_is_checked_too():
    """The parsed item may carry the column under either name; checking only
    the canonical one would miss half the real inputs."""
    notes = flag_constant_columns(_rows(3, age_group=AGE_RANGE), SUSPECTS)
    assert len(notes) == 1, notes
    assert "Target Age(s)" in notes[0]


def test_a_column_that_varies_is_not_flagged():
    """The point is a column answered from the question, not a column that
    happens to contain a configured value on one row."""
    items = [
        {"name": "a", "target_ages": AGE_RANGE},
        {"name": "b", "target_ages": "all ages"},
        {"name": "c", "target_ages": "8+"},
    ]
    assert flag_constant_columns(items, SUSPECTS) == []


def test_a_constant_that_is_not_a_configured_value_is_not_flagged():
    """*Constant* alone is not the bug. Several events genuinely sharing a
    value is ordinary; firing on it would train the reader to ignore this."""
    assert flag_constant_columns(_rows(4, target_ages="all ages"), SUSPECTS) == []


def test_empty_cells_are_not_a_constant_column():
    """An empty cell is an honest 'unknown'. The failure is a column uniformly
    POPULATED from the prompt -- flagging uniformly blank would invert the
    lesson that empty beats fabricated.

    The second half uses a suspects list that contains the empty string. That
    is the only input under which the emptiness guard is load-bearing: with a
    normal list, a blank column is already rejected because `""` matches no
    suspect, so asserting only the normal case would pass whether or not the
    guard exists -- a test agreeing with code it does not actually pin.
    """
    assert flag_constant_columns(_rows(4, target_ages=""), SUSPECTS) == []
    assert flag_constant_columns(_rows(4), SUSPECTS) == []

    pathological = {**SUSPECTS, "Target Age(s)": [AGE_RANGE, ""]}
    assert flag_constant_columns(_rows(4, target_ages=""), pathological) == []
    assert flag_constant_columns(_rows(4), pathological) == []


def test_one_row_is_never_a_constant_column():
    """A single row is trivially 'constant'. A check that fires on every
    one-row table is one the reader learns to skip."""
    assert flag_constant_columns(_rows(1, target_ages=AGE_RANGE), SUSPECTS) == []
    assert flag_constant_columns([], SUSPECTS) == []


def test_the_check_never_drops_a_row():
    """Enforcement papering over a judgement trades a wrong answer for an empty
    one -- how this report went honest and hollow once already. This reports
    and returns notes only; it has no way to remove a row."""
    items = _rows(5, target_ages=AGE_RANGE)
    before = [dict(i) for i in items]
    flag_constant_columns(items, SUSPECTS)
    assert items == before, "flag_constant_columns mutated its input"


def test_matching_is_case_and_space_insensitive():
    """`6-13 ` and `2-3 Hours` are the same answer wearing different clothes."""
    notes = flag_constant_columns(
        _rows(3, target_ages=f" {AGE_RANGE} ", duration="2-3 Hours"), SUSPECTS
    )
    joined = " ".join(notes)
    assert "Target Age(s)" in joined, notes
    assert "Duration" in joined, notes
