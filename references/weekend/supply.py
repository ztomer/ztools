"""Making the in-window candidates visible to the model, without starving it.

THE PROBLEM (PENDING 5.1). Event searches are month-scoped, and the aggregator
pages they return list a whole month. The draft phase therefore chose events
that mostly could not happen on the planned weekend, and the deterministic
window filter downstream removed nearly all of them:

    40 raw -> 29 in-region -> 3 followed -> 20 candidates -> 10 draft
    -> 8 refined -> 7 structured -> 6 after exclusions -> 0 after window

THE FIX THAT DID NOT WORK, and why this one is shaped the way it is. The
obvious move — filter candidates to the window BEFORE the draft — was tried and
reverted. It did not make the model return fewer events; it made the model
INVENT them. A constraint a component cannot satisfy honestly will be satisfied
dishonestly, and fabricated output is strictly worse than sparse output.

So this REMOVES NOTHING. It marks the candidates that mention a date inside the
plan window and floats them to the top, leaving every other candidate present
and selectable below. Supply is unchanged, so the model can never be starved
into inventing; what changes is only what it sees first.

That split is the house rule: the verifiable question ("does this text mention a
date in the window?") is answered deterministically here, and the judgement
("is this a good family outing?") stays with the model.
"""

from __future__ import annotations

from datetime import date

from lib.dates import find_dates_in

# What a marked line is prefixed with. The model is told what it means in the
# prompt; here it just has to be unmistakable and stable.
IN_WINDOW_MARK = "[THIS WEEKEND]"


def mentions_window(text: str, start: date, end: date) -> bool:
    """Does this candidate mention any date inside the plan window?

    Uses the same scanner the report checkers use, so a candidate this floats
    cannot be one the checker would later call out-of-window on the same
    evidence.
    """
    return any(start <= d <= end for d in find_dates_in(text, start.year))


def prioritise_in_window(corpus: str, start: date, end: date) -> str:
    """Float in-window candidates to the top of the corpus and mark them.

    Order is preserved within each group, so this is a stable partition rather
    than a re-ranking — two runs over the same corpus produce the same text.

    Returns the corpus unchanged when nothing matches: a corpus with no dated
    candidates is a real situation (evergreen venue listings), and inventing a
    marker for it would tell the model something untrue.
    """
    lines = corpus.split("\n")
    marked: list[str] = []
    rest: list[str] = []
    for line in lines:
        if line.strip() and mentions_window(line, start, end):
            marked.append(f"{IN_WINDOW_MARK} {line}")
        else:
            rest.append(line)
    if not marked:
        return corpus
    return "\n".join(marked + rest)


def in_window_count(corpus: str, start: date, end: date) -> int:
    """How many candidates actually land in the window.

    Reported to the operator, because it is the number that explains a thin
    plan: 20 candidates of which 0 are in-window is a SUPPLY problem, and it
    looks identical to a model problem unless someone counts.
    """
    return sum(
        1 for line in corpus.split("\n") if line.strip() and mentions_window(line, start, end)
    )
