"""Did the summary attribute each claim to the person who actually made it?

This is the failure mode `tw` exists to avoid, and the eval could not see it.
Feeding validate_summary a summary with every quote moved to the wrong author
scored 65 -- identical to the correct one -- because the old check asked only
whether the handle and the timestamp each appeared SOMEWHERE in the source. In
a shuffled summary both tokens are present, just not together, so every bullet
counted as grounded.

Faithful attribution is a stricter claim, and a checkable one:

  1. the (handle, timestamp) PAIR must head a real source line -- not a handle
     from one tweet wearing a timestamp from another; and
  2. the bullet's content must come from THAT line, not from a different tweet
     by a different author.

Both are deterministic. Neither needs a model to judge.
"""

from __future__ import annotations

import os
import re

__all__ = [
    "CLAIM_OVERLAP_THRESHOLD",
    "attribution_faithfulness",
    "source_lines_by_author",
]

# Source lines look like `[@TechCrunch | Aug 10 14:30]: the tweet text`, which is
# the shape every model prompt tells the model to copy from.
_SOURCE_LINE_RE = re.compile(r"\[@([A-Za-z][\w]*)\s*\|\s*([^\]]+)\]\s*:?\s*(.*)")
# Bullets carry the same pair at the end: `- claim (@TechCrunch | Aug 10 14:30)`.
#
# Trailing punctuation and wrapping brackets are tolerated after the tag, and that
# is not cosmetic. The pattern used to anchor on `\)\s*$`, which made it measure
# PUNCTUATION rather than attribution:
#
#   `- claim (@Reuters | 07:10).`     full stop      -> no match
#   `- claim ((@mchen | 07:10))`      double parens  -> no match
#
# Both are correctly attributed bullets. Neither matched, so the summary counted
# ZERO tagged bullets, and every attribution check downstream was skipped --
# validate_summary's misattribution cap is gated on `total_bullets`, so a model that
# punctuates its bullets was never checked for attribution at all.
#
# Found twice, the same way, by running real models against the misattribution task:
# gemma-4-12b scored 0 "no attributed bullets" while tagging every bullet correctly
# and getting the quoted-speaker trap right, and foundation scored 0 while actually
# FAILING that trap. An instrument that returns 0 for both a right and a wrong answer
# is not measuring the thing it claims to.
_BULLET_TAG_RE = re.compile(r"\(@([A-Za-z][\w]*)\s*\|\s*([^)]+)\)[\s.,;:!)\]]*$")

_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    to was were will with about after all also been more new now other over than
    their there these they this those we you your our but not can into out up""".split()
)

# A bullet compresses a tweet, so it shares only some of its words. Below this
# share of the bullet's own content words, it is describing something else.
CLAIM_OVERLAP_THRESHOLD = float(os.environ.get("ZTOOLS_CLAIM_OVERLAP", "0.25"))


def _content_words(text: str) -> set[str]:
    """Meaningful lowercase words, with handles and punctuation stripped out."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def source_lines_by_author(source_text: str) -> dict[tuple[str, str], str]:
    """Map each source line's (handle, timestamp) to what that author said.

    Keyed by the PAIR rather than by handle alone: one author can post several
    times, and a bullet that borrows a sibling tweet's timestamp is exactly the
    error this is here to catch.
    """
    lines: dict[tuple[str, str], str] = {}
    for raw in (source_text or "").splitlines():
        match = _SOURCE_LINE_RE.search(raw)
        if not match:
            continue
        handle, stamp, content = match.group(1), match.group(2).strip(), match.group(3)
        lines[(handle.lower(), stamp)] = content
    return lines


def attribution_faithfulness(text: str, source_text: str) -> tuple[int, int, list[str]]:
    """(faithful, total, reasons) over bullets that carry an attribution tag.

    `total` counts only tagged bullets, so a summary with no tags at all is
    scored elsewhere (as a formatting failure) rather than being silently
    credited as 100% faithful here.
    """
    by_author = source_lines_by_author(source_text)
    faithful = total = 0
    reasons: list[str] = []

    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if not line.lstrip().startswith(("-", "*")):
            continue
        tag = _BULLET_TAG_RE.search(line)
        if not tag:
            continue
        total += 1
        handle, stamp = tag.group(1).lower(), tag.group(2).strip()

        said = by_author.get((handle, stamp))
        if said is None:
            # Distinguish an invented pair from a real handle wearing a
            # borrowed timestamp: the second is the subtler error and the
            # message should say which one happened.
            if any(h == handle for h, _ in by_author):
                reasons.append(f"@{handle} did not post at {stamp}")
            else:
                reasons.append(f"@{handle} is not in the source")
            continue

        claim = _content_words(line[: tag.start()])
        if not claim:
            reasons.append(f"@{handle} bullet has no content")
            continue
        overlap = len(claim & _content_words(said)) / len(claim)
        if overlap >= CLAIM_OVERLAP_THRESHOLD:
            faithful += 1
        else:
            reasons.append(f"@{handle}'s bullet does not match what they posted")

    return faithful, total, reasons


# An untagged summary is a formatting failure, not an attribution one, and scoring
# it here would credit "produced no attributions" as "made no mistakes".
NO_TAGS_SCORE = 0
# Below this share of correctly-attributed bullets a summary is actively misleading
# rather than merely incomplete, and the message says so.
ATTRIBUTION_POOR_RATIO = 0.5


def validate_attribution(data, source_text: str = ""):
    """Score a summary on attribution faithfulness alone, as a graded ratio.

    Deliberately NOT `validate_summary`'s all-or-nothing cap. That cap is right for
    `tw` -- one wrong attribution is disqualifying for something a user will act on --
    but it makes a poor measuring instrument: every model with a single slip lands on
    the same number, so the task separates nobody. The first full-roster sweep had the
    opposite problem for the same reason -- ten of eleven models tied at 100.

    A ratio ranks. `faithful / tagged` is the fraction of bullets whose (handle,
    timestamp) pair heads a real source line AND whose content came from that line.
    """
    text = data if isinstance(data, str) else str(data or "")
    if not text.strip():
        return 0, "empty response"
    if not source_text:
        # Cannot be assessed either way. Failing here would punish a task for how it
        # was wired rather than for what the model produced -- the same mistake as
        # scoring a summary 100 for attributions nobody checked.
        return 0, "no source to check attribution against"

    faithful, total, reasons = attribution_faithfulness(text, source_text)
    if not total:
        return NO_TAGS_SCORE, "no attributed bullets (every bullet must end with (@handle | time))"

    score = round(100 * faithful / total)
    if faithful == total:
        return score, ""
    detail = "; ".join(dict.fromkeys(reasons))[:200]
    severity = "misattributed" if faithful / total < ATTRIBUTION_POOR_RATIO else "attribution slips"
    return score, f"{severity} {total - faithful}/{total}: {detail}"


__all__ += ["ATTRIBUTION_POOR_RATIO", "NO_TAGS_SCORE", "validate_attribution"]
