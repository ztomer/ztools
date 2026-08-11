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
_BULLET_TAG_RE = re.compile(r"\(@([A-Za-z][\w]*)\s*\|\s*([^)]+)\)\s*$")

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
