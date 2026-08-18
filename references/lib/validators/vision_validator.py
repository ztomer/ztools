"""Did the model actually SEE the images, or is it describing nothing?

The failure this is built to catch does not look like a failure. `rn`'s vision path
sent images in a format osaurus silently ignores, so the model answered from the text
prompt alone and produced confident, well-formed, entirely invented descriptions --
"large brown dog" for a picture of a red circle. A validator that only checked shape
("is this 3-4 words?") scored that a perfect 100.

So the score is keyword recall against KNOWN image contents, and the fixtures are
chosen so a blind model cannot pass: three unmistakable, mutually unrelated subjects.
Guessing "red circle" gets you one third at best.
"""

import re
from typing import Any, List, Tuple

MAX_SCORE = 100
#: A blind model still emits plausible words, so the floor is not zero -- it is
#: whatever one lucky guess is worth. Reported, not silently tolerated.
__all__ = ["validate_image_description", "matched_fixtures"]


def _words(text: str) -> set:
    return set(re.findall(r"[a-z]+", (text or "").lower()))


def matched_fixtures(text: str, fixtures: List[dict]) -> List[Tuple[str, bool]]:
    """(fixture name, was it recognised) for each fixture, in order."""
    seen = _words(text)
    return [(f["name"], bool(seen & {w.lower() for w in f["accept"]})) for f in fixtures]


def validate_image_description(
    data: Any, source_text: str = "", fixtures: List[dict] = None
) -> Tuple[int, str]:
    """Score how many of the known images the description actually accounts for."""
    if fixtures is None:
        from eval.vision_fixtures import VISION_FIXTURES

        fixtures = VISION_FIXTURES

    text = data if isinstance(data, str) else str(data or "")
    if not text.strip():
        return 0, "empty response"
    if not fixtures:
        # No ground truth means nothing can be judged. Returning a score here would
        # be inventing one, which is the whole failure mode this file exists for.
        return 0, "no fixtures to check against"

    results = matched_fixtures(text, fixtures)
    hits = sum(1 for _name, ok in results if ok)
    score = round(MAX_SCORE * hits / len(results))
    if hits == len(results):
        return score, ""
    missed = [name for name, ok in results if not ok]
    detail = f"described {hits}/{len(results)} images; missed {', '.join(missed)}"
    if hits == 0:
        # The signature of a model that received no image at all, as opposed to one
        # that saw them and described them poorly.
        detail += " (no image content recognised at all -- is the payload reaching it?)"
    return score, detail
