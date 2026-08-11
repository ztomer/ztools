"""Follow an aggregator page and read the events out of it.

Class C14 established that a directory page is not an activity, and the model now
correctly refuses to list one. That was right, and it left the plan nearly empty:
web search for "events" returns almost nothing BUT aggregator pages, so rejecting
them rejects the whole harvest.

The events are INSIDE those pages. So the pipeline follows the most promising
ones and adds their text to the corpus the extract phase reads, instead of either
listing the page as an activity (wrong) or discarding it (wasteful).

Probed on the real scraper before building this: DDGS returns an `href` for every
result and the snippet bodies already name individual events ("TAIWANfest,
Harbourfront Centre, August 28-30"). The predecessor threw the href away in
`_clean_search_results`, which is why following was not possible at all.

Bounded on purpose -- a scheduled unattended run must not hang on a slow site:
at most FOLLOW_LIMIT pages, FETCH_TIMEOUT seconds each, MAX_PAGE_CHARS kept.
"""

from __future__ import annotations

import os

from lib.tui import STEP, WARN

__all__ = ["AGGREGATOR_MARKERS", "looks_like_aggregator", "fetch_page_text", "follow_aggregators"]

# Phrases that mark a page as a DIRECTORY of activities rather than an activity.
# Single source of truth: eval/report_classes.py imports these rather than
# keeping its own copy. A checker with a private copy of a list drifts from the
# code it measures and then agrees with the bug -- that is how C8b and the C5
# "library"/"libraries" miss both hid.
AGGREGATOR_MARKERS = (
    "things to do", "what's on", "whats on", "events guide", "activities guide",
    "events & activities", "events and activities", "guides for",
    "calendar of events", "event calendar", "event listings", "directory",
    "round-up", "roundup", "archives", "best places", "top 10", "your guide",
    "festivals", "fairs", "events in", "what to do",
)

FOLLOW_LIMIT = int(os.environ.get("WEEKEND_FOLLOW_LIMIT", "3"))
FETCH_TIMEOUT = int(os.environ.get("WEEKEND_FOLLOW_TIMEOUT", "8"))
MAX_PAGE_CHARS = int(os.environ.get("WEEKEND_FOLLOW_MAX_CHARS", "4000"))
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Safari/537.36"


def looks_like_aggregator(title: str) -> bool:
    """Is this result a page that LISTS events?

    Used here to decide what is worth FOLLOWING -- the opposite of how the same
    signal is used in the report checker, where it flags a row that should never
    have been listed as an activity.
    """
    from weekend.enforce import normalize_for_match

    normalized = normalize_for_match(title)
    return any(marker in normalized for marker in AGGREGATOR_MARKERS)


def fetch_page_text(url: str, timeout: int = FETCH_TIMEOUT, max_chars: int = MAX_PAGE_CHARS) -> str:
    """Readable text from `url`, or "" on any failure.

    Returns "" rather than raising: a scheduled run must degrade to "fewer
    events" and say so, never abort because one site was slow.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, timeout=timeout, headers={"User-Agent": _UA})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "form", "noscript"]):
            tag.decompose()
        # Keep the page's line structure. Collapsing everything into one line
        # made each followed page a single 4000-char blob, so the extractor had
        # to find events inside a wall of text and the corpus line count
        # understated real supply. One source line per listing is what the rest
        # of the pipeline is built to consume.
        lines = [
            " ".join(part.split())
            for part in soup.get_text(separator="\n", strip=True).splitlines()
        ]
        text = "\n".join(line for line in lines if line)
        return text[:max_chars]
    except Exception as exc:
        print(f"{WARN} Could not read {url[:60]}: {type(exc).__name__}")
        return ""



# Every candidate the pipeline can use must be ONE line starting with "- ":
# `weekend.phases.extract_sources` keeps only those. This function used to
# return a single "- Events listed on 'X':" header followed by raw page text,
# so the header survived and every actual event on the page was discarded --
# the whole follow-aggregators feature was dead downstream. Reformatting the
# same text per line flipped a real run from 0 transient rows to 2 correct ones.
# Deliberately permissive: nav chrome is 1-2 words ("Home", "About"), while a
# real entry can be short ("TAIWANfest August 28-30" is 23 chars). Starving
# the pipeline is the failure this whole change exists to fix, so a little
# noise reaching the extractor is the cheaper error.
MIN_CANDIDATE_CHARS = int(os.environ.get("WEEKEND_FOLLOW_MIN_LINE", "8"))
MAX_LINES_PER_PAGE = int(os.environ.get("WEEKEND_FOLLOW_MAX_LINES", "60"))


def as_candidate_lines(text: str, title: str) -> str:
    """Render fetched page text as one `- ` candidate per line.

    The source title is carried on every line so a row can still be traced back
    to the page it came from after the corpus is flattened.
    """
    lines = []
    for raw in (text or "").splitlines():
        line = " ".join(raw.split())
        if len(line) < MIN_CANDIDATE_CHARS:
            continue
        lines.append(f"- [{title}] {line}")
        if len(lines) >= MAX_LINES_PER_PAGE:
            break
    return "\n".join(lines)


def follow_aggregators(results, limit: int = FOLLOW_LIMIT) -> str:
    """Fetch the most promising aggregator pages and return their text.

    Only results that look like a directory AND carry in-region evidence are
    followed, so the budget is spent on pages likely to list local events.
    """
    from weekend.region import has_region_evidence

    followed = []
    for r in results:
        if len(followed) >= limit:
            break
        title, url = (r.get("title") or "").strip(), (r.get("href") or "").strip()
        if not url or not looks_like_aggregator(title):
            continue
        if not has_region_evidence(f"{title} {r.get('body', '')}"):
            continue
        text = fetch_page_text(url)
        if text:
            followed.append(as_candidate_lines(text, title))
    if followed:
        print(f"{STEP} Followed {len(followed)} event listing page(s)")
    return "\n".join(block for block in followed if block)
