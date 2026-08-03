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
        text = " ".join(soup.get_text(separator=" ", strip=True).split())
        return text[:max_chars]
    except Exception as exc:
        print(f"{WARN} Could not read {url[:60]}: {type(exc).__name__}")
        return ""


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
            followed.append(f"- Events listed on '{title}':\n{text}")
    if followed:
        print(f"{STEP} Followed {len(followed)} event listing page(s)")
    return "\n".join(followed)
