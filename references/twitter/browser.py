#!/usr/bin/env python3
"""
Browser automation helpers for twitter_summarizer.
"""

import os
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

try:
    from playwright.sync_api import TimeoutError as PWTimeout
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = PWTimeout = None

from lib.config_toml import load_config
from lib.signal_handling import GracefulShutdown, is_shutdown_requested
from lib.tui import STEP, WARN

from twitter.browser_launch import (
    BACKEND_CAMOUFOX,
    BACKEND_CHROMIUM,
    backend_explicitly_requested,
    launch_camoufox,
    resolve_backend,
)
from twitter.browser_parse import parse_tweets_from_response
from twitter.cookies import SESSION_COOKIE_NAME, get_browser_cookies, has_session_cookie

_TWITTER_CONFIG_PATH = Path(__file__).parent.parent / "conf" / "twitter.toml"
_TWITTER_CFG = load_config(_TWITTER_CONFIG_PATH) or {}

MAX_SCROLLS = int(os.environ.get("TWITTER_MAX_SCROLLS", str(_TWITTER_CFG.get("max_scrolls", 1200))))
SCROLL_PAUSE_MS = int(os.environ.get("TWITTER_SCROLL_PAUSE_MS", "1800"))

# Stop conditions that bound the scroll loop independently of MAX_SCROLLS.
# Without these the loop only ever exits on "reached the requested time
# window", so a timeline that stops serving new tweets spins the full
# MAX_SCROLLS x SCROLL_PAUSE_MS (20+ minutes) collecting nothing.
MAX_RUNTIME_S = float(os.environ.get("TWITTER_MAX_RUNTIME_S", "300"))
STAGNANT_SCROLL_LIMIT = int(os.environ.get("TWITTER_STAGNANT_SCROLL_LIMIT", "8"))

# Scroll pauses are chunked so an interrupt is noticed within a step rather
# than at the end of a full SCROLL_PAUSE_MS.
SLEEP_STEP_S = 0.1

# Bound to the real clock at import so the runtime budget stays wall-clock
# even when callers stub the module's `time` for sleeping.
_monotonic = time.monotonic

# Timeouts (milliseconds)
PAGE_LOAD_TIMEOUT_MS = int(os.environ.get("TWITTER_PAGE_LOAD_TIMEOUT_MS", "30000"))
# Humanized (animated) cursor movement makes a click take far longer than a
# synthetic one; 5s was not enough and the Following tab silently fell back to
# the "For You" feed on every run.
CLICK_TIMEOUT_MS = int(os.environ.get("TWITTER_CLICK_TIMEOUT_MS", "15000"))

# Sleep durations (seconds)
INITIAL_PAGE_WAIT = int(os.environ.get("TWITTER_INITIAL_PAGE_WAIT", "3"))
TAB_SWITCH_WAIT = int(os.environ.get("TWITTER_TAB_SWITCH_WAIT", "2"))

# Payload-shape constants live in twitter.browser_parse, next to the only code
# that reads them.
TWITTER_HOME_URL = os.environ.get("TWITTER_HOME_URL", "https://x.com/home")
SCROLL_INNER_HEIGHT_MULTIPLIER = 2
MS_PER_SECOND = 1000.0
# Scrolls and reports the resulting offset in one round trip, so the loop can
# tell "the page refuses to move" apart from "the page moved but had nothing
# new on it".
SCROLL_SCRIPT = (
    "() => { window.scrollBy(0, window.innerHeight * "
    f"{SCROLL_INNER_HEIGHT_MULTIPLIER}); return window.scrollY; }}"
)
EXACT_MATCH_PREVIEW_LIMIT = 80
CONTENT_MATCH_PREVIEW_LIMIT = 100
_login_kw_str = os.environ.get("TWITTER_LOGIN_KEYWORDS", "log in,login,sign in,signin")
LOGIN_KEYWORDS = tuple(k.strip() for k in _login_kw_str.split(",") if k.strip())
LOGGED_OUT_URL_MARKERS = ("/login", "/signin", "/i/flow/login")
# We always request /home; being left on the bare root means we were bounced.
LOGGED_OUT_ROOT_URLS = ("https://x.com", "https://twitter.com", "https://www.x.com")
EXIT_ERROR = 1

# Pre-compiled regular expressions for performance (John Carmack optimization)
RT_PREFIX_RE = re.compile(r"^RT @\w+: ")


def _click_tab(tab) -> None:
    """Click a timeline tab, tolerating camoufox's humanized cursor.

    x.com's tabs are `div[role=tab]` behind hover-revealed overlays, and
    camoufox's `humanize` animates the pointer, so a plain click routinely
    burns the actionability timeout. Scroll it in, hover it, then force the
    click: force still dispatches a trusted event, it only skips the
    visibility/stability wait that was timing out.
    """
    try:
        tab.scroll_into_view_if_needed(timeout=CLICK_TIMEOUT_MS)
    except Exception:
        pass
    try:
        tab.hover(timeout=CLICK_TIMEOUT_MS)
    except Exception:
        pass
    tab.click(timeout=CLICK_TIMEOUT_MS, force=True)


def _inject_cookies(context, cookies: list[dict]) -> None:
    """Load cookies into the context, and verify the session one survived.

    Adding cookies one at a time keeps a single malformed cookie from dropping
    the whole batch, but the per-cookie failures must NOT be swallowed: a
    rejected `expires` field once silently discarded all 17 cookies, and the
    only symptom was x.com serving its logged-out page.
    """
    rejected: list[str] = []
    for cookie in cookies:
        try:
            context.add_cookies([cookie])
        except Exception as e:
            rejected.append(f"{cookie.get('name')} ({str(e).splitlines()[0][-90:]})")

    if rejected:
        print(f"{WARN} {len(rejected)}/{len(cookies)} cookies rejected by the browser:")
        for line in rejected[:3]:
            print(f"{WARN}   {line}")

    if not has_session_cookie(context.cookies()):
        print(
            f"{WARN} The session cookie ({SESSION_COOKIE_NAME}) did not survive "
            f"injection — x.com would serve a logged-out page."
        )
        sys.exit(EXIT_ERROR)


def _is_logged_out(page) -> bool:
    """True when x.com served an anonymous page instead of the timeline.

    Stale cookies do not always land on /login: requesting /home while signed
    out bounces to the site root, whose title carries none of LOGIN_KEYWORDS.
    That page scrolls like any other, so missing it looks like a hung scroll.
    """
    url = (page.url or "").lower()
    if any(marker in url for marker in LOGGED_OUT_URL_MARKERS):
        return True
    if _strip_url(url) in LOGGED_OUT_ROOT_URLS:
        return True
    return any(kw in page.title().lower() for kw in LOGIN_KEYWORDS)


def _strip_url(url: str) -> str:
    """Drop query/fragment and any trailing slash for exact root comparison."""
    return url.split("?")[0].split("#")[0].rstrip("/")


def _close_quietly(closable) -> None:
    """Close a browser context/page, ignoring an already-dead handle."""
    try:
        closable.close()
    except Exception:
        pass


def _scroll_timeline(page, since_time, collected, get_oldest_seen) -> tuple[int, str]:
    """Scroll until one of the stop conditions trips.

    Returns (scrolls_performed, human-readable stop reason). Every exit path
    leaves `collected` intact so partial results survive an interrupt.
    """
    started_at = _monotonic()
    scrolls = 0
    stagnant = 0
    last_count = len(collected)
    last_offset = None

    try:
        while scrolls < MAX_SCROLLS:
            if is_shutdown_requested():
                return scrolls, "interrupted"
            if _monotonic() - started_at >= MAX_RUNTIME_S:
                return scrolls, f"hit the {MAX_RUNTIME_S:.0f}s runtime budget"

            try:
                offset = page.evaluate(SCROLL_SCRIPT)
            except Exception as e:
                return scrolls, f"the page stopped responding ({e})"
            scrolls += 1

            if not _sleep_between_scrolls(SCROLL_PAUSE_MS / MS_PER_SECOND):
                return scrolls, "interrupted"

            oldest_seen = get_oldest_seen()
            if oldest_seen and oldest_seen < since_time:
                return scrolls, "reached the requested time window"

            grew = len(collected) > last_count
            moved = last_offset is None or _scroll_offset_changed(offset, last_offset)
            if grew or moved:
                stagnant = 0
            else:
                stagnant += 1
                if stagnant >= STAGNANT_SCROLL_LIMIT:
                    return scrolls, (
                        f"no new tweets and no page movement for "
                        f"{STAGNANT_SCROLL_LIMIT} consecutive scrolls"
                    )
            last_count = len(collected)
            last_offset = offset
    except KeyboardInterrupt:
        # Only reachable when no drain-mode signal handler is installed —
        # setup_signals(drain=True) sets a flag instead of raising.
        return scrolls, "interrupted"

    return scrolls, f"reached the {MAX_SCROLLS}-scroll limit"


def _sleep_between_scrolls(seconds: float) -> bool:
    """Pause between scrolls. Returns False if a shutdown was requested."""
    remaining = seconds
    while remaining > 0:
        if is_shutdown_requested():
            return False
        time.sleep(min(SLEEP_STEP_S, remaining))
        remaining -= SLEEP_STEP_S
    return not is_shutdown_requested()


def _scroll_offset_changed(current, previous) -> bool:
    """True when the page actually moved between two scrolls.

    A non-numeric offset means we cannot tell, so assume it moved — treating
    an unreadable offset as "stuck" would end collection early.
    """
    try:
        return abs(float(current) - float(previous)) > 0.5
    except (TypeError, ValueError):
        return True


def _launch_chromium(pw, debug):
    """Launch stock chromium, installing the browser binary if it is missing."""
    try:
        return pw.chromium.launch(headless=not debug)
    except Exception as e:
        if "Executable doesn't exist" not in str(e):
            raise
        print(f"{WARN} Browser binary missing — running playwright install chromium ...")
        import subprocess

        # Drive playwright through the running interpreter: a bare "playwright"
        # on PATH belongs to whichever environment happens to be first, and
        # installs a binary this process will not find.
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"], check=True
        )
        return pw.chromium.launch(headless=not debug)


@contextmanager
def open_browser(debug: bool):
    """Yield (browser, backend_name).

    Prefers camoufox (anti-detect Firefox) and falls back to stock chromium,
    naming the reason for the downgrade. A backend pinned via
    TWITTER_BROWSER_BACKEND fails loudly instead of falling back.
    """
    if resolve_backend() == BACKEND_CAMOUFOX:
        try:
            browser, close_camoufox = launch_camoufox(debug)
        except Exception as e:
            if backend_explicitly_requested():
                raise
            print(f"{WARN} camoufox unavailable ({e}) — falling back to chromium.")
        else:
            print(f"{STEP} Browser backend: camoufox (anti-detect Firefox)")
            # GracefulShutdown also closes the browser if an interrupt lands
            # somewhere this `finally` would never run.
            try:
                with GracefulShutdown(close_camoufox):
                    yield browser, BACKEND_CAMOUFOX
            finally:
                close_camoufox()
            return

    if sync_playwright is None:
        print(f"{WARN} Playwright is not available. Install it with: uv sync")
        sys.exit(EXIT_ERROR)

    with sync_playwright() as pw:
        browser = _launch_chromium(pw, debug)
        print(f"{STEP} Browser backend: chromium")

        def close_browser():
            _close_quietly(browser)

        try:
            with GracefulShutdown(close_browser):
                yield browser, BACKEND_CHROMIUM
        finally:
            close_browser()


def collect_tweets_via_browser(since_time: datetime, debug: bool) -> list[dict]:
    print(f"{STEP} Looking for an x.com session in your browsers ...")
    cookies, source = get_browser_cookies()
    if not has_session_cookie(cookies):
        # Guest cookies are handed to anonymous visitors, so without this check
        # x.com quietly serves the logged-out landing page and we scroll static
        # marketing copy instead of a timeline.
        if cookies:
            print(
                f"{WARN} {source} has {len(cookies)} x.com cookies but no session "
                f"token ({SESSION_COOKIE_NAME}) — that browser is not signed in."
            )
        else:
            print(f"{WARN} No Twitter/X cookies found in any installed browser.")
        print(f"{WARN} Either log in to x.com in Chrome/Zen/Firefox, or run:")
        print(f"{WARN}   python3 -m twitter --login")
        sys.exit(EXIT_ERROR)
    print(f"{STEP} Using the x.com session from {source}.")

    all_tweets: list[dict] = []
    oldest_seen: datetime | None = None

    def handle_response(response):
        nonlocal oldest_seen
        url = response.url
        if "HomeTimeline" not in url and "HomeLatestTimeline" not in url:
            return
        try:
            body = response.json()
        except Exception:
            return
        batch = parse_tweets_from_response(body)
        for tweet in batch:
            all_tweets.append(tweet)
            if oldest_seen is None or tweet["created_at"] < oldest_seen:
                oldest_seen = tweet["created_at"]

    with open_browser(debug) as (browser, backend):
        context = browser.new_context()
        _inject_cookies(context, cookies)

        page = context.new_page()
        page.on("response", handle_response)

        try:
            page.goto(TWITTER_HOME_URL, wait_until="domcontentloaded", timeout=PAGE_LOAD_TIMEOUT_MS)
        except PWTimeout:
            pass

        try:
            page.wait_for_selector(
                '[data-testid="primaryColumn"], [role="tablist"], [autocomplete="username"]',
                timeout=INITIAL_PAGE_WAIT * MS_PER_SECOND,
            )
        except Exception:
            pass

        if _is_logged_out(page):
            print(
                f"{WARN} x.com served a logged-out page — the {source} session is stale."
            )
            print(f"{WARN} Re-open x.com in that browser, or run: python3 -m twitter --login")
            sys.exit(EXIT_ERROR)

        try:
            _following_env = os.environ.get("TWITTER_FOLLOWING_TAB", "")
            if _following_env:
                localized_following = [t.strip() for t in _following_env.split(",")]
            else:
                localized_following = (
                    "Following", "Abonnements", "Siguiendo", "Sigo", "Gefolgt", "关注", "フォロー中"
                )
            following_tab = None
            for term in localized_following:
                loc = page.locator('[role="tab"]', has_text=term).first
                if loc.count() > 0:
                    following_tab = loc
                    break

            if following_tab is None:
                tabs = page.locator('[role="tab"]')
                if tabs.count() > 1:
                    following_tab = tabs.nth(1)

            if following_tab is None:
                msg_warn = (
                    f"{WARN} Could not locate 'Following' tab "
                    "(defaulting to current view)."
                )
                print(msg_warn, file=sys.stderr)
            else:
                _click_tab(following_tab)
                try:
                    page.wait_for_selector(
                        '[role="tab"][aria-selected="true"]',
                        timeout=TAB_SWITCH_WAIT * MS_PER_SECOND,
                    )
                except Exception:
                    pass
                print(f"{STEP} Switched to the 'Following' tab.")
        except Exception as e:
            msg_fail = (
                f"{WARN} Failed to switch to 'Following' tab: {e}. "
                "Defaulting to current view."
            )
            print(msg_fail, file=sys.stderr)

        print(f"{STEP} Scrolling timeline (collecting tweets since {since_time.isoformat()}) ...")
        try:
            scrolls, stop_reason = _scroll_timeline(
                page, since_time, all_tweets, lambda: oldest_seen
            )
            print(f"{STEP} Stopped after {scrolls} scrolls — {stop_reason}.")
        finally:
            _close_quietly(context)

    filtered = [t for t in all_tweets if t["created_at"] >= since_time]
    rt_prefix = RT_PREFIX_RE
    seen_exact = set()
    seen_content = set()
    unique = []
    for t in filtered:
        exact_key = (t["screen_name"], t["text"][:EXACT_MATCH_PREVIEW_LIMIT])
        if exact_key in seen_exact:
            continue
        seen_exact.add(exact_key)

        content_key = rt_prefix.sub("", t["text"])[:CONTENT_MATCH_PREVIEW_LIMIT]
        if content_key in seen_content:
            continue
        seen_content.add(content_key)

        unique.append(t)

    unique.sort(key=lambda t: t["created_at"])
    return unique
