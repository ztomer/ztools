#!/usr/bin/env python3
"""
Browser automation helpers for twitter_summarizer.
"""

import os
import re
import sys
import time
from datetime import datetime, timezone

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
except Exception:
    sync_playwright = PWTimeout = None

from twitter.cookies import get_chrome_cookies
from lib.tui import STEP, WARN

MAX_SCROLLS = int(os.environ.get("TWITTER_MAX_SCROLLS", "1200"))
SCROLL_PAUSE_MS = int(os.environ.get("TWITTER_SCROLL_PAUSE_MS", "1800"))

# Timeouts (milliseconds)
PAGE_LOAD_TIMEOUT_MS = int(os.environ.get("TWITTER_PAGE_LOAD_TIMEOUT_MS", "30000"))
CLICK_TIMEOUT_MS = int(os.environ.get("TWITTER_CLICK_TIMEOUT_MS", "5000"))

# Sleep durations (seconds)
INITIAL_PAGE_WAIT = 3
TAB_SWITCH_WAIT = 2

# Constants to eliminate magic numbers/strings (Mitchell Hashimoto design)
TWITTER_TYPENAME_VISIBILITY = "TweetWithVisibilityResults"
TWITTER_ITEM_TIMELINE_TWEET = "TimelineTweet"
TWITTER_TYPE_TIMELINE_ADD_ENTRIES = "TimelineAddEntries"
UNKNOWN_USER = "unknown"
TWITTER_DATE_FORMAT = "%a %b %d %H:%M:%S +0000 %Y"
TWITTER_HOME_URL = os.environ.get("TWITTER_HOME_URL", "https://x.com/home")
SCROLL_INNER_HEIGHT_MULTIPLIER = 2
MS_PER_SECOND = 1000.0
EXACT_MATCH_PREVIEW_LIMIT = 80
CONTENT_MATCH_PREVIEW_LIMIT = 100
LOGIN_KEYWORDS = ("log in", "login", "sign in", "signin")
EXIT_ERROR = 1

# Pre-compiled regular expressions for performance (John Carmack optimization)
RT_PREFIX_RE = re.compile(r"^RT @\w+: ")


def parse_tweets_from_response(data: dict) -> list[dict]:
    tweets = []
    try:
        instructions = (
            data.get("data", {})
            .get("home", {})
            .get("home_timeline_urt", {})
            .get("instructions", [])
        )
        for instruction in instructions:
            if instruction.get("type") != TWITTER_TYPE_TIMELINE_ADD_ENTRIES:
                continue
            for entry in instruction.get("entries", []):
                content = entry.get("content", {})
                item_content = content.get("itemContent", {})
                if item_content.get("itemType") != TWITTER_ITEM_TIMELINE_TWEET:
                    continue
                tweet_result = item_content.get(
                    "tweet_results", {}).get("result", {})
                if tweet_result.get("__typename") == TWITTER_TYPENAME_VISIBILITY:
                    tweet_result = tweet_result.get("tweet", tweet_result)

                legacy = tweet_result.get("legacy", {})
                user_result = (
                    tweet_result.get("core", {})
                    .get("user_results", {})
                    .get("result", {})
                )
                user_core = user_result.get("core", {})
                user_legacy = user_result.get("legacy", {})
                full_text = legacy.get("full_text", "")
                created_at_str = legacy.get("created_at", "")
                screen_name = (
                    user_core.get("screen_name")
                    or user_legacy.get("screen_name")
                    or UNKNOWN_USER
                )
                favorite_count = legacy.get("favorite_count", 0)
                retweet_count = legacy.get("retweet_count", 0)
                reply_count = legacy.get("reply_count", 0)
                tweet_id = legacy.get("id_str", "")
                in_reply_to = legacy.get("in_reply_to_screen_name", "")

                if not full_text or not created_at_str:
                    continue

                try:
                    created_at = datetime.strptime(
                        created_at_str, TWITTER_DATE_FORMAT
                    )
                    created_at = created_at.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

                tweet = {
                    "screen_name": screen_name,
                    "text": full_text,
                    "created_at": created_at,
                    "favorite_count": favorite_count,
                    "retweet_count": retweet_count,
                    "reply_count": reply_count,
                }
                if tweet_id:
                    tweet["id_str"] = tweet_id
                if in_reply_to:
                    tweet["in_reply_to_screen_name"] = in_reply_to

                tweets.append(tweet)
    except Exception as e:
        if os.environ.get("DEBUG"):
            print(f"Error: {e}")
    return tweets


def collect_tweets_via_browser(since_time: datetime, debug: bool) -> list[dict]:
    print(f"{STEP} Extracting Twitter/X cookies from Chrome profile ...")
    cookies = get_chrome_cookies()
    if not cookies:
        print(f"{WARN} No Twitter/X cookies found. Are you logged in to x.com in Chrome?")
        sys.exit(EXIT_ERROR)

    if sync_playwright is None:
        print(f"{WARN} Playwright is not available. Install it with: uv sync")
        sys.exit(EXIT_ERROR)

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

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=not debug)
        except Exception as e:
            if "Executable doesn't exist" in str(e):
                print(f"{WARN} Browser binary missing — running playwright install chromium ...")
                import subprocess
                subprocess.run(["playwright", "install", "chromium"], check=True)
                browser = pw.chromium.launch(headless=not debug)
            else:
                raise
        context = browser.new_context()
        for cookie in cookies:
            try:
                context.add_cookies([cookie])
            except Exception:
                pass

        page = context.new_page()
        page.on("response", handle_response)

        try:
            page.goto(
                TWITTER_HOME_URL, wait_until="domcontentloaded", timeout=PAGE_LOAD_TIMEOUT_MS
            )
        except PWTimeout:
            pass

        time.sleep(INITIAL_PAGE_WAIT)
        if any(
            kw in page.title().lower()
            for kw in LOGIN_KEYWORDS
        ):
            print(
                f"{WARN} Not logged in — cookies may be stale. Log into x.com in Chrome and retry."
            )
            sys.exit(EXIT_ERROR)

        try:
            following_tab = page.locator(
                '[role="tab"]', has_text="Following").first
            following_tab.click(timeout=CLICK_TIMEOUT_MS)
            time.sleep(TAB_SWITCH_WAIT)
        except Exception:
            pass

        print(
            f"{STEP} Scrolling timeline (collecting tweets since {since_time.isoformat()}) ..."
        )
        scrolls = 0
        try:
            while scrolls < MAX_SCROLLS:
                page.evaluate(f"window.scrollBy(0, window.innerHeight * {SCROLL_INNER_HEIGHT_MULTIPLIER})")
                time.sleep(SCROLL_PAUSE_MS / MS_PER_SECOND)
                scrolls += 1
                if oldest_seen and oldest_seen < since_time:
                    break
        except KeyboardInterrupt:
            print(f"\n{WARN} Scroll interrupted by user. Processing collected tweets...")
        finally:
            context.close()
            browser.close()

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
