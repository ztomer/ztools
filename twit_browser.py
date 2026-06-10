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

from twit_cookies import get_chrome_cookies
from lib.tui import STEP, WARN

MAX_SCROLLS = 1200
SCROLL_PAUSE_MS = 1800


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
            if instruction.get("type") != "TimelineAddEntries":
                continue
            for entry in instruction.get("entries", []):
                content = entry.get("content", {})
                item_content = content.get("itemContent", {})
                if item_content.get("itemType") != "TimelineTweet":
                    continue
                tweet_result = item_content.get(
                    "tweet_results", {}).get("result", {})
                if tweet_result.get("__typename") == "TweetWithVisibilityResults":
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
                    or "unknown"
                )

                if not full_text or not created_at_str:
                    continue

                try:
                    created_at = datetime.strptime(
                        created_at_str, "%a %b %d %H:%M:%S +0000 %Y"
                    )
                    created_at = created_at.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

                tweets.append(
                    {
                        "screen_name": screen_name,
                        "text": full_text,
                        "created_at": created_at,
                    }
                )
    except Exception as e:
        if os.environ.get("DEBUG"):
            print(f"Error: {e}")
    return tweets


def collect_tweets_via_browser(since_time: datetime, debug: bool) -> list[dict]:
    print(f"{STEP} Extracting Twitter/X cookies from Chrome profile ...")
    cookies = get_chrome_cookies()
    if not cookies:
        print(f"{WARN} No Twitter/X cookies found. Are you logged in to x.com in Chrome?")
        sys.exit(1)

    if sync_playwright is None:
        print(f"{WARN} Playwright is not available. Install it with: uv sync")
        sys.exit(1)

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
                "https://x.com/home", wait_until="domcontentloaded", timeout=30000
            )
        except PWTimeout:
            pass

        time.sleep(3)
        if any(
            kw in page.title().lower()
            for kw in ("log in", "login", "sign in", "signin")
        ):
            print(
                f"{WARN} Not logged in — cookies may be stale. Log into x.com in Chrome and retry."
            )
            sys.exit(1)

        try:
            following_tab = page.locator(
                '[role="tab"]', has_text="Following").first
            following_tab.click(timeout=5000)
            time.sleep(2)
        except Exception:
            pass

        print(
            f"{STEP} Scrolling timeline (collecting tweets since {since_time.isoformat()}) ..."
        )
        scrolls = 0
        while scrolls < MAX_SCROLLS:
            page.evaluate("window.scrollBy(0, window.innerHeight * 2)")
            time.sleep(SCROLL_PAUSE_MS / 1000)
            scrolls += 1
            if oldest_seen and oldest_seen < since_time:
                break

        context.close()
        browser.close()

    filtered = [t for t in all_tweets if t["created_at"] >= since_time]
    rt_prefix = re.compile(r"^RT @\w+: ")
    seen_exact = set()
    seen_content = set()
    unique = []
    for t in filtered:
        exact_key = (t["screen_name"], t["text"][:80])
        if exact_key in seen_exact:
            continue
        seen_exact.add(exact_key)

        content_key = rt_prefix.sub("", t["text"])[:100]
        if content_key in seen_content:
            continue
        seen_content.add(content_key)

        unique.append(t)

    unique.sort(key=lambda t: t["created_at"])
    return unique
