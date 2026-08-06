from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from twitter.browser import parse_tweets_from_response

# A guest cookie set plus the session token, matching what a signed-in Chrome
# profile actually holds. Collection refuses to start without auth_token.
SIGNED_IN_COOKIES = [
    {"name": "guest_id", "value": "v1%3A1234", "domain": ".x.com"},
    {"name": "auth_token", "value": "deadbeefcafe", "domain": ".x.com"},
    {"name": "ct0", "value": "csrf123", "domain": ".x.com"},
]
GUEST_ONLY_COOKIES = [
    {"name": "guest_id", "value": "v1%3A1234", "domain": ".x.com"},
    {"name": "__cf_bm", "value": "abc", "domain": ".x.com"},
]


@pytest.fixture(autouse=True)
def clean_signal_state():
    """Backend pinning lives in conftest; this only resets shutdown state."""
    from lib.signal_handling import reset_signal_state

    yield
    reset_signal_state()


def _make_tweet_data(screen_name, text, date_str, typename="TimelineTweet",
                     fav_count=0, rt_count=0, reply_count=0, id_str="",
                     in_reply_to=""):
    legacy = {
        "full_text": text,
        "created_at": date_str,
        "favorite_count": fav_count,
        "retweet_count": rt_count,
        "reply_count": reply_count,
    }
    if id_str:
        legacy["id_str"] = id_str
    if in_reply_to:
        legacy["in_reply_to_screen_name"] = in_reply_to
    return {
        "data": {
            "home": {
                "home_timeline_urt": {
                    "instructions": [
                        {
                            "type": "TimelineAddEntries",
                            "entries": [
                                {
                                    "content": {
                                        "itemContent": {
                                            "itemType": "TimelineTweet",
                                            "tweet_results": {
                                                "result": {
                                                    "__typename": typename,
                                                    "legacy": legacy,
                                                    "core": {
                                                        "user_results": {
                                                            "result": {
                                                                "core": {
                                                                    "screen_name": screen_name,
                                                                },
                                                                "legacy": {},
                                                            }
                                                        }
                                                    },
                                                }
                                            },
                                        }
                                    },
                                }
                            ],
                        }
                    ],
                }
            }
        }
    }


class TestParseTweetsFromResponse:
    def test_empty_data(self):
        assert parse_tweets_from_response({}) == []

    def test_missing_fields(self):
        assert parse_tweets_from_response({"data": {}}) == []

    def test_parse_single_tweet(self):
        data = _make_tweet_data(
            "user1",
            "hello world",
            "Mon Jun 01 12:00:00 +0000 2026",
        )
        tweets = parse_tweets_from_response(data)
        assert len(tweets) == 1
        assert tweets[0]["screen_name"] == "user1"
        assert tweets[0]["text"] == "hello world"
        assert tweets[0]["favorite_count"] == 0
        assert tweets[0]["retweet_count"] == 0
        assert tweets[0]["reply_count"] == 0
        assert "id_str" not in tweets[0]
        assert "in_reply_to_screen_name" not in tweets[0]

    def test_datetime_parsed(self):
        data = _make_tweet_data(
            "user1", "test", "Mon Jun 01 12:00:00 +0000 2026"
        )
        tweets = parse_tweets_from_response(data)
        # +0000 parsed as UTC
        assert tweets[0]["created_at"].tzinfo is timezone.utc
        assert tweets[0]["created_at"].utcoffset().total_seconds() == 0
        assert tweets[0]["created_at"].year == 2026
        assert tweets[0]["created_at"].month == 6
        assert tweets[0]["created_at"].day == 1

    def test_invalid_date_skipped(self):
        data = _make_tweet_data("user1", "text", "not a date")
        tweets = parse_tweets_from_response(data)
        assert len(tweets) == 0

    def test_visibility_results_unwrapped(self):
        data_map = _make_tweet_data(
            "user1", "visible text", "Mon Jun 01 12:00:00 +0000 2026",
            typename="TweetWithVisibilityResults",
        )
        tweet_result = (
            data_map["data"]["home"]["home_timeline_urt"]
            ["instructions"][0]["entries"][0]["content"]
            ["itemContent"]["tweet_results"]["result"]
        )
        tweet_result["tweet"] = dict(tweet_result)
        tweets = parse_tweets_from_response(data_map)
        assert len(tweets) == 1
        assert tweets[0]["text"] == "visible text"

    def test_unknown_item_type_skipped(self):
        data = {
            "data": {
                "home": {
                    "home_timeline_urt": {
                        "instructions": [
                            {
                                "type": "TimelineAddEntries",
                                "entries": [
                                    {
                                        "content": {
                                            "itemContent": {
                                                "itemType": "TimelinePrompt",
                                            }
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                }
            }
        }
        assert parse_tweets_from_response(data) == []

    def test_wrong_instruction_type_skipped(self):
        data = {
            "data": {
                "home": {
                    "home_timeline_urt": {
                        "instructions": [
                            {
                                "type": "TimelinePinEntry",
                                "entries": [{}],
                            }
                        ],
                    }
                }
            }
        }
        assert parse_tweets_from_response(data) == []

    def test_empty_text_skipped(self):
        """Tweet with empty full_text is skipped."""
        data = _make_tweet_data("user1", "", "Mon Jun 01 12:00:00 +0000 2026")
        # Replace full_text with empty
        data["data"]["home"]["home_timeline_urt"]["instructions"][0]["entries"][0]["content"]["itemContent"]["tweet_results"]["result"]["legacy"]["full_text"] = ""
        assert parse_tweets_from_response(data) == []

    def test_empty_date_skipped(self):
        data = _make_tweet_data("user1", "text", "Mon Jun 01 12:00:00 +0000 2026")
        data["data"]["home"]["home_timeline_urt"]["instructions"][0]["entries"][0]["content"]["itemContent"]["tweet_results"]["result"]["legacy"]["created_at"] = ""
        assert parse_tweets_from_response(data) == []

    def test_screen_name_falls_back(self):
        """When user_core has no screen_name, fall back to user_legacy."""
        data = _make_tweet_data("user1", "text", "Mon Jun 01 12:00:00 +0000 2026")
        data["data"]["home"]["home_timeline_urt"]["instructions"][0]["entries"][0]["content"]["itemContent"]["tweet_results"]["result"]["core"]["user_results"]["result"]["core"] = {}
        data["data"]["home"]["home_timeline_urt"]["instructions"][0]["entries"][0]["content"]["itemContent"]["tweet_results"]["result"]["core"]["user_results"]["result"]["legacy"] = {"screen_name": "legacy_user"}
        tweets = parse_tweets_from_response(data)
        assert tweets[0]["screen_name"] == "legacy_user"

    def test_screen_name_unknown(self):
        """No screen_name anywhere → 'unknown'."""
        data = _make_tweet_data("user1", "text", "Mon Jun 01 12:00:00 +0000 2026")
        data["data"]["home"]["home_timeline_urt"]["instructions"][0]["entries"][0]["content"]["itemContent"]["tweet_results"]["result"]["core"]["user_results"]["result"]["core"] = {}
        data["data"]["home"]["home_timeline_urt"]["instructions"][0]["entries"][0]["content"]["itemContent"]["tweet_results"]["result"]["core"]["user_results"]["result"]["legacy"] = {}
        tweets = parse_tweets_from_response(data)
        assert tweets[0]["screen_name"] == "unknown"

    def test_top_level_exception_caught(self, monkeypatch, capsys):
        """Exception in parsing is caught, prints debug error, returns empty list."""
        # Use a malformed data structure that triggers AttributeError deep in the loop
        # Without try/except, this would raise AttributeError
        data = {"data": "not a dict"}
        monkeypatch.setenv("DEBUG", "1")
        result = parse_tweets_from_response(data)
        # Exception was caught
        assert result == []
        # DEBUG error was printed
        out = capsys.readouterr()
        assert "Error" in out.out
        assert "get" in out.out  # AttributeError: 'str' object has no attribute 'get'

    def test_import_error(self, monkeypatch):
        """When playwright fails to import, sync_playwright/PWTimeout are None."""
        import sys

        # Block playwright import
        monkeypatch.setitem(sys.modules, "playwright", None)
        monkeypatch.setitem(sys.modules, "playwright.sync_api", None)

        # Force re-import
        if "twitter.browser" in sys.modules:
            del sys.modules["twitter.browser"]
        import twitter.browser as twit_browser

        assert twit_browser.sync_playwright is None
        assert twit_browser.PWTimeout is None

        # Cleanup
        del sys.modules["twitter.browser"]
        # Re-import normally
        from twitter.browser import parse_tweets_from_response
        assert parse_tweets_from_response({}) == []


class TestCollectTweetsViaBrowser:
    """Tests for collect_tweets_via_browser - the main browser automation function."""

    def test_no_cookies_exits(self, monkeypatch):
        """When no cookies found, sys.exit(1) is called."""
        from twitter.browser import collect_tweets_via_browser
        with patch("twitter.browser.get_browser_cookies", return_value=([], "")), \
             patch("twitter.browser.sys") as mock_sys:
            mock_sys.exit.side_effect = SystemExit(1)
            with pytest.raises(SystemExit):
                collect_tweets_via_browser(since_time=datetime.now(timezone.utc), debug=False)
            mock_sys.exit.assert_called_with(1)

    def test_login_page_detected_exits(self, monkeypatch):
        """When page title indicates login, exit."""
        from twitter.browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.cookies.return_value = SIGNED_IN_COOKIES
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Log in to X / X"

        # Track the response handler so we can invoke it
        captured_handlers = []
        def on_response(event, handler):
            captured_handlers.append(handler)
        mock_page.on.side_effect = on_response

        with patch("twitter.browser.get_browser_cookies", return_value=(SIGNED_IN_COOKIES, "Zen")), \
             patch("twitter.browser.sync_playwright", return_value=mock_pw), \
             patch("twitter.browser.time"), \
             patch("twitter.browser.sys") as mock_sys:
            mock_sys.exit.side_effect = SystemExit(1)
            with pytest.raises(SystemExit):
                collect_tweets_via_browser(since_time=since, debug=False)
            mock_sys.exit.assert_called_with(1)

    def test_collects_tweets(self, monkeypatch):
        """Happy path: tweets are actually returned in the result list."""
        from twitter.browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.cookies.return_value = SIGNED_IN_COOKIES
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        # On page.on("response", handler), invoke handler immediately with mock data
        def on_response(event, handler):
            mock_response = MagicMock()
            mock_response.url = "https://x.com/api/graphql/HomeTimeline"
            mock_response.json.return_value = _make_tweet_data(
                "user1", "Hello world!", "Mon Jun 01 12:00:00 +0000 2026"
            )
            handler(mock_response)
        mock_page.on.side_effect = on_response

        with patch("twitter.browser.get_browser_cookies", return_value=(SIGNED_IN_COOKIES, "Zen")), \
             patch("twitter.browser.sync_playwright", return_value=mock_pw), \
             patch("twitter.browser.time"), \
             patch("twitter.browser.MAX_SCROLLS", 0):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)

        # Verify the tweet was actually collected and returned
        assert len(tweets) == 1
        assert tweets[0]["screen_name"] == "user1"
        assert tweets[0]["text"] == "Hello world!"
        assert tweets[0]["created_at"].year == 2026

    def test_following_tab_click_succeeds(self, monkeypatch):
        """When following tab click works, no exception."""
        from twitter.browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.cookies.return_value = SIGNED_IN_COOKIES
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        # Following tab exists and clicks
        mock_page.locator.return_value.first.click.return_value = None

        with patch("twitter.browser.get_browser_cookies", return_value=(SIGNED_IN_COOKIES, "Zen")), \
             patch("twitter.browser.sync_playwright", return_value=mock_pw), \
             patch("twitter.browser.time"), \
             patch("twitter.browser.MAX_SCROLLS", 0):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        assert tweets == []

    def test_following_tab_click_fails(self, monkeypatch):
        """When following tab click fails, swallowed."""
        from twitter.browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.cookies.return_value = SIGNED_IN_COOKIES
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        # Click raises exception
        mock_page.locator.return_value.first.click.side_effect = Exception("not found")

        with patch("twitter.browser.get_browser_cookies", return_value=(SIGNED_IN_COOKIES, "Zen")), \
             patch("twitter.browser.sync_playwright", return_value=mock_pw), \
             patch("twitter.browser.time"), \
             patch("twitter.browser.MAX_SCROLLS", 0):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        assert tweets == []

    def test_goto_timeout_caught(self, monkeypatch):
        """PWTimeout on page.goto is caught."""
        import twitter.browser as twit_browser
        from twitter.browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.cookies.return_value = SIGNED_IN_COOKIES
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        # page.goto raises PWTimeout
        class FakeTimeout(Exception):
            pass
        twit_browser.PWTimeout = FakeTimeout
        mock_page.goto.side_effect = FakeTimeout("timeout")

        with patch("twitter.browser.get_browser_cookies", return_value=(SIGNED_IN_COOKIES, "Zen")), \
             patch("twitter.browser.sync_playwright", return_value=mock_pw), \
             patch("twitter.browser.time"), \
             patch("twitter.browser.MAX_SCROLLS", 0):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        assert tweets == []

    def test_response_handler_url_not_match(self, monkeypatch):
        """Response URL doesn't match HomeTimeline — handler ignores it, oldest_seen stays None."""
        from twitter.browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.cookies.return_value = SIGNED_IN_COOKIES
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        # Capture handler and inject it into the scroll loop
        captured_handler = []
        def on_response(event, handler):
            captured_handler.append(handler)
        mock_page.on.side_effect = on_response

        # page.evaluate side effect: invoke the captured handler on first call
        # to verify the URL-doesn't-match branch
        def eval_side_effect(*args, **kwargs):
            if len(captured_handler) == 0:
                return None
            fake_response = MagicMock()
            fake_response.url = "https://x.com/some/other/endpoint"  # not HomeTimeline
            captured_handler[0](fake_response)
            return None
        mock_page.evaluate.side_effect = eval_side_effect

        with patch("twitter.browser.get_browser_cookies", return_value=(SIGNED_IN_COOKIES, "Zen")), \
             patch("twitter.browser.sync_playwright", return_value=mock_pw), \
             patch("twitter.browser.time"), \
             patch("twitter.browser.MAX_SCROLLS", 3):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        # Handler was called, but URL didn't match → oldest_seen stayed None → loop ran
        # all 3 scrolls (didn't break early). No tweets returned.
        assert tweets == []
        assert mock_page.evaluate.call_count == 3

    def test_response_handler_json_fails(self, monkeypatch):
        """Response.json() fails — handler returns silently, loop continues all scrolls."""
        from twitter.browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.cookies.return_value = SIGNED_IN_COOKIES
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        captured_handler = []
        def on_response(event, handler):
            captured_handler.append(handler)
        mock_page.on.side_effect = on_response

        def eval_side_effect(*args, **kwargs):
            if len(captured_handler) == 0:
                return None
            fake_response = MagicMock()
            fake_response.url = "https://x.com/api/graphql/HomeLatestTimeline"  # matches
            fake_response.json.side_effect = Exception("not json")
            captured_handler[0](fake_response)
            return None
        mock_page.evaluate.side_effect = eval_side_effect

        with patch("twitter.browser.get_browser_cookies", return_value=(SIGNED_IN_COOKIES, "Zen")), \
             patch("twitter.browser.sync_playwright", return_value=mock_pw), \
             patch("twitter.browser.time"), \
             patch("twitter.browser.MAX_SCROLLS", 3):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        # URL matched, but json() raised → handler returned silently → no oldest_seen
        # → loop ran all 3 scrolls
        assert tweets == []
        assert mock_page.evaluate.call_count == 3

    def test_loop_breaks_on_oldest_seen(self, monkeypatch):
        """Loop breaks when oldest_seen < since_time."""
        from twitter.browser import collect_tweets_via_browser
        since = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.cookies.return_value = SIGNED_IN_COOKIES
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        # Tweet older than since_time
        old_tweet_data = _make_tweet_data(
            "u", "old", "Mon Jan 01 12:00:00 +0000 2020"
        )

        captured_handler = []
        def on_response_capture(event, handler):
            captured_handler.append(handler)
        mock_page.on.side_effect = on_response_capture

        # Make page.evaluate trigger the handler on first call
        call_count = [0]
        def fake_evaluate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1 and captured_handler:
                fake_response = MagicMock()
                fake_response.url = "https://x.com/api/graphql/HomeTimeline"
                fake_response.json.return_value = old_tweet_data
                captured_handler[0](fake_response)
        mock_page.evaluate.side_effect = fake_evaluate

        with patch("twitter.browser.get_browser_cookies", return_value=(SIGNED_IN_COOKIES, "Zen")), \
             patch("twitter.browser.sync_playwright", return_value=mock_pw), \
             patch("twitter.browser.time"), \
             patch("twitter.browser.MAX_SCROLLS", 5):
            collect_tweets_via_browser(since_time=since, debug=False)
        # If we got here without looping 5 times, the break worked
        # Broke on first iteration (tweet already seen)
        assert call_count[0] <= 2  # max 2, actually 1 in this scenario

    def test_unique_filtering_exact_dedup(self, monkeypatch):
        """Exact duplicate tweets (same screen_name + first 80 chars) are deduped."""
        from twitter.browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.cookies.return_value = SIGNED_IN_COOKIES
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        captured_handler = []
        def on_response_capture(event, handler):
            captured_handler.append(handler)
        mock_page.on.side_effect = on_response_capture

        # On the first page.evaluate, invoke handler with the tweet.
        # On the second, invoke it again to test exact_key dedup.
        tweet_data = _make_tweet_data(
            "u", "Same exact text", "Mon Jun 01 12:00:00 +0000 2026"
        )
        call_count = [0]
        def fake_evaluate(*args, **kwargs):
            call_count[0] += 1
            if captured_handler and call_count[0] <= 2:
                mock_response = MagicMock()
                mock_response.url = "https://x.com/api/graphql/HomeTimeline"
                mock_response.json.return_value = tweet_data
                captured_handler[0](mock_response)
        mock_page.evaluate.side_effect = fake_evaluate

        with patch("twitter.browser.get_browser_cookies", return_value=(SIGNED_IN_COOKIES, "Zen")), \
             patch("twitter.browser.sync_playwright", return_value=mock_pw), \
             patch("twitter.browser.time"), \
             patch("twitter.browser.MAX_SCROLLS", 2):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        # First invocation adds, second is deduped
        assert len(tweets) == 1

    def test_signin_keyword_detected(self, monkeypatch):
        """'signin' in title also triggers exit."""
        from twitter.browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.cookies.return_value = SIGNED_IN_COOKIES
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Sign in to X"

        with patch("twitter.browser.get_browser_cookies", return_value=(SIGNED_IN_COOKIES, "Zen")), \
             patch("twitter.browser.sync_playwright", return_value=mock_pw), \
             patch("twitter.browser.time"), \
             patch("twitter.browser.sys") as mock_sys:
            mock_sys.exit.side_effect = SystemExit(1)
            with pytest.raises(SystemExit):
                collect_tweets_via_browser(since_time=since, debug=False)

    def test_loop_max_scrolls_reached(self, monkeypatch):
        """Loop completes when MAX_SCROLLS reached."""
        from twitter.browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.cookies.return_value = SIGNED_IN_COOKIES
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        with patch("twitter.browser.get_browser_cookies", return_value=(SIGNED_IN_COOKIES, "Zen")), \
             patch("twitter.browser.sync_playwright", return_value=mock_pw), \
             patch("twitter.browser.time"), \
             patch("twitter.browser.MAX_SCROLLS", 3):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        assert tweets == []

    def test_response_handler_with_tweets(self, monkeypatch):
        """Handler processes a valid HomeTimeline response with tweets — loop sees oldest_seen."""
        from twitter.browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.cookies.return_value = SIGNED_IN_COOKIES
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        # Tweet older than since_time so the loop breaks
        tweet_data = _make_tweet_data(
            "user1", "Hello", "Mon Jan 01 12:00:00 +0000 2018"
        )

        captured_handler = []
        def on_response(event, handler):
            captured_handler.append(handler)
        mock_page.on.side_effect = on_response

        def eval_side_effect(*args, **kwargs):
            if len(captured_handler) == 0:
                return None
            fake_response = MagicMock()
            fake_response.url = "https://x.com/api/graphql/HomeLatestTimeline"
            fake_response.json.return_value = tweet_data
            captured_handler[0](fake_response)
            return None
        mock_page.evaluate.side_effect = eval_side_effect

        with patch("twitter.browser.get_browser_cookies", return_value=(SIGNED_IN_COOKIES, "Zen")), \
             patch("twitter.browser.sync_playwright", return_value=mock_pw), \
             patch("twitter.browser.time"), \
             patch("twitter.browser.MAX_SCROLLS", 10):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        # Handler ran, set oldest_seen to 2018 which is < since (2020) → loop broke
        # after first scroll. Only 1 page.evaluate call.
        assert mock_page.evaluate.call_count == 1
        # The tweet (2018) is older than since_time (2020) → filtered out
        assert tweets == []

    def test_cookie_add_fails(self, monkeypatch):
        """When add_cookies fails, swallowed."""
        from twitter.browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.cookies.return_value = SIGNED_IN_COOKIES
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"
        mock_context.add_cookies.side_effect = Exception("bad cookie")

        with patch("twitter.browser.get_browser_cookies", return_value=(SIGNED_IN_COOKIES, "Zen")), \
             patch("twitter.browser.sync_playwright", return_value=mock_pw), \
             patch("twitter.browser.time"), \
             patch("twitter.browser.MAX_SCROLLS", 0):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        assert tweets == []

    def test_basic_collection(self, monkeypatch):
        """A single tweet is returned with all expected fields."""
        from twitter.browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.cookies.return_value = SIGNED_IN_COOKIES
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        # Use a function to capture handler
        def on_response(event, handler):
            # Invoke handler immediately with mock data
            mock_response = MagicMock()
            mock_response.url = "https://x.com/api/graphql/HomeTimeline"
            mock_response.json.return_value = _make_tweet_data(
                "u", "Some text", "Mon Jun 01 12:00:00 +0000 2026"
            )
            handler(mock_response)
        mock_page.on.side_effect = on_response

        with patch("twitter.browser.get_browser_cookies", return_value=(SIGNED_IN_COOKIES, "Zen")), \
             patch("twitter.browser.sync_playwright", return_value=mock_pw), \
             patch("twitter.browser.time"), \
             patch("twitter.browser.MAX_SCROLLS", 0):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        # Sanity: one tweet, sorted by created_at ascending
        assert len(tweets) == 1
        assert tweets[0]["text"] == "Some text"

    def test_returns_sorted_by_date(self, monkeypatch):
        """Tweets are sorted by created_at ascending."""
        from twitter.browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.cookies.return_value = SIGNED_IN_COOKIES
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        # Provide 3 tweets in non-sorted order
        def on_response(event, handler):
            mock_response = MagicMock()
            mock_response.url = "https://x.com/api/graphql/HomeTimeline"
            mock_response.json.return_value = {
                "data": {"home": {"home_timeline_urt": {"instructions": [{
                    "type": "TimelineAddEntries",
                    "entries": [
                        {"content": {"itemContent": {"itemType": "TimelineTweet", "tweet_results": {"result": {
                            "__typename": "Tweet",
                            "legacy": {"full_text": "third", "created_at": "Mon Jun 03 12:00:00 +0000 2026"},
                            "core": {"user_results": {"result": {"core": {"screen_name": "u"}, "legacy": {}}}},
                        }}}}},
                        {"content": {"itemContent": {"itemType": "TimelineTweet", "tweet_results": {"result": {
                            "__typename": "Tweet",
                            "legacy": {"full_text": "first", "created_at": "Mon Jun 01 12:00:00 +0000 2026"},
                            "core": {"user_results": {"result": {"core": {"screen_name": "u"}, "legacy": {}}}},
                        }}}}},
                        {"content": {"itemContent": {"itemType": "TimelineTweet", "tweet_results": {"result": {
                            "__typename": "Tweet",
                            "legacy": {"full_text": "second", "created_at": "Mon Jun 02 12:00:00 +0000 2026"},
                            "core": {"user_results": {"result": {"core": {"screen_name": "u"}, "legacy": {}}}},
                        }}}}},
                    ],
                }]}}}
            }
            handler(mock_response)
        mock_page.on.side_effect = on_response

        with patch("twitter.browser.get_browser_cookies", return_value=(SIGNED_IN_COOKIES, "Zen")), \
             patch("twitter.browser.sync_playwright", return_value=mock_pw), \
             patch("twitter.browser.time"), \
             patch("twitter.browser.MAX_SCROLLS", 0):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        # 3 tweets returned, sorted by date ascending
        assert len(tweets) == 3
        assert tweets[0]["text"] == "first"
        assert tweets[1]["text"] == "second"
        assert tweets[2]["text"] == "third"

    def test_unique_filtering_rt_dedup(self, monkeypatch):
        """RTs with same content after stripping prefix are deduped."""
        from twitter.browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.cookies.return_value = SIGNED_IN_COOKIES
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        def on_response(event, handler):
            # Two tweets with same underlying content but different RT prefixes
            data = {
                "data": {
                    "home": {
                        "home_timeline_urt": {
                            "instructions": [
                                {
                                    "type": "TimelineAddEntries",
                                    "entries": [
                                        {
                                            "content": {
                                                "itemContent": {
                                                    "itemType": "TimelineTweet",
                                                    "tweet_results": {
                                                        "result": {
                                                            "__typename": "Tweet",
                                                            "legacy": {
                                                                "full_text": "Original content here, exactly matching",
                                                                "created_at":
                                                                "Mon Jun 01 12:00:00 +0000 2026",
                                                            },
                                                            "core": {
                                                                "user_results": {
                                                                    "result": {
                                                                        "core": {"screen_name": "u1"},
                                                                        "legacy": {},
                                                                    }
                                                                }
                                                            },
                                                        }
                                                    },
                                                }
                                            },
                                        },
                                        {
                                            "content": {
                                                "itemContent": {
                                                    "itemType": "TimelineTweet",
                                                    "tweet_results": {
                                                        "result": {
                                                            "__typename": "Tweet",
                                                            "legacy": {
                                                                "full_text": "RT @someone: Original content here, exactly matching",
                                                                "created_at":
                                                                "Mon Jun 01 12:00:00 +0000 2026",
                                                            },
                                                            "core": {
                                                                "user_results": {
                                                                    "result": {
                                                                        "core": {"screen_name": "u2"},
                                                                        "legacy": {},
                                                                    }
                                                                }
                                                            },
                                                        }
                                                    },
                                                }
                                            },
                                        },
                                    ],
                                }
                            ],
                        }
                    }
                }
            }
            mock_response = MagicMock()
            mock_response.url = "https://x.com/api/graphql/HomeTimeline"
            mock_response.json.return_value = data
            handler(mock_response)
        mock_page.on.side_effect = on_response

        with patch("twitter.browser.get_browser_cookies", return_value=(SIGNED_IN_COOKIES, "Zen")), \
             patch("twitter.browser.sync_playwright", return_value=mock_pw), \
             patch("twitter.browser.time"), \
             patch("twitter.browser.MAX_SCROLLS", 0):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        # First tweet kept, second (RT with same content) deduped
        assert len(tweets) == 1

    def test_filtered_by_since_time(self, monkeypatch):
        """Tweets older than since_time are filtered out, recent ones kept."""
        from twitter.browser import collect_tweets_via_browser
        since = datetime(2025, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.cookies.return_value = SIGNED_IN_COOKIES
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        # Two tweets: one before since_time (filtered), one after (kept)
        def on_response(event, handler):
            mock_response = MagicMock()
            mock_response.url = "https://x.com/api/graphql/HomeTimeline"
            mock_response.json.return_value = {
                "data": {"home": {"home_timeline_urt": {"instructions": [{
                    "type": "TimelineAddEntries",
                    "entries": [
                        {"content": {"itemContent": {"itemType": "TimelineTweet", "tweet_results": {"result": {
                            "__typename": "Tweet",
                            "legacy": {"full_text": "old tweet", "created_at": "Mon Jan 01 12:00:00 +0000 2020"},
                            "core": {"user_results": {"result": {"core": {"screen_name": "u1"}, "legacy": {}}}},
                        }}}}},
                        {"content": {"itemContent": {"itemType": "TimelineTweet", "tweet_results": {"result": {
                            "__typename": "Tweet",
                            "legacy": {"full_text": "recent tweet", "created_at": "Mon Jun 01 12:00:00 +0000 2026"},
                            "core": {"user_results": {"result": {"core": {"screen_name": "u2"}, "legacy": {}}}},
                        }}}}},
                    ],
                }]}}}
            }
            handler(mock_response)
        mock_page.on.side_effect = on_response

        with patch("twitter.browser.get_browser_cookies", return_value=(SIGNED_IN_COOKIES, "Zen")), \
             patch("twitter.browser.sync_playwright", return_value=mock_pw), \
             patch("twitter.browser.time"), \
             patch("twitter.browser.MAX_SCROLLS", 0):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        # Only the recent tweet survives
        assert len(tweets) == 1
        assert tweets[0]["text"] == "recent tweet"
        assert tweets[0]["screen_name"] == "u2"

    def test_keyboard_interrupt_handling(self, monkeypatch):
        """KeyboardInterrupt during scroll loop is handled gracefully and returns collected tweets."""
        from twitter.browser import collect_tweets_via_browser
        since = datetime(2025, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.cookies.return_value = SIGNED_IN_COOKIES
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        def on_response(event, handler):
            mock_response = MagicMock()
            mock_response.url = "https://x.com/api/graphql/HomeTimeline"
            mock_response.json.return_value = {
                "data": {"home": {"home_timeline_urt": {"instructions": [{
                    "type": "TimelineAddEntries",
                    "entries": [
                        {"content": {"itemContent": {"itemType": "TimelineTweet", "tweet_results": {"result": {
                            "__typename": "Tweet",
                            "legacy": {"full_text": "collected tweet", "created_at": "Mon Jun 01 12:00:00 +0000 2026"},
                            "core": {"user_results": {"result": {"core": {"screen_name": "u1"}, "legacy": {}}}},
                        }}}}},
                    ],
                }]}}}
            }
            handler(mock_response)
        mock_page.on.side_effect = on_response

        # Raise KeyboardInterrupt on page.evaluate to simulate Ctrl+C during scrolling
        mock_page.evaluate.side_effect = KeyboardInterrupt("Ctrl+C")

        with patch("twitter.browser.get_browser_cookies", return_value=(SIGNED_IN_COOKIES, "Zen")), \
             patch("twitter.browser.sync_playwright", return_value=mock_pw), \
             patch("twitter.browser.time"), \
             patch("twitter.browser.print"):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        # Even with interrupt, the tweet collected in response is processed and returned
        assert len(tweets) == 1
        assert tweets[0]["text"] == "collected tweet"
        # Context and browser must be closed
        mock_context.close.assert_called_once()
        mock_browser.close.assert_called_once()


class TestScrollStopConditions:
    """The scroll loop must terminate on something other than MAX_SCROLLS.

    Its only original exit was "reached the requested time window", so a
    timeline that stopped serving tweets ran MAX_SCROLLS x SCROLL_PAUSE_MS
    (20+ minutes by default) collecting nothing.
    """

    SINCE = datetime(2020, 1, 1, tzinfo=timezone.utc)

    def _run(self, page, collected=None, oldest_seen=None, **overrides):
        import twitter.browser as tb

        settings = {"MAX_SCROLLS": 100, "SCROLL_PAUSE_MS": 0, "STAGNANT_SCROLL_LIMIT": 8}
        settings.update(overrides)
        collected = [] if collected is None else collected

        stack = [patch.object(tb, k, v) for k, v in settings.items()]
        for ctx in stack:
            ctx.start()
        try:
            return tb._scroll_timeline(page, self.SINCE, collected, lambda: oldest_seen)
        finally:
            for ctx in stack:
                ctx.stop()

    def test_stops_when_page_stops_moving(self):
        """Same scroll offset + no new tweets => give up well before MAX_SCROLLS."""
        page = MagicMock()
        page.evaluate.return_value = 500.0  # page pinned at the same offset

        scrolls, reason = self._run(page)

        # First scroll establishes the baseline, then 8 stagnant ones trip it.
        assert scrolls == 9
        assert "no new tweets and no page movement" in reason
        assert page.evaluate.call_count == 9

    def test_keeps_scrolling_while_the_page_still_moves(self):
        page = MagicMock()
        offsets = iter([100.0 * i for i in range(1, 200)])
        page.evaluate.side_effect = lambda _script: next(offsets)

        scrolls, reason = self._run(page, MAX_SCROLLS=12)

        assert scrolls == 12
        assert reason == "reached the 12-scroll limit"

    def test_keeps_scrolling_while_new_tweets_arrive(self):
        """A static offset is not stagnation if tweets are still landing."""
        page = MagicMock()
        collected = []
        page.evaluate.side_effect = lambda _script: (collected.append({"n": 1}), 500.0)[1]

        scrolls, reason = self._run(page, collected=collected, MAX_SCROLLS=15)

        assert scrolls == 15
        assert len(collected) == 15
        assert "no page movement" not in reason

    def test_stops_at_runtime_budget(self):
        import twitter.browser as tb

        page = MagicMock()
        offsets = iter([100.0 * i for i in range(1, 200)])
        page.evaluate.side_effect = lambda _script: next(offsets)
        clock = iter([0.0, 10.0, 20.0, 30.0, 40.0])

        with patch.object(tb, "_monotonic", lambda: next(clock)):
            scrolls, reason = self._run(page, MAX_RUNTIME_S=25.0)

        assert scrolls == 2
        assert "25s runtime budget" in reason

    def test_stops_when_time_window_reached(self):
        page = MagicMock()
        page.evaluate.side_effect = lambda _script: 100.0
        older_than_since = datetime(2019, 6, 1, tzinfo=timezone.utc)

        scrolls, reason = self._run(page, oldest_seen=older_than_since)

        assert scrolls == 1
        assert reason == "reached the requested time window"

    def test_stops_when_page_dies(self):
        page = MagicMock()
        page.evaluate.side_effect = Exception("Target page closed")

        scrolls, reason = self._run(page)

        assert scrolls == 0
        assert "stopped responding" in reason
        assert "Target page closed" in reason

    def test_shutdown_request_stops_and_preserves_collected(self):
        """Ctrl+C must not discard the tweets already gathered."""
        import lib.signal_handling as sig

        collected = [{"text": "already collected"}]
        page = MagicMock()

        def evaluate(_script):
            sig._shutdown_requested = True
            return 100.0

        page.evaluate.side_effect = evaluate

        scrolls, reason = self._run(page, collected=collected)

        assert reason == "interrupted"
        assert scrolls == 1
        assert collected == [{"text": "already collected"}]

    def test_keyboard_interrupt_stops_without_propagating(self):
        """Library use without setup_signals() still raises KeyboardInterrupt."""
        page = MagicMock()
        page.evaluate.side_effect = KeyboardInterrupt()

        scrolls, reason = self._run(page)

        assert scrolls == 0
        assert reason == "interrupted"


class TestBackendSelection:
    """Camoufox is preferred, chromium is the fallback, pinning is honoured."""

    def test_auto_prefers_camoufox_when_installed(self):
        import twitter.browser_launch as launch

        with patch.object(launch, "BROWSER_BACKEND", launch.BACKEND_AUTO), \
             patch.object(launch, "Camoufox", MagicMock()):
            assert launch.resolve_backend() == launch.BACKEND_CAMOUFOX

    def test_auto_uses_chromium_when_camoufox_missing(self):
        import twitter.browser_launch as launch

        with patch.object(launch, "BROWSER_BACKEND", launch.BACKEND_AUTO), \
             patch.object(launch, "Camoufox", None):
            assert launch.resolve_backend() == launch.BACKEND_CHROMIUM

    def test_pinned_chromium_wins_over_installed_camoufox(self):
        import twitter.browser_launch as launch

        with patch.object(launch, "BROWSER_BACKEND", launch.BACKEND_CHROMIUM), \
             patch.object(launch, "Camoufox", MagicMock()):
            assert launch.resolve_backend() == launch.BACKEND_CHROMIUM

    def test_unknown_backend_value_falls_back_to_auto(self):
        import twitter.browser_launch as launch

        with patch.object(launch, "BROWSER_BACKEND", "netscape"), \
             patch.object(launch, "Camoufox", None):
            assert launch.resolve_backend() == launch.BACKEND_CHROMIUM

    def test_open_browser_yields_camoufox_and_closes_it(self):
        import twitter.browser as tb

        camoufox_browser = MagicMock()
        closer = MagicMock()

        with patch.object(tb, "resolve_backend", return_value=tb.BACKEND_CAMOUFOX), \
             patch.object(tb, "launch_camoufox", return_value=(camoufox_browser, closer)):
            with tb.open_browser(debug=False) as (browser, backend):
                assert browser is camoufox_browser
                assert backend == tb.BACKEND_CAMOUFOX
                closer.assert_not_called()

        closer.assert_called_once()

    def test_open_browser_falls_back_to_chromium_and_says_why(self, capsys):
        import twitter.browser as tb

        mock_pw = MagicMock()
        chromium_browser = MagicMock()
        mock_pw.__enter__.return_value.chromium.launch.return_value = chromium_browser

        with patch.object(tb, "resolve_backend", return_value=tb.BACKEND_CAMOUFOX), \
             patch.object(tb, "launch_camoufox", side_effect=RuntimeError("binary missing")), \
             patch.object(tb, "backend_explicitly_requested", return_value=False), \
             patch.object(tb, "sync_playwright", return_value=mock_pw):
            with tb.open_browser(debug=False) as (browser, backend):
                assert browser is chromium_browser
                assert backend == tb.BACKEND_CHROMIUM

        out = capsys.readouterr().out
        assert "camoufox unavailable" in out
        assert "binary missing" in out  # the reason, not just "falling back"
        chromium_browser.close.assert_called_once()

    def test_pinned_camoufox_failure_is_not_silently_downgraded(self):
        import twitter.browser as tb

        with patch.object(tb, "resolve_backend", return_value=tb.BACKEND_CAMOUFOX), \
             patch.object(tb, "launch_camoufox", side_effect=RuntimeError("binary missing")), \
             patch.object(tb, "backend_explicitly_requested", return_value=True):
            with pytest.raises(RuntimeError, match="binary missing"):
                with tb.open_browser(debug=False):
                    pass


class TestLoggedOutDetection:
    """A guest session scrolls a static landing page — it must be caught early.

    This was the real cause of the "stuck scrolling" report: x.com bounced
    /home to its logged-out root, whose title matches none of LOGIN_KEYWORDS,
    so collection scrolled marketing copy until MAX_SCROLLS ran out.
    """

    def _page(self, url, title="X. It's what's happening / X"):
        page = MagicMock()
        page.url = url
        page.title.return_value = title
        return page

    def test_bounce_to_root_is_logged_out(self):
        from twitter.browser import _is_logged_out

        assert _is_logged_out(self._page("https://x.com/")) is True
        assert _is_logged_out(self._page("https://x.com")) is True
        assert _is_logged_out(self._page("https://twitter.com/?lang=en")) is True

    def test_login_flow_url_is_logged_out(self):
        from twitter.browser import _is_logged_out

        assert _is_logged_out(self._page("https://x.com/i/flow/login")) is True

    def test_login_title_is_logged_out(self):
        from twitter.browser import _is_logged_out

        assert _is_logged_out(self._page("https://x.com/home", "Log in to X / X")) is True

    def test_real_timeline_is_not_logged_out(self):
        from twitter.browser import _is_logged_out

        assert _is_logged_out(self._page("https://x.com/home", "Home / X")) is False

    def test_guest_cookies_abort_before_launching_a_browser(self, capsys):
        """No session token => say so, do not open a browser and scroll."""
        import twitter.browser as tb

        launch = MagicMock()
        with patch.object(tb, "get_browser_cookies", return_value=(GUEST_ONLY_COOKIES, "Chrome")), \
             patch.object(tb, "open_browser", launch):
            with pytest.raises(SystemExit) as excinfo:
                tb.collect_tweets_via_browser(
                    since_time=datetime(2020, 1, 1, tzinfo=timezone.utc), debug=False
                )

        assert excinfo.value.code == 1
        launch.assert_not_called()
        out = capsys.readouterr().out
        assert "auth_token" in out
        assert "not signed in" in out


@pytest.mark.real_cookie_discovery
class TestSessionCookieDetection:
    def test_session_token_recognised(self):
        from twitter.cookies import has_session_cookie

        assert has_session_cookie(SIGNED_IN_COOKIES) is True

    def test_guest_cookies_are_not_a_session(self):
        from twitter.cookies import has_session_cookie

        assert has_session_cookie(GUEST_ONLY_COOKIES) is False

    def test_empty_session_token_does_not_count(self):
        from twitter.cookies import has_session_cookie

        assert has_session_cookie([{"name": "auth_token", "value": ""}]) is False

    def test_signed_in_profile_wins_over_an_earlier_guest_profile(self, tmp_path):
        """Default holds guest cookies; the logged-in profile is elsewhere."""
        import twitter.cookies as ck

        default_db = tmp_path / "Default" / "Cookies"
        profile_db = tmp_path / "Profile 1" / "Cookies"
        for p in (default_db, profile_db):
            p.parent.mkdir(parents=True)
            p.touch()

        by_path = {default_db: GUEST_ONLY_COOKIES, profile_db: SIGNED_IN_COOKIES}
        with patch.object(ck, "CHROME_COOKIES_DB", default_db), \
             patch.object(ck, "_get_chrome_keychain_key", return_value=b"k" * 16), \
             patch.object(ck, "_read_profile_cookies", side_effect=lambda p, d, k: by_path[p]):
            cookies = ck.get_chrome_cookies()

        assert ck.has_session_cookie(cookies) is True
        assert cookies == SIGNED_IN_COOKIES

    def test_guest_cookies_returned_when_no_profile_is_signed_in(self, tmp_path):
        """Still return something so the caller can report what it found."""
        import twitter.cookies as ck

        default_db = tmp_path / "Default" / "Cookies"
        default_db.parent.mkdir(parents=True)
        default_db.touch()

        with patch.object(ck, "CHROME_COOKIES_DB", default_db), \
             patch.object(ck, "_get_chrome_keychain_key", return_value=b"k" * 16), \
             patch.object(ck, "_read_profile_cookies", return_value=GUEST_ONLY_COOKIES):
            cookies = ck.get_chrome_cookies()

        assert cookies == GUEST_ONLY_COOKIES


class TestFirefoxCookieExpiry:
    """moz_cookies.expiry unit drift silently discarded every cookie.

    Newer Firefox builds (Zen included) store expiry in MILLISECONDS. Passing
    that straight to Playwright makes add_cookies reject the cookie outright,
    so all 17 x.com cookies vanished and x.com served its logged-out page.
    """

    def test_millisecond_expiry_is_converted_to_seconds(self):
        from twitter.cookies_firefox import normalize_expiry

        # Real value observed in a Zen profile: ms since epoch.
        assert normalize_expiry(1818470613360) == 1818470613

    def test_second_expiry_is_left_alone(self):
        from twitter.cookies_firefox import normalize_expiry

        assert normalize_expiry(1818470613) == 1818470613

    def test_session_cookie_has_no_expiry(self):
        from twitter.cookies_firefox import normalize_expiry

        assert normalize_expiry(0) is None
        assert normalize_expiry(None) is None
        assert normalize_expiry(-1) is None

    def test_converted_expiry_is_accepted_as_a_unix_timestamp(self):
        """The whole point: the result must be a plausible future date."""
        import datetime

        from twitter.cookies_firefox import normalize_expiry

        got = normalize_expiry(1818470613360)
        year = datetime.datetime.fromtimestamp(got, datetime.timezone.utc).year
        assert 2020 < year < 2100, f"expiry landed in year {year}"

    def test_host_filter_does_not_match_unrelated_domains(self):
        """`LIKE '%x.com'` would also match netflix.com and dropbox.com."""
        from twitter.cookies_firefox import _host_matches_clause

        where, params = _host_matches_clause((".x.com",))
        assert "netflix" not in where
        assert params == ["x.com", ".x.com", "%.x.com"]


class TestCookieInjection:
    """A rejected cookie must never be swallowed into a silent logged-out run."""

    def test_reports_rejections_and_aborts_when_session_lost(self, capsys):
        import twitter.browser as tb

        context = MagicMock()
        context.add_cookies.side_effect = Exception("Cookie should have a valid expires")
        context.cookies.return_value = []

        with pytest.raises(SystemExit) as excinfo:
            tb._inject_cookies(context, SIGNED_IN_COOKIES)

        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert "3/3 cookies rejected" in out
        assert "valid expires" in out  # the browser's own reason, not a generic message
        assert "did not survive injection" in out

    def test_silent_success_when_session_lands(self, capsys):
        import twitter.browser as tb

        context = MagicMock()
        context.cookies.return_value = SIGNED_IN_COOKIES

        tb._inject_cookies(context, SIGNED_IN_COOKIES)

        assert context.add_cookies.call_count == 3
        assert "rejected" not in capsys.readouterr().out

    def test_partial_rejection_still_proceeds_if_session_survives(self, capsys):
        """One bad cookie must not abort a run that is genuinely authenticated."""
        import twitter.browser as tb

        context = MagicMock()
        context.add_cookies.side_effect = [Exception("bad"), None, None]
        context.cookies.return_value = SIGNED_IN_COOKIES

        tb._inject_cookies(context, SIGNED_IN_COOKIES)

        assert "1/3 cookies rejected" in capsys.readouterr().out


@pytest.mark.real_cookie_discovery
class TestBrowserCookieDiscovery:
    """Whichever browser is actually signed in wins, Firefox-family first."""

    def test_firefox_session_preferred_over_chrome(self):
        import twitter.cookies as ck

        with patch.object(ck, "firefox_profile_dbs", return_value=[Path("/zen/p/cookies.sqlite")]), \
             patch.object(ck, "read_firefox_cookies", return_value=SIGNED_IN_COOKIES), \
             patch.object(ck, "get_chrome_cookies", return_value=GUEST_ONLY_COOKIES):
            cookies, source = ck.get_browser_cookies()

        assert ck.has_session_cookie(cookies) is True
        assert "zen" in source or "p" in source

    def test_falls_through_to_chrome_when_firefox_is_guest_only(self):
        import twitter.cookies as ck

        with patch.object(ck, "firefox_profile_dbs", return_value=[Path("/zen/p/cookies.sqlite")]), \
             patch.object(ck, "read_firefox_cookies", return_value=GUEST_ONLY_COOKIES), \
             patch.object(ck, "get_chrome_cookies", return_value=SIGNED_IN_COOKIES):
            cookies, source = ck.get_browser_cookies()

        assert cookies == SIGNED_IN_COOKIES
        assert source == "Chrome"

    def test_reports_guest_source_when_nothing_is_signed_in(self):
        """Caller needs the source name to tell the user which browser to fix."""
        import twitter.cookies as ck

        with patch.object(ck, "firefox_profile_dbs", return_value=[Path("/zen/p/cookies.sqlite")]), \
             patch.object(ck, "read_firefox_cookies", return_value=GUEST_ONLY_COOKIES), \
             patch.object(ck, "get_chrome_cookies", return_value=[]):
            cookies, source = ck.get_browser_cookies()

        assert cookies == GUEST_ONLY_COOKIES
        assert source != ""


class TestFollowingTabReporting:
    """The 'could not locate' warning used to fire on the SUCCESS path."""

    def _collect(self, tab_count):
        import twitter.browser as tb

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.cookies.return_value = SIGNED_IN_COOKIES
        mock_context.new_page.return_value = mock_page
        mock_context.cookies.return_value = SIGNED_IN_COOKIES
        mock_page.title.return_value = "Home / X"
        mock_page.url = "https://x.com/home"
        mock_page.locator.return_value.first.count.return_value = tab_count
        mock_page.locator.return_value.count.return_value = tab_count

        with patch.object(tb, "get_browser_cookies", return_value=(SIGNED_IN_COOKIES, "Zen")), \
             patch.object(tb, "sync_playwright", return_value=mock_pw), \
             patch.object(tb, "time"), \
             patch.object(tb, "MAX_SCROLLS", 0):
            tb.collect_tweets_via_browser(
                since_time=datetime(2020, 1, 1, tzinfo=timezone.utc), debug=False
            )

    def test_no_false_warning_when_the_tab_is_found(self, capsys):
        self._collect(tab_count=1)
        captured = capsys.readouterr()
        assert "Could not locate" not in captured.err
        assert "Switched to the 'Following' tab" in captured.out

    def test_warns_when_the_tab_is_genuinely_missing(self, capsys):
        self._collect(tab_count=0)
        assert "Could not locate" in capsys.readouterr().err


class TestCamoufoxLaunchOptions:
    """humanize must stay off — it silently downgraded the feed."""

    def test_humanize_is_off_by_default(self):
        import twitter.browser_launch as launch

        assert launch.CAMOUFOX_HUMANIZE is False, (
            "humanize breaks the Following-tab click, so runs fall back to "
            "the For You feed and collect roughly half as many tweets"
        )

    def test_launch_options_carry_the_humanize_setting(self):
        import twitter.browser_launch as launch

        opts = launch._camoufox_options(headless=True)
        assert opts["humanize"] is False
        assert opts["headless"] is True

    def test_headful_option_for_login(self):
        import twitter.browser_launch as launch

        assert launch._camoufox_options(headless=False)["headless"] is False


class TestScrollOffsetComparison:
    def test_detects_movement(self):
        from twitter.browser import _scroll_offset_changed

        assert _scroll_offset_changed(1200.0, 800.0) is True

    def test_detects_stillness(self):
        from twitter.browser import _scroll_offset_changed

        assert _scroll_offset_changed(800.0, 800.0) is False

    def test_unreadable_offset_counts_as_movement(self):
        """Assuming "stuck" on an unreadable offset would end collection early."""
        from twitter.browser import _scroll_offset_changed

        assert _scroll_offset_changed(MagicMock(), 800.0) is True
        assert _scroll_offset_changed(None, 800.0) is True
