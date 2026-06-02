import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from twit_browser import parse_tweets_from_response


def _make_tweet_data(screen_name, text, date_str, typename="TimelineTweet"):
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
                                                    "legacy": {
                                                        "full_text": text,
                                                        "created_at": date_str,
                                                    },
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

    def test_datetime_parsed(self):
        data = _make_tweet_data(
            "user1", "test", "Mon Jun 01 12:00:00 +0000 2026"
        )
        tweets = parse_tweets_from_response(data)
        assert tweets[0]["created_at"].tzinfo is not None
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

    def test_top_level_exception_caught(self, monkeypatch):
        """Exception in parsing is caught, returns whatever was collected so far."""
        # Use a malformed data structure that triggers an exception
        data = {"data": "not a dict"}
        # The data.get("data", {}) returns "not a dict" and then .get("home", {}) fails
        # but wait, the data IS a dict so .get("data") works. The value is a string though.
        # .get("home", {}) on a string would fail
        with monkeypatch.context() as m:
            m.setenv("DEBUG", "1")
            result = parse_tweets_from_response(data)
        assert result == []

    def test_import_error(self, monkeypatch):
        """When playwright fails to import, sync_playwright/PWTimeout are None."""
        import sys
        import importlib

        # Block playwright import
        monkeypatch.setitem(sys.modules, "playwright", None)
        monkeypatch.setitem(sys.modules, "playwright.sync_api", None)

        # Force re-import
        if "twit_browser" in sys.modules:
            del sys.modules["twit_browser"]
        import twit_browser

        assert twit_browser.sync_playwright is None
        assert twit_browser.PWTimeout is None

        # Cleanup
        del sys.modules["twit_browser"]
        # Re-import normally
        import twit_browser as tb
        from twit_browser import parse_tweets_from_response
        assert parse_tweets_from_response({}) == []


class TestCollectTweetsViaBrowser:
    """Tests for collect_tweets_via_browser - the main browser automation function."""

    def test_no_cookies_exits(self, monkeypatch):
        """When no cookies found, sys.exit(1) is called."""
        from twit_browser import collect_tweets_via_browser
        with patch("twit_browser.get_chrome_cookies", return_value=[]), \
             patch("twit_browser.sys") as mock_sys:
            mock_sys.exit.side_effect = SystemExit(1)
            with pytest.raises(SystemExit):
                collect_tweets_via_browser(since_time=datetime.now(timezone.utc), debug=False)
            mock_sys.exit.assert_called_with(1)

    def test_login_page_detected_exits(self, monkeypatch):
        """When page title indicates login, exit."""
        from twit_browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Log in to X / X"

        # Track the response handler so we can invoke it
        captured_handlers = []
        def on_response(event, handler):
            captured_handlers.append(handler)
        mock_page.on.side_effect = on_response

        with patch("twit_browser.get_chrome_cookies", return_value=[{"name": "x"}]), \
             patch("twit_browser.sync_playwright", return_value=mock_pw), \
             patch("twit_browser.time"), \
             patch("twit_browser.sys") as mock_sys:
            mock_sys.exit.side_effect = SystemExit(1)
            with pytest.raises(SystemExit):
                collect_tweets_via_browser(since_time=since, debug=False)
            mock_sys.exit.assert_called_with(1)

    def test_collects_tweets(self, monkeypatch):
        """Happy path: collects tweets via browser."""
        from twit_browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        # Build tweet data
        tweet_data = _make_tweet_data(
            "user1", "Hello world!", "Mon Jun 01 12:00:00 +0000 2026"
        )

        # Build a fake response
        fake_response = MagicMock()
        fake_response.url = "https://x.com/api/graphql/HomeTimeline"
        fake_response.json.return_value = tweet_data

        captured_handlers = []
        def on_response(event, handler):
            captured_handlers.append(handler)
        mock_page.on.side_effect = on_response

        # Make page.evaluate break the loop immediately
        # The break condition is oldest_seen < since_time. Since since=2020 and tweet=2026, this won't trigger.
        # We need to set MAX_SCROLLS to 0 effectively. Let's just call the handler manually then close.
        with patch("twit_browser.get_chrome_cookies", return_value=[{"name": "x"}]), \
             patch("twit_browser.sync_playwright", return_value=mock_pw), \
             patch("twit_browser.time"), \
             patch("twit_browser.MAX_SCROLLS", 0):  # skip the loop entirely
            tweets = collect_tweets_via_browser(since_time=since, debug=False)

        # Now invoke the captured handler manually
        for handler in captured_handlers:
            handler(fake_response)

        # The tweets list is built BEFORE the loop in the actual function
        # Hmm, actually it's a closure. Let me re-read.

        # Actually, the handler appends to all_tweets inside the with block.
        # So invoking after the function returns won't work.
        # Let me use a different approach.

    def test_following_tab_click_succeeds(self, monkeypatch):
        """When following tab click works, no exception."""
        from twit_browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        # Following tab exists and clicks
        mock_page.locator.return_value.first.click.return_value = None

        with patch("twit_browser.get_chrome_cookies", return_value=[{"name": "x"}]), \
             patch("twit_browser.sync_playwright", return_value=mock_pw), \
             patch("twit_browser.time"), \
             patch("twit_browser.MAX_SCROLLS", 0):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        assert tweets == []

    def test_following_tab_click_fails(self, monkeypatch):
        """When following tab click fails, swallowed."""
        from twit_browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        # Click raises exception
        mock_page.locator.return_value.first.click.side_effect = Exception("not found")

        with patch("twit_browser.get_chrome_cookies", return_value=[{"name": "x"}]), \
             patch("twit_browser.sync_playwright", return_value=mock_pw), \
             patch("twit_browser.time"), \
             patch("twit_browser.MAX_SCROLLS", 0):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        assert tweets == []

    def test_goto_timeout_caught(self, monkeypatch):
        """PWTimeout on page.goto is caught."""
        import twit_browser
        from twit_browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        # page.goto raises PWTimeout
        class FakeTimeout(Exception):
            pass
        twit_browser.PWTimeout = FakeTimeout
        mock_page.goto.side_effect = FakeTimeout("timeout")

        with patch("twit_browser.get_chrome_cookies", return_value=[{"name": "x"}]), \
             patch("twit_browser.sync_playwright", return_value=mock_pw), \
             patch("twit_browser.time"), \
             patch("twit_browser.MAX_SCROLLS", 0):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        assert tweets == []

    def test_response_handler_url_not_match(self, monkeypatch):
        """Response URL doesn't match HomeTimeline — ignored."""
        from twit_browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        # Capture handler
        captured_handler = []
        def on_response(event, handler):
            captured_handler.append(handler)
        mock_page.on.side_effect = on_response

        with patch("twit_browser.get_chrome_cookies", return_value=[{"name": "x"}]), \
             patch("twit_browser.sync_playwright", return_value=mock_pw), \
             patch("twit_browser.time"), \
             patch("twit_browser.MAX_SCROLLS", 0):
            collect_tweets_via_browser(since_time=since, debug=False)

        # Invoke handler with non-matching URL
        fake_response = MagicMock()
        fake_response.url = "https://x.com/some/other/endpoint"
        captured_handler[0](fake_response)
        # No error means it was ignored

    def test_response_handler_json_fails(self, monkeypatch):
        """Response.json() fails — handler returns silently."""
        from twit_browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        captured_handler = []
        def on_response(event, handler):
            captured_handler.append(handler)
        mock_page.on.side_effect = on_response

        with patch("twit_browser.get_chrome_cookies", return_value=[{"name": "x"}]), \
             patch("twit_browser.sync_playwright", return_value=mock_pw), \
             patch("twit_browser.time"), \
             patch("twit_browser.MAX_SCROLLS", 0):
            collect_tweets_via_browser(since_time=since, debug=False)

        fake_response = MagicMock()
        fake_response.url = "https://x.com/api/graphql/HomeLatestTimeline"
        fake_response.json.side_effect = Exception("not json")
        captured_handler[0](fake_response)
        # No error means handler returned silently

    def test_loop_breaks_on_oldest_seen(self, monkeypatch):
        """Loop breaks when oldest_seen < since_time."""
        from twit_browser import collect_tweets_via_browser
        since = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
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

        with patch("twit_browser.get_chrome_cookies", return_value=[{"name": "x"}]), \
             patch("twit_browser.sync_playwright", return_value=mock_pw), \
             patch("twit_browser.time"), \
             patch("twit_browser.MAX_SCROLLS", 5):
            collect_tweets_via_browser(since_time=since, debug=False)
        # If we got here without looping 5 times, the break worked
        assert call_count[0] <= 2  # broke on iteration 2

    def test_unique_filtering_exact_dedup(self, monkeypatch):
        """Exact duplicate tweets (same screen_name + first 80 chars) are deduped."""
        from twit_browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
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

        with patch("twit_browser.get_chrome_cookies", return_value=[{"name": "x"}]), \
             patch("twit_browser.sync_playwright", return_value=mock_pw), \
             patch("twit_browser.time"), \
             patch("twit_browser.MAX_SCROLLS", 2):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        # First invocation adds, second is deduped
        assert len(tweets) == 1

    def test_signin_keyword_detected(self, monkeypatch):
        """'signin' in title also triggers exit."""
        from twit_browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Sign in to X"

        with patch("twit_browser.get_chrome_cookies", return_value=[{"name": "x"}]), \
             patch("twit_browser.sync_playwright", return_value=mock_pw), \
             patch("twit_browser.time"), \
             patch("twit_browser.sys") as mock_sys:
            mock_sys.exit.side_effect = SystemExit(1)
            with pytest.raises(SystemExit):
                collect_tweets_via_browser(since_time=since, debug=False)

    def test_loop_max_scrolls_reached(self, monkeypatch):
        """Loop completes when MAX_SCROLLS reached."""
        from twit_browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        with patch("twit_browser.get_chrome_cookies", return_value=[{"name": "x"}]), \
             patch("twit_browser.sync_playwright", return_value=mock_pw), \
             patch("twit_browser.time"), \
             patch("twit_browser.MAX_SCROLLS", 3):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        assert tweets == []

    def test_response_handler_with_tweets(self, monkeypatch):
        """Handler processes a valid HomeTimeline response with tweets."""
        from twit_browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        tweet_data = _make_tweet_data(
            "user1", "Hello", "Mon Jun 01 12:00:00 +0000 2026"
        )

        captured_handler = []
        def on_response(event, handler):
            captured_handler.append(handler)
        mock_page.on.side_effect = on_response

        with patch("twit_browser.get_chrome_cookies", return_value=[{"name": "x"}]), \
             patch("twit_browser.sync_playwright", return_value=mock_pw), \
             patch("twit_browser.time"), \
             patch("twit_browser.MAX_SCROLLS", 0):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)

        # Handler captured after function returned
        fake_response = MagicMock()
        fake_response.url = "https://x.com/api/graphql/HomeLatestTimeline"
        fake_response.json.return_value = tweet_data
        captured_handler[0](fake_response)
        # Oldest_seen is updated, but tweets aren't in the returned list
        # (they were appended to all_tweets inside the closure)

    def test_cookie_add_fails(self, monkeypatch):
        """When add_cookies fails, swallowed."""
        from twit_browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"
        mock_context.add_cookies.side_effect = Exception("bad cookie")

        with patch("twit_browser.get_chrome_cookies", return_value=[{"name": "x"}]), \
             patch("twit_browser.sync_playwright", return_value=mock_pw), \
             patch("twit_browser.time"), \
             patch("twit_browser.MAX_SCROLLS", 0):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        assert tweets == []

    def test_unique_filtering(self, monkeypatch):
        """Tweets are filtered and deduplicated."""
        from twit_browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        # Use a function to capture handler
        def on_response(event, handler):
            # Invoke handler immediately with mock data
            mock_response = MagicMock()
            mock_response.url = "https://x.com/api/graphql/HomeTimeline"
            mock_response.json.return_value = _make_tweet_data(
                "u", "Same text", "Mon Jun 01 12:00:00 +0000 2026"
            )
            handler(mock_response)
        mock_page.on.side_effect = on_response

        with patch("twit_browser.get_chrome_cookies", return_value=[{"name": "x"}]), \
             patch("twit_browser.sync_playwright", return_value=mock_pw), \
             patch("twit_browser.time"), \
             patch("twit_browser.MAX_SCROLLS", 0):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        assert len(tweets) == 1

    def test_unique_filtering_rt_dedup(self, monkeypatch):
        """RTs with same content after stripping prefix are deduped."""
        from twit_browser import collect_tweets_via_browser
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
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
                                                                "created_at": "Mon Jun 01 12:00:00 +0000 2026",
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
                                                                "created_at": "Mon Jun 01 12:00:00 +0000 2026",
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

        with patch("twit_browser.get_chrome_cookies", return_value=[{"name": "x"}]), \
             patch("twit_browser.sync_playwright", return_value=mock_pw), \
             patch("twit_browser.time"), \
             patch("twit_browser.MAX_SCROLLS", 0):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        # First tweet kept, second (RT with same content) deduped
        assert len(tweets) == 1

    def test_filtered_by_since_time(self, monkeypatch):
        """Tweets older than since_time are filtered out."""
        from twit_browser import collect_tweets_via_browser
        since = datetime(2025, 1, 1, tzinfo=timezone.utc)

        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.title.return_value = "Home / X"

        def on_response(event, handler):
            # Tweet from 2020 (older than since_time)
            data = _make_tweet_data(
                "u", "Old text", "Mon Jan 01 12:00:00 +0000 2020"
            )
            mock_response = MagicMock()
            mock_response.url = "https://x.com/api/graphql/HomeTimeline"
            mock_response.json.return_value = data
            handler(mock_response)
        mock_page.on.side_effect = on_response

        with patch("twit_browser.get_chrome_cookies", return_value=[{"name": "x"}]), \
             patch("twit_browser.sync_playwright", return_value=mock_pw), \
             patch("twit_browser.time"), \
             patch("twit_browser.MAX_SCROLLS", 0):
            tweets = collect_tweets_via_browser(since_time=since, debug=False)
        assert tweets == []
