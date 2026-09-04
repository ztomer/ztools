from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

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


class TestCollectTweetsViaBrowser:
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
