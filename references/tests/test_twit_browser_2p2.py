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
