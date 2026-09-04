from datetime import timezone

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
