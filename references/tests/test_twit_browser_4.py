from datetime import datetime, timezone
from pathlib import Path
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
