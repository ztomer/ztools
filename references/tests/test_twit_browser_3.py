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
