#!/usr/bin/env python3
"""Persistent camoufox profile holding the x.com login.

Scraping cookies out of Chrome is fragile — it depends on the keychain, on
guessing which of N Chrome profiles is signed in, and on the user happening to
be logged in there at all. A persistent camoufox profile removes all of that:
the user signs in once through `--login`, x.com issues its cookies to that
profile, and every later run reuses them headlessly.

We never see or store a password. Only the cookies x.com hands out live on
disk, in the profile directory.
"""

import os
import time
from pathlib import Path

from lib.tui import OK, STEP, WARN

from twitter.browser_launch import launch_camoufox_persistent
from twitter.cookies import SESSION_COOKIE_NAME

PROFILE_DIR = Path(
    os.environ.get("TWITTER_PROFILE_DIR", "~/.twitter-camoufox-profile")
).expanduser()

LOGIN_URL = os.environ.get("TWITTER_LOGIN_URL", "https://x.com/i/flow/login")
LOGIN_TIMEOUT_S = float(os.environ.get("TWITTER_LOGIN_TIMEOUT_S", "300"))
LOGIN_POLL_S = float(os.environ.get("TWITTER_LOGIN_POLL_S", "2"))
# Firefox stores a populated profile as a directory tree; an empty dir means
# `--login` was never completed.
_PROFILE_MARKER = "cookies.sqlite"


def profile_exists() -> bool:
    """True when a camoufox profile directory has actually been populated."""
    return (PROFILE_DIR / _PROFILE_MARKER).exists()


def context_has_session(context) -> bool:
    """True when the live browser context holds an x.com session token."""
    try:
        cookies = context.cookies()
    except Exception:
        return False
    return any(
        c.get("name") == SESSION_COOKIE_NAME and c.get("value") for c in cookies
    )


def open_session(headless: bool = True):
    """Open the persistent profile. Returns (context, closer)."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    return launch_camoufox_persistent(PROFILE_DIR, headless=headless)


def has_saved_session() -> bool:
    """True when the on-disk profile holds an x.com session token.

    Reads the profile's own cookies.sqlite rather than launching a browser, so
    the caller can choose a backend without paying a browser start-up.
    """
    if not profile_exists():
        return False
    import shutil
    import sqlite3
    import tempfile

    tf = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp = Path(tf.name)
    tf.close()
    try:
        shutil.copy2(PROFILE_DIR / _PROFILE_MARKER, tmp)
        conn = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
        row = conn.execute(
            "SELECT COUNT(*) FROM moz_cookies "
            "WHERE name = ? AND value != '' AND (host = '.x.com' OR host = 'x.com')",
            (SESSION_COOKIE_NAME,),
        ).fetchone()
        conn.close()
        return bool(row and row[0])
    except Exception:
        return False
    finally:
        tmp.unlink(missing_ok=True)


def login(timeout_s: float = LOGIN_TIMEOUT_S) -> bool:
    """Open a real window so the user can sign in to x.com themselves.

    Polls until x.com issues a session cookie to the profile. Returns True on
    success. The password is typed by the user into the browser — it never
    passes through this process.
    """
    print(f"{STEP} Opening x.com in a camoufox window — sign in there.")
    print(f"{STEP} Profile: {PROFILE_DIR}")
    print(f"{STEP} Waiting up to {timeout_s:.0f}s for the session cookie ...")

    context, closer = open_session(headless=False)
    try:
        page = context.pages[0] if context.pages else context.new_page()
        try:
            page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=60_000)
        except Exception as e:
            print(f"{WARN} Could not open {LOGIN_URL}: {e}")

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if context_has_session(context):
                print(f"{OK} Signed in — session saved to {PROFILE_DIR}")
                print(f"{OK} Future runs are headless; no login needed.")
                return True
            time.sleep(LOGIN_POLL_S)

        print(f"{WARN} Timed out after {timeout_s:.0f}s without a session cookie.")
        print(f"{WARN} Re-run the login and complete the sign-in in the window.")
        return False
    finally:
        # Closing the context is what flushes cookies.sqlite to disk.
        closer()
