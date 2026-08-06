#!/usr/bin/env python3
"""Camoufox launch support for the twitter timeline collector.

Camoufox is an anti-detect Firefox build driven through the Playwright API, so
everything downstream of the launch (contexts, cookies, response interception)
is identical to the stock chromium path in `twitter.browser`. Only the launch
itself differs, which is all this module owns.
"""

import os

try:
    from camoufox.addons import DefaultAddons
    from camoufox.sync_api import Camoufox
except Exception:  # camoufox not installed
    Camoufox = DefaultAddons = None

BACKEND_AUTO = "auto"
BACKEND_CAMOUFOX = "camoufox"
BACKEND_CHROMIUM = "chromium"
VALID_BACKENDS = (BACKEND_AUTO, BACKEND_CAMOUFOX, BACKEND_CHROMIUM)

# "auto" prefers camoufox and silently falls back to chromium; naming a backend
# explicitly makes the failure loud instead of downgrading behind your back.
BROWSER_BACKEND = os.environ.get("TWITTER_BROWSER_BACKEND", BACKEND_AUTO).strip().lower()

# humanize animates the cursor. It is OFF by default because it buys nothing
# here — the timeline is scrolled via JS, and the one real click (the
# "Following" tab) never completes with it on: the animated pointer blows even
# a 15s actionability timeout, force=True included, so every run silently fell
# back to the "For You" feed. Measured: humanize on -> 27 tweets from For You;
# off -> 56 from Following. Set TWITTER_CAMOUFOX_HUMANIZE=1 to restore it.
CAMOUFOX_HUMANIZE = os.environ.get("TWITTER_CAMOUFOX_HUMANIZE", "0") != "0"
CAMOUFOX_OS = os.environ.get("TWITTER_CAMOUFOX_OS", "macos")
# Camoufox ships uBlock Origin enabled. It never touches x.com's own GraphQL
# endpoints, and dropping ad/analytics requests means fewer response events to
# sift per scroll — but allow turning it off if a filter list ever misfires.
CAMOUFOX_UBLOCK = os.environ.get("TWITTER_CAMOUFOX_UBLOCK", "1") != "0"


def camoufox_available() -> bool:
    """True when the camoufox python package imported successfully."""
    return Camoufox is not None


def resolve_backend() -> str:
    """Pick a backend from TWITTER_BROWSER_BACKEND and what is installed."""
    requested = BROWSER_BACKEND if BROWSER_BACKEND in VALID_BACKENDS else BACKEND_AUTO
    if requested == BACKEND_CHROMIUM:
        return BACKEND_CHROMIUM
    if requested == BACKEND_CAMOUFOX:
        return BACKEND_CAMOUFOX
    return BACKEND_CAMOUFOX if camoufox_available() else BACKEND_CHROMIUM


def backend_explicitly_requested() -> bool:
    """True when the user pinned a backend rather than leaving it on auto."""
    return BROWSER_BACKEND in (BACKEND_CAMOUFOX, BACKEND_CHROMIUM)


def _camoufox_options(headless: bool) -> dict:
    options = {
        "headless": headless,
        "humanize": CAMOUFOX_HUMANIZE,
        "os": CAMOUFOX_OS,
        "block_webrtc": True,
        "enable_cache": True,
    }
    if not CAMOUFOX_UBLOCK and DefaultAddons is not None:
        options["exclude_addons"] = [DefaultAddons.UBO]
    return options


def launch_camoufox_persistent(user_data_dir, headless: bool = True):
    """Start camoufox against a persistent profile. Returns (context, closer).

    persistent_context yields a BrowserContext rather than a Browser: the
    profile on disk *is* the storage state, so cookies survive between runs.
    """
    if Camoufox is None:
        raise RuntimeError(
            "camoufox is not installed — pip install camoufox && python3 -m camoufox fetch"
        )

    manager = Camoufox(
        persistent_context=True,
        user_data_dir=str(user_data_dir),
        **_camoufox_options(headless),
    )
    context = manager.__enter__()

    def closer():
        try:
            manager.__exit__(None, None, None)
        except Exception:
            pass

    return context, closer


def launch_camoufox(debug: bool):
    """Start camoufox and return (browser, closer).

    Raises on launch failure so the caller can decide whether to fall back.
    Call closer() to shut the browser down.
    """
    if Camoufox is None:
        raise RuntimeError(
            "camoufox is not installed — pip install camoufox && python3 -m camoufox fetch"
        )

    manager = Camoufox(**_camoufox_options(headless=not debug))
    browser = manager.__enter__()

    def closer():
        try:
            manager.__exit__(None, None, None)
        except Exception:
            pass

    return browser, closer
