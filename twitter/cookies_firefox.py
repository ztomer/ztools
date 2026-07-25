#!/usr/bin/env python3
"""Cookie extraction for Firefox-family browsers (Zen, Firefox, LibreWolf, ...).

Unlike the Chromium family these store cookies in a plain `cookies.sqlite` with
no encryption and no keychain round trip, so extraction is just a query. That
also makes them the better source for camoufox, which *is* Firefox — the cookie
dicts drop straight into `context.add_cookies()`.
"""

import shutil
import sqlite3
import tempfile
from pathlib import Path

# Profile roots under ~/Library/Application Support. Order is preference order
# when more than one browser holds a session.
FIREFOX_FAMILY_ROOTS = (
    "zen/Profiles",
    "Firefox/Profiles",
    "LibreWolf/Profiles",
    "Waterfox/Profiles",
)
COOKIE_DB_NAME = "cookies.sqlite"
DEFAULT_COOKIE_PATH = "/"
# Only the default container. Firefox container tabs duplicate a cookie per
# container, and mixing them yields two conflicting auth_tokens.
DEFAULT_CONTAINER = ""

_QUERY = (
    "SELECT name, value, host, path, expiry, isSecure, isHttpOnly, originAttributes "
    "FROM moz_cookies WHERE {where}"
)

# moz_cookies.expiry is seconds in classic Firefox but MILLISECONDS in newer
# builds (Zen included). A second-based timestamp this large would be year 5138,
# so anything above the bound is milliseconds. Playwright rejects the raw value
# outright ("only -1 or a positive number for the unix timestamp in s"), which
# silently drops every cookie including the session token.
_MAX_PLAUSIBLE_EPOCH_SECONDS = 100_000_000_000


def normalize_expiry(expiry) -> int | None:
    """Convert a moz_cookies expiry to unix seconds, or None if it has none."""
    try:
        value = int(expiry)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if value > _MAX_PLAUSIBLE_EPOCH_SECONDS:
        value //= 1000
    return value


def firefox_profile_dbs(app_support: Path | None = None) -> list[Path]:
    """Every Firefox-family cookie DB on this machine, in preference order."""
    base = app_support or (Path.home() / "Library" / "Application Support")
    found: list[Path] = []
    for root in FIREFOX_FAMILY_ROOTS:
        for db in sorted((base / root).glob(f"*/{COOKIE_DB_NAME}")):
            if db.exists():
                found.append(db)
    return found


def _host_matches_clause(domains: tuple[str, ...]) -> tuple[str, list]:
    """Build an exact-suffix host filter.

    A naive `host LIKE '%x.com'` also matches netflix.com and dropbox.com, so
    match the dotted suffix or the bare domain explicitly.
    """
    clauses, params = [], []
    for d in domains:
        bare = d.lstrip(".")
        clauses.append("(host = ? OR host = ? OR host LIKE ?)")
        params.extend([bare, f".{bare}", f"%.{bare}"])
    return " OR ".join(clauses), params


def read_firefox_cookies(db_path: Path, domains: tuple[str, ...]) -> list[dict]:
    """Read matching cookies out of one Firefox profile, newest container first."""
    tf = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp = Path(tf.name)
    tf.close()
    try:
        # Copy first: the DB may be WAL-locked by a running browser.
        shutil.copy2(db_path, tmp)
        where, params = _host_matches_clause(domains)
        conn = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
        rows = conn.execute(_QUERY.format(where=where), params).fetchall()
        conn.close()
    except Exception:
        return []
    finally:
        tmp.unlink(missing_ok=True)

    cookies: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for name, value, host, path, expiry, is_secure, is_http_only, origin_attrs in rows:
        if (origin_attrs or "") != DEFAULT_CONTAINER:
            continue
        if not value:
            continue
        key = (name, host, path or DEFAULT_COOKIE_PATH)
        if key in seen:
            continue
        seen.add(key)
        cookie: dict = {
            "name": name,
            "value": value,
            "domain": host,
            "path": path or DEFAULT_COOKIE_PATH,
            "secure": bool(is_secure),
            "httpOnly": bool(is_http_only),
        }
        expires = normalize_expiry(expiry)
        if expires is not None:
            cookie["expires"] = expires
        cookies.append(cookie)
    return cookies
