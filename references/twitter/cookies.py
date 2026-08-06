#!/usr/bin/env python3
"""
Cookie extraction helpers for twitter_summarizer.
"""

import hashlib
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from lib.tui import WARN
from twitter.cookies_firefox import firefox_profile_dbs, read_firefox_cookies

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
except ImportError:
    Cipher = algorithms = modes = None
    import warnings

    warnings.warn("cryptography module not installed — Chrome cookie decryption will fail")

from lib.config_toml import load_config

_TWITTER_CONFIG_PATH = Path(__file__).parent.parent / "conf" / "twitter.toml"
_TWITTER_CFG = load_config(_TWITTER_CONFIG_PATH) or {}

_configured_cookies = _TWITTER_CFG.get("chrome_cookies_db")
_DEFAULT_CHROME_PATH = (
    Path.home() / "Library" / "Application Support" / "Google" / "Chrome" / "Default" / "Cookies"
)
CHROME_COOKIES_DB = (
    Path(_configured_cookies).expanduser() if _configured_cookies else _DEFAULT_CHROME_PATH
)

# Keychain and Cryptographic Constants to eliminate magic numbers/strings
# (Mitchell Hashimoto design)
KEYCHAIN_COMMAND = ["security", "find-generic-password", "-w", "-s", "Chrome Safe Storage"]
PBKDF2_ALGORITHM = "sha1"
PBKDF2_SALT = b"saltysalt"
PBKDF2_ITERATIONS = 1003
KEY_LENGTH = 16
V10_HEADER = b"v10"
IV_START_OFFSET = 3
IV_END_OFFSET = 19
CIPHERTEXT_START_OFFSET = 19
AES_BLOCK_SIZE = 16
PLAINTEXT_START_OFFSET = 16
FILETIME_TO_UNIX_DIVISOR = 1_000_000
FILETIME_TO_UNIX_OFFSET = 11_644_473_600
DB_TEMP_FILE_SUFFIX = ".db"
DEFAULT_DOMAINS = (".twitter.com", ".x.com")
# The logged-in session token. Guest cookies (guest_id, __cf_bm, gt, ...) are
# handed out to anonymous visitors too, so their presence says nothing about
# whether the profile is signed in.
SESSION_COOKIE_NAME = "auth_token"
EXIT_ERROR = 1
DEFAULT_COOKIE_PATH = "/"
MIN_VALID_UNIX_TIMESTAMP = 0
SQL_COOKIES_QUERY = (
    "SELECT name, encrypted_value, value, path, host_key, expires_utc, is_secure FROM cookies"
)


def _get_chrome_keychain_key() -> bytes:
    password = subprocess.check_output(
        KEYCHAIN_COMMAND,
        stderr=subprocess.DEVNULL,
    ).strip()
    return hashlib.pbkdf2_hmac(
        PBKDF2_ALGORITHM, password, PBKDF2_SALT, PBKDF2_ITERATIONS, dklen=KEY_LENGTH
    )


def _decrypt_cookie(encrypted_value: bytes, key: bytes) -> str:
    if not encrypted_value or encrypted_value[:IV_START_OFFSET] != V10_HEADER:
        return encrypted_value.decode("utf-8", errors="replace")

    if not all((Cipher, algorithms, modes)):
        return encrypted_value.decode("utf-8", errors="replace")

    try:
        iv = encrypted_value[IV_START_OFFSET:IV_END_OFFSET]
        ciphertext = encrypted_value[CIPHERTEXT_START_OFFSET:]
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        raw = decryptor.update(ciphertext) + decryptor.finalize()
        pad = raw[-1]
        if pad < 1 or pad > AES_BLOCK_SIZE or len(raw) < PLAINTEXT_START_OFFSET + pad:
            raise ValueError("Invalid PKCS#7 padding length")
        if not all(b == pad for b in raw[-pad:]):
            raise ValueError("Invalid PKCS#7 padding bytes")
        return raw[PLAINTEXT_START_OFFSET:-pad].decode("utf-8", errors="replace")
    except Exception:
        return encrypted_value.decode("utf-8", errors="replace")


def has_session_cookie(cookies: list[dict]) -> bool:
    """True when the cookie set actually authenticates a signed-in account."""
    return any(c.get("name") == SESSION_COOKIE_NAME and c.get("value") for c in cookies)


def get_browser_cookies(
    domains: tuple[str, ...] = DEFAULT_DOMAINS,
) -> tuple[list[dict], str]:
    """Find an x.com session in any installed browser.

    Returns (cookies, source_label). Firefox-family browsers are tried first:
    their cookie DB is unencrypted, needs no keychain prompt, and comes from the
    same engine camoufox runs, so the cookies transplant cleanly.

    Falls back to the first source holding *any* x.com cookies so the caller can
    report what it actually found rather than an empty list.
    """
    guest_only: list[dict] = []
    guest_source = ""

    for db in firefox_profile_dbs():
        cookies = read_firefox_cookies(db, domains)
        source = f"{db.parent.parent.parent.name}:{db.parent.name}"
        if has_session_cookie(cookies):
            return cookies, source
        if cookies and not guest_only:
            guest_only, guest_source = cookies, source

    chromium = get_chrome_cookies(domains)
    if has_session_cookie(chromium):
        return chromium, "Chrome"
    if chromium and not guest_only:
        guest_only, guest_source = chromium, "Chrome"

    return guest_only, guest_source


def _read_profile_cookies(db_path: Path, domains: tuple[str, ...], key: bytes) -> list[dict]:
    """Read and decrypt the matching cookies out of one Chrome profile."""
    tf = tempfile.NamedTemporaryFile(suffix=DB_TEMP_FILE_SUFFIX, delete=False)
    tmp_db = Path(tf.name)
    tf.close()

    try:
        shutil.copy2(db_path, tmp_db)
        placeholders = " OR ".join("host_key LIKE ?" for _ in domains)
        params = [f"%{d}" for d in domains]
        conn = sqlite3.connect(f"file:{tmp_db}?mode=ro", uri=True)
        rows = conn.execute(f"{SQL_COOKIES_QUERY} WHERE {placeholders}", params).fetchall()
        conn.close()
    except Exception:
        return []
    finally:
        tmp_db.unlink(missing_ok=True)

    cookies = []
    for name, enc_val, plain_val, path, host_key, expires_utc, is_secure in rows:
        value = _decrypt_cookie(enc_val, key) if enc_val else plain_val
        if not value:
            continue
        cookie: dict = {
            "name": name,
            "value": value,
            "domain": host_key,
            "path": path or DEFAULT_COOKIE_PATH,
            "secure": bool(is_secure),
        }
        if expires_utc:
            unix_ts = int((expires_utc / FILETIME_TO_UNIX_DIVISOR) - FILETIME_TO_UNIX_OFFSET)
            if unix_ts > MIN_VALID_UNIX_TIMESTAMP:
                cookie["expires"] = unix_ts
        cookies.append(cookie)
    return cookies


def get_chrome_cookies(
    domains: tuple[str, ...] = DEFAULT_DOMAINS,
) -> list[dict]:
    cookie_paths = []
    if CHROME_COOKIES_DB.exists():
        cookie_paths.append(CHROME_COOKIES_DB)
    chrome_dir = CHROME_COOKIES_DB.parent.parent
    if chrome_dir.exists():
        for p in chrome_dir.glob("Profile */Cookies"):
            if p.exists() and p not in cookie_paths:
                cookie_paths.append(p)

    if not cookie_paths:
        msg_err = f"{WARN} Chrome Cookies DB not found at {CHROME_COOKIES_DB} or Profile dirs"
        print(msg_err)
        sys.exit(EXIT_ERROR)

    try:
        key = _get_chrome_keychain_key()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            f"{WARN} Failed to read Chrome keychain key — cookies cannot be decrypted",
            file=sys.stderr,
        )
        return []

    # Return the first profile that is actually signed in. Stopping at the
    # first profile with *any* cookies would hand back a guest-only session
    # from Default and never look at the profile you are logged into.
    guest_only: list[dict] = []
    for db_path in cookie_paths:
        cookies = _read_profile_cookies(db_path, domains, key)
        if has_session_cookie(cookies):
            return cookies
        if cookies and not guest_only:
            guest_only = cookies

    return guest_only
