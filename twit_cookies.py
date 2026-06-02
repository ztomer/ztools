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

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
except ImportError:
    Cipher = algorithms = modes = None

CHROME_COOKIES_DB = (
    Path.home()
    / "Library"
    / "Application Support"
    / "Google"
    / "Chrome"
    / "Default"
    / "Cookies"
)


def _get_chrome_keychain_key() -> bytes:
    password = subprocess.check_output(
        ["security", "find-generic-password", "-w", "-s", "Chrome Safe Storage"],
        stderr=subprocess.DEVNULL,
    ).strip()
    return hashlib.pbkdf2_hmac("sha1", password, b"saltysalt", 1003, dklen=16)


def _decrypt_cookie(encrypted_value: bytes, key: bytes) -> str:
    if not encrypted_value or encrypted_value[:3] != b"v10":
        return encrypted_value.decode("utf-8", errors="replace")

    if not all((Cipher, algorithms, modes)):
        return encrypted_value.decode("utf-8", errors="replace")

    try:
        iv = encrypted_value[3:19]
        ciphertext = encrypted_value[19:]
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        raw = decryptor.update(ciphertext) + decryptor.finalize()
        pad = raw[-1]
        return raw[16:-pad].decode("utf-8", errors="replace")
    except Exception:
        return encrypted_value.decode("utf-8", errors="replace")


def get_chrome_cookies(
    domains: tuple[str, ...] = (".twitter.com", ".x.com"),
) -> list[dict]:
    if not CHROME_COOKIES_DB.exists():
        print(f"{WARN} Chrome Cookies DB not found at {CHROME_COOKIES_DB}")
        sys.exit(1)

    tmp_db = Path(tempfile.mktemp(suffix=".db"))
    shutil.copy2(CHROME_COOKIES_DB, tmp_db)

    try:
        key = _get_chrome_keychain_key()
        domain_clauses = " OR ".join(f"host_key LIKE '%{d}'" for d in domains)
        conn = sqlite3.connect(f"file:{tmp_db}?mode=ro", uri=True)
        rows = conn.execute(
            f"SELECT name, encrypted_value, value, path, host_key, expires_utc, is_secure "
            f"FROM cookies WHERE {domain_clauses}"
        ).fetchall()
        conn.close()
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
            "path": path or "/",
            "secure": bool(is_secure),
        }
        if expires_utc:
            unix_ts = int((expires_utc / 1_000_000) - 11_644_473_600)
            if unix_ts > 0:
                cookie["expires"] = unix_ts
        cookies.append(cookie)
    return cookies
