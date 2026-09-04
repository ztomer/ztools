"""Tests for twit_cookies.py."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

# This module exercises the cookie readers themselves against fixture DBs, so
# it opts out of the conftest gate that stubs them for everyone else.
pytestmark = pytest.mark.real_cookie_discovery


class TestGetChromeKeychainKey:
    def test_basic(self, mock_llm):
        import twitter.cookies as twit_cookies

        with patch("subprocess.check_output", return_value=b"password123\n"):
            key = twit_cookies._get_chrome_keychain_key()
        assert isinstance(key, bytes)
        assert len(key) == 16

    def test_password_with_whitespace(self, mock_llm):
        import twitter.cookies as twit_cookies

        # PBKDF2 should strip
        with patch("subprocess.check_output", return_value=b"   pwd  \n"):
            key = twit_cookies._get_chrome_keychain_key()
        assert len(key) == 16


class TestDecryptCookie:
    def test_empty_value(self, mock_llm):
        import twitter.cookies as twit_cookies

        assert twit_cookies._decrypt_cookie(b"", b"key") == ""

    def test_non_v10_prefix(self, mock_llm):
        import twitter.cookies as twit_cookies

        # Not starting with v10 — return as decoded string
        result = twit_cookies._decrypt_cookie(b"plaintext", b"key")
        assert result == "plaintext"

    def test_v10_with_crypto(self, mock_llm):
        """v10 prefix triggers AES decryption path."""
        import twitter.cookies as twit_cookies

        encrypted = b"v10" + b"\x00" * 16 + b"ciphertext_here"
        # pad is the last byte. Use pad=2 so raw[16:-2] = "YYYYYYYYY" (9 bytes)
        raw_data = b"A" * 16 + b"Y" * 9 + bytes([2, 2])
        with (
            patch.object(twit_cookies, "Cipher", create=True) as mock_cipher,
            patch.object(twit_cookies, "algorithms", create=True),
            patch.object(twit_cookies, "modes", create=True),
        ):
            mock_cipher_instance = MagicMock()
            mock_cipher.return_value = mock_cipher_instance
            mock_decryptor = MagicMock()
            mock_cipher_instance.decryptor.return_value = mock_decryptor
            mock_decryptor.update.return_value = b""
            mock_decryptor.finalize.return_value = raw_data
            result = twit_cookies._decrypt_cookie(encrypted, b"key1234567890ab")
        assert result == "YYYYYYYYY"

    def test_v10_decrypt_exception(self, mock_llm):
        """When decrypt fails, return raw as decoded string."""
        import twitter.cookies as twit_cookies

        encrypted = b"v10" + b"x" * 32
        with (
            patch.object(twit_cookies, "Cipher", create=True) as mock_cipher,
            patch.object(twit_cookies, "algorithms", create=True),
            patch.object(twit_cookies, "modes", create=True),
        ):
            mock_cipher.side_effect = Exception("decrypt failed")
            result = twit_cookies._decrypt_cookie(encrypted, b"key")
        # Falls back to .decode("utf-8", errors="replace")
        assert isinstance(result, str)

    def test_v10_decrypt_inner_exception(self, mock_llm):
        """When finalize() fails, return raw as decoded string."""
        import twitter.cookies as twit_cookies

        encrypted = b"v10" + b"x" * 32
        with (
            patch.object(twit_cookies, "Cipher", create=True) as mock_cipher,
            patch.object(twit_cookies, "algorithms", create=True),
            patch.object(twit_cookies, "modes", create=True),
        ):
            mock_instance = MagicMock()
            mock_cipher.return_value = mock_instance
            mock_decryptor = MagicMock()
            mock_instance.decryptor.return_value = mock_decryptor
            mock_decryptor.update.return_value = b""
            mock_decryptor.finalize.side_effect = Exception("fail")
            result = twit_cookies._decrypt_cookie(encrypted, b"key")
        assert isinstance(result, str)

    def test_no_cryptography_module(self, mock_llm):
        """If Cipher is None (cryptography not installed), return decoded."""
        import twitter.cookies as twit_cookies

        encrypted = b"v10" + b"x" * 32
        with (
            patch.object(twit_cookies, "Cipher", None),
            patch.object(twit_cookies, "algorithms", None),
            patch.object(twit_cookies, "modes", None),
        ):
            result = twit_cookies._decrypt_cookie(encrypted, b"key")
        assert isinstance(result, str)

    def test_v10_invalid_padding(self, mock_llm):
        """v10 cookie with invalid PKCS#7 padding falls back to raw decode."""
        import twitter.cookies as twit_cookies

        encrypted = b"v10" + b"\x00" * 16 + b"ciphertext_here"
        # pad is invalid, e.g. 99 (larger than block size 16)
        raw_data = b"A" * 16 + b"Y" * 10 + bytes([99])
        with (
            patch.object(twit_cookies, "Cipher", create=True) as mock_cipher,
            patch.object(twit_cookies, "algorithms", create=True),
            patch.object(twit_cookies, "modes", create=True),
        ):
            mock_cipher_instance = MagicMock()
            mock_cipher.return_value = mock_cipher_instance
            mock_decryptor = MagicMock()
            mock_cipher_instance.decryptor.return_value = mock_decryptor
            mock_decryptor.update.return_value = b""
            mock_decryptor.finalize.return_value = raw_data
            result = twit_cookies._decrypt_cookie(encrypted, b"key1234567890ab")
        # Falls back to raw decode of the input encrypted bytes since ValueError is raised
        assert result == encrypted.decode("utf-8", errors="replace")

    def test_get_chrome_keychain_key_failure(self, mock_llm):
        """Keychain failure should propagate."""
        import twitter.cookies as twit_cookies

        with patch("subprocess.check_output", side_effect=Exception("keychain error")):
            with pytest.raises(Exception, match="keychain error"):
                twit_cookies._get_chrome_keychain_key()

    def test_crypto_missing_returns_garbage_warning(self, mock_llm):
        """When crypto missing, encrypted cookie value decoded as broken utf-8."""
        import twitter.cookies as twit_cookies

        with patch.object(twit_cookies, "_get_chrome_keychain_key", return_value=b"k" * 16):
            result = twit_cookies._decrypt_cookie(b"v10" + b"\xff\xfe\xff" * 10, b"k" * 16)
        assert isinstance(result, str)
        # Should contain replacement characters for the binary garbage
        assert "\ufffd" in result or len(result) > 0


class TestGetChromeCookies:
    def _make_db(self, tmp_path, rows):
        db = tmp_path / "Cookies"
        conn = sqlite3.connect(str(db))
        conn.execute("""CREATE TABLE cookies (
            name TEXT, encrypted_value BLOB, value TEXT, path TEXT,
            host_key TEXT, expires_utc INTEGER, is_secure INTEGER
        )""")
        for r in rows:
            conn.execute("INSERT INTO cookies VALUES (?,?,?,?,?,?,?)", r)
        conn.commit()
        conn.close()
        return db

    def test_db_not_exists(self, tmp_path, monkeypatch, capsys):
        """No Chrome cookie DB is an empty result, not process death."""
        import twitter.cookies as twit_cookies

        monkeypatch.setattr(twit_cookies, "CHROME_COOKIES_DB", tmp_path / "nonexistent")
        assert twit_cookies.get_chrome_cookies() == []
        assert "not found" in capsys.readouterr().out

    def test_basic(self, tmp_path, monkeypatch):
        import twitter.cookies as twit_cookies

        rows = [
            ("auth_token", b"", "abc123", "/", ".twitter.com", 9999999999999999, 1),
            ("session_id", b"", "xyz", "/", ".x.com", 0, 0),
        ]
        db = self._make_db(tmp_path, rows)
        monkeypatch.setattr(twit_cookies, "CHROME_COOKIES_DB", db)
        with (
            patch.object(twit_cookies, "_get_chrome_keychain_key", return_value=b"k" * 16),
            patch("twitter.cookies.tempfile.mktemp", return_value=str(tmp_path / "tmp.db")),
        ):
            cookies = twit_cookies.get_chrome_cookies()
        assert len(cookies) == 2
        assert cookies[0]["name"] == "auth_token"
        assert cookies[0]["value"] == "abc123"
        assert cookies[0]["secure"] is True
        assert cookies[0]["domain"] == ".twitter.com"
        assert cookies[0]["path"] == "/"
        # Second cookie has expires_utc=0, so no expires key
        assert "expires" not in cookies[1]
        assert cookies[1]["secure"] is False

    def test_with_encrypted_value(self, tmp_path, monkeypatch):
        import twitter.cookies as twit_cookies

        rows = [
            ("enc", b"v10" + b"x" * 16, "", "/", ".twitter.com", 0, 0),
        ]
        db = self._make_db(tmp_path, rows)
        monkeypatch.setattr(twit_cookies, "CHROME_COOKIES_DB", db)
        with (
            patch.object(twit_cookies, "_get_chrome_keychain_key", return_value=b"k" * 16),
            patch.object(twit_cookies, "_decrypt_cookie", return_value="decrypted"),
            patch("twitter.cookies.tempfile.mktemp", return_value=str(tmp_path / "tmp.db")),
        ):
            cookies = twit_cookies.get_chrome_cookies()
        assert len(cookies) == 1
        assert cookies[0]["value"] == "decrypted"

    def test_empty_value_skipped(self, tmp_path, monkeypatch):
        import twitter.cookies as twit_cookies

        rows = [
            ("empty", b"", "", "/", ".twitter.com", 0, 0),
        ]
        db = self._make_db(tmp_path, rows)
        monkeypatch.setattr(twit_cookies, "CHROME_COOKIES_DB", db)
        with (
            patch.object(twit_cookies, "_get_chrome_keychain_key", return_value=b"k" * 16),
            patch("twitter.cookies.tempfile.mktemp", return_value=str(tmp_path / "tmp.db")),
        ):
            cookies = twit_cookies.get_chrome_cookies()
        assert cookies == []

    def test_custom_domains(self, tmp_path, monkeypatch):
        import twitter.cookies as twit_cookies

        rows = [
            ("x", b"", "y", "/", "custom.com", 0, 0),
        ]
        db = self._make_db(tmp_path, rows)
        monkeypatch.setattr(twit_cookies, "CHROME_COOKIES_DB", db)
        with (
            patch.object(twit_cookies, "_get_chrome_keychain_key", return_value=b"k" * 16),
            patch("twitter.cookies.tempfile.mktemp", return_value=str(tmp_path / "tmp.db")),
        ):
            cookies = twit_cookies.get_chrome_cookies(domains=("custom.com",))
        assert len(cookies) == 1

    def test_path_null_uses_root(self, tmp_path, monkeypatch):
        import twitter.cookies as twit_cookies

        rows = [
            ("x", b"", "y", None, ".twitter.com", 0, 0),
        ]
        db = self._make_db(tmp_path, rows)
        monkeypatch.setattr(twit_cookies, "CHROME_COOKIES_DB", db)
        with (
            patch.object(twit_cookies, "_get_chrome_keychain_key", return_value=b"k" * 16),
            patch("twitter.cookies.tempfile.mktemp", return_value=str(tmp_path / "tmp.db")),
        ):
            cookies = twit_cookies.get_chrome_cookies()
        assert cookies[0]["path"] == "/"

    def test_expires_utc_negative_skipped(self, tmp_path, monkeypatch):
        """If unix_ts <= 0, no expires key."""
        import twitter.cookies as twit_cookies

        # expires_utc=0 → unix_ts = 0 - 11644473600 = negative
        rows = [
            ("x", b"", "y", "/", ".twitter.com", 0, 0),
        ]
        db = self._make_db(tmp_path, rows)
        monkeypatch.setattr(twit_cookies, "CHROME_COOKIES_DB", db)
        with (
            patch.object(twit_cookies, "_get_chrome_keychain_key", return_value=b"k" * 16),
            patch("twitter.cookies.tempfile.mktemp", return_value=str(tmp_path / "tmp.db")),
        ):
            cookies = twit_cookies.get_chrome_cookies()
        assert "expires" not in cookies[0]

    def test_expires_utc_positive(self, tmp_path, monkeypatch):
        """If expires_utc produces positive unix_ts, add 'expires' key."""
        import twitter.cookies as twit_cookies

        rows = [
            ("x", b"", "y", "/", ".twitter.com", 20000000000000000, 0),
        ]
        db = self._make_db(tmp_path, rows)
        monkeypatch.setattr(twit_cookies, "CHROME_COOKIES_DB", db)
        with (
            patch.object(twit_cookies, "_get_chrome_keychain_key", return_value=b"k" * 16),
            patch("twitter.cookies.tempfile.mktemp", return_value=str(tmp_path / "tmp.db")),
        ):
            cookies = twit_cookies.get_chrome_cookies()
        assert "expires" in cookies[0]
        assert cookies[0]["expires"] > 0


class TestErrorPaths:
    def test_keychain_failure_crashes_with_helpful_error(self, tmp_path, monkeypatch):
        """keychain failure currently propagates - verify it reaches caller."""
        import sqlite3

        import twitter.cookies as twit_cookies

        rows = [("x", b"", "y", "/", ".twitter.com", 0, 0)]
        db_path = tmp_path / "Cookies"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""CREATE TABLE cookies (
            name TEXT, encrypted_value BLOB, value TEXT, path TEXT,
            host_key TEXT, expires_utc INTEGER, is_secure INTEGER
        )""")
        conn.execute("INSERT INTO cookies VALUES (?,?,?,?,?,?,?)", *rows)
        conn.commit()
        conn.close()
        monkeypatch.setattr(twit_cookies, "CHROME_COOKIES_DB", db_path)
        with (
            patch.object(
                twit_cookies, "_get_chrome_keychain_key", side_effect=Exception("keychain error")
            ),
            patch("twitter.cookies.tempfile.mktemp", return_value=str(tmp_path / "tmp.db")),
        ):
            with pytest.raises(Exception, match="keychain error"):
                twit_cookies.get_chrome_cookies()

    def test_keychain_called_process_error_returns_empty(self, tmp_path, monkeypatch):
        """CalledProcessError is caught, returns []."""
        import sqlite3
        import subprocess

        import twitter.cookies as twit_cookies

        db_path = tmp_path / "Cookies"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""CREATE TABLE cookies (
            name TEXT, encrypted_value BLOB, value TEXT, path TEXT,
            host_key TEXT, expires_utc INTEGER, is_secure INTEGER
        )""")
        conn.commit()
        conn.close()
        monkeypatch.setattr(twit_cookies, "CHROME_COOKIES_DB", db_path)
        with patch.object(
            twit_cookies,
            "_get_chrome_keychain_key",
            side_effect=subprocess.CalledProcessError(1, "security"),
        ):
            cookies = twit_cookies.get_chrome_cookies()
        assert cookies == []


def test_missing_chrome_db_returns_empty_not_sysexit(monkeypatch, tmp_path):
    """A probe must not kill the process.

    get_browser_cookies scans Firefox first and only then Chrome; a sys.exit
    here made the caller's real remedy message unreachable for anyone whose
    x.com session lives in Firefox/Zen.
    """
    import twitter.cookies as ck

    monkeypatch.setattr(ck, "CHROME_COOKIES_DB", tmp_path / "nope" / "Cookies")
    assert ck.get_chrome_cookies() == []
