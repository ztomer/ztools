"""Tests for twit_cookies.py."""
import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestGetChromeKeychainKey:
    def test_basic(self, mock_llm):
        import twit_cookies
        with patch("subprocess.check_output", return_value=b"password123\n"):
            key = twit_cookies._get_chrome_keychain_key()
        assert isinstance(key, bytes)
        assert len(key) == 16

    def test_password_with_whitespace(self, mock_llm):
        import twit_cookies
        # PBKDF2 should strip
        with patch("subprocess.check_output", return_value=b"   pwd  \n"):
            key = twit_cookies._get_chrome_keychain_key()
        assert len(key) == 16


class TestDecryptCookie:
    def test_empty_value(self, mock_llm):
        import twit_cookies
        assert twit_cookies._decrypt_cookie(b"", b"key") == ""

    def test_non_v10_prefix(self, mock_llm):
        import twit_cookies
        # Not starting with v10 — return as decoded string
        result = twit_cookies._decrypt_cookie(b"plaintext", b"key")
        assert result == "plaintext"

    def test_v10_with_crypto(self, mock_llm):
        """v10 prefix triggers AES decryption path."""
        import twit_cookies
        encrypted = b"v10" + b"\x00" * 16 + b"ciphertext_here"
        # pad is the last byte. Use pad=2 so raw[16:-2] = "YYYYYYYYY" (8 bytes)
        raw_data = b"A" * 16 + b"Y" * 10 + bytes([2])
        with patch.object(twit_cookies, "Cipher", create=True) as mock_cipher, \
             patch.object(twit_cookies, "algorithms", create=True), \
             patch.object(twit_cookies, "modes", create=True):
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
        import twit_cookies
        encrypted = b"v10" + b"x" * 32
        with patch.object(twit_cookies, "Cipher", create=True) as mock_cipher, \
             patch.object(twit_cookies, "algorithms", create=True), \
             patch.object(twit_cookies, "modes", create=True):
            mock_cipher.side_effect = Exception("decrypt failed")
            result = twit_cookies._decrypt_cookie(encrypted, b"key")
        # Falls back to .decode("utf-8", errors="replace")
        assert isinstance(result, str)

    def test_v10_decrypt_inner_exception(self, mock_llm):
        """When finalize() fails, return raw as decoded string."""
        import twit_cookies
        encrypted = b"v10" + b"x" * 32
        with patch.object(twit_cookies, "Cipher", create=True) as mock_cipher, \
             patch.object(twit_cookies, "algorithms", create=True), \
             patch.object(twit_cookies, "modes", create=True):
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
        import twit_cookies
        encrypted = b"v10" + b"x" * 32
        with patch.object(twit_cookies, "Cipher", None), \
             patch.object(twit_cookies, "algorithms", None), \
             patch.object(twit_cookies, "modes", None):
            result = twit_cookies._decrypt_cookie(encrypted, b"key")
        assert isinstance(result, str)


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

    def test_db_not_exists(self, tmp_path, monkeypatch):
        import twit_cookies
        monkeypatch.setattr(twit_cookies, "CHROME_COOKIES_DB", tmp_path / "nonexistent")
        with patch.object(twit_cookies, "print"), \
             patch.object(twit_cookies, "sys") as mock_sys:
            mock_sys.exit.side_effect = SystemExit(1)
            with pytest.raises(SystemExit) as e:
                twit_cookies.get_chrome_cookies()
        assert e.value.code == 1

    def test_basic(self, tmp_path, monkeypatch):
        import twit_cookies
        rows = [
            ("auth_token", b"", "abc123", "/", ".twitter.com", 9999999999999999, 1),
            ("session_id", b"", "xyz", "/", ".x.com", 0, 0),
        ]
        db = self._make_db(tmp_path, rows)
        monkeypatch.setattr(twit_cookies, "CHROME_COOKIES_DB", db)
        with patch.object(twit_cookies, "_get_chrome_keychain_key", return_value=b"k" * 16), \
             patch("twit_cookies.tempfile.mktemp", return_value=str(tmp_path / "tmp.db")):
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
        import twit_cookies
        rows = [
            ("enc", b"v10" + b"x" * 16, "", "/", ".twitter.com", 0, 0),
        ]
        db = self._make_db(tmp_path, rows)
        monkeypatch.setattr(twit_cookies, "CHROME_COOKIES_DB", db)
        with patch.object(twit_cookies, "_get_chrome_keychain_key", return_value=b"k" * 16), \
             patch.object(twit_cookies, "_decrypt_cookie", return_value="decrypted"), \
             patch("twit_cookies.tempfile.mktemp", return_value=str(tmp_path / "tmp.db")):
            cookies = twit_cookies.get_chrome_cookies()
        assert len(cookies) == 1
        assert cookies[0]["value"] == "decrypted"

    def test_empty_value_skipped(self, tmp_path, monkeypatch):
        import twit_cookies
        rows = [
            ("empty", b"", "", "/", ".twitter.com", 0, 0),
        ]
        db = self._make_db(tmp_path, rows)
        monkeypatch.setattr(twit_cookies, "CHROME_COOKIES_DB", db)
        with patch.object(twit_cookies, "_get_chrome_keychain_key", return_value=b"k" * 16), \
             patch("twit_cookies.tempfile.mktemp", return_value=str(tmp_path / "tmp.db")):
            cookies = twit_cookies.get_chrome_cookies()
        assert cookies == []

    def test_custom_domains(self, tmp_path, monkeypatch):
        import twit_cookies
        rows = [
            ("x", b"", "y", "/", "custom.com", 0, 0),
        ]
        db = self._make_db(tmp_path, rows)
        monkeypatch.setattr(twit_cookies, "CHROME_COOKIES_DB", db)
        with patch.object(twit_cookies, "_get_chrome_keychain_key", return_value=b"k" * 16), \
             patch("twit_cookies.tempfile.mktemp", return_value=str(tmp_path / "tmp.db")):
            cookies = twit_cookies.get_chrome_cookies(domains=("custom.com",))
        assert len(cookies) == 1

    def test_path_null_uses_root(self, tmp_path, monkeypatch):
        import twit_cookies
        rows = [
            ("x", b"", "y", None, ".twitter.com", 0, 0),
        ]
        db = self._make_db(tmp_path, rows)
        monkeypatch.setattr(twit_cookies, "CHROME_COOKIES_DB", db)
        with patch.object(twit_cookies, "_get_chrome_keychain_key", return_value=b"k" * 16), \
             patch("twit_cookies.tempfile.mktemp", return_value=str(tmp_path / "tmp.db")):
            cookies = twit_cookies.get_chrome_cookies()
        assert cookies[0]["path"] == "/"

    def test_expires_utc_negative_skipped(self, tmp_path, monkeypatch):
        """If unix_ts <= 0, no expires key."""
        import twit_cookies
        # expires_utc=0 → unix_ts = 0 - 11644473600 = negative
        rows = [
            ("x", b"", "y", "/", ".twitter.com", 0, 0),
        ]
        db = self._make_db(tmp_path, rows)
        monkeypatch.setattr(twit_cookies, "CHROME_COOKIES_DB", db)
        with patch.object(twit_cookies, "_get_chrome_keychain_key", return_value=b"k" * 16), \
             patch("twit_cookies.tempfile.mktemp", return_value=str(tmp_path / "tmp.db")):
            cookies = twit_cookies.get_chrome_cookies()
        assert "expires" not in cookies[0]

    def test_expires_utc_positive(self, tmp_path, monkeypatch):
        """If expires_utc produces positive unix_ts, add 'expires' key."""
        import twit_cookies
        # Use a value that produces positive unix_ts after the conversion
        # unix_ts = expires_utc/1_000_000 - 11_644_473_600
        # We need > 0: expires_utc/1_000_000 > 11_644_473_600
        # 11_644_473_600 * 1_000_000 = 1.16e16
        rows = [
            ("x", b"", "y", "/", ".twitter.com", 20000000000000000, 0),
        ]
        db = self._make_db(tmp_path, rows)
        monkeypatch.setattr(twit_cookies, "CHROME_COOKIES_DB", db)
        with patch.object(twit_cookies, "_get_chrome_keychain_key", return_value=b"k" * 16), \
             patch("twit_cookies.tempfile.mktemp", return_value=str(tmp_path / "tmp.db")):
            cookies = twit_cookies.get_chrome_cookies()
        assert "expires" in cookies[0]
        assert cookies[0]["expires"] > 0
