"""Tests for the Apple Foundation Models bridge (lib/foundation_lib.py).

All tests mock apple_fm_sdk entirely — the SDK is macOS-only and not
available in CI.
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_fm():
    """Ensure each test starts with a clean fm import state."""
    import lib.foundation_lib as mod
    original_fm = mod.fm
    yield
    mod.fm = original_fm


def _make_mock_fm(available=True, reason="ok", response="hello"):
    """Build a fake apple_fm_sdk module."""
    fm = MagicMock()
    model_instance = MagicMock()
    model_instance.is_available.return_value = (available, reason)
    fm.SystemLanguageModel.return_value = model_instance

    session_instance = MagicMock()


    async def _respond(prompt=""):
        return response

    session_instance.respond = _respond
    fm.LanguageModelSession.return_value = session_instance
    return fm


class TestFoundationAvailable:
    def test_returns_false_when_sdk_is_none(self):
        import lib.foundation_lib as mod
        mod.fm = None
        assert mod.foundation_available() is False

    def test_returns_true_when_model_is_available(self):
        import lib.foundation_lib as mod
        mod.fm = _make_mock_fm(available=True)
        assert mod.foundation_available() is True

    def test_returns_false_when_model_is_not_available(self):
        import lib.foundation_lib as mod
        mod.fm = _make_mock_fm(available=False, reason="no model")
        assert mod.foundation_available() is False

    def test_returns_false_on_exception(self):
        import lib.foundation_lib as mod
        fm = MagicMock()
        fm.SystemLanguageModel.side_effect = RuntimeError("boom")
        mod.fm = fm
        assert mod.foundation_available() is False


class TestCallFoundation:
    def test_returns_none_when_sdk_is_none(self):
        import lib.foundation_lib as mod
        mod.fm = None
        assert mod.call_foundation("sys", "user") is None

    def test_returns_response_on_success(self):
        import lib.foundation_lib as mod
        mod.fm = _make_mock_fm(available=True, response="  result text  ")
        result = mod.call_foundation("system prompt", "user prompt")
        assert result == "result text"

    def test_returns_none_when_model_not_available(self):
        import lib.foundation_lib as mod
        mod.fm = _make_mock_fm(available=False, reason="not ready")
        assert mod.call_foundation("sys", "user") is None

    def test_returns_none_on_exception(self):
        import lib.foundation_lib as mod
        fm = MagicMock()
        fm.SystemLanguageModel.side_effect = RuntimeError("crash")
        mod.fm = fm
        assert mod.call_foundation("sys", "user") is None

    def test_returns_none_when_response_is_not_string(self):
        import lib.foundation_lib as mod
        mod.fm = _make_mock_fm(available=True, response=12345)
        assert mod.call_foundation("sys", "user") is None

    def test_empty_system_prompt(self):
        import lib.foundation_lib as mod
        mod.fm = _make_mock_fm(available=True, response="ok")
        result = mod.call_foundation("", "user prompt")
        assert result == "ok"
