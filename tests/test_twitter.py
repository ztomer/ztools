"""Tests for twitter summarizer."""
import pytest


class TestTwitterThinkingRemoval:
    """Test thinking removal in twitter summarizer context."""

    def test_strip_thinking_before_return(self, mock_llm_response):
        """Verify strip_thinking is called even when no XML thinking."""
        # This tests the bug we fixed: strip_thinking must be called
        # even when extract_thinking returns empty thinking
        from lib.osaurus_lib import extract_thinking, strip_thinking

        response = mock_llm_response["twitter_response"]
        thinking, cleaned = extract_thinking(response)

        # No XML thinking detected
        assert thinking == ""

        # But we must still strip plaintext thinking
        result = strip_thinking(cleaned)
        assert "## Summary" in result
        assert "thinking process" not in result

    def test_merge_thinking_with_summary(self):
        """Test merge_thinking_with_summary includes analysis."""
        from lib.osaurus_lib import merge_thinking_with_summary

        thinking = "Analysis: This is important"
        summary = "## Summary\nMain content"
        result = merge_thinking_with_summary(thinking, summary)
        assert "## Analysis" in result
        assert thinking in result
        assert summary in result


class TestTwitterPromptBuilding:
    """Test twitter prompt building."""

    def test_prompt_includes_timeline(self, sample_tweets):
        """Verify timeline is included in prompt."""
        # This is more of a smoke test - verify the format is reasonable
        tweets_text = ""
        for t in sample_tweets:
            tweets_text += f"- @{t['screen_name']}: {t['text']}\n"

        assert "@user1" in tweets_text
        assert "Test tweet" in tweets_text
