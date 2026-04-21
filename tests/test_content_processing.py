"""Tests for content processing - thinking removal."""
import pytest
from lib.content_processing import remove_thinking_blocks, remove_inline_thinking


class TestThinkingRemoval:
    """Test cases for thinking removal."""

    def test_remove_xml_thinking(self):
        """Remove <think> XML thinking blocks."""
        text = "<think> Think about this Output: actual content"
        result = remove_thinking_blocks(text)
        assert "actual content" in result
        assert "</think>" not in result

    def test_remove_qwen_plaintext_thinking(self):
        """Remove qwen plaintext thinking."""
        text = """Here's a thinking process:
1. Analyze
2. Decide

Output Generation.
## Real Summary
Content here"""
        result = remove_thinking_blocks(text)
        assert "## Real Summary" in result
        assert "thinking process" not in result

    def test_remove_draft_marker(self):
        """Remove content before Draft: marker."""
        text = """Draft:
## Summary
Real content"""
        result = remove_thinking_blocks(text)
        assert "## Summary" in result

    def test_remove_stats_tokens(self):
        """Remove trailing stats tokens."""
        text = """## Summary
Content here
stats:1234"""
        result = remove_thinking_blocks(text)
        assert "stats:" not in result
        assert "Content here" in result

    def test_remove_self_correction(self):
        """Remove self-correction blocks after recognized markers."""
        # Self-correction is only removed after recognized thinking markers
        text = """Here's a thinking process:
Draft:
## Summary
Content here
*(Self-Correction during draft)*"""
        result = remove_thinking_blocks(text)
        assert "Self-Correction" not in result
        assert "Content here" in result

    def test_empty_input(self):
        """Handle empty input."""
        result = remove_thinking_blocks("")
        assert result == ""

    def test_no_thinking(self):
        """Handle text without thinking."""
        text = "## Summary\nJust regular content"
        result = remove_thinking_blocks(text)
        assert result == text


class TestStripThinking:
    """Test strip_thinking alias function."""

    def test_strip_thinking_alias(self):
        """Test that strip_thinking is aliased correctly."""
        from lib.osaurus_lib import strip_thinking

        text = "Think: Output: Real content"
        result = strip_thinking(text)
        assert "Real content" in result
