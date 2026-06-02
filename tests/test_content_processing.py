"""Tests for content processing - thinking removal."""
import pytest
from lib.content_processing import (
    remove_thinking_blocks,
    remove_inline_thinking,
    remove_stats_tokens,
    remove_markdown_blocks,
    extract_content_from_code_blocks,
    strip_backtick_value,
    clean_model_output,
)


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

    def test_gemma_channel_thought_paired(self):
        """Remove Gemma's matched channel thought patterns."""
        text = "before <|channel>thought internal stuff<channel|> after"
        result = remove_thinking_blocks(text)
        assert "internal stuff" not in result
        assert "before" in result
        assert "after" in result

    def test_gemma_channel_unmatched(self):
        """Remove Gemma's unmatched channel| thought pattern."""
        text = "before <channel|>thought trailing text without close"
        result = remove_thinking_blocks(text)
        assert "trailing text" not in result
        assert "before" in result

    def test_gemma_bare_channel(self):
        """Remove Gemma's bare <|channel> with everything after."""
        text = "before <|channel>leaked internal token"
        result = remove_thinking_blocks(text)
        assert "leaked" not in result

    def test_gemma_internal_tokens(self):
        """Remove Gemma's internal channel tokens."""
        text = "before <|channel|internal|> after"
        result = remove_thinking_blocks(text)
        assert "internal" not in result
        assert "before" in result

    def test_other_channel_tokens(self):
        """Remove other <|xxx|> channel tokens."""
        text = "before <|something|> after"
        result = remove_thinking_blocks(text)
        assert "<|something|>" not in result

    def test_unmatched_close_think(self):
        """Handle </think> without matching <think>."""
        text = "before </think> after the think close"
        result = remove_thinking_blocks(text)
        assert "after" in result

    def test_think_marker(self):
        """Handle 'Think:' marker."""
        text = "garbage Think: actual content"
        result = remove_thinking_blocks(text)
        assert "actual content" in result

    def test_qwen_plaintext_with_json(self):
        """Qwen thinking with JSON output - JSON match path."""
        text = """Here's a thinking process: lots of reasoning
[{"name": "x"}]"""
        result = remove_thinking_blocks(text)
        assert "name" in result

    def test_let_me_think(self):
        """Handle 'Let me think' thinking markers."""
        text = """Let me think
This is reasoning
Output Generation.
Final answer"""
        result = remove_thinking_blocks(text)
        assert "Final answer" in result

    def test_let_me_carefully(self):
        """Handle 'Let me carefully' marker."""
        text = """Let me carefully analyze this
Final Answer: stuff
## Result"""
        result = remove_thinking_blocks(text)
        assert "## Result" in result

    def test_let_me_analyze(self):
        """Handle 'Let me analyze' marker."""
        text = """Let me analyze step by step
Response: 
## Final"""
        result = remove_thinking_blocks(text)
        assert "## Final" in result

    def test_thinking_process(self):
        """Handle 'Thinking Process:' marker."""
        text = """Thinking Process:
1. step
2. step
Draft:
## Final"""
        result = remove_thinking_blocks(text)
        assert "## Final" in result

    def test_here_is_my_thinking_process(self):
        """Handle 'Here is my thinking process:' marker."""
        text = """Here is my thinking process:
1. step
Output:
## Final"""
        result = remove_thinking_blocks(text)
        assert "## Final" in result

    def test_stats_with_decimal(self):
        """Remove stats with decimal like 'stats:1234;56'."""
        text = "content\nstats:1234;56"
        result = remove_thinking_blocks(text)
        # The function removes the trailing stats
        assert "content" in result
        assert "1234" not in result

    def test_stats_simple(self):
        """Remove stats with simple digits."""
        text = "content\nstats:1234"
        result = remove_thinking_blocks(text)
        assert "content" in result
        assert "stats" not in result


class TestStripThinking:
    """Test strip_thinking alias function."""

    def test_strip_thinking_alias(self):
        """Test that strip_thinking is aliased correctly."""
        from lib.osaurus_lib import strip_thinking

        text = "Think: Output: Real content"
        result = strip_thinking(text)
        assert "Real content" in result


class TestRemoveInlineThinking:
    def test_empty(self):
        assert remove_inline_thinking("") == ""

    def test_none(self):
        assert remove_inline_thinking(None) is None

    def test_no_inline(self):
        text = "just regular content"
        assert remove_inline_thinking(text) == text

    def test_gemma_self_correction_loops(self):
        text = "Let's pick Toronto? No. Let's pick Vaughan? No. Let's pick Markham? No."
        result = remove_inline_thinking(text)
        # Should collapse to a shorter string
        assert "truncated" in result.lower() or len(result) < len(text)

    def test_gemma_pick_with_asterisks(self):
        text = "Let's pick *Toronto Zoo*? No. Let's pick *Vaughan Mills*? No. Let's pick *Markham*? No."
        result = remove_inline_thinking(text)
        assert "truncated" in result.lower() or len(result) < len(text)

    def test_qwen_inline_with_json(self):
        # Long preamble followed by JSON
        preamble = "a" * 2500
        text = f"{preamble}\n\n{{\"key\": \"value\"}}"
        result = remove_inline_thinking(text)
        # Should strip the preamble
        assert preamble not in result
        assert "key" in result

    def test_qwen_inline_with_array(self):
        preamble = "b" * 2500
        text = f"{preamble}\n\n[1, 2, 3]"
        result = remove_inline_thinking(text)
        assert "1, 2, 3" in result

    def test_short_preamble_kept(self):
        text = "short\n\n[1,2,3]"
        result = remove_inline_thinking(text)
        # Preamble is short, kept
        assert "short" in result


class TestRemoveStatsTokens:
    def test_empty(self):
        assert remove_stats_tokens("") == ""

    def test_none(self):
        assert remove_stats_tokens(None) == ""

    def test_inline_stats(self):
        text = "content stats:1234;56.78 more content"
        result = remove_stats_tokens(text)
        assert "stats:" not in result

    def test_trailing_number_with_dot(self):
        text = "content 1234;56.78"
        result = remove_stats_tokens(text)
        # Trailing number with dot removed
        assert "1234" not in result

    def test_unicode_replacement(self):
        text = "content \ufffe more"
        result = remove_stats_tokens(text)
        assert "\ufffe" not in result


class TestRemoveMarkdownBlocks:
    def test_empty(self):
        assert remove_markdown_blocks("") == ""

    def test_none(self):
        assert remove_markdown_blocks(None) == ""

    def test_with_fences(self):
        text = "before ```python\ncode\n``` after"
        result = remove_markdown_blocks(text)
        assert "before" in result
        assert "after" in result
        assert "```" not in result

    def test_with_lang_no_newline(self):
        text = "before ```python code ``` after"
        result = remove_markdown_blocks(text)
        assert "```" not in result


class TestExtractContentFromCodeBlocks:
    def test_empty(self):
        assert extract_content_from_code_blocks("") is None

    def test_none(self):
        assert extract_content_from_code_blocks(None) is None

    def test_no_code_blocks(self):
        assert extract_content_from_code_blocks("just text") is None

    def test_with_code_block(self):
        text = "before ```python\ncode here\n``` after"
        result = extract_content_from_code_blocks(text)
        assert result == "code here"

    def test_multiple_blocks_returns_last(self):
        text = "```\nfirst\n``` middle ```\nlast\n```"
        result = extract_content_from_code_blocks(text)
        assert result == "last"

    def test_with_lang(self):
        text = "```python\nprint('hi')\n```"
        result = extract_content_from_code_blocks(text)
        assert "print('hi')" in result


class TestStripBacktickValue:
    def test_empty(self):
        assert strip_backtick_value("") == ""

    def test_none(self):
        assert strip_backtick_value(None) is None

    def test_no_backticks(self):
        text = "no backticks"
        assert strip_backtick_value(text) == text

    def test_single_token(self):
        text = "`my_file`"
        assert strip_backtick_value(text) == "my_file"

    def test_with_bold(self):
        text = "** `value`"
        assert strip_backtick_value(text) == "value"

    def test_with_multiple_asterisks(self):
        text = "*** `value`"
        assert strip_backtick_value(text) == "value"

    def test_inline_backticks_unchanged(self):
        text = "before `x` after"
        result = strip_backtick_value(text)
        # Multiple backticks or non-isolated - return as is
        assert result == text


class TestCleanModelOutput:
    def test_empty(self):
        assert clean_model_output("") == ""

    def test_none(self):
        assert clean_model_output(None) == ""

    def test_combined_cleanup(self):
        text = "<think>reasoning</think> ```python\ncode\n``` content"
        result = clean_model_output(text)
        assert "<think>" not in result
        assert "```" not in result

    def test_stripped(self):
        text = "   content   "
        result = clean_model_output(text)
        assert result == "content"

