"""Tests for lib.content_processing."""
import pytest


class TestRemoveThinkingBlocks:
    def test_removes_think_tags(self):
        from lib.content_processing import remove_thinking_blocks
        result = remove_thinking_blocks("prefix <think>inner</think> suffix")
        assert result == "prefix  suffix"

    def test_qwen_marker_with_output_tag(self):
        from lib.content_processing import remove_thinking_blocks
        text = "Here's a thinking process: I should analyze this. Output: The answer is 42."
        result = remove_thinking_blocks(text)
        assert result == "The answer is 42."

    def test_qwen_marker_without_output_tag_finds_json(self):
        from lib.content_processing import remove_thinking_blocks
        text = "Here's a thinking process: Let me think about this. {\"key\": \"value\"}"
        result = remove_thinking_blocks(text)
        assert result == "{\"key\": \"value\"}"

    def test_qwen_marker_truncated_no_crash(self):
        from lib.content_processing import remove_thinking_blocks
        text = "Let me think about this. Output: The truncated part."
        result = remove_thinking_blocks(text)
        assert result == "The truncated part."

    def test_marker_not_present(self):
        from lib.content_processing import remove_thinking_blocks
        text = "Just normal text without any thinking markers."
        result = remove_thinking_blocks(text)
        assert result == text

    def test_gemma_channel_blocks(self):
        from lib.content_processing import remove_thinking_blocks
        text = "hello <|channel>thought inside<channel|> world"
        result = remove_thinking_blocks(text)
        assert result == "hello  world"

    def test_end_think_tag(self):
        from lib.content_processing import remove_thinking_blocks
        text = "keep this</think>discard this"
        result = remove_thinking_blocks(text)
        assert result == "discard this"  # keeps content after </think>

    def test_think_colon(self):
        from lib.content_processing import remove_thinking_blocks
        text = "keep\nThink: discard"
        result = remove_thinking_blocks(text)
        assert result == "discard"  # keeps content after Think:

    def test_remove_thinking_gemma_internal_tokens(self):
        from lib.content_processing import remove_thinking_blocks
        text = "hello <|channel|some_token|> world"
        result = remove_thinking_blocks(text)
        assert result == "hello  world"

    def test_stats_tokens_via_remove_stats_tokens(self):
        from lib.content_processing import remove_stats_tokens
        text = "content\nstats:2114;97.2952"
        result = remove_stats_tokens(text)
        assert result == "content"

    def test_self_correction_in_full_flow(self):
        from lib.content_processing import remove_thinking_blocks
        text = "Here's a thinking process: ... Output: hello [Self-Correction: let me fix that"
        result = remove_thinking_blocks(text)
        assert "hello" in result
        assert "Self-Correction" not in result


class TestRemoveInlineThinking:
    def test_gemma_self_correction_collapsed(self):
        from lib.content_processing import remove_inline_thinking
        text = "Let's pick Zoo? No. Let's pick Park? No. Let's pick Museum? No. final"
        result = remove_inline_thinking(text)
        assert "reasoning truncated" in result
        assert "Zoo" not in result

    def test_qwen_long_preamble_before_json(self):
        from lib.content_processing import remove_inline_thinking
        preamble = "A" * 2500
        text = f"{preamble}\n\n{{\"key\": \"value\"}}"
        result = remove_inline_thinking(text)
        assert result.startswith("{")

    def test_short_preamble_no_truncation(self):
        from lib.content_processing import remove_inline_thinking
        text = "short preamble {\"key\": \"value\"}"
        result = remove_inline_thinking(text)
        assert result == text

    def test_empty_content(self):
        from lib.content_processing import remove_inline_thinking
        assert remove_inline_thinking("") == ""
        assert remove_inline_thinking(None) is None


class TestRemoveStatsTokens:
    def test_trailing_stats_removed(self):
        from lib.content_processing import remove_stats_tokens
        result = remove_stats_tokens("content\nstats:2114;97.2952")
        assert result == "content"

    def test_inline_stats_removed(self):
        from lib.content_processing import remove_stats_tokens
        result = remove_stats_tokens("stats:123;45.67")
        assert result == ""

    def test_unicode_replacement_removed(self):
        from lib.content_processing import remove_stats_tokens
        result = remove_stats_tokens("text\ufffetext")
        assert result == "texttext"

    def test_empty(self):
        from lib.content_processing import remove_stats_tokens
        assert remove_stats_tokens("") == ""


class TestCleanModelOutput:
    def test_full_pipeline(self):
        from lib.content_processing import clean_model_output
        text = "<think>skip</think> body\nstats:100;50"
        result = clean_model_output(text)
        assert result == "body"

    def test_empty(self):
        from lib.content_processing import clean_model_output
        assert clean_model_output("") == ""


class TestExtractContentFromCodeBlocks:
    def test_present(self):
        from lib.content_processing import extract_content_from_code_blocks
        text = "```json\n{\"key\": \"value\"}\n```"
        result = extract_content_from_code_blocks(text)
        assert result == "{\"key\": \"value\"}"

    def test_absent(self):
        from lib.content_processing import extract_content_from_code_blocks
        result = extract_content_from_code_blocks("plain text")
        assert result is None


class TestStripBacktickValue:
    def test_single_backtick(self):
        from lib.content_processing import strip_backtick_value
        assert strip_backtick_value("`filename.txt`") == "filename.txt"

    def test_leading_asterisks(self):
        from lib.content_processing import strip_backtick_value
        assert strip_backtick_value("** `value`") == "value"

    def test_no_match(self):
        from lib.content_processing import strip_backtick_value
        assert strip_backtick_value("plain") == "plain"
