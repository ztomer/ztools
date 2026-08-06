"""
Content Processing - Shared utilities for cleaning LLM output.
Removes thinking blocks, stats tokens, and other model artifacts.
"""

import re
from typing import Optional

# Pre-compiled regular expressions for performance (John Carmack optimization)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
GEMMA_THOUGHT_MATCH_RE = re.compile(r"<\|channel>thought.*?<channel\|>", re.DOTALL)
GEMMA_THOUGHT_UNMATCH_RE = re.compile(r"<channel\|>thought\b[^<]*", re.DOTALL)
GEMMA_CHANNEL_RE = re.compile(r"<\|channel>.*", re.DOTALL)
CHANNEL_RE = re.compile(r"<channel\|>")
THINKING_MARKER_RE = re.compile(r"<\|.*?\|>")
GEMMA_INTERNAL_RE = re.compile(r"<\|channel\|[^|]*\|>")
OUTPUT_MARKER_RE = re.compile(
    r"(?:Output Generation|Output|Final Answer|Response|Proceeds|"
    r"I will now generate|I'll now generate|Let's draft|Draft)"
    r"\s*[\.\:]\s*",
    re.IGNORECASE,
)
SELF_CORRECTION_RE = re.compile(r"\n?\*?\[?\(?[Ss]elf-[Cc]orrection.*", re.DOTALL)
JSON_START_RE = re.compile(r"[\[{]")
TRAILING_STATS_RE = re.compile(r"\n*stats:\d+([;.]\d+)?\s*$")

# Content cleanup limits and indicators (Mitchell Hashimoto design)
JSON_SEARCH_PREVIEW_LIMIT = 2000
UNICODE_REPLACEMENT_CHAR = "\ufffe"
THINKING_END_TAG = "</think>"
THINKING_PREFIX_MARKER = "Think:"

QWEN_THINKING_MARKERS = [
    "Here's a thinking process:",
    "Thinking Process:",
    "Here is my thinking process:",
    "Let me think",
    "Let me carefully",
    "Let me analyze",
]

GEMMA_CORRECTION_LOOP_RE = re.compile(r"(\s*Let'?s? pick [^\n]+\? No\.){3,}", re.IGNORECASE)
BLANK_JSON_START_RE = re.compile(r"\n\s*\n\s*([\[{])")

STATS1_RE = re.compile(r"stats:\d+;[\d.]+")
STATS2_RE = re.compile(r"\d+;[\d.]+$")
STATS3_RE = re.compile(r"stats:.*$")

MD_BLOCK1_RE = re.compile(r"```(?:\w+)?\s*\n?")
MD_BLOCK2_RE = re.compile(r"```\s*")

CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\s*(.*?)```", re.DOTALL)

BACKTICK1_RE = re.compile(r"^`([^`]+)`$")
BACKTICK2_RE = re.compile(r"^\*+\s*`([^`]+)`")


def remove_thinking_blocks(content: str) -> str:
    """Remove <think>, </think>, and similar model thinking tags."""
    if not content:
        return ""

    # Remove <think>...</think> blocks
    content = THINK_RE.sub("", content)

    # Remove Gemma <|channel>thought ... <channel|> loops (matched pairs)
    content = GEMMA_THOUGHT_MATCH_RE.sub("", content)
    # Remove <channel|>thought\ntext... blocks (unmatched, trailing)
    content = GEMMA_THOUGHT_UNMATCH_RE.sub("", content)
    # Remove any remaining bare <channel|> or <|channel> and everything after
    content = GEMMA_CHANNEL_RE.sub("", content)
    content = CHANNEL_RE.sub("", content)

    # Remove other thinking markers: <|xxx|>, etc.
    content = THINKING_MARKER_RE.sub("", content)

    # Remove Gemma internal tokens
    content = GEMMA_INTERNAL_RE.sub("", content)

    # Remove Qwen 3.6 plaintext thinking: "Here's a thinking process:" or "Thinking Process:"
    for marker in QWEN_THINKING_MARKERS:
        marker_idx = content.find(marker)
        if marker_idx < 0:
            continue
        # Try to find explicit output markers after thinking
        output_match = OUTPUT_MARKER_RE.search(content)
        if output_match:
            content = content[output_match.end() :]
            content = SELF_CORRECTION_RE.sub("", content)
        else:
            json_match = JSON_START_RE.search(content[marker_idx:])
            if json_match:
                content = content[marker_idx + json_match.start() :]
        break

    # Handle </think> tag without matching </think>
    if THINKING_END_TAG in content:
        content = content.split(THINKING_END_TAG)[-1]
    elif THINKING_PREFIX_MARKER in content:
        content = content.split(THINKING_PREFIX_MARKER)[-1]

    # Remove trailing stats like "stats:2114;97.2952" or "stats:1234"
    content = TRAILING_STATS_RE.sub("", content)

    return content.strip()


def remove_inline_thinking(content: str) -> str:
    """Remove verbose inline chain-of-thought reasoning that precedes a JSON/plain answer.

    Handles two patterns seen in Qwen and Gemma models:
    1. Qwen3.6: Long prose reasoning before the first JSON bracket.
       Pattern: text... then '\n\n{' or '\n\n[' (a blank-line-separated JSON block).
    2. Gemma self-correction loops: "Let's pick X? No." repeated many times.
    """
    if not content:
        return content

    # Pattern 1 – Gemma self-correction loops: collapse "Let's pick X? No." chains
    content = GEMMA_CORRECTION_LOOP_RE.sub(" [reasoning truncated]", content)

    # Pattern 2 – Qwen inline reasoning: huge block of prose before a JSON object.
    first_json = min(
        (content.find(c) for c in "[{" if content.find(c) >= 0),
        default=-1,
    )
    if first_json > JSON_SEARCH_PREVIEW_LIMIT:
        # Look for a JSON block that starts after a blank line
        blank_json_match = BLANK_JSON_START_RE.search(content)
        if blank_json_match:
            content = content[blank_json_match.start(1) :]

    return content.strip()


def remove_stats_tokens(content: str) -> str:
    """Remove model-generated stats tokens and control characters."""
    if not content:
        return ""

    # Remove stats tokens: "stats:123;45.67" anywhere in the text
    content = STATS1_RE.sub("", content)
    content = STATS2_RE.sub("", content)
    content = STATS3_RE.sub("", content)

    # Remove unicode replacement character
    content = content.replace(UNICODE_REPLACEMENT_CHAR, "")

    return content.strip()


def remove_markdown_blocks(content: str) -> str:
    """Remove markdown code block markers (``` ... ```)."""
    if not content:
        return ""

    # Remove markdown code block fences (``` and ``` language identifiers)
    content = MD_BLOCK1_RE.sub("", content)
    content = MD_BLOCK2_RE.sub("", content)

    return content.strip()


def extract_content_from_code_blocks(content: str) -> Optional[str]:
    """Extract content from markdown code blocks if present."""
    if not content:
        return None

    # Extract from code blocks: ```language ... ```
    code_blocks = CODE_BLOCK_RE.findall(content)
    if code_blocks:
        return code_blocks[-1].strip()

    return None


def strip_backtick_value(content: str) -> str:
    """Extract value wrapped in backticks, e.g. `some_file.txt` -> some_file.txt.

    Used for filename tasks where models like Qwen wrap the answer in backticks.
    Only strips when the ENTIRE content (after trimming) is a single backtick-wrapped token.
    """
    if not content:
        return content
    stripped = content.strip()
    # Single `token` pattern
    m = BACKTICK1_RE.match(stripped)
    if m:
        return m.group(1).strip()
    # Allow leading ** prefix like `** `value``
    m = BACKTICK2_RE.match(stripped)
    if m:
        return m.group(1).strip()
    return content


def clean_model_output(content: str) -> str:
    """
    Comprehensive cleanup: remove thinking, stats, markdown.

    Cleans all model artifacts in order:
    1. Remove thinking blocks (explicit tags)
    2. Remove inline chain-of-thought reasoning (Qwen/Gemma verbose patterns)
    3. Remove stats tokens
    4. Remove markdown wrappers

    Args:
        content: Raw model output

    Returns:
        Cleaned text ready for parsing
    """
    if not content:
        return ""

    content = remove_thinking_blocks(content)
    content = remove_inline_thinking(content)
    content = remove_stats_tokens(content)
    content = remove_markdown_blocks(content)

    return content.strip()
