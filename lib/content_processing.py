"""
Content Processing - Shared utilities for cleaning LLM output.
Removes thinking blocks, stats tokens, and other model artifacts.
"""

import re
from typing import Optional


def remove_thinking_blocks(content: str) -> str:
    """Remove <think>, </think>, and similar model thinking tags."""
    if not content:
        return ""

    # Remove <think>...</think> blocks
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    # Remove Gemma <|channel>thought ... <channel|> loops (matched pairs)
    content = re.sub(r"<\|channel>thought.*?<channel\|>", "", content, flags=re.DOTALL)
    # Remove <channel|>thought\ntext... blocks (unmatched, trailing)
    content = re.sub(r"<channel\|>thought\b[^<]*", "", content, flags=re.DOTALL)
    # Remove any remaining bare <channel|> or <|channel> and everything after
    content = re.sub(r"<\|channel>.*", "", content, flags=re.DOTALL)
    content = re.sub(r"<channel\|>", "", content)

    # Remove other thinking markers: <|xxx|>, etc.
    content = re.sub(r"<\|.*?\|>", "", content)

    # Remove Gemma internal tokens
    content = re.sub(r"<\|channel\|[^|]*\|>", "", content)

    # Remove Qwen 3.6 plaintext thinking: "Here's a thinking process:" or "Thinking Process:"
    for marker in ["Here's a thinking process:", "Thinking Process:", "Here is my thinking process:",
                   "Let me think", "Let me carefully", "Let me analyze"]:
        if marker in content:
            # Try to find explicit output markers after thinking
            output_match = re.search(
                r"(?:Output Generation|Output|Final Answer|Response|Proceeds|I will now generate|I'll now generate|Let's draft|Draft)\s*[\.\:]\s*",
                content, re.IGNORECASE
            )
            if output_match:
                content = content[output_match.end():]
                # Also remove any trailing self-correction/verification blocks
                content = re.sub(r"\n?\*?\[?\(?[Ss]elf-[Cc]orrection.*", "", content, flags=re.DOTALL)
            else:
                # Try to find JSON start after thinking
                json_match = re.search(r'[\[{]', content[content.index(marker):])
                if json_match:
                    content = content[content.index(marker) + json_match.start():]
            break

    # Handle </think> tag without matching </think>
    if "</think>" in content:
        content = content.split("</think>")[-1]
    elif "Think:" in content:
        content = content.split("Think:")[-1]

    # Remove trailing stats like "stats:2114;97.2952" or "stats:1234"
    content = re.sub(r"\n*stats:\d+([;.]\d+)?\s*$", "", content)

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
    # These often appear as lines like: "   Let's pick *Toronto Zoo*? No."
    content = re.sub(
        r"(\s*Let'?s? pick [^\n]+\? No\.){3,}",
        " [reasoning truncated]",
        content,
        flags=re.IGNORECASE,
    )

    # Pattern 2 – Qwen inline reasoning: huge block of prose before a JSON object.
    # If the content has more than 2000 chars before the first { or [,
    # and there is a JSON block separated by a blank line, skip the preamble.
    first_json = min(
        (content.find(c) for c in "[{" if content.find(c) >= 0),
        default=-1,
    )
    if first_json > 2000:
        # Look for a JSON block that starts after a blank line
        blank_json_match = re.search(r"\n\s*\n\s*([\[{])", content)
        if blank_json_match:
            content = content[blank_json_match.start(1):]

    return content.strip()


def remove_stats_tokens(content: str) -> str:
    """Remove model-generated stats tokens and control characters."""
    if not content:
        return ""

    # Remove stats tokens: "stats:123;45.67" anywhere in the text
    content = re.sub(r"stats:\d+;[\d.]+", "", content)
    content = re.sub(r"\d+;[\d.]+$", "", content)
    content = re.sub(r"￾stats:.*$", "", content)

    # Remove unicode replacement character
    content = content.replace("\ufffe", "")

    return content.strip()


def remove_markdown_blocks(content: str) -> str:
    """Remove markdown code block markers (``` ... ```)."""
    if not content:
        return ""

    # Remove markdown code block fences (``` and ``` language identifiers)
    content = re.sub(r"```(?:\w+)?\s*\n?", "", content)
    content = re.sub(r"```\s*", "", content)

    return content.strip()


def extract_content_from_code_blocks(content: str) -> Optional[str]:
    """Extract content from markdown code blocks if present."""
    if not content:
        return None

    # Extract from code blocks: ```language ... ```
    code_blocks = re.findall(r"```(?:\w+)?\s*(.*?)```", content, re.DOTALL)
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
    m = re.match(r"^`([^`]+)`$", stripped)
    if m:
        return m.group(1).strip()
    # Allow leading ** prefix like `** `value``
    m = re.match(r"^\*+\s*`([^`]+)`", stripped)
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
