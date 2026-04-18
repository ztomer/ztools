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

    # Remove other thinking markers: <|xxx|>, etc.
    content = re.sub(r"<\|.*?\|>", "", content)

    # Remove Gemma internal tokens
    content = re.sub(r"<\|channel\|[^|]*\|>", "", content)

    # Remove Qwen "Thinking Process:" sections
    if "Thinking Process:" in content:
        output_match = re.search(
            r"(?:Output|Final Answer|Response):\s*", content, re.IGNORECASE
        )
        if output_match:
            content = content[output_match.end():]
        else:
            # Try to find JSON start if no explicit output marker
            first_brace = content.find("{")
            if first_brace >= 0:
                content = content[first_brace:]

    # Handle </think> tag
    if "</think>" in content:
        content = content.split("</think>")[-1]
    elif "Think:" in content:
        content = content.split("Think:")[-1]

    return content.strip()


def remove_stats_tokens(content: str) -> str:
    """Remove model-generated stats tokens and control characters."""
    if not content:
        return ""

    # Remove stats tokens: "stats:123;45.67"
    content = re.sub(r"^.+?stats:\d+;[\d.]+", "", content)
    content = re.sub(r"^.+?\d+;[\d.]+", "", content)

    # Remove unicode replacement character
    content = content.replace("\ufffe", "")

    # Remove stats token at end
    stats_match = re.search(r"\d+;\d+\.\d+$", content)
    if stats_match:
        content = content[: stats_match.start()].strip()

    return content.strip()


def remove_markdown_blocks(content: str) -> str:
    """Remove markdown code block markers (``` ... ```)."""
    if not content:
        return ""

    # Remove markdown code blocks completely: ```json ... ```
    while "```" in content:
        start = content.find("```")
        end = content.find("```", start + 3)
        if end < 0:
            content = content[:start]
        else:
            content = content[:start] + content[end + 3:]

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


def clean_model_output(content: str) -> str:
    """
    Comprehensive cleanup: remove thinking, stats, markdown.

    Cleans all model artifacts in order:
    1. Remove thinking blocks
    2. Remove stats tokens
    3. Remove markdown wrappers

    Args:
        content: Raw model output

    Returns:
        Cleaned text ready for parsing
    """
    if not content:
        return ""

    content = remove_thinking_blocks(content)
    content = remove_stats_tokens(content)
    content = remove_markdown_blocks(content)

    return content.strip()
