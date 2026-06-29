# Validation helper functions

import re
import json
from typing import List, Tuple, Any

from lib.validators.constants import (
    FILENAME_SEPARATORS, MIN_FILENAME_LINE_LEN, MAX_FILENAME_LINE_LEN,
    DEFAULT_CANDIDATE_FALLBACK_LIMIT, CODE_FENCE_LEN, SPACE_CHAR,
    MIN_KEYS_FOR_DETAILS,
)

# Pre-compiled regexes for validation performance (John Carmack optimization)
TEXT_HEADERS_RE = re.compile(r'^#{2,}\s+\w+', re.MULTILINE)
WHITESPACE_RE = re.compile(r'\s+')
JSON_LIST_RE = re.compile(r'\[[\s\S]*\]')


def has_text_headers(text: str) -> bool:
    """Check if text has ## or ### headers."""
    return bool(TEXT_HEADERS_RE.search(text))


def count_content_lines(text: str) -> int:
    """Count non-empty content lines (excluding headers)."""
    if not text:
        return 0
    lines = [l.strip() for l in text.split('\n')]
    return sum(1 for l in lines if l and not l.startswith('#'))


def is_valid_filename_char(char: str) -> bool:
    """Check if character is valid for filenames."""
    return char.isalnum() or char in '_-.'


def has_filename_format(filename: str) -> bool:
    """Check if filename has good format (separators or extension)."""
    return any(char in filename for char in FILENAME_SEPARATORS)


def _extract_best_filename_candidate(text: str) -> str:
    """Extract best filename candidate from multi-line reasoning."""
    if not text:
        return ""
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines:
        # Skip lines that look like reasoning
        if line.startswith('```') or line.startswith('#'):
            continue
        # Take first valid-looking candidate
        if MIN_FILENAME_LINE_LEN < len(line) < MAX_FILENAME_LINE_LEN:
            return line
    return text.strip()[:DEFAULT_CANDIDATE_FALLBACK_LIMIT] if text.strip() else ""


def strip_backtick_value(value: str) -> str:
    """Strip markdown backtick wrappers from value."""
    if not value:
        return ""
    text = str(value).strip()
    # Remove markdown code blocks
    if text.startswith('```'):
        text = text[CODE_FENCE_LEN:]
        if text.endswith('```'):
            text = text[:-CODE_FENCE_LEN]
    # Remove single backticks
    text = text.strip('`').strip()
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    if not text:
        return ""
    # Replace multiple spaces with single space
    text = WHITESPACE_RE.sub(SPACE_CHAR, text)
    return text.strip()


def extract_json_list(content: str) -> List[dict]:
    """Extract JSON list from content."""
    if not content:
        return []
    
    # Try to find JSON array
    match = JSON_LIST_RE.search(content)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    
    return []


def has_item_details(item: dict) -> bool:
    """Check if item has sufficient details beyond name only."""
    if not isinstance(item, dict):
        return False
    return len(item.keys()) > MIN_KEYS_FOR_DETAILS