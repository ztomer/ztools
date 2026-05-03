# Validation helper functions

import re
import json
from typing import List, Tuple, Any


def has_text_headers(text: str) -> bool:
    """Check if text has ## headers."""
    return bool(re.search(r'^##\s+\w+', text, re.MULTILINE))


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
    return any(char in filename for char in ["_", "-", "."])


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
        if 3 < len(line) < 60:
            return line
    return text.strip()[:50] if text.strip() else ""


def strip_backtick_value(value: str) -> str:
    """Strip markdown backtick wrappers from value."""
    if not value:
        return ""
    text = str(value).strip()
    # Remove markdown code blocks
    if text.startswith('```'):
        text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
    # Remove single backticks
    text = text.strip('`').strip()
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    if not text:
        return ""
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_json_list(content: str) -> List[dict]:
    """Extract JSON list from content."""
    if not content:
        return []
    
    # Try to find JSON array
    match = re.search(r'\[[\s\S]*\]', content)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    
    return []


def has_item_details(item: dict) -> bool:
    """Check if item has sufficient details beyond name only."""
    if not isinstance(item, dict):
        return False
    # Has any field beyond just name
    return len(item.keys()) > 1
    
    # Try to find JSON array
    match = re.search(r'\[[\s\S]*\]', content)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    
    return []