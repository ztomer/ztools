# Text Validation functions

from typing import Tuple, Any

from lib.validators.constants import (
    MAX_SCORE, SUMMARY_HEADERS_WEIGHT, SUMMARY_LENGTH_GOOD,
    SUMMARY_LENGTH_OK, SUMMARY_LENGTH_THRESHOLD, SUMMARY_LENGTH_THRESHOLD_OK,
    SUMMARY_CONTENT_WEIGHT, SUMMARY_LINES_GOOD, SUMMARY_LINES_OK,
    FILENAME_LENGTH_MIN, FILENAME_LENGTH_MAX, FILENAME_LENGTH_WEIGHT,
    FILENAME_CHARS_WEIGHT, FILENAME_FORMAT_WEIGHT,
    FILE_SUMMARY_ITEMS_WEIGHT, FILE_SUMMARY_QUALITY_WEIGHT,
    FILE_SUMMARY_MIN_ITEMS,
)

from lib.validators.helpers import (
    has_text_headers, count_content_lines, is_valid_filename_char,
    has_filename_format, strip_backtick_value, _extract_best_filename_candidate,
)


# Filename validation characters
FILENAME_VALID_CHARS = set('_.-')


def validate_filename(data: Any) -> Tuple[int, str]:
    """Score filenames based on validity and format quality."""
    if not data:
        return 0, "empty response"
    
    raw = str(data).strip()
    clean = strip_backtick_value(raw)
    
    # Try to extract better candidate from reasoning output
    if (len(clean) >= FILENAME_LENGTH_MAX
            or not all(is_valid_filename_char(c) for c in clean)):
        clean = _extract_best_filename_candidate(raw)
    
    failures = []
    score = 0
    
    # Valid length
    if FILENAME_LENGTH_MIN < len(clean) < FILENAME_LENGTH_MAX:
        score += FILENAME_LENGTH_WEIGHT
    else:
        failures.append(f"length {len(clean)} not in {FILENAME_LENGTH_MIN}-{FILENAME_LENGTH_MAX - 1}")
    
    # Valid characters
    if all(is_valid_filename_char(c) for c in clean):
        score += FILENAME_CHARS_WEIGHT
    else:
        failures.append("invalid chars")
    
    # Good format
    if has_filename_format(clean):
        score += FILENAME_FORMAT_WEIGHT
    else:
        failures.append("no separators/extension")
    
    return min(MAX_SCORE, score), "; ".join(failures)


def validate_summary(data: Any) -> Tuple[int, str]:
    """Score summaries based on structure and content quality."""
    if not data:
        return 0, "empty response"
    if isinstance(data, dict):
        data = str(data)
    
    data_str = str(data).strip()
    failures = []
    score = 0
    
    # Check for proper headers
    if has_text_headers(data_str):
        score += SUMMARY_HEADERS_WEIGHT
    else:
        failures.append("no headers")
    
    # Check for adequate length
    if len(data_str) >= SUMMARY_LENGTH_THRESHOLD:
        score += SUMMARY_LENGTH_GOOD
    elif len(data_str) >= SUMMARY_LENGTH_THRESHOLD_OK:
        score += SUMMARY_LENGTH_OK
        failures.append(f"short ({len(data_str)} chars)")
    else:
        failures.append(f"too short ({len(data_str)} chars)")
    
    # Check for multiple sections/bullet points
    content_line_count = count_content_lines(data_str)
    if content_line_count >= SUMMARY_LINES_GOOD:
        score += SUMMARY_CONTENT_WEIGHT
    elif content_line_count >= SUMMARY_LINES_OK:
        score += SUMMARY_LENGTH_OK
    else:
        failures.append(f"only {content_line_count} content line(s)")
    
    return min(MAX_SCORE, score), "; ".join(failures)


def validate_file_summary(data: Any, source_text: str = "") -> Tuple[int, str]:
    """Score file summaries based on uniqueness and specificity."""
    if not data:
        return 0, "empty response"
    
    if isinstance(data, dict):
        data = [data]
    
    items = data if isinstance(data, list) else []
    if not items:
        return 0, "no items"
    
    failures = []
    score = 0
    
    # Check item count
    if len(items) >= FILE_SUMMARY_MIN_ITEMS:
        score += FILE_SUMMARY_ITEMS_WEIGHT
    else:
        failures.append(f"only {len(items)} items (need {FILE_SUMMARY_MIN_ITEMS}+)")
    
    # Check for unique, specific descriptions
    descriptions = [item.get("desc", "") or item.get("description", "") for item in items]
    unique_descs = set(d for d in descriptions if d)
    
    if len(unique_descs) >= len(descriptions) * 0.7:
        score += FILE_SUMMARY_QUALITY_WEIGHT
    elif unique_descs:
        score += FILE_SUMMARY_QUALITY_WEIGHT // 2
        failures.append(f"{len(descriptions) - len(unique_descs)} duplicate descriptions")
    else:
        failures.append("no valid descriptions")
    
    return min(MAX_SCORE, score), "; ".join(failures)