"""Validators for model evaluation tasks."""

import re
from typing import Tuple, List, Dict, Any
from .content_processing import strip_backtick_value

# Stopwords for source matching
STOPWORDS = {"the", "and", "for", "with", "this", "that", "from", "are", "was", "has", "have", "but", "not", "you", "all", "can", "her", "his", "had", "they", "been", "will", "would", "could", "what", "when", "where", "who", "which", "why", "how"}

# ==========================================================
# SCORING CONSTANTS
# ==========================================================

# JSON Validator scoring weights
JSON_STRUCTURE_WEIGHT = 20      # Points for valid JSON structure
JSON_COUNT_GOOD = 25            # Points for 10+ items
JSON_COUNT_OK = 15              # Points for 5+ items
JSON_VALIDITY_WEIGHT = 30        # Points for valid item content
JSON_VALIDITY_THRESHOLD = 0.7   # % of items must be valid
JSON_SOURCE_WEIGHT = 25         # Points for extracting from INPUT (not hallucinating)

# Detailed JSON validator scoring weights
DETAILED_STRUCTURE_WEIGHT = 15  # Points for valid JSON structure
DETAILED_COUNT_GOOD = 15        # Points for 10+ items
DETAILED_COUNT_OK = 10          # Points for 5+ items
# Points for detail quality (name + location/weather/description)
DETAILED_QUALITY_WEIGHT = 40
DETAIL_REQUIRED_FIELDS = 3      # Score 100% with all items having details
DETAIL_THRESHOLD_HIGH = 0.8     # 80%+ items have details = full*0.8
DETAIL_THRESHOLD_MID = 0.5      # 50%+ items have details = full*0.5
DETAILED_SOURCE_WEIGHT = 30   # Points for extracting from INPUT
# Partial scoring (computed as fraction of weight)
DETAIL_PARTIAL_HIGH = DETAILED_QUALITY_WEIGHT * 8 // 10
DETAIL_PARTIAL_MID = DETAILED_QUALITY_WEIGHT * 5 // 10
DETAIL_PARTIAL_LOW = DETAILED_QUALITY_WEIGHT * 2 // 10

# Summary validator scoring weights
SUMMARY_HEADERS_WEIGHT = 40     # Points for having headers (## or **)
SUMMARY_LENGTH_GOOD = 30        # Points for 500+ chars (40 tweets = more content)
SUMMARY_LENGTH_OK = 15          # Points for 200-499 chars
SUMMARY_CONTENT_WEIGHT = 30     # Points for 5+ content lines (more topics)
SUMMARY_LENGTH_THRESHOLD = 500  # Minimum chars for good score
SUMMARY_LENGTH_THRESHOLD_OK = 200  # Minimum chars for ok score
SUMMARY_LINES_GOOD = 5          # Required content lines for full score
SUMMARY_LINES_OK = 3            # Content lines for partial score

# Filename validator scoring weights
FILENAME_LENGTH_WEIGHT = 40     # Points for valid length
FILENAME_CHARS_WEIGHT = 30      # Points for valid characters
FILENAME_FORMAT_WEIGHT = 30     # Points for good format (separators/extension)
FILENAME_LENGTH_MIN = 4         # Minimum length
FILENAME_LENGTH_MAX = 59        # Maximum length
FILENAME_VALID_CHARS = "_-."    # Additional valid characters beyond alphanumeric

# General scoring
MAX_SCORE = 100                 # Maximum score value
MIN_ITEMS_GOOD = 10             # Good item count
MIN_ITEMS_OK = 5                 # Acceptable item count


# ==========================================================
# HELPER FUNCTIONS (Single Responsibility Principle)
# ==========================================================


def extract_list_from_dict(data: Dict[str, Any], keys: List[str] = None, _depth: int = 0) -> List[Any]:
    """Extract list from nested dict, trying multiple key options recursively.

    Strategy (in priority order):
    1. Check known activity/result keys at current level.
    2. Recurse into any nested dict values up to depth 4.
    3. If no named list found, collect all list values from the dict and
       return the longest one (handles models that use ad-hoc keys).
    """
    if keys is None:
        keys = [
            "activities", "items", "results", "data",
            "fixed_activities", "transient_events",
            "events", "places", "venues", "recommendations",
        ]

    if not isinstance(data, dict) or _depth > 4:
        return []

    # Direct key match at this level
    for key in keys:
        if key in data:
            inner = data[key]
            if isinstance(inner, list):
                return inner

    # Recurse into nested dicts (e.g. {"weekend_summary": {"events": [...]}})
    for value in data.values():
        if isinstance(value, dict):
            found = extract_list_from_dict(value, keys, _depth + 1)
            if found:
                return found

    # Fallback: return the longest list found under any key at this level
    # This handles models that invent arbitrary wrapper keys (Gemma, etc.)
    best: List[Any] = []
    for value in data.values():
        if isinstance(value, list) and len(value) > len(best):
            best = value
    return best


def is_valid_list_item(item: Any, required_fields: List[str] = None) -> bool:
    """Check if an item is valid (string or dict with required fields)."""
    if required_fields is None:
        required_fields = ["name", "activity"]

    if isinstance(item, str):
        return bool(item.strip())
    elif isinstance(item, dict):
        return any(item.get(field) for field in required_fields)
    return False


def has_item_details(item: Dict[str, Any], detail_fields: List[str] = None) -> bool:
    """Check if dict item has required detail fields.

    Uses flexible key matching - accepts alternate key names from different models.
    """
    if detail_fields is None:
        # Main detail fields from our schema + alternates Gemma uses
        detail_fields = [
            # Core identity
            "name", "event", "title", "activity", "place",
            # Location variants
            "location", "venue", "address", "where",
            # Time variants
            "day", "date", "when", "time", "duration",
            # Audience variants
            "target_ages", "age_group", "ages", "audience", "who",
            # Price variants
            "price", "cost", "pricing",
            # Weather variants
            "weather", "type", "indoor_outdoor", "setting",
            # Additional Gemma variants
            "price", "cost", "pricing", "target_ages", "age_group",
        ]

    if not isinstance(item, dict):
        return False

    # Must have any identity field
    name_fields = ["name", "event", "title", "activity", "place"]
    has_name = any(item.get(f) for f in name_fields)
    if not has_name:
        return False

    # Must have at least one detail field (location/weather/price/etc)
    detail_fields = ["location", "venue", "address", "where", "day", "date", "when", "time",
                    "duration", "target_ages", "age_group", "ages", "audience", "price",
                    "cost", "pricing", "weather", "type", "indoor_outdoor", "setting"]
    return any(item.get(f) for f in detail_fields)


def check_source_extraction(items: List[Dict], source_text: str) -> float:
    """Check if items use data from the INPUT source text.

    Returns ratio of items that contain text from source (not hallucinated).
    This is the KEY quality signal - model should extract from provided data, not invent.
    """
    if not items or not source_text:
        return 0.0

    source_lower = source_text.lower()
    source_terms = set(t for t in source_lower.split() if len(t) >= 3 and t not in STOPWORDS)

    if not source_terms:
        return 0.0

    matches = 0
    matched_items = []
    unmatched_items = []

    for item in items:
        if isinstance(item, dict):
            item_text = " ".join(str(v).lower() for v in item.values() if v)
            item_name = item.get("name", item.get("event", item.get("title", "")))
        elif isinstance(item, str):
            item_text = item.lower()
            item_name = item
        else:
            item_text = str(item).lower()
            item_name = item_text

        if not item_text:
            continue

        item_terms = set(t for t in item_text.split() if len(t) >= 3 and t not in STOPWORDS)
        common = item_terms & source_terms

        if len(common) >= 2:
            matches += 1
            matched_items.append(item_name if item_name else "unnamed")
        else:
            unmatched_items.append(item_name if item_name else "unnamed")

    ratio = matches / len(items) if items else 0.0
    return ratio


def get_source_matching_details(items: List[Dict], source_text: str) -> Dict[str, Any]:
    """Return detailed info about source matching for diagnostics."""
    if not items or not source_text:
        return {"matched": [], "unmatched": [], "ratio": 0.0, "source_preview": ""}

    source_lower = source_text.lower()
    source_terms = set(t for t in source_lower.split() if len(t) >= 3 and t not in STOPWORDS)

    matched = []
    unmatched = []

    for item in items:
        if isinstance(item, dict):
            item_text = " ".join(str(v).lower() for v in item.values() if v)
            item_name = item.get("name", item.get("event", item.get("title", "")))
        elif isinstance(item, str):
            item_text = item.lower()
            item_name = item
        else:
            item_text = str(item).lower()
            item_name = item_text

        if not item_text:
            continue

        item_terms = set(t for t in item_text.split() if len(t) >= 3 and t not in STOPWORDS)
        common = item_terms & source_terms

        if len(common) >= 2:
            matched.append({"name": item_name, "matched_terms": list(common)[:5]})
        else:
            unmatched.append({"name": item_name, "terms": list(item_terms)[:5], "in_source": list(common)})

    return {
        "matched": matched,
        "unmatched": unmatched,
        "ratio": len(matched) / len(items) if items else 0.0,
        "source_preview": source_text[:500] if source_text else ""
    }


def has_text_headers(text: str, header_markers: List[str] = None) -> bool:
    """Check if text has proper headers."""
    if header_markers is None:
        header_markers = ["##", "**"]

    return any(marker in text for marker in header_markers)


def count_content_lines(text: str) -> int:
    """Count non-header content lines."""
    lines = text.split("\n")
    return len([l.strip() for l in lines if l.strip() and not l.strip().startswith("#")])


def is_valid_filename_char(char: str) -> bool:
    """Check if character is valid for filename."""
    return char.isalnum() or char in FILENAME_VALID_CHARS


def has_filename_format(filename: str) -> bool:
    """Check if filename has good format (separators or extension)."""
    return any(char in filename for char in ["_", "-", "."])


def validate_json(data: Any, source_text: str = "") -> Tuple[int, str]:
    """Score simple JSON lists based on validity and structure.

    Args:
        data: Parsed JSON data (list or dict)
        source_text: Original input text - used to check if model extracts from input
    """
    if not data:
        return 0, "empty response"

    # Extract list from potentially nested dict
    if isinstance(data, dict):
        data = extract_list_from_dict(data)

    items = data if isinstance(data, list) else []
    if not items:
        return 0, "no items found"

    # Quality-based scoring: check if items have valid structure
    score = 0
    failures = []

    # Valid JSON structure
    score += JSON_STRUCTURE_WEIGHT

    # Proper count of items (3+ is good)
    if len(items) >= MIN_ITEMS_GOOD:
        score += JSON_COUNT_GOOD
    elif len(items) >= MIN_ITEMS_OK:
        score += JSON_COUNT_OK
    else:
        failures.append(f"only {len(items)} items (need {MIN_ITEMS_GOOD}+)")

    # Each item should be valid
    valid_items = sum(1 for item in items if is_valid_list_item(item))

    if valid_items == len(items):
        score += JSON_VALIDITY_WEIGHT
    elif valid_items >= len(items) * JSON_VALIDITY_THRESHOLD:
        score += JSON_COUNT_OK
        failures.append(f"only {valid_items}/{len(items)} items are valid")
    else:
        failures.append(f"only {valid_items}/{len(items)} items are valid")

    # KEY QUALITY SIGNAL: Check if items use input data (not hallucinated)
    if source_text and items:
        source_ratio = check_source_extraction(items, source_text)
        if source_ratio >= 0.8:
            score += JSON_SOURCE_WEIGHT
        elif source_ratio >= 0.5:
            score += JSON_SOURCE_WEIGHT // 2
        elif source_ratio > 0:
            score += JSON_SOURCE_WEIGHT // 4
        else:
            failures.append("not from input (hallucinated)")

    return min(MAX_SCORE, score), "; ".join(failures)


def validate_detailed_json(data: Any, source_text: str = "") -> Tuple[int, str]:
    """Score objects with details based on quality, not count.

    Args:
        data: Parsed JSON data (list or dict)
        source_text: Original input text - used to check if model extracts from input
    """
    if not data:
        return 0, "empty response"

    # Extract list from potentially nested dict
    if isinstance(data, dict):
        data = extract_list_from_dict(data)

    items = data if isinstance(data, list) else []
    if not items:
        return 0, "no items found"

    # Convert string arrays to objects (Gemma returns strings like ["item1", "item2"])
    converted_items = []
    for item in items:
        if isinstance(item, str):
            converted_items.append({"name": item.strip()})
        elif isinstance(item, dict):
            converted_items.append(item)
        else:
            converted_items.append({"name": str(item)})
    items = converted_items

    score = 0
    failures = []

    # Valid JSON structure with items
    score += DETAILED_STRUCTURE_WEIGHT

    # Count: good if 3+ items
    if len(items) >= MIN_ITEMS_GOOD:
        score += DETAILED_COUNT_GOOD
    elif len(items) >= MIN_ITEMS_OK:
        score += DETAILED_COUNT_OK
    else:
        failures.append(f"only {len(items)} items")

    # Check each item quality
    valid_with_details = sum(1 for item in items if has_item_details(item))

    # Award points based on proportion with full details
    if valid_with_details == len(items):
        score += DETAILED_QUALITY_WEIGHT  # Perfect: all items have name AND details
    elif valid_with_details >= len(items) * DETAIL_THRESHOLD_HIGH:
        score += DETAIL_PARTIAL_HIGH
    elif valid_with_details >= len(items) * DETAIL_THRESHOLD_MID:
        score += DETAIL_PARTIAL_MID
    elif valid_with_details > 0:
        score += DETAIL_PARTIAL_LOW
    else:
        failures.append("no items with details")

    # KEY QUALITY SIGNAL: Check if items use input data (not hallucinated)
    source_ratio = 0.0
    if source_text and items:
        source_ratio = check_source_extraction(items, source_text)
        if source_ratio >= 0.8:
            score += DETAILED_SOURCE_WEIGHT
        elif source_ratio >= 0.5:
            score += DETAILED_SOURCE_WEIGHT // 2
        elif source_ratio > 0:
            score += DETAILED_SOURCE_WEIGHT // 4
        else:
            failures.append("not from input (hallucinated)")

    return min(MAX_SCORE, score), "; ".join(failures[:DETAIL_REQUIRED_FIELDS])


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


def _extract_best_filename_candidate(raw: str) -> str:
    """Scan raw model output for the best filename candidate.

    Models like Qwen often embed the actual filename in a reasoning block.
    We look for the first short line that:
    - Is within valid length bounds
    - Contains only valid filename chars (alphanumeric + _ - .)
    - Has at least one separator (underscore, hyphen, or dot)
    """
    import re as _re
    # Try backtick-wrapped first
    m = _re.search(r"`([A-Za-z0-9_\-.]{4,58})`", raw)
    if m:
        return m.group(1)
    # Try each line
    for line in raw.splitlines():
        candidate = line.strip().strip("`*\" '")
        if (FILENAME_LENGTH_MIN < len(candidate) < FILENAME_LENGTH_MAX
                and all(c.isalnum() or c in FILENAME_VALID_CHARS for c in candidate)
                and has_filename_format(candidate)):
            return candidate
    return raw.strip()


def validate_filename(data: Any) -> Tuple[int, str]:
    """Score filenames based on validity and format quality."""
    if not data:
        return 0, "empty response"
    raw = str(data).strip()
    # Strip backtick wrappers (e.g. Qwen outputs `filename.txt`)
    clean = strip_backtick_value(raw)
    # If after stripping it's still too long or has invalid chars, try to
    # extract a better candidate from multi-line reasoning output.
    if (len(clean) >= FILENAME_LENGTH_MAX
            or not all(is_valid_filename_char(c) for c in clean)):
        clean = _extract_best_filename_candidate(raw)

    failures = []
    score = 0

    # Valid length - points
    if FILENAME_LENGTH_MIN < len(clean) < FILENAME_LENGTH_MAX:
        score += FILENAME_LENGTH_WEIGHT
    else:
        failures.append(
            f"length {len(clean)} not in {FILENAME_LENGTH_MIN}-{FILENAME_LENGTH_MAX - 1}")

    # Valid characters only - points
    if all(is_valid_filename_char(c) for c in clean):
        score += FILENAME_CHARS_WEIGHT
    else:
        failures.append("invalid chars")

    # Good format - points
    if has_filename_format(clean):
        score += FILENAME_FORMAT_WEIGHT
    else:
        failures.append("no separators/extension")

    return min(MAX_SCORE, score), "; ".join(failures)


VALIDATORS = {
    "json": validate_json,
    "detailed_json": validate_detailed_json,
    "summarize": validate_summary,
    "filename": validate_filename,
}
