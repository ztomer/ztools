# Text Validation functions
# Quality-based scoring - checks output signals, not just format

import re
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

# Generic filenames that indicate the model didn't understand the task
GENERIC_FILENAMES = {
    "filename.txt", "file.txt", "text.txt", "output.txt", "document.txt",
    "note.txt", "image.png", "screenshot.png", "unnamed", "file",
    "filename", "output", "document", "image", "photo", "screenshot",
}


def validate_filename(data: Any, source_text: str = "") -> Tuple[int, str]:
    """Score filenames based on quality and specificity."""
    if not data:
        return 0, "empty response"

    raw = str(data).strip()
    clean = strip_backtick_value(raw)

    if (len(clean) >= FILENAME_LENGTH_MAX
            or not all(is_valid_filename_char(c) for c in clean)):
        clean = _extract_best_filename_candidate(raw)

    failures = []
    score = 0

    # Check for generic/unhelpful output (model didn't understand task)
    if clean.lower() in GENERIC_FILENAMES or len(clean) < 4:
        return 0, f"generic: {clean}"

    # Valid length (30 pts)
    if FILENAME_LENGTH_MIN < len(clean) < FILENAME_LENGTH_MAX:
        score += 30
    else:
        failures.append(f"length {len(clean)} not in {FILENAME_LENGTH_MIN}-{FILENAME_LENGTH_MAX - 1}")

    # Valid characters (20 pts)
    if all(is_valid_filename_char(c) for c in clean):
        score += 20
    else:
        failures.append("invalid chars")

    # Format quality - has meaningful structure (25 pts)
    if has_filename_format(clean) or "_" in clean or "-" in clean:
        score += 25
    else:
        failures.append("no separators/structure")

    # Non-generic specificity (25 pts)
    # Penalize outputs that look like the model asked a question or explained
    clean_lower = clean.lower()
    has_question_parts = "?" in clean or "please" in clean_lower or "which" in clean_lower or "what" in clean_lower
    has_explanation = len(clean) > 70 or clean_lower.startswith(("the ", "this ", "a "))
    if has_question_parts:
        failures.append("question-like output")
    elif has_explanation:
        score += 15  # Some structure but too wordy
        failures.append("wordy")
    else:
        score += 25

    return min(MAX_SCORE, score), "; ".join(failures)


def validate_summary(data: Any) -> Tuple[int, str]:
    """Score summaries based on quality signals: structure, user mentions, content depth."""
    if not data:
        return 0, "empty response"
    if isinstance(data, dict):
        data = str(data)

    data_str = str(data).strip()
    failures = []
    score = 0
    data_lower = data_str.lower()

    # Structure - headers + bullet points (15 pts)
    has_headers = has_text_headers(data_str)
    has_bullets = "•" in data_str or "* " in data_str or "- " in data_str
    if has_headers and has_bullets:
        score += 15
    elif has_headers:
        score += 10
    elif has_bullets and len(data_str) >= 300:
        score += 8
    elif len(data_str) >= 200:
        score += 5
    else:
        failures.append("no structure")

    # User coverage (15 pts)
    found_users = len(re.findall(r'@?[Uu]ser\s*\d+', data_str))
    if found_users >= 3:
        score += 15
    elif found_users == 2:
        score += 10
    elif found_users == 1:
        score += 5
    else:
        failures.append("no user mentions")

    # Event specificity + narrative depth (25 pts)
    has_timestamps = bool(re.search(r'\d{1,2}:\d{2}', data_str))
    narrative_words = len(re.findall(r'(?i)\b(?:ask(?:s|ed|ing)?|respond(?:s|ed|ing)?|thank(?:s|ed|ing)?|report(?:s|ed|ing)?|confirm(?:s|ed|ing)?|direct(?:s|ed|ing)?|inquire(?:s|d|ing)?|announce(?:s|d|ing)?|share(?:s|d|ing)?|request(?:s|ed|ing)?|provide(?:s|d|ing)?)\b', data_str))

    specificity_score = 0
    if has_timestamps:
        specificity_score += 10
    specificity_score += min(15, narrative_words * 5)  # up to 15 for narrative words
    if specificity_score == 0:
        failures.append("no timestamps or narrative words")
    score += specificity_score

    # Synthesis depth (20 pts)
    # Penalize template-driven output (repeated **Who:/What:/When:/Where: fields)
    template_fields = len(re.findall(r'\*\*(Who|What|When|Where):', data_str))
    is_template_driven = template_fields >= 3

    # Check for "Not specified" / boilerplate filler
    has_boilerplate = bool(re.search(r'(?i)(not specified|n/a|unknown|not provided)', data_str))

    # Check for top-level synthesis (content before first ## or ### header)
    top_level = ""
    header_match = re.search(r'\n#{2,}\s+\w+', data_str)
    if header_match:
        first_header = header_match.start()
        if first_header > 0:
            top_level = data_str[:first_header].strip()
    elif not re.search(r'^#{2,}\s+\w+', data_str, re.MULTILINE):
        top_level = data_str  # no headers at all

    has_synthesis = bool(re.search(r'(?i)(overall|summary|in (short|summary)|key (points?|takeaways?)|tl;dr|(the|this|that) (conversation|discussion|thread|interaction))', top_level)) if top_level else False

    if is_template_driven:
        failures.append("template-driven (repeated field structure)")
        synthesis_score = 0
    elif has_boilerplate:
        synthesis_score = 5
        failures.append("boilerplate filler")
    else:
        synthesis_score = 10

    if has_synthesis:
        synthesis_score += 10

    score += synthesis_score

    # Topic coverage (25 pts)
    topic_markers = len(re.findall(r'^#{2,}\s+\w+|^[A-Z][^a-z]{2,}:\s', data_str, re.MULTILINE))
    if topic_markers >= 2:
        score += 25
    elif topic_markers == 1:
        score += 15
    elif bool(re.search(r'(?i)(first|second|third|then|also|additionally|meanwhile)', data_str)):
        score += 10
    else:
        failures.append("no topic structure")

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

    # Item count (30 pts)
    if len(items) >= FILE_SUMMARY_MIN_ITEMS:
        score += 30
    else:
        failures.append(f"only {len(items)} items (need {FILE_SUMMARY_MIN_ITEMS}+)")

    # Path realism - check paths look like real files (30 pts)
    paths = [str(item.get("path", "")) for item in items]
    real_paths = sum(1 for p in paths if "." in p or "/" in p)
    if real_paths >= len(items) * 0.7:
        score += 30
    elif real_paths >= len(items) * 0.4:
        score += 20
    else:
        failures.append(f"unrealistic paths ({real_paths}/{len(items)})")

    # Quality descriptions (40 pts)
    descs = [str(item.get("desc", "") or item.get("description", "")) for item in items]
    unique_descs = set(d for d in descs if d)

    # Check descriptions are specific (not generic phrases)
    generic_phrases = ["personal", "document", "system", "user's", "folder"]
    specific = sum(1 for d in descs if not any(g in d.lower() for g in generic_phrases) and len(d) > 10)

    quality_score = 0
    if specific >= 2:
        quality_score = 40
    elif specific >= 1:
        quality_score = 25
    elif unique_descs:
        quality_score = 15
    else:
        failures.append("generic descriptions only")

    score += quality_score

    return min(MAX_SCORE, score), "; ".join(failures)