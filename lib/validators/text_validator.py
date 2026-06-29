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
    MIN_SPECIFIC_FILENAME_LEN, FILENAME_LENGTH_SCORE, FILENAME_CHARS_SCORE,
    FILENAME_FORMAT_SCORE, MAX_EXPLANATORY_FILENAME_LEN,
    FILENAME_EXPLANATION_PENALTY_SCORE, FILENAME_SPECIFIC_SCORE,
    STRUCT_HEADERS_BULLETS_SCORE, STRUCT_HEADERS_ONLY_SCORE,
    STRUCT_BULLET_LONG_LEN, STRUCT_BULLET_LONG_SCORE,
    STRUCT_BULLET_SHORT_LEN, STRUCT_BULLET_SHORT_SCORE,
    USERS_COVERAGE_HIGH_COUNT, USERS_COVERAGE_HIGH_SCORE,
    USERS_COVERAGE_MED_COUNT, USERS_COVERAGE_MED_SCORE,
    USERS_COVERAGE_LOW_COUNT, USERS_COVERAGE_LOW_SCORE,
    TIMESTAMP_SPECIFICITY_SCORE, MAX_NARRATIVE_SPECIFICITY_SCORE,
    NARRATIVE_WORD_SCORE_MULTIPLIER, TEMPLATE_DRIVEN_FIELD_LIMIT,
    BOILERPLATE_SYNTHESIS_SCORE, DEFAULT_SYNTHESIS_SCORE,
    SYNTHESIS_MATCH_BONUS, TOPIC_COVERAGE_HIGH_COUNT,
    TOPIC_COVERAGE_HIGH_SCORE, TOPIC_COVERAGE_MED_COUNT,
    TOPIC_COVERAGE_MED_SCORE, TOPIC_TRANSITION_WORD_SCORE,
    FILE_SUMMARY_ITEMS_SCORE, PATH_REALISM_HIGH_RATIO,
    PATH_REALISM_HIGH_SCORE, PATH_REALISM_MED_RATIO,
    PATH_REALISM_MED_SCORE, SPECIFIC_DESC_HIGH_COUNT,
    SPECIFIC_DESC_HIGH_SCORE, SPECIFIC_DESC_MED_COUNT,
    SPECIFIC_DESC_MED_SCORE, SPECIFIC_DESC_LOW_SCORE,
    MIN_SPECIFIC_DESC_LEN,
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

# Pre-compiled validation regular expressions (John Carmack optimization)
USER_MENTIONS_RE = re.compile(r'@?[Uu]ser\s*\d+')
TIMESTAMP_RE = re.compile(r'\d{1,2}:\d{2}')
NARRATIVE_WORDS_RE = re.compile(
    r'\b(?:ask(?:s|ed|ing)?|respond(?:s|ed|ing)?|thank(?:s|ed|ing)?|report(?:s|ed|ing)?|confirm(?:s|ed|ing)?|direct(?:s|ed|ing)?|inquire(?:s|d|ing)?|announce(?:s|d|ing)?|share(?:s|d|ing)?|request(?:s|ed|ing)?|provide(?:s|d|ing)?)\b',
    re.IGNORECASE
)
TEMPLATE_FIELDS_RE = re.compile(r'\*\*(Who|What|When|Where):')
BOILERPLATE_RE = re.compile(r'(not specified|n/a|unknown|not provided)', re.IGNORECASE)
NEWLINE_HEADER_RE = re.compile(r'\n#{2,}\s+\w+')
START_HEADER_RE = re.compile(r'^#{2,}\s+\w+', re.MULTILINE)
SYNTHESIS_RE = re.compile(
    r'(overall|summary|in (short|summary)|key (points?|takeaways?)|tl;dr|(the|this|that) (conversation|discussion|thread|interaction))',
    re.IGNORECASE
)
TOPIC_MARKERS_RE = re.compile(r'^#{2,}\s+\w+|^[A-Z][^a-z]{2,}:\s', re.MULTILINE)
TRANSITION_WORDS_RE = re.compile(
    r'(first|second|third|then|also|additionally|meanwhile)',
    re.IGNORECASE
)


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
    if clean.lower() in GENERIC_FILENAMES or len(clean) < MIN_SPECIFIC_FILENAME_LEN:
        return 0, f"generic: {clean}"

    # Valid length (30 pts)
    if FILENAME_LENGTH_MIN < len(clean) < FILENAME_LENGTH_MAX:
        score += FILENAME_LENGTH_SCORE
    else:
        failures.append(f"length {len(clean)} not in {FILENAME_LENGTH_MIN}-{FILENAME_LENGTH_MAX - 1}")

    # Valid characters (20 pts)
    if all(is_valid_filename_char(c) for c in clean):
        score += FILENAME_CHARS_SCORE
    else:
        failures.append("invalid chars")

    # Format quality - has meaningful structure (25 pts)
    if has_filename_format(clean) or "_" in clean or "-" in clean:
        score += FILENAME_FORMAT_SCORE
    else:
        failures.append("no separators/structure")

    # Non-generic specificity (25 pts)
    # Score outputs based on whether they look like filenames vs prose explanations
    clean_lower = clean.lower()
    has_question_parts = "?" in clean or "please" in clean_lower or "which" in clean_lower or "what" in clean_lower
    has_explanation = len(clean) > MAX_EXPLANATORY_FILENAME_LEN or clean_lower.startswith(("the ", "this ", "a "))
    if has_question_parts:
        failures.append("question-like output")
    elif has_explanation:
        score += FILENAME_EXPLANATION_PENALTY_SCORE  # Some structure but too wordy
        failures.append("wordy")
    else:
        score += FILENAME_SPECIFIC_SCORE

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
        score += STRUCT_HEADERS_BULLETS_SCORE
    elif has_headers:
        score += STRUCT_HEADERS_ONLY_SCORE
    elif has_bullets and len(data_str) >= STRUCT_BULLET_LONG_LEN:
        score += STRUCT_BULLET_LONG_SCORE
    elif len(data_str) >= STRUCT_BULLET_SHORT_LEN:
        score += STRUCT_BULLET_SHORT_SCORE
    else:
        failures.append("no structure")

    # User coverage (15 pts)
    found_users = len(USER_MENTIONS_RE.findall(data_str))
    if found_users >= USERS_COVERAGE_HIGH_COUNT:
        score += USERS_COVERAGE_HIGH_SCORE
    elif found_users == USERS_COVERAGE_MED_COUNT:
        score += USERS_COVERAGE_MED_SCORE
    elif found_users == USERS_COVERAGE_LOW_COUNT:
        score += USERS_COVERAGE_LOW_SCORE
    else:
        failures.append("no user mentions")

    # Event specificity + narrative depth (25 pts)
    has_timestamps = bool(TIMESTAMP_RE.search(data_str))
    narrative_words = len(NARRATIVE_WORDS_RE.findall(data_str))

    specificity_score = 0
    if has_timestamps:
        specificity_score += TIMESTAMP_SPECIFICITY_SCORE
    specificity_score += min(MAX_NARRATIVE_SPECIFICITY_SCORE, narrative_words * NARRATIVE_WORD_SCORE_MULTIPLIER)  # up to 15 for narrative words
    if specificity_score == 0:
        failures.append("no timestamps or narrative words")
    score += specificity_score

    # Synthesis depth (20 pts)
    # Penalize template-driven output (repeated **Who:/What:/When:/Where: fields)
    template_fields = len(TEMPLATE_FIELDS_RE.findall(data_str))
    is_template_driven = template_fields >= TEMPLATE_DRIVEN_FIELD_LIMIT

    # Check for "Not specified" / boilerplate filler
    has_boilerplate = bool(BOILERPLATE_RE.search(data_str))

    # Check for top-level synthesis (content before first ## or ### header)
    top_level = ""
    header_match = NEWLINE_HEADER_RE.search(data_str)
    if header_match:
        first_header = header_match.start()
        if first_header > 0:
            top_level = data_str[:first_header].strip()
    elif not START_HEADER_RE.search(data_str):
        top_level = data_str  # no headers at all

    has_synthesis = bool(SYNTHESIS_RE.search(top_level)) if top_level else False

    if is_template_driven:
        failures.append("template-driven (repeated field structure)")
        synthesis_score = 0
    elif has_boilerplate:
        synthesis_score = BOILERPLATE_SYNTHESIS_SCORE
        failures.append("boilerplate filler")
    else:
        synthesis_score = DEFAULT_SYNTHESIS_SCORE

    if has_synthesis:
        synthesis_score += SYNTHESIS_MATCH_BONUS

    score += synthesis_score

    # Topic coverage (25 pts)
    topic_markers = len(TOPIC_MARKERS_RE.findall(data_str))
    if topic_markers >= TOPIC_COVERAGE_HIGH_COUNT:
        score += TOPIC_COVERAGE_HIGH_SCORE
    elif topic_markers == TOPIC_COVERAGE_MED_COUNT:
        score += TOPIC_COVERAGE_MED_SCORE
    elif bool(TRANSITION_WORDS_RE.search(data_str)):
        score += TOPIC_TRANSITION_WORD_SCORE
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
        score += FILE_SUMMARY_ITEMS_SCORE
    else:
        failures.append(f"only {len(items)} items (need {FILE_SUMMARY_MIN_ITEMS}+)")

    # Path realism - check paths look like real files (30 pts)
    paths = [str(item.get("path", "")) for item in items]
    real_paths = sum(1 for p in paths if "." in p or "/" in p)
    if real_paths >= len(items) * PATH_REALISM_HIGH_RATIO:
        score += PATH_REALISM_HIGH_SCORE
    elif real_paths >= len(items) * PATH_REALISM_MED_RATIO:
        score += PATH_REALISM_MED_SCORE
    else:
        failures.append(f"unrealistic paths ({real_paths}/{len(items)})")

    # Quality descriptions (40 pts)
    descs = [str(item.get("desc", "") or item.get("description", "")) for item in items]
    unique_descs = set(d for d in descs if d)

    # Check descriptions are specific (not generic phrases)
    generic_phrases = ["personal", "document", "system", "user's", "folder"]
    specific = sum(1 for d in descs if not any(g in d.lower() for g in generic_phrases) and len(d) > MIN_SPECIFIC_DESC_LEN)

    quality_score = 0
    if specific >= SPECIFIC_DESC_HIGH_COUNT:
        quality_score = SPECIFIC_DESC_HIGH_SCORE
    elif specific >= SPECIFIC_DESC_MED_COUNT:
        quality_score = SPECIFIC_DESC_MED_SCORE
    elif unique_descs:
        quality_score = SPECIFIC_DESC_LOW_SCORE
    else:
        failures.append("generic descriptions only")

    score += quality_score

    return min(MAX_SCORE, score), "; ".join(failures)