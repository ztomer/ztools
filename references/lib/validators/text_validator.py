# Text Validation functions
# Quality-based scoring - checks output signals, not just format

import re
from typing import Any, Tuple

from lib.quality_models import GENERIC_FILENAMES
from lib.validators.attribution import attribution_faithfulness
from lib.validators.constants import (
    BOILERPLATE_SYNTHESIS_SCORE,
    DEFAULT_SYNTHESIS_SCORE,
    FILE_SUMMARY_ITEMS_SCORE,
    FILE_SUMMARY_MIN_ITEMS,
    FILENAME_CHARS_SCORE,
    FILENAME_EXPLANATION_PENALTY_SCORE,
    FILENAME_FORMAT_SCORE,
    FILENAME_LENGTH_MAX,
    FILENAME_LENGTH_MIN,
    FILENAME_LENGTH_SCORE,
    FILENAME_SPECIFIC_SCORE,
    MAX_EXPLANATORY_FILENAME_LEN,
    MAX_NARRATIVE_SPECIFICITY_SCORE,
    MAX_SCORE,
    MIN_SPECIFIC_DESC_LEN,
    MIN_SPECIFIC_FILENAME_LEN,
    NARRATIVE_WORD_SCORE_MULTIPLIER,
    PATH_REALISM_HIGH_RATIO,
    PATH_REALISM_HIGH_SCORE,
    PATH_REALISM_MED_RATIO,
    PATH_REALISM_MED_SCORE,
    SPECIFIC_DESC_HIGH_COUNT,
    SPECIFIC_DESC_HIGH_SCORE,
    SPECIFIC_DESC_LOW_SCORE,
    SPECIFIC_DESC_MED_COUNT,
    SPECIFIC_DESC_MED_SCORE,
    STRUCT_BULLET_LONG_LEN,
    STRUCT_BULLET_LONG_SCORE,
    STRUCT_BULLET_SHORT_LEN,
    STRUCT_BULLET_SHORT_SCORE,
    STRUCT_HEADERS_BULLETS_SCORE,
    STRUCT_HEADERS_ONLY_SCORE,
    SYNTHESIS_MATCH_BONUS,
    TEMPLATE_DRIVEN_FIELD_LIMIT,
    TIMESTAMP_SPECIFICITY_SCORE,
    TOPIC_COVERAGE_HIGH_COUNT,
    TOPIC_COVERAGE_HIGH_SCORE,
    TOPIC_COVERAGE_MED_COUNT,
    TOPIC_COVERAGE_MED_SCORE,
    TOPIC_TRANSITION_WORD_SCORE,
    USERS_COVERAGE_HIGH_COUNT,
    USERS_COVERAGE_HIGH_SCORE,
    USERS_COVERAGE_LOW_COUNT,
    USERS_COVERAGE_LOW_SCORE,
    USERS_COVERAGE_MED_COUNT,
    USERS_COVERAGE_MED_SCORE,
)
from lib.validators.helpers import (
    _extract_best_filename_candidate,
    has_filename_format,
    has_text_headers,
    is_valid_filename_char,
    strip_backtick_value,
)

# Re-exported from the other half of this module (split for the 500-line limit),
# so `from lib.validators.text_validator import validate_mixed_*` keeps working.
from lib.validators.text_validator_mixed import (  # noqa: F401,E402
    _entry_hit,
    _extract_file_paths,
    _extract_mixed_filenames,
    _extract_tweet_senders,
    _file_summary_paths,
    _name_overlap,
    _parse_noise_entries,
    _split_signal_noise,
    detect_instruction_leak,
    validate_factual_accuracy,
    validate_factual_coverage,
    validate_mixed_file_summary,
    validate_mixed_filename,
    validate_mixed_summary,
    validate_no_contradiction,
    validate_no_leak,
    validate_strict_schema,
)

# Filename validation characters
FILENAME_VALID_CHARS = set("_.-")

# Pre-compiled validation regular expressions (John Carmack optimization)
# The prompts order every bullet to end with `(@handle | timestamp)`, and the
# real timelines carry real handles (@TechCrunch). Counting only `user N` tokens
# scored a prompt-perfect summary as "no user mentions" while `@user 1 @user 2`
# padding scored full marks. Count both shapes, de-duplicated, so coverage means
# "distinct people referenced" rather than "how many times the token appears".
# The `(?<![\w.])` guard keeps email and domain tokens out of the count:
# "contact support@example.com" is not three users.
USER_HANDLE_RE = re.compile(r"(?<![\w.])@([A-Za-z][A-Za-z0-9_]{1,})")
LEGACY_USER_RE = re.compile(r"\b[Uu]ser\s*(\d+)\b")
TIMESTAMP_RE = re.compile(r"\d{1,2}:\d{2}")

# Format placeholders reproduced verbatim instead of filled in. Foundation once
# ended all 31 bullets with the literal `(@TechCrunch | Mon DD HH:MM)` and still
# scored 90 "ok" with an empty failure reason: nothing looked for the template.
PLACEHOLDER_LEAK_MAX_SCORE = 40
PLACEHOLDER_RE = re.compile(
    r"Mon DD|DD HH|HH:MM|@username|@handle|<handle>|YYYY-MM-DD|\{\w+\}"
)
# A bullet that actually attributes: `... (@handle | 08:15)` or `(@handle | Aug 10 14:30)`.
ATTRIBUTED_BULLET_RE = re.compile(r"\(@([A-Za-z][\w]*)\s*\|\s*([^)]+)\)\s*$")
NARRATIVE_WORDS_RE = re.compile(
    r"\b(?:ask(?:s|ed|ing)?|respond(?:s|ed|ing)?|thank(?:s|ed|ing)?|report(?:s|ed|ing)?|confirm(?:s|ed|ing)?|direct(?:s|ed|ing)?|inquire(?:s|d|ing)?|announce(?:s|d|ing)?|share(?:s|d|ing)?|request(?:s|ed|ing)?|provide(?:s|d|ing)?)\b",
    re.IGNORECASE,
)
TEMPLATE_FIELDS_RE = re.compile(r"\*\*(Who|What|When|Where):")
BOILERPLATE_RE = re.compile(r"(not specified|n/a|unknown|not provided)", re.IGNORECASE)
NEWLINE_HEADER_RE = re.compile(r"\n#{2,}\s+\w+")
START_HEADER_RE = re.compile(r"^#{2,}\s+\w+", re.MULTILINE)
SYNTHESIS_RE = re.compile(
    r"(overall|summary|in (short|summary)|key (points?|takeaways?)|"
    r"tl;dr|(the|this|that) (conversation|discussion|thread|interaction))",
    re.IGNORECASE,
)
LEADING_SUMMARY_SECTION_RE = re.compile(
    r"^#{2,}\s+(?:executive\s+summary|summary|overview|tl;?dr)\s*\n(?P<body>.*?)(?=\n#{2,}\s|\Z)",
    re.IGNORECASE | re.DOTALL,
)
TOPIC_MARKERS_RE = re.compile(r"^#{2,}\s+\w+|^[A-Z][^a-z]{2,}:\s", re.MULTILINE)
TRANSITION_WORDS_RE = re.compile(
    r"(first|second|third|then|also|additionally|meanwhile)", re.IGNORECASE
)


FILENAME_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
        "for", "and", "or", "with", "that", "this", "it", "its", "please",
        "try", "again", "showing", "show",
    }
)
# A name that shares no content word with the input identifies the wrong thing,
# however well-formed it is. Below this coverage the shape score cannot rescue it.
FILENAME_RELEVANCE_FLOOR = 0.2
FILENAME_RELEVANCE_GOOD = 0.6
FILENAME_IRRELEVANT_MAX_SCORE = 40


def filename_relevance(name: str, source_text: str) -> float:
    """Fraction of the input's content words that appear in the filename.

    The eval used to send RENAME_PROMPT with `{text}` unfilled and score the
    result shape-only, so `summary_request` — a name for nothing — earned
    100/100 and decided `best_models.filename`.
    """
    if not source_text:
        return -1.0
    words = {w for w in re.findall(r"[a-z0-9]{3,}", source_text.lower())}
    words -= FILENAME_STOPWORDS
    if not words:
        return -1.0
    name_tokens = set(re.findall(r"[a-z0-9]+", name.lower()))
    hits = sum(1 for w in words if w in name_tokens or any(w in t for t in name_tokens))
    return hits / len(words)


def validate_filename(data: Any, source_text: str = "") -> Tuple[int, str]:
    """Score filenames based on quality and specificity."""
    if not data:
        return 0, "empty response"

    raw = str(data).strip()

    # Mitigation: reject instruction leakage (nemotron-style "Here is the filename:")
    leaks = detect_instruction_leak(raw)
    if leaks:
        return 0, f"instruction leak: {leaks[0]}"

    clean = strip_backtick_value(raw)

    if len(clean) >= FILENAME_LENGTH_MAX or not all(is_valid_filename_char(c) for c in clean):
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
        failures.append(
            f"length {len(clean)} not in {FILENAME_LENGTH_MIN}-{FILENAME_LENGTH_MAX - 1}"
        )

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
    has_question_parts = (
        "?" in clean or "please" in clean_lower or "which" in clean_lower or "what" in clean_lower
    )
    has_explanation = len(clean) > MAX_EXPLANATORY_FILENAME_LEN or clean_lower.startswith(
        ("the ", "this ", "a ")
    )
    if has_question_parts:
        failures.append("question-like output")
    elif has_explanation:
        score += FILENAME_EXPLANATION_PENALTY_SCORE  # Some structure but too wordy
        failures.append("wordy")
    else:
        score += FILENAME_SPECIFIC_SCORE

    coverage = filename_relevance(clean, source_text)
    if coverage >= 0.0:
        if coverage < FILENAME_RELEVANCE_FLOOR:
            failures.append(f"unrelated to input (coverage {coverage:.0%})")
            return min(FILENAME_IRRELEVANT_MAX_SCORE, score), "; ".join(failures)
        if coverage < FILENAME_RELEVANCE_GOOD:
            failures.append(f"weak input coverage ({coverage:.0%})")
            score -= 15

    return min(MAX_SCORE, max(0, score)), "; ".join(failures)



def grounded_attribution_ratio(text: str, source_text: str) -> tuple[int, int]:
    """(grounded, total) over bullets that end with an `(@handle | stamp)` tag.

    A bullet is grounded when BOTH its handle and its timestamp appear in the
    source. That is what makes the prompt's CRITICAL instruction measurable:
    a copied stamp is grounded, an invented weekday is not, and a leaked
    placeholder is not.
    """
    grounded = total = 0
    for line in text.splitlines():
        line = line.rstrip()
        if not line.lstrip().startswith(("-", "*")):
            continue
        m = ATTRIBUTED_BULLET_RE.search(line)
        if not m:
            continue
        total += 1
        handle, stamp = m.group(1), m.group(2).strip()
        if source_text and handle.lower() in source_text.lower() and stamp in source_text:
            grounded += 1
    return grounded, total


def count_distinct_users(text: str) -> int:
    """Distinct people referenced, counting real @handles and legacy `user N`."""
    handles = {h.lower() for h in USER_HANDLE_RE.findall(text)}
    handles.discard("user")  # "@user 1" is the legacy token, not a handle
    legacy = {f"user{n}" for n in LEGACY_USER_RE.findall(text)}
    return len(handles | legacy)


# A summary with any misattributed bullet cannot score above this, however well
# formed it is. Sits just ABOVE PLACEHOLDER_LEAK_MAX_SCORE deliberately: a
# template leak ("Mon DD at HH:MM") is visibly unusable and the reader discards
# it, whereas a misattributed bullet is plausible and gets believed. Both are
# far below passing, and keeping them distinct preserves the ordering the suite
# already asserts -- collapsing them to one number would trade this blindness
# for another.
MISATTRIBUTION_MAX_SCORE = 45


def validate_summary(data: Any, source_text: str = "") -> Tuple[int, str]:
    """Score summaries based on quality signals: structure, user mentions, content depth."""
    if not data:
        return 0, "empty response"
    if isinstance(data, dict):
        data = str(data)

    data_str = str(data).strip()
    failures = []
    score = 0

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
    found_users = count_distinct_users(data_str)
    if found_users >= USERS_COVERAGE_HIGH_COUNT:
        score += USERS_COVERAGE_HIGH_SCORE
    elif found_users == USERS_COVERAGE_MED_COUNT:
        score += USERS_COVERAGE_MED_SCORE
    elif found_users == USERS_COVERAGE_LOW_COUNT:
        score += USERS_COVERAGE_LOW_SCORE
    else:
        failures.append("no user mentions")

    # Event specificity + narrative depth (25 pts)
    # `TIMESTAMP_RE.search` was satisfied by the literal string "HH:MM", so a
    # summary with no real attribution at all collected the full timestamp
    # points. Score the fraction of bullets whose (handle, stamp) actually
    # appears in the source instead; fall back to the sniff with no source.
    narrative_words = len(NARRATIVE_WORDS_RE.findall(data_str))

    specificity_score = 0
    # Faithfulness, not mere presence. grounded_attribution_ratio asked whether
    # the handle and the stamp each appeared SOMEWHERE in the source, which a
    # summary with every quote moved to the wrong author satisfies completely --
    # it scored 65, exactly the same as the correct summary. Requiring the PAIR
    # to head one real source line, and the claim to come from that same line,
    # is what makes misattribution visible.
    faithful, total_bullets, attribution_reasons = attribution_faithfulness(
        data_str, source_text
    )
    if source_text and total_bullets:
        ratio = faithful / total_bullets
        if ratio >= 0.8:
            specificity_score += TIMESTAMP_SPECIFICITY_SCORE
        elif ratio >= 0.5:
            specificity_score += TIMESTAMP_SPECIFICITY_SCORE // 2
        elif ratio == 0:
            failures.append(f"no faithful attribution (0/{total_bullets} bullets)")
        if attribution_reasons:
            failures.extend(attribution_reasons[:3])
    elif TIMESTAMP_RE.search(data_str):
        specificity_score += TIMESTAMP_SPECIFICITY_SCORE

    specificity_score += min(
        MAX_NARRATIVE_SPECIFICITY_SCORE, narrative_words * NARRATIVE_WORD_SCORE_MULTIPLIER
    )  # up to 15 for narrative words
    if specificity_score == 0:
        failures.append("no timestamps or narrative words")
    score += specificity_score

    # Synthesis depth (20 pts)
    # Penalize template-driven output (repeated **Who:/What:/When:/Where: fields)
    template_fields = len(TEMPLATE_FIELDS_RE.findall(data_str))
    is_template_driven = template_fields >= TEMPLATE_DRIVEN_FIELD_LIMIT

    # Check for "Not specified" / boilerplate filler
    has_boilerplate = bool(BOILERPLATE_RE.search(data_str))

    # Check for top-level synthesis: prose before the first ## header, or — since
    # the prompts order "Start with a ## Executive Summary paragraph" — the body
    # of a leading summary section. Requiring prose ABOVE the first header made
    # the bonus unreachable for output that followed the prompt exactly.
    top_level = ""
    header_match = NEWLINE_HEADER_RE.search(data_str)
    if header_match:
        first_header = header_match.start()
        if first_header > 0:
            top_level = data_str[:first_header].strip()
    elif not START_HEADER_RE.search(data_str):
        top_level = data_str  # no headers at all
    if not top_level:
        lead = LEADING_SUMMARY_SECTION_RE.match(data_str.lstrip())
        if lead:
            top_level = lead.group("body").strip()

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

    leaks = PLACEHOLDER_RE.findall(data_str)
    if leaks:
        # Unfilled template text means the output did not do the task, whatever
        # else it got right. Cap hard rather than deducting a few points.
        failures.append(f"placeholder leak ({len(leaks)} occurrences)")
        return min(PLACEHOLDER_LEAK_MAX_SCORE, score), "; ".join(failures)

    # Misattribution is disqualifying, not a deduction. A summary that tells the
    # user the wrong person said a thing is worse than one that omits it: the
    # reader has no way to spot the error, and acting on it means repeating a
    # false claim about a real person. Everything else here -- structure, user
    # coverage, depth -- describes a summary that is at least true.
    if source_text and total_bullets and faithful < total_bullets:
        score = min(score, MISATTRIBUTION_MAX_SCORE)

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

    # Guard against list-of-strings responses — convert to dicts
    # (models sometimes return ["path1.py", "path2.py"] instead of [{...}, ...])
    items = [item if isinstance(item, dict) else {"path": str(item), "desc": ""} for item in items]

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
    specific = sum(
        1
        for d in descs
        if not any(g in d.lower() for g in generic_phrases) and len(d) > MIN_SPECIFIC_DESC_LEN
    )

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
