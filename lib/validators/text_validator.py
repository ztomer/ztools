# Text Validation functions
# Quality-based scoring - checks output signals, not just format

import re
import json
from typing import List, Tuple, Any

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


from lib.quality_models import GENERIC_FILENAMES

# Filename validation characters
FILENAME_VALID_CHARS = set('_.-')

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

    # Mitigation: reject instruction leakage (nemotron-style "Here is the filename:")
    leaks = detect_instruction_leak(raw)
    if leaks:
        return 0, f"instruction leak: {leaks[0]}"

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


def validate_summary(data: Any, source_text: str = "") -> Tuple[int, str]:
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


# ============================================================
# MIXED-SIGNAL VALIDATORS (signal-from-noise filtering)
# ============================================================

def _split_signal_noise(source_text: str) -> Tuple[str, str]:
    """Split a mixed prompt into (signal_part, noise_part) on the NOISE marker."""
    if "NOISE" not in source_text:
        return source_text, ""
    signal_part, noise_part = source_text.split("NOISE", 1)
    return signal_part, noise_part


def _extract_tweet_senders(text: str) -> List[str]:
    """Extract @Sender handles from tweet lines: [@Sender | time]: ..."""
    senders = []
    for line in text.split("\n"):
        m = re.search(r'\[@([\w]+)\s*\|', line)
        if m:
            senders.append("@" + m.group(1).lower())
    return senders


_COMMON = {
    "about", "above", "after", "again", "their", "there", "these", "those",
    "would", "could", "should", "which", "while", "where", "when", "what",
    "with", "from", "this", "that", "then", "than", "they", "them", "have",
    "been", "were", "will", "your", "said", "says", "into", "over", "also",
}


def _parse_noise_entries(noise_part: str) -> List[str]:
    entries = []
    for line in noise_part.split("\n"):
        line = line.strip()
        if line.startswith("- "):
            content = line[2:].strip()
            text = content.split(":", 1)[-1].strip()
            if text:
                entries.append(text)
    return entries


def _entry_hit(entry: str, text: str) -> bool:
    """True if >=2 distinctive tokens of a noise entry appear in the text (phrase match)."""
    toks = {t for t in re.sub(r'[^a-z0-9 ]', ' ', entry.lower()).split() if len(t) >= 4}
    if not toks:
        return False
    return sum(1 for t in toks if t in text) >= 2


def validate_mixed_summary(data: Any, source_text: str = "") -> Tuple[int, str]:
    """Score tweet-summary filtering by NOISE EXCLUSION.

    Models summarize by content (not by echoing @senders), so we detect contamination
    by phrase-matching the NOISE entries (>=2 of an entry's distinctive tokens present).
    Signal coverage (by distinctive content tokens) is reported and used as a floor so a
    summary unrelated to the timeline cannot score high.
    """
    if not data:
        return 0, "empty response"
    summary = str(data).lower()

    signal_part, noise_part = _split_signal_noise(source_text)
    noise_entries = _parse_noise_entries(noise_part)
    noise_hits = sum(1 for e in noise_entries if _entry_hit(e, summary))
    noise_total = len(noise_entries)

    score = 100
    failures = []
    if noise_total:
        score -= round(100 * noise_hits / noise_total)
        if noise_hits:
            failures.append(f"included {noise_hits}/{noise_total} noise items")

    # Signal coverage: distinctive (>=5 char) content tokens from the real timeline.
    sig_toks = {
        t for t in re.sub(r'[^a-z0-9 ]', ' ', signal_part.lower()).split()
        if len(t) >= 5 and t not in _COMMON
    }
    if sig_toks:
        covered = sum(1 for t in sig_toks if t in summary)
        cov = covered / len(sig_toks)
        failures.append(f"signal coverage {covered}/{len(sig_toks)}")
        if covered == 0:
            score = min(score, 30)  # says nothing about the actual timeline
    return score, "; ".join(failures)


def _extract_file_paths(text: str) -> List[str]:
    paths = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("/") and ("." in line or "/" in line):
            paths.append(line.split()[0])
    return paths


def _file_summary_paths(data: Any) -> List[str]:
    if isinstance(data, str):
        # Markdown form: "## path: summary" or "## path"
        paths = []
        for line in data.split("\n"):
            line = line.strip()
            if line.startswith("##"):
                header = line[2:].strip().rstrip(":").strip()
                if header:
                    paths.append(header)
        return paths
    if isinstance(data, dict):
        # Accept {"path": "desc"} form
        if all(isinstance(v, str) for v in data.values()):
            return [str(k) for k in data.keys()]
        items = [v for v in data.values() if isinstance(v, list)]
        data = items[0] if items else []
    if isinstance(data, list):
        out = []
        for item in data:
            if isinstance(item, dict):
                p = item.get("path") or item.get("file") or ""
                if p:
                    out.append(str(p))
            elif isinstance(item, str):
                out.append(item)
        return out
    return []


def validate_mixed_file_summary(data: Any, source_text: str = "") -> Tuple[int, str]:
    """Score file-summary filtering: real files summarized, noise files excluded."""
    if not data:
        return 0, "empty response"

    # Models may return JSON (per system prompt) or markdown ## headers. Normalize.
    if isinstance(data, str):
        parsed = None
        try:
            parsed = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if parsed is not None:
            data = parsed

    signal_part, noise_part = _split_signal_noise(source_text)
    signal_paths = [p.lower() for p in _extract_file_paths(signal_part)]
    noise_paths = [p.lower() for p in _extract_file_paths(noise_part)]
    output_paths = [p.lower() for p in _file_summary_paths(data)]

    if not output_paths:
        return 0, "no file entries in output"

    # Recall: signal paths covered (by prefix match, since output may abbreviate)
    tp = 0
    for sp in signal_paths:
        if any(sp in op or op in sp for op in output_paths):
            tp += 1
    recall = tp / len(signal_paths) if signal_paths else 1.0

    # Precision: output entries that are NOT noise
    fp = 0
    for op in output_paths:
        if any(np in op or op in np for np in noise_paths):
            fp += 1
    precision = (len(output_paths) - fp) / len(output_paths) if output_paths else 0.0

    score = int(100 * (0.5 * recall + 0.5 * precision))
    failures = []
    if fp > 0:
        failures.append(f"included {fp}/{len(noise_paths)} noise files")
    if signal_paths and tp < len(signal_paths):
        failures.append(f"missed {len(signal_paths) - tp}/{len(signal_paths)} real files")
    return score, "; ".join(failures)


def _extract_mixed_filenames(source_text: str) -> Tuple[List[str], List[str]]:
    """Parse signal snippets (before NOISE) and noise entries (after NOISE)."""
    signal_part, noise_part = _split_signal_noise(source_text)
    signal = []
    for line in signal_part.split("\n"):
        line = line.strip()
        m = re.match(r'^\d+\.\s*(.+)$', line)
        if m:
            signal.append(m.group(1).strip())
    noise = []
    for line in noise_part.split("\n"):
        line = line.strip()
        if line.startswith("- "):
            content = line[2:].strip()
            text = content.split(":", 1)[-1].strip()
            if text:
                noise.append(text)
    return signal, noise


def _name_overlap(a: str, b: str) -> bool:
    norm = lambda s: re.sub(r'[^a-z0-9 ]', ' ', s.lower()).split()
    ta = {t for t in norm(a) if len(t) >= 3}
    tb = {t for t in norm(b) if len(t) >= 3}
    return bool(ta & tb)


def validate_mixed_filename(data: Any, source_text: str = "") -> Tuple[int, str]:
    """Score filename-extraction filtering: signal snippets renamed, noise excluded."""
    if not data:
        return 0, "empty response"

    if isinstance(data, dict):
        data = [data]
    if isinstance(data, list):
        outputs = [str(x) for x in data if x]
    else:
        outputs = [str(data)]

    signal, noise = _extract_mixed_filenames(source_text)
    if not signal:
        return 0, "no signal snippets in source"

    # Recall: each signal snippet produced a filename
    tp = 0
    for sig in signal:
        if any(_name_overlap(sig, out) for out in outputs):
            tp += 1
    recall = tp / len(signal)

    # Precision: no output derived from noise entries
    fp = 0
    for out in outputs:
        if any(_name_overlap(n, out) for n in noise):
            fp += 1
    precision = (len(outputs) - fp) / len(outputs) if outputs else 0.0

    score = int(100 * (0.5 * recall + 0.5 * precision))
    failures = []
    if fp > 0:
        failures.append(f"included {fp}/{len(noise)} noise-derived names")
    if tp < len(signal):
        failures.append(f"missed {len(signal) - tp}/{len(signal)} signal snippets")
    return score, "; ".join(failures)


# ============================================================
# FAITHFULNESS / SCHEMA / LEAK CHECKS (Round 1-2 quality tests)
# ============================================================

_LEAK_PATTERNS = [
    re.compile(r"here\s+is\s+(the\s+)?filename", re.I),
    re.compile(r"here's\s+(the\s+)?filename", re.I),
    re.compile(r"the\s+filename\s+is", re.I),
    re.compile(r"filename\s*:", re.I),
    re.compile(r"here\s+is\s+your\s+(file|output)", re.I),
]


def detect_instruction_leak(text: str) -> List[str]:
    """Detect nemotron-style leakage where the model echoes the instruction
    ('Here is the filename: ...') instead of emitting the bare value."""
    if not text:
        return []
    return [p.search(text).group(0) for p in _LEAK_PATTERNS if p.search(text)]


def validate_no_leak(text: str, source_text: str = "") -> Tuple[int, str]:
    """Score 0 if the output leaks instruction text; 100 if clean."""
    leaks = detect_instruction_leak(text)
    if leaks:
        return 0, f"instruction leak: {leaks[0]}"
    return 100, ""


def validate_strict_schema(raw: str, source_text: str = "", kind: str = "json") -> Tuple[int, str]:
    """Verify the RAW output is exactly the contracted shape — no prose preamble,
    no markdown fences, no trailing commentary. Gates the prose-before-JSON
    failure mode (qwen35b / qwopus)."""
    if not raw or not raw.strip():
        return 0, "empty response"
    text = raw.strip()
    if "```" in text:
        return 0, "contains code fence (not strict schema)"
    if kind == "json":
        brackets = [i for i, c in enumerate(text) if c in "{["]
        ends = [i for i, c in enumerate(text) if c in "}]"]
        if not brackets or not ends or max(ends) < min(brackets):
            return 0, "no JSON object/array found"
        before = text[: min(brackets)].strip()
        after = text[max(ends) + 1:].strip()
        if before or after:
            return 0, f"prose outside JSON (before={before!r}, after={after!r})"
        return 100, ""
    if kind == "filename":
        if detect_instruction_leak(text):
            return 0, "instruction leak"
        if len(text) > 80:
            return 0, f"filename too long ({len(text)} chars)"
        return 100, ""
    # markdown / free-text: accept as-is
    return 100, ""


def validate_factual_accuracy(output: str, source_text: str = "",
                              falsehood_phrases: list = None) -> Tuple[int, str]:
    """Multi-falsehood faithfulness probe.
    Checks the output against MULTIPLE planted falsehoods.
    Scores 100 - (33 * parroted_count) for 3 falsehoods.
    `falsehood_phrases` is a list of sentinel phrases; each one found in
    the output counts as a failure."""
    if not output:
        return 100, ""
    if not falsehood_phrases:
        return 100, ""
    out = output.lower()
    found = []
    for phrase in falsehood_phrases:
        p = phrase.lower().strip()
        if p and p in out:
            found.append(phrase)
        elif p:
            ptoks = {t for t in re.sub(r'[^a-z0-9 ]', ' ', p).split() if len(t) >= 4}
            if ptoks and sum(1 for t in ptoks if t in out) >= 2:
                found.append(phrase)
    if not found:
        return 100, ""
    pct = max(0, 100 - int(100 * len(found) / len(falsehood_phrases)))
    return pct, f"parrots {len(found)}/{len(falsehood_phrases)} falsehoods: {found[0]!r}"


def validate_factual_coverage(output: str, source_text: str = "",
                              key_facts: list = None) -> Tuple[int, str]:
    """Fact-coverage scoring.
    Checks how many of the given `key_facts` (case-insensitive substrings) are
    present in the output. High coverage = thorough summarization."""
    if not output:
        return 0, "empty response"
    if not key_facts:
        return 100, ""
    out = output.lower()
    found = sum(1 for fact in key_facts if fact.lower() in out)
    pct = int(100 * found / len(key_facts))
    failures = []
    if pct < 30:
        failures.append(f"covered {found}/{len(key_facts)} key facts")
    return pct, "; ".join(failures)


def validate_no_contradiction(output: str, source_text: str = "",
                              contradiction_phrase: str = "") -> Tuple[int, str]:
    """Faithfulness probe: assert the model does NOT parrot a planted falsehood.
    `contradiction_phrase` is the sentinel fact injected into the source; if it
    (or its distinctive tokens) appears in the output, the model failed to filter
    it out."""
    if not output or not contradiction_phrase:
        return 100, ""
    out = output.lower()
    phrase = contradiction_phrase.lower().strip()
    if phrase and phrase in out:
        return 0, f"parrots contradiction: {contradiction_phrase!r}"
    ptoks = {t for t in re.sub(r'[^a-z0-9 ]', ' ', phrase).split() if len(t) >= 4}
    if ptoks and sum(1 for t in ptoks if t in out) >= 2:
        return 0, f"parrots contradiction: {contradiction_phrase!r}"
    return 100, ""