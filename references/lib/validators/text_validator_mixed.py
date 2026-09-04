"""Mixed-signal, faithfulness, schema and leak validators.

Split out of text_validator.py, which had grown past the repo's 500-line limit.
Both halves are re-exported from lib.validators, so importers are unaffected.
"""

# Text Validation functions
# Quality-based scoring - checks output signals, not just format

import json
import re
from typing import Any, List, Tuple

from lib.validators.text_match import identifying_tokens, phrase_overlap, tokenize

# ============================================================
# MIXED-SIGNAL VALIDATORS (signal-from-noise filtering)
# ============================================================

# Both of these are CHOSEN, not measured -- there is no throughput to time here,
# only a judgement about how much rewording still counts as the same statement.
# Each was set by running controls rather than by taste, and each is written to
# fail in its own safe direction.
#
# COVERAGE: half the identifying tokens. Calibrated against four controls, where
# a coverage metric has to separate the middle two:
#     source timeline (upper bound)          100
#     all 18 topics, heavily paraphrased       77
#     4 of 18 topics                           16
#     no facts at all                           0
# Lower thresholds credit a summary for a passing name-drop; higher ones start
# demanding the verbatim copying this replaced. A missed fact under-credits a
# good summary, so the failure lands on the harmless side.
COVERAGE_TOKEN_RATIO = 0.5
# FALSEHOOD: two distinctive tokens, the sensitivity this check already had. A
# missed falsehood scores a hoax-repeating summary as clean, so this one errs
# toward over-detection instead.
FALSEHOOD_TOKEN_HITS = 2


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
        m = re.search(r"\[@([\w]+)\s*\|", line)
        if m:
            senders.append("@" + m.group(1).lower())
    return senders


_COMMON = {
    "about",
    "above",
    "after",
    "again",
    "their",
    "there",
    "these",
    "those",
    "would",
    "could",
    "should",
    "which",
    "while",
    "where",
    "when",
    "what",
    "with",
    "from",
    "this",
    "that",
    "then",
    "than",
    "they",
    "them",
    "have",
    "been",
    "were",
    "will",
    "your",
    "said",
    "says",
    "into",
    "over",
    "also",
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
    toks = {t for t in re.sub(r"[^a-z0-9 ]", " ", entry.lower()).split() if len(t) >= 4}
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
        t
        for t in re.sub(r"[^a-z0-9 ]", " ", signal_part.lower()).split()
        if len(t) >= 5 and t not in _COMMON
    }
    if sig_toks:
        covered = sum(1 for t in sig_toks if t in summary)
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
        m = re.match(r"^\d+\.\s*(.+)$", line)
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
    def norm(s: str) -> list[str]:
        return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()

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
        after = text[max(ends) + 1 :].strip()
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


def validate_factual_accuracy(
    output: str, source_text: str = "", falsehood_phrases: list = None
) -> Tuple[int, str]:
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
            # A COUNT, not a fraction, and deliberately so: a couple of
            # distinctive tokens is enough to show the hoax was repeated, and a
            # fraction rule would make the LONGEST falsehoods the hardest to
            # detect -- the unsafe direction for this check.
            ptoks = set(identifying_tokens(p))
            present = set(tokenize(out))
            if ptoks and sum(1 for t in ptoks if t in present) >= FALSEHOOD_TOKEN_HITS:
                found.append(phrase)
    if not found:
        return 100, ""
    pct = max(0, 100 - int(100 * len(found) / len(falsehood_phrases)))
    return pct, f"parrots {len(found)}/{len(falsehood_phrases)} falsehoods: {found[0]!r}"


def validate_factual_coverage(
    output: str, source_text: str = "", key_facts: list = None
) -> Tuple[int, str]:
    """Fact-coverage scoring: how many of `key_facts` the output actually covers.

    Matched on identifying tokens, NOT as exact substrings, because the task
    prompt orders the model to reword ("use narrative verbs and connecting
    phrases"). Substring matching therefore scored verbatim COPYING and called
    it coverage, and the two controls said so plainly: a summary covering all
    18 topics in its own words scored 5, and the source timeline itself -- the
    trivial upper bound -- managed only 94, because one key fact ('Amazon
    launches drone delivery in Toronto') is not a substring of its own source
    line ('...in select Toronto neighborhoods'). Every model failed this task,
    and none of them was the reason.
    """
    if not output:
        return 0, "empty response"
    if not key_facts:
        return 100, ""
    out = output.lower()
    found = sum(1 for fact in key_facts if phrase_overlap(fact, out) >= COVERAGE_TOKEN_RATIO)
    pct = int(100 * found / len(key_facts))
    failures = []
    if pct < 30:
        failures.append(f"covered {found}/{len(key_facts)} key facts")
    return pct, "; ".join(failures)


def validate_no_contradiction(
    output: str, source_text: str = "", contradiction_phrase: str = ""
) -> Tuple[int, str]:
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
    ptoks = {t for t in re.sub(r"[^a-z0-9 ]", " ", phrase).split() if len(t) >= 4}
    if ptoks and sum(1 for t in ptoks if t in out) >= 2:
        return 0, f"parrots contradiction: {contradiction_phrase!r}"
    return 100, ""
