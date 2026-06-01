"""
Dimension-based quality scoring for model evaluation.

Each task has 3-5 named quality dimensions, each scored 0-100 independently.
Composite score = weighted average of dimensions.
Scorers are calibrated against human-judged reference outputs.
"""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Callable

from lib.config import get_model_prompt, Task, _safe_format_prompt
from lib.osaurus_lib import call as llm_call


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Score:
    name: str
    score: float
    weight: float
    failures: List[str] = field(default_factory=list)

    @property
    def weighted(self) -> float:
        return self.score * self.weight


@dataclass
class ScoreCard:
    model: str
    task: str
    case_id: str
    dimensions: List[Score]
    output: str
    elapsed: float = 0.0

    @property
    def composite(self) -> float:
        if not self.dimensions:
            return 0.0
        return sum(s.weighted for s in self.dimensions)

    @property
    def total_weight(self) -> float:
        return sum(s.weight for s in self.dimensions)

    def report(self) -> str:
        comp = self.composite
        lines = [
            f"  {self.task:12s} {comp:5.1f}%  ({self.elapsed:5.1f}s)  {self.case_id}",
        ]
        for d in self.dimensions:
            if d.failures:
                lines.append(f"    {d.name:18s} {d.score:5.1f}%  FAIL: {'; '.join(d.failures)}")
            else:
                lines.append(f"    {d.name:18s} {d.score:5.1f}%")
        return "\n".join(lines)


# ============================================================
# TEST CASES WITH REFERENCE OUTPUTS
# ============================================================

@dataclass
class TestCase:
    task: str
    input_text: str
    reference: str
    description: str


FILENAME_CASES = [
    TestCase(
        task="filename",
        input_text="Screenshot showing login error: Invalid credentials. Please try again.",
        reference="login_error_invalid_credentials",
        description="Login error screenshot",
    ),
    TestCase(
        task="filename",
        input_text="Summer Festival 2024 - Family Fun Day at Central Park",
        reference="summer_festival_2024_family_fun_day",
        description="Event with clear subject",
    ),
    TestCase(
        task="filename",
        input_text="10 powerful sentences by Scott Adams navigating failure and ambition",
        reference="scott_adams_sentences_failure_ambition",
        description="Quote with multiple keywords",
    ),
    TestCase(
        task="filename",
        input_text="Meeting Notes - Project Alpha - Q1 Review",
        reference="meeting_notes_project_alpha_q1_review",
        description="Work meeting notes",
    ),
    TestCase(
        task="filename",
        input_text="Screen Shot 2024-03-15 at 14.30.22.png",
        reference="screen_shot_20240315_143022",
        description="Generic screenshot (less info)",
    ),
]

SUMMARIZE_CASES = [
    TestCase(
        task="summarize",
        input_text=(
            "[@user1 | 10:00] Just launched our new product! Excited to share.\n"
            "[@user2 | 10:15] Looks great! How do I get it?\n"
            "[@user1 | 10:30] Check the website for early access.\n"
            "[@user3 | 10:45] Got it, thanks!\n"
            "[@user2 | 11:00] Anyone tried the beta yet?\n"
            "[@user4 | 11:15] Been using it all morning, very smooth.\n"
            "[@user1 | 11:30] Great feedback!"
        ),
        reference=(
            "This conversation follows a product launch through four stages: announcement, access setup, beta testing, and community feedback. "
            "@user1 drives the narrative — announcing the product, directing users to early access, and acknowledging feedback. "
            "@user2 acts as the engaged user asking questions, @user3 confirms receipt, and @user4 provides the positive beta review.\n\n"
            "## Launch & Access\n"
            "- @user1 announced a new product at 10:00\n"
            "- @user2 asked how to get it at 10:15\n"
            "- @user1 directed users to the website for early access at 10:30\n"
            "- @user3 confirmed receiving the info at 10:45\n\n"
            "## Beta & Feedback\n"
            "- @user2 asked about beta testing at 11:00\n"
            "- @user4 reported smooth beta experience at 11:15\n"
            "- @user1 thanked everyone for the feedback at 11:30"
        ),
        description="Product launch with user interaction",
    ),
    TestCase(
        task="summarize",
        input_text=(
            "[@user5 | 09:00] Server migration starting. ETA 2 hours.\n"
            "[@user6 | 09:15] Database backup complete.\n"
            "[@user7 | 09:30] DNS propagation initiated.\n"
            "[@user5 | 10:00] Config sync in progress.\n"
            "[@user6 | 10:30] All services restored. Monitoring active.\n"
            "[@user7 | 11:00] No issues detected. Migration complete.\n"
            "[@user5 | 11:30] Post-mortem scheduled for tomorrow."
        ),
        reference=(
            "A server migration completed successfully over 2.5 hours. "
            "@user5 led the process with @user6 and @user7 handling backup, DNS, and monitoring phases. "
            "All services were restored by 10:30 with no issues detected.\n\n"
            "## Migration Steps\n"
            "- @user5 started server migration at 09:00\n"
            "- @user6 completed database backup at 09:15\n"
            "- @user7 initiated DNS propagation at 09:30\n"
            "- @user5 ran config sync at 10:00\n\n"
            "## Completion & Monitoring\n"
            "- @user6 restored all services at 10:30\n"
            "- @user7 confirmed no issues at 11:00\n"
            "- @user5 scheduled post-mortem at 11:30"
        ),
        description="Server migration timeline",
    ),
]

FILE_SUMMARY_CASES = [
    TestCase(
        task="file_summary",
        input_text=json.dumps([
            {"path": "eval_lib.py", "desc": "model evaluation functions"},
            {"path": "validators.py", "desc": "validation logic for JSON and text output"},
            {"path": "config.py", "desc": "configuration management and model prompts"},
            {"path": "osaurus_lib.py", "desc": "LLM API client library"},
        ]),
        reference=json.dumps([
            {"path": "eval_lib.py", "desc": "model evaluation functions for quality scoring"},
            {"path": "validators.py", "desc": "validation logic for JSON and text output"},
            {"path": "config.py", "desc": "configuration management and model prompts"},
            {"path": "osaurus_lib.py", "desc": "LLM API client for Ollama/OAI-compatible servers"},
        ]),
        description="4 files with known descriptions",
    ),
]

ALL_TEST_CASES = FILENAME_CASES + SUMMARIZE_CASES + FILE_SUMMARY_CASES


# ============================================================
# DIMENSION SCORERS
# ============================================================

def _str(x):
    return str(x) if x is not None else ""

def _lower(x):
    return _str(x).lower()


# -- Filename Dimensions --

def _score_filename_relevance(output: str, case: TestCase) -> Score:
    """How well the filename captures the key concepts from input."""
    out = _lower(output).strip()
    inp = _lower(case.input_text)
    ref = _lower(case.reference)

    if not out:
        return Score("Relevance", 0, 0.40, failures=["empty"])

    # Extract key tokens from input (skip stopwords)
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "to", "of", "in", "for", "on", "with", "at", "by", "from",
                 "and", "or", "but", "not", "please", "try", "again", "showing"}
    inp_tokens = set(re.findall(r'[a-z0-9]+', inp)) - stopwords
    out_tokens = set(re.findall(r'[a-z0-9]+', out))

    # Count how many input tokens appear as substrings in any output token
    # (handles merged words like "summerfestival" containing "summer" and "festival")
    matches = set()
    for it in inp_tokens:
        if it in out_tokens:
            matches.add(it)
        else:
            for ot in out_tokens:
                if it in ot:
                    matches.add(it)
                    break
    coverage = len(matches) / len(inp_tokens) if inp_tokens else 0

    # Also check against reference — overlapped tokens (with substring fallback)
    ref_tokens = set(re.findall(r'[a-z0-9]+', ref))
    ref_matches = set()
    for rt in ref_tokens:
        if rt in out_tokens:
            ref_matches.add(rt)
        else:
            for ot in out_tokens:
                if rt in ot:
                    ref_matches.add(rt)
                    break

    failures = []
    score = 0
    if coverage >= 0.6:
        score = 100
    elif coverage >= 0.4:
        score = 75
    elif coverage >= 0.2:
        score = 50
    else:
        score = 25
        failures.append(f"only {len(matches)}/{len(inp_tokens)} input tokens covered")

    # Penalize if too far from reference (missing key concepts)
    if ref_matches and len(ref_matches) < len(ref_tokens) * 0.4:
        score = min(score, 60)
        failures.append("missing key concepts from input")

    return Score("Relevance", score, 0.40, failures)


def _score_filename_format(output: str, case: TestCase) -> Score:
    """Is the output a valid filename (no spaces, proper length, no filler)."""
    out = output.strip()
    if not out:
        return Score("Format", 0, 0.35, failures=["empty"])

    failures = []
    deduction = 0

    # Empty/generic check
    GENERIC = {"filename.txt", "file.txt", "text.txt", "output.txt",
               "document.txt", "note.txt", "screenshot.png", "unnamed", "file"}
    if _lower(out) in GENERIC:
        return Score("Format", 0, 0.35, failures=["generic filename"])

    # No question-like text
    if "?" in out or "please" in _lower(out):
        deduction += 50
        failures.append("has question/instruction text")

    # Valid chars (0-9, a-z, _, -, .)
    # Spaces are INVALID in filenames
    valid_part = re.sub(r'[a-zA-Z0-9_.-]', '', out)
    if valid_part:
        space_count = valid_part.count(" ")
        non_space = valid_part.replace(" ", "")
        if space_count > 0:
            deduction += 40 + (space_count * 5)
            failures.append(f"has {space_count} space(s)")
        if non_space:
            deduction += 20
            failures.append(f"invalid chars: {non_space[:10]}")

    # Length check
    if len(out) > 60:
        deduction += 20
        failures.append(f"too long ({len(out)} chars)")

    # Uppercase check (filenames should be lowercase)
    if out != _lower(out) and any(c.isupper() for c in out):
        deduction += 10
        failures.append("has uppercase")

    # Separator presence (underscores or dashes)
    if "_" not in out and "-" not in out and "." not in out:
        deduction += 10
        failures.append("no separators")

    score = max(0, 100 - deduction)
    return Score("Format", score, 0.35, failures)


def _score_filename_conciseness(output: str, case: TestCase) -> Score:
    """Is the filename concise and not overly verbose."""
    out = output.strip()
    if not out:
        return Score("Conciseness", 0, 0.25, failures=["empty"])

    failures = []
    score = 100

    # If output is not a real filename (question, has spaces), conciseness is irrelevant
    if "?" in out or "please" in _lower(out):
        return Score("Conciseness", 0, 0.25, failures=["not a filename (question)"])
    if " " in out:
        return Score("Conciseness", 10, 0.25, failures=["has spaces — not a filename"])

    # Ideal: 10-40 chars
    length = len(out)
    if length < 5:
        score = 50
        failures.append(f"too short ({length} chars)")
    elif length < 10:
        score = 75
    elif 10 <= length <= 45:
        score = 100
    elif length <= 60:
        score = 80
        failures.append(f"slightly long ({length} chars)")
    else:
        score = 40
        failures.append(f"too long ({length} chars)")

    # Check for unnecessary words
    filler = ["the", "and", "of", "for", "with", "from", "this", "that"]
    if any(f in _lower(re.sub(r'[_-]', ' ', out)).split() for f in filler):
        score = max(score - 15, 0)
        failures.append("has filler words")

    return Score("Conciseness", score, 0.25, failures)


# -- Summarize Dimensions --

def _score_summarize_completeness(output: str, case: TestCase) -> Score:
    """All events and users covered."""
    out = _str(output)
    inp = _lower(case.input_text)
    if not out or len(out) < 30:
        return Score("Completeness", 0, 0.30, failures=["empty or too short"])

    failures = []

    # Count expected users in output (by user number, ignoring @ vs plain)
    users_ref = set(re.findall(r'user\s*(\d+)', inp, re.IGNORECASE))
    users_out = set(re.findall(r'user\s*(\d+)', out, re.IGNORECASE))
    user_ratio = len(users_out & users_ref) / len(users_ref) if users_ref else 1
    if user_ratio < 0.75:
        failures.append(f"users: {len(users_out & users_ref)}/{len(users_ref)}")

    # Count timestamped events in input vs output
    events = len(re.findall(r'\d{1,2}:\d{2}', inp))
    out_events = len(re.findall(r'\d{1,2}:\d{2}', out))
    event_ratio = min(1.0, out_events / events) if events else 1
    if event_ratio < 0.5:
        failures.append(f"events: {out_events}/{events} timestamped")

    # Check for key topic words from input
    topics = {"launch", "access", "beta", "feedback", "migration",
              "backup", "dns", "services"}
    inp_topics = {t for t in topics if t in inp}
    out_topics = {t for t in topics if t in _lower(out)}
    topic_ratio = len(out_topics & inp_topics) / len(inp_topics) if inp_topics else 1
    if topic_ratio < 0.5:
        failures.append(f"topics: {len(out_topics & inp_topics)}/{len(inp_topics)}")

    # Composite completeness score
    raw = (user_ratio + event_ratio + topic_ratio) / 3
    score = raw * 100
    if not failures:
        score = min(100, score + 10)

    return Score("Completeness", score, 0.30, failures)


def _score_summarize_synthesis(output: str, case: TestCase) -> Score:
    """TL;DR, narrative connecting events, relationship awareness."""
    out = _str(output)
    if not out:
        return Score("Synthesis", 0, 0.25, failures=["empty"])

    failures = []
    score = 0

    # 1. Has TL;DR / top-level synthesis (40 pts)
    header_match = re.search(r'\n#{2,}\s+\w+', out)
    if header_match:
        top_level = out[:header_match.start()].strip()
    elif not re.search(r'^#{2,}\s+\w+', out, re.MULTILINE):
        top_level = out  # no headers at all
    else:
        top_level = ""

    has_synthesis = bool(re.search(
        r'(?i)(overall|summary|in (short|summary)|tl;dr|(the|this|that) '
        r'(conversation|discussion|thread|interaction|timeline|migration|launch))',
        top_level
    )) if top_level else False

    # 2. Narrative connecting language (30 pts)
    narrative_verbs = len(re.findall(
        r'(?i)\b(?:ask(?:s|ed|ing)?|respond(?:s|ed|ing)?|thank(?:s|ed|ing)?|'
        r'report(?:s|ed|ing)?|confirm(?:s|ed|ing)?|direct(?:s|ed|ing)?|'
        r'inquire(?:s|d|ing)?|announce(?:s|d|ing)?|share(?:s|d|ing)?|'
        r'request(?:s|ed|ing)?|provide(?:s|d|ing)?|drive(?:s|n)?|lead?|'
        r'act(?:s|ed|ing)?|handle(?:s|d)?|manage(?:s|d)?|coordinate(?:s|d)?)\b',
        out
    ))

    # 3. Relationship awareness — connects users to actions (30 pts)
    user_action = len(re.findall(
        r'(?i)@?[Uu]ser\s*\d+\s+(?:announce|ask|direct|confirm|report|thank|inquire)',
        out
    ))
    relationship_patterns = len(re.findall(
        r'(?i)(in response|follow(?:ing|ed|s)? (?:up|that)|'
        r'the(?:n| (?:discussion|conversation|thread)) (?:shift|move|transition|turn)|'
        r'wrapped? up|kicked? off|stepped? in)',
        out
    ))

    synthesis_score = 40 if has_synthesis else 0
    narrative_score = min(30, narrative_verbs * 6)
    relationship_score = min(30, (user_action + relationship_patterns) * 8)

    score = synthesis_score + narrative_score + relationship_score

    if not has_synthesis:
        failures.append("no TL;DR/synthesis paragraph")
    if narrative_verbs == 0:
        failures.append("no narrative verbs")
    if user_action == 0 and relationship_patterns == 0:
        failures.append("no relationship awareness")

    return Score("Synthesis", score, 0.25, failures)


def _score_summarize_structure(output: str, case: TestCase) -> Score:
    """Headers, bullets, readability."""
    out = _str(output)
    if not out:
        return Score("Structure", 0, 0.20, failures=["empty"])

    failures = []
    score = 0

    has_headers = bool(re.search(r'^#{2,}\s+\w+', out, re.MULTILINE))
    has_bullets = bool(re.search(r'^[\s]*[-*•]', out, re.MULTILINE))

    if has_headers and has_bullets:
        score = 100
    elif has_headers:
        score = 70
    elif has_bullets:
        score = 50
    else:
        score = 20
        failures.append("no headers or bullet points")

    # Prevents template-driven penalty as a structure sub-issue
    template_fields = len(re.findall(r'\*\*(Who|What|When|Where):', out))
    if template_fields >= 3:
        score = max(30, score - 40)
        failures.append("template-like structure")

    # Length range
    if len(out) < 100:
        score = min(score, 50)
        failures.append("too short")
    elif len(out) > 2000:
        score = min(score, 80)
        failures.append("too long")

    return Score("Structure", score, 0.20, failures)


def _score_summarize_specificity(output: str, case: TestCase) -> Score:
    """Timestamps, @mentions, concrete details from input."""
    out = _str(output)
    if not out:
        return Score("Specificity", 0, 0.25, failures=["empty"])

    failures = []
    score = 0

    # Timestamps (40 pts)
    timestamps = len(re.findall(r'\d{1,2}:\d{2}', out))
    expected_events = len(re.findall(r'\d{1,2}:\d{2}', case.input_text))
    ts_score = min(40, (timestamps / expected_events * 40) if expected_events else 0)

    # @mentions (30 pts) — unique users by number
    user_numbers = set(re.findall(r'user\s*(\d+)', out, re.IGNORECASE))
    expected_users = set(re.findall(r'user\s*(\d+)', case.input_text, re.IGNORECASE))
    user_coverage = len(user_numbers & expected_users) / len(expected_users) if expected_users else 0
    mention_score = user_coverage * 30

    # Concrete details (30 pts) — unique user+timestamp combos
    inp_details = set(re.findall(r'\d{1,2}:\d{2}', case.input_text))
    inp_users = set(re.findall(r'user\s*(\d+)', case.input_text, re.IGNORECASE))
    out_ts = set(re.findall(r'\d{1,2}:\d{2}', out))
    out_users = set(re.findall(r'user\s*(\d+)', out, re.IGNORECASE))
    inp_total = len(inp_details) + len(inp_users)
    out_total = len(out_ts & inp_details) + len(out_users & inp_users)
    detail_ratio = out_total / inp_total if inp_total else 0
    detail_score = min(30, detail_ratio * 30)

    score = int(ts_score + mention_score + detail_score)

    if timestamps < expected_events * 0.5 and expected_events:
        failures.append(f"missing timestamps ({timestamps}/{expected_events})")
    if not user_numbers:
        failures.append("no user mentions")

    return Score("Specificity", score, 0.25, failures)


# -- File Summary Dimensions --

def _score_file_completeness(output: str, case: TestCase) -> Score:
    """All expected files present (by path)."""
    out = _str(output)
    if not out:
        return Score("Completeness", 0, 0.40, failures=["empty"])

    failures = []
    try:
        data = json.loads(out)
    except (json.JSONDecodeError, TypeError):
        return Score("Completeness", 0, 0.40, failures=["invalid JSON"])

    if not isinstance(data, list):
        return Score("Completeness", 0, 0.40, failures=["not a list"])

    # Check expected paths
    ref = json.loads(case.reference)
    exp_paths = {item["path"] for item in ref}
    out_paths = {item.get("path", "") for item in data if isinstance(item, dict)}
    found = exp_paths & out_paths
    ratio = len(found) / len(exp_paths) if exp_paths else 0

    score = ratio * 100
    if ratio < 1.0:
        missing = exp_paths - out_paths
        failures.append(f"missing files: {', '.join(sorted(missing))}")

    return Score("Completeness", score, 0.40, failures)


def _score_file_accuracy(output: str, case: TestCase) -> Score:
    """Descriptions match file purpose (keyword overlap with expected)."""
    out = _str(output)
    if not out:
        return Score("Accuracy", 0, 0.30, failures=["empty"])

    try:
        data = json.loads(out)
    except (json.JSONDecodeError, TypeError):
        return Score("Accuracy", 0, 0.30, failures=["invalid JSON"])

    if not isinstance(data, list):
        return Score("Accuracy", 0, 0.30, failures=["not a list"])

    ref = json.loads(case.reference)
    failures = []
    total_score = 0
    count = 0

    for ref_item in ref:
        ref_path = ref_item["path"]
        ref_desc = ref_item["desc"]
        # Find matching output item
        match = next(
            (item for item in data if isinstance(item, dict)
             and item.get("path", "") == ref_path),
            None
        )
        if match is None:
            failures.append(f"'{ref_path}' not found")
            continue

        out_desc = _str(match.get("desc", ""))
        if not out_desc:
            failures.append(f"'{ref_path}' has no description")
            continue

        # Score by token overlap between description and expected (with substring fallback)
        ref_tokens = set(re.findall(r'[a-z]+', _lower(ref_desc)))
        out_tokens = set(re.findall(r'[a-z]+', _lower(out_desc)))
        if len(ref_tokens) == 0:
            continue

        overlap = set()
        for rt in ref_tokens:
            if rt in out_tokens:
                overlap.add(rt)
            else:
                for ot in out_tokens:
                    if rt in ot:
                        overlap.add(rt)
                        break
        ratio = len(overlap) / len(ref_tokens)
        item_score = min(100, ratio * 100)
        total_score += item_score
        count += 1

        if ratio < 0.3:
            failures.append(f"'{ref_path}' desc mismatch")

    if count == 0:
        return Score("Accuracy", 0, 0.30, failures=["no items scored"])

    return Score("Accuracy", total_score / count, 0.30, failures)


def _score_file_format(output: str, case: TestCase) -> Score:
    """Valid JSON, correct array structure, valid items."""
    out = _str(output)
    if not out:
        return Score("Format", 0, 0.30, failures=["empty"])

    failures = []
    try:
        data = json.loads(out)
    except (json.JSONDecodeError, TypeError):
        return Score("Format", 0, 0.30, failures=["invalid JSON"])

    if not isinstance(data, list):
        return Score("Format", 0, 0.30, failures=["not a list"])

    if len(data) == 0:
        return Score("Format", 30, 0.30, failures=["empty array"])

    valid = sum(1 for item in data if isinstance(item, dict)
                and "path" in item and "desc" in item)
    ratio = valid / len(data)
    score = ratio * 100

    if ratio < 1.0:
        failures.append(f"{valid}/{len(data)} items have valid schema")

    return Score("Format", score, 0.30, failures)


# ============================================================
# SCORE CARD BUILDERS
# ============================================================

TASK_SCORERS: Dict[str, List[Callable]] = {
    "filename": [
        _score_filename_relevance,
        _score_filename_format,
        _score_filename_conciseness,
    ],
    "summarize": [
        _score_summarize_completeness,
        _score_summarize_synthesis,
        _score_summarize_structure,
        _score_summarize_specificity,
    ],
    "file_summary": [
        _score_file_completeness,
        _score_file_accuracy,
        _score_file_format,
    ],
}


def score_output(output: str, task: str, case: TestCase) -> ScoreCard:
    """Score a model output against a test case, returning per-dimension scores."""
    out = output.strip()

    # Critical failure gate: empty or generic produces all zeros
    if not out:
        return ScoreCard(
            model="", task=task, case_id=case.description,
            dimensions=[],
            output=output,
        )

    if task == "filename":
        GENERIC = {"filename.txt", "file.txt", "text.txt", "output.txt",
                   "document.txt", "note.txt", "screenshot.png", "unnamed", "file"}
        if _lower(out) in GENERIC:
            return ScoreCard(
                model="", task=task, case_id=case.description,
                dimensions=[Score("Relevance", 0, 0.40, failures=["generic"]),
                            Score("Format", 0, 0.35, failures=["generic"]),
                            Score("Conciseness", 0, 0.25, failures=["generic"])],
                output=output,
            )

    scorers = TASK_SCORERS.get(task, [])
    dimensions = [scorer(output, case) for scorer in scorers]
    return ScoreCard(
        model="",
        task=task,
        case_id=case.description,
        dimensions=dimensions,
        output=output,
    )


# ============================================================
# MODEL CALLING
# ============================================================

def query_model(model: str, prompt: str, input_text: str, task: str) -> Optional[str]:
    """Call the model and return its text output."""
    try:
        filled = _safe_format_prompt(prompt, input_text)
        result = llm_call(
            model=model,
            messages=[{"role": "user", "content": filled}],
            timeout=600,
            task=task,
        )
        return _str(result.get("content"))
    except Exception as e:
        return None


# ============================================================
# RUNNER
# ============================================================

def run_suite(models: List[str], cases: List[TestCase] = None,
              verbose: bool = True) -> List[ScoreCard]:
    """Run quality suite against models and return ScoreCards."""
    if cases is None:
        cases = ALL_TEST_CASES

    results = []
    total = len(models) * len(cases)

    for i, model in enumerate(models):
        for j, case in enumerate(cases):
            idx = i * len(cases) + j + 1
            if verbose:
                print(f"  [{idx}/{total}] {model[:30]:30s} {case.task:12s} {case.description}",
                      end=" ", flush=True)

            prompt = get_model_prompt(model, Task(case.task))
            if not prompt:
                if verbose:
                    print("SKIP (no prompt)")
                continue

            t0 = time.time()
            output = query_model(model, prompt, case.input_text, case.task)
            elapsed = time.time() - t0

            if output is None:
                if verbose:
                    print("ERROR")
                results.append(ScoreCard(
                    model=model, task=case.task, case_id=case.description,
                    dimensions=[], output="", elapsed=elapsed,
                ))
                continue

            sc = score_output(output, case.task, case)
            sc.model = model
            sc.elapsed = elapsed
            results.append(sc)

            if verbose:
                comp = sc.composite
                if sc.dimensions:
                    worst = min(d.score for d in sc.dimensions)
                    prefix = "✓" if worst >= 60 else ("△" if comp >= 40 else "✗")
                    failures = [d.failures for d in sc.dimensions if d.failures]
                    fail_str = f" [{'; '.join(f for f in failures[0])}]" if failures else ""
                    print(f"{prefix}  {comp:5.1f}%  ({elapsed:.1f}s){fail_str}")
                else:
                    print("✗  0.0%")

    return results


# ============================================================
# REPORT
# ============================================================

def generate_report(results: List[ScoreCard]) -> str:
    """Generate a formatted comparison report from ScoreCards."""
    lines = [
        "=" * 90,
        "  QUALITY REPORT — Per-Dimension Scoring",
        "=" * 90,
    ]

    # Group by model
    by_model: Dict[str, List[ScoreCard]] = {}
    for sc in results:
        by_model.setdefault(sc.model, []).append(sc)

    for model in sorted(by_model.keys()):
        cards = by_model[model]
        lines.append(f"\n  ── {model} ──")

        # Per-task average dimension scores
        by_task: Dict[str, List[ScoreCard]] = {}
        for sc in cards:
            by_task.setdefault(sc.task, []).append(sc)

        for task in ["filename", "summarize", "file_summary"]:
            task_cards = by_task.get(task, [])
            if not task_cards:
                continue

            avg_dim = {}
            for sc in task_cards:
                for d in sc.dimensions:
                    avg_dim.setdefault(d.name, []).append(d.score)

            dim_avgs = {name: sum(scores)/len(scores)
                        for name, scores in avg_dim.items()}

            # Composite average
            composites = [sc.composite for sc in task_cards]
            avg_comp = sum(composites) / len(composites)

            lines.append(f"\n    {task}:")
            for name, avg in sorted(dim_avgs.items()):
                bar = "█" * int(avg / 10) + "░" * (10 - int(avg / 10))
                lines.append(f"      {name:18s} {avg:5.1f}%  {bar}")
            lines.append(f"      {'─' * 40}")
            lines.append(f"      {'Composite':18s} {avg_comp:5.1f}%")

            # Per-case timing
            times = [sc.elapsed for sc in task_cards]
            avg_time = sum(times) / len(times) if times else 0
            lines.append(f"      {'Avg time':18s} {avg_time:5.1f}s  ({len(times)} cases)")

    # Cross-model comparison
    lines.extend([
        "\n",
        "=" * 90,
        "  CROSS-MODEL COMPARISON",
        "=" * 90,
        f"  {'Model':35s} {'Filename':>10} {'Summarize':>12} {'FileSum':>10} {'Speed':>8} {'Fail':>6}",
        f"  {'-'*35} {'-'*10} {'-'*12} {'-'*10} {'-'*8} {'-'*6}",
    ])

    for model in sorted(by_model.keys()):
        cards = by_model[model]
        task_avgs = {}
        task_times = {}
        failures = 0
        for sc in cards:
            task_avgs.setdefault(sc.task, []).append(sc.composite)
            task_times.setdefault(sc.task, []).append(sc.elapsed)
            for d in sc.dimensions:
                if d.failures:
                    failures += 1

        def avg(vals):
            return sum(vals) / len(vals) if vals else 0

        f_avg = avg(task_avgs.get("filename", []))
        s_avg = avg(task_avgs.get("summarize", []))
        fs_avg = avg(task_avgs.get("file_summary", []))
        all_times = [t for times in task_times.values() for t in times]
        speed = avg(all_times) if all_times else 0

        lines.append(
            f"  {model:35s} {f_avg:8.1f}%  {s_avg:10.1f}%  {fs_avg:8.1f}%  "
            f"{speed:6.1f}s  {failures:5}"
        )

    lines.append("")
    return "\n".join(lines)


# ============================================================
# REGRESSION DETECTION (Baseline)
# ============================================================

BASELINE_PATH = Path(__file__).parent.parent / "docs" / "eval_baseline.json"


def _model_task_key(model: str, task: str, case_id: str) -> str:
    return f"{model}::{task}::{case_id}"


def save_baseline(results: List[ScoreCard]):
    """Save current results as the baseline for regression detection."""
    baseline = {}
    for sc in results:
        key = _model_task_key(sc.model, sc.task, sc.case_id)
        baseline[key] = {
            "composite": sc.composite,
            "dimensions": {d.name: d.score for d in sc.dimensions},
            "elapsed": sc.elapsed,
        }

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
    return baseline


def load_baseline() -> dict:
    """Load the baseline file. Returns empty dict if not found."""
    if not BASELINE_PATH.exists():
        return {}
    with open(BASELINE_PATH) as f:
        return json.load(f)


def compare_to_baseline(results: List[ScoreCard]) -> List[str]:
    """Compare results to baseline, return regression warnings."""
    baseline = load_baseline()
    if not baseline:
        return ["  No baseline found. Run with --save-baseline to create one."]

    warnings = []
    for sc in results:
        key = _model_task_key(sc.model, sc.task, sc.case_id)
        prev = baseline.get(key)
        if not prev:
            continue

        curr_comp = sc.composite
        prev_comp = prev["composite"]
        delta = curr_comp - prev_comp

        if delta < -10:
            dim_deltas = []
            for d in sc.dimensions:
                prev_d = prev.get("dimensions", {}).get(d.name, 0)
                dd = d.score - prev_d
                if dd < -15:
                    dim_deltas.append(f"{d.name}: {prev_d:.0f}→{d.score:.0f} ({dd:+.0f})")
            detail = f" [{'; '.join(dim_deltas)}]" if dim_deltas else ""
            warnings.append(
                f"  ⚠ REGRESSION: {sc.model} / {sc.task} / {sc.case_id}\n"
                f"    {prev_comp:.1f}% → {curr_comp:.1f}% ({delta:+.1f}pts){detail}"
            )
        elif delta > 10:
            warnings.append(
                f"  ↑ IMPROVEMENT: {sc.model} / {sc.task} / {sc.case_id}\n"
                f"    {prev_comp:.1f}% → {curr_comp:.1f}% ({delta:+.1f}pts)"
            )

    return warnings


# ============================================================
# MAIN
# ============================================================

def main():
    """Run quality suite from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Quality evaluation suite")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Models to test (default: all with prompts)")
    parser.add_argument("--tasks", nargs="*", default=None,
                        choices=["filename", "summarize", "file_summary"],
                        help="Tasks to test (default: all)")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save results as baseline for regression detection")
    parser.add_argument("--regression-only", action="store_true",
                        help="Only compare to baseline, don't re-run models")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output")
    args = parser.parse_args()

    # Filter cases by task
    cases = ALL_TEST_CASES
    if args.tasks:
        cases = [c for c in cases if c.task in args.tasks]

    # Default models
    models = args.models or ["foundation", "qwopus3.6-27b-v2-mlx-4bit",
                             "nemotron-3-nano-omni-30b-a3b-mxfp4"]

    if args.regression_only:
        # Load latest results and compare
        baseline = load_baseline()
        if not baseline:
            print("No baseline found at", BASELINE_PATH)
            return
        print(f"Loaded baseline with {len(baseline)} entries from {BASELINE_PATH}")

        # Reconstruct ScoreCards from baseline (no real results available)
        print("(Regression-only mode — re-run without --regression-only to compare new results)")
        return

    # Run suite
    print(f"Quality Suite: {len(models)} models × {len(cases)} cases")
    results = run_suite(models, cases, verbose=not args.quiet)

    # Report
    print(generate_report(results))

    # Regression check
    print("\n  ── Regression Check ──")
    warnings = compare_to_baseline(results)
    if warnings:
        for w in warnings:
            print(w)
    else:
        print("  No regressions detected.")

    # Save baseline if requested
    if args.save_baseline:
        save_baseline(results)
        print(f"\n  Baseline saved to {BASELINE_PATH}")


if __name__ == "__main__":
    main()
