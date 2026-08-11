"""Scorers for the summarize task.

Split out of quality_scorers.py for the repo's 500-line limit. Importing this
module registers its scorers; quality_scorers.py does that import.
"""

import re

from lib.quality_models import Score, TestCase, _lower, _str
from lib.quality_scorers_core import register_scorer


@register_scorer("summarize")
def _score_summarize_completeness(output: str, case: TestCase) -> Score:
    out = _str(output)
    inp = _lower(case.input_text)
    if not out or len(out) < 30:
        return Score("Completeness", 0, 0.30, failures=["empty or too short"])

    failures = []

    users_ref = set(re.findall(r"user\s*(\d+)", inp, re.IGNORECASE))
    users_out = set(re.findall(r"user\s*(\d+)", out, re.IGNORECASE))
    user_ratio = len(users_out & users_ref) / len(users_ref) if users_ref else 1
    if user_ratio < 0.75:
        failures.append(f"users: {len(users_out & users_ref)}/{len(users_ref)}")

    events = len(re.findall(r"\d{1,2}:\d{2}", inp))
    out_events = len(re.findall(r"\d{1,2}:\d{2}", out))
    event_ratio = min(1.0, out_events / events) if events else 1
    if event_ratio < 0.5:
        failures.append(f"events: {out_events}/{events} timestamped")

    topics = {"launch", "access", "beta", "feedback", "migration", "backup", "dns", "services"}
    inp_topics = {t for t in topics if t in inp}
    out_topics = {t for t in topics if t in _lower(out)}
    topic_ratio = len(out_topics & inp_topics) / len(inp_topics) if inp_topics else 1
    if topic_ratio < 0.5:
        failures.append(f"topics: {len(out_topics & inp_topics)}/{len(inp_topics)}")

    raw = (user_ratio + event_ratio + topic_ratio) / 3
    score = raw * 100
    if not failures:
        score = min(100, score + 10)

    return Score("Completeness", score, 0.30, failures)


@register_scorer("summarize")
def _score_summarize_synthesis(output: str, case: TestCase) -> Score:
    out = _str(output)
    if not out:
        return Score("Synthesis", 0, 0.25, failures=["empty"])

    failures = []
    score = 0

    header_match = re.search(r"\n#{2,}\s+\w+", out)
    if header_match:
        top_level = out[: header_match.start()].strip()
    elif not re.search(r"^#{2,}\s+\w+", out, re.MULTILINE):
        top_level = out
    else:
        top_level = ""

    has_synthesis = (
        bool(
            re.search(
                r"(?i)(overall|summary|in (short|summary)|tl;dr|(the|this|that) "
                r"(conversation|discussion|thread|interaction|timeline|migration|launch))",
                top_level,
            )
        )
        if top_level
        else False
    )

    narrative_verbs = len(
        re.findall(
            r"(?i)\b(?:ask(?:s|ed|ing)?|respond(?:s|ed|ing)?|thank(?:s|ed|ing)?|"
            r"report(?:s|ed|ing)?|confirm(?:s|ed|ing)?|direct(?:s|ed|ing)?|"
            r"inquire(?:s|d|ing)?|announce(?:s|d|ing)?|share(?:s|d|ing)?|"
            r"request(?:s|ed|ing)?|provide(?:s|d|ing)?|drive(?:s|n)?|lead?|"
            r"act(?:s|ed|ing)?|handle(?:s|d)?|manage(?:s|d)?|coordinate(?:s|d)?)\b",
            out,
        )
    )

    user_action = len(
        re.findall(
            r"(?i)@?[Uu]ser\s*\d+\s+(?:announce|ask|direct|confirm|report|thank|inquire)", out
        )
    )
    relationship_patterns = len(
        re.findall(
            r"(?i)(in response|follow(?:ing|ed|s)? (?:up|that)|"
            r"the(?:n| (?:discussion|conversation|thread)) (?:shift|move|transition|turn)|"
            r"wrapped? up|kicked? off|stepped? in)",
            out,
        )
    )

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


@register_scorer("summarize")
def _score_summarize_structure(output: str, case: TestCase) -> Score:
    out = _str(output)
    if not out:
        return Score("Structure", 0, 0.20, failures=["empty"])

    failures = []
    score = 0

    has_headers = bool(re.search(r"^#{2,}\s+\w+", out, re.MULTILINE))
    has_bullets = bool(re.search(r"^[\s]*[-*•]", out, re.MULTILINE))

    if has_headers and has_bullets:
        score = 100
    elif has_headers:
        score = 70
    elif has_bullets:
        score = 50
    else:
        score = 20
        failures.append("no headers or bullet points")

    template_fields = len(re.findall(r"\*\*(Who|What|When|Where):", out))
    if template_fields >= 3:
        score = max(30, score - 40)
        failures.append("template-like structure")

    if len(out) < 100:
        score = min(score, 50)
        failures.append("too short")
    elif len(out) > 2000:
        score = min(score, 80)
        failures.append("too long")

    return Score("Structure", score, 0.20, failures)


@register_scorer("summarize")
def _score_summarize_specificity(output: str, case: TestCase) -> Score:
    out = _str(output)
    if not out:
        return Score("Specificity", 0, 0.25, failures=["empty"])

    failures = []
    score = 0

    timestamps = len(re.findall(r"\d{1,2}:\d{2}", out))
    expected_events = len(re.findall(r"\d{1,2}:\d{2}", case.input_text))
    ts_score = min(40, (timestamps / expected_events * 40) if expected_events else 0)

    user_numbers = set(re.findall(r"user\s*(\d+)", out, re.IGNORECASE))
    expected_users = set(re.findall(r"user\s*(\d+)", case.input_text, re.IGNORECASE))
    user_coverage = (
        len(user_numbers & expected_users) / len(expected_users) if expected_users else 0
    )
    mention_score = user_coverage * 30

    inp_details = set(re.findall(r"\d{1,2}:\d{2}", case.input_text))
    inp_users = set(re.findall(r"user\s*(\d+)", case.input_text, re.IGNORECASE))
    out_ts = set(re.findall(r"\d{1,2}:\d{2}", out))
    out_users = set(re.findall(r"user\s*(\d+)", out, re.IGNORECASE))
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
