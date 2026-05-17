"""Validators for the taxes_* tasks ported from
github.com/ztomer/Taxes (see eval_tasks/data/taxes/*.json).

The rubric mirrors `core.llm_model_eval._score_output` from the
source repo, but in self-contained form so this validator has no
external dependencies beyond the JSON snapshot.

Score is 0–100. Three weighted components:
  - grounding (0–40): how many cross-border signals the output
    references. 5+ = 40, 4 = 32, 3 = 24, 2 = 16, 1 = 8, 0 = 0.
  - no_leak (0–30): 30 by default; drops to 0 if any GT-flavored
    substring appears verbatim.
  - substance (0–30): full marks for prose with specific dollar
    figures and not list-form-only; -10 per missing signal,
    floor 0.

The snapshot's `rubric` dict carries `expected_signals` and
`gt_forbidden` so this validator is purely a JSON consumer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Tuple


def _load_rubric(task_name: str) -> dict[str, Any]:
    """Read the snapshot's rubric block by task short-name."""
    data_dir = Path(__file__).resolve().parent.parent.parent / "eval_tasks" / "data" / "taxes"
    fp = data_dir / f"taxes_{task_name}.sanitized.json"
    if not fp.exists():
        return {}
    return (json.loads(fp.read_text(encoding="utf-8")).get("rubric") or {})


def _grounding_score(output: str, expected_signals: list[str]) -> tuple[int, int]:
    """Returns (score 0-40, hits count). Signals are matched
    case-insensitively as substrings."""
    if not expected_signals:
        return 40, 0
    out_lower = (output or "").lower()
    hits = sum(1 for sig in expected_signals if sig.lower() in out_lower)
    if hits >= 5:  return 40, hits
    if hits == 4: return 32, hits
    if hits == 3: return 24, hits
    if hits == 2: return 16, hits
    if hits == 1: return 8, hits
    return 0, hits


def _no_leak_score(output: str, gt_forbidden: list[str]) -> int:
    """30 unless the output verbatim-quotes a GT-flavored term.
    Reaching 0 here means the eval surface leaked — that's an
    LLM-context contract failure, not an LLM quality issue, but it
    still drops the score because the output is dangerous to ship."""
    if not output:
        return 30
    if any(term in output for term in (gt_forbidden or [])):
        return 0
    return 30


def _substance_score(output: str) -> int:
    """0–30 based on output depth heuristics:
    -10 for <600 chars, -10 for no specific $ amounts cited,
    -10 for list-form-only (≥3 bullets, thin connective prose).
    Floor at 0."""
    if not output:
        return 0
    score = 30
    chars = len(output.strip())
    if chars < 600:
        score -= 10
    if not re.findall(r"\$([\d,]+(?:\.\d{2})?)", output):
        score -= 10
    bullet_count = len(re.findall(r"^\s*(?:\d+[.)]\s+|[-•*]\s+)",
                                  output, re.MULTILINE))
    non_bullet = re.sub(r"^[\s\d.\-•*]+", "", output, flags=re.MULTILINE)
    non_bullet = re.sub(r"\s+", " ", non_bullet)
    if bullet_count >= 3 and len(non_bullet) < bullet_count * 60:
        score -= 10
    return max(0, score)


def _validate_taxes_task(output: str, task_name: str) -> Tuple[int, str]:
    """Common rubric: grounding (40) + no_leak (30) + substance (30)."""
    rubric = _load_rubric(task_name)
    expected = rubric.get("expected_signals", [])
    forbidden = rubric.get("gt_forbidden", [])

    g_score, g_hits = _grounding_score(output, expected)
    l_score = _no_leak_score(output, forbidden)
    s_score = _substance_score(output)
    total = g_score + l_score + s_score

    reason = (f"grounding={g_score}/40 ({g_hits} signals)  "
              f"no_leak={l_score}/30  substance={s_score}/30")
    if l_score == 0:
        reason += "  ⚠ GT-flavored term leaked"
    return total, reason


def validate_taxes_anomalies(output: Any) -> Tuple[int, str]:
    return _validate_taxes_task(str(output or ""), "anomalies")


def validate_taxes_audit_readiness(output: Any) -> Tuple[int, str]:
    """Same rubric + additional schema check: must be valid JSON
    with a `risk_items` list. If schema fails, halve the total."""
    raw = str(output or "")
    score, reason = _validate_taxes_task(raw, "audit_readiness")
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("risk_items"), list):
            reason += "  schema=ok"
        else:
            score = score // 2
            reason += "  schema=bad-shape (score halved)"
    except (json.JSONDecodeError, ValueError):
        score = score // 2
        reason += "  schema=not-json (score halved)"
    return score, reason


def validate_taxes_synthesis(output: Any) -> Tuple[int, str]:
    """Same rubric + markdown-section check: at least 4 of 5
    expected `**N. Section**` headings present."""
    raw = str(output or "")
    score, reason = _validate_taxes_task(raw, "synthesis")
    rubric = _load_rubric("synthesis")
    expected_sections = rubric.get("expected_sections", [])
    if expected_sections:
        hits = sum(1 for s in expected_sections if s in raw)
        if hits < len(expected_sections) - 1:  # allow one missing
            score = max(0, score - 10)
            reason += f"  sections={hits}/{len(expected_sections)} (−10)"
        else:
            reason += f"  sections={hits}/{len(expected_sections)}"
    return score, reason


__all__ = [
    "validate_taxes_anomalies",
    "validate_taxes_audit_readiness",
    "validate_taxes_synthesis",
]
