"""Validators for the three GROUNDED taxes tasks (yoy_narrative, qa, slip_qa).

These are deliberately unlike `taxes_validator.py`. That module scores against a
`rubric` block -- keyword signals, forbidden substrings, length heuristics. This
one scores against a `grounding` block, where the verdict is ARITHMETIC or
SET-MEMBERSHIP and therefore cannot be satisfied by writing the right words.
That is the whole reason these three were imported: the keyword tasks saturate.

Each snapshot carries `grounding.rule` stating the check in prose, so the
contract does not have to be reverse-engineered from field names. The functions
below implement exactly those three rules and nothing else -- notably NOT the
prompts' stylistic asks (word counts, "explain the cause not the effect"),
which no grounding field can adjudicate.

These snapshots carry no `rubric`, so grounding is the only signal. If the
arithmetic here is wrong there is no second opinion to disagree with it, which
is why `test_taxes_grounded.py` builds each ideal answer FROM the grounding
block and asserts it scores 100 before any model score is trusted.

Dollar figures in prose are recognised by money shape -- a `$` sigil, a `CAD`
suffix, or comma-grouped/2-decimal digits. A bare integer ("the RRSP line moved
by 50357") is NOT counted, so this under-detects rather than inventing
violations; years and T1 line numbers stay out of the match for the same reason.
"""

from __future__ import annotations

import json
import re
import sys
from itertools import combinations
from typing import Any, Iterable, Tuple

from lib.paths import eval_tasks_path
from lib.tui import WARN

MAX_SCORE = 100

# Money shapes: "$1,234.56" / "1,234.56 CAD" / "1,234.56" / "1234.56".
_MONEY_RE = re.compile(
    r"\$\s*\d[\d,]*(?:\.\d+)?"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:CAD|cad|dollars)"
    r"|\d{1,3}(?:,\d{3})+(?:\.\d{2})?"
    r"|\d+\.\d{2}\b"
)

# Beyond this many attribution values the subset-sum enumeration is skipped and
# only single values (and the full total) are accepted as traceable. 2**16 is
# already 65k combinations; the real snapshot carries 13.
_MAX_SUBSET_VALUES = 16

_warned_missing: set[str] = set()


def _load_grounding(task_name: str) -> dict[str, Any]:
    """Read the snapshot's grounding block by task short-name.

    A missing grounding block would score every output against nothing, which
    reads as a passing grade, so say so once per task rather than degrade
    quietly -- the same failure `_load_rubric` guards against next door.
    """
    fp = eval_tasks_path("data", "taxes", f"taxes_{task_name}.sanitized.json")
    if not fp.exists():
        if task_name not in _warned_missing:
            _warned_missing.add(task_name)
            print(
                f"{WARN} No grounding for taxes task '{task_name}' at {fp} — "
                "scores for it are not grounded",
                file=sys.stderr,
            )
        return {}
    return json.loads(fp.read_text(encoding="utf-8")).get("grounding") or {}


def _parse_output(raw: str) -> Tuple[Any, str]:
    """Return (parsed, error). Tolerates a markdown fence the prompt forbade.

    Fence-stripping is deliberate: emitting ```json is a formatting slip, and
    charging it the whole schema component would drown the arithmetic signal
    these tasks exist to measure. The slip is still named in the reason string.
    """
    text = (raw or "").strip()
    if not text:
        return None, "empty output"
    fenced = False
    if text.startswith("```"):
        fenced = True
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text), ("fenced" if fenced else "")
    except (json.JSONDecodeError, ValueError):
        # Fall back to the outermost {...} span; models prepend preambles.
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group()), "extracted-from-prose"
            except (json.JSONDecodeError, ValueError):
                pass
        return None, "not-json"


def _cents(value: Any) -> float | None:
    """Round a numeric to cents, or None when it is not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), 2)


def _prose_amounts(prose: str) -> list[float]:
    """Every money-shaped figure in prose, as positive cent-rounded floats."""
    found: list[float] = []
    for match in _MONEY_RE.findall(prose or ""):
        cleaned = re.sub(r"[^\d.]", "", match)
        if not cleaned or cleaned.count(".") > 1:
            continue
        try:
            found.append(abs(round(float(cleaned), 2)))
        except ValueError:
            continue
    return found


def _known_set(known_amounts: Iterable[Any]) -> set[float]:
    """Known amounts as positive cent-rounded floats, for membership tests."""
    out: set[float] = set()
    for amount in known_amounts or []:
        cent = _cents(amount)
        if cent is not None:
            out.add(abs(cent))
    return out


def _score_prose_amounts(prose: str, known: set[float], weight: int) -> Tuple[int, str]:
    """Proportional credit for prose dollar figures that trace to known amounts.

    No figures at all earns full marks: quoting nothing invents nothing. The
    slip_qa empty-flags case is the one place that is wrong, and it is scored
    by its own rule rather than here.
    """
    amounts = _prose_amounts(prose)
    if not amounts:
        return weight, f"prose_amounts=0/0 ({weight}/{weight})"
    grounded = sum(1 for a in amounts if a in known)
    score = round(weight * grounded / len(amounts))
    return score, f"prose_amounts={grounded}/{len(amounts)} ({score}/{weight})"


def _traceable_sums(values: list[float]) -> set[float]:
    """Every value reachable as a sum of a non-empty subset of `values`."""
    sums: set[float] = set()
    if not values:
        return sums
    if len(values) > _MAX_SUBSET_VALUES:
        sums.update(values)
        sums.add(round(sum(values), 2))
        return sums
    for size in range(1, len(values) + 1):
        for combo in combinations(values, size):
            sums.add(round(sum(combo), 2))
    return sums


def validate_taxes_yoy_narrative(output: Any, source_text: str = "") -> Tuple[int, str]:
    """arithmetic_reconciliation — see grounding.rule in the snapshot.

    schema 20 | drivers traceable to tax effects 30 | drivers reconcile to the
    total within tolerance 30 | prose figures grounded 20.
    """
    grounding = _load_grounding("yoy_narrative")
    parsed, note = _parse_output(str(output or ""))
    bits = [note] if note else []

    if not isinstance(parsed, dict):
        return 0, f"schema=0/20 ({note or 'not-an-object'})"

    prose = parsed.get("prose") if isinstance(parsed.get("prose"), str) else ""
    drivers = parsed.get("drivers")
    drivers = drivers if isinstance(drivers, list) else []
    well_formed = [
        d for d in drivers if isinstance(d, dict) and _cents(d.get("delta_cad")) is not None
    ]

    schema = 0
    if prose:
        schema += 10
    if well_formed:
        schema += 10
    bits.append(f"schema={schema}/20")

    attribution = grounding.get("attribution") or {}
    effects = [
        c
        for c in (_cents(d.get("tax_effect_cad")) for d in attribution.get("drivers") or [])
        if c is not None
    ]
    rules_effect = _cents(attribution.get("rules_effect_cad"))
    if rules_effect is not None:
        effects.append(rules_effect)
    traceable = _traceable_sums(effects)

    reported = [_cents(d.get("delta_cad")) for d in well_formed]
    reported = [r for r in reported if r is not None]
    if reported and traceable:
        hits = sum(1 for r in reported if r in traceable)
        trace_score = round(30 * hits / len(reported))
        bits.append(f"traceable={hits}/{len(reported)} ({trace_score}/30)")
    else:
        trace_score = 0
        bits.append("traceable=0/0 (0/30)")

    total_delta = _cents(grounding.get("total_tax_delta"))
    tol_abs = _cents(grounding.get("tolerance_abs_cad")) or 0.0
    tol_pct = grounding.get("tolerance_pct") or 0.0
    if reported and total_delta is not None:
        tolerance = max(tol_abs, abs(total_delta * float(tol_pct)))
        error = abs(round(sum(reported), 2) - total_delta)
        if error <= tolerance:
            recon_score = 30
        else:
            # Linear decay from the tolerance edge to a full-magnitude miss.
            span = max(abs(total_delta), 1.0)
            recon_score = max(0, round(30 * (1 - (error - tolerance) / span)))
        bits.append(f"reconcile err={error:.2f} tol={tolerance:.2f} ({recon_score}/30)")
    else:
        recon_score = 0
        bits.append("reconcile=n/a (0/30)")

    prose_score, prose_note = _score_prose_amounts(
        prose, _known_set(grounding.get("known_amounts")), 20
    )
    bits.append(prose_note)

    total = schema + trace_score + recon_score + prose_score
    return min(MAX_SCORE, total), "  ".join(bits)


def validate_taxes_qa(output: Any, source_text: str = "") -> Tuple[int, str]:
    """citation_and_number_grounding — see grounding.rule in the snapshot.

    schema 20 | every cited fact_id known 40 | prose figures grounded 40.

    An empty `citations` list scores 0 for the citation component whenever the
    snapshot ships facts: the prompt only licenses `[]` when nothing in the
    facts answers the question, and this snapshot's facts do.
    """
    grounding = _load_grounding("qa")
    parsed, note = _parse_output(str(output or ""))
    bits = [note] if note else []

    if not isinstance(parsed, dict):
        return 0, f"schema=0/20 ({note or 'not-an-object'})"

    prose = parsed.get("prose") if isinstance(parsed.get("prose"), str) else ""
    citations = parsed.get("citations")
    citations = citations if isinstance(citations, list) else []

    schema = 0
    if prose:
        schema += 10
    if isinstance(parsed.get("citations"), list):
        schema += 10
    bits.append(f"schema={schema}/20")

    known_ids = {str(i) for i in grounding.get("known_fact_ids") or []}
    cited = [
        str(c.get("fact_id")) for c in citations if isinstance(c, dict) and c.get("fact_id")
    ]
    if cited:
        hits = sum(1 for c in cited if c in known_ids)
        cite_score = round(40 * hits / len(cited))
        bits.append(f"citations={hits}/{len(cited)} ({cite_score}/40)")
    elif known_ids:
        cite_score = 0
        bits.append("citations=0 (0/40, facts were available)")
    else:
        cite_score = 40
        bits.append("citations=0 (40/40, no facts to cite)")

    prose_score, prose_note = _score_prose_amounts(
        prose, _known_set(grounding.get("known_amounts")), 40
    )
    bits.append(prose_note)

    return min(MAX_SCORE, schema + cite_score + prose_score), "  ".join(bits)


def validate_taxes_slip_qa(output: Any, source_text: str = "") -> Tuple[int, str]:
    """flag_subset_and_number_grounding — see grounding.rule in the snapshot.

    schema 30 | highlighted ids a subset of known flag ids 35 | prose figures
    grounded 35.

    This snapshot's flag list is EMPTY, so the rule's last clause bites: an
    empty flags[] means prose must carry no dollar amounts at all. That makes
    this a hallucination gate -- a model that reassures the filer with an
    invented figure scores 0 on both grounded components.
    """
    grounding = _load_grounding("slip_qa")
    parsed, note = _parse_output(str(output or ""))
    bits = [note] if note else []

    if not isinstance(parsed, dict):
        return 0, f"schema=0/30 ({note or 'not-an-object'})"

    prose = parsed.get("prose") if isinstance(parsed.get("prose"), str) else ""
    ids = parsed.get("highlighted_flag_ids")
    ids = ids if isinstance(ids, list) else None

    schema = 0
    if prose:
        schema += 15
    if ids is not None:
        schema += 15
    bits.append(f"schema={schema}/30")

    known_flags = {str(i) for i in grounding.get("known_flag_ids") or []}
    reported_ids = [str(i) for i in ids or []]
    if reported_ids:
        hits = sum(1 for i in reported_ids if i in known_flags)
        flag_score = round(35 * hits / len(reported_ids))
        bits.append(f"flags={hits}/{len(reported_ids)} ({flag_score}/35)")
    else:
        # No ids claimed is correct precisely when none were available.
        flag_score = 35 if not known_flags else 0
        bits.append(f"flags=0 claimed, {len(known_flags)} known ({flag_score}/35)")

    known_amounts = _known_set(grounding.get("known_amounts"))
    amounts = _prose_amounts(prose)
    if not known_amounts:
        # The empty-flags clause: any figure at all is unsourced by definition.
        num_score = 35 if not amounts else 0
        bits.append(f"prose_amounts={len(amounts)} with none sourceable ({num_score}/35)")
    else:
        num_score, prose_note = _score_prose_amounts(prose, known_amounts, 35)
        bits.append(prose_note)

    word_count = len(prose.split())
    if word_count >= 30:
        bits.append(f"note: prose is {word_count} words, prompt asked for under 30")

    return min(MAX_SCORE, schema + flag_score + num_score), "  ".join(bits)


__all__ = [
    "validate_taxes_qa",
    "validate_taxes_slip_qa",
    "validate_taxes_yoy_narrative",
]
