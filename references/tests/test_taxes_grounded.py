"""Winnability gate for the three grounded taxes tasks.

These snapshots ship no `rubric`, so `grounding` is the only signal. If the
arithmetic in `taxes_grounded.py` is wrong there is no second opinion to
disagree with it -- which is exactly why every ideal answer below is built FROM
the grounding block rather than typed out by hand, and asserted to score 100
BEFORE any model score from these tasks is trusted.

Each ideal is paired with mutants that perturb one grounded quantity. The
mutant assertions name the component that must drop, not just "< 100": a bound
that a mutant can satisfy cannot see the change it was written to catch.
"""

from __future__ import annotations

import json
import re

import lib.validators.taxes_grounded as tg
import pytest
from lib.paths import eval_tasks_path
from lib.validators.taxes_grounded import (
    validate_taxes_qa,
    validate_taxes_slip_qa,
    validate_taxes_yoy_narrative,
)


def _grounding(name: str) -> dict:
    fp = eval_tasks_path("data", "taxes", f"taxes_{name}.sanitized.json")
    return json.loads(fp.read_text(encoding="utf-8"))["grounding"]


def _money(value: float) -> str:
    return f"${value:,.2f}"


# --------------------------------------------------------------------------
# yoy_narrative -- arithmetic_reconciliation
# --------------------------------------------------------------------------


def _ideal_yoy() -> dict:
    """Top four tax effects + the rules effect + one grouped remainder.

    Six drivers keeps the prompt's 3-6 band while still summing to the exact
    total: the rule allows a driver whose delta_cad is a SUM of tax effects,
    which is the only way a bounded driver list can reconcile.
    """
    g = _grounding("yoy_narrative")
    attribution = g["attribution"]
    effects = [d["tax_effect_cad"] for d in attribution["drivers"]]
    ordered = sorted(effects, key=lambda v: -abs(v))
    top, rest = ordered[:4], ordered[4:]

    drivers = [
        {"label": f"line-level driver {i}", "delta_cad": v, "note": "root cause"}
        for i, v in enumerate(top)
    ]
    drivers.append(
        {
            "label": "bracket and indexation changes",
            "delta_cad": attribution["rules_effect_cad"],
            "note": "belongs to no single line",
        }
    )
    drivers.append(
        {
            "label": "remaining lines, grouped",
            "delta_cad": round(sum(rest), 2),
            "note": "root cause for the smaller lines",
        }
    )

    quoted = [a for a in g["known_amounts"] if a > 1000][:3]
    prose = (
        "The filer's total tax moved for three reasons this year. "
        + " ".join(f"One component was {_money(a)}." for a in quoted)
        + " Each figure is taken from the supplied diff."
    )
    return {"prose": prose, "drivers": drivers}


def test_yoy_ideal_answer_built_from_grounding_scores_100():
    score, reason = validate_taxes_yoy_narrative(json.dumps(_ideal_yoy()))
    assert score == 100, f"ideal answer is not winnable: {score} — {reason}"


def test_yoy_untraceable_driver_drops_the_traceable_component():
    answer = _ideal_yoy()
    answer["drivers"][0]["delta_cad"] = 12345.67  # not any tax effect, nor a sum
    score, reason = validate_taxes_yoy_narrative(json.dumps(answer))
    assert "traceable=5/6" in reason, reason
    assert score < 100


def test_yoy_drivers_that_do_not_reconcile_lose_the_reconcile_component():
    """Drop the grouped remainder: every driver is still individually
    traceable, so only the sum-to-total check can catch this."""
    answer = _ideal_yoy()
    del answer["drivers"][-1]
    score, reason = validate_taxes_yoy_narrative(json.dumps(answer))
    assert "traceable=5/5" in reason, reason
    assert "reconcile err=" in reason
    assert score < 100


def test_yoy_invented_prose_figure_drops_the_prose_component():
    answer = _ideal_yoy()
    answer["prose"] += " We also note $999,999.99 of unexplained movement."
    score, reason = validate_taxes_yoy_narrative(json.dumps(answer))
    assert "prose_amounts=3/4" in reason, reason
    assert score < 100


# --------------------------------------------------------------------------
# qa -- citation_and_number_grounding
# --------------------------------------------------------------------------


def _ideal_qa() -> dict:
    g = _grounding("qa")
    ids = list(g["known_fact_ids"])[:3]
    quoted = [a for a in g["known_amounts"] if a > 1000][:2]
    prose = (
        "Your refund is smaller because the underlying lines moved. "
        + " ".join(f"One of them is {_money(a)}." for a in quoted)
    )
    return {
        "prose": prose,
        "citations": [{"fact_id": i, "note": "supports the explanation"} for i in ids],
    }


def test_qa_ideal_answer_built_from_grounding_scores_100():
    score, reason = validate_taxes_qa(json.dumps(_ideal_qa()))
    assert score == 100, f"ideal answer is not winnable: {score} — {reason}"


def test_qa_invented_fact_id_drops_the_citation_component():
    answer = _ideal_qa()
    answer["citations"][0]["fact_id"] = "t1.2025.invented.line"
    score, reason = validate_taxes_qa(json.dumps(answer))
    assert "citations=2/3" in reason, reason
    assert score < 100


def test_qa_empty_citations_score_zero_when_facts_were_available():
    answer = _ideal_qa()
    answer["citations"] = []
    score, reason = validate_taxes_qa(json.dumps(answer))
    assert "facts were available" in reason, reason
    assert score == 60  # schema 20 + citations 0 + prose 40


def test_qa_invented_prose_figure_drops_the_prose_component():
    answer = _ideal_qa()
    answer["prose"] += " That works out to $123,456.78 overall."
    score, reason = validate_taxes_qa(json.dumps(answer))
    assert "prose_amounts=2/3" in reason, reason
    assert score < 100


# --------------------------------------------------------------------------
# slip_qa -- flag_subset_and_number_grounding (the empty-flags case)
# --------------------------------------------------------------------------


def _ideal_slip_qa() -> dict:
    return {
        "prose": "No issues were found across your slips for this tax year.",
        "highlighted_flag_ids": [],
    }


def test_slip_qa_snapshot_really_is_the_empty_flags_case():
    """The hallucination-gate reading of this task depends on it."""
    g = _grounding("slip_qa")
    assert g["flags"] == []
    assert g["known_flag_ids"] == []
    assert g["known_amounts"] == []


def test_slip_qa_ideal_answer_scores_100():
    score, reason = validate_taxes_slip_qa(json.dumps(_ideal_slip_qa()))
    assert score == 100, f"ideal answer is not winnable: {score} — {reason}"


def test_slip_qa_any_dollar_figure_is_unsourced_and_scores_zero_there():
    answer = _ideal_slip_qa()
    answer["prose"] = "No issues found; your $1,234.56 of credits all reconcile."
    score, reason = validate_taxes_slip_qa(json.dumps(answer))
    assert "none sourceable (0/35)" in reason, reason
    assert score == 65  # schema 30 + flags 35 + numbers 0


def test_slip_qa_invented_flag_id_scores_zero_on_flags():
    answer = _ideal_slip_qa()
    answer["highlighted_flag_ids"] = ["slip.t4.mismatch"]
    score, reason = validate_taxes_slip_qa(json.dumps(answer))
    assert "flags=0/1" in reason, reason
    assert score == 65  # schema 30 + flags 0 + numbers 35


def test_slip_qa_over_long_prose_is_reported_but_not_scored():
    """The prompt asks for under 30 words; no grounding field adjudicates it,
    so it is named in the reason and left out of the score deliberately."""
    answer = _ideal_slip_qa()
    answer["prose"] = " ".join(["word"] * 40)
    score, reason = validate_taxes_slip_qa(json.dumps(answer))
    assert "40 words" in reason, reason
    assert score == 100


# --------------------------------------------------------------------------
# shared parsing behaviour
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "validator",
    [validate_taxes_yoy_narrative, validate_taxes_qa, validate_taxes_slip_qa],
)
def test_non_json_output_scores_zero(validator):
    score, reason = validator("I cannot answer that.")
    assert score == 0
    assert "not-json" in reason


def test_fenced_json_is_still_graded_but_the_slip_is_named():
    fenced = "```json\n" + json.dumps(_ideal_slip_qa()) + "\n```"
    score, reason = validate_taxes_slip_qa(fenced)
    assert score == 100
    assert "fenced" in reason


# --------------------------------------------------------------------------
# Defensive branches. Each of these is a path a real model output can reach,
# so they are pinned rather than left to the 95% floor to notice.
# --------------------------------------------------------------------------


class TestGroundingLoad:
    def test_a_missing_snapshot_warns_once_and_grounds_nothing(self, capsys, monkeypatch):
        """A silently-empty grounding block scores every output against nothing,
        which reads as a passing grade. It must say so."""
        monkeypatch.setattr(tg, "_warned_missing", set())
        assert tg._load_grounding("no_such_task") == {}
        first = capsys.readouterr().err
        assert "No grounding" in first and "no_such_task" in first

        assert tg._load_grounding("no_such_task") == {}
        assert capsys.readouterr().err == "", "the warning repeated per call"


class TestOutputParsing:
    def test_empty_output_is_named_as_empty_not_as_bad_json(self):
        parsed, note = tg._parse_output("   ")
        assert parsed is None
        assert note == "empty output"

    def test_json_buried_in_a_preamble_is_still_graded(self):
        raw = 'Sure! Here is the answer:\n{"prose": "x", "highlighted_flag_ids": []}'
        parsed, note = tg._parse_output(raw)
        assert parsed == {"prose": "x", "highlighted_flag_ids": []}
        assert note == "extracted-from-prose"

    def test_a_brace_span_that_is_not_json_falls_through_to_not_json(self):
        parsed, note = tg._parse_output("preamble {not: valid, json} trailer")
        assert parsed is None
        assert note == "not-json"


class TestNumericHelpers:
    @pytest.mark.parametrize("value", ["12.50", None, True, False, {"a": 1}])
    def test_non_numbers_are_not_treated_as_amounts(self, value):
        """`True` is an int in Python; letting it through would score a boolean
        delta_cad as 1.0 rather than rejecting the field."""
        assert tg._cents(value) is None

    def test_a_malformed_money_shape_is_skipped_not_crashed(self):
        assert tg._prose_amounts("version 1.2.3 and $10.00") == [10.0]

    def test_prose_with_no_figures_cannot_be_ungrounded(self):
        score, note = tg._score_prose_amounts("no numbers here", {1.0}, 20)
        assert score == 20
        assert "0/0" in note

    def test_traceable_sums_of_nothing_is_empty(self):
        assert tg._traceable_sums([]) == set()

    def test_too_many_values_falls_back_to_singles_and_the_full_total(self):
        """Guards the 2**n blowup. The fallback must be narrower, not wider:
        a pair sum stops being accepted once enumeration is skipped.

        Powers of two on purpose. With 1..17 the pair sum 1+2 equals the single
        value 3, so the assertion below passed for the wrong reason and could
        not see the branch it was written to catch.
        """
        values = [2.0**i for i in range(tg._MAX_SUBSET_VALUES + 1)]
        sums = tg._traceable_sums(values)
        assert values[0] in sums
        assert round(sum(values), 2) in sums
        assert values[0] + values[1] not in sums


class TestDegenerateModelOutputs:
    def test_yoy_with_no_usable_drivers_loses_both_arithmetic_components(self):
        answer = {"prose": "The tax fell.", "drivers": []}
        score, reason = validate_taxes_yoy_narrative(json.dumps(answer))
        assert "traceable=0/0 (0/30)" in reason
        assert "reconcile=n/a (0/30)" in reason
        assert score == 30  # schema 10 (prose only) + prose_amounts 20

    def test_qa_with_no_facts_to_cite_does_not_punish_an_empty_citation_list(
        self, monkeypatch
    ):
        """The prompt licenses `[]` when nothing in the facts answers the
        question. Scoring that 0 would make the honest answer unwinnable."""
        monkeypatch.setattr(
            tg, "_load_grounding", lambda name: {"known_fact_ids": [], "known_amounts": []}
        )
        answer = {"prose": "Nothing here answers that.", "citations": []}
        score, reason = validate_taxes_qa(json.dumps(answer))
        assert "no facts to cite" in reason
        assert score == 100

    def test_slip_qa_scores_figures_proportionally_when_flags_do_exist(self, monkeypatch):
        """The empty-flags clause is the snapshot's case, not the only case."""
        monkeypatch.setattr(
            tg,
            "_load_grounding",
            lambda name: {"known_flag_ids": ["f1"], "known_amounts": [100.0]},
        )
        answer = {
            "prose": "One slip shows $100.00, another shows $250.00.",
            "highlighted_flag_ids": ["f1"],
        }
        score, reason = validate_taxes_slip_qa(json.dumps(answer))
        assert "prose_amounts=1/2" in reason
        assert score == 83  # schema 30 + flags 35 + round(35 * 1/2) = 18


class TestMoneyGuardsSurviveARegexChange:
    """`_prose_amounts` cleans a regex match to digits-and-dots before float().

    Fuzzing 300k inputs shows the CURRENT `_MONEY_RE` can never yield a span
    that fails to parse, so these two guards are unreachable through it. They
    are kept, and tested here through a substituted pattern, because the failure
    they prevent is invisible: `eval/run.py` wraps validators in a blanket
    except, so an uncaught ValueError becomes a permanent score of 0 rather than
    a crash anyone would notice. Widening the pattern must not be able to do
    that silently.
    """

    def test_a_span_with_several_dots_is_skipped(self, monkeypatch):
        monkeypatch.setattr(tg, "_MONEY_RE", re.compile(r"[\d.]+"))
        assert tg._prose_amounts("version 1.2.3 here") == []

    def test_a_span_that_is_not_a_number_is_skipped_not_raised(self, monkeypatch):
        monkeypatch.setattr(tg, "_MONEY_RE", re.compile(r"[.]"))
        assert tg._prose_amounts("a sentence. and another.") == []
