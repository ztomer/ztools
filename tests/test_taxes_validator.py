"""Tests for the taxes_* validators (ported from
github.com/ztomer/Taxes — see lib/validators/taxes_validator.py).

Pins:
  - Substantive prose with 5+ signals + $ amounts scores high.
  - List-form-only / thin output loses substance points.
  - GT-leak terms drop no_leak to 0 regardless of other axes.
  - audit_readiness halves score on non-JSON output.
  - synthesis dings missing markdown section headers.
"""

from __future__ import annotations

import pytest

from lib.validators.taxes_validator import (
    validate_taxes_anomalies,
    validate_taxes_audit_readiness,
    validate_taxes_synthesis,
)


# 600+ chars of prose hitting 5 signals with $ amounts
_GOOD_PROSE = (
    "The 2025 cross-border filing has several items worth flagging. "
    "T1135 foreign-property reporting is triggered because the IBKR "
    "brokerage holdings exceeded $100,000 CAD at year-end. We see "
    "specifically $XXX,XXX in US securities + an ILS-denominated fund "
    "valued at $XX,XXX CAD that the filer should disclose. "
    "Form 106 reconciliation against the T4 box 38 RSU vest values "
    "shows a $X,XXX delta the accountant will ask about — likely the "
    "prior-year true-up at the higher Israeli rate. "
    "Quarterly tax payments to the Israeli tax authority for the "
    "rental income ($XX,XXX gross, $X,XXX expenses) are documented in "
    "the rental statement; verify the four remittance receipts match "
    "the reported total before filing. "
    "Spousal split for the medical expense claim looks reasonable "
    "given the income split. No prior baseline is available — this is "
    "the first multi-year run, so anomaly detection has reduced "
    "sensitivity for YoY drift."
)


def test_anomalies_substantive_prose_scores_high():
    score, reason = validate_taxes_anomalies(_GOOD_PROSE)
    assert score >= 90, f"Expected ≥90, got {score}. Reason: {reason}"
    assert "grounding=40/40" in reason


def test_anomalies_thin_list_form_low_substance():
    thin = "1. T1135\n2. Form 106\n3. quarterly tax"
    score, reason = validate_taxes_anomalies(thin)
    # 3 signals → 24 grounding; no_leak 30; substance 0 (thin+no$+list)
    assert score == 54, f"Expected 54, got {score}. Reason: {reason}"
    assert "substance=0" in reason


def test_anomalies_gt_leak_zeroes_no_leak():
    leaky = ("Filed (GT): $XX,XXX refund. " + _GOOD_PROSE)
    score, reason = validate_taxes_anomalies(leaky)
    assert "no_leak=0" in reason
    assert "GT-flavored term leaked" in reason


def test_audit_readiness_valid_json_keeps_score():
    """Pure JSON output with `risk_items` list → schema=ok, no halving."""
    import json
    items = [
        {"title": f"T1135 finding {i}",
         "severity": "high",
         "rationale": (
             "Foreign property exceeds $XXX,XXX CAD threshold without "
             "T1135 disclosure. Form 106 + IBKR statements both show "
             "balance above $XX,XXX. Box 38 RSU vest values reconcile "
             "to T4. Quarterly tax compliance verified. No prior "
             "baseline for cross-year comparison."),
         "documents": ["IBKR statement", "Form 106"],
         "mitigation": "file T1135 with return"}
        for i in range(2)
    ]
    payload = json.dumps({"risk_items": items}, indent=2)
    assert len(payload) > 600  # confirm substance threshold met
    score, reason = validate_taxes_audit_readiness(payload)
    assert "schema=ok" in reason, f"Reason: {reason}"
    # Score should NOT be halved
    assert score > 50, f"Expected unhalved (>50), got {score}. Reason: {reason}"


def test_audit_readiness_non_json_halves_score():
    """Plain prose for audit_readiness → schema fail → score halved."""
    score, reason = validate_taxes_audit_readiness(_GOOD_PROSE)
    assert "score halved" in reason
    # _GOOD_PROSE scores 90+ as anomalies; halved = ~45
    assert score < 60, f"Expected <60 after halving, got {score}"


def test_audit_readiness_wrong_json_shape_halves():
    """Valid JSON without `risk_items` list → still halved."""
    score, reason = validate_taxes_audit_readiness('{"foo": "bar"}')
    assert "schema=bad-shape" in reason or "score halved" in reason


def test_synthesis_missing_sections_dings():
    """Synthesis expects 5 markdown section headers; missing ≥2 → -10."""
    short = "Just a paragraph with T1135 and Form 106 and box 38 mentions. " * 15
    score, reason = validate_taxes_synthesis(short)
    assert "sections=0/5" in reason or "−10" in reason


def test_synthesis_with_all_sections_no_ding():
    body = (
        "**1. Missing Documents**: T1135 disclosure pending.\n"
        "**2. Estimated Tax Impact**: $X,XXX refund, $XX,XXX FTC.\n"
        "**3. Top 5 Action Items**: file T1135; reconcile Form 106; "
        "verify quarterly tax; check box 38 vs vesting; confirm no prior.\n"
        "**4. Key Risks**: late T1135 → daily penalty.\n"
        "**5. Timeline**: April 30 filing deadline.\n"
    ) * 3  # pad for substance
    score, reason = validate_taxes_synthesis(body)
    assert "sections=5/5" in reason
    assert "−10" not in reason


def test_empty_output_scores_zero():
    """Empty output → 0 grounding, but no_leak=30 (vacuously true)."""
    score, reason = validate_taxes_anomalies("")
    assert "grounding=0/40" in reason
    # Only no_leak (30) fires; substance and grounding both 0
    assert score == 30


def test_load_rubric_missing_file(tmp_path, monkeypatch):
    """When the rubric file doesn't exist, _load_rubric returns {}."""
    import lib.validators.taxes_validator as tv
    # Patch the data_dir to a path that doesn't have the file
    monkeypatch.setattr(tv, "Path", lambda *a, **kw: tmp_path)
    from lib.validators.taxes_validator import _load_rubric
    result = _load_rubric("nonexistent_task")
    assert result == {}


def test_grounding_score_empty_signals():
    """When expected_signals is empty, full 40 score."""
    from lib.validators.taxes_validator import _grounding_score
    score, hits = _grounding_score("any output", [])
    assert score == 40
    assert hits == 0
