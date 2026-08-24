use super::*;
use serde_json::json;

#[test]
fn test_prose_amounts_extraction() {
    let prose = "The balance was $1,234.56 with additional 500.00 CAD and $50.";
    let amounts = prose_amounts(prose);
    assert_eq!(amounts, vec![1234.56, 500.0, 50.0]);
}

#[test]
fn test_traceable_sums_enumeration() {
    let values = vec![10.0, 20.0, 30.0];
    let sums = traceable_sums(&values);
    assert!(sums.contains(&1000));
    assert!(sums.contains(&3000)); // 10+20 or 30
    assert!(sums.contains(&5000)); // 20+30
    assert!(sums.contains(&6000)); // 10+20+30
    assert!(!sums.contains(&7000));
}

#[test]
fn test_validate_taxes_slip_qa_empty_flags_rewarded() {
    let output = json!({
        "prose": "Everything is consistent with T4 slips.",
        "highlighted_flag_ids": []
    });
    let grounding = json!({
        "known_flag_ids": [],
        "known_amounts": []
    });
    let (score, note) = validate_taxes_slip_qa(&output, Some(&grounding));
    assert_eq!(score, 100);
    assert!(note.contains("flags=0 claimed, 0 known (35/35)"));
}

#[test]
fn test_cents_variants_and_rounding() {
    assert_eq!(cents(&json!(10)), Some(10.0));
    assert_eq!(cents(&json!(12.346)), Some(12.35));
    assert_eq!(cents(&json!("3.75")), Some(3.75));
    assert_eq!(cents(&json!("   ")), None);
    assert_eq!(cents(&json!("not a number")), None);
    assert_eq!(cents(&json!(null)), None);
    assert_eq!(cents(&json!({"a": 1})), None);
}

#[test]
fn test_known_set_ignores_invalid_and_uses_absolute_value() {
    let set = known_set(&[
        json!(10),
        json!("2.5"),
        json!("bad"),
        json!(-3),
        json!(true),
    ]);
    assert_eq!(set.len(), 3);
    assert!(set.contains(&1000));
    assert!(set.contains(&250));
    assert!(set.contains(&300));
}

#[test]
fn test_score_prose_amounts_zero_and_partial_grounding() {
    let known = known_set(&[json!(50)]);
    let (score, note) = score_prose_amounts("no amounts here", &known, 20);
    assert_eq!(score, 20);
    assert_eq!(note, "prose_amounts=0/0 (20/20)");

    // $50 is grounded, $75 is not: half credit
    let (score, note) = score_prose_amounts("costs $50 plus $75 in fees", &known, 20);
    assert_eq!(score, 10);
    assert_eq!(note, "prose_amounts=1/2 (10/20)");
}

#[test]
fn test_traceable_sums_empty_input() {
    assert!(traceable_sums(&[]).is_empty());
}

#[test]
fn test_traceable_sums_large_input_only_individuals_and_total() {
    let values: Vec<f64> = (0..=MAX_SUBSET_VALUES)
        .map(|i| 100.0 + i as f64)
        .collect();
    let sums = traceable_sums(&values);
    assert!(sums.contains(&10000)); // first element alone
    assert!(sums.contains(&11600)); // last element alone
    assert!(sums.contains(&183600)); // grand total 1836.00
    // pair sums are NOT enumerated past MAX_SUBSET_VALUES (101+102 = 203.00)
    assert!(!sums.contains(&20300));
}

#[test]
fn test_yoy_narrative_fenced_json_partial_reconciliation() {
    let output = json!(
        "```json\n{\"prose\":\"Net change was $50.\",\"drivers\":[{\"delta_cad\":-150},{\"delta_cad\":50}]}\n```"
    );
    let grounding = json!({
        "attribution": {
            "drivers": [{"tax_effect_cad": -150}, {"tax_effect_cad": 50}],
            "rules_effect_cad": -25
        },
        "total_tax_delta": -175,
        "tolerance_abs_cad": 5,
        "tolerance_pct": 0.02,
        "known_amounts": [50]
    });
    let (score, note) = validate_taxes_yoy_narrative(&output, Some(&grounding));
    // schema 20 + traceable 30 + recon 18 + prose 20
    assert_eq!(score, 88);
    assert!(note.contains("fenced"), "note was: {note}");
    assert!(note.contains("schema=20/20"), "note was: {note}");
    assert!(note.contains("traceable=2/2 (30/30)"), "note was: {note}");
    assert!(
        note.contains("reconcile err=75.00 tol=5.00 (18/30)"),
        "note was: {note}"
    );
    assert!(note.contains("prose_amounts=1/1 (20/20)"), "note was: {note}");
}

#[test]
fn test_yoy_narrative_exact_reconciliation_scores_full() {
    let output = json!({"prose": "text", "drivers": [{"delta_cad": -175}]});
    let grounding = json!({
        "attribution": {
            "drivers": [{"tax_effect_cad": -150}],
            "rules_effect_cad": -25
        },
        "total_tax_delta": -175,
        "tolerance_abs_cad": 5
    });
    // -175 is traceable as (-150)+(-25); error 0 is inside tolerance 5
    let (score, note) = validate_taxes_yoy_narrative(&output, Some(&grounding));
    assert_eq!(score, 100);
    assert!(
        note.contains("reconcile err=0.00 tol=5.00 (30/30)"),
        "note was: {note}"
    );
}

#[test]
fn test_yoy_narrative_reconciliation_error_clamps_at_zero() {
    let output = json!({"prose": "text", "drivers": [{"delta_cad": 1000}]});
    let grounding = json!({
        "attribution": {"rules_effect_cad": 0},
        "total_tax_delta": -175,
        "tolerance_abs_cad": 5
    });
    let (score, note) = validate_taxes_yoy_narrative(&output, Some(&grounding));
    // scaled penalty goes negative and floors at 0: schema 20 + trace 0 + recon 0 + prose 20
    assert_eq!(score, 40);
    assert!(
        note.contains("reconcile err=1175.00 tol=5.00 (0/30)"),
        "note was: {note}"
    );
    assert!(note.contains("traceable=0/1 (0/30)"), "note was: {note}");
}

#[test]
fn test_yoy_narrative_empty_report_and_grounding() {
    let output = json!({"prose": "", "drivers": []});
    let grounding = json!({});
    let (score, note) = validate_taxes_yoy_narrative(&output, Some(&grounding));
    assert_eq!(score, 20);
    assert!(note.contains("schema=0/20"), "note was: {note}");
    assert!(note.contains("traceable=0/0 (0/30)"), "note was: {note}");
    assert!(note.contains("reconcile=n/a (0/30)"), "note was: {note}");
    assert!(note.contains("prose_amounts=0/0 (20/20)"), "note was: {note}");
}

#[test]
fn test_yoy_narrative_filters_malformed_drivers_and_effects() {
    let output = json!({"prose": "x", "drivers": [
        5, {"delta_cad": "abc"}, {}, {"delta_cad": -100}
    ]});
    let grounding = json!({"attribution": {"drivers": [{"tax_effect_cad": "nope"}]}});
    let (score, note) = validate_taxes_yoy_narrative(&output, Some(&grounding));
    // only delta_cad=-100 survives; attribution effect is unusable -> no
    // traceable sums, no reconciliation target
    assert_eq!(score, 40);
    assert!(note.contains("traceable=0/0 (0/30)"), "note was: {note}");
    assert!(note.contains("reconcile=n/a (0/30)"), "note was: {note}");
    assert!(note.contains("schema=20/20"), "note was: {note}");
}

#[test]
fn test_yoy_narrative_unparseable_output_scores_zero() {
    assert_eq!(
        validate_taxes_yoy_narrative(&json!(""), None),
        (0, "schema=0/20 (empty output)".to_string())
    );
}

#[test]
fn test_taxes_qa_citation_scoring_paths() {
    let grounding = json!({"known_fact_ids": ["f1", "f2"], "known_amounts": [10]});
    let output = json!({"prose": "about $10",
                        "citations": [{"fact_id": "f1"}, {"fact_id": "zz"}]});
    let (score, note) = validate_taxes_qa(&output, Some(&grounding));
    // schema 20 + citations 1/2 -> 20 + prose fully grounded 40
    assert_eq!(score, 80);
    assert!(note.contains("citations=1/2 (20/40)"), "note was: {note}");

    // entries without a usable fact_id are dropped before scoring
    let output = json!({"prose": "",
                        "citations": [{"fact_id": "f1"}, {"other": 1}, {"fact_id": 42}]});
    let grounding = json!({"known_fact_ids": ["f1"], "known_amounts": []});
    let (score, _) = validate_taxes_qa(&output, Some(&grounding));
    assert_eq!(score, 10 + 40 + 40);

    // citations present but every id unknown
    let grounding = json!({"known_amounts": []});
    let output = json!({"prose": "", "citations": [{"fact_id": "x"}]});
    let (score, note) = validate_taxes_qa(&output, Some(&grounding));
    assert_eq!(score, 10 + 40);
    assert!(note.contains("citations=0/1 (0/40)"), "note was: {note}");

    // no citations while facts were available
    let grounding = json!({"known_fact_ids": ["f1"], "known_amounts": [10]});
    let output = json!({"prose": "about $10"});
    let (score, note) = validate_taxes_qa(&output, Some(&grounding));
    assert_eq!(score, 10 + 40);
    assert!(
        note.contains("citations=0 (0/40, facts were available)"),
        "note was: {note}"
    );

    // no citations and no facts to cite: citation slot awarded anyway
    let output = json!({"prose": ""});
    let (score, note) = validate_taxes_qa(&output, Some(&json!({})));
    assert_eq!(score, 40 + 40);
    assert!(
        note.contains("citations=0 (40/40, no facts to cite)"),
        "note was: {note}"
    );
}

#[test]
fn test_taxes_qa_extracts_json_from_prose() {
    let output = json!("Sure! Here: {\"prose\":\"costs $5\"} end");
    let grounding = json!({"known_amounts": [5]});
    let (score, note) = validate_taxes_qa(&output, Some(&grounding));
    assert_eq!(score, 10 + 40 + 40);
    assert!(note.starts_with("extracted-from-prose"), "note was: {note}");
}

#[test]
fn test_taxes_qa_non_object_and_empty_outputs_fail_closed() {
    // array parses as JSON but is not an object
    assert_eq!(
        validate_taxes_qa(&json!([1]), None),
        (0, "schema=0/20 (not-an-object)".to_string())
    );
    // braces reversed -> extraction impossible
    assert_eq!(
        validate_taxes_qa(&json!("} reverse {"), None),
        (0, "schema=0/20 (not-json)".to_string())
    );
    // opening brace with no closer
    assert_eq!(
        validate_taxes_qa(&json!("junk {\"x\": 1"), None),
        (0, "schema=0/20 (not-json)".to_string())
    );
    // braces present but the span between them is not JSON
    assert_eq!(
        validate_taxes_qa(&json!("junk {\"x\": } tail"), None),
        (0, "schema=0/20 (not-json)".to_string())
    );
    assert_eq!(
        validate_taxes_qa(&json!(""), None),
        (0, "schema=0/20 (empty output)".to_string())
    );
}

#[test]
fn test_taxes_slip_qa_flag_scoring_paths() {
    let grounding = json!({"known_flag_ids": ["k1", "k2"], "known_amounts": [20]});
    let output = json!({"prose": "see the $20 line",
                        "highlighted_flag_ids": ["k1", "zz"]});
    let (score, note) = validate_taxes_slip_qa(&output, Some(&grounding));
    // schema 30 + flags 1/2 -> round(17.5)=18 + prose fully grounded 35
    assert_eq!(score, 83);
    assert!(note.contains("flags=1/2 (18/35)"), "note was: {note}");

    // flags claimed while none were known: flag slot lost
    let grounding = json!({"known_flag_ids": [], "known_amounts": [5]});
    let output = json!({"prose": "x", "highlighted_flag_ids": ["anything"]});
    let (score, _) = validate_taxes_slip_qa(&output, Some(&grounding));
    assert_eq!(score, 30 + 35);

    // flags omitted while known flags existed; empty prose forfeits its slot too
    let grounding = json!({"known_flag_ids": ["k1"]});
    let output = json!({"prose": ""});
    let (score, note) = validate_taxes_slip_qa(&output, Some(&grounding));
    assert_eq!(score, 35);
    assert!(
        note.contains("flags=0 claimed, 1 known (0/35)"),
        "note was: {note}"
    );

    // flags omitted, none known, but prose invents an amount
    let grounding = json!({"known_flag_ids": [], "known_amounts": []});
    let output = json!({"prose": "costs $9"});
    let (score, note) = validate_taxes_slip_qa(&output, Some(&grounding));
    assert_eq!(score, (15 + 35));
    assert!(
        note.contains("prose_amounts=1 with none sourceable (0/35)"),
        "note was: {note}"
    );
}

#[test]
fn test_taxes_validators_load_grounding_without_explicit_input() {
    // exercises the load_grounding seam (missing task data -> Null grounding);
    // the outputs below fail parsing, so the verdict is independent of whatever
    // the on-disk sanitized fixtures contain
    assert_eq!(
        validate_taxes_yoy_narrative(&json!(""), None),
        (0, "schema=0/20 (empty output)".to_string())
    );
    assert_eq!(
        validate_taxes_slip_qa(&json!(""), None),
        (0, "schema=0/30 (empty output)".to_string())
    );
}
