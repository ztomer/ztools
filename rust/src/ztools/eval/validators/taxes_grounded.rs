//! Grounded arithmetic and citation validation for taxes tasks.
//!
//! Port of `lib/validators/taxes_grounded.py`.

use regex::Regex;
use serde_json::Value;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

pub const MAX_SCORE: i64 = 100;
pub const MAX_SUBSET_VALUES: usize = 16;

static MONEY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?x)
        \$\s*\d[\d,]*(?:\.\d+)?
        | \d[\d,]*(?:\.\d+)?\s*(?:CAD|cad|dollars)
        | \d{1,3}(?:,\d{3})+(?:\.\d{2})?
        | \d+\.\d{2}\b
    ",
    )
    .unwrap()
});

pub fn cents(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64().map(|f| (f * 100.0).round() / 100.0),
        Value::String(s) => s
            .trim()
            .parse::<f64>()
            .ok()
            .map(|f| (f * 100.0).round() / 100.0),
        _ => None,
    }
}

pub fn prose_amounts(prose: &str) -> Vec<f64> {
    let mut found = Vec::new();
    for mat in MONEY_RE.find_iter(prose) {
        let cleaned: String = mat
            .as_str()
            .chars()
            .filter(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        if cleaned.is_empty() || cleaned.matches('.').count() > 1 {
            continue;
        }
        if let Ok(f) = cleaned.parse::<f64>() {
            let val = (f * 100.0).round() / 100.0;
            found.push(val.abs());
        }
    }
    found
}

pub fn known_set(known_amounts: &[Value]) -> HashSet<i64> {
    let mut out = HashSet::new();
    for val in known_amounts {
        if let Some(c) = cents(val) {
            out.insert((c.abs() * 100.0).round() as i64);
        }
    }
    out
}

pub fn score_prose_amounts(prose: &str, known: &HashSet<i64>, weight: i64) -> (i64, String) {
    let amounts = prose_amounts(prose);
    if amounts.is_empty() {
        return (weight, format!("prose_amounts=0/0 ({}/{})", weight, weight));
    }
    let grounded = amounts
        .iter()
        .filter(|a| known.contains(&((a.abs() * 100.0).round() as i64)))
        .count();
    let score = (weight as f64 * grounded as f64 / amounts.len() as f64).round() as i64;
    (
        score,
        format!(
            "prose_amounts={}/{} ({}/{})",
            grounded,
            amounts.len(),
            score,
            weight
        ),
    )
}

pub fn traceable_sums(values: &[f64]) -> HashSet<i64> {
    let mut sums = HashSet::new();
    if values.is_empty() {
        return sums;
    }
    if values.len() > MAX_SUBSET_VALUES {
        for v in values {
            sums.insert((v * 100.0).round() as i64);
        }
        let total: f64 = values.iter().sum();
        sums.insert((total * 100.0).round() as i64);
        return sums;
    }

    let n = values.len();
    for mask in 1..(1usize << n) {
        let mut sub_sum = 0.0;
        for (i, v) in values.iter().enumerate() {
            if (mask & (1 << i)) != 0 {
                sub_sum += *v;
            }
        }
        sums.insert((sub_sum * 100.0).round() as i64);
    }
    sums
}

fn load_grounding(task_name: &str) -> Value {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let candidate = Path::new(manifest)
        .parent()
        .map(|p| {
            p.join("eval_tasks/data/taxes")
                .join(format!("taxes_{}.sanitized.json", task_name))
        })
        .unwrap_or_else(|| {
            PathBuf::from(format!(
                "eval_tasks/data/taxes/taxes_{}.sanitized.json",
                task_name
            ))
        });

    if let Ok(content) = std::fs::read_to_string(&candidate) {
        if let Ok(val) = serde_json::from_str::<Value>(&content) {
            if let Some(grounding) = val.get("grounding") {
                return grounding.clone();
            }
        }
    }
    Value::Null
}

fn parse_output(raw: &Value) -> (Option<Value>, String) {
    if let Value::Object(_) = raw {
        return (Some(raw.clone()), String::new());
    }
    let text = match raw {
        Value::String(s) => s.trim().to_string(),
        other => other.to_string(),
    };
    if text.is_empty() {
        return (None, "empty output".to_string());
    }
    let mut clean_text = text.as_str();
    let mut note = String::new();
    if clean_text.starts_with("```") {
        note = "fenced".to_string();
        clean_text = clean_text
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();
    }
    if let Ok(val) = serde_json::from_str::<Value>(clean_text) {
        return (Some(val), note);
    }
    if let Some(start) = clean_text.find('{') {
        if let Some(end) = clean_text.rfind('}') {
            if start < end {
                if let Ok(val) = serde_json::from_str::<Value>(&clean_text[start..=end]) {
                    return (Some(val), "extracted-from-prose".to_string());
                }
            }
        }
    }
    (None, "not-json".to_string())
}

pub fn validate_taxes_yoy_narrative(
    output: &Value,
    explicit_grounding: Option<&Value>,
) -> (i64, String) {
    let loaded = load_grounding("yoy_narrative");
    let grounding = explicit_grounding.unwrap_or(&loaded);
    let (parsed, note) = parse_output(output);
    let mut bits = if !note.is_empty() {
        vec![note]
    } else {
        Vec::new()
    };

    let map = match parsed {
        Some(Value::Object(m)) => m,
        _ => {
            return (
                0,
                format!(
                    "schema=0/20 ({})",
                    bits.first()
                        .cloned()
                        .unwrap_or_else(|| "not-an-object".to_string())
                ),
            )
        }
    };

    let prose = map.get("prose").and_then(|v| v.as_str()).unwrap_or("");
    let drivers = map
        .get("drivers")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let well_formed: Vec<&serde_json::Map<String, Value>> = drivers
        .iter()
        .filter_map(|d| {
            let m = d.as_object()?;
            if cents(m.get("delta_cad")?).is_some() {
                Some(m)
            } else {
                None
            }
        })
        .collect();

    let mut schema = 0;
    if !prose.is_empty() {
        schema += 10;
    }
    if !well_formed.is_empty() {
        schema += 10;
    }
    bits.push(format!("schema={}/20", schema));

    let attribution = grounding.get("attribution").and_then(|v| v.as_object());
    let mut effects = Vec::new();
    if let Some(attr) = attribution {
        if let Some(drivers) = attr.get("drivers").and_then(|v| v.as_array()) {
            for d in drivers {
                if let Some(c) = d.get("tax_effect_cad").and_then(cents) {
                    effects.push(c);
                }
            }
        }
        if let Some(r) = attr.get("rules_effect_cad").and_then(cents) {
            effects.push(r);
        }
    }
    let traceable = traceable_sums(&effects);

    let reported: Vec<f64> = well_formed
        .iter()
        .filter_map(|d| cents(d.get("delta_cad")?))
        .collect();
    let trace_score = if !reported.is_empty() && !traceable.is_empty() {
        // SIGNED match, like the Python original: drivers report negative
        // deltas for a tax decrease, and the attribution effects carry the same
        // sign. Abs-ing here made every honest answer untraceable (0/4 against
        // Python's 4/4 on identical output -- found by the A/B parity run).
        let hits = reported
            .iter()
            .filter(|r| traceable.contains(&((**r * 100.0).round() as i64)))
            .count();
        let s = (30.0 * hits as f64 / reported.len() as f64).round() as i64;
        bits.push(format!("traceable={}/{} ({}/30)", hits, reported.len(), s));
        s
    } else {
        bits.push("traceable=0/0 (0/30)".to_string());
        0
    };

    let total_delta = grounding.get("total_tax_delta").and_then(cents);
    let tol_abs = grounding
        .get("tolerance_abs_cad")
        .and_then(cents)
        .unwrap_or(0.0);
    let tol_pct = grounding
        .get("tolerance_pct")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    let recon_score = if let (false, Some(target)) = (reported.is_empty(), total_delta) {
        let tolerance = tol_abs.max((target * tol_pct).abs());
        let rep_sum: f64 = reported.iter().sum();
        let error = (rep_sum - target).abs();
        let s = if error <= tolerance {
            30
        } else {
            let span = target.abs().max(1.0);
            (30.0 * (1.0 - (error - tolerance) / span)).round().max(0.0) as i64
        };
        bits.push(format!(
            "reconcile err={:.2} tol={:.2} ({}/30)",
            error, tolerance, s
        ));
        s
    } else {
        bits.push("reconcile=n/a (0/30)".to_string());
        0
    };

    let known = known_set(
        grounding
            .get("known_amounts")
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[]),
    );
    let (prose_score, prose_note) = score_prose_amounts(prose, &known, 20);
    bits.push(prose_note);

    let total = schema + trace_score + recon_score + prose_score;
    (total.min(MAX_SCORE), bits.join("  "))
}

pub fn validate_taxes_qa(output: &Value, explicit_grounding: Option<&Value>) -> (i64, String) {
    let loaded = load_grounding("qa");
    let grounding = explicit_grounding.unwrap_or(&loaded);
    let (parsed, note) = parse_output(output);
    let mut bits = if !note.is_empty() {
        vec![note]
    } else {
        Vec::new()
    };

    let map = match parsed {
        Some(Value::Object(m)) => m,
        _ => {
            return (
                0,
                format!(
                    "schema=0/20 ({})",
                    bits.first()
                        .cloned()
                        .unwrap_or_else(|| "not-an-object".to_string())
                ),
            )
        }
    };

    let prose = map.get("prose").and_then(|v| v.as_str()).unwrap_or("");
    let citations = map.get("citations").and_then(|v| v.as_array());

    let mut schema = 0;
    if !prose.is_empty() {
        schema += 10;
    }
    if citations.is_some() {
        schema += 10;
    }
    bits.push(format!("schema={}/20", schema));

    let known_ids: HashSet<String> = grounding
        .get("known_fact_ids")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .map(|v| v.to_string().trim_matches('"').to_string())
                .collect()
        })
        .unwrap_or_default();

    let cited: Vec<String> = citations
        .map(|arr| {
            arr.iter()
                .filter_map(|c| {
                    c.as_object()?
                        .get("fact_id")?
                        .as_str()
                        .map(|s| s.to_string())
                })
                .collect()
        })
        .unwrap_or_default();

    let cite_score = if !cited.is_empty() {
        let hits = cited.iter().filter(|c| known_ids.contains(*c)).count();
        let s = (40.0 * hits as f64 / cited.len() as f64).round() as i64;
        bits.push(format!("citations={}/{} ({}/40)", hits, cited.len(), s));
        s
    } else if !known_ids.is_empty() {
        bits.push("citations=0 (0/40, facts were available)".to_string());
        0
    } else {
        bits.push("citations=0 (40/40, no facts to cite)".to_string());
        40
    };

    let known = known_set(
        grounding
            .get("known_amounts")
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[]),
    );
    let (prose_score, prose_note) = score_prose_amounts(prose, &known, 40);
    bits.push(prose_note);

    let total = schema + cite_score + prose_score;
    (total.min(MAX_SCORE), bits.join("  "))
}

pub fn validate_taxes_slip_qa(output: &Value, explicit_grounding: Option<&Value>) -> (i64, String) {
    let loaded = load_grounding("slip_qa");
    let grounding = explicit_grounding.unwrap_or(&loaded);
    let (parsed, note) = parse_output(output);
    let mut bits = if !note.is_empty() {
        vec![note]
    } else {
        Vec::new()
    };

    let map = match parsed {
        Some(Value::Object(m)) => m,
        _ => {
            return (
                0,
                format!(
                    "schema=0/30 ({})",
                    bits.first()
                        .cloned()
                        .unwrap_or_else(|| "not-an-object".to_string())
                ),
            )
        }
    };

    let prose = map.get("prose").and_then(|v| v.as_str()).unwrap_or("");
    let ids = map.get("highlighted_flag_ids").and_then(|v| v.as_array());

    let mut schema = 0;
    if !prose.is_empty() {
        schema += 15;
    }
    if ids.is_some() {
        schema += 15;
    }
    bits.push(format!("schema={}/30", schema));

    let known_flags: HashSet<String> = grounding
        .get("known_flag_ids")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .map(|v| v.to_string().trim_matches('"').to_string())
                .collect()
        })
        .unwrap_or_default();

    let reported_ids: Vec<String> = ids
        .map(|arr| {
            arr.iter()
                .map(|v| v.to_string().trim_matches('"').to_string())
                .collect()
        })
        .unwrap_or_default();

    let flag_score = if !reported_ids.is_empty() {
        let hits = reported_ids
            .iter()
            .filter(|i| known_flags.contains(*i))
            .count();
        let s = (35.0 * hits as f64 / reported_ids.len() as f64).round() as i64;
        bits.push(format!("flags={}/{} ({}/35)", hits, reported_ids.len(), s));
        s
    } else {
        let s = if known_flags.is_empty() { 35 } else { 0 };
        bits.push(format!(
            "flags=0 claimed, {} known ({}/35)",
            known_flags.len(),
            s
        ));
        s
    };

    let known = known_set(
        grounding
            .get("known_amounts")
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[]),
    );
    let amounts = prose_amounts(prose);
    let num_score = if known.is_empty() {
        let s = if amounts.is_empty() { 35 } else { 0 };
        bits.push(format!(
            "prose_amounts={} with none sourceable ({}/35)",
            amounts.len(),
            s
        ));
        s
    } else {
        let (s, prose_note) = score_prose_amounts(prose, &known, 35);
        bits.push(prose_note);
        s
    };

    let total = schema + flag_score + num_score;
    (total.min(MAX_SCORE), bits.join("  "))
}

#[cfg(test)]
mod tests {
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
}
