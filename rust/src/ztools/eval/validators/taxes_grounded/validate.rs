//! The three public taxes validators: YoY narrative, QA, and slip QA.
//!
//! Split out of taxes_grounded.rs for the 500-line production cap. These are the
//! entry points; the arithmetic lives in `amounts`, the fixtures in `grounding`.

use serde_json::Value;
use std::collections::HashSet;

use super::amounts::{
    cents, known_set, prose_amounts, score_prose_amounts, traceable_sums, MAX_SCORE,
};
use super::grounding::{load_grounding, parse_output};

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
