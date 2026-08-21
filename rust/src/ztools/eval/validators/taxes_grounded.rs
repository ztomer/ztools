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
}
