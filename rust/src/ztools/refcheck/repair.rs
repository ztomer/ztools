//! Choosing where a missing reference should point: the best INSTALLED member
//! of the same model family.
//!
//! The family key is the Hugging Face `base_model` tag when it is known (it is
//! cached in `families.json`), else the name stem; both are reduced to the same
//! stem so the two kinds of key compare. "Best" is the highest think-slot score
//! in `eval_history.json`; a cracked build wins a tie.

use super::refs::short_name;
use std::cmp::Ordering;
use std::collections::BTreeMap;

/// Tokens that start the quantisation or variant suffix of a model name.
fn is_suffix_token(tok: &str) -> bool {
    let digits_then = |s: &str, tail: &str| {
        s.strip_suffix(tail)
            .is_some_and(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit()))
    };
    digits_then(tok, "bit")
        || tok.starts_with("jang")
        || tok.starts_with("mxfp")
        || tok.starts_with("optiq")
        || matches!(
            tok,
            "crack"
                | "it"
                | "instruct"
                | "bf16"
                | "fp16"
                | "fp8"
                | "awq"
                | "gptq"
                | "mlx"
                | "uncensored"
                | "abliterated"
        )
        || (tok.len() > 1
            && tok.starts_with('q')
            && tok[1..].starts_with(|c: char| c.is_ascii_digit()))
}

/// `Qwen3.8-27B-8bit` and `qwen3.8-27b-jang_6d-crack` both reduce to
/// `qwen3.8-27b`: the name up to its first quantisation or variant token.
#[must_use]
pub fn stem(name: &str) -> String {
    let lower = short_name(name);
    let mut kept: Vec<&str> = Vec::new();
    for (i, tok) in lower.split('-').enumerate() {
        if i > 0 && is_suffix_token(tok) {
            break;
        }
        kept.push(tok);
    }
    kept.join("-")
}

/// The key two models share when they come from one base model.
#[must_use]
pub fn family_key(base_model: Option<&str>, repo: &str) -> String {
    stem(base_model.unwrap_or(repo))
}

/// A cracked build, by its name.
#[must_use]
pub fn is_cracked(model: &str) -> bool {
    model.to_lowercase().contains("crack")
}

fn better(a: &(&String, Option<f64>), b: &(&String, Option<f64>)) -> Ordering {
    let by_score = match (a.1, b.1) {
        (Some(x), Some(y)) if (x - y).abs() < 1e-9 => Ordering::Equal,
        (Some(x), Some(y)) => y.total_cmp(&x),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    };
    by_score
        .then_with(|| is_cracked(b.0).cmp(&is_cracked(a.0)))
        .then_with(|| a.0.cmp(b.0))
}

/// The best installed member of `missing`'s family, if any.
///
/// `base` maps a repo id to its cached base model. `scores` maps a lowercased
/// short name to its think-slot score.
#[must_use]
pub fn choose(
    missing: &str,
    installed: &[String],
    base: &dyn Fn(&str) -> Option<String>,
    scores: &BTreeMap<String, f64>,
) -> Option<String> {
    let want = family_key(base(missing).as_deref(), missing);
    let mut members: Vec<(&String, Option<f64>)> = installed
        .iter()
        .filter(|id| id.as_str() != missing)
        .filter(|id| family_key(base(id).as_deref(), id) == want)
        .map(|id| (id, scores.get(&short_name(id)).copied()))
        .collect();
    members.sort_by(better);
    members.first().map(|(id, _)| (*id).clone())
}
