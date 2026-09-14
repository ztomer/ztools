//! Model selection and fallback-chain construction for the summarizer.
//!
//! Port of the selection half of `references/twitter/summarize.py`
//! (`select_best_model` lives in `lib/osaurus_models.py`): which model to
//! try first and in what order to fall back when an attempt yields nothing.
//! Pure list arithmetic — no transport — so the retry loop ([`super::chain`])
//! can be tested without a server. Preference and fallback names arrive from
//! `conf/twitter.toml [fallback]`; nothing here names a model.

/// Pick a model from a roster by preference-substring match, case-insensitive.
/// First preference wins; with no match the first listed model wins; an empty
/// roster selects nothing.
#[must_use]
pub fn select_best_model(models: &[String], preferred: &[String]) -> Option<String> {
    if models.is_empty() {
        return None;
    }
    for pref in preferred {
        let needle = pref.to_lowercase();
        if let Some(hit) = models.iter().find(|m| m.to_lowercase().contains(&needle)) {
            return Some(hit.clone());
        }
    }
    models.first().cloned()
}

/// Ordered deduped fallback chain: the target first, then the extra names.
///
/// When a roster is known, models the server does not serve are dropped —
/// except `foundation`, the on-device last resort, which is always kept.
/// With no roster the chain is unfiltered: nothing is known to be absent.
#[must_use]
pub fn fallback_chain(
    target: &str,
    available: &[String],
    extra_fallbacks: &[String],
) -> Vec<String> {
    let mut chain = Vec::new();
    for name in std::iter::once(&target.to_string()).chain(extra_fallbacks.iter()) {
        let name = name.trim();
        if name.is_empty() || chain.iter().any(|m: &String| m == name) {
            continue;
        }
        if !available.is_empty()
            && !available.iter().any(|m| m == name)
            && !name.to_lowercase().contains("foundation")
        {
            continue;
        }
        chain.push(name.to_string());
    }
    chain
}

/// Resolve which model to try first: a served target is kept; a missing one
/// resolves through [`select_best_model`]; with no roster the target stands.
#[must_use]
pub fn resolve_target_model(target: &str, available: &[String], preferred: &[String]) -> String {
    if !available.is_empty() && !available.iter().any(|m| m == target) {
        select_best_model(available, preferred).unwrap_or_else(|| target.to_string())
    } else {
        target.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_roster_selects_nothing() {
        assert_eq!(select_best_model(&[], &["qwen".to_string()]), None);
    }

    #[test]
    fn preferred_substring_wins_case_insensitively() {
        let models = vec!["m1".to_string(), "QWEN3.6-35b".to_string()];
        assert_eq!(
            select_best_model(&models, &["qwen".to_string()]),
            Some("QWEN3.6-35b".to_string())
        );
    }

    #[test]
    fn first_preference_beats_later_ones() {
        let models = vec!["gemma-4".to_string(), "qwen3".to_string()];
        assert_eq!(
            select_best_model(&models, &["qwen".to_string(), "gemma".to_string()]),
            Some("qwen3".to_string())
        );
    }

    #[test]
    fn no_match_falls_back_to_first_listed() {
        let models = vec!["m2".to_string(), "m3".to_string()];
        assert_eq!(
            select_best_model(&models, &["qwen".to_string()]),
            Some("m2".to_string())
        );
    }

    #[test]
    fn chain_dedupes_strips_and_drops_unserved_models() {
        let available = vec!["m1".to_string(), "qwen3".to_string()];
        let chain = fallback_chain("m1", &available, &[" qwen3 ".to_string(), "m9".to_string()]);
        assert_eq!(chain, vec!["m1".to_string(), "qwen3".to_string()]);
    }

    #[test]
    fn chain_always_keeps_foundation_as_last_resort() {
        let available = vec!["m1".to_string()];
        let chain = fallback_chain("m1", &available, &["foundation".to_string()]);
        assert_eq!(chain, vec!["m1".to_string(), "foundation".to_string()]);
    }

    #[test]
    fn chain_is_unfiltered_when_no_roster_is_known() {
        let chain = fallback_chain("m1", &[], &["m9".to_string()]);
        assert_eq!(chain, vec!["m1".to_string(), "m9".to_string()]);
    }

    #[test]
    fn served_target_is_kept() {
        let available = vec!["m1".to_string(), "m2".to_string()];
        assert_eq!(
            resolve_target_model("m1", &available, &["qwen".to_string()]),
            "m1".to_string()
        );
    }

    #[test]
    fn missing_target_resolves_to_best_match() {
        let available = vec!["m2".to_string(), "qwen3".to_string()];
        assert_eq!(
            resolve_target_model("m1", &available, &["qwen".to_string()]),
            "qwen3".to_string()
        );
    }

    #[test]
    fn missing_target_without_match_resolves_to_first_listed() {
        let available = vec!["m2".to_string()];
        assert_eq!(
            resolve_target_model("m1", &available, &["qwen".to_string()]),
            "m2".to_string()
        );
    }

    #[test]
    fn unknown_roster_leaves_the_target_alone() {
        assert_eq!(
            resolve_target_model("m1", &[], &["qwen".to_string()]),
            "m1".to_string()
        );
    }
}
