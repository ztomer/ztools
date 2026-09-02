//! `ztools model-eval --capabilities`: what each servable model IS, without
//! running anything.
//!
//! Split from `cli_ztools.rs` for the house 500-line cap, along a seam the
//! file already had. Everything else in that module RUNS eval tasks -- it
//! takes the GPU lock, refuses oversize models, and reports scores. This
//! reports family, generative verdict, on-disk footprint and viability from
//! what is already recorded, and touches none of that machinery.

use anyhow::Result;

use super::resolve_models;
use crate::config::ZtoolsConfig;

/// `--capabilities`: probe what each servable model IS -- family (recorded
/// architecture first, name match as fallback), generative verdict, on-disk
/// weight footprint, and viability (packaging defects + learned decode rate)
/// -- WITHOUT running a single task. Port of `ev --capabilities`.
pub(super) fn print_capabilities(url: &str, model_selector: &str) -> Result<()> {
    let models = resolve_models(url, model_selector, &ZtoolsConfig::default())?;
    if models.is_empty() {
        println!("no models found");
        return Ok(());
    }
    println!("Model                               Family        Disk GB   Gen?  Viability");
    for m in &models {
        let family = recorded_family_or_name(m);
        let disk_gb = crate::ztools::eval::model_disk_bytes(m)
            .map(|b| format!("{:.1}", b as f64 / 1024.0 / 1024.0 / 1024.0))
            .unwrap_or_else(|| "-".to_string());
        let gen = if crate::ztools::eval::is_generative_model(m) {
            "yes"
        } else {
            "NO"
        };
        let decode = decode_rate(m);
        let viability = match crate::ztools::model_health::assess_viability(m, decode, None) {
            Ok(()) => "ok".to_string(),
            Err(reason) => reason,
        };
        println!(
            "{:<36} {:<12} {:>9} {:>6}  {}",
            truncate_col(m, 36),
            truncate_col(&family, 12),
            disk_gb,
            gen,
            truncate_col(&viability, 60)
        );
    }
    Ok(())
}

fn recorded_family_or_name(model: &str) -> String {
    // Same resolution chain as budgets.rs: recorded architecture trimmed to a
    // conf/models file, else the name's family token, else "default".
    crate::ztools::eval::config_family(model).unwrap_or_else(|| "default".to_string())
}

fn decode_rate(model: &str) -> Option<f64> {
    let signals = crate::ztools::eval::load_signals();
    let caps = signals.get(model)?.get("_capabilities")?;
    caps.get("decode_tokens_per_sec")?.as_f64()
}

fn truncate_col(s: &str, width: usize) -> String {
    if s.chars().count() <= width {
        s.to_string()
    } else {
        let cut: String = s.chars().take(width.saturating_sub(3)).collect();
        format!("{cut}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A literal model selector never reaches the network — `resolve_models`
    /// short-circuits before the HTTP call — so the whole report is printable
    /// against a URL nothing is listening on. Every lookup underneath
    /// (`config_family`, `model_disk_bytes`, `load_signals`) degrades to a
    /// stated placeholder when its file is absent, which is exactly the state
    /// a fresh checkout is in.
    #[test]
    fn a_literal_model_prints_a_row_without_touching_the_network() {
        print_capabilities("http://127.0.0.1:1", "some-unknown-model")
            .expect("a literal selector must not need a live server");
    }

    /// `--capabilities all` DOES need the server, and says so rather than
    /// printing an empty table that reads like "no models installed".
    #[test]
    fn asking_for_all_models_without_a_server_is_an_error_not_an_empty_table() {
        let err = print_capabilities("http://127.0.0.1:1", "all")
            .expect_err("an unreachable server cannot be reported as zero models");
        assert!(!err.to_string().is_empty(), "the failure says something");
    }

    /// The family chain: a recorded architecture wins, a name match is the
    /// fallback, and "default" is what an unrecognised name gets. The last one
    /// is the branch that matters — it must be a stated word, not an empty
    /// column that reads as a missing value.
    #[test]
    fn an_unrecognised_model_reports_the_default_family_rather_than_nothing() {
        assert_eq!(
            recorded_family_or_name("not-a-real-model-name-xyz"),
            "default"
        );
    }

    #[test]
    fn a_model_with_no_recorded_signals_has_no_decode_rate() {
        assert_eq!(decode_rate("not-a-real-model-name-xyz"), None);
    }

    /// The column widths are what keep the table readable; an over-long value
    /// must be cut with a visible marker, never silently clipped to look like
    /// a shorter real name.
    #[test]
    fn a_column_wider_than_its_budget_is_cut_with_an_ellipsis() {
        assert_eq!(truncate_col("short", 10), "short");
        assert_eq!(truncate_col("exactlyten", 10), "exactlyten");
        assert_eq!(truncate_col("elevenchars", 10), "elevenc...");
        assert_eq!(truncate_col("elevenchars", 10).chars().count(), 10);
    }

    /// Multi-byte names must be cut on character boundaries. `&s[..n]` would
    /// panic here, and a model list is exactly where an unusual name shows up.
    #[test]
    fn truncation_counts_characters_not_bytes() {
        let wide = "модель-с-очень-длинным-именем";
        let cut = truncate_col(wide, 10);
        assert_eq!(cut.chars().count(), 10, "{cut}");
        assert!(cut.ends_with("..."), "{cut}");
    }

    /// A width smaller than the ellipsis itself must not underflow. The
    /// `saturating_sub` is the guard; this is what proves it is load-bearing.
    #[test]
    fn a_width_narrower_than_the_ellipsis_does_not_underflow() {
        assert_eq!(truncate_col("abcdef", 2), "...");
        assert_eq!(truncate_col("abcdef", 0), "...");
    }
}
