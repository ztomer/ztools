//! Engine order learned from the runs before this one.
//!
//! `SearchHealth` records which engine walled which query on every run, and
//! until 2026-09-19 nothing read it back: a planner whose first engine had
//! been walled for a month still paid sixteen challenged POSTs per plan
//! before reaching the one that answered. The rule here demotes an engine to
//! the back of the order when it walled most of its queries across the last
//! few runs, and restores it the run after it answers again. Thresholds are
//! DATA (`conf/weekend.toml [search]`); the record is a small JSON file the
//! planner appends to after each run.

use serde::{Deserialize, Serialize};

use super::health::SearchHealth;
use super::search::{ENGINES, ENGINE_COUNT};

/// One run's per-engine walls, as the record stores them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunRecord {
    pub date: String,
    pub queries: usize,
    /// Indexed like [`ENGINES`].
    pub blocked: [usize; ENGINE_COUNT],
}

/// The rolling record.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SearchRecord {
    pub runs: Vec<RunRecord>,
}

/// How many runs the record keeps; enough for a threshold of a few.
const KEEP_RUNS: usize = 10;

/// The thresholds, from `conf/weekend.toml [search]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DemotionPolicy {
    /// An engine is judged only once this many runs are on record.
    pub after_runs: usize,
    /// ...and demoted when it walled at least this fraction of its queries
    /// over those runs.
    pub wall_ratio: f64,
}

impl Default for DemotionPolicy {
    fn default() -> Self {
        Self {
            after_runs: 3,
            wall_ratio: 0.8,
        }
    }
}

impl DemotionPolicy {
    /// Read `[search] demote_after_runs` / `demote_wall_ratio` from the first
    /// weekend.toml that carries them; defaults otherwise.
    #[must_use]
    pub fn load(paths: &[String]) -> Self {
        let mut policy = Self::default();
        for raw in paths {
            let path = crate::manifest::expand_tilde(raw);
            let Ok(content) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Ok(val) = toml::from_str::<toml::Value>(&content) else {
                continue;
            };
            let Some(search) = val.get("search") else {
                continue;
            };
            if let Some(n) = search
                .get("demote_after_runs")
                .and_then(toml::Value::as_integer)
            {
                policy.after_runs = usize::try_from(n).unwrap_or(policy.after_runs).max(1);
            }
            if let Some(r) = search
                .get("demote_wall_ratio")
                .and_then(toml::Value::as_float)
            {
                policy.wall_ratio = r.clamp(0.0, 1.0);
            }
            return policy;
        }
        policy
    }
}

impl SearchRecord {
    #[must_use]
    pub fn load(path: &std::path::Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Append this run and keep the tail; a write failure is reported, never
    /// fatal — the order is a saving, not a correctness property.
    pub fn record(&mut self, path: &std::path::Path, date: &str, health: &SearchHealth) {
        self.runs.push(RunRecord {
            date: date.to_string(),
            queries: health.queries,
            blocked: health.blocked,
        });
        if self.runs.len() > KEEP_RUNS {
            let drop = self.runs.len() - KEEP_RUNS;
            self.runs.drain(..drop);
        }
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        match serde_json::to_string_pretty(self) {
            Ok(json) => {
                if let Err(e) = std::fs::write(path, json) {
                    eprintln!(
                        "\u{26a0} search record not written to {}: {e}",
                        path.display()
                    );
                }
            }
            Err(e) => eprintln!("\u{26a0} search record not serialised: {e}"),
        }
    }

    /// Engine indexes in the order to try them: the default order with every
    /// engine that walled `wall_ratio` of its queries over the last
    /// `after_runs` runs moved to the back, in their own relative order.
    /// Fewer runs on record than the policy asks for means no judgement.
    #[must_use]
    pub fn order(&self, policy: &DemotionPolicy) -> [usize; ENGINE_COUNT] {
        let mut order = [0usize; ENGINE_COUNT];
        for (i, slot) in order.iter_mut().enumerate() {
            *slot = i;
        }
        if self.runs.len() < policy.after_runs {
            return order;
        }
        let recent = &self.runs[self.runs.len() - policy.after_runs..];
        let queries: usize = recent.iter().map(|r| r.queries).sum();
        if queries == 0 {
            return order;
        }
        let demoted: Vec<bool> = (0..ENGINE_COUNT)
            .map(|i| {
                let blocked: usize = recent.iter().map(|r| r.blocked[i]).sum();
                ratio(blocked, queries) >= policy.wall_ratio
            })
            .collect();
        let mut kept: Vec<usize> = (0..ENGINE_COUNT).filter(|&i| !demoted[i]).collect();
        kept.extend((0..ENGINE_COUNT).filter(|&i| demoted[i]));
        order.copy_from_slice(&kept);
        order
    }

    /// One line for the operator when the order is not the default.
    #[must_use]
    pub fn describe(
        &self,
        policy: &DemotionPolicy,
        order: &[usize; ENGINE_COUNT],
    ) -> Option<String> {
        let default: Vec<usize> = (0..ENGINE_COUNT).collect();
        if order[..] == default[..] {
            return None;
        }
        let names: Vec<&str> = order.iter().map(|&i| ENGINES[i]).collect();
        let recent = &self.runs[self.runs.len().saturating_sub(policy.after_runs)..];
        let queries: usize = recent.iter().map(|r| r.queries).sum();
        let walled: Vec<String> = (0..ENGINE_COUNT)
            .filter(|&i| {
                let blocked: usize = recent.iter().map(|r| r.blocked[i]).sum();
                ratio(blocked, queries) >= policy.wall_ratio
            })
            .map(|i| {
                let blocked: usize = recent.iter().map(|r| r.blocked[i]).sum();
                format!(
                    "{} walled {blocked} of {queries} queries over the last {} runs",
                    ENGINES[i],
                    recent.len()
                )
            })
            .collect();
        Some(format!("{} ({})", names.join(" → "), walled.join("; ")))
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "query counts per run are in the tens; f64 is exact far beyond that"
)]
fn ratio(part: usize, whole: usize) -> f64 {
    part as f64 / whole as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(blocked: [usize; ENGINE_COUNT]) -> RunRecord {
        RunRecord {
            date: "2026-09-19".into(),
            queries: 16,
            blocked,
        }
    }

    #[test]
    fn too_few_runs_keep_the_default_order() {
        let rec = SearchRecord {
            runs: vec![run([16, 0, 0]), run([16, 0, 0])],
        };
        assert_eq!(rec.order(&DemotionPolicy::default()), [0, 1, 2]);
        assert!(rec
            .describe(&DemotionPolicy::default(), &[0, 1, 2])
            .is_none());
    }

    #[test]
    fn a_walled_first_engine_moves_to_the_back_and_comes_back_when_it_answers() {
        let policy = DemotionPolicy::default();
        let mut rec = SearchRecord {
            runs: vec![run([16, 0, 0]), run([14, 0, 0]), run([13, 0, 0])],
        };
        assert_eq!(rec.order(&policy), [1, 2, 0]);
        let line = rec.describe(&policy, &[1, 2, 0]).unwrap();
        assert!(line.starts_with("Bing → Brave → DuckDuckGo ("), "{line}");
        assert!(
            line.contains("DuckDuckGo walled 43 of 48 queries over the last 3 runs"),
            "{line}"
        );
        // Two clean runs pull the ratio under the bar: back to the front.
        rec.runs.push(run([0, 0, 0]));
        rec.runs.push(run([0, 0, 0]));
        assert_eq!(rec.order(&policy), [0, 1, 2]);
    }

    #[test]
    fn two_walled_engines_keep_their_relative_order_at_the_back() {
        let rec = SearchRecord {
            runs: vec![run([16, 16, 0]); 3],
        };
        assert_eq!(rec.order(&DemotionPolicy::default()), [2, 0, 1]);
    }

    #[test]
    fn the_record_round_trips_and_keeps_only_the_tail() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("nested/search_health.json");
        let mut rec = SearchRecord::default();
        let health = SearchHealth {
            queries: 16,
            blocked: [9, 0, 0],
            ..Default::default()
        };
        for _ in 0..(KEEP_RUNS + 2) {
            rec.record(&path, "2026-09-19", &health);
        }
        assert_eq!(rec.runs.len(), KEEP_RUNS);
        let loaded = SearchRecord::load(&path);
        assert_eq!(loaded, rec);
        assert_eq!(
            SearchRecord::load(td.path().join("missing.json").as_path()),
            SearchRecord::default()
        );
    }

    #[test]
    fn the_policy_reads_weekend_toml_and_defaults_otherwise() {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().join("weekend.toml");
        std::fs::write(
            &p,
            "[search]\ndemote_after_runs = 5\ndemote_wall_ratio = 0.5\n",
        )
        .unwrap();
        let policy = DemotionPolicy::load(&[p.to_string_lossy().into_owned()]);
        assert_eq!(policy.after_runs, 5);
        assert!((policy.wall_ratio - 0.5).abs() < 1e-9);
        assert_eq!(
            DemotionPolicy::load(&["/nonexistent".into()]),
            DemotionPolicy::default()
        );
    }
}
