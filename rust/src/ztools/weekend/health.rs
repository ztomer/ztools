//! What the plan's two external dependencies actually did during a run, so a
//! plan with no transient events can say WHY.
//!
//! Class: an empty result that cannot distinguish "nothing to find" from
//! "could not look". For a month every plan carried the same warning —
//! "search/extraction yielded 0 events" — while the truth was that every
//! search was bot-walled and the model never finished loading. The warning
//! was accurate and useless. These records travel from the fetch into the
//! rendered plan so the sentence names the cause.

use super::search::{EngineVerdict, QueryOutcome, ENGINES, ENGINE_COUNT};

/// Tally of what the search engines said across a run's fan-out.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SearchHealth {
    pub queries: usize,
    /// Queries that produced at least one result, from any engine.
    pub answered: usize,
    /// Queries that produced nothing AND hit at least one bot wall.
    pub starved: usize,
    /// Per engine, indexed like [`ENGINES`].
    pub blocked: [usize; ENGINE_COUNT],
    pub unreachable: [usize; ENGINE_COUNT],
}

impl SearchHealth {
    pub fn record(&mut self, outcome: &QueryOutcome) {
        self.queries += 1;
        if !outcome.results.is_empty() {
            self.answered += 1;
        }
        if outcome.starved_by_bot_wall() {
            self.starved += 1;
        }
        for (i, verdict) in outcome.verdicts.iter().enumerate() {
            match verdict {
                EngineVerdict::Blocked => self.blocked[i] += 1,
                EngineVerdict::Unreachable => self.unreachable[i] += 1,
                _ => {}
            }
        }
    }

    /// True when a bot wall cost the run at least one query outright.
    #[must_use]
    pub const fn bot_walled(&self) -> bool {
        self.starved > 0
    }

    /// One sentence for the plan, or `None` when every query was answered.
    #[must_use]
    pub fn summary(&self) -> Option<String> {
        if self.queries == 0 || self.answered == self.queries {
            return None;
        }
        let mut parts = Vec::new();
        for (i, engine) in ENGINES.iter().enumerate() {
            if self.blocked[i] > 0 {
                parts.push(format!("{engine} bot-walled {}", self.blocked[i]));
            }
            if self.unreachable[i] > 0 {
                parts.push(format!("{engine} unreachable {}", self.unreachable[i]));
            }
        }
        let detail = if parts.is_empty() {
            String::new()
        } else {
            format!(" ({})", parts.join(", "))
        };
        Some(format!(
            "{} of {} searches answered{detail}",
            self.answered, self.queries
        ))
    }
}

/// Whether the model was there to do the extraction at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelHealth {
    /// Warm-up answered; `secs` is how long the cold start took.
    Ready { model: String, secs: u64 },
    /// Warm-up did not answer; no phase was attempted.
    Unavailable { model: String, reason: String },
}

impl ModelHealth {
    #[must_use]
    pub const fn is_ready(&self) -> bool {
        matches!(self, Self::Ready { .. })
    }
}

/// What the enforcement gates did to the extracted rows.
///
/// So a week of plans can say whether a noisier corpus moves the
/// invented-row rate. Counted where the rows are dropped
/// (`cli_ztools::weekend_plan`), written into the plan as its provenance
/// line, and read back by `ztools status`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Provenance {
    /// Transient rows the model produced before any gate.
    pub extracted: usize,
    /// Dropped because nothing in the fetched corpus supports them.
    pub unsourced: usize,
    /// Dropped because their dates fall outside the plan's weekend.
    pub outside_window: usize,
    /// Dropped by the operator's exclusion list.
    pub excluded: usize,
}

impl Provenance {
    /// The plan's provenance line — always written, zeros included, so the
    /// reading exists for the runs where nothing was dropped too.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "_Provenance: {} extracted, {} unsourced, {} outside the window, {} excluded._",
            self.extracted, self.unsourced, self.outside_window, self.excluded
        )
    }

    /// Parse the line back out of a stored plan.
    #[must_use]
    pub fn parse(plan: &str) -> Option<Self> {
        let line = plan.lines().find(|l| l.starts_with("_Provenance: "))?;
        let nums: Vec<usize> = line
            .split(|c: char| !c.is_ascii_digit())
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.parse().ok())
            .collect();
        match nums[..] {
            [extracted, unsourced, outside_window, excluded, ..] => Some(Self {
                extracted,
                unsourced,
                outside_window,
                excluded,
            }),
            _ => None,
        }
    }
}

/// Everything the plan's degraded-warning needs to name its cause.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanHealth {
    pub search: SearchHealth,
    pub model: ModelHealth,
    pub provenance: Provenance,
}

impl PlanHealth {
    /// A health record that claims nothing went wrong, for callers that
    /// render a plan without having run the pipeline (tests, fixtures).
    #[must_use]
    pub fn nominal() -> Self {
        Self {
            search: SearchHealth::default(),
            model: ModelHealth::Ready {
                model: String::new(),
                secs: 0,
            },
            provenance: Provenance::default(),
        }
    }

    /// The WARNING body for a plan with no transient events. Names the cause
    /// when there is one; otherwise says the quiet weekend is the finding.
    #[must_use]
    pub fn degraded_reason(&self) -> String {
        let mut causes = Vec::new();
        if let ModelHealth::Unavailable { model, reason } = &self.model {
            causes.push(format!(
                "the model `{model}` did not answer ({reason}), so nothing was extracted"
            ));
        }
        if self.search.bot_walled() {
            causes.push(format!(
                "{} of {} searches were blocked by a bot wall — the corpus was starved, \
                 not the weekend quiet",
                self.search.starved, self.search.queries
            ));
        } else if let Some(summary) = self.search.summary() {
            causes.push(summary);
        }
        if causes.is_empty() {
            "No live transient events found for this weekend (search/extraction yielded \
             0 events). Showing year-round fixed venues as fallback."
                .to_string()
        } else {
            format!(
                "No live transient events: {}. Showing year-round fixed venues as fallback.",
                causes.join("; ")
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ztools::weekend::search::SearchResult;

    fn outcome(results: usize, verdicts: [EngineVerdict; ENGINE_COUNT]) -> QueryOutcome {
        QueryOutcome {
            query: "q".into(),
            results: (0..results)
                .map(|i| SearchResult {
                    title: format!("t{i}"),
                    href: String::new(),
                    body: String::new(),
                })
                .collect(),
            verdicts,
        }
    }

    #[test]
    fn a_query_walled_at_every_engine_counts_as_starved() {
        let mut h = SearchHealth::default();
        h.record(&outcome(
            0,
            [
                EngineVerdict::Blocked,
                EngineVerdict::Blocked,
                EngineVerdict::Empty,
            ],
        ));
        h.record(&outcome(
            3,
            [
                EngineVerdict::Blocked,
                EngineVerdict::Answered(3),
                EngineVerdict::Skipped,
            ],
        ));
        assert_eq!(h.queries, 2);
        assert_eq!(h.answered, 1);
        assert_eq!(h.starved, 1, "a walled query Bing rescued is not starved");
        assert_eq!(h.blocked, [2, 1, 0]);
        assert!(h.bot_walled());
        assert_eq!(
            h.summary().unwrap(),
            "1 of 2 searches answered (DuckDuckGo bot-walled 2, Bing bot-walled 1)"
        );
    }

    #[test]
    fn a_fully_answered_run_has_no_summary_and_no_wall() {
        let mut h = SearchHealth::default();
        h.record(&outcome(
            2,
            [
                EngineVerdict::Answered(2),
                EngineVerdict::Skipped,
                EngineVerdict::Skipped,
            ],
        ));
        assert!(!h.bot_walled());
        assert_eq!(h.summary(), None);
    }

    #[test]
    fn degraded_reason_names_the_wall_and_the_dead_model() {
        let mut search = SearchHealth::default();
        search.record(&outcome(
            0,
            [
                EngineVerdict::Blocked,
                EngineVerdict::Unreachable,
                EngineVerdict::Blocked,
            ],
        ));
        let health = PlanHealth {
            search,
            model: ModelHealth::Unavailable {
                model: "qwen".into(),
                reason: "no answer within 900s".into(),
            },
            provenance: Provenance::default(),
        };
        let reason = health.degraded_reason();
        assert!(
            reason.contains("`qwen` did not answer (no answer within 900s)"),
            "{reason}"
        );
        assert!(
            reason.contains("1 of 1 searches were blocked by a bot wall"),
            "{reason}"
        );
        assert!(reason.starts_with("No live transient events:"), "{reason}");
    }

    #[test]
    fn the_provenance_line_round_trips_zeros_included() {
        let p = Provenance {
            extracted: 12,
            unsourced: 3,
            outside_window: 0,
            excluded: 1,
        };
        let plan = format!("# Weekend Plan\n\n| a |\n\n{}\n", p.line());
        assert_eq!(Provenance::parse(&plan), Some(p));
        assert_eq!(Provenance::parse("# Weekend Plan\n"), None);
        assert_eq!(
            Provenance::default().line(),
            "_Provenance: 0 extracted, 0 unsourced, 0 outside the window, 0 excluded._"
        );
    }

    #[test]
    fn nominal_health_keeps_the_original_sentence() {
        let reason = PlanHealth::nominal().degraded_reason();
        assert!(
            reason.contains("search/extraction yielded 0 events"),
            "{reason}"
        );
    }
}
