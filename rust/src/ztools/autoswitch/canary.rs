//! The canary: the next [`CANARY_RUNS`] real runs of a switched slot's
//! consumer must succeed, pass the consumer's own gate and stay within the
//! speed floor. A failure rolls back inside that same run.

use super::{CANARY_RUNS, Event, Kind, Ledger, Paths, switch};

/// The slowest a consumer run may be, in wall seconds, on `slot`.
///
/// `summarize`: the rule's own floor for the slot's slowest task, cold load
/// included (`Rule::DESIGN.summarize_max_seconds`, 600 s); the summarize
/// step is one such call. `json`: the weekend plan's whole pipeline, which
/// makes a structure call per batch plus the search, at most 30 minutes.
/// The others have no consumer this job watches yet: no floor, so a run there
/// is inconclusive rather than passing on nothing.
#[must_use]
pub fn floor_seconds(slot: &str) -> Option<f64> {
    match slot {
        "summarize" => Some(crate::ztools::select::Rule::DESIGN.summarize_max_seconds),
        "json" => Some(1800.0),
        _ => None,
    }
}

/// What the consumer's own gate said of one run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Passed,
    /// The run failed, or its gate refused what it made.
    Failed(String),
    /// The run says nothing about the model (no input, a bot wall).
    Inconclusive(String),
}

/// One real consumer run.
#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    pub slot: String,
    /// The model the run used.
    pub model: String,
    pub verdict: Verdict,
    pub seconds: f64,
}

/// What the canary did with a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// No canary is open on this slot and model.
    NotWatched,
    Inconclusive(String),
    Passing(u32),
    Confirmed(String),
    RolledBack {
        candidate: String,
        restored: String,
        why: String,
    },
}

/// Count `run` against the open canary.
///
/// # Errors
/// When the ledger or the selection file cannot be read or written.
pub fn observe(paths: &Paths, (now, date): (f64, &str), run: &Run) -> anyhow::Result<Step> {
    if !paths.ledger().exists() {
        return Ok(Step::NotWatched);
    }
    let mut ledger = Ledger::load(paths)?;
    let Some(canary) = ledger.canary.clone() else {
        return Ok(Step::NotWatched);
    };
    if (&canary.slot, &canary.candidate) != (&run.slot, &run.model) {
        return Ok(Step::NotWatched);
    }
    let failed = match &run.verdict {
        Verdict::Inconclusive(why) => return Ok(Step::Inconclusive(why.clone())),
        Verdict::Failed(why) => Some(why.clone()),
        Verdict::Passed => match floor_seconds(&run.slot) {
            None => return Ok(Step::Inconclusive(format!("{}: no speed floor", run.slot))),
            Some(f) if run.seconds > f => Some(format!("took {:.0}s > {f:.0}s floor", run.seconds)),
            Some(_) => None,
        },
    };
    if let Some(why) = failed {
        let ev = switch::rollback(paths, (now, date), &format!("canary: {why}"))?;
        paths.append_finding(&format!(
            "LM {date}: {} canary failed on {} ({why}); rolled back to {}, {} barred {:.0} days",
            ev.slot,
            ev.from,
            ev.to,
            ev.from,
            super::BAR_DAYS
        ))?;
        return Ok(Step::RolledBack {
            candidate: ev.from,
            restored: ev.to,
            why,
        });
    }
    let passed = canary.passed + 1;
    let kind = if passed >= CANARY_RUNS {
        Kind::Confirm
    } else {
        Kind::CanaryPass
    };
    ledger.events.push(Event {
        at: now,
        date: date.to_string(),
        slot: canary.slot.clone(),
        kind,
        from: canary.incumbent.clone(),
        to: canary.candidate.clone(),
        why: format!("canary {passed}/{CANARY_RUNS}, {:.0}s", run.seconds),
        previous: None,
        written: None,
    });
    if kind == Kind::Confirm {
        ledger.canary = None;
        ledger.save(paths)?;
        let line = format!(
            "LM {date}: {} -> {} confirmed (canary {CANARY_RUNS}/{CANARY_RUNS})",
            canary.slot, canary.candidate
        );
        paths.append_finding(&line)?;
        return Ok(Step::Confirmed(canary.candidate));
    }
    ledger.canary = Some(super::Canary { passed, ..canary });
    ledger.save(paths)?;
    Ok(Step::Passing(passed))
}

/// [`observe`] for a consumer process: this user's paths and clock, one line
/// on stderr, and a fault that never fails the consumer's own run.
pub fn observe_live(slot: &str, model: &str, verdict: Verdict, seconds: f64) {
    let paths = Paths::from_home();
    let (at, date) = super::clock();
    let run = Run {
        slot: slot.into(),
        model: model.into(),
        verdict,
        seconds,
    };
    match observe(&paths, (at, &date), &run) {
        Ok(Step::NotWatched) => {}
        Ok(Step::Passing(n)) => eprintln!("→ canary {slot} {model}: {n}/{CANARY_RUNS} passed"),
        Ok(Step::Confirmed(m)) => eprintln!("✓ canary {slot}: {m} confirmed"),
        Ok(Step::Inconclusive(why)) => eprintln!("→ canary {slot} {model}: not counted ({why})"),
        Ok(Step::RolledBack {
            candidate,
            restored,
            why,
        }) => eprintln!(
            "✗ canary {slot}: {candidate} failed ({why}); the selection is back on {restored}"
        ),
        Err(e) => eprintln!("⚠ canary {slot}: {e:#}"),
    }
}

/// The summarize consumer's gate, read off its result.
///
/// A failed run fails, and so does a DEGRADED summary, because then the
/// model's own answer was refused by `twitter/quality.rs` (or never came) and
/// a fallback answered.
#[must_use]
pub fn twitter_verdict(result: &anyhow::Result<std::path::PathBuf>) -> Verdict {
    match result {
        Err(e) => Verdict::Failed(format!("the run failed: {}", e.root_cause())),
        Ok(path) => match std::fs::read_to_string(path) {
            Err(e) => Verdict::Failed(format!("{}: {e}", path.display())),
            Ok(doc) if doc.contains("**DEGRADED**") => {
                Verdict::Failed("degraded: the model's answer was refused or missing".into())
            }
            Ok(_) => Verdict::Passed,
        },
    }
}

/// The json consumer's gate (the weekend plan): the model was there, and the
/// structure phase turned the fetched corpus into rows. A run whose search
/// answered nothing says nothing about the model.
#[must_use]
pub fn weekend_verdict(health: &crate::ztools::weekend::PlanHealth) -> Verdict {
    use crate::ztools::weekend::ModelHealth;
    match &health.model {
        ModelHealth::Ready { .. } => {}
        ModelHealth::Unavailable { reason, .. } => {
            return Verdict::Failed(format!("the model did not answer: {reason}"));
        }
        ModelHealth::NotInstalled { .. } => {
            return Verdict::Failed("the model is not installed".into());
        }
    }
    if health.provenance.extracted > 0 {
        Verdict::Passed
    } else if health.search.answered == 0 {
        Verdict::Inconclusive("no search answered, nothing to extract".into())
    } else {
        Verdict::Failed(format!(
            "{} searches answered and no row came back: the JSON phase gave nothing usable",
            health.search.answered
        ))
    }
}
