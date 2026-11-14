//! Promote a candidate into one slot, and restore the previous selection.

use super::{Canary, DAY, Event, Kind, Ledger, Mode, Paths, SLOT_DAYS, SWITCHABLE, selection};
use std::collections::BTreeMap;

/// What a promotion did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Promotion {
    /// The selection file now names `to`; the canary is open.
    Switched {
        slot: String,
        from: String,
        to: String,
    },
    /// The rule chose it, the mode only proposes it.
    Proposed {
        slot: String,
        to: String,
        why: String,
    },
    /// A guard rail refused it.
    Refused {
        slot: String,
        to: String,
        why: String,
    },
}

impl Promotion {
    /// The findings fragment.
    #[must_use]
    pub fn line(&self) -> String {
        match self {
            Self::Switched { slot, from, to } => {
                format!(
                    "{slot} {from} -> {to} switched, canary 0/{}",
                    super::CANARY_RUNS
                )
            }
            Self::Proposed { slot, to, why } => format!("{slot} -> {to} propose-only ({why})"),
            Self::Refused { slot, to, why } => format!("{slot} -> {to} not switched ({why})"),
        }
    }
}

/// Why `slot` may not take `to` now, if it may not.
fn refusal(ledger: &Ledger, slot: &str, from: &str, to: &str, now: f64) -> Option<String> {
    if slot.contains("embed") {
        return Some("embedding models are never switched automatically".into());
    }
    if !SWITCHABLE.contains(&slot) {
        return Some(format!("{slot} is not a switchable slot"));
    }
    if super::canary::floor_seconds(slot).is_none() {
        return Some(format!(
            "{slot} has no canary speed floor, so no consumer run could confirm or roll it back"
        ));
    }
    if from == to {
        return Some("already selected".into());
    }
    if let Some(c) = &ledger.canary {
        return Some(format!(
            "a canary is open: {} on {} ({}/{})",
            c.candidate,
            c.slot,
            c.passed,
            super::CANARY_RUNS
        ));
    }
    if let Some(why) = ledger.bar(to, now) {
        return Some(why);
    }
    if let Some(last) = ledger.last_switch(slot)
        && now - last.at < SLOT_DAYS * DAY
    {
        return Some(format!(
            "{slot} switched {} on {}; one switch a slot per {SLOT_DAYS:.0} days",
            last.to, last.date
        ));
    }
    None
}

/// Switch `slot` from `from` to `to` when the mode and every guard rail allow.
///
/// `base` is the selection as the running config holds it, used to seed a
/// selection file that does not yet exist.
///
/// # Errors
/// When the ledger or the selection file cannot be read or written.
pub fn promote(
    paths: &Paths,
    mode: &Mode,
    (now, date): (f64, &str),
    base: &BTreeMap<String, String>,
    slot: &str,
    to: &str,
) -> anyhow::Result<Promotion> {
    let from = base.get(slot).cloned().unwrap_or_default();
    let mut ledger = super::Ledger::load(paths)?;
    let (slot_s, to_s) = (slot.to_string(), to.to_string());
    if let Some(why) = refusal(&ledger, slot, &from, to, now) {
        return Ok(Promotion::Refused {
            slot: slot_s,
            to: to_s,
            why,
        });
    }
    if let Mode::Propose(why) = mode {
        return Ok(Promotion::Proposed {
            slot: slot_s,
            to: to_s,
            why: why.clone(),
        });
    }
    let previous = selection::read(&paths.selection)?;
    let written = selection::with_slot(previous.as_deref(), base, slot, to)?;
    std::fs::create_dir_all(&paths.state)?;
    std::fs::write(paths.previous(), previous.as_deref().unwrap_or(""))?;
    ledger.events.push(Event {
        at: now,
        date: date.to_string(),
        slot: slot_s.clone(),
        kind: Kind::Switch,
        from: from.clone(),
        to: to_s.clone(),
        why: String::new(),
        previous,
        written: Some(written.clone()),
    });
    ledger.canary = Some(Canary {
        slot: slot_s.clone(),
        candidate: to_s.clone(),
        incumbent: from.clone(),
        opened: now,
        passed: 0,
    });
    // The ledger first: a crash between the two leaves a canary that names the
    // file it would restore, never a switched file with no way back.
    ledger.save(paths)?;
    selection::write(&paths.selection, Some(&written))?;
    Ok(Promotion::Switched {
        slot: slot_s,
        from,
        to: to_s,
    })
}

/// Put the selection back as it was before `switch`: the whole previous file
/// when nothing has touched the file since, else only that slot.
fn restore(paths: &Paths, switch: &Event) -> anyhow::Result<()> {
    let now = selection::read(&paths.selection)?;
    if now.is_some() && now == switch.written {
        return selection::write(&paths.selection, switch.previous.as_deref());
    }
    let base = BTreeMap::new();
    let text = selection::with_slot(now.as_deref(), &base, &switch.slot, &switch.from)?;
    selection::write(&paths.selection, Some(&text))
}

/// Restore the selection before the latest switch not yet rolled back, bar its
/// candidate when its canary was open, and record it.
///
/// # Errors
/// When there is nothing to roll back, or a file cannot be read or written.
pub fn rollback(paths: &Paths, (now, date): (f64, &str), why: &str) -> anyhow::Result<Event> {
    let mut ledger = Ledger::load(paths)?;
    let open = ledger.canary.clone();
    let target = ledger
        .events
        .iter()
        .enumerate()
        .rev()
        .find(|(i, e)| {
            e.kind == Kind::Switch
                && !ledger.events[i + 1..]
                    .iter()
                    .any(|r| r.kind == Kind::Rollback && r.slot == e.slot && r.from == e.to)
        })
        .map(|(_, e)| e.clone())
        .ok_or_else(|| anyhow::anyhow!("nothing to roll back: the ledger records no switch"))?;
    restore(paths, &target)?;
    if open
        .as_ref()
        .is_some_and(|c| (&c.slot, &c.candidate) == (&target.slot, &target.to))
    {
        ledger.canary = None;
        ledger
            .barred
            .insert(target.to.clone(), super::BAR_DAYS.mul_add(DAY, now));
    }
    let event = Event {
        at: now,
        date: date.to_string(),
        slot: target.slot.clone(),
        kind: Kind::Rollback,
        from: target.to.clone(),
        to: target.from,
        why: why.to_string(),
        previous: None,
        written: None,
    };
    ledger.events.push(event.clone());
    ledger.save(paths)?;
    Ok(event)
}
