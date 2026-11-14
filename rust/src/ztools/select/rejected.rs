//! Candidates the decision rule rejected (LM.RENOMINATE).
//!
//! A rejected candidate is deleted that night. Before this record nothing
//! remembered the verdict, so the scout could nominate the same repo the next
//! night and the night re-downloaded and re-measured it to the same answer
//! (seen by LM.RAPTOR07HANG, 2026-11-14). A rejection rests on three things,
//! and it stands until one of them changes, the way `unloadable.json` bars a
//! candidate until the Osaurus version changes:
//!
//! - the candidate's revision (the Hub commit it was downloaded at),
//! - each slot's measure: the incumbent it was judged against and the
//!   identities of the tasks it was judged on,
//! - the Osaurus version it ran on.
//!
//! An input unknown on either side cannot be shown to have changed, so it
//! does not lift the record; one that is known on both sides and differs does.

use super::measure::{is_refusal_task, slot_tasks};
use crate::ztools::eval::task_fingerprint::TaskIdentities;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// What one slot's verdict was measured against.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotBasis {
    pub incumbent: String,
    /// Task name to its identity: the slot's tasks and the refusal probe.
    pub tasks: BTreeMap<String, String>,
}

/// One candidate the rule rejected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rejected {
    /// Short name the eval addresses it by.
    pub model: String,
    /// The Hub commit it was downloaded at; `None` when it was unknown.
    pub revision: Option<String>,
    /// The Osaurus version it ran on; `None` when it was unknown.
    pub osaurus: Option<String>,
    pub slots: BTreeMap<String, SlotBasis>,
    pub date: String,
    pub why: String,
}

/// The rule's inputs as they stand now.
pub struct Now<'a> {
    /// The repo's current revision, asked only for a repo that has a record.
    pub revision: &'a dyn Fn(&str) -> Option<String>,
    pub osaurus: Option<&'a str>,
    pub slots: &'a BTreeMap<String, String>,
    pub current: &'a TaskIdentities,
}

/// The tasks a slot is judged on, with their identities: the slot's own tasks
/// and the whole refusal probe (the nightly's task list).
#[must_use]
pub fn slot_measure(current: &TaskIdentities, slot: &str) -> BTreeMap<String, String> {
    slot_tasks(current, slot)
        .into_iter()
        .chain(current.keys().filter(|t| is_refusal_task(t)).cloned())
        .filter_map(|t| current.get(&t).map(|id| (t, id.clone())))
        .collect()
}

/// The basis of each of `of` that has an incumbent.
#[must_use]
pub fn basis(
    current: &TaskIdentities,
    slots: &BTreeMap<String, String>,
    of: &[String],
) -> BTreeMap<String, SlotBasis> {
    of.iter()
        .filter_map(|slot| {
            let incumbent = slots.get(slot)?.clone();
            Some((
                slot.clone(),
                SlotBasis {
                    incumbent,
                    tasks: slot_measure(current, slot),
                },
            ))
        })
        .collect()
}

/// `state/rejected.json`: repo to its record.
#[must_use]
pub fn path(state_dir: &Path) -> PathBuf {
    state_dir.join("rejected.json")
}

/// The records on disk; a missing or unreadable file is none.
#[must_use]
pub fn load(state_dir: &Path) -> BTreeMap<String, Rejected> {
    std::fs::read_to_string(path(state_dir))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

/// Record `repo` as rejected by the rule.
///
/// # Errors
/// When the state directory or the file cannot be written.
pub fn record(state_dir: &Path, repo: &str, rec: Rejected) -> std::io::Result<()> {
    let mut all = load(state_dir);
    all.insert(repo.to_string(), rec);
    std::fs::create_dir_all(state_dir)?;
    let text = serde_json::to_string_pretty(&all).map_err(std::io::Error::other)?;
    let tmp = path(state_dir).with_extension("json.tmp");
    std::fs::write(&tmp, text + "\n")?;
    std::fs::rename(tmp, path(state_dir))
}

fn differs(then: Option<&str>, now: Option<&str>) -> bool {
    matches!((then, now), (Some(a), Some(b)) if a != b)
}

/// Whether anything `rec` rested on has changed. The revision is asked last,
/// and only when the record has one: it is the one input that costs a request.
fn changed(repo: &str, rec: &Rejected, now: &Now<'_>) -> bool {
    let slots: Vec<String> = rec.slots.keys().cloned().collect();
    differs(rec.osaurus.as_deref(), now.osaurus)
        || basis(now.current, now.slots, &slots) != rec.slots
        || (rec.revision.is_some()
            && differs(rec.revision.as_deref(), (now.revision)(repo).as_deref()))
}

/// Why `name` (a repo id or its short name) is not to be nominated: the rule
/// rejected it and nothing that rejection rested on has changed.
#[must_use]
pub fn why(state_dir: &Path, name: &str, now: &Now<'_>) -> Option<String> {
    let lower = name.to_lowercase();
    load(state_dir)
        .into_iter()
        .find(|(repo, r)| repo.to_lowercase() == lower || r.model.to_lowercase() == lower)
        .filter(|(repo, r)| !changed(repo, r, now))
        .map(|(repo, r)| {
            format!(
                "{repo} was rejected by the rule on {} ({}); it stands until its revision, a \
                 slot's measure or the Osaurus version changes",
                r.date, r.why
            )
        })
}
