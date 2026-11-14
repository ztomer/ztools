//! A replaced incumbent's weights are kept [`RETIRE_DAYS`] days after its
//! successor is confirmed, then deleted (`rm -rf`, never Trash; LM.DESIGN
//! section 6, LM.CANARYFLOORS).
//!
//! Only a directory this job staged itself is ever deleted (the design's
//! scope rule): a model the owner installed is said, once, and kept. A model
//! that a slot selects again by then is kept too.

use super::{DAY, Ledger, Paths, RETIRE_DAYS};
use std::collections::BTreeSet;
use std::path::PathBuf;

/// What the sweep did with one retired incumbent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Swept {
    /// Its weights (every listed directory) are gone.
    Deleted(String),
    /// Kept for good, and why.
    Kept(String, String),
}

impl Swept {
    /// The findings fragment.
    #[must_use]
    pub fn line(&self) -> String {
        match self {
            Self::Deleted(m) => {
                format!("{m} replaced {RETIRE_DAYS:.0} days ago: weights deleted")
            }
            Self::Kept(m, why) => format!("replaced {m} kept ({why})"),
        }
    }
}

/// Delete the weights of every incumbent replaced [`RETIRE_DAYS`] or more
/// days before `now`.
///
/// `in_use` is every model a slot selects now; `owned`
/// names the directories this job staged for a model, `None` when it staged
/// none (the owner installed it).
///
/// A removal that fails is said and its entry stays, so the next night tries
/// again.
///
/// # Errors
/// When the ledger cannot be read or written.
pub fn sweep(
    paths: &Paths,
    now: f64,
    in_use: &BTreeSet<String>,
    owned: &dyn Fn(&str) -> Option<Vec<PathBuf>>,
) -> anyhow::Result<Vec<Swept>> {
    if !paths.ledger().exists() {
        return Ok(Vec::new());
    }
    let mut ledger = Ledger::load(paths)?;
    let mut done = Vec::new();
    let mut keep = Vec::new();
    for r in std::mem::take(&mut ledger.retired) {
        if in_use.contains(&r.model) {
            done.push(Swept::Kept(r.model, "a slot selects it again".into()));
        } else if now - r.since < RETIRE_DAYS * DAY {
            keep.push(r);
        } else if let Some(dirs) = owned(&r.model) {
            let failed = dirs
                .iter()
                .filter(|d| d.exists())
                .find_map(|d| std::fs::remove_dir_all(d).err().map(|e| (d.clone(), e)));
            if let Some((d, e)) = failed {
                let why = format!("{}: {e}; tried again next night", d.display());
                done.push(Swept::Kept(r.model.clone(), why));
                keep.push(r);
            } else {
                done.push(Swept::Deleted(r.model));
            }
        } else {
            done.push(Swept::Kept(
                r.model,
                "not staged by this job: yours to delete".into(),
            ));
        }
    }
    ledger.retired = keep;
    ledger.save(paths)?;
    Ok(done)
}
