//! `ztools models scout`: who could replace each slot's model, PROPOSE-ONLY
//! (LM.DESIGN sections 1-2, LM.SCOUT).
//!
//! It reads the Hugging Face Hub API and the Osaurus catalog, takes every
//! allowlisted member of each incumbent's family (the HF `base_model` tag) and
//! every catalog successor of a retired incumbent, and runs the four filters of
//! [`rules`] on each. It never downloads, installs or switches anything: the
//! most it says is which ONE candidate it would stage. Outside sources only
//! nominate; the local eval is what promotes (LM.SELECT).
//!
//! A feed that refuses is never read as "nothing there": what it would have
//! decided reads `unverified`, and the report counts the feeds that answered
//! against the feeds asked.

pub mod catalog;
pub mod feeds;
pub mod local;
mod render;
pub mod rules;

#[cfg(test)]
mod parts_tests;
#[cfg(test)]
mod tests;

pub use feeds::{Feed, Fixture, Live};
pub use local::Local;
pub use render::render;
pub use rules::Check;

use crate::ztools::refcheck::{is_cracked, short_name};
use serde_json::Value;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

/// Every key asked, once, with its answer: the report's denominator.
struct Reader<'a> {
    feed: &'a dyn Feed,
    asked: RefCell<BTreeMap<String, Result<Value, String>>>,
}

impl Reader<'_> {
    fn get(&self, key: &str) -> Result<Value, String> {
        if let Some(hit) = self.asked.borrow().get(key) {
            return hit.clone();
        }
        let got = self.feed.get(key);
        self.asked.borrow_mut().insert(key.to_string(), got.clone());
        got
    }

    fn rows(&self, key: &str) -> Result<Vec<Value>, String> {
        self.get(key)
            .map(|v| v.as_array().cloned().unwrap_or_default())
    }
}

fn ids(rows: &[Value]) -> Vec<String> {
    rows.iter()
        .filter_map(|r| r.get("id").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

/// One nominated repo and what each filter said of it. A filter that was not
/// reached (an earlier one failed) is `None`.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub repo: String,
    /// Why it is on the list: `family <base>` or `successor of <repo>`.
    pub nominated: String,
    pub provenance: Check,
    pub format: Option<Check>,
    pub fit: Option<Check>,
    pub disk: Option<Check>,
}

impl Candidate {
    fn checks(&self) -> impl Iterator<Item = &Check> {
        std::iter::once(&self.provenance)
            .chain(&self.format)
            .chain(&self.fit)
            .chain(&self.disk)
    }

    /// Every filter reached and passed.
    #[must_use]
    pub fn passes(&self) -> bool {
        self.disk.is_some() && self.fit.is_some() && self.checks().all(Check::passed)
    }
}

/// One incumbent model and the slots that name it.
#[derive(Debug, Clone)]
pub struct Incumbent {
    pub slots: Vec<String>,
    pub name: String,
    /// The installed repo id, if the slot's name is installed.
    pub repo: Option<String>,
    /// The HF `base_model`, or why it is not known.
    pub family: Result<String, String>,
    /// Retired from the Osaurus catalog this version ships.
    pub retired: bool,
    pub candidates: Vec<Candidate>,
}

/// What a scout run found.
#[derive(Debug, Clone)]
pub struct Proposal {
    pub asked: usize,
    /// Each feed that refused, and why.
    pub refused: Vec<(String, String)>,
    pub incumbents: Vec<Incumbent>,
    /// The one candidate it would stage, if any passes every filter.
    pub stage: Option<String>,
    pub notes: Vec<String>,
}

/// What the Osaurus catalog says, or why it could not be read.
struct Catalog {
    retired: BTreeSet<String>,
    /// Curated ids, then org-fetch ids by downloads.
    listed: Vec<String>,
    served: BTreeSet<String>,
    complete: bool,
}

fn read_catalog(r: &Reader, local: &Local, notes: &mut Vec<String>) -> Catalog {
    let mut cat = Catalog {
        retired: BTreeSet::new(),
        listed: Vec::new(),
        served: local
            .installed
            .iter()
            .filter_map(|i| i.model_type.clone())
            .collect(),
        complete: false,
    };
    match &local.osaurus_version {
        None => notes.push("⚠ Osaurus version unknown: the catalog source is unverified".into()),
        Some(v) => match r.get(&feeds::catalog_key(v)) {
            Ok(c) => {
                let list = |k: &str| c.get(k).and_then(Value::as_array).cloned().unwrap_or_default();
                cat.retired = list("retired").iter().filter_map(Value::as_str).map(str::to_string).collect();
                for row in list("curated") {
                    cat.listed.extend(row.get("id").and_then(Value::as_str).map(str::to_string));
                    cat.served.extend(row.get("model_type").and_then(Value::as_str).map(str::to_string));
                }
                cat.complete = true;
            }
            Err(why) => notes.push(format!(
                "⚠ Osaurus {v} catalog unverified ({why}): model types are judged by the installed ones"
            )),
        },
    }
    if let Ok(rows) = r.rows(&feeds::org_key()) {
        cat.listed.extend(ids(&rows));
    }
    cat
}

/// The leading letters of a repo name: `raptor` for `Raptor-0.6-4B-JANG_6M`.
fn series(repo: &str) -> String {
    rules::name(repo)
        .chars()
        .take_while(char::is_ascii_alphabetic)
        .collect::<String>()
        .to_lowercase()
}

fn base_model(card: &Value) -> Option<String> {
    card.get("tags")?
        .as_array()?
        .iter()
        .filter_map(Value::as_str)
        .filter_map(|t| t.strip_prefix("base_model:"))
        .find(|b| !b.contains(':'))
        .map(str::to_string)
}

struct Ctx<'a> {
    r: &'a Reader<'a>,
    local: &'a Local,
    cat: &'a Catalog,
    gpu_budget: Option<u64>,
}

fn judge(cx: &Ctx, repo: &str, nominated: &str, allowlisted: &BTreeSet<String>) -> Candidate {
    let mut c = Candidate {
        repo: repo.to_string(),
        nominated: nominated.to_string(),
        provenance: rules::provenance(repo, allowlisted),
        format: None,
        fit: None,
        disk: None,
    };
    if !c.provenance.passed() {
        return c;
    }
    let files: Vec<rules::File> = match cx.r.rows(&feeds::tree_key(repo)) {
        Ok(rows) => rows
            .iter()
            .filter_map(|f| {
                Some(rules::File {
                    path: f.get("path")?.as_str()?.to_string(),
                    size: f.get("size")?.as_u64()?,
                })
            })
            .collect(),
        Err(why) => {
            c.format = Some(Check::Unverified(format!("file list: {why}")));
            return c;
        }
    };
    if let pre @ Check::Fail(_) = rules::format(&files, &Value::Null, &cx.cat.served) {
        c.format = Some(pre);
        return c;
    }
    let config = match cx.r.get(&feeds::config_key(repo)) {
        Ok(v) => v,
        Err(why) => {
            c.format = Some(Check::Unverified(format!("config.json: {why}")));
            return c;
        }
    };
    let format = match rules::format(&files, &config, &cx.cat.served) {
        Check::Fail(why) if !cx.cat.complete && why.starts_with("model_type") => {
            Check::Unverified(format!("{why} (catalog unverified)"))
        }
        other => other,
    };
    let ok = format.passed();
    c.format = Some(format);
    if ok {
        c.fit = Some(rules::fit(
            rules::weights_bytes(&files),
            &config,
            cx.gpu_budget,
        ));
        c.disk = Some(rules::disk(
            rules::download_bytes(&files),
            cx.local.free_bytes,
            cx.local.staged,
        ));
    }
    c
}

fn nominate(cx: &Ctx, inc: &mut Incumbent, repo: &str) {
    let installed: BTreeSet<String> = cx
        .local
        .installed
        .iter()
        .map(|i| i.repo.to_lowercase())
        .collect();
    let mut seen: BTreeSet<String> = installed;
    let mut queue: Vec<(String, String)> = Vec::new();
    let mut push = |id: String, why: String, queue: &mut Vec<(String, String)>| {
        if seen.insert(id.to_lowercase()) {
            queue.push((id, why));
        }
    };
    if inc.retired {
        let want = series(repo);
        for id in &cx.cat.listed {
            if rules::author(id) == rules::author(repo)
                && series(id) == want
                && !cx.cat.retired.contains(&id.to_lowercase())
            {
                push(id.clone(), format!("successor of {repo}"), &mut queue);
            }
        }
    }
    let mut allowlisted: BTreeSet<String> = BTreeSet::new();
    if let Ok(base) = &inc.family {
        for author in rules::ALLOWLIST {
            let Ok(rows) = cx.r.rows(&feeds::family_key(author, base)) else {
                continue;
            };
            let mut members = ids(&rows);
            members.sort_by_key(|id| !is_cracked(id));
            for id in members {
                allowlisted.insert(rules::name(&id).to_lowercase());
                push(id, format!("family {base}"), &mut queue);
            }
        }
        allowlisted.insert(rules::name(repo).to_lowercase());
        if let Ok(rows) = cx.r.rows(&feeds::search_key(rules::name(repo))) {
            for id in ids(&rows) {
                if !rules::ALLOWLIST.contains(&rules::author(&id)) {
                    push(id, format!("search {}", rules::name(repo)), &mut queue);
                }
            }
        }
    }
    inc.candidates = queue
        .iter()
        .map(|(id, why)| judge(cx, id, why, &allowlisted))
        .collect();
}

/// Run the scout over `feed` for `local`'s slots.
#[must_use]
pub fn scout(feed: &dyn Feed, local: &Local) -> Proposal {
    let r = Reader {
        feed,
        asked: RefCell::new(BTreeMap::new()),
    };
    let mut notes = Vec::new();
    let gpu_budget = r
        .get(feeds::HEALTH)
        .ok()
        .and_then(|h| h.get("gpu_budget_bytes")?.as_u64());
    let cat = read_catalog(&r, local, &mut notes);
    let cx = Ctx {
        r: &r,
        local,
        cat: &cat,
        gpu_budget,
    };
    let mut by_name: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (slot, model) in &local.slots {
        by_name
            .entry(model.to_lowercase())
            .or_default()
            .push(slot.clone());
    }
    let mut incumbents = Vec::new();
    for (name, slots) in by_name {
        let repo = local
            .installed
            .iter()
            .map(|i| i.repo.clone())
            .find(|id| short_name(id) == name);
        let mut inc = Incumbent {
            slots,
            name,
            repo: repo.clone(),
            family: Err("not installed".into()),
            retired: false,
            candidates: Vec::new(),
        };
        if let Some(repo) = repo {
            inc.retired = cat.retired.contains(&repo.to_lowercase());
            inc.family = r.get(&feeds::model_key(&repo)).and_then(|card| {
                base_model(&card).ok_or_else(|| "its card has no base_model tag".into())
            });
            nominate(&cx, &mut inc, &repo);
        }
        incumbents.push(inc);
    }
    incumbents.sort_by_key(|i| (!i.retired, !i.slots.iter().any(|s| s == "think")));
    let stage = incumbents
        .iter()
        .flat_map(|i| &i.candidates)
        .find(|c| c.passes())
        .map(|c| c.repo.clone());
    let asked = r.asked.borrow();
    let refused = asked
        .iter()
        .filter_map(|(k, v)| v.as_ref().err().map(|why| (k.clone(), why.clone())))
        .collect();
    Proposal {
        asked: asked.len(),
        refused,
        incumbents,
        stage,
        notes,
    }
}
