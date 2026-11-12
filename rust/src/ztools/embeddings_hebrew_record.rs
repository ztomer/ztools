//! The live calibration of a Hebrew judge: `ztools models hebrew-pairing`.
//!
//! It asks the summarize slot's model about the pairs of the committed 42-tweet
//! Hebrew fixture, under the GPU lock through the residency seam, and records
//! every answer with its time. The record is committed beside the fixture; a
//! model enters [`super::CALIBRATED_JUDGES`] only with a record that
//! [`verdict`] passes, and the pinning test re-checks it on every run.

use std::time::Instant;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::{PREFILTER_FLOOR, SYSTEM, ask, candidates, is_hebrew, overlap, parse_verdict};

/// One fixture tweet: its labelled group and its text.
#[derive(Debug, Clone, Deserialize)]
pub struct Row {
    pub group: String,
    pub text: String,
}

#[derive(Deserialize)]
struct Fixture {
    tweets: Vec<Row>,
}

/// The Hebrew rows of a fixture file (any extra fields, such as vectors, are
/// ignored).
///
/// # Errors
///
/// When the file cannot be read or is not a fixture.
pub fn hebrew_rows(json: &str) -> Result<Vec<Row>> {
    let fx: Fixture = serde_json::from_str(json).context("not an embeddings fixture")?;
    Ok(fx
        .tweets
        .into_iter()
        .filter(|r| is_hebrew(&r.text))
        .collect())
}

/// `he_football_basket` is a neighbour of `he_football`'s story.
#[must_use]
pub fn story(group: &str) -> String {
    group.split('_').take(2).collect::<Vec<_>>().join("_")
}

/// A pair's class: `same` story, a neighbouring story (`trap`), `unrelated`.
#[must_use]
pub fn class(a: &Row, b: &Row) -> &'static str {
    if a.group == b.group {
        "same"
    } else if story(&a.group) == story(&b.group) {
        "trap"
    } else {
        "unrelated"
    }
}

/// One judged pair.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Judged {
    pub a: usize,
    pub b: usize,
    pub class: String,
    pub answer: String,
    pub seconds: f64,
}

/// One recorded run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Record {
    pub model: String,
    pub recorded: String,
    /// The machine's load averages when the run started: a loaded machine's
    /// times are loaded numbers.
    pub load: String,
    pub system: String,
    pub prefilter_floor: f64,
    /// Every pair of the fixture, or the prefilter's candidates and the traps.
    pub all_pairs: bool,
    pub load_seconds: f64,
    pub pairs: Vec<Judged>,
}

/// Per class: (pairs, right answers).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Tally {
    pub pairs: usize,
    pub right: usize,
}

/// What a record says about the PIPELINE that ships: the prefilter, then the model.
///
/// A pair is merged when its overlap reaches [`PREFILTER_FLOOR`]
/// and the model said yes. The tallies count the pipeline's right outcomes;
/// `wrong` lists its wrong ones (an unreadable answer on a candidate is wrong
/// whatever the class). It passes when `wrong` is empty and every labelled
/// pair and every trap was asked.
///
/// `judge_alone` lists the model's wrong answers on pairs the prefilter never
/// sends it: what the model would get wrong without the prefilter, said so
/// that a record cannot pass on the prefilter's work unseen.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Verdict {
    pub same: Tally,
    pub trap: Tally,
    pub unrelated: Tally,
    pub median_seconds: f64,
    pub wrong: Vec<String>,
    pub judge_alone: Vec<String>,
}

impl Verdict {
    #[must_use]
    pub fn passes(&self, rows: &[Row]) -> bool {
        let (mut same, mut trap) = (0, 0);
        for (i, a) in rows.iter().enumerate() {
            for b in &rows[i + 1..] {
                match class(a, b) {
                    "same" => same += 1,
                    "trap" => trap += 1,
                    _ => {}
                }
            }
        }
        self.wrong.is_empty() && self.same.pairs == same && self.trap.pairs == trap && same > 0
    }
}

/// Score `record` against the fixture `rows`.
#[must_use]
pub fn verdict(record: &Record, rows: &[Row]) -> Verdict {
    let mut v = Verdict::default();
    let mut times: Vec<f64> = Vec::new();
    for p in &record.pairs {
        let (Some(a), Some(b)) = (rows.get(p.a), rows.get(p.b)) else {
            v.wrong
                .push(format!("pair {}-{} is not in the fixture", p.a, p.b));
            continue;
        };
        let cls = class(a, b);
        let want = cls == "same";
        let got = parse_verdict(&p.answer);
        let candidate = overlap(&a.text, &b.text) >= PREFILTER_FLOOR;
        let tally = match cls {
            "same" => &mut v.same,
            "trap" => &mut v.trap,
            _ => &mut v.unrelated,
        };
        tally.pairs += 1;
        let line = format!("{cls} {} / {}: answered {:?}", a.group, b.group, p.answer);
        if candidate && got == Some(want) || !candidate && !want {
            tally.right += 1;
        } else {
            v.wrong.push(line.clone());
        }
        if !candidate && got != Some(want) {
            v.judge_alone.push(format!("{line} (below the prefilter)"));
        }
        if cls != p.class {
            v.wrong.push(format!(
                "pair {}-{} recorded as {}, is {cls}",
                p.a, p.b, p.class
            ));
        }
        times.push(p.seconds);
    }
    times.sort_by(f64::total_cmp);
    v.median_seconds = times.get(times.len() / 2).copied().unwrap_or(0.0);
    v
}

/// The pairs a run judges: every pair, or the prefilter's candidates plus any
/// trap the prefilter missed (a trap must be asked to be passed).
#[must_use]
pub fn pairs_to_judge(rows: &[Row], all_pairs: bool) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if all_pairs {
        for i in 0..rows.len() {
            for j in i + 1..rows.len() {
                out.push((i, j));
            }
        }
        return out;
    }
    let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();
    out = candidates(&texts, PREFILTER_FLOOR);
    for i in 0..rows.len() {
        for j in i + 1..rows.len() {
            if class(&rows[i], &rows[j]) != "unrelated" && !out.contains(&(i, j)) {
                out.push((i, j));
            }
        }
    }
    out
}

/// The load averages, as `sysctl -n vm.loadavg` says them.
fn load_averages() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// Ask `model` about `rows`' pairs, loaded and released by the residency
/// seam, and record every answer.
///
/// A model still resident after the run keeps the record (`with_model_kept`):
/// the first all-pairs run lost its 861 answers to that release error.
///
/// # Errors
///
/// From the load; a failed call is recorded as its error text, which
/// [`verdict`] counts as wrong, and a failed release is printed.
pub fn run(base_url: &str, model: &str, rows: &[Row], all_pairs: bool) -> Result<Record> {
    let load = load_averages();
    let pairs = pairs_to_judge(rows, all_pairs);
    let timing = crate::ztools::residency::Timing::default();
    let ran = crate::ztools::residency::with_model_kept(base_url, model, timing, |held| {
        let mut judged = Vec::with_capacity(pairs.len());
        for (n, &(i, j)) in pairs.iter().enumerate() {
            let started = Instant::now();
            let answer = ask(base_url, model, &rows[i].text, &rows[j].text)
                .unwrap_or_else(|e| format!("error: {e:#}"));
            let seconds = started.elapsed().as_secs_f64();
            eprintln!(
                "· {}/{} {} {answer:?} {seconds:.2}s",
                n + 1,
                pairs.len(),
                class(&rows[i], &rows[j])
            );
            judged.push(Judged {
                a: i,
                b: j,
                class: class(&rows[i], &rows[j]).to_string(),
                answer,
                seconds: (seconds * 1000.0).round() / 1000.0,
            });
        }
        Ok(Record {
            model: model.to_string(),
            recorded: chrono::Local::now()
                .format("%Y-%m-%d %H:%M %:z")
                .to_string(),
            load,
            system: SYSTEM.to_string(),
            prefilter_floor: PREFILTER_FLOOR,
            all_pairs,
            load_seconds: (held.load_seconds() * 1000.0).round() / 1000.0,
            pairs: judged,
        })
    })?;
    ran.released.report("the record is kept");
    ran.out
}

#[cfg(test)]
#[path = "embeddings_hebrew_record_tests.rs"]
mod tests;
