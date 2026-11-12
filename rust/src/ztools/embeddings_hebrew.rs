//! Hebrew same-story pairing by the summarize slot's local model (LM.EMBEDHE).
//!
//! THE RULING (owner, 2026-11-12): "Use the local llm for Hebrew pairing." Both
//! embedding candidates failed the Hebrew calibration (`embeddings_calibration.rs`
//! records why), and `potion-base-4M` cannot read Hebrew at all, so Hebrew tweets
//! are paired by the model the `summarize` slot names, never by a vector:
//!
//! 1. A cheap prefilter picks the candidate pairs: the Jaccard overlap of the two
//!    tweets' character trigrams (each word padded with `#`, so a prefix letter
//!    such as `ה` or `ב` costs a gram or two, not the word) at or above
//!    [`PREFILTER_FLOOR`]. It is a recall filter, not the verdict.
//! 2. The model judges each candidate with a strict yes/no prompt ([`SYSTEM`]),
//!    loaded, used and released through the residency seam, so under the GPU
//!    lock and never outside it. An answer that is neither `yes` nor `no` is a
//!    failed verdict: the pair is kept apart and counted, never guessed.
//! 3. The yes pairs are joined into clusters (connected components).
//!
//! A model judges only when its calibration on the committed Hebrew fixture
//! passed ([`CALIBRATED_JUDGES`], each pinned by its recorded run under
//! `tests/fixtures/embeddings/`): every labelled pair yes, the neighbouring-story
//! traps no, every unrelated pair no. A summarize slot that names another model
//! keeps Hebrew apart and says so, so a switched model is never trusted on the
//! old model's numbers. `ztools models hebrew-pairing` records a model's run.

use std::collections::BTreeSet;

use anyhow::{Result, bail};

use crate::ztools::llm::{ChatBudget, ChatRequest, Sampling, chat_with};
use crate::ztools::twitter::Tweet;

#[path = "embeddings_hebrew_record.rs"]
pub mod record;

/// The judge's instructions. A recorded calibration holds the prompt it was
/// measured with, and the pinning test fails when this text moves.
pub const SYSTEM: &str = "You decide whether two tweets report the same news story: the same \
single event, announcement, decision, match or incident, told by two people. Tweets about the \
same topic, person, organisation or team but about different events are NOT the same story. \
Answer with exactly one word: yes or no.";

/// The lowest trigram overlap a pair needs to be judged.
///
/// Calibrated on the 42-tweet Hebrew fixture: its lowest labelled same-story pair overlaps 0.225,
/// and this floor sits well under it, because a miss here is a pair the model
/// never sees.
pub const PREFILTER_FLOOR: f64 = 0.08;

/// The most pairs one run judges, highest overlap first. The rest are counted.
pub const MAX_PAIRS: usize = 60;

/// Models whose Hebrew judging passed calibration on the committed fixture.
///
/// `raptor-v0.5-8b-a1b-jang_6m` (2026-11-11, every one of the fixture's 861 pairs
/// asked, load average 20-85): the pipeline gets all 861 right (8 of 8 labelled pairs
/// merged, 8 of 8 neighbouring-story traps and 845 of 845 unrelated pairs apart); the
/// model alone said a wrong yes on 3 unrelated pairs, all below the prefilter (overlap
/// 0.033-0.053). Median 0.10 s a pair, mean 0.48 s.
pub const CALIBRATED_JUDGES: &[&str] = &["raptor-v0.5-8b-a1b-jang_6m"];

/// Whether `model` (with or without an organisation prefix) passed calibration.
#[must_use]
pub fn judge_calibrated(model: &str, calibrated: &[&str]) -> bool {
    let bare = model.rsplit('/').next().unwrap_or(model);
    calibrated.iter().any(|m| m.eq_ignore_ascii_case(bare))
}

/// Mostly Hebrew letters.
#[must_use]
pub fn is_hebrew(text: &str) -> bool {
    let (he, letters) =
        text.chars()
            .filter(|c| c.is_alphabetic())
            .fold((0usize, 0usize), |(h, n), c| {
                (
                    h + usize::from(('\u{0590}'..='\u{05ff}').contains(&c)),
                    n + 1,
                )
            });
    letters > 0 && he * 2 >= letters
}

fn trigrams(text: &str) -> BTreeSet<String> {
    let mut grams = BTreeSet::new();
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        if word.is_empty() {
            continue;
        }
        let padded: Vec<char> = std::iter::once('#')
            .chain(word.chars())
            .chain(std::iter::once('#'))
            .collect();
        for w in padded.windows(3) {
            grams.insert(w.iter().collect());
        }
    }
    grams
}

/// The Jaccard overlap of two texts' character trigrams, 0.0 to 1.0.
#[must_use]
pub fn overlap(a: &str, b: &str) -> f64 {
    let (ga, gb) = (trigrams(a), trigrams(b));
    let union = ga.union(&gb).count();
    if union == 0 {
        return 0.0;
    }
    let shared = ga.intersection(&gb).count();
    f64::from(u32::try_from(shared).unwrap_or(u32::MAX))
        / f64::from(u32::try_from(union).unwrap_or(u32::MAX))
}

/// The pairs of `texts` at or above `floor`, highest overlap first (ties in
/// index order).
#[must_use]
pub fn candidates(texts: &[&str], floor: f64) -> Vec<(usize, usize)> {
    let mut scored = Vec::new();
    for i in 0..texts.len() {
        for j in i + 1..texts.len() {
            let o = overlap(texts[i], texts[j]);
            if o >= floor {
                scored.push((o, i, j));
            }
        }
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2))));
    scored.into_iter().map(|(_, i, j)| (i, j)).collect()
}

/// `yes` or `no`, in any case, with surrounding punctuation; anything else is
/// no verdict.
#[must_use]
pub fn parse_verdict(raw: &str) -> Option<bool> {
    let word = raw.trim().trim_matches(|c: char| !c.is_alphanumeric());
    if word.eq_ignore_ascii_case("yes") {
        Some(true)
    } else if word.eq_ignore_ascii_case("no") {
        Some(false)
    } else {
        None
    }
}

/// The user message for one pair.
#[must_use]
pub fn pair_prompt(a: &str, b: &str) -> String {
    format!("Tweet 1: {a}\nTweet 2: {b}\nSame story?")
}

/// Decides whether two texts tell the same story. The seam: production asks
/// the local model ([`LlmJudge`]); a unit test answers from a table.
pub trait Judge {
    /// # Errors
    ///
    /// When no verdict could be had (the call failed, or the answer was
    /// neither yes nor no).
    fn same_story(&mut self, a: &str, b: &str) -> Result<bool>;
}

/// The summarize slot's model on the Osaurus server, already loaded by the
/// residency seam that holds the GPU lock.
pub struct LlmJudge<'a> {
    pub base_url: &'a str,
    pub model: &'a str,
}

/// A verdict is one word: a few tokens, and a stall guard a loaded model
/// never comes near.
const JUDGE_BUDGET: ChatBudget = ChatBudget {
    stall_secs: 120,
    cap_secs: 300,
    max_tokens: 4,
};

/// The model's raw answer about one pair.
///
/// # Errors
///
/// As [`chat_with`].
pub fn ask(base_url: &str, model: &str, a: &str, b: &str) -> Result<String> {
    let user = pair_prompt(a, b);
    chat_with(
        &ChatRequest {
            base_url,
            model,
            system: Some(SYSTEM),
            user: &user,
            json: false,
        },
        &JUDGE_BUDGET,
        &Sampling::GREEDY,
    )
}

impl Judge for LlmJudge<'_> {
    fn same_story(&mut self, a: &str, b: &str) -> Result<bool> {
        let raw = ask(self.base_url, self.model, a, b)?;
        match parse_verdict(&raw) {
            Some(v) => Ok(v),
            None => bail!("{} answered {raw:?}, not yes or no", self.model),
        }
    }
}

/// What one run's pairing did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Pairing {
    /// Pairs the model gave a verdict on.
    pub judged: usize,
    /// The yes pairs, as indices into the texts.
    pub same: Vec<(usize, usize)>,
    /// Candidate pairs past [`MAX_PAIRS`], never asked.
    pub unjudged: usize,
    /// Pairs with no verdict, kept apart.
    pub failed: usize,
    /// The first failure, for the run's line.
    pub first_error: Option<String>,
}

/// Judge the candidate pairs of `texts`, at most `cap` of them.
pub fn pair(texts: &[&str], judge: &mut dyn Judge, cap: usize) -> Pairing {
    let pairs = candidates(texts, PREFILTER_FLOOR);
    let mut out = Pairing {
        unjudged: pairs.len().saturating_sub(cap),
        ..Pairing::default()
    };
    for (i, j) in pairs.into_iter().take(cap) {
        match judge.same_story(texts[i], texts[j]) {
            Ok(same) => {
                out.judged += 1;
                if same {
                    out.same.push((i, j));
                }
            }
            Err(e) => {
                out.failed += 1;
                out.first_error.get_or_insert_with(|| format!("{e:#}"));
            }
        }
    }
    out
}

/// Each index's component under `edges`, named by its lowest member.
#[must_use]
pub fn components(n: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    const fn find(root: &mut [usize], mut i: usize) -> usize {
        while root[i] != i {
            root[i] = root[root[i]];
            i = root[i];
        }
        i
    }
    let mut root: Vec<usize> = (0..n).collect();
    for &(a, b) in edges {
        let (ra, rb) = (find(&mut root, a), find(&mut root, b));
        root[ra.max(rb)] = ra.min(rb);
    }
    (0..n).map(|i| find(&mut root, i)).collect()
}

/// Join the singleton clusters `eligible` picks by `judge`'s yes pairs.
///
/// A joined cluster takes the place of its first member; every other cluster
/// keeps its place. Returns the clusters, how many tweets were eligible, and
/// the pairing.
pub fn join(
    clusters: Vec<Vec<Tweet>>,
    eligible: impl Fn(&Tweet) -> bool,
    judge: &mut dyn Judge,
    cap: usize,
) -> (Vec<Vec<Tweet>>, usize, Pairing) {
    let slots: Vec<usize> = clusters
        .iter()
        .enumerate()
        .filter(|(_, c)| c.len() == 1 && eligible(&c[0]))
        .map(|(i, _)| i)
        .collect();
    let texts: Vec<&str> = slots
        .iter()
        .map(|&i| clusters[i][0].text.as_str())
        .collect();
    let pairing = pair(&texts, judge, cap);
    let comp = components(slots.len(), &pairing.same);
    let mut groups: Vec<Vec<Tweet>> = vec![Vec::new(); slots.len()];
    for (k, &slot) in slots.iter().enumerate() {
        groups[comp[k]].push(clusters[slot][0].clone());
    }
    let position: std::collections::HashMap<usize, usize> =
        slots.iter().enumerate().map(|(k, &s)| (s, k)).collect();
    let mut out = Vec::with_capacity(clusters.len());
    for (i, cluster) in clusters.into_iter().enumerate() {
        match position.get(&i) {
            None => out.push(cluster),
            Some(&k) if comp[k] == k => out.push(std::mem::take(&mut groups[k])),
            Some(_) => {}
        }
    }
    (out, slots.len(), pairing)
}

/// What became of the Hebrew tweets the embedding model cannot read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Hebrew {
    /// There were none.
    None,
    /// Kept as their own clusters, for `reason`; `fault` when something
    /// failed, rather than a step not yet taken.
    Apart {
        tweets: usize,
        reason: String,
        fault: bool,
    },
    /// Paired by `model`.
    Paired {
        model: String,
        tweets: usize,
        pairing: Pairing,
    },
}

impl Hebrew {
    /// The clause the run's status line carries, if any.
    #[must_use]
    pub fn clause(&self) -> Option<String> {
        match self {
            Self::None => None,
            Self::Apart { tweets, reason, .. } => {
                Some(format!("{tweets} Hebrew tweets kept apart: {reason}"))
            }
            Self::Paired {
                model,
                tweets,
                pairing,
            } => {
                let mut parts = vec![format!(
                    "{tweets} Hebrew tweets paired by {model}: {} pairs judged, {} the same story",
                    pairing.judged,
                    pairing.same.len()
                )];
                if pairing.unjudged > 0 {
                    parts.push(format!(
                        "{} candidate pairs past the cap of {MAX_PAIRS} not judged",
                        pairing.unjudged
                    ));
                }
                if pairing.failed > 0 {
                    parts.push(format!(
                        "{} without a verdict kept apart ({})",
                        pairing.failed,
                        pairing.first_error.as_deref().unwrap_or("no reason")
                    ));
                }
                Some(parts.join(", "))
            }
        }
    }

    /// The tweets this accounts for.
    #[must_use]
    pub const fn tweets(&self) -> usize {
        match self {
            Self::None => 0,
            Self::Apart { tweets, .. } | Self::Paired { tweets, .. } => *tweets,
        }
    }
}

/// Pair the Hebrew singletons of `clusters` with `model`, the summarize
/// slot's, when it is in `calibrated`. `unreadable` says which tweets the
/// embedding model could not compare.
///
/// The model is loaded, used and released through the residency seam, which
/// holds the GPU lock throughout; a run with fewer than two Hebrew tweets, or
/// no candidate pair, loads nothing.
pub fn pair_hebrew(
    clusters: Vec<Vec<Tweet>>,
    unreadable: impl Fn(&Tweet) -> bool,
    base_url: &str,
    model: &str,
    calibrated: &[&str],
    timing: crate::ztools::residency::Timing,
) -> (Vec<Vec<Tweet>>, Hebrew) {
    let eligible = |t: &Tweet| unreadable(t) && is_hebrew(&t.text);
    let texts: Vec<&str> = clusters
        .iter()
        .filter(|c| c.len() == 1 && eligible(&c[0]))
        .map(|c| c[0].text.as_str())
        .collect();
    let tweets = texts.len();
    if tweets == 0 {
        return (clusters, Hebrew::None);
    }
    let apart = |reason: String, fault: bool| Hebrew::Apart {
        tweets,
        reason,
        fault,
    };
    if !judge_calibrated(model, calibrated) {
        let reason = format!(
            "the summarize model '{model}' has no Hebrew same-story calibration \
             (ztools models hebrew-pairing records one)"
        );
        return (clusters, apart(reason, false));
    }
    if candidates(&texts, PREFILTER_FLOOR).is_empty() {
        let pairing = Pairing::default();
        let model = model.to_string();
        return (
            clusters,
            Hebrew::Paired {
                model,
                tweets,
                pairing,
            },
        );
    }
    // The release comes back beside the value, which is every tweet of the
    // run: a model that outstays the pairing must not cost the clusters.
    let mut kept = Some(clusters);
    let run = crate::ztools::residency::with_model_kept(base_url, model, timing, |_| {
        let mut judge = LlmJudge { base_url, model };
        let all = kept.take().unwrap_or_default();
        Ok(join(all, eligible, &mut judge, MAX_PAIRS))
    });
    match run {
        Ok(crate::ztools::residency::Ran {
            out: Ok((joined, _, pairing)),
            released,
        }) => {
            released.report("the Hebrew pairing is kept");
            let model = model.to_string();
            (
                joined,
                Hebrew::Paired {
                    model,
                    tweets,
                    pairing,
                },
            )
        }
        // The body cannot fail; a load that failed never ran it, so the
        // clusters are still here.
        Ok(crate::ztools::residency::Ran { out: Err(e), .. }) | Err(e) => {
            (kept.unwrap_or_default(), apart(format!("{e:#}"), true))
        }
    }
}

#[cfg(test)]
#[path = "embeddings_hebrew_judge_tests.rs"]
mod tests;
