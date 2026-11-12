//! The decision rule of LM.DESIGN section 4, and section 3's cracked band.
//!
//! Switch slot S from incumbent I to candidate C only when all of these hold:
//! 1. C passes every gate (complete run, no gate-task regression against I,
//!    refusal at most I's, the speed floor).
//! 2. `score_C >= score_I + max(3.0, 2*sd)`, or the cracked band applies: C is
//!    refusal-free (at most 2% refusals), I is not, and `score_C >= score_I - 2.0`.
//! 3. A second measure agrees: grader preference at least 50%, or speed at least
//!    I's, or an outside nomination plus Hebrew at least I's. Disagreement keeps.
//!
//! Ties keep the incumbent. Everything here is a pure function of [`Measure`]s,
//! so the same rule decides the nightly and the replay of `eval_history.json`.

use std::collections::BTreeMap;

/// The rule's numbers. [`Rule::DESIGN`] is the one the owner accepted (D3, D4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rule {
    /// The least a candidate must gain, in slot-score points.
    pub margin_floor: f64,
    /// How many run-to-run spreads the gain must also clear.
    pub sd_mult: f64,
    /// How far below a refusal-free model may score and still win (section 3).
    pub cracked_band: f64,
    /// The refusal share at or under which a model is refusal-free.
    pub refusal_free: f64,
    /// `summarize`: the slowest task, cold load included, in seconds.
    pub summarize_max_seconds: f64,
    /// `think`: decode at 8k, tokens a second.
    pub think_min_tps: f64,
}

impl Rule {
    /// LM.DESIGN D3/D4 as accepted 2026-11-11.
    pub const DESIGN: Self = Self {
        margin_floor: 3.0,
        sd_mult: 2.0,
        cracked_band: 2.0,
        refusal_free: 0.02,
        summarize_max_seconds: 600.0,
        think_min_tps: 8.0,
    };

    /// The gain a candidate must clear given both sides' spreads.
    #[must_use]
    pub fn needed(&self, sd_c: f64, sd_i: f64) -> f64 {
        self.margin_floor.max(self.sd_mult * sd_c.max(sd_i))
    }
}

/// One model on one slot: what the eval measured of it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Measure {
    pub model: String,
    /// The slot score: the mean over repeats of the mean over the slot's tasks.
    pub score: f64,
    /// Sample standard deviation of the slot score over the repeats (0 for one).
    pub sd: f64,
    pub repeats: usize,
    /// Gate task to its score.
    pub gates: BTreeMap<String, f64>,
    /// Slot tasks (and, in the nightly, refusal tasks) with no score.
    pub missing: Vec<String>,
    /// Share of probe items refused, 0..=1; `None` when never measured.
    pub refusal: Option<f64>,
    /// Mean wall seconds per slot task.
    pub seconds: Option<f64>,
    /// The slowest slot task, wall seconds.
    pub max_seconds: Option<f64>,
    /// Decode tokens a second at 8k, from `osaurus bench`.
    pub decode_tps: Option<f64>,
    /// Blind pairwise grader preference for this model over the incumbent, 0..=1.
    pub grader: Option<f64>,
    /// An outside source nominated it.
    pub nominated: bool,
    /// The Hebrew tasks' mean.
    pub hebrew: Option<f64>,
}

impl Measure {
    /// At most `rule.refusal_free` of the probe refused. Unmeasured is never free.
    #[must_use]
    pub fn refusal_free(&self, rule: &Rule) -> bool {
        self.refusal.is_some_and(|r| r <= rule.refusal_free + 1e-9)
    }
}

/// A task whose regression against the incumbent is a gate (section 4).
#[must_use]
pub fn is_gate_task(task: &str) -> bool {
    task.ends_with("_injection")
        || matches!(
            task,
            "summarize_contradiction"
                | "summarize_factual_accuracy"
                | "weekend_fabrication"
                | "json"
        )
}

/// Every gate `c` fails against `i` on `slot`; empty when it passes them all.
#[must_use]
pub fn gate_failures(slot: &str, c: &Measure, i: &Measure, rule: &Rule) -> Vec<String> {
    let mut out = Vec::new();
    if !c.missing.is_empty() {
        out.push(format!("incomplete ({} unscored)", c.missing.join(", ")));
    }
    for (task, want) in &i.gates {
        match c.gates.get(task) {
            Some(got) if got + 1e-9 >= *want => {}
            Some(got) => out.push(format!("{task} {got:.0} < {want:.0}")),
            None => out.push(format!("{task} unscored")),
        }
    }
    if let (Some(rc), Some(ri)) = (c.refusal, i.refusal)
        && rc > ri + 1e-9
    {
        out.push(format!("refuses {:.1}% > {:.1}%", rc * 100.0, ri * 100.0));
    }
    if slot == "summarize"
        && let Some(s) = c.max_seconds
        && s > rule.summarize_max_seconds
    {
        out.push(format!(
            "slowest task {s:.0}s > {:.0}s",
            rule.summarize_max_seconds
        ));
    }
    if slot == "think"
        && let Some(t) = c.decode_tps
        && t < rule.think_min_tps
    {
        out.push(format!("decode {t:.1} tok/s < {:.1}", rule.think_min_tps));
    }
    out
}

/// The second measure that agrees with switching to `c`, if any.
#[must_use]
pub fn second_measure(c: &Measure, i: &Measure) -> Option<String> {
    if let Some(g) = c.grader
        && g >= 0.5
    {
        return Some(format!("grader {:.0}%", g * 100.0));
    }
    if let (Some(sc), Some(si)) = (c.seconds, i.seconds)
        && sc <= si
    {
        return Some(format!("speed {sc:.1}s <= {si:.1}s a task"));
    }
    if c.nominated
        && let (Some(hc), Some(hi)) = (c.hebrew, i.hebrew)
        && hc >= hi
    {
        return Some(format!("nominated, Hebrew {hc:.1} >= {hi:.1}"));
    }
    None
}

/// What the rule said of one candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct Judged {
    pub model: String,
    pub score: f64,
    /// `score_C - score_I`.
    pub delta: f64,
    /// `max(floor, mult*sd)`.
    pub needed: f64,
    pub gates: Vec<String>,
    /// The cracked band carried condition 2.
    pub band: bool,
    pub second: Option<String>,
    /// Conditions 1-3 all hold.
    pub wins: bool,
    /// One line: why it wins or keeps.
    pub why: String,
}

/// One slot's decision.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub slot: String,
    pub incumbent: String,
    /// The incumbent, or the candidate it switches to.
    pub chosen: String,
    pub judged: Vec<Judged>,
}

impl Decision {
    #[must_use]
    pub fn switched(&self) -> bool {
        self.chosen != self.incumbent
    }
}

/// Judge one candidate against the incumbent.
#[must_use]
pub fn judge(slot: &str, c: &Measure, i: &Measure, rule: &Rule) -> Judged {
    let gates = gate_failures(slot, c, i, rule);
    let delta = c.score - i.score;
    let needed = rule.needed(c.sd, i.sd);
    let by_margin = delta > 1e-9 && delta + 1e-9 >= needed;
    let band = !by_margin
        && c.refusal_free(rule)
        && !i.refusal_free(rule)
        && c.score + 1e-9 >= i.score - rule.cracked_band;
    let second = second_measure(c, i);
    let wins = gates.is_empty() && (by_margin || band) && second.is_some();
    let why = if !gates.is_empty() {
        format!("gated: {}", gates.join("; "))
    } else if !(by_margin || band) {
        format!("{delta:+.1} inside the {needed:.1} margin")
    } else if let Some(s) = &second {
        let by = if band { "cracked band" } else { "margin" };
        format!("{delta:+.1} by {by} ({needed:.1}), {s}")
    } else {
        format!("{delta:+.1} clears, but no second measure agrees")
    };
    Judged {
        model: c.model.clone(),
        score: c.score,
        delta,
        needed,
        gates,
        band,
        second,
        wins,
        why,
    }
}

/// Section 3's order among candidates that pass: a refusal-free one beats a
/// non-refusal-free one within the band, otherwise the higher score; a name
/// breaks a tie so the order never depends on input order.
fn preferred(
    a: &Judged,
    ma: &Measure,
    b: &Judged,
    mb: &Measure,
    rule: &Rule,
) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let (fa, fb) = (ma.refusal_free(rule), mb.refusal_free(rule));
    if fa && !fb && a.score + 1e-9 >= b.score - rule.cracked_band {
        return Ordering::Less;
    }
    if fb && !fa && b.score + 1e-9 >= a.score - rule.cracked_band {
        return Ordering::Greater;
    }
    b.score
        .total_cmp(&a.score)
        .then_with(|| a.model.cmp(&b.model))
}

/// Decide `slot`: keep `incumbent`, or switch to the most preferred winner.
#[must_use]
pub fn decide(slot: &str, incumbent: &Measure, candidates: &[Measure], rule: &Rule) -> Decision {
    let mut judged: Vec<(Judged, &Measure)> = candidates
        .iter()
        .filter(|c| c.model != incumbent.model)
        .map(|c| (judge(slot, c, incumbent, rule), c))
        .collect();
    judged.sort_by(|(a, ma), (b, mb)| {
        b.wins
            .cmp(&a.wins)
            .then_with(|| preferred(a, ma, b, mb, rule))
    });
    let chosen = judged
        .first()
        .filter(|(j, _)| j.wins)
        .map_or_else(|| incumbent.model.clone(), |(j, _)| j.model.clone());
    Decision {
        slot: slot.to_string(),
        incumbent: incumbent.model.clone(),
        chosen,
        judged: judged.into_iter().map(|(j, _)| j).collect(),
    }
}
