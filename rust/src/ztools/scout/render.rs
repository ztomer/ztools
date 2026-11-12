//! A [`Proposal`] as the lines `ztools models scout` prints.

use super::{Candidate, Check, Proposal};

fn candidate_line(c: &Candidate) -> String {
    let head = format!("{} [{}]", c.repo, c.nominated);
    if !c.provenance.passed() {
        return format!("  ✗ {head}: provenance: {}", c.provenance.text());
    }
    let Some(format) = &c.format else {
        return format!("  ⚠ {head}: format not reached");
    };
    if !format.passed() {
        return format!("  {} {head}: format: {}", format.glyph(), format.text());
    }
    let part = |label: &str, check: Option<&Check>| {
        check.map_or_else(
            || format!("{label} ⚠ not reached"),
            |ch| format!("{label} {} {}", ch.glyph(), ch.text()),
        )
    };
    let glyph = if c.passes() {
        "✓"
    } else if [&c.fit, &c.disk]
        .iter()
        .any(|x| matches!(x, Some(Check::Fail(_))))
    {
        "✗"
    } else {
        "⚠"
    };
    format!(
        "  {glyph} {head}: {}; {}; {}",
        format.text(),
        part("fit", c.fit.as_ref()),
        part("disk", c.disk.as_ref())
    )
}

/// The report, one line per fact.
#[must_use]
pub fn render(p: &Proposal) -> Vec<String> {
    let mut out =
        vec!["scout (propose-only): nothing is downloaded, installed or switched".to_string()];
    let answered = p.asked - p.refused.len();
    if p.refused.is_empty() {
        out.push(format!("✓ feeds: {answered} of {} answered", p.asked));
    } else {
        out.push(format!(
            "⚠ feeds: {answered} of {} answered; {} unverified",
            p.asked,
            p.refused.len()
        ));
        for (key, why) in &p.refused {
            out.push(format!("  ⚠ unverified {key}: {why}"));
        }
    }
    out.extend(p.notes.iter().cloned());
    for inc in &p.incumbents {
        let slots = inc.slots.join(", ");
        let Some(repo) = &inc.repo else {
            out.push(format!(
                "✗ {slots}: {} is not installed (models refcheck)",
                inc.name
            ));
            continue;
        };
        let family = match &inc.family {
            Ok(base) => format!("family {base}"),
            Err(why) => format!("family unverified: {why}"),
        };
        out.push(format!("→ {slots}: incumbent {repo} ({family})"));
        if inc.retired {
            out.push(format!(
                "  ⚠ {repo} is retired from the Osaurus catalog; its successors are nominated first"
            ));
        }
        out.extend(inc.candidates.iter().map(candidate_line));
    }
    out.push(p.stage.as_ref().map_or_else(
        || "→ would stage: nothing passes every filter".to_string(),
        |repo| {
            format!(
                "→ would stage: {repo} (one candidate; the local eval decides, this list only \
                 nominates)"
            )
        },
    ));
    out
}
