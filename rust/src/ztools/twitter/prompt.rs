//! The summary prompt: the shared instructions, the timeline that fits, and
//! the citation rule worked on that timeline's own first line.
//!
//! Split out of `mod.rs` for the 500-line cap.

use super::{Tweet, deduplicate_tweets};

/// Build executive summary prompt for LLM. `instructions` is the shared
/// instruction block (canonical text: `conf/prompts.toml` `[twitter.summarize]`).
#[must_use]
pub fn build_prompt(tweets: &[Tweet], max_chars: usize, instructions: &str) -> (String, usize) {
    let (prompt, shown) = build_prompt_shown(tweets, max_chars, instructions);
    (prompt, shown.len())
}

/// The prompt, and the tweets it SHOWS, in timeline order: exactly what an
/// answer may cite.
///
/// A line that does not fit `max_chars` is skipped and a later, shorter one
/// may still fit, so the shown tweets are not a prefix of the input. The
/// caller used to take `tweets[..shown]` as the citable set, which named the
/// wrong tweets whenever the budget skipped one (2026-11-09).
///
/// The prompt closes with the citation rule worked on the timeline's OWN
/// first source line. The rule's example used to be `@TechCrunch | 08:00`,
/// and raptor cited that example handle on every bullet of two timelines that
/// held no such account (2026-10-30: 54 of 54 citations; 2026-11-08: 31 of 33).
/// An example drawn from the timeline cannot name an account it lacks.
#[must_use]
pub fn build_prompt_shown(
    tweets: &[Tweet],
    max_chars: usize,
    instructions: &str,
) -> (String, Vec<Tweet>) {
    let deduped = deduplicate_tweets(tweets);
    let mut lines = Vec::new();
    let mut shown = Vec::new();
    let mut used = 0;

    for t in deduped.iter().rev() {
        let mut prefix_parts = vec![source_prefix(t)];
        if t.favorite_count > 0 || t.retweet_count > 0 {
            prefix_parts.push(format!(
                "{} favs, {} RTs",
                t.favorite_count, t.retweet_count
            ));
        }
        if let Some(ref r) = t.reply_to {
            prefix_parts.push(format!("-> @{r}"));
        }
        let line = format!("[{}]: {}", prefix_parts.join(" | "), t.text.trim());
        if used + line.len() + 1 > max_chars {
            continue;
        }
        used += line.len() + 1;
        lines.push(line);
        shown.push(t.clone());
    }
    lines.reverse();
    shown.reverse();
    let timeline = lines.join("\n");
    let example = shown.first().map_or_else(String::new, |t| {
        format!(
            "\n\nCite each tweet by its source line's first two fields only: the line \
             beginning `[{p} | ...]:` is cited `({p})`.",
            p = source_prefix(t)
        )
    });

    let prompt = format!(
        "{instructions}\n\n\
        <timeline>\n{timeline}\n</timeline>{example}"
    );

    (prompt, shown)
}

/// A source line's first two fields, which are also its citation's body.
fn source_prefix(t: &Tweet) -> String {
    format!("@{} | {}", t.screen_name, t.created_at)
}
