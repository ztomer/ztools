//! Class-level fixtures for the summary quality gate: one per degenerate shape
//! the summarizer has actually saved, plus the healthy shapes that must still
//! pass. Each degenerate fixture is built to trip ONE rule, so a rule that
//! stops firing turns exactly its own test red.

use super::*;

/// A healthy bullet for tweet `i`: distinct words, cited in the required form.
fn cited(i: usize) -> String {
    let topics = [
        "rail line closed between two stations",
        "funding round closes friday for small groups",
        "compiler release adds const generics",
        "city council votes on the new budget",
        "storm warning issued for the lakeshore",
        "museum opens a weekend exhibit for kids",
        "chip maker reports record data center revenue",
    ];
    format!(
        "- Item {i}: {} (@account_{i} | 0{}:00)",
        topics[i % topics.len()],
        i % 10
    )
}

fn doc(bullets: &[String]) -> String {
    format!(
        "## Executive Summary\n\nThe timeline covered several unrelated stories today.\n\n## Topics\n\n{}",
        bullets.join("\n")
    )
}

/// The shape seen in production: 43 tweets in, 58 bullets out, 35 of them
/// repeats of a handful of lines, all saved as primary output.
fn the_production_loop() -> String {
    let mut lines: Vec<String> = (0..23).map(cited).collect();
    for i in 0..35 {
        lines.push(cited(i % 5));
    }
    doc(&lines)
}

#[test]
fn a_healthy_cited_summary_passes() {
    let lines: Vec<String> = (0..6).map(cited).collect();
    let q = check_summary_quality(&doc(&lines), 10);
    assert!(!q.rejected(), "{q:?}");
}

#[test]
fn the_production_repetition_loop_is_rejected() {
    let q = check_summary_quality(&the_production_loop(), 43);
    assert!(q.rejected(), "a 58-bullet loop over 43 tweets was accepted");
    assert!(
        q.rejections
            .iter()
            .any(|r| r.contains("repeat an earlier bullet")),
        "{q:?}"
    );
}

/// Duplicates alone, with fewer bullets than tweets and every bullet cited, so
/// only the duplicate rule can reject it.
#[test]
fn exact_duplicates_beyond_the_allowance_are_rejected_on_their_own() {
    let mut lines: Vec<String> = (0..4).map(cited).collect();
    for _ in 0..=MAX_DUPLICATE_BULLETS {
        lines.push(cited(1));
    }
    let q = check_summary_quality(&doc(&lines), 100);
    assert_eq!(q.rejections.len(), 1, "{q:?}");
    assert!(q.rejections[0].contains("3 of 7 bullets repeat"), "{q:?}");
}

#[test]
fn a_small_number_of_duplicates_is_tolerated() {
    let mut lines: Vec<String> = (0..4).map(cited).collect();
    for _ in 0..MAX_DUPLICATE_BULLETS {
        lines.push(cited(1));
    }
    let q = check_summary_quality(&doc(&lines), 100);
    assert!(!q.rejected(), "{q:?}");
}

/// A loop rarely repeats byte for byte: case, punctuation and one changed word
/// must not make a repeat look new.
#[test]
fn near_duplicates_count_as_duplicates() {
    let base = "- The compiler release adds const generics and faster builds for everyone (@rust_lang | 19:00)";
    let lines = vec![
        base.to_string(),
        base.to_uppercase().replace("@RUST_LANG", "@rust_lang"),
        "- The compiler release adds const generics, and faster builds for everyone! (@rust_lang | 19:00)"
            .to_string(),
        "- The compiler release adds const generics and faster builds for anyone (@rust_lang | 19:00)"
            .to_string(),
    ];
    assert_eq!(duplicate_bullets(&bullets(&doc(&lines))), 3);
}

/// Two different facts from one account share the citation and little else.
#[test]
fn distinct_facts_from_one_account_are_not_duplicates() {
    let lines = vec![
        "- Rust 1.95 is released with const generics (@rust_lang | 19:00)".to_string(),
        "- The 2027 edition call for proposals opens next month (@rust_lang | 19:05)".to_string(),
    ];
    assert_eq!(duplicate_bullets(&bullets(&doc(&lines))), 0);
}

#[test]
fn more_bullets_than_input_tweets_is_rejected_on_its_own() {
    let lines: Vec<String> = (0..5).map(cited).collect();
    let q = check_summary_quality(&doc(&lines), 4);
    assert_eq!(q.rejections.len(), 1, "{q:?}");
    assert!(
        q.rejections[0].contains("5 bullets for 4 input tweets"),
        "{q:?}"
    );
    assert!(!check_summary_quality(&doc(&lines), 5).rejected());
}

/// The bracket shape seen in production: `[@handle | ts]` where the prompt
/// requires `(@handle | ts)`.
#[test]
fn square_bracket_attribution_is_rejected() {
    let lines: Vec<String> = (0..5)
        .map(|i| cited(i).replace('(', "[").replace(')', "]"))
        .collect();
    let q = check_summary_quality(&doc(&lines), 10);
    assert_eq!(q.rejections.len(), 1, "{q:?}");
    assert!(
        q.rejections[0].contains("5 of 5 bullets lack the `(@handle | timestamp)`"),
        "{q:?}"
    );
}

/// "Mostly" means more than half: a minority of uncited bullets is weak, not a
/// different format.
#[test]
fn a_minority_of_uncited_bullets_passes_and_a_majority_rejects() {
    let mut lines: Vec<String> = (0..3).map(cited).collect();
    lines.push("- an uncited aside about the weather".to_string());
    lines.push("- another uncited aside about traffic".to_string());
    assert!(!check_summary_quality(&doc(&lines), 10).rejected());
    lines.push("- a third uncited aside about sports".to_string());
    lines.push("- a fourth uncited aside about music".to_string());
    assert!(check_summary_quality(&doc(&lines), 10).rejected());
}

/// A citation wrapped onto the bullet's continuation line, and a merged bullet
/// citing two sources, are both attributed.
#[test]
fn wrapped_and_merged_citations_are_attributed() {
    let text = "## Topic\n\n- A rail shutdown on the Lakeshore line, with shuttle buses\n  between Union and Bathurst (@transit_watch | 12:00).\n- Two accounts report the vote (@a | 08:00) (@b | 08:15)\n";
    let found = bullets(text);
    assert_eq!(found.len(), 2, "{found:?}");
    assert!(found.iter().all(|b| is_attributed(b)), "{found:?}");
}

/// A header line or an unindented paragraph ends a bullet: its text must not
/// lend the bullet a citation it does not carry.
#[test]
fn a_following_paragraph_does_not_join_the_bullet() {
    let text = "## Topic\n- uncited bullet\nA paragraph (@x | 1).\n## Next (@y | 2)\n";
    let found = bullets(text);
    assert_eq!(found, vec!["uncited bullet".to_string()]);
}

#[test]
fn the_structural_floor_still_holds() {
    let empty = check_summary_quality("", 3);
    assert!(empty.rejected());
    assert!(empty.rejections[0].contains("empty"));

    let raw = check_summary_quality("Just raw text without headers", 3);
    assert!(raw.rejected());
    assert!(
        raw.warnings.iter().any(|w| w.contains("headers")),
        "{raw:?}"
    );

    let short = check_summary_quality("## Short Header\n- Bullet item (@a | 1)", 3);
    assert!(!short.rejected(), "{short:?}");
    assert!(short.warnings.iter().any(|w| w.contains("Very short")));
}

/// A run with no tweets has nothing to cite: any bullet is an invention. This
/// is the "Please provide the timeline" document saved five times in August.
#[test]
fn bullets_with_zero_input_tweets_are_rejected() {
    let q = check_summary_quality(&doc(&[cited(0)]), 0);
    assert!(q.rejected(), "{q:?}");
}

/// The live shape of 2026-10-10: a healthy-looking answer whose LAST bullet
/// stops mid-citation. Every other rule passes it (7 of 8 cited, no repeats,
/// fewer bullets than tweets), so only the truncation rule can reject it.
#[test]
fn an_answer_cut_off_mid_citation_is_rejected() {
    let mut lines: Vec<String> = (0..7).map(cited).collect();
    lines.push(
        "- A game announcement was posted on Steam. (@AION2Official | Sat Oct 10 336:51 +..."
            .into(),
    );
    let q = check_summary_quality(&doc(&lines), 45);
    assert_eq!(q.rejections.len(), 1, "{q:?}");
    assert!(q.rejections[0].contains("cut off"), "{q:?}");

    // A complete last bullet that merely mentions an ellipsis is not cut.
    let mut whole: Vec<String> = (0..7).map(cited).collect();
    whole.push("- He said \"wait...\" and left. (@someone | 09:00)".into());
    assert!(!check_summary_quality(&doc(&whole), 45).rejected());
}

/// The tweets `cited(i)` cites, as the prompt showed them.
fn sources(n: usize) -> Vec<(String, String)> {
    (0..n)
        .map(|i| (format!("account_{i}"), format!("0{}:00", i % 10)))
        .collect()
}

/// Ground truth, not a shape: every citation must name a tweet the model was
/// given. Live on 2026-10-10 the summarizer wrote `(@danielamram3 | Sat Oct 10
/// 15:33:43 +0000 2о)` (a Cyrillic о) and `(@unusual_whales | Sat Oct 1`, and
/// both passed, because "is it cited" only checked the parentheses.
#[test]
fn citations_that_name_no_input_tweet_are_rejected() {
    let healthy: Vec<String> = (0..7).map(cited).collect();
    assert_eq!(unmatched_citations(&doc(&healthy), &sources(7)), None);

    let mut garbled: Vec<String> = (0..5).map(cited).collect();
    garbled.push("- A shot was fired over parking. (@account_5 | 05:0о)".into());
    garbled.push("- Elections were postponed. (@nobody_given | 06:00)".into());
    let why = unmatched_citations(&doc(&garbled), &sources(7)).expect("rejected");
    assert!(why.contains("2 of 7 citations"), "{why}");
}

/// The model thinking aloud inside a topic section ("Actually wait - I realize
/// I've been overthinking this") is not a summary. Prose belongs only in the
/// Executive Summary.
#[test]
fn prose_inside_a_topic_section_is_rejected() {
    let mut lines: Vec<String> = (0..6).map(cited).collect();
    lines.push(String::new());
    lines.push("Actually wait - I realize I've been overthinking this.".into());
    let q = check_summary_quality(&doc(&lines), 10);
    assert_eq!(q.rejections.len(), 1, "{q:?}");
    assert!(q.rejections[0].contains("talking to itself"), "{q:?}");
    // The Executive Summary paragraph `doc` writes is prose, and allowed.
    let healthy: Vec<String> = (0..6).map(cited).collect();
    assert!(!check_summary_quality(&doc(&healthy), 10).rejected());
}

/// Live on 2026-10-10 one tweet came back as three bullets (two identical) and
/// two more as two each -- too few repeats to read as a loop, so the gate
/// passed them. A bullet whose every citation names a tweet an earlier bullet
/// already cited adds no source the reader lacks: it is dropped, and a topic
/// left with no bullets loses its header. A bullet that cites one new tweet
/// alongside a cited one stays.
#[test]
fn bullets_that_only_recite_cited_tweets_are_dropped() {
    let answer = "## Executive Summary\nPolicy and transit (@a | 1).\n\n\
                  ## Policy\n- policy fact (@a | 1)\n- transit fact (@b | 2)\n\n\
                  ## Transit\n- transit fact, retold\n  (@b | 2)\n- merged (@b | 2) (@c | 3)\n\n\
                  ## Other\n- policy fact (@a | 1)\n";
    let (kept, dropped) = drop_recited(answer);
    assert_eq!(dropped, 2, "{kept}");
    assert_eq!(
        kept,
        "## Executive Summary\nPolicy and transit (@a | 1).\n\n\
         ## Policy\n- policy fact (@a | 1)\n- transit fact (@b | 2)\n\n\
         ## Transit\n- merged (@b | 2) (@c | 3)\n"
    );
}

/// Nothing repeated: the answer is returned byte for byte.
#[test]
fn an_answer_with_no_recited_tweet_is_unchanged() {
    let answer = "## Topic\n- one (@a | 1)\n- two (@b | 2)\n- uncited\n";
    assert_eq!(drop_recited(answer), (answer.to_string(), 0));
}
