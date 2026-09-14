//! Summary-quality scorer with misattribution and placeholder caps.
//!
//! Port of `lib/validators/text_validator.py::validate_summary`. Scores
//! structure, user coverage, specificity, synthesis, and topic coverage out
//! of 100, then applies two hard caps: a placeholder leak (`Mon DD`, `HH:MM`
//! and friends) caps at 40, and any misattributed bullet caps at 45 —
//! misattribution is disqualifying, not a deduction, because a plausible
//! wrong author gets believed while template text gets discarded.

pub const MISATTRIBUTION_MAX_SCORE: i64 = 45;
const PLACEHOLDER_LEAK_MAX_SCORE: i64 = 40;
const MAX_SCORE: i64 = 100;

const STRUCT_HEADERS_BULLETS_SCORE: i64 = 15;
const STRUCT_HEADERS_ONLY_SCORE: i64 = 10;
const STRUCT_BULLET_LONG_LEN: usize = 300;
const STRUCT_BULLET_LONG_SCORE: i64 = 8;
const STRUCT_BULLET_SHORT_LEN: usize = 200;
const STRUCT_BULLET_SHORT_SCORE: i64 = 5;
const USERS_COVERAGE_HIGH_COUNT: usize = 3;
const USERS_COVERAGE_HIGH_SCORE: i64 = 15;
const USERS_COVERAGE_MED_COUNT: usize = 2;
const USERS_COVERAGE_MED_SCORE: i64 = 10;
const USERS_COVERAGE_LOW_COUNT: usize = 1;
const USERS_COVERAGE_LOW_SCORE: i64 = 5;
const TIMESTAMP_SPECIFICITY_SCORE: i64 = 10;
const MAX_NARRATIVE_SPECIFICITY_SCORE: i64 = 15;
const NARRATIVE_WORD_SCORE_MULTIPLIER: i64 = 5;
const TEMPLATE_DRIVEN_FIELD_LIMIT: usize = 3;
const BOILERPLATE_SYNTHESIS_SCORE: i64 = 5;
const DEFAULT_SYNTHESIS_SCORE: i64 = 10;
const SYNTHESIS_MATCH_BONUS: i64 = 10;
const TOPIC_COVERAGE_HIGH_COUNT: usize = 2;
const TOPIC_COVERAGE_HIGH_SCORE: i64 = 25;
const TOPIC_COVERAGE_MED_COUNT: usize = 1;
const TOPIC_COVERAGE_MED_SCORE: i64 = 15;
const TOPIC_TRANSITION_WORD_SCORE: i64 = 10;

use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

use super::attribution::attribution_faithfulness;
use crate::ztools::eval::validate::has_text_headers;

static NARRATIVE_WORDS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:ask(?:s|ed|ing)?|respond(?:s|ed|ing)?|thank(?:s|ed|ing)?|report(?:s|ed|ing)?|confirm(?:s|ed|ing)?|direct(?:s|ed|ing)?|inquire(?:s|d|ing)?|announce(?:s|d|ing)?|share(?:s|d|ing)?|request(?:s|ed|ing)?|provide(?:s|d|ing)?)\b",
    )
    .expect("narrative words regex is static")
});
static TEMPLATE_FIELDS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\*\*(Who|What|When|Where):").expect("template fields regex is static")
});
static BOILERPLATE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(not specified|n/a|unknown|not provided)")
        .expect("boilerplate regex is static")
});
static NEWLINE_HEADER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\n#{2,}\s+\w+").expect("newline header regex is static"));
static START_HEADER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^#{2,}\s+\w+").expect("start header regex is static"));
static SYNTHESIS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(overall|summary|in (short|summary)|key (points?|takeaways?)|tl;dr|(the|this|that) (conversation|discussion|thread|interaction))",
    )
    .expect("synthesis regex is static")
});
static LEADING_SUMMARY_HEAD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^#{2,}\s+(?:executive\s+summary|summary|overview|tl;?dr)\s*\n")
        .expect("leading summary head regex is static")
});
static SECTION_BREAK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\n#{2,}\s").expect("section break regex is static"));
static TOPIC_MARKERS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^#{2,}\s+\w+|^[A-Z][^a-z]{2,}:\s").expect("topic markers regex is static")
});
static TRANSITION_WORDS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(first|second|third|then|also|additionally|meanwhile)")
        .expect("transition words regex is static")
});
static PLACEHOLDER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"Mon DD|DD HH|HH:MM|@username|@handle|<handle>|YYYY-MM-DD|\{\w+\}")
        .expect("placeholder regex is static")
});
static TIMESTAMP_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\d{1,2}:\d{2}").expect("timestamp regex is static"));
static HANDLE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"@([A-Za-z][A-Za-z0-9_]{1,})").expect("handle regex is static"));
static LEGACY_USER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[Uu]ser\s*(\d+)\b").expect("legacy user regex is static"));

/// Distinct people referenced, counting real `@handles` and legacy `user N`.
///
/// The handle pattern needs a preceding-character guard (`(?<![\w.])` in
/// Python) that the `regex` crate cannot express, so the guard is checked by
/// hand: a match preceded by a word character or `.` (as in an email address)
/// is not a handle.
fn count_distinct_users(text: &str) -> usize {
    let mut handles = HashSet::new();
    for m in HANDLE_RE.find_iter(text) {
        let preceding_ok = text[..m.start()]
            .chars()
            .next_back()
            .is_none_or(|c| !(c.is_alphanumeric() || c == '_' || c == '.'));
        if preceding_ok {
            handles.insert(m.as_str()[1..].to_lowercase());
        }
    }
    handles.remove("user");
    for caps in LEGACY_USER_RE.captures_iter(text) {
        handles.insert(format!("user{}", &caps[1]));
    }
    handles.len()
}
/// Score summaries on structure, user coverage, depth, synthesis, and topics.
///
/// A placeholder leak caps at 40 and any misattributed bullet caps at 45;
/// both return-or-cap rather than deducting, because neither failure mode is
/// offset by polish elsewhere.
///
/// Each scoring arm is its own function so the 100-line cap holds and a
/// future arm can be pinned without re-reading the whole scorer; all arms
/// append to the shared failure list in Python order.
#[must_use]
pub fn validate_summary(data: &str, source_text: &str) -> (i64, String) {
    if data.is_empty() {
        return (0, "empty response".to_string());
    }
    let data_str = data.trim();
    let mut failures = Vec::new();
    let mut score: i64 = 0;

    score += structure_points(data_str, &mut failures);
    score += coverage_points(data_str, &mut failures);
    let (specificity, faithful, total_bullets) =
        specificity_points(data_str, source_text, &mut failures);
    score += specificity;
    score += synthesis_points(data_str, &mut failures);
    score += topic_points(data_str, &mut failures);

    // Placeholder leaks cap hard: unfilled template text means the output
    // did not do the task, whatever else it got right.
    let leak_count = PLACEHOLDER_RE.find_iter(data_str).count();
    if leak_count > 0 {
        failures.push(format!("placeholder leak ({leak_count} occurrences)"));
        return (
            MAX_SCORE.min(PLACEHOLDER_LEAK_MAX_SCORE).min(score),
            failures.join("; "),
        );
    }

    if !source_text.is_empty() && total_bullets > 0 && faithful < total_bullets {
        score = score.min(MISATTRIBUTION_MAX_SCORE);
    }
    (score.min(MAX_SCORE), failures.join("; "))
}

fn structure_points(data_str: &str, failures: &mut Vec<String>) -> i64 {
    let has_headers = has_text_headers(data_str);
    let has_bullets = data_str.contains('•') || data_str.contains("* ") || data_str.contains("- ");
    if has_headers && has_bullets {
        STRUCT_HEADERS_BULLETS_SCORE
    } else if has_headers {
        STRUCT_HEADERS_ONLY_SCORE
    } else if has_bullets && data_str.chars().count() >= STRUCT_BULLET_LONG_LEN {
        STRUCT_BULLET_LONG_SCORE
    } else if data_str.chars().count() >= STRUCT_BULLET_SHORT_LEN {
        STRUCT_BULLET_SHORT_SCORE
    } else {
        failures.push("no structure".to_string());
        0
    }
}

fn coverage_points(data_str: &str, failures: &mut Vec<String>) -> i64 {
    match count_distinct_users(data_str) {
        n if n >= USERS_COVERAGE_HIGH_COUNT => USERS_COVERAGE_HIGH_SCORE,
        n if n == USERS_COVERAGE_MED_COUNT => USERS_COVERAGE_MED_SCORE,
        n if n == USERS_COVERAGE_LOW_COUNT => USERS_COVERAGE_LOW_SCORE,
        _ => {
            failures.push("no user mentions".to_string());
            0
        }
    }
}

#[expect(
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "narrative-word and bullet counts in one summary -- dozens -- and their ratio. Both exact in their targets by orders of magnitude"
)]
fn specificity_points(
    data_str: &str,
    source_text: &str,
    failures: &mut Vec<String>,
) -> (i64, usize, usize) {
    let narrative_words = NARRATIVE_WORDS_RE.find_iter(data_str).count() as i64;
    let (faithful, total_bullets, attribution_reasons) =
        attribution_faithfulness(data_str, source_text);
    let mut specificity_score: i64 = 0;
    if !source_text.is_empty() && total_bullets > 0 {
        let ratio = faithful as f64 / total_bullets as f64;
        if ratio >= 0.8 {
            specificity_score += TIMESTAMP_SPECIFICITY_SCORE;
        } else if ratio >= 0.5 {
            specificity_score += TIMESTAMP_SPECIFICITY_SCORE / 2;
        } else if faithful == 0 {
            failures.push(format!(
                "no faithful attribution (0/{total_bullets} bullets)"
            ));
        }
        failures.extend(attribution_reasons.into_iter().take(3));
    } else if TIMESTAMP_RE.is_match(data_str) {
        specificity_score += TIMESTAMP_SPECIFICITY_SCORE;
    }
    specificity_score +=
        MAX_NARRATIVE_SPECIFICITY_SCORE.min(narrative_words * NARRATIVE_WORD_SCORE_MULTIPLIER);
    if specificity_score == 0 {
        failures.push("no timestamps or narrative words".to_string());
    }
    (specificity_score, faithful, total_bullets)
}

fn synthesis_points(data_str: &str, failures: &mut Vec<String>) -> i64 {
    let template_fields = TEMPLATE_FIELDS_RE.find_iter(data_str).count();
    let is_template_driven = template_fields >= TEMPLATE_DRIVEN_FIELD_LIMIT;
    let has_boilerplate = BOILERPLATE_RE.is_match(data_str);
    let mut top_level = String::new();
    if let Some(header_match) = NEWLINE_HEADER_RE.find(data_str) {
        if header_match.start() > 0 {
            top_level = data_str[..header_match.start()].trim().to_string();
        }
    } else if START_HEADER_RE.find(data_str).is_none() {
        top_level = data_str.to_string();
    }
    if top_level.is_empty() {
        let lstripped = data_str.trim_start();
        if let Some(head) = LEADING_SUMMARY_HEAD_RE.find(lstripped) {
            let rest = &lstripped[head.end()..];
            let body = SECTION_BREAK_RE
                .find(rest)
                .map_or(rest, |section| &rest[..section.start()]);
            top_level = body.trim().to_string();
        }
    }
    let has_synthesis = !top_level.is_empty() && SYNTHESIS_RE.is_match(&top_level);
    let mut synthesis_score;
    if is_template_driven {
        failures.push("template-driven (repeated field structure)".to_string());
        synthesis_score = 0;
    } else if has_boilerplate {
        synthesis_score = BOILERPLATE_SYNTHESIS_SCORE;
        failures.push("boilerplate filler".to_string());
    } else {
        synthesis_score = DEFAULT_SYNTHESIS_SCORE;
    }
    if has_synthesis {
        synthesis_score += SYNTHESIS_MATCH_BONUS;
    }
    synthesis_score
}

fn topic_points(data_str: &str, failures: &mut Vec<String>) -> i64 {
    let topic_markers = TOPIC_MARKERS_RE.find_iter(data_str).count();
    if topic_markers >= TOPIC_COVERAGE_HIGH_COUNT {
        TOPIC_COVERAGE_HIGH_SCORE
    } else if topic_markers == TOPIC_COVERAGE_MED_COUNT {
        TOPIC_COVERAGE_MED_SCORE
    } else if TRANSITION_WORDS_RE.is_match(data_str) {
        TOPIC_TRANSITION_WORD_SCORE
    } else {
        failures.push("no topic structure".to_string());
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write as _;

    const SOURCE: &str = "[@alice | 08:15]: We shipped the new billing pipeline today after three weeks of work.\n[@bob | 09:02]: The migration finished cleanly and the dashboards look correct.\n[@carol | 10:30]: I am writing the postmortem for last week's outage now.\n";

    const FAITHFUL: &str = "## Engineering\n- shipped the new billing pipeline after three weeks of work (@alice | 08:15)\n- migration finished cleanly and dashboards look correct (@bob | 09:02)\n- writing the postmortem for last week's outage (@carol | 10:30)\n";

    const MISATTRIBUTED: &str = "## Engineering\n- shipped the new billing pipeline after three weeks of work (@carol | 10:30)\n- migration finished cleanly and dashboards look correct (@alice | 08:15)\n- writing the postmortem for last week's outage (@bob | 09:02)\n";

    #[test]
    fn misattributed_summary_is_capped() {
        let (score, _) = validate_summary(MISATTRIBUTED, SOURCE);
        assert!(score <= MISATTRIBUTION_MAX_SCORE, "{score}");
    }

    #[test]
    fn faithful_summary_scores_above_the_cap() {
        // Without this the cap could pass for the wrong reason — if every
        // summary scored below 45 the assertion above would hold trivially.
        let (score, _) = validate_summary(FAITHFUL, SOURCE);
        assert!(score > MISATTRIBUTION_MAX_SCORE, "{score}");
    }

    #[test]
    fn omission_beats_misattribution() {
        let omits = "## Engineering\n- shipped the new billing pipeline (@alice | 08:15)\n";
        let (misattributing, _) = validate_summary(MISATTRIBUTED, SOURCE);
        let (omitting, _) = validate_summary(omits, SOURCE);
        assert!(misattributing < omitting, "{misattributing} vs {omitting}");
    }

    #[test]
    fn misattribution_reason_names_the_person() {
        let (_, msg) = validate_summary(MISATTRIBUTED, SOURCE);
        assert!(msg.contains("faithful attribution"), "{msg}");
        assert!(
            ["@alice", "@bob", "@carol"].iter().any(|h| msg.contains(h)),
            "{msg}"
        );
    }

    #[test]
    fn partial_misattribution_is_also_capped() {
        let mostly_right = "## Engineering\n- shipped the new billing pipeline after three weeks of work (@alice | 08:15)\n- migration finished cleanly and dashboards look correct (@bob | 09:02)\n- writing the postmortem for last week's outage (@alice | 08:15)\n";
        let (score, _) = validate_summary(mostly_right, SOURCE);
        assert!(score <= MISATTRIBUTION_MAX_SCORE, "{score}");
    }

    fn flattening_summary(n_faithful: usize) -> (String, String) {
        let claims = [
            "shipped the billing pipeline after three weeks",
            "migration finished cleanly and dashboards recovered",
            "writing the postmortem for last week outage",
            "upgraded the search index to the new analyzer",
            "cut the nightly batch runtime by half",
        ];
        let mut src = String::new();
        for (i, c) in claims.iter().enumerate() {
            writeln!(src, "[@u{} | 0{}:00]: We {c} today.", i + 1, i + 1).unwrap();
        }
        let mut out = vec!["## Update".to_string()];
        for (i, c) in claims.iter().enumerate() {
            let author = if i < n_faithful {
                i + 1
            } else {
                ((i + 1) % claims.len()) + 1
            };
            out.push(format!("- {c} (@u{author} | 0{author}:00)"));
        }
        (out.join("\n") + "\n", src)
    }

    #[test]
    fn every_partial_ratio_lands_on_the_cap() {
        for n in [4, 3, 2, 0] {
            let (summary, src) = flattening_summary(n);
            let (score, _) = validate_summary(&summary, &src);
            assert_eq!(score, MISATTRIBUTION_MAX_SCORE, "n_faithful={n}: {score}");
        }
    }

    #[test]
    fn complete_faithfulness_escapes_the_cap() {
        let (summary, src) = flattening_summary(5);
        let (score, _) = validate_summary(&summary, &src);
        assert!(score > MISATTRIBUTION_MAX_SCORE, "{score}");
    }

    #[test]
    fn each_bullet_marker_is_recognised_alone() {
        for marker in ["•", "* ", "- "] {
            let body: String = (0..6)
                .map(|_| format!("{marker}a reasonably long claim about the work done today"))
                .collect::<Vec<_>>()
                .join("\n");
            assert!(
                body.chars().count() >= 300,
                "fixture must clear the long-bullet floor"
            );
            let (scored, _) = validate_summary(&body, "");
            let (bare, _) =
                validate_summary("just a paragraph of prose with no markers at all", "");
            assert!(
                scored > bare,
                "{marker:?} was not recognised as a bullet marker"
            );
        }
    }

    #[test]
    fn prose_with_no_structure_is_reported() {
        let (_, msg) = validate_summary("short prose", "");
        assert!(msg.contains("no structure"), "{msg}");
    }

    #[test]
    fn output_with_neither_timestamps_nor_narrative_is_reported() {
        let body: String = (0..6).map(|_| "- item").collect::<Vec<_>>().join("\n");
        let (_, msg) = validate_summary(&body, "");
        assert!(msg.contains("no timestamps or narrative words"), "{msg}");
    }

    #[test]
    fn output_with_a_timestamp_is_not_reported_as_lacking_one() {
        let body = "## Update\n- the deploy finished at 14:30 and the dashboards recovered\n";
        let (_, msg) = validate_summary(body, "");
        assert!(!msg.contains("no timestamps or narrative words"), "{msg}");
    }

    const ORDER_SOURCE: &str = "[@TechCrunch | 08:00]: OpenAI announced GPT-5\n[@TheVerge | 08:15]: Apple Vision Pro 2 mass production\n[@Wired | 17:45]: Meta unveiled AR glasses prototype\n";
    const ORDER_GROUNDED: &str = "## Executive Summary\nA dynamic period across AI and consumer hardware, with launches converging\non inference cost and on-device capability.\n\n## AI\n- OpenAI announced GPT-5 with advanced reasoning (@TechCrunch | 08:00)\n- Apple Vision Pro 2 entered mass production (@TheVerge | 08:15)\n- Meta unveiled AR glasses (@Wired | 17:45)\n";
    const ORDER_LEAKED: &str = "## Executive Summary\nA dynamic period across AI and consumer hardware, with launches converging\non inference cost and on-device capability.\n\n## AI\n- OpenAI announced GPT-5 with advanced reasoning (@TechCrunch | Mon DD HH:MM)\n- Apple Vision Pro 2 entered mass production (@TheVerge | Mon DD HH:MM)\n- Meta unveiled AR glasses (@Wired | Mon DD HH:MM)\n";
    const ORDER_INVENTED: &str = "## Executive Summary\nA dynamic period across AI and consumer hardware, with launches converging\non inference cost and on-device capability.\n\n## AI\n- OpenAI announced GPT-5 with advanced reasoning (@TechCrunch | Mon 03:00)\n- Apple Vision Pro 2 entered mass production (@TheVerge | Tue 04:15)\n- Meta unveiled AR glasses (@Wired | Wed 05:45)\n";

    const GOOD_SUMMARY: &str = "## Executive Summary\nFunding and model releases dominated the week, with inference cost recurring\nacross threads. Participants reported benchmarks and confirmed pricing.\n\n## Funding\n- Series B closed at $40M (@TechCrunch | Mar 15 08:00)\n- Follow-on announced for infrastructure (@benedictevans | Mar 15 09:30)\n\n## Models\n- Lower latency confirmed (@simonw | Mar 16 11:05)\n- Early evaluation numbers shared (@karpathy | Mar 16 14:20)\n";
    const BAD_SUMMARY: &str = "stuff happened. things were said. no idea who or when.";

    #[test]
    fn good_summary_outscores_bad() {
        let (good, _) = validate_summary(GOOD_SUMMARY, "");
        let (bad, _) = validate_summary(BAD_SUMMARY, "");
        assert!(good > bad, "{good} vs {bad}");
    }

    #[test]
    fn conformant_summary_clears_the_gate() {
        assert!(validate_summary(GOOD_SUMMARY, "").0 >= 90);
    }

    #[test]
    fn padding_with_user_tokens_does_not_help() {
        let padded =
            format!("{GOOD_SUMMARY}\n- @user 1 responded, @user 2 asked, @user 3 confirmed\n");
        assert!(validate_summary(&padded, "").0 <= validate_summary(GOOD_SUMMARY, "").0);
    }

    #[test]
    fn placeholder_leak_is_capped_and_named() {
        let (score, failures) = validate_summary(ORDER_LEAKED, ORDER_SOURCE);
        assert!(score <= 40, "{score}");
        assert!(failures.contains("placeholder leak"), "{failures}");
    }

    #[test]
    fn invented_timestamps_lose_attribution_points() {
        let (invented, invented_msg) = validate_summary(ORDER_INVENTED, ORDER_SOURCE);
        let (grounded, _) = validate_summary(ORDER_GROUNDED, ORDER_SOURCE);
        assert!(invented < grounded, "{invented} vs {grounded}");
        assert!(
            invented_msg.contains("faithful attribution"),
            "{invented_msg}"
        );
    }

    #[test]
    fn grounded_attribution_scores_clean() {
        let (score, failures) = validate_summary(ORDER_GROUNDED, ORDER_SOURCE);
        assert!(score >= 90, "{score}: {failures}");
        assert!(failures.is_empty(), "{failures}");
    }

    #[test]
    fn leaked_below_invented_below_grounded() {
        let leaked = validate_summary(ORDER_LEAKED, ORDER_SOURCE).0;
        let invented = validate_summary(ORDER_INVENTED, ORDER_SOURCE).0;
        let grounded = validate_summary(ORDER_GROUNDED, ORDER_SOURCE).0;
        assert!(
            leaked < invented && invented < grounded,
            "{leaked} {invented} {grounded}"
        );
    }
}
