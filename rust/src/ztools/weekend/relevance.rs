//! Did the engine answer THIS query, or just answer?
//!
//! A results page that parses into results used to be an answer, whatever it
//! was about. On 2026-10-10 Bing served every one of the planner's sixteen
//! queries a page about something else -- "kids activities Vaughan October
//! 2026" got `YouTube Kids` and `PBS KIDS`, "Vaughan family events October 2026"
//! got a condo on Isla Mujeres, every "GTA ..." query got Grand Theft Auto --
//! while echoing the full query back in its own search box, so the request
//! was right and the ANSWER was the wall. Each page parsed, so each was
//! recorded "answered"; Bing stood first in the learned order (`DuckDuckGo`
//! having been demoted for real walls), Brave was never asked, and the corpus
//! was 88 lines of nothing: "0/88 mention a date this weekend".
//!
//! An answer is ON-TOPIC when enough of its results carry enough of the
//! query's distinctive words. An off-topic answer is a soft bot wall: the
//! caller treats it exactly like a hard one -- recorded as walled, so the
//! learned order demotes that engine -- and asks the next engine instead.

use super::SearchResult;

/// Words that carry no topic: joiners, and month names (a query names the
/// month, and so does nearly every dated page on the web, relevant or not).
const NON_TOPIC: &[&str] = &[
    "and",
    "the",
    "for",
    "with",
    "this",
    "near",
    "from",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

/// The query's distinctive words: lower-cased, three letters or more, not a
/// bare number (the year is on every page), not a [`NON_TOPIC`] word.
#[must_use]
pub fn query_terms(query: &str) -> Vec<String> {
    let mut terms: Vec<String> = Vec::new();
    for word in query
        .to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
    {
        let keep = word.len() >= 3
            && !word.chars().all(|c| c.is_ascii_digit())
            && !NON_TOPIC.contains(&word);
        if keep && !terms.iter().any(|t| t == word) {
            terms.push(word.to_string());
        }
    }
    terms
}

/// How many of `terms` one result must carry to be about the query: half of
/// them, and never fewer than two (a lone shared word is how "kids ..."
/// matched `YouTube Kids`), but never more than the query has.
fn needed(terms: usize) -> usize {
    terms.min(terms.div_ceil(2).max(2))
}

/// Is this answer about the query?
///
/// On-topic when at least a THIRD of the results each carry [`needed`] of the
/// query's terms in their title or body. Calibrated on the live results of
/// 2026-10-10 (the probe that reproduced "0/88"): Bing's sixteen soft-walled
/// answers scored at most 2 in 10 under this rule, the 24 real answers Brave
/// gave the same and the window-dated queries scored 5 in 10 or more, so the
/// line sits between 20% and 50%. A query with no distinctive word cannot be
/// judged and is always on-topic.
#[must_use]
pub fn answer_is_on_topic(query: &str, results: &[SearchResult]) -> bool {
    let terms = query_terms(query);
    if terms.is_empty() || results.is_empty() {
        return true;
    }
    let need = needed(terms.len());
    let on_topic = results
        .iter()
        .filter(|r| {
            let text = format!("{} {}", r.title, r.body).to_lowercase();
            terms.iter().filter(|t| text.contains(t.as_str())).count() >= need
        })
        .count();
    on_topic * 3 >= results.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(title: &str, body: &str) -> SearchResult {
        SearchResult {
            title: title.into(),
            href: String::new(),
            body: body.into(),
        }
    }

    /// Months, years, joiners and short words are not the topic.
    #[test]
    fn the_terms_are_the_distinctive_words_once_each() {
        assert_eq!(
            query_terms("Vaughan events this weekend October 9-12 2026"),
            ["vaughan", "events", "weekend"]
        );
        assert_eq!(
            query_terms("kids KIDS activities near Toronto"),
            ["kids", "activities", "toronto"]
        );
        assert_empty!(query_terms("October 2026"));
    }

    /// Two shared words at least, half of a long query's, never more than it has.
    #[test]
    fn a_result_needs_half_the_terms_and_at_least_two() {
        assert_eq!(needed(1), 1);
        assert_eq!(needed(2), 2);
        assert_eq!(needed(3), 2);
        assert_eq!(needed(5), 3);
        assert_eq!(needed(7), 4);
    }

    /// The boundary is a third: three on-topic results in nine pass, two fail.
    /// Bing's worst soft-walled answer scored 2 in 10.
    #[test]
    fn a_third_of_the_results_must_be_on_topic() {
        let on = || hit("Kids activities in Vaughan", "");
        let off = || hit("YouTube Kids", "videos for kids");
        let q = "kids activities Vaughan October 2026";
        let mut answer = vec![on(), on(), on()];
        answer.extend((0..6).map(|_| off()));
        assert!(answer_is_on_topic(q, &answer));
        answer.remove(0);
        answer.push(off());
        assert!(!answer_is_on_topic(q, &answer));
    }

    /// Nothing to judge is never a wall: no terms, or no results.
    #[test]
    fn an_unjudgeable_answer_is_on_topic() {
        assert!(answer_is_on_topic("October 2026", &[hit("anything", "")]));
        assert!(answer_is_on_topic("kids activities Vaughan", &[]));
    }
}
