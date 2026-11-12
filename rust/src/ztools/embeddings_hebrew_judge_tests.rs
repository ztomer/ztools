//! LM.EMBEDHE: the prefilter and the pairing on the committed Hebrew fixture,
//! with a table in place of the model; the live model's verdicts are a recorded
//! run, pinned by `embeddings_hebrew_record_tests.rs`.

use std::collections::BTreeMap;
use std::time::Duration;

use super::*;
use crate::test_env::TestEnv;
use crate::ztools::residency::Timing;
use crate::ztools::residency::fake::{FakeOsaurus, Spec};
use serial_test::serial;

use super::record::{Row, class, hebrew_rows};

/// The 42 Hebrew tweets of the committed fixture (the vectors are not read).
pub(super) fn hebrew_fixture() -> Vec<Row> {
    hebrew_rows(include_str!(
        "../../../tests/fixtures/embeddings/potion_multilingual_128m.json"
    ))
    .expect("the committed fixture parses")
}

fn tweet(text: &str) -> Tweet {
    Tweet {
        id: String::new(),
        screen_name: "u".to_string(),
        text: text.to_string(),
        created_at: "now".to_string(),
        favorite_count: 0,
        retweet_count: 0,
        reply_to: None,
    }
}

/// Says yes exactly to the pairs of one group, and records what it was asked.
struct ByGroup {
    group: BTreeMap<String, String>,
    asked: usize,
}

impl ByGroup {
    fn new(rows: &[Row]) -> Self {
        Self {
            group: rows
                .iter()
                .map(|r| (r.text.clone(), r.group.clone()))
                .collect(),
            asked: 0,
        }
    }
}

impl Judge for ByGroup {
    fn same_story(&mut self, a: &str, b: &str) -> Result<bool> {
        self.asked += 1;
        Ok(self.group.get(a) == self.group.get(b))
    }
}

struct Mumbles;

impl Judge for Mumbles {
    fn same_story(&mut self, _: &str, _: &str) -> Result<bool> {
        bail!("answered \"maybe\", not yes or no")
    }
}

#[test]
fn a_verdict_is_one_word_and_nothing_else_is_guessed() {
    assert_eq!(parse_verdict("yes"), Some(true));
    assert_eq!(parse_verdict(" No. "), Some(false));
    assert_eq!(parse_verdict("YES!"), Some(true));
    assert_eq!(parse_verdict("yes, they are"), None);
    assert_eq!(parse_verdict("maybe"), None);
    assert_eq!(parse_verdict("כן"), None);
    assert_eq!(parse_verdict(""), None);
}

#[test]
fn hebrew_is_told_from_latin_and_from_letterless_text() {
    assert!(is_hebrew("בנק ישראל הוריד את הריבית"));
    assert!(is_hebrew("טיפ לפייתון: kill"));
    assert!(!is_hebrew("Rust 1.95 is out today"));
    assert!(!is_hebrew("2:1 90' !!!"));
}

#[test]
fn the_prefilter_passes_every_labelled_pair_and_few_others() {
    let rows = hebrew_fixture();
    assert_eq!(rows.len(), 42);
    let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();
    let picked = candidates(&texts, PREFILTER_FLOOR);
    let count = |cls: &str| {
        picked
            .iter()
            .filter(|(i, j)| class(&rows[*i], &rows[*j]) == cls)
            .count()
    };
    assert_eq!(count("same"), 8, "every labelled pair reaches the model");
    assert_eq!(count("trap"), 8, "the neighbouring stories reach it too");
    assert_eq!(count("unrelated"), 24);
    assert!(picked.len() <= MAX_PAIRS, "one run judges them all");
    // The first pair is the closest one: the two football tweets.
    let (i, j) = picked[0];
    assert_eq!(
        (rows[i].group.as_str(), rows[j].group.as_str()),
        ("he_football", "he_football")
    );
}

/// The prefilter can lose a pair: at 0.25 the Bank of Israel and Knesset pairs
/// (0.265 and 0.225 overlap) are the first to go, so the floor matters.
#[test]
fn a_tight_prefilter_loses_labelled_pairs() {
    let rows = hebrew_fixture();
    let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();
    let same = |floor| {
        candidates(&texts, floor)
            .iter()
            .filter(|(i, j)| class(&rows[*i], &rows[*j]) == "same")
            .count()
    };
    assert_eq!(same(0.25), 7);
    assert_eq!(same(0.30), 5);
    assert_eq!(same(1.0), 0);
}

#[test]
fn overlap_is_a_jaccard_of_trigrams() {
    assert!((overlap("abc", "abc") - 1.0).abs() < 1e-9);
    assert!(overlap("abc", "xyz").abs() < 1e-9);
    assert!(overlap("", "").abs() < 1e-9);
    // "#ab" "abc" "bc#" against "#ab" "abd" "bd#": one of five.
    assert!((overlap("abc", "abd") - 0.2).abs() < 1e-9);
}

#[test]
fn a_judge_that_knows_the_stories_rebuilds_the_eight_pairs() {
    let rows = hebrew_fixture();
    let clusters: Vec<Vec<Tweet>> = std::iter::once(vec![tweet("Rust 1.95 is out")])
        .chain(rows.iter().map(|r| vec![tweet(&r.text)]))
        .collect();
    let mut judge = ByGroup::new(&rows);
    let (out, eligible, pairing) = join(clusters, |t| is_hebrew(&t.text), &mut judge, MAX_PAIRS);
    assert_eq!(eligible, 42);
    assert_eq!(judge.asked, 40);
    assert_eq!(pairing.judged, 40);
    assert_eq!(pairing.same.len(), 8);
    assert_eq!(out.len(), 1 + 42 - 8, "eight pairs joined");
    assert_eq!(
        out[0][0].text, "Rust 1.95 is out",
        "a Latin cluster keeps its place"
    );
    assert_eq!(out.iter().filter(|c| c.len() == 2).count(), 8);
    assert_eq!(
        out[1][0].text, rows[0].text,
        "a joined cluster sits at its first member"
    );
    assert_eq!(out[1][1].text, rows[1].text);
}

#[test]
fn the_cap_bounds_the_calls_and_counts_the_rest() {
    let rows = hebrew_fixture();
    let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();
    let mut judge = ByGroup::new(&rows);
    let pairing = pair(&texts, &mut judge, 5);
    assert_eq!(judge.asked, 5);
    assert_eq!(pairing.judged, 5);
    assert_eq!(pairing.unjudged, 35);
}

#[test]
fn a_pair_without_a_verdict_is_kept_apart_and_counted() {
    let rows = hebrew_fixture();
    let clusters: Vec<Vec<Tweet>> = rows.iter().map(|r| vec![tweet(&r.text)]).collect();
    let (out, _, pairing) = join(clusters, |t| is_hebrew(&t.text), &mut Mumbles, MAX_PAIRS);
    assert_eq!(out.len(), 42, "nothing merged on no verdict");
    assert_eq!(pairing.failed, 40);
    assert_eq!(pairing.judged, 0);
    assert!(pairing.first_error.unwrap().contains("maybe"));
}

#[test]
fn components_join_chains_under_their_lowest_member() {
    assert_eq!(components(5, &[(3, 4), (1, 3)]), vec![0, 1, 2, 1, 1]);
    assert_eq!(components(3, &[]), vec![0, 1, 2]);
}

#[test]
fn a_model_is_calibrated_by_its_bare_name() {
    assert!(judge_calibrated("org/raptor", &["raptor"]));
    assert!(judge_calibrated("RAPTOR", &["raptor"]));
    assert!(!judge_calibrated("raptor-2", &["raptor"]));
    assert!(!judge_calibrated("raptor", &[]));
}

fn fast() -> Timing {
    Timing {
        lock_wait: Duration::from_secs(2),
        load_request: Duration::from_secs(5),
        settle: Duration::from_millis(300),
        release_grace: Duration::from_millis(300),
        release_cap: Duration::from_secs(2),
        poll_min: Duration::from_millis(5),
        poll_max: Duration::from_millis(20),
    }
}

fn the_pairs() -> Vec<Vec<Tweet>> {
    [
        "מכבי תל אביב גברה 2:1 על הפועל באר שבע עם שער ניצחון בדקה ה-90",
        "Rust 1.95 is out today",
        "איזה סיום! שער בדקה ה-90 העניק למכבי תל אביב ניצחון 2:1 על הפועל באר שבע",
        "מכבי תל אביב הפסידה 85:80 לריאל מדריד ביורוליג אחרי הארכה",
    ]
    .iter()
    .map(|t| vec![tweet(t)])
    .collect()
}

/// The fake model says yes only when both tweets name the 90th-minute goal.
fn football_judge() -> Spec {
    Spec {
        answer_for: Some(std::sync::Arc::new(|body: &str| {
            let yes = body.matches("90").count() >= 2;
            let word = if yes { "yes" } else { "no" };
            format!(r#"{{"choices":[{{"message":{{"content":"{word}"}}}}]}}"#)
        })),
        ..Spec::chat("{}")
    }
}

#[test]
#[serial]
fn a_calibrated_summarize_model_pairs_under_the_lock_and_is_released() {
    let env = TestEnv::new();
    let fake = FakeOsaurus::serve(football_judge());
    let (out, hebrew) = pair_hebrew(
        the_pairs(),
        |_| true,
        &fake.url,
        "raptor",
        &["raptor"],
        fast(),
    );
    let Hebrew::Paired {
        model,
        tweets,
        pairing,
    } = &hebrew
    else {
        panic!("{hebrew:?}");
    };
    assert_eq!((model.as_str(), *tweets), ("raptor", 3));
    assert_eq!(pairing.same.len(), 1, "{pairing:?}");
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].len(), 2, "the two football tweets");
    assert_eq!(fake.loads(), vec!["raptor".to_string()]);
    assert_eq!(fake.last_health(), Some(vec![]), "released");
    assert_eq!(fake.unlocked_requests(), Vec::<String>::new());
    let clause = hebrew.clause().unwrap();
    assert!(
        clause.contains("3 Hebrew tweets paired by raptor"),
        "{clause}"
    );
    drop(env);
}

#[test]
#[serial]
fn an_uncalibrated_summarize_model_keeps_hebrew_apart_and_loads_nothing() {
    let env = TestEnv::new();
    let fake = FakeOsaurus::serve(football_judge());
    let (out, hebrew) = pair_hebrew(
        the_pairs(),
        |_| true,
        &fake.url,
        "other",
        &["raptor"],
        fast(),
    );
    assert_eq!(out.len(), 4);
    assert!(fake.calls().is_empty(), "{:?}", fake.calls());
    let Hebrew::Apart {
        tweets: 3,
        fault: false,
        reason,
    } = &hebrew
    else {
        panic!("{hebrew:?}");
    };
    assert!(
        reason.contains("'other' has no Hebrew same-story calibration"),
        "{reason}"
    );
    drop(env);
}

#[test]
#[serial]
fn a_model_that_will_not_load_keeps_every_tweet_and_says_why() {
    let env = TestEnv::new();
    let (out, hebrew) = pair_hebrew(
        the_pairs(),
        |_| true,
        "http://127.0.0.1:59998",
        "raptor",
        &["raptor"],
        fast(),
    );
    assert_eq!(out.len(), 4, "no tweet lost");
    assert!(
        matches!(
            hebrew,
            Hebrew::Apart {
                tweets: 3,
                fault: true,
                ..
            }
        ),
        "{hebrew:?}"
    );
    drop(env);
}

#[test]
#[serial]
fn a_failed_release_keeps_the_pairing() {
    let env = TestEnv::new();
    let fake = FakeOsaurus::serve(Spec {
        keep_loaded: true,
        ..football_judge()
    });
    let (out, hebrew) = pair_hebrew(
        the_pairs(),
        |_| true,
        &fake.url,
        "raptor",
        &["raptor"],
        fast(),
    );
    assert_eq!(
        out.len(),
        3,
        "the joined clusters survive the release error"
    );
    assert!(matches!(hebrew, Hebrew::Paired { .. }), "{hebrew:?}");
    drop(env);
}

#[test]
fn no_hebrew_or_no_candidate_loads_nothing() {
    let latin = vec![vec![tweet("Rust 1.95 is out")]];
    let (out, hebrew) = pair_hebrew(
        latin,
        |_| true,
        "http://127.0.0.1:59997",
        "m",
        &["m"],
        fast(),
    );
    assert_eq!((out.len(), hebrew), (1, Hebrew::None));
    let far = vec![
        vec![tweet("החתול שלנו אכל עוף")],
        vec![tweet("הביטקוין ירד היום")],
    ];
    let (_, hebrew) = pair_hebrew(far, |_| true, "http://127.0.0.1:59997", "m", &["m"], fast());
    assert!(
        matches!(&hebrew, Hebrew::Paired { pairing, .. } if pairing.judged == 0),
        "{hebrew:?}"
    );
}
