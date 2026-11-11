//! LM.EMBEDHE: the Hebrew half of an embedding calibration, and the verdicts on
//! the two in-process multilingual candidates, `potion-multilingual-128M` and its
//! escalation `Qwen3-Embedding-0.6B`.
//!
//! The rule (design section 5): a model's Hebrew tweets enter clustering only when
//! some threshold puts every labelled same-story Hebrew pair at or above it and
//! every other Hebrew pair below it, with the unrelated Hebrew median under 0.5.
//! Until then the model's calibration stays `latin_only`, or it gets no entry.

use super::super::*;

#[derive(serde::Deserialize)]
struct Recorded {
    model: String,
    tweets: Vec<Row>,
}

#[derive(serde::Deserialize)]
struct Row {
    group: String,
    text: String,
    embedding: Vec<f32>,
}

fn multilingual() -> Recorded {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/embeddings/potion_multilingual_128m.json"
    ))
    .expect("the committed fixture parses")
}

fn qwen3() -> Recorded {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/embeddings/qwen3_embedding_0_6b.json"
    ))
    .expect("the committed fixture parses")
}

/// The story a group is about: `he_football_basket` is a related-not-same tweet
/// of `he_football`'s topic.
fn story(group: &str) -> String {
    group.split('_').take(2).collect::<Vec<_>>().join("_")
}

/// Mostly Hebrew letters.
fn is_hebrew(text: &str) -> bool {
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

/// The Hebrew pairs' cosines: (labelled same-story, every other pair).
fn hebrew_pairs(fx: &Recorded) -> (Vec<f32>, Vec<f32>) {
    let he: Vec<&Row> = fx.tweets.iter().filter(|r| is_hebrew(&r.text)).collect();
    let (mut same, mut other) = (Vec::new(), Vec::new());
    for (i, a) in he.iter().enumerate() {
        for b in &he[i + 1..] {
            let c = cosine_similarity(&a.embedding, &b.embedding);
            if a.group == b.group {
                same.push(c);
            } else {
                other.push(c);
            }
        }
    }
    (same, other)
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// Whether `threshold` passes the Hebrew half on `fx`.
fn hebrew_half_passes(fx: &Recorded, threshold: f32) -> bool {
    let (same, other) = hebrew_pairs(fx);
    !same.is_empty()
        && same.iter().all(|c| *c >= threshold)
        && other.iter().all(|c| *c < threshold)
        && median(other) < 0.5
}

#[test]
fn the_fixture_holds_forty_hebrew_tweets_with_labelled_pairs() {
    let fx = multilingual();
    assert_eq!(fx.model, "minishlab/potion-multilingual-128M");
    let he = fx.tweets.iter().filter(|r| is_hebrew(&r.text)).count();
    assert!(he >= 40, "{he} Hebrew tweets");
    let (same, _) = hebrew_pairs(&fx);
    assert_eq!(same.len(), 8, "eight labelled same-story pairs");
}

/// Measured 2026-11-11: the unrelated Hebrew median is 0.171 (potion-base-4M: 0.947),
/// so the model does read Hebrew, and 7 of the 8 labelled pairs sit above every
/// unrelated pair. The eighth, the Knesset-dissolution pair told with different verb
/// forms (`פיזורה` / `התפזרה`), is at 0.449, under the related-not-same train/bus
/// pair (0.533) and an unrelated one (0.535): a static token table does not see
/// Hebrew morphology. No threshold passes, so the model gets no calibration entry.
#[test]
fn potion_multilingual_128m_has_no_threshold_that_passes_the_hebrew_half() {
    let fx = multilingual();
    let (same, other) = hebrew_pairs(&fx);
    assert!(median(other.clone()) < 0.5, "the model reads Hebrew");
    let highest_other = other.iter().copied().fold(f32::MIN, f32::max);
    assert_eq!(same.iter().filter(|c| **c > highest_other).count(), 7);
    for step in 0..=100_u8 {
        let threshold = f32::from(step) / 100.0;
        assert!(
            !hebrew_half_passes(&fx, threshold),
            "threshold {threshold} passes the Hebrew half: calibrate the model instead"
        );
    }
    assert_eq!(calibration_for(&fx.model), None);
}

/// The instrument can say yes: with the one unseparated pair taken out, 0.60
/// passes, so the verdict above is the data's and not a check that always fails.
#[test]
fn the_hebrew_half_passes_when_every_pair_is_separable() {
    let mut fx = multilingual();
    fx.tweets.retain(|r| r.group != "he_election");
    assert!(hebrew_half_passes(&fx, 0.60));
    assert!(!hebrew_half_passes(&fx, 0.0), "0.0 merges everything");
    assert!(!hebrew_half_passes(&fx, 1.0), "1.0 merges nothing");
}

#[test]
fn both_fixtures_hold_the_same_texts_and_groups() {
    let (a, b) = (multilingual(), qwen3());
    assert_eq!(b.model, "Qwen/Qwen3-Embedding-0.6B");
    let key = |f: &Recorded| -> Vec<(String, String)> {
        f.tweets
            .iter()
            .map(|r| (r.group.clone(), r.text.clone()))
            .collect()
    };
    assert_eq!(
        key(&a),
        key(&b),
        "the escalation is measured on the SAME fixture"
    );
    assert!(b.tweets.iter().all(|r| r.embedding.len() == 1024));
}

/// Measured 2026-11-11 (candle on Metal, no instruction): Qwen3-Embedding-0.6B reads
/// Hebrew well -- unrelated median 0.277, and every labelled pair (lowest 0.579, the
/// Bank of Israel rate cut) sits above every pair from different stories (highest
/// 0.530) -- but it merges a story with its neighbour: Maccabi Tel Aviv's football
/// win and its basketball loss sit at 0.641, Nvidia's and Apple's earnings at 0.637.
/// A threshold at or under 0.579 merges those, so no threshold passes the half and
/// the model gets no calibration entry. With the instruction prompt it is worse
/// (unrelated median 0.534). On the 266 real Latin tweets it also merges different
/// stories at 0.742 ("Stop telling Claude" with an unrelated Claude Code tip), above
/// two of the four known same-story pairs (0.704, 0.681), where potion-base-4M
/// recovers all four.
#[test]
fn qwen3_embedding_0_6b_has_no_threshold_that_passes_the_hebrew_half() {
    let fx = qwen3();
    let (same, other) = hebrew_pairs(&fx);
    assert!(median(other) < 0.5, "the model reads Hebrew");
    let he: Vec<&Row> = fx.tweets.iter().filter(|r| is_hebrew(&r.text)).collect();
    let (mut related, mut unrelated) = (f32::MIN, f32::MIN);
    for (i, a) in he.iter().enumerate() {
        for b in &he[i + 1..] {
            if a.group == b.group {
                continue;
            }
            let c = cosine_similarity(&a.embedding, &b.embedding);
            let slot = if story(&a.group) == story(&b.group) {
                &mut related
            } else {
                &mut unrelated
            };
            *slot = slot.max(c);
        }
    }
    let lowest_same = same.iter().copied().fold(f32::MAX, f32::min);
    assert!(
        lowest_same > unrelated,
        "same-story pairs clear every other story"
    );
    assert!(
        related > lowest_same,
        "a neighbouring story is closer than a same-story pair"
    );
    for step in 0..=100_u8 {
        let threshold = f32::from(step) / 100.0;
        assert!(
            !hebrew_half_passes(&fx, threshold),
            "threshold {threshold} passes the Hebrew half: calibrate the model instead"
        );
    }
    assert_eq!(calibration_for(&fx.model), None);
}

/// The instrument can say yes on this fixture too: without the related-not-same
/// tweets, 0.55 passes; 0.0 and 1.0 still fail.
#[test]
fn qwen3_passes_the_hebrew_half_without_the_neighbouring_stories() {
    let mut fx = qwen3();
    fx.tweets.retain(|r| r.group == story(&r.group));
    assert!(hebrew_half_passes(&fx, 0.55));
    assert!(!hebrew_half_passes(&fx, 0.0), "0.0 merges everything");
    assert!(!hebrew_half_passes(&fx, 1.0), "1.0 merges nothing");
}
