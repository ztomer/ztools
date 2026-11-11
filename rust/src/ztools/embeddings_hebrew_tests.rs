//! LM.EMBEDHE: the Hebrew half of an embedding calibration, and the verdict on
//! `potion-multilingual-128M`, the in-process multilingual candidate.
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
