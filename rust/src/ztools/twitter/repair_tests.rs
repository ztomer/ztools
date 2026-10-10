//! T3. Synthetic tweets shaped like the 2026-10-10 failures: citations that
//! drop the year, an answer that loops after seven real bullets, one that
//! invents its sources.

use std::fmt::Write as _;

use super::*;

fn given() -> Vec<(String, String)> {
    [
        ("alice", "Sat Oct 10 09:46:22 +0000 2026"),
        ("bob", "Sat Oct 10 10:32:00 +0000 2026"),
        ("carol", "Sat Oct 10 13:30:45 +0000 2026"),
        ("dave", "Sat Oct 10 14:02:53 +0000 2026"),
        ("erin", "Fri Oct 09 22:51:00 +0000 2026"),
        ("frank", "Sat Oct 10 00:49:52 +0000 2026"),
        ("grace", "Sat Oct 10 07:54:52 +0000 2026"),
        ("grace", "Sat Oct 10 07:54:59 +0000 2026"),
    ]
    .iter()
    .map(|(h, t)| ((*h).to_string(), (*t).to_string()))
    .collect()
}

/// A citation names a given tweet when its timestamp is that tweet's, or a
/// prefix of exactly one of the handle's tweets down to the minute -- the
/// model drops the year, or the seconds, and the tweet is still unambiguous.
#[test]
fn a_shortened_timestamp_still_names_its_tweet() {
    let g = given();
    let full = "Sat Oct 10 09:46:22 +0000 2026";
    assert_eq!(resolve("alice", full, &g), Some(full));
    assert_eq!(
        resolve("alice", "Sat Oct 10 09:46:22 +0000", &g),
        Some(full)
    );
    assert_eq!(
        resolve("erin", "Fri Oct 09 22:51", &g),
        Some("Fri Oct 09 22:51:00 +0000 2026")
    );
    // Two of grace's tweets share the minute: ambiguous, so unresolved.
    assert_eq!(resolve("grace", "Sat Oct 10 07:54", &g), None);
    // Shorter than the minute says too little.
    assert_eq!(resolve("alice", "Sat Oct 10", &g), None);
    assert_eq!(resolve("mallory", full, &g), None);
}

fn bullet(handle: &str, stamp: &str, fact: &str) -> String {
    format!("- {fact} (@{handle} | {stamp})\n")
}

/// Seven real bullets, then the same two recited into every later section.
fn looping_answer() -> String {
    let mut s = String::from("## Executive Summary\n\nA busy Saturday.\n\n## Topic one\n\n");
    for (h, t, f) in [
        (
            "alice",
            "Sat Oct 10 09:46:22 +0000",
            "Alice reports the first fact",
        ),
        (
            "bob",
            "Sat Oct 10 10:32:00 +0000",
            "Bob reports a second fact",
        ),
        (
            "carol",
            "Sat Oct 10 13:30:45 +0000",
            "Carol reports a third fact",
        ),
        (
            "dave",
            "Sat Oct 10 14:02:53 +0000",
            "Dave reports a fourth fact",
        ),
    ] {
        s.push_str(&bullet(h, t, f));
    }
    s.push_str("\n## Topic two\n\n");
    for (h, t, f) in [
        ("erin", "Fri Oct 09 22:51", "Erin reports a fifth fact"),
        (
            "frank",
            "Sat Oct 10 00:49:52 +0000",
            "Frank reports a sixth fact",
        ),
        (
            "grace",
            "Sat Oct 10 07:54:52 +0000 2026",
            "Grace reports a seventh fact",
        ),
    ] {
        s.push_str(&bullet(h, t, f));
    }
    for n in 0..6 {
        let _ = write!(s, "\n## Looped topic {n}\n\n");
        s.push_str(&bullet(
            "alice",
            "Sat Oct 10 09:46:22 +0000",
            "Alice reports the first fact",
        ));
        s.push_str(&bullet(
            "bob",
            "Sat Oct 10 10:32:00 +0000",
            "Bob reports a second fact",
        ));
    }
    s
}

/// The loop's repeats carry nothing; the seven bullets before it each cite a
/// given tweet. Repaired: the seven stay, every citation in its full form, the
/// looped sections go, and the repair says what it dropped.
#[test]
fn a_looping_answer_is_repaired_to_its_verified_bullets() {
    let repaired = repair(&looping_answer(), &given()).expect("seven verified bullets survive");
    let kept = crate::ztools::twitter::quality::bullets(&repaired.body);
    assert_eq!(kept.len(), 7, "{}", repaired.body);
    assert!(!repaired.body.contains("Looped topic"), "{}", repaired.body);
    assert!(
        repaired
            .body
            .contains("(@erin | Fri Oct 09 22:51:00 +0000 2026)")
    );
    assert!(
        !repaired.body.contains("+0000)"),
        "every citation is canonical: {}",
        repaired.body
    );
    assert!(
        repaired
            .notes
            .iter()
            .any(|n| n.contains("12") && n.contains("repeat")),
        "{:?}",
        repaired.notes
    );
}

/// An answer whose citations mostly name nothing it was given is inventing,
/// and is rejected rather than trimmed to its few real lines.
#[test]
fn an_answer_that_mostly_invents_its_sources_is_rejected() {
    let mut s = String::from("## Executive Summary\n\nx\n\n## Topic\n\n");
    s.push_str(&bullet("alice", "Sat Oct 10 09:46:22 +0000 2026", "real"));
    for n in 0..4 {
        s.push_str(&bullet(
            "nobody",
            &format!("Sat Oct 10 0{n}:00:00"),
            "invented",
        ));
    }
    let why = repair(&s, &given()).unwrap_err();
    assert!(why.contains("4 of 5 citations"), "{why}");
}

/// A bullet citing a tweet it was not given goes; its neighbours stay.
#[test]
fn one_bad_citation_drops_its_bullet_not_the_answer() {
    let mut s = looping_answer();
    s.push_str(&bullet("nobody", "Sat Oct 10 01:00:00", "invented"));
    let repaired = repair(&s, &given()).unwrap();
    assert!(!repaired.body.contains("invented"), "{}", repaired.body);
    assert!(
        repaired
            .notes
            .iter()
            .any(|n| n.contains("1 bullet") && n.contains("no tweet")),
        "{:?}",
        repaired.notes
    );
}

/// The gate judges the REPAIRED answer, and asks for enough of it: three
/// verified bullets out of a loop is not a summary of the timeline.
#[test]
fn too_few_verified_bullets_is_still_a_rejection() {
    let mut s = String::from("## Executive Summary\n\nx\n\n## Topic\n\n");
    for (h, t) in [
        ("alice", "Sat Oct 10 09:46:22 +0000 2026"),
        ("bob", "Sat Oct 10 10:32:00 +0000 2026"),
        ("carol", "Sat Oct 10 13:30:45 +0000 2026"),
    ] {
        s.push_str(&bullet(h, t, "fact"));
    }
    for _ in 0..10 {
        s.push_str(&bullet("alice", "Sat Oct 10 09:46:22 +0000 2026", "fact"));
    }
    let why = crate::ztools::twitter::handle_model_output(&s, &given(), false).unwrap_err();
    assert!(why.contains("3 verified bullet"), "{why}");
}

/// A cut answer says so in what is saved.
#[test]
fn a_cut_answer_is_saved_saying_it_was_cut() {
    let answer = crate::ztools::twitter::handle_model_output(&looping_answer(), &given(), true)
        .expect("repairable");
    assert!(
        answer.repairs.iter().any(|n| n.contains("length limit")),
        "{:?}",
        answer.repairs
    );
}
