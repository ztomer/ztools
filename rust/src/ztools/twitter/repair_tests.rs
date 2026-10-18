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
    // Eighteen tweets: the floor is five, where the eight of `given` make it three.
    let mut sources = given();
    sources.extend((0..10).map(|i| {
        (
            format!("extra{i}"),
            format!("Sat Oct 10 11:{i:02}:00 +0000 2026"),
        )
    }));
    let why = crate::ztools::twitter::handle_model_output(&s, &sources, false).unwrap_err();
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

/// A citation in brackets is a citation. On 2026-10-10 raptor led every bullet
/// with `**[@handle | stamp]** --` and lost the whole answer to "69 of 69
/// bullets lack the attribution"; the fallback mixed `[@h | ts)`. Each names a
/// given tweet, so the repair writes it in the one form the gate reads.
#[test]
fn a_bracketed_citation_is_rewritten_not_rejected() {
    let g = given();
    let answer = "## Topic\n\n\
        - **[@alice | Sat Oct 10 09:46:22 +0000 2026]** -- a thing happened\n\
        - [@bob | Sat Oct 10 10:32) said another\n\
        - Carol noted it (@carol | Sat Oct 10 13:30:45 +0000 2026)\n";
    let fixed = repair(answer, &g).expect("every citation names a given tweet");
    assert!(
        fixed
            .body
            .contains("- (@alice | Sat Oct 10 09:46:22 +0000 2026) -- a thing happened"),
        "{}",
        fixed.body
    );
    assert!(
        fixed
            .body
            .contains("- (@bob | Sat Oct 10 10:32:00 +0000 2026) said another"),
        "{}",
        fixed.body
    );
    assert_empty!(fixed.notes);
    let bullets = super::super::quality::bullets(&fixed.body);
    assert!(
        bullets
            .iter()
            .all(|b| super::super::quality::is_attributed(b)),
        "{bullets:?}"
    );
}

/// A bullet citing more than three tweets is several unrelated claims run
/// together (the fallback's 19-citation single bullet); no accepted summary of
/// 300+ cites more than three in one bullet. It goes, and the note says so.
#[test]
fn a_run_on_bullet_is_dropped() {
    let g = given();
    let answer = "## Topic\n\n\
        - one (@alice | Sat Oct 10 09:46:22 +0000 2026)\n\
        - two (@bob | Sat Oct 10 10:32:00 +0000 2026), (@carol | Sat Oct 10 13:30:45 +0000 2026), \
          (@dave | Sat Oct 10 14:02:53 +0000 2026)\n\
        - run on (@erin | Fri Oct 09 22:51); then (@frank | Sat Oct 10 00:49); \
          then (@grace | Sat Oct 10 07:54:52 +0000 2026); then (@grace | Sat Oct 10 07:54:59 +0000 2026)\n";
    let fixed = repair(answer, &g).expect("every citation names a given tweet");
    assert!(!fixed.body.contains("run on"), "{}", fixed.body);
    assert!(fixed.body.contains("- two"), "three citations is a bullet");
    assert_eq!(
        fixed.notes,
        vec!["dropped 1 bullet(s) citing more than 3 tweets (unrelated claims run together)"]
    );
}

/// The floor is the timeline's, not the repair's: one clean bullet for eight
/// tweets is not a summary of them, repaired or not.
#[test]
fn one_clean_bullet_for_many_tweets_is_rejected() {
    let g = given();
    let answer = "## Topic\n\n- one (@alice | Sat Oct 10 09:46:22 +0000 2026)\n";
    let why = handle_model_output(answer, &g, false).expect_err("one bullet for eight tweets");
    assert!(why.contains("too few"), "{why}");
}

/// Every form a model has written a citation of ONE given tweet in, from the
/// rejected answers in the store (2026-10-10..16). The class: a faithful
/// citation in a form the repair's grammar did not parse, so the answer was
/// thrown away whole and the chain ran out of models -- `twitter-evening`
/// 2026-10-16: raptor "23 of 25 bullets lack the `(@handle | timestamp)`
/// attribution" (it wrote `(TomerPersico | Fri Oct 16 14:21:56 +0000 2026)`),
/// then foundation "42 of 42 citations name no tweet the model was given" (it
/// wrote `(@Uvalajnr | 04:11:19)`). Each form must come back canonical.
fn drifted_forms(handle: &str, full: &str) -> Vec<String> {
    let year_dropped = full.rsplit_once(' ').map_or(full, |(head, _)| head);
    let time = full.split_whitespace().nth(3).expect("a time field");
    let minute = &time[..5];
    vec![
        format!("(@{handle} | {full})"),
        format!("[@{handle} | {full}]"),
        format!("**[@{handle} | {full}]**"),
        format!("[@{handle} | {full})"),
        format!("(@{handle} | {year_dropped})"),
        format!("({handle} | {full})"),
        format!("({handle} | {year_dropped})"),
        format!("(@{handle} | {time})"),
        format!("(@{handle} | {minute})"),
        format!("({handle} | {time})"),
    ]
}

#[test]
fn every_drifted_form_of_a_given_citation_comes_back_canonical() {
    let g = given();
    // grace's two tweets share a minute; her full stamps are still unique.
    for (handle, full) in g.iter().filter(|(h, _)| h != "grace") {
        for form in drifted_forms(handle, full) {
            let answer = format!("## Topic\n\n- a fact {form}\n");
            let fixed = repair(&answer, &g).unwrap_or_else(|why| panic!("{form}: {why}"));
            assert!(
                fixed
                    .body
                    .contains(&format!("- a fact (@{handle} | {full})")),
                "{form} -> {}",
                fixed.body
            );
            assert_empty!(fixed.notes);
        }
    }
}

/// Foundation joins the sources of one bullet inside one bracket:
/// `(@shipwreckedcrew | 18:08:50; @fedesimio | 18:31:56)` (2026-10-13). Each
/// names a given tweet; they come back as one canonical citation apiece.
#[test]
fn several_sources_in_one_bracket_become_one_citation_each() {
    let g = given();
    let answer =
        "## Topic\n\n- two facts (@alice | 09:46:22; bob | Sat Oct 10 10:32:00 +0000 2026)\n";
    let fixed = repair(answer, &g).expect("both name a given tweet");
    assert!(
        fixed.body.contains(
            "- two facts (@alice | Sat Oct 10 09:46:22 +0000 2026) \
             (@bob | Sat Oct 10 10:32:00 +0000 2026)"
        ),
        "{}",
        fixed.body
    );
}

/// The looser forms resolve only what is unambiguous. A time of day two of a
/// handle's tweets share names neither; a bracket with a bar whose handle the
/// model was never given is prose, left as written and never counted.
#[test]
fn a_loose_form_resolves_only_what_is_unambiguous() {
    let g = given();
    assert_eq!(resolve("grace", "07:54", &g), None);
    assert_eq!(
        resolve("grace", "07:54:59", &g),
        Some("Sat Oct 10 07:54:59 +0000 2026")
    );
    assert_eq!(resolve("alice", "9:46", &g), None);
    assert_eq!(resolve("mallory", "09:46:22", &g), None);
    let answer = "## Topic\n\n- a fact (@alice | 09:46:22), ratio (wins | losses) unchanged\n";
    let fixed = repair(answer, &g).expect("the one citation names a given tweet");
    assert!(
        fixed.body.contains("ratio (wins | losses) unchanged"),
        "{}",
        fixed.body
    );
}

/// The 2026-10-16 evening answers, end to end: every bullet's citation drifted
/// (raptor dropped the `@`, foundation kept only the time of day), and every
/// one names a given tweet. Both are summaries, not rejections.
#[test]
fn the_2026_10_16_evening_answers_are_accepted() {
    let g = given();
    let mut raptor = String::from("## Executive Summary\n\nA busy Saturday.\n\n## Topic\n\n");
    let mut foundation = raptor.clone();
    for (n, (handle, full)) in g.iter().filter(|(h, _)| h != "grace").enumerate() {
        let _ = writeln!(
            raptor,
            "- Fact number {n} from the timeline ({handle} | {full})"
        );
        let time = full.split_whitespace().nth(3).expect("a time field");
        let _ = writeln!(
            foundation,
            "- Fact number {n} from the timeline (@{handle} | {time})"
        );
    }
    let mut refused = Vec::new();
    for (name, answer) in [("raptor", raptor), ("foundation", foundation)] {
        match handle_model_output(&answer, &g, false) {
            Ok(kept) => {
                assert_eq!(
                    super::super::quality::bullets(&kept.body).len(),
                    6,
                    "{name}"
                );
                assert_empty!(kept.repairs);
            }
            Err(why) => refused.push(format!("{name}: {why}")),
        }
    }
    assert_empty!(refused);
}

/// Raptor's unbracketed citation (2026-10-13 noon): `..., per @h | stamp.`
/// with the sentence's own full stop after it. It names one given tweet, so it
/// is rewritten to the canonical `(@h | stamp)` and the stop stays outside.
/// Without the `@` a bare bar is prose, never a citation.
#[test]
fn a_bracketless_citation_is_repaired_to_the_canonical_form() {
    let g = given();
    let answer = "## Topic\n\n\
        - Alice reports a thing, per @alice | Sat Oct 10 09:46:22 +0000 2026.\n\
        - Bob reports another, per @bob | Sat Oct 10 10:32 and nothing else.\n\
        - Carol at the time, @carol | 13:30:45\n\
        - The ratio was wins | losses, per nobody\n";
    let fixed = repair(answer, &g).expect("every bare citation names a given tweet");
    assert!(
        fixed
            .body
            .contains("per (@alice | Sat Oct 10 09:46:22 +0000 2026)."),
        "{}",
        fixed.body
    );
    assert!(
        fixed
            .body
            .contains("per (@bob | Sat Oct 10 10:32:00 +0000 2026) and nothing else."),
        "{}",
        fixed.body
    );
    assert!(
        fixed
            .body
            .contains("(@carol | Sat Oct 10 13:30:45 +0000 2026)"),
        "{}",
        fixed.body
    );
    assert!(
        fixed.body.contains("wins | losses, per nobody"),
        "{}",
        fixed.body
    );
}

/// A bare citation naming no given tweet counts against the answer like a
/// bracketed one: it is not silently trusted.
#[test]
fn an_unresolvable_bracketless_citation_still_counts_against_the_answer() {
    let g = given();
    let answer = "## Topic\n\n- a fact, per @mallory | Sat Oct 10 09:46:22 +0000 2026.\n";
    let why = repair(answer, &g).expect_err("the only citation names no given tweet");
    assert!(why.contains("1 of 1 citations"), "{why}");
}
