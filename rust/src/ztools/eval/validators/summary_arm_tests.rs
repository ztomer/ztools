//! Arm-level cross-checks ported from `test_text_validator_1.py`'s
//! `TestValidateSummary`. The comments there have rotted (stale
//! arithmetic), but every asserted TOTAL matches the implementation, so
//! these pin the port arm by arm.
use super::summary::validate_summary;

#[test]
fn headers_and_bullets_score_40() {
    assert_eq!(
        validate_summary("## Section 1\n- bullet one\n- bullet two", "").0,
        40
    );
}

#[test]
fn headers_only_scores_45() {
    assert_eq!(
        validate_summary("## Section 1\n## Section 2\nsome content here", "").0,
        45
    );
}

#[test]
fn long_bullet_body_scores_18() {
    let text = format!("- {}", "a ".repeat(200));
    assert_eq!(validate_summary(&text, "").0, 18);
}

#[test]
fn long_body_without_markers_scores_15() {
    let text = "a ".repeat(200);
    assert_eq!(validate_summary(&text, "").0, 15);
}

#[test]
fn user_counts_score_25_20_15() {
    assert_eq!(validate_summary("@user1 @user2 @user3 hello", "").0, 25);
    assert_eq!(validate_summary("@user1 @user2 hello", "").0, 20);
    assert_eq!(validate_summary("@user1 hello", "").0, 15);
    assert_eq!(
        validate_summary("user 1 user 2 user 3 did things", "").0,
        25
    );
}

#[test]
fn timestamp_body_scores_20() {
    assert_eq!(validate_summary("At 10:30 something happened", "").0, 20);
}

#[test]
fn narrative_verbs_score_35() {
    assert_eq!(
        validate_summary("user asks and then responds and thanks", "").0,
        35
    );
}

#[test]
fn synthesis_variants_score_20_except_tldr() {
    assert_eq!(
        validate_summary("Overall, this is a summary of events. ## Section", "").0,
        20
    );
    assert_eq!(
        validate_summary("Key takeaways: lots happened. ## Section", "").0,
        20
    );
    assert_eq!(
        validate_summary("TL;DR: short version. ## Section", "").0,
        35
    );
    assert_eq!(
        validate_summary("In short, things happened. ## Section", "").0,
        20
    );
    assert_eq!(
        validate_summary("This conversation was interesting. ## Section", "").0,
        20
    );
    assert_eq!(validate_summary("no headers here just text", "").0, 10);
}

#[test]
fn topic_counts_score_50_40_20() {
    assert_eq!(
        validate_summary("## Topic1\n## Topic2\n- bullets", "").0,
        50
    );
    assert_eq!(validate_summary("## Topic1\n- bullets", "").0, 40);
    assert_eq!(
        validate_summary(
            "First this happened. Then that. Also meanwhile something.",
            ""
        )
        .0,
        20
    );
}

#[test]
fn template_and_boilerplate_are_named() {
    let (_, msg) = validate_summary("**Who: x\n**What: y\n**When: z\nbody text here", "");
    assert!(msg.contains("template-driven"), "{msg}");
    let (_, msg) = validate_summary("Some content with not specified value here", "");
    assert!(msg.contains("boilerplate"), "{msg}");
}
