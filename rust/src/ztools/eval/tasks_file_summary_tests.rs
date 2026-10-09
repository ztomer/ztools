//! The roster's file-summary rows, asserted on the RENDERED prompt.
//!
//! Split out of `tasks_tests.rs` and into its own file for one reason: the pins in
//! `prompts/file_summary_pins_tests.rs` render from the TEMPLATES, so every one of
//! them stays green if `tasks.rs` hands the runner the un-rendered path list — the
//! exact prompt this whole change replaced. The gap is closed by building the real
//! roster and looking at what the model would be sent.
//!
//! It also holds the two things that can only be asserted HERE, now that the rows
//! are repo-relative and the root is resolved once in `roster()`: that the root is
//! resolved from the checkout derivation rather than guessed, and that a row the
//! resolved checkout does not have stops the roster instead of producing a
//! shorter prompt. Both need a fixture, which is why they moved out of the pins
//! file — a row used to be absolute, so there was no way to make one unreadable
//! without touching the working tree, and the gap was recorded instead of closed.

use super::*;
use crate::test_env::TestEnv;
use crate::ztools::eval::prompts::file_summary::{
    FILE_SUMMARY_CONTENTS_SLOT, FILE_SUMMARY_EXCERPT_BYTES, FILE_SUMMARY_FILE_LIST, live_root,
    live_root_from, render_from,
};
use std::path::Path;

fn roster_tasks() -> Vec<EvalTask> {
    roster(&RosterInputs::in_dir(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("conf"),
    ))
    .expect("roster loads from the shipped conf")
}

fn user_prompt(tasks: &[EvalTask], name: &str) -> String {
    tasks
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("{name} is not in the roster"))
        .messages
        .last()
        .expect("every task has a user message")
        .content
        .clone()
}

/// THE FIX AT THE SEAM, not at the template: the prompt the runner would send
/// carries every listed file's own text.
///
/// Calibrated: making `file_summary_tasks` pass the bare templates — which every
/// template-level pin still accepts — fails here.
#[test]
fn the_roster_sends_the_files_own_content_not_just_their_paths() {
    let tasks = roster_tasks();
    for name in ["file_summary", "file_summary_mixed"] {
        let prompt = user_prompt(&tasks, name);
        assert!(
            !prompt.contains(FILE_SUMMARY_CONTENTS_SLOT),
            "{name} shipped the unfilled slot: the model is told to rely on content \
             it was never given"
        );
        for row in FILE_SUMMARY_FILE_LIST.lines() {
            let row = row.trim();
            assert!(
                prompt.contains(&format!("--- BEGIN {row} ")),
                "{name} does not fence {row}'s content"
            );
        }
        assert_eq!(
            prompt.matches("--- BEGIN ").count(),
            FILE_SUMMARY_FILE_LIST.lines().count(),
            "{name}: exactly one excerpt per listed row"
        );
        assert!(
            prompt.contains("DO NOT infer functionality from file names,"),
            "{name} lost the rule the content makes followable"
        );
    }
}

/// The bound holds at the seam too, in BYTES, because that is what the context
/// window is spent in.
#[test]
fn the_rosters_prompts_are_inside_the_context_budget_they_claim() {
    let tasks = roster_tasks();
    for name in ["file_summary", "file_summary_mixed"] {
        let prompt = user_prompt(&tasks, name);
        let ceiling = (FILE_SUMMARY_FILE_LIST.lines().count() * FILE_SUMMARY_EXCERPT_BYTES) + 4096;
        assert!(
            prompt.len() <= ceiling,
            "{name} is {} bytes over its {ceiling}-byte ceiling",
            prompt.len()
        );
        // And the excerpt bounds are visible in what was sent, so a later edit to
        // either constant shows up here rather than only in the template tests.
        assert!(
            prompt.contains("(showing"),
            "{name}: a truncated excerpt is silent"
        );
    }
}

/// The mixed row's validator is handed the SAME text the model is shown — it
/// reads its signal set out of it. Handing it the template instead would score
/// the mixed variant against a prompt the model never saw, which is the drift
/// `the_source_is_the_prompt_the_model_was_shown` states as a rule.
#[test]
fn the_mixed_validator_is_handed_the_prompt_the_model_was_shown() {
    let tasks = roster_tasks();
    let prompt = user_prompt(&tasks, "file_summary_mixed");
    let Check::Graded(Graded::MixedFileSummary { source }) = tasks
        .iter()
        .find(|t| t.name == "file_summary_mixed")
        .expect("row exists")
        .checks
        .first()
        .expect("one check")
    else {
        panic!("file_summary_mixed is graded by validate_mixed_file_summary with a source");
    };
    assert_eq!(
        source.as_str(),
        prompt.as_str(),
        "the validator's source must be the rendered prompt the model saw"
    );
}

/// A slotless template, and a row the resolved checkout does not have, both stop
/// the render instead of shipping a prompt that names files and explains none of
/// them. `roster()` is on the path of every eval run, so a silent skip would be a
/// sweep that quietly measures a shorter task.
///
/// The slot half used to be the whole test, with an apology under it: the rows
/// were absolute, so there was no way to make a listed file unreadable without
/// editing the working tree. A row is relative now, so both halves are reachable
/// from a fixture checkout and both are asserted.
#[test]
fn a_missing_listed_file_is_an_error_not_a_shorter_prompt() {
    let stripped = FILE_SUMMARY_PROMPT.replace(FILE_SUMMARY_CONTENTS_SLOT, "");
    let err = crate::ztools::eval::prompts::file_summary::render_both(
        &stripped,
        FILE_SUMMARY_PROMPT_MIXED,
    )
    .expect_err("a slotless template must not render")
    .to_string();
    assert!(err.contains(FILE_SUMMARY_CONTENTS_SLOT), "{err}");

    let fixture = fixture_checkout();
    std::fs::remove_file(fixture.path().join("conf/rename.toml")).expect("a fixture row");
    let err = render_from(fixture.path(), FILE_SUMMARY_PROMPT)
        .expect_err("a row the checkout does not have must not render")
        .to_string();
    assert!(
        err.contains("listed file conf/rename.toml"),
        "the error must name the row: {err}"
    );
    assert!(
        err.contains(fixture.path().to_str().expect("a utf-8 temp path")),
        "the error must name the root it looked in: {err}"
    );
}

/// A fixture checkout in a per-test temp dir: both markers `manifest` accepts,
/// every listed row written with text no file in this repo contains. Same shape as
/// the pins file's, and separate on purpose — a shared helper would make the two
/// suites one change apart from each other.
fn fixture_checkout() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().expect("a fixture temp dir is creatable");
    write_fixture_rows(tmp.path());
    tmp
}

/// A fixture checkout AT a known directory, so several can exist at once with a
/// known order between them — which is what a precedence pin needs.
fn fixture_checkout_at(root: &Path) {
    std::fs::create_dir_all(root.parent().expect("a fixture root has a parent"))
        .expect("a fixture root's parent");
    write_fixture_rows(root);
}

fn write_fixture_rows(root: &Path) {
    std::fs::create_dir_all(root.join("rust")).expect("a fixture marker directory");
    std::fs::create_dir_all(root.join("conf")).expect("a fixture marker directory");
    std::fs::write(root.join("rust/Cargo.toml"), "[package]\n").expect("a marker");
    std::fs::write(root.join("conf/config.toml"), "[best_models]\n").expect("a marker");
    for row in FILE_SUMMARY_FILE_LIST.lines() {
        let path = root.join(row);
        std::fs::create_dir_all(path.parent().expect("every row has a parent"))
            .expect("a fixture row's directory");
        std::fs::write(&path, format!("fixture row: {row}\n")).expect("a fixture row");
    }
}

/// THE ROOT IS DERIVED, NOT GUESSED. The roster resolves it once, above every row
/// that needs it, from the checkout derivation every other shipped default uses:
/// the checkout above the running executable first, the home checkout second, the
/// checkout this binary was built from last. It is NOT the working directory,
/// which is right for `cargo test` in a repo and wrong for an installed binary and
/// for every CI runner.
///
/// Written against the `*_from` twin rather than the live default, for the reason
/// every other twin in this crate exists: the live answer is decided by the BUILD
/// LAYOUT and by whatever `ZTOOLS_EXE` and `$HOME` a concurrently running sandbox
/// has left behind, so a test written against it would pin the machine rather than
/// the order. The live path is pinned by the roster tests above, and by
/// `the_live_seam_is_not_decided_by_a_peers_sandbox`.
#[test]
fn the_checkout_is_derived_in_order_not_guessed_from_the_working_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    fixture_checkout_at(&home.join("Projects/ztools"));
    let installed = tmp.path().join("opt/ztools");
    fixture_checkout_at(&installed);
    let built = tmp.path().join("build/ztools");
    fixture_checkout_at(&built);
    let exe = installed.join("bin/ztools");

    // Everything present: the install-from checkout wins. The other two are in the
    // list too, which is what tells an order apart from a coincidence.
    assert_eq!(
        live_root_from(Some(&exe), Some(&home), Some(&built)).expect("a checkout"),
        installed,
        "the checkout the executable was INSTALLED FROM beats the home fallback and \
         the build checkout"
    );
    // No executable: the home checkout is next, and it is not the build checkout.
    assert_eq!(
        live_root_from(None, Some(&home), Some(&built)).expect("a checkout"),
        home.join("Projects/ztools"),
        "with no executable to walk, the home checkout is the operator's answer"
    );
    // Neither: the build checkout — the candidate that makes an installed binary
    // and a shared build dir outside the repo both work.
    assert_eq!(
        live_root_from(None, None, Some(&built)).expect("a checkout"),
        built,
        "the checkout this binary was BUILT FROM is last, and it is the one no \
         environment redirect can move"
    );
    // And it is not appended twice when it is the same directory as a derived one.
    assert_eq!(
        live_root_from(Some(&exe), Some(&home), Some(&installed)).expect("a checkout"),
        installed,
        "a build checkout that IS a derived root is one candidate, not two"
    );
}

/// A root that holds only SOME of the rows is not a checkout of these rows: it is
/// an unrelated `conf/`, a different checkout, or a partial worktree. Accepting it
/// would render a prompt naming eighteen files and explaining seventeen.
#[test]
fn a_root_holding_some_rows_is_not_the_checkout() {
    let tmp = tempfile::tempdir().unwrap();
    // Both markers, sixteen of the seventeen rows, and one row missing — the shape
    // an unrelated `conf/` or a partial worktree has.
    let partial = tmp.path().join("Projects/ztools");
    std::fs::create_dir_all(partial.join("rust")).unwrap();
    std::fs::create_dir_all(partial.join("conf")).unwrap();
    std::fs::write(partial.join("rust/Cargo.toml"), "[package]\n").unwrap();
    std::fs::write(partial.join("conf/config.toml"), "[best_models]\n").unwrap();
    for row in FILE_SUMMARY_FILE_LIST.lines().take(16) {
        let path = partial.join(row);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "partial\n").unwrap();
    }
    let good = tmp.path().join("good");
    fixture_checkout_at(&good);

    assert_eq!(
        live_root_from(None, Some(tmp.path()), Some(&good)).expect("a checkout"),
        good,
        "the root missing ONE row is skipped and the next candidate is taken"
    );
}

/// NO candidate holding every row is an ERROR naming every candidate tried —
/// never a silent fall back to the working directory, and never `.`.
///
/// The candidates are arguments, so every branch is reachable: none at all, one
/// that exists but holds nothing of ours, and one that is not even a directory.
#[test]
fn no_checkout_holding_every_row_names_every_candidate_it_tried() {
    let err = live_root_from(None, None, None)
        .expect_err("no candidate at all")
        .to_string();
    assert!(
        err.contains("no candidate"),
        "an empty candidate list must say so rather than print nothing: {err}"
    );

    let tmp = tempfile::tempdir().unwrap();
    // A real checkout root that holds NONE of the rows: the markers are there, the
    // files are not. This is a different something — a fresh clone of another
    // project's ztools, a half-checked-out worktree.
    let markers_only = tmp.path().join("opt/ztools");
    std::fs::create_dir_all(markers_only.join("rust")).unwrap();
    std::fs::create_dir_all(markers_only.join("conf")).unwrap();
    std::fs::write(markers_only.join("rust/Cargo.toml"), "[package]\n").unwrap();
    std::fs::write(markers_only.join("conf/config.toml"), "[best_models]\n").unwrap();
    let exe = markers_only.join("bin/ztools");
    std::fs::create_dir_all(exe.parent().unwrap()).unwrap();
    let empty_home = tmp.path().join("empty");
    std::fs::create_dir_all(&empty_home).unwrap();

    let err = live_root_from(Some(&exe), Some(&empty_home), None)
        .expect_err("neither candidate holds the rows")
        .to_string();
    // BOTH candidates, spelled the way the derivation spells them. An error that
    // named one of them would hide where the next reader should look.
    assert!(
        err.contains(&markers_only.display().to_string()),
        "the checkout above the executable must be named: {err}"
    );
    assert!(
        err.contains(&empty_home.join("Projects/ztools").display().to_string()),
        "the home fallback must be named: {err}"
    );
    assert!(
        !err.contains("tried ."),
        "the working directory must not stand in for a checkout: {err}"
    );
}

/// The live seam is not decided by a peer's sandbox.
///
/// WHY THIS TEST EXISTS, and what it cost to learn. `live_root` reads `ZTOOLS_EXE`
/// and `$HOME`, and `TestEnv` redirects both — but a `TestEnv` excludes other
/// `TestEnv`s, not the non-serial tests running beside it. So the first version of
/// this change failed seventeen tests in this suite whenever a sandbox happened
/// to be live: a roster built while a peer held the lock resolved to a directory
/// that does not exist. Measured 2026-10-08, not theorised. The fix was to make the
/// build-time checkout the last candidate, so the ambient answer cannot be moved
/// by an environment write; this test pins that a live root is still resolvable
/// while a sandbox is held.
#[test]
fn the_live_seam_is_not_decided_by_a_peers_sandbox() {
    let env = TestEnv::new();
    let root = live_root().expect("the live root resolves while a sandbox is held");
    assert!(
        root.join("rust/Cargo.toml").is_file() && root.join("conf/config.toml").is_file(),
        "{root:?} is not a checkout"
    );
    for row in FILE_SUMMARY_FILE_LIST.lines() {
        assert!(root.join(row).is_file(), "{row} is not under {root:?}");
    }
    drop(env);
}
