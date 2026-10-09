//! What a task's identity covers, and the registry that carries it.
//!
//! The cases that matter are the ones a NAME-ONLY store gets wrong: the same
//! task before and after its prompt changed (`file_summary`, 2026-10-08), and
//! the same task rendered from two different directories (which must be the
//! SAME identity, or the digest would be measuring the checkout rather than
//! the question).

use super::*;
use crate::ztools::eval::task_loader::{Check, EvalTask};
use crate::ztools::eval::tasks::RosterInputs;

fn task(name: &str, prompt: &str) -> EvalTask {
    EvalTask::new(name, prompt, vec![Check::Contains("x".to_string())])
}

/// The shipped conf, for the loader-driven cases below.
fn shipped_conf() -> RosterInputs {
    let conf = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the checkout that holds rust/")
        .join("conf");
    RosterInputs {
        inputs: conf.join("eval_inputs.toml"),
        vision: conf.join("eval_vision.toml"),
    }
}

/// The shipped `filename` input, verbatim, so the "same bytes elsewhere" case
/// below is the same bytes rather than a second guess at them.
const SHIPPED_FILENAME: &str =
    "Screenshot showing login error: Invalid credentials. Please try again.";

/// A copy of the shipped `eval_inputs.toml` under a different directory, with
/// its `filename` row replaced by `filename_prompt`.
///
/// The copy is what makes the case real: the same bytes, read from a path that
/// is not this checkout. The temp dir is RETURNED rather than dropped, because
/// the roster reads it after this function returns.
fn inputs_elsewhere_with(filename_prompt: &str) -> (tempfile::TempDir, RosterInputs) {
    let dir = tempfile::tempdir().expect("a temp dir");
    let shipped = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the checkout that holds rust/")
            .join("conf/eval_inputs.toml"),
    )
    .expect("the shipped eval_inputs.toml");
    let needle = format!("filename = \"{SHIPPED_FILENAME}\"");
    assert!(
        shipped.contains(&needle),
        "the shipped conf still spells the filename row this test edits"
    );
    let edited = shipped.replace(&needle, &format!("filename = \"{filename_prompt}\""));
    assert!(
        edited.contains(&format!("filename = \"{filename_prompt}\"")),
        "the copy must carry the row this case asks for"
    );
    let path = dir.path().join("eval_inputs.toml");
    std::fs::write(&path, edited).expect("write the edited inputs");
    let files = RosterInputs {
        inputs: path,
        vision: shipped_conf().vision,
    };
    (dir, files)
}

fn fingerprint_of(files: &RosterInputs, name: &str) -> String {
    let tasks = crate::ztools::eval::load_all_eval_tasks(files, None)
        .expect("the roster loads from the shipped conf");
    let found = tasks
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("the roster has no {name}"));
    task_fingerprint(found)
}

#[test]
fn the_same_task_always_fingerprints_the_same() {
    let t = task("t", "summarise this");
    let first = task_fingerprint(&t);
    // Rebuilt from scratch, not cloned: a digest that depended on where the
    // value came from would differ between the two.
    let second = task_fingerprint(&task("t", "summarise this"));
    assert_eq!(first, second);
    // 16 hex chars, no more: the digest is stored on every row of two files.
    assert_eq!(first.len(), 16, "{first}");
    assert!(first.chars().all(|c| c.is_ascii_hexdigit()), "{first}");
    // And PINNED: the digest is a pure function of the task, so the same task
    // must produce the same string in every process on every machine, forever.
    // A deliberate change to the payload regenerates this deliberately.
    assert_eq!(first, "3dd13fa25e7b1aea");
}

#[test]
fn one_character_in_a_message_changes_the_identity() {
    let a = task_fingerprint(&task("t", "summarise this"));
    let b = task_fingerprint(&task("t", "summarise this."));
    let c = task_fingerprint(&task("t", "summarise  this"));
    assert_ne!(a, b, "{a} vs {b}");
    assert_ne!(a, c, "a space is a character: {a} vs {c}");
}

#[test]
fn the_name_the_checks_and_parse_json_all_count() {
    let base = task_fingerprint(&task("t", "p"));
    assert_ne!(base, task_fingerprint(&task("u", "p")), "the name counts");
    assert_ne!(
        base,
        task_fingerprint(&EvalTask::new(
            "t",
            "p",
            vec![
                Check::Contains("x".to_string()),
                Check::NotContains("y".to_string())
            ]
        )),
        "a check counts"
    );
    assert_ne!(
        base,
        task_fingerprint(&EvalTask::new("t", "p", Vec::new())),
        "the ABSENCE of a check counts"
    );
    assert_ne!(
        base,
        task_fingerprint(&task("t", "p").json()),
        "parse_json counts"
    );
}

#[test]
fn a_message_role_change_and_an_image_change_count() {
    let plain = task_fingerprint(&EvalTask::new("t", "p", Vec::new()));
    let system = task_fingerprint(&EvalTask::with_system("t", "sys", "p", Vec::new()));
    assert_ne!(plain, system, "the role of a message counts");
    let with_image = task_fingerprint(&EvalTask {
        name: "t".to_string(),
        messages: vec![ChatMessage::user_with_images(
            "p",
            vec!["data:image/png;base64,AAAA".to_string()],
        )],
        checks: Vec::new(),
        parse_json: false,
    });
    assert_ne!(plain, with_image, "an image is part of what the task sends");
}

#[test]
fn the_check_order_counts() {
    let forward = task_fingerprint(&EvalTask::new(
        "t",
        "p",
        vec![
            Check::Contains("x".to_string()),
            Check::NotContains("y".to_string()),
        ],
    ));
    let reversed = task_fingerprint(&EvalTask::new(
        "t",
        "p",
        vec![
            Check::NotContains("y".to_string()),
            Check::Contains("x".to_string()),
        ],
    ));
    assert_ne!(forward, reversed);
}

/// The digest must describe the QUESTION, not the checkout it was rendered in.
/// The same roster bytes read from a different directory is the same task.
#[test]
fn the_same_roster_from_a_different_directory_is_the_same_task() {
    let here = fingerprint_of(&shipped_conf(), "filename");
    let (_dir, elsewhere) = inputs_elsewhere_with(SHIPPED_FILENAME);
    let there = fingerprint_of(&elsewhere, "filename");
    assert_eq!(
        here, there,
        "a path is not part of a task's identity: {here} vs {there}"
    );
}

/// The same proof at the place it actually bit: the file-summary prompt is
/// RENDERED out of a checkout root, and a digest that folded that root in would
/// fingerprint the same question differently per checkout. Two temp roots with
/// identical bytes, so the only thing that differs is where they live.
#[test]
fn the_same_file_contents_under_two_roots_are_the_same_task() {
    use crate::ztools::eval::prompts::file_summary::FILE_SUMMARY_FILE_LIST;
    use crate::ztools::eval::prompts::{FILE_SUMMARY_PROMPT, FILE_SUMMARY_PROMPT_MIXED};
    use crate::ztools::eval::task_loader::ChatMessage;
    use std::path::PathBuf;

    fn checkout() -> PathBuf {
        let root = tempfile::tempdir().expect("a temp root").keep();
        for row in FILE_SUMMARY_FILE_LIST
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
        {
            let path = root.join(row);
            std::fs::create_dir_all(path.parent().expect("a row has a parent"))
                .expect("make the row's directory");
            std::fs::write(&path, "the same bytes, wherever they live\n").expect("write the row");
        }
        root
    }

    let task_from = |root: &PathBuf| -> EvalTask {
        let plain = crate::ztools::eval::prompts::file_summary::render_both_from(
            root,
            FILE_SUMMARY_PROMPT,
            FILE_SUMMARY_PROMPT_MIXED,
        )
        .expect("the fixture checkout holds every row")
        .0;
        EvalTask {
            name: "file_summary".to_string(),
            messages: vec![
                ChatMessage::system("Output JSON now."),
                ChatMessage::user(plain),
            ],
            checks: Vec::new(),
            parse_json: true,
        }
    };

    let (first, second) = (checkout(), checkout());
    assert_ne!(first, second, "two different roots");
    assert_eq!(
        task_fingerprint(&task_from(&first)),
        task_fingerprint(&task_from(&second)),
        "the checkout a prompt was rendered from is not part of its identity"
    );

    // And the contents ARE: change one row's bytes in the second root and the
    // digest must move -- that is the whole point of the 2026-10-08 fix.
    let row = FILE_SUMMARY_FILE_LIST
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .expect("a listed row");
    std::fs::write(second.join(row), "DIFFERENT bytes\n").expect("rewrite the row");
    assert_ne!(
        task_fingerprint(&task_from(&first)),
        task_fingerprint(&task_from(&second)),
        "the contents the prompt embeds are part of its identity"
    );
}

/// A config change that alters a prompt DOES move the identity — the prompt is
/// the task. This is the seam that makes `file_summary`'s 2026-10-08 change
/// visible to every aggregate, without any of them knowing what M1 was.
#[test]
fn a_config_change_that_alters_a_prompt_moves_the_identity() {
    let before = fingerprint_of(&shipped_conf(), "filename");
    let (_dir, elsewhere) =
        inputs_elsewhere_with("Screenshot showing login error: Invalid credentials.");
    let after = fingerprint_of(&elsewhere, "filename");
    assert_ne!(before, after, "{before} vs {after}");
}

/// The registry is how a writer that only knows a task NAME resolves its
/// digest; an unregistered name is UNKNOWN, never silently current.
#[test]
fn the_registry_resolves_what_the_loader_registered_and_nothing_else() {
    let registered = task("registry-only-task", "p");
    remember_current_tasks(std::slice::from_ref(&registered));
    assert_eq!(
        current_task_fingerprint("registry-only-task").as_deref(),
        Some(task_fingerprint(&registered).as_str())
    );
    assert!(
        current_task_fingerprint("nobody-registered-this").is_none(),
        "an unknown name must not resolve to anything"
    );
    let identities = current_task_identities();
    assert_eq!(
        identities.get("registry-only-task").map(String::as_str),
        Some(task_fingerprint(&registered).as_str())
    );
    // Registering again is idempotent: the same task, the same digest.
    remember_current_tasks(&[registered]);
    assert_eq!(
        current_task_fingerprint("registry-only-task").as_deref(),
        Some(task_fingerprint(&task("registry-only-task", "p")).as_str())
    );
}

#[test]
fn standing_splits_current_superseded_and_unrecorded() {
    let mut current = TaskIdentities::new();
    current.insert("t".to_string(), "aaaa".to_string());

    assert_eq!(
        standing_of(Some("aaaa"), &current, "t"),
        Standing::Current,
        "the same task: countable"
    );
    assert!(standing_of(Some("aaaa"), &current, "t").counts());
    assert_eq!(
        standing_of(Some("bbbb"), &current, "t"),
        Standing::Superseded,
        "the task was replaced under the same name"
    );
    assert_eq!(
        standing_of(None, &current, "t"),
        Standing::Unrecorded,
        "an entry written before fingerprints existed"
    );
    assert_eq!(
        standing_of(Some("aaaa"), &current, "gone"),
        Standing::Unrecorded,
        "a task this process has no current identity for"
    );
    for standing in [Standing::Superseded, Standing::Unrecorded] {
        assert!(!standing.counts(), "{standing:?} must not be averaged");
    }
}
