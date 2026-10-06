//! The file-summary family's pins: the row-for-row file list, the two prompts
//! that wrap it verbatim, the content block that makes "rely only on provided
//! content" followable, and the two structural checks that make a row a FILE
//! rather than a plausible-looking path.
//!
//! Split out of `mod.rs` for the house 500-line cap. It was one test there; it
//! is three files' worth of property now, because a row-for-row pin is not
//! enough on its own — see `every_listed_path_is_a_file_a_fresh_clone_has`.

use std::collections::HashSet;
use std::path::PathBuf;

use super::file_summary::{
    FILE_SUMMARY_CONTENTS_SLOT, FILE_SUMMARY_EXCERPT_LINES, FILE_SUMMARY_GROUND_TRUTH, excerpt,
    excerpts, render, render_both,
};
use super::{FILE_SUMMARY_FILE_LIST, FILE_SUMMARY_PROMPT, FILE_SUMMARY_PROMPT_MIXED};

/// `<repo>`, from `CARGO_MANIFEST_DIR` (which is `<repo>/rust`).
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate lives in <repo>/rust")
        .to_path_buf()
}

/// `git ls-files`, recorded. A path that is not in a fresh clone must not be
/// named by a prompt, and `git ls-files` is the only honest answer to "is this
/// file in a clone?" — a filesystem probe says `pyproject.toml` is right here
/// on this machine, which is exactly how an untracked file got into this list.
///
/// The fixture is a SUBSET of the tracked set by construction (it was written
/// by `git ls-files > tests/fixtures/repo_tracked_paths.txt`), so it cannot
/// bless an untracked file, and it fails in the safe direction: a newly tracked
/// file missing from it makes this go RED, never green.
fn tracked_paths() -> HashSet<String> {
    let path = repo_root().join("tests/fixtures/repo_tracked_paths.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines().map(str::to_string).collect()
}

/// The rows, repo-relative. The list is absolute and rooted at one repo, so it
/// is pinned by the tail: the machine prefix is not a property of the prompt.
const LISTED: &[&str] = &[
    "README.md",
    "CLAUDE.md",
    "conf/config.toml",
    "conf/weekend.toml",
    "conf/twitter.toml",
    "conf/rename.toml",
    "conf/models/foundation.toml",
    "conf/models/gemma.toml",
    "conf/models/qwen.toml",
    "docs/MODEL_QUIRKS.md",
    "docs/TESTING.md",
    "rust/src/manifest.rs",
    "rust/src/units.rs",
    "rust/src/ztools/image_renamer.rs",
    "rust/src/ztools/store.rs",
    "rust/src/ztools/twitter/mod.rs",
    "rust/src/ztools/weekend/format.rs",
];

const NOISE: &[&str] = &[
    "/fake/path/nonexistent_file.txt",
    "/totally/made/up/directory/garbage.log",
    "/random/asdfghjkl/qwertyuiop.zxc",
    "/spam/buy_now/click_here.exe",
    "/irrelevant/crypto_price_predictions.md",
    "/hallucinated/alien_landing_report.pdf",
];

/// One list, wrapped by both prompts, plus the noise block only the mixed one
/// carries.
#[test]
fn the_file_summary_prompts_wrap_one_pinned_file_list() {
    let listed: Vec<&str> = FILE_SUMMARY_FILE_LIST
        .lines()
        .map(|line| {
            line.rsplit_once("/Users/ztomer/Projects/ztools/")
                .unwrap_or_else(|| panic!("{line:?} is not a repo-absolute path"))
                .1
        })
        .collect();
    assert_eq!(listed, LISTED, "FILE_SUMMARY_FILE_LIST, row for row");
    // Both prompts must wrap THAT list verbatim — the single-source property
    // that makes the rows above worth pinning once.
    for (name, prompt) in [
        ("FILE_SUMMARY_PROMPT", FILE_SUMMARY_PROMPT),
        ("FILE_SUMMARY_PROMPT_MIXED", FILE_SUMMARY_PROMPT_MIXED),
    ] {
        assert!(
            prompt.contains(FILE_SUMMARY_FILE_LIST),
            "{name} no longer wraps FILE_SUMMARY_FILE_LIST verbatim"
        );
        assert!(
            prompt.starts_with("Read the file list below and give one-line summary for each file."),
            "{name} lost its opening instruction"
        );
        assert!(
            prompt.contains("DO NOT infer functionality from file names,"),
            "{name} lost the anti-inference rule it exists to test"
        );
    }
    for noise in NOISE {
        assert!(
            FILE_SUMMARY_PROMPT_MIXED.contains(noise),
            "the mixed prompt lost the noise file {noise}"
        );
        assert!(
            !FILE_SUMMARY_PROMPT.contains(noise),
            "the honest prompt must not carry the noise file {noise}"
        );
    }
    assert!(FILE_SUMMARY_PROMPT_MIXED.contains("NOISE FILES (Ignore - test your filtering):"));
    assert!(!FILE_SUMMARY_PROMPT.contains("NOISE FILES"));
}

/// The class, closed: a row is a path a model is asked to describe, so it must
/// be a file that EXISTS and that a fresh CLONE has.
///
/// Before this, the list named sixteen `references/**` paths deleted with the
/// Python runtime (ddb7d53) and one untracked `pyproject.toml`, for a month,
/// with nothing red: the row-for-row pin above agreed with the constant, and
/// the constant agreed with a dead tree. A pin that cannot see the working tree
/// pins the mistake instead of catching it.
///
/// Both halves are needed and they fail differently. Existence catches a
/// deletion; trackedness catches a file that is real here and absent from every
/// clone — the case a filesystem probe reports as fine.
#[test]
fn every_listed_path_is_a_file_a_fresh_clone_has() {
    let tracked = tracked_paths();
    assert_nonempty!(
        &tracked,
        "tests/fixtures/repo_tracked_paths.txt is empty, so this check would pass by \
         vacuity; regenerate it with `git ls-files > tests/fixtures/repo_tracked_paths.txt`"
    );
    for row in LISTED {
        let path = repo_root().join(row);
        assert!(
            path.is_file(),
            "the prompt names {row}, which is not a file at {}. A row the model cannot \
             read is a row it is told not to guess about and then graded on guessing \
             about. Remove it, or add the file.",
            path.display()
        );
        assert!(
            tracked.contains(*row),
            "the prompt names {row}, which is UNTRACKED: a fresh clone does not have it. \
             `git add` it (the orchestrator commits), or remove the row — the filesystem \
             check above passes on this machine, which is why this one exists."
        );
    }
    // The regression, named: the deleted tree and the untracked row are the two
    // ways this list was fiction, and one line each keeps them from returning
    // by accident.
    for dead in ["references/", "pyproject.toml"] {
        assert!(
            !FILE_SUMMARY_FILE_LIST.contains(dead),
            "{dead} is back in the file list; see the header for why it left"
        );
    }
}

/// THE FIX, pinned: the prompt says "Rely ONLY on provided content context ...
/// DO NOT infer functionality from file names", so the content has to be IN it.
///
/// Before this, both templates carried seventeen paths and no content at all, and
/// `validate_file_summary` scored the resulting guesses anyway. A pin that only
/// checked the list and the instruction would have passed on that prompt for
/// another month, which is why the slot and what fills it are both asserted here.
#[test]
fn the_rendered_prompt_carries_the_files_own_content() {
    for (name, template) in [
        ("FILE_SUMMARY_PROMPT", FILE_SUMMARY_PROMPT),
        ("FILE_SUMMARY_PROMPT_MIXED", FILE_SUMMARY_PROMPT_MIXED),
    ] {
        assert!(
            template.contains(FILE_SUMMARY_CONTENTS_SLOT),
            "{name} lost its content slot, so it renders to a path list again"
        );
    }

    let block = excerpts().expect("the listed files are readable");
    for row in FILE_SUMMARY_FILE_LIST.lines() {
        let row = row.trim();
        assert!(
            block.contains(&format!("--- BEGIN {row} ")),
            "the content block does not fence {row}"
        );
        let body: String = block
            .lines()
            .skip_while(|l| !l.starts_with(&format!("--- BEGIN {row} ")))
            .skip(1)
            .take_while(|l| !l.starts_with("--- END "))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            body.chars().count() > 20,
            "{row} got a fence and no text: the model is back to guessing"
        );
    }

    let rendered = render(FILE_SUMMARY_PROMPT).expect("renders");
    assert!(
        !rendered.contains(FILE_SUMMARY_CONTENTS_SLOT),
        "the slot survived rendering"
    );
    assert!(
        rendered.contains(&format!(
            "--- BEGIN {} ",
            FILE_SUMMARY_FILE_LIST.lines().next().unwrap()
        )),
        "the rendered prompt does not carry the first listed file's content"
    );
}

/// The bound, stated as a NUMBER and enforced against it.
///
/// The excerpt is bounded because the prompt shares a context window with the
/// OUTPUT budget — `foundation` has 4096 tokens for both — and because the point
/// is to summarise a file, not to ingest it.
///
/// The ceiling below is ABSOLUTE on purpose. It was first written as
/// `LISTED.len() * FILE_SUMMARY_EXCERPT_BYTES + 4096`, which is a gate that
/// cannot fail: raising `FILE_SUMMARY_EXCERPT_BYTES` tenfold moved the ceiling
/// with it and this stayed green (calibrated). A budget derived from the knob it
/// polices measures nothing. The measured prompt is 23,150 bytes at the shipped
/// bounds, and 24,576 is the ceiling — about 5.8k tokens.
///
/// So a listed file that grows, a row added to the list, or a loosened constant
/// all go RED here rather than quietly costing every model its context window.
#[test]
fn the_rendered_prompt_stays_inside_its_stated_budget() {
    const CEILING_BYTES: usize = 24_576;
    let rendered = render(FILE_SUMMARY_PROMPT).expect("renders");
    assert!(
        rendered.len() <= CEILING_BYTES,
        "the rendered file-summary prompt is {} bytes, over its {CEILING_BYTES}-byte \
         ceiling. MODEL_QUIRKS.md alone is 65KB; lower FILE_SUMMARY_EXCERPT_BYTES or \
         FILE_SUMMARY_EXCERPT_LINES deliberately, or raise this ceiling as its own \
         decision with the token cost stated.",
        rendered.len()
    );
    assert!(
        rendered.len() >= LISTED.len() * 400,
        "the content block collapsed to {} bytes: the bound is now tighter than any \
         of these files' heads, so the model cannot read any of them",
        rendered.len()
    );
}

/// The truncation is SAID, not silent: a model told it is seeing a head of the
/// file can hedge honestly, and one told nothing will guess at the rest. The
/// smallest listed file is whole, so the note must not be a blanket lie.
#[test]
fn truncation_is_stated_and_short_files_are_not_announced_as_cut() {
    let block = excerpts().expect("the listed files are readable");
    let mut saw_cut = false;
    let mut saw_whole = false;
    for (i, row) in FILE_SUMMARY_FILE_LIST.lines().enumerate() {
        let text = std::fs::read_to_string(row.trim()).unwrap_or_else(|e| panic!("{row}: {e}"));
        let (_, shown, total) = excerpt(&text);
        let note = format!("--- BEGIN {row} (showing {shown} of {total} lines)");
        let has_note = block.contains(&note);
        assert_eq!(
            shown < total,
            has_note,
            "row {i} ({row}): {shown} of {total} lines sent, so the truncation note \
             must be {has_note} -- a model told nothing will guess at the rest"
        );
        saw_cut |= has_note;
        saw_whole |= !has_note;
    }
    assert!(
        saw_cut && saw_whole,
        "both cases must occur in the shipped list"
    );
    assert!(
        block.contains("EXCERPTS"),
        "the block must label itself as excerpts"
    );
    assert!(
        block.contains(&format!("(showing {FILE_SUMMARY_EXCERPT_LINES} of")),
        "a file over the LINE bound must be cut at exactly the line bound"
    );
}

/// Every listed row has pinned ground truth, and the truth is keyed by the row.
/// A listed file with no truth has nothing to be graded against: the scorer falls
/// back to counting content verbs there, which is the measurement this whole
/// change removed.
#[test]
fn every_listed_row_has_pinned_ground_truth() {
    let pinned: HashSet<&str> = FILE_SUMMARY_GROUND_TRUTH
        .iter()
        .map(|(rel, _)| *rel)
        .collect();
    assert_eq!(
        pinned.len(),
        FILE_SUMMARY_GROUND_TRUTH.len(),
        "two pinned rows share a key, so one of them can never be reached"
    );
    for row in LISTED {
        assert!(
            pinned.contains(row),
            "{row} is listed but has no pinned truth, so a description of it is \
             graded on verb count alone"
        );
    }
    assert_eq!(pinned.len(), LISTED.len(), "a pinned row is not listed");

    // A row with an EMPTY phrase list is worse than an absent one: `grounded`
    // returns false for it, so the row scores zero however well the model read
    // the file, and the score reads as a model failure. Calibrated: emptying one
    // row's phrases left the key-presence half of this gate green.
    for (row, facts) in FILE_SUMMARY_GROUND_TRUTH {
        assert!(
            facts.len() >= 2,
            "{row} carries {} pinned phrase(s); the table's documented shape is two, \
             and fewer makes the row unscoreable while looking like a bad answer",
            facts.len()
        );
        for fact in *facts {
            assert!(
                !fact.trim().is_empty(),
                "{row} has a blank pinned phrase, which no description can match"
            );
        }
    }
}

/// Both templates must keep their own shape after the slot was added: the noise
/// block BELOW it, so `validate_mixed_file_summary` reads the content as signal
/// and the noise as noise. Content above the marker would score as precision loss
/// on every model.
#[test]
fn the_content_block_sits_above_the_noise_marker() {
    let slot = FILE_SUMMARY_PROMPT_MIXED
        .find(FILE_SUMMARY_CONTENTS_SLOT)
        .expect("the mixed template carries the slot");
    let noise = FILE_SUMMARY_PROMPT_MIXED
        .find("NOISE FILES")
        .expect("the mixed template carries the noise block");
    assert!(
        slot < noise,
        "the mixed prompt must carry file contents BEFORE its noise block: slot at          {slot}, noise at {noise}"
    );
    assert!(!FILE_SUMMARY_PROMPT.contains("NOISE"));
}

/// The dedup, pinned: `render_both` exists so seventeen files are read ONCE for
/// the pair, not once each. That optimisation is only safe while both prompts
/// carry the same slot and the same block, so that is asserted rather than
/// assumed — a template that drifted would otherwise render a different prompt
/// from `render_both` than from `render`, silently.
#[test]
fn one_content_block_serves_both_prompts() {
    let (plain, mixed) =
        render_both(FILE_SUMMARY_PROMPT, FILE_SUMMARY_PROMPT_MIXED).expect("renders");
    assert_eq!(plain, render(FILE_SUMMARY_PROMPT).expect("renders"));
    assert_eq!(mixed, render(FILE_SUMMARY_PROMPT_MIXED).expect("renders"));
    // The block is identical, and each prompt keeps exactly one copy of it: two
    // copies would mean the same files read and paid for twice in the request.
    let occurrences = |text: &str| text.matches("--- BEGIN ").count();
    assert_eq!(occurrences(&plain), LISTED.len());
    assert_eq!(occurrences(&mixed), LISTED.len());
}

/// `render_both` refuses a template that lost its slot rather than shipping a
/// prompt that names files and explains none of them.
#[test]
fn a_template_without_its_slot_is_an_error_not_a_silent_path_only_prompt() {
    let stripped = FILE_SUMMARY_PROMPT.replace(FILE_SUMMARY_CONTENTS_SLOT, "");
    let err = render_both(&stripped, FILE_SUMMARY_PROMPT_MIXED)
        .expect_err("a slotless template must not render")
        .to_string();
    assert!(err.contains(FILE_SUMMARY_CONTENTS_SLOT), "{err}");
    let err = render_both(FILE_SUMMARY_PROMPT, &stripped)
        .expect_err("a slotless template must not render")
        .to_string();
    assert!(err.contains("mixed"), "{err}");
}
