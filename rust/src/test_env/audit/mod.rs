//! The gates that keep [`super::TestEnv`] from going stale, and from being
//! forgettable.
//!
//! A guard is only as good as the list inside it, and a list nobody re-checks
//! rots the moment a new variable is read. So each of these re-derives its
//! expectation from the SOURCES on every test run and fails when they disagree
//! — the property "you cannot add a path-reading variable without the sandbox
//! knowing" is enforced by the build, not by whoever writes the next test.
//!
//! Three of them carry an allowlist, [`OUTSIDE_THE_CONTRACT`], which is EMPTY.
//! It is kept as an empty list rather than deleted, and it is a ratchet in both
//! directions: an entry whose file has been fixed fails the gate, so the next
//! file that needs one has to re-open the question here rather than invent a
//! private exemption that nothing re-checks.
//!
//! The DETECTORS these gates call live in [`text`], beside the snippets that
//! calibrate them.

mod text;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use super::table::{EXTERNAL, MANAGED};
use text::{Body, blank_out_literals, env_vars_read_by, fixed_temp_paths, hazard_in, test_bodies};

/// The gate reads the crate's own sources, not a copy of them.
fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every `.rs` file the crate is built from, minus this module's own.
///
/// `src/test_env/` is excluded because its sources are the MANIFEST and the
/// detectors' calibration snippets: a literal like `NOT_A_ZTOOLS_VARIABLE` in
/// a test fixture is the detector proving it can fail, not a variable the
/// crate reads. Scanning them would make every gate self-satisfying.
fn rust_sources() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    let root = crate_root();
    let mut out = Vec::new();
    walk(&root.join("src"), &mut out);
    if let Ok(entries) = std::fs::read_dir(root.join("tests")) {
        out.extend(
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "rs")),
        );
    }
    let manifest = relative(&root.join("src/test_env/mod.rs"));
    let _ = manifest;
    out.retain(|p| !relative(p).starts_with("src/test_env"));
    out.sort();
    out
}

fn relative(path: &Path) -> String {
    path.strip_prefix(crate_root())
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

/// Files whose tests are outside the sandbox contract.
///
/// EMPTY, and kept as an empty list rather than deleted, because the two gates
/// that consult it ([`every_hazard_test_constructs_a_test_env`] and
/// [`no_test_names_a_fixed_directory_under_the_system_temp_dir`]) read it, and
/// because the next file that needs one should have to re-open the question in
/// the same place rather than re-invent a private exemption.
///
/// It held seven entries on 2026-10-05 — five files whose tests built a
/// `ZtoolsConfig` without the guard, and two that named a fixed directory under
/// the system temp dir — and they are gone because the files were fixed, not
/// because the list was. `OUTSIDE_THE_CONTRACT` is a ratchet in both
/// directions: [`the_outside_the_contract_allowlist_only_names_files_that_still_exist`]
/// fails on an entry whose file has nothing left to allowlist, and the debt
/// line it prints is the number this list is supposed to drive to zero.
const OUTSIDE_THE_CONTRACT: &[&str] = &[];

// --- the gates ---------------------------------------------------------------

/// GATE 1. Every variable the crate reads a path or a policy out of is managed
/// by the sandbox.
#[test]
fn every_env_var_the_crate_reads_is_managed() {
    let mut unmanaged: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for path in rust_sources() {
        let text = std::fs::read_to_string(&path).expect("a source file is readable");
        for name in env_vars_read_by(&text) {
            if !MANAGED.contains(&name.as_str()) && !EXTERNAL.contains(&name.as_str()) {
                unmanaged.entry(name).or_default().push(relative(&path));
            }
        }
    }
    assert!(
        unmanaged.is_empty(),
        "these variables are read but not sandboxed, so a test cannot keep its \
         reads and writes off the operator's machine through TestEnv: {unmanaged:?}"
    );
}

/// GATE 2. A test that could reach the operator's disk takes the guard.
#[test]
fn every_hazard_test_constructs_a_test_env() {
    let mut unguarded: Vec<String> = Vec::new();
    for path in rust_sources() {
        let rel = relative(&path);
        if OUTSIDE_THE_CONTRACT.contains(&rel.as_str()) {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("a source file is readable");
        for body in test_bodies(&text) {
            report_if_unguarded(&rel, &body, &mut unguarded);
        }
    }
    assert!(
        unguarded.is_empty(),
        "these tests can read or write the operator's real HOME and take no \
         TestEnv, so their isolation is a coincidence of code order: {unguarded:?}"
    );
}

/// One finding, or none — the whole of GATE 2's per-test decision.
///
/// Split out so the gate and the debt counter below cannot drift into two
/// different definitions of "unguarded". The guard is looked for in `code`, not
/// `source`, for the same reason the hazards are: a comment that says
/// `TestEnv::new()` is not a test that calls it.
fn report_if_unguarded(rel: &str, body: &Body, out: &mut Vec<String>) {
    let Some(hazard) = hazard_in(&body.code) else {
        return;
    };
    if body.code.contains("TestEnv::new()") || body.code.contains("TestEnv::new_locked()") {
        return;
    }
    let name = body
        .source
        .lines()
        .find_map(|l| l.trim().strip_prefix("fn "))
        .unwrap_or("<unnamed>")
        .split(['(', ' '])
        .next()
        .unwrap_or("<unnamed>")
        .to_string();
    out.push(format!("{rel}: {name} ({hazard})"));
}

/// GATE 3. No test names a fixed directory under the system temp dir.
#[test]
fn no_test_names_a_fixed_directory_under_the_system_temp_dir() {
    let mut fixed: Vec<String> = Vec::new();
    for path in rust_sources() {
        let rel = relative(&path);
        if OUTSIDE_THE_CONTRACT.contains(&rel.as_str()) {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("a source file is readable");
        for (line, snippet) in fixed_temp_paths(&text) {
            fixed.push(format!("{rel}:{line}: {snippet}"));
        }
    }
    assert!(
        fixed.is_empty(),
        "a fixed name under the system temp dir collides between two cargo test \
         runs on one Mac; use tempfile::tempdir(): {fixed:?}"
    );
}

/// GATE 3b. The allowlist is a ratchet, not a rug: an entry whose file has been
/// deleted or moved is dead weight that would silently hide the next violation
/// in a path that no longer exists.
#[test]
fn the_outside_the_contract_allowlist_only_names_files_that_still_exist() {
    let present: BTreeSet<String> = rust_sources().iter().map(|p| relative(p)).collect();
    let dead: Vec<&str> = OUTSIDE_THE_CONTRACT
        .iter()
        .copied()
        .filter(|entry| !present.contains(*entry))
        .collect();
    assert!(
        dead.is_empty(),
        "these OUTSIDE_THE_CONTRACT entries no longer name a file, so they can \
         only be hiding nothing; delete them: {dead:?}"
    );

    // The exact debt, printed rather than implied, so the list is a number
    // somebody can watch fall instead of a file name that reads like a
    // permanent exemption. Granularity is per FILE, which is the seam: a new
    // unguarded test added to one of these files is covered by its file's
    // entry, not tracked individually.
    let mut debt: Vec<String> = Vec::new();
    for entry in OUTSIDE_THE_CONTRACT {
        let path = crate_root().join(entry);
        let text = std::fs::read_to_string(&path).expect("an allowlisted file is readable");
        let hazards = test_bodies(&text)
            .iter()
            .filter(|body| hazard_in(&body.code).is_some())
            .count();
        let fixed = fixed_temp_paths(&text).len();
        if hazards + fixed > 0 {
            debt.push(format!("{entry}: {hazards} hazard, {fixed} fixed-temp"));
        }
    }
    println!("sandbox-contract debt outside this change: {debt:?}");
    assert_eq!(
        debt.len(),
        OUTSIDE_THE_CONTRACT.len(),
        "an allowlisted file with nothing to allowlist has been fixed; delete its \
         entry so the next unguarded test in it is caught"
    );
}

// --- calibration -------------------------------------------------------------
// A gate that has only ever seen a passing tree is not evidence. Each detector
// is run against a snippet that SHOULD trip it.

#[test]
fn the_env_var_detector_finds_a_literal_read_and_an_env_named_const() {
    let text = r#"
        let a = std::env::var("EVAL_OUTPUT_DIR");
        let b = std::env::var_os(DIR_ENV);
        pub const OVERSIZE_OVERRIDE_ENV: &str = "EVAL_ALLOW_OVERSIZE";
        std::env::set_var(OVERSIZE_OVERRIDE_ENV, "1");
        let c = "PARSE";
        pub const FAIL_PARSE: &str = "PARSE";
    "#;
    assert_eq!(
        env_vars_read_by(text),
        vec!["EVAL_OUTPUT_DIR", "EVAL_ALLOW_OVERSIZE"],
        "a literal read and an *_ENV seam are found; a var-shaped call site with \\
         no literal and an uppercase CONSTANT are not variables"
    );
}

#[test]
fn the_hazard_detector_names_the_hazard_it_found() {
    let body = "#[test]\nfn t() {\n    let c = RunnerConfig { record_signals: true };\n}\n";
    assert_eq!(hazard_in(body), Some("record_signals: true"));
    let literal = "#[test]\nfn t() {\n    let c = ZtoolsConfig { ..Default::default() };\n}\n";
    assert_eq!(
        hazard_in(literal),
        Some("ZtoolsConfig {"),
        "the struct-update spelling of a default config is the same hazard"
    );
    let unrelated = "#[test]\nfn t() {\n    let c = RunnerConfig { ..Default::default() };\n}\n";
    assert_eq!(
        hazard_in(unrelated),
        None,
        "a config with no ~/ default is not this hazard; matching the syntax \
         would flag tests that cannot touch anything"
    );
    let safe = "#[test]\nfn t() {\n    let _env = TestEnv::new();\n}\n";
    assert_eq!(hazard_in(safe), None);
}

#[test]
fn the_test_body_splitter_stops_at_the_end_of_its_own_function() {
    // Two tests, the second of which is a hazard: attributing it to the first
    // would either hide it or blame the wrong function.
    let text = "
        #[test]
        fn safe_one() {
            let a = 1;
        }
        #[test]
        fn hazard_two() {
            let c = ZtoolsConfig::default();
        }
    ";
    let bodies = test_bodies(text);
    assert_eq!(bodies.len(), 2);
    assert!(bodies[0].source.contains("safe_one"));
    assert!(hazard_in(&bodies[0].code).is_none());
    assert!(hazard_in(&bodies[1].code).is_some());
}

/// The regression for the defect the naive splitter had, and the reason
/// [`blank_out_literals`] exists: a brace inside a raw string is not structure,
/// and one test's fixture used to decide where the NEXT test's body ended.
/// Before the fix this saw ONE body and attributed the second test's hazard to
/// the first -- both a miss and a false blame, from the same three lines.
///
/// The hazard string sits in a comment here rather than in code, which is the
/// other half: a detector must not fire on prose, or every file documenting
/// what it looks for would go red.
#[test]
fn a_brace_inside_a_literal_does_not_move_the_next_tests_end() {
    let text = r##"
        #[test]
        fn holds_a_json_fixture() {
            let raw = r#"{"choices": [{"#;
            assert!(raw.contains("choices"));
        }
        #[test]
        fn later_and_guarded() {
            let _env = TestEnv::new();
            // ZtoolsConfig::default() named in a COMMENT is not a hazard.
            let c = cfg_default();
            let _ = c;
        }
        #[test]
        fn later_and_unguarded() {
            let c = ZtoolsConfig::default();
            let _ = c;
        }
    "##;
    let bodies = test_bodies(text);
    assert_eq!(
        bodies.len(),
        3,
        "each `#[test]` is its own body; a brace inside a string must not close one"
    );
    assert!(bodies[0].source.contains("holds_a_json_fixture"));
    assert!(
        hazard_in(&bodies[1].code).is_none(),
        "the guarded body is not a finding, and the hazard it names in a comment is not a hazard"
    );
    assert_eq!(
        hazard_in(&bodies[2].code),
        Some("ZtoolsConfig::default()"),
        "the third body owns its own hazard rather than the second's"
    );
    // The literal really WAS unbalanced: this is the input the fix exists for,
    // so an input that balanced would prove nothing about the brace counting.
    assert_ne!(
        bodies[0].source.matches('{').count(),
        bodies[0].source.matches('}').count(),
        "the fixture must contain an unbalanced brace, or this proves nothing"
    );
}

/// `blank_out_literals` keeps every line, because two forms of one body are
/// sliced out of it by line number. If it dropped or joined a line, the source
/// and the code would stop describing the same test.
#[test]
fn blanking_literals_preserves_every_line_and_only_blanks() {
    let text = "let a = 1; // note\n/* block\n still */ let b = \"x{y\";\nlet c = '}';\nlet d = r#\"{\"q\": 1}\"#;\n";
    let blanked = blank_out_literals(text);
    assert_eq!(
        blanked.lines().count(),
        text.lines().count(),
        "line numbering must survive, or the two body forms are not the same test"
    );
    assert!(blanked.contains("let a = 1;"));
    assert!(blanked.contains("let c ="));
    assert!(
        !blanked.contains('{') && !blanked.contains('}'),
        "every brace here is inside a comment or a literal, and every one of them \
         must be gone: {blanked}"
    );
    assert!(
        text.matches('{').count() > 0,
        "the input must contain braces"
    );
}

#[test]
fn the_fixed_temp_path_detector_spots_a_literal_and_spares_a_pid_suffix() {
    let fixed = "    let d = std::env::temp_dir().join(\"ztools_test_images\");";
    assert_eq!(fixed_temp_paths(fixed).len(), 1);
    let pid = "    let d = std::env::temp_dir().join(format!(\"t_{}\", std::process::id()));";
    assert_eq!(
        fixed_temp_paths(pid).len(),
        0,
        "a pid suffix is the existing workaround; tempfile is the fix, and this \
         detector must not pretend the workaround does not exist yet"
    );
    let unrelated = "    let d = env::temp_dir()";
    assert_eq!(fixed_temp_paths(unrelated).len(), 0, "{unrelated}");
}
