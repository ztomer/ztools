//! `ztools weekend exclude` through the real binary: the contract the `routines`
//! row button parses -- one JSON object on stdout and an exit status that says
//! which kind of answer it is (`docs/ROADMAP.md`, "Contracts other repos build on").

mod support;

use std::fs;
use std::path::Path;
// Re-exported, as the other suites do, so the helpers this one does not call are
// not dead code in this test binary.
pub use support::*;

const PLAN: &str = "### Fixed / Year-Round Activities (Ranked by Fit Score)

| Score | Activity & Location | Dates |
| :--- | :--- | :--- |
| * 3.0/5 | **Kortright Centre for Conservation** (Vaughan) | Year-Round |
";

/// A sandbox whose config names its own weekend.toml and whose plan store holds
/// one plan. Returns the weekend.toml path.
fn sandbox(home: &Path) -> std::path::PathBuf {
    let weekend = home.join("weekend.toml");
    fs::write(
        &weekend,
        "# keep me\nexclude_places = [\n  \"ROM\",\n]\n\n[[children]]\nbirthday = \"2015-01-01\"\n",
    )
    .unwrap();
    write_config(
        home,
        &format!("weekend_exclusions_paths = [\"{}\"]\n", weekend.display()),
    );
    let plans = home.join("plans");
    fs::create_dir_all(&plans).unwrap();
    fs::write(
        plans.join("weekend_plan_October_16_to_October_18_2026.md"),
        PLAN,
    )
    .unwrap();
    weekend
}

fn run(home: &Path, args: &[&str]) -> (serde_json::Value, i32) {
    let out = ztool(home)
        .env("WEEKEND_OUTPUT_DIR", home.join("plans"))
        .arg("weekend")
        .arg("exclude")
        .args(args)
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let json = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout is not one JSON object ({e}): {stdout:?}"));
    (json, out.status.code().unwrap())
}

#[test]
fn exclude_repeat_undo_and_refuse_through_the_binary() {
    let home = fresh("weekend-exclude");
    let weekend = sandbox(&home);
    let place = "Kortright Centre for Conservation";

    let (json, code) = run(&home, &[place]);
    assert_eq!(
        (json["result"].as_str(), code),
        (Some("added"), 0),
        "{json}"
    );
    assert_eq!(json["entry"], place);
    assert_eq!(json["entries"], 2);
    let text = fs::read_to_string(&weekend).unwrap();
    assert!(text.starts_with("# keep me\nexclude_places = [\n  \"ROM\",\n  \"Kortright"));

    let (json, code) = run(&home, &[place]);
    assert_eq!(
        (json["result"].as_str(), code),
        (Some("unchanged"), 0),
        "{json}"
    );
    assert_eq!(fs::read_to_string(&weekend).unwrap(), text);

    let (json, code) = run(&home, &[place, "--undo"]);
    assert_eq!(
        (json["result"].as_str(), code),
        (Some("removed"), 0),
        "{json}"
    );
    assert_eq!(json["action"], "include");

    let (json, code) = run(&home, &["Row 3"]);
    assert_eq!(
        (json["result"].as_str(), code),
        (Some("refused"), 3),
        "{json}"
    );
    assert!(json["reason"].as_str().unwrap().contains("Row 3"));

    let _ = fs::remove_dir_all(&home);
}

#[test]
fn a_missing_config_file_fails_with_status_1() {
    let home = fresh("weekend-exclude-missing");
    sandbox(&home);
    write_config(
        &home,
        &format!(
            "weekend_exclusions_paths = [\"{}\"]\n",
            home.join("absent.toml").display()
        ),
    );
    let (json, code) = run(&home, &["Kortright Centre for Conservation"]);
    assert_eq!(
        (json["result"].as_str(), code),
        (Some("failed"), 1),
        "{json}"
    );
    let _ = fs::remove_dir_all(&home);
}
