//! F1: `ztools weekend exclude <place>`, the one writer of `exclude_places`.
//!
//! Every edit is re-read through [`load_exclusions`] -- the loader the planner
//! actually runs -- and through a plain TOML parse, because the C8 trap is a list
//! that LOOKS right in the file and parses as a key of the last table.

use super::*;
use crate::config::ZtoolsConfig;
use std::path::{Path, PathBuf};

const CONFIG: &str = r#"# Weekend Planner Configuration

# NOTE: top-level keys MUST precede the first [table].
exclude_places = [
  "Toronto Zoo",
  # "Commented Out Place",
  "ROM",
]

[location]
city = "Vaughan"

[[children]]
birthday = "2015-01-01"
"#;

const PLAN: &str = "# Weekend Plan: Oct 16 to Oct 18 (Vaughan/Toronto)

### Fixed / Year-Round Activities (Ranked by Fit Score)

| Score | Activity & Location | Dates | Target Age(s) |
| :--- | :--- | :--- | :--- |
| * 3.0/5 | **Kortright Centre for Conservation** (Vaughan) | Year-Round | All Ages |
| * 3.0/5 | **Air Riderz Trampoline Park** (Vaughan) | Year-Round | 6-13 |
| * 2.5/5 | **Toronto Zoo** (Scarborough) | Year-Round | All Ages |

### Transient / Limited-Time Events (Ranked by Fit Score)

| Score | Event & Location | Dates | Day & Time |
| :--- | :--- | :--- | :--- |
| * 2.5/5 | **Legends of Horror** (Casa Loma, Toronto) | 2026-10-16 | Friday |
";

/// A sandbox: the config under `dir/weekend.toml`, the plan store under `dir/plans`.
struct Sandbox {
    _dir: tempfile::TempDir,
    config: PathBuf,
    plans: PathBuf,
}

impl Sandbox {
    fn new(config: &str) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weekend.toml");
        std::fs::write(&path, config).unwrap();
        let plans = dir.path().join("plans");
        std::fs::create_dir(&plans).unwrap();
        std::fs::write(
            plans.join("weekend_plan_October_16_to_October_18_2026.md"),
            PLAN,
        )
        .unwrap();
        Self {
            _dir: dir,
            config: path,
            plans,
        }
    }

    fn paths(&self) -> Vec<String> {
        vec![self.config.to_string_lossy().into_owned()]
    }

    fn edit(&self, action: Action, place: &str) -> Result<Outcome, EditError> {
        edit(&self.paths(), &self.plans, action, place)
    }

    fn text(&self) -> String {
        std::fs::read_to_string(&self.config).unwrap()
    }

    /// The list as the planner reads it.
    fn loaded(&self) -> Vec<String> {
        let config = ZtoolsConfig {
            weekend_exclusions_paths: self.paths(),
            ..ZtoolsConfig::default()
        };
        crate::ztools::weekend::load_exclusions(&config)
    }
}

/// `exclude_places` as a plain TOML parse sees it, at the TOP level only.
fn top_level_list(text: &str) -> Vec<String> {
    let value: toml::Value = toml::from_str(text).unwrap();
    value
        .get("exclude_places")
        .and_then(toml::Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn assert_no_nested_list(text: &str) {
    let value: toml::Value = toml::from_str(text).unwrap();
    for (key, table) in value.as_table().unwrap() {
        let nested = match table {
            toml::Value::Table(t) => t.contains_key("exclude_places"),
            toml::Value::Array(a) => a.iter().any(|v| v.get("exclude_places").is_some()),
            _ => false,
        };
        assert!(!nested, "exclude_places landed under [{key}] (C8):\n{text}");
    }
}

#[test]
fn exclude_appends_a_plan_place_to_the_top_level_list() {
    let s = Sandbox::new(CONFIG);
    let out = s
        .edit(Action::Exclude, "Kortright Centre for Conservation")
        .unwrap();
    assert_eq!(out.change, Change::Added);
    assert_eq!(out.entry, "Kortright Centre for Conservation");
    assert_eq!(out.entries, 3);
    let text = s.text();
    assert_eq!(
        top_level_list(&text),
        ["Toronto Zoo", "ROM", "Kortright Centre for Conservation"]
    );
    assert_no_nested_list(&text);
    assert_eq!(
        s.loaded(),
        ["Toronto Zoo", "ROM", "Kortright Centre for Conservation"]
    );
    // Comments survive, the trailing table is untouched.
    assert!(text.contains("# NOTE: top-level keys MUST precede the first [table]."));
    assert!(text.contains("# \"Commented Out Place\","));
    assert!(text.ends_with("[[children]]\nbirthday = \"2015-01-01\"\n"));
}

#[test]
fn the_plan_spelling_is_what_is_stored() {
    let s = Sandbox::new(CONFIG);
    let out = s
        .edit(Action::Exclude, "  air riderz TRAMPOLINE park ")
        .unwrap();
    assert_eq!(out.entry, "Air Riderz Trampoline Park");
    assert!(
        s.loaded()
            .contains(&"Air Riderz Trampoline Park".to_string())
    );
}

#[test]
fn excluding_twice_is_a_no_op() {
    let s = Sandbox::new(CONFIG);
    s.edit(Action::Exclude, "Legends of Horror").unwrap();
    let before = s.text();
    let again = s.edit(Action::Exclude, "Legends of Horror").unwrap();
    assert_eq!(again.change, Change::Unchanged);
    assert_eq!(s.text(), before, "a repeat must not rewrite the file");
    // Already listed counts as known even when today's plan no longer has it.
    let listed = s.edit(Action::Exclude, "rom").unwrap();
    assert_eq!(listed.change, Change::Unchanged);
    assert_eq!(listed.entry, "ROM");
}

#[test]
fn undo_removes_and_repeat_undo_is_a_no_op() {
    let s = Sandbox::new(CONFIG);
    s.edit(Action::Exclude, "Legends of Horror").unwrap();
    let out = s.edit(Action::Include, "legends of horror").unwrap();
    assert_eq!(out.change, Change::Removed);
    assert_eq!(out.entry, "Legends of Horror");
    assert_eq!(s.loaded(), ["Toronto Zoo", "ROM"]);
    let before = s.text();
    let again = s.edit(Action::Include, "Legends of Horror").unwrap();
    assert_eq!(again.change, Change::Unchanged);
    assert_eq!(s.text(), before);
    assert!(before.contains("# \"Commented Out Place\","));
}

#[test]
fn an_unknown_place_is_refused_with_a_reason() {
    let s = Sandbox::new(CONFIG);
    let before = s.text();
    let Err(EditError::Refused(reason)) = s.edit(Action::Exclude, "Row 3") else {
        panic!("an unknown place must be refused");
    };
    assert!(reason.contains("Row 3"), "{reason}");
    assert!(reason.contains("weekend_plan_October_16"), "{reason}");
    assert_eq!(s.text(), before);
}

#[test]
fn a_malformed_place_is_refused() {
    let s = Sandbox::new(CONFIG);
    for bad in ["", "   ", "Toronto Zoo\n[evil]", "a\u{7}b"] {
        assert!(
            matches!(s.edit(Action::Exclude, bad), Err(EditError::Refused(_))),
            "{bad:?}"
        );
    }
}

#[test]
fn no_stored_plan_means_nothing_to_validate_against() {
    let s = Sandbox::new(CONFIG);
    std::fs::remove_dir_all(&s.plans).unwrap();
    let Err(EditError::Refused(reason)) = s.edit(Action::Exclude, "Toronto Zoo North") else {
        panic!("must refuse");
    };
    assert!(reason.contains("no stored plan"), "{reason}");
}

#[test]
fn concurrent_writers_lose_no_entry() {
    let s = Sandbox::new(CONFIG);
    let places = [
        "Kortright Centre for Conservation",
        "Air Riderz Trampoline Park",
        "Legends of Horror",
    ];
    // Many writers per place, all at once: every distinct place must land,
    // exactly once, whatever the interleaving.
    std::thread::scope(|scope| {
        for round in 0..8 {
            for place in places {
                let (paths, plans) = (s.paths(), s.plans.clone());
                scope.spawn(move || {
                    let _ = round;
                    edit(&paths, &plans, Action::Exclude, place).unwrap();
                });
            }
        }
    });
    let mut got = s.loaded();
    got.sort();
    let mut want: Vec<String> = ["Toronto Zoo", "ROM"]
        .iter()
        .chain(places.iter())
        .map(ToString::to_string)
        .collect();
    want.sort();
    assert_eq!(got, want);
    assert_no_nested_list(&s.text());
}

#[test]
fn a_file_without_a_list_gets_one_seeded_with_what_the_planner_used() {
    let s = Sandbox::new("[location]\ncity = \"Vaughan\"\n");
    let before = s.loaded(); // the built-in fallback list
    s.edit(Action::Exclude, "Legends of Horror").unwrap();
    let text = s.text();
    assert_no_nested_list(&text);
    let mut want = before;
    want.push("Legends of Horror".to_string());
    assert_eq!(top_level_list(&text), want);
    assert!(text.starts_with("exclude_places"), "{text}");
    assert!(text.contains("[location]\ncity = \"Vaughan\"\n"));
}

#[test]
fn a_list_misplaced_under_a_table_is_refused_not_duplicated() {
    let s = Sandbox::new("[location]\ncity = \"Vaughan\"\nexclude_places = [\"ROM\"]\n");
    let Err(EditError::Refused(reason)) = s.edit(Action::Exclude, "Legends of Horror") else {
        panic!("must refuse a C8 file");
    };
    assert!(reason.contains("[location]"), "{reason}");
}

#[test]
fn the_loader_reads_toml_not_lines() {
    // The old line scanner read a commented-out entry as an exclusion and a
    // list under a table as top-level; the planner now reads what TOML reads.
    let s = Sandbox::new(CONFIG);
    assert_eq!(s.loaded(), ["Toronto Zoo", "ROM"]);
    let nested = Sandbox::new("[location]\nexclude_places = [\"Kortright\"]\n");
    assert!(!nested.loaded().contains(&"Kortright".to_string()));
    let inline = Sandbox::new("exclude_places = [\"A\", \"B\"]\n");
    assert_eq!(inline.loaded(), ["A", "B"]);
}

#[test]
fn the_written_file_keeps_its_permissions() {
    use std::os::unix::fs::PermissionsExt;
    let s = Sandbox::new(CONFIG);
    std::fs::set_permissions(&s.config, std::fs::Permissions::from_mode(0o644)).unwrap();
    s.edit(Action::Exclude, "Legends of Horror").unwrap();
    let mode = std::fs::metadata(&s.config).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o644);
}

#[test]
fn plan_places_are_the_bold_names() {
    assert_eq!(
        plan_places(PLAN),
        [
            "Kortright Centre for Conservation",
            "Air Riderz Trampoline Park",
            "Toronto Zoo",
            "Legends of Horror"
        ]
    );
}

#[test]
fn the_result_contract() {
    let s = Sandbox::new(CONFIG);
    let ok = s.edit(Action::Exclude, "Legends of Horror");
    let (json, code) = report(Action::Exclude, "Legends of Horror", &ok);
    assert_eq!(code, 0);
    assert_eq!(json["schema"], 1);
    assert_eq!(json["action"], "exclude");
    assert_eq!(json["result"], "added");
    assert_eq!(json["entry"], "Legends of Horror");
    assert_eq!(json["entries"], 3);
    let canonical = std::fs::canonicalize(&s.config).unwrap();
    assert_eq!(json["file"], canonical.to_string_lossy().as_ref());
    assert!(json["reason"].is_null());

    let again = s.edit(Action::Include, "Legends of Horror");
    let (json, code) = report(Action::Include, "Legends of Horror", &again);
    assert_eq!((json["result"].as_str(), code), (Some("removed"), 0));

    let refused = s.edit(Action::Exclude, "Nowhere");
    let (json, code) = report(Action::Exclude, "Nowhere", &refused);
    assert_eq!((json["result"].as_str(), code), (Some("refused"), 3));
    assert!(json["reason"].as_str().unwrap().contains("Nowhere"));
    assert!(json["entry"].is_null());

    let failed: Result<Outcome, EditError> = Err(EditError::Failed("disk".into()));
    let (json, code) = report(Action::Exclude, "x", &failed);
    assert_eq!((json["result"].as_str(), code), (Some("failed"), 1));
}

#[test]
fn no_config_file_at_all_is_a_failure_not_a_new_file() {
    let s = Sandbox::new(CONFIG);
    let missing: Vec<String> = vec![s.plans.join("absent.toml").to_string_lossy().into_owned()];
    assert!(matches!(
        edit(&missing, &s.plans, Action::Exclude, "Legends of Horror"),
        Err(EditError::Failed(_))
    ));
    assert!(!Path::new(&missing[0]).exists());
}
