//! The miss → install → use path, against a fake installer seam.

use super::*;
use std::cell::RefCell;
use std::collections::HashSet;

/// A machine where `present` names what the probe finds, and installing a
/// row's command makes that row present (unless the command is in `broken`,
/// or `inert`, which exits 0 and leaves nothing behind).
#[derive(Default)]
struct FakeHost {
    present: RefCell<HashSet<String>>,
    broken: HashSet<String>,
    inert: HashSet<String>,
    installs: RefCell<Vec<String>>,
    said: RefCell<Vec<String>>,
}

impl FakeHost {
    fn with(present: &[&str]) -> Self {
        let host = Self::default();
        for p in present {
            host.present.borrow_mut().insert((*p).to_owned());
        }
        host
    }
}

impl Host for FakeHost {
    fn probe(&self, dep: &Dependency) -> Option<String> {
        self.present
            .borrow()
            .contains(dep.name)
            .then(|| format!("/fake/{}", dep.name))
    }

    fn install(&self, argv: &[&str]) -> Result<(), String> {
        let line = argv.join(" ");
        self.installs.borrow_mut().push(line.clone());
        if self.broken.contains(&line) {
            return Err("exit status 1".to_owned());
        }
        if !self.inert.contains(&line) {
            let name = argv.last().copied().unwrap_or_default();
            self.present.borrow_mut().insert(name.to_owned());
        }
        Ok(())
    }

    fn say(&self, line: &str) {
        self.said.borrow_mut().push(line.to_owned());
    }
}

const TABLE: &[Dependency] = &[
    Dependency {
        name: "python3",
        pin: "3.15",
        needed_for: "the installer",
        probe: Probe::Command("python3"),
        install: &[],
        requires: &[],
        manual: "install Python 3 with Homebrew",
    },
    Dependency {
        name: "pkg",
        pin: "1.0",
        needed_for: "the browser fetch",
        probe: Probe::PythonModule("pkg"),
        install: &["pip", "install", "pkg"],
        requires: &["python3"],
        manual: "",
    },
    Dependency {
        name: "browser",
        pin: "2.0",
        needed_for: "the collector",
        probe: Probe::Located("browser"),
        install: &["fetch", "browser"],
        requires: &["pkg"],
        manual: "",
    },
];

#[test]
fn a_present_dependency_is_used_and_nothing_is_installed() {
    let host = FakeHost::with(&["python3", "pkg", "browser"]);
    assert_eq!(
        ensure(TABLE, "browser", &host),
        Health::Present("/fake/browser".to_owned())
    );
    assert_eq!(*host.installs.borrow(), Vec::<String>::new());
    assert_eq!(
        *host.said.borrow(),
        Vec::<String>::new(),
        "a present row says nothing"
    );
}

#[test]
fn a_miss_installs_its_prerequisites_then_itself_and_is_used() {
    let host = FakeHost::with(&["python3"]);
    let health = ensure(TABLE, "browser", &host);
    assert_eq!(health, Health::Installed("/fake/browser".to_owned()));
    assert_eq!(health.location(), Some("/fake/browser"));
    assert_eq!(
        *host.installs.borrow(),
        vec!["pip install pkg".to_owned(), "fetch browser".to_owned()],
        "prerequisite first, then the row"
    );
    let said = host.said.borrow();
    assert!(
        said.iter()
            .any(|l| l.starts_with("→ browser 2.0 is missing")
                && l.contains("needed for the collector")
                && l.contains("fetch browser")),
        "the install is announced with its reason and command: {said:?}"
    );
    assert!(
        said.iter()
            .any(|l| l.starts_with("✓ installed browser 2.0"))
    );
}

#[test]
fn a_failed_install_degrades_with_the_reason() {
    let mut host = FakeHost::with(&["python3", "pkg"]);
    host.broken.insert("fetch browser".to_owned());
    let health = ensure(TABLE, "browser", &host);
    let Health::Unavailable(why) = &health else {
        panic!("expected unavailable, got {health:?}");
    };
    assert!(
        why.contains("could not be installed (exit status 1)"),
        "{why}"
    );
    assert!(why.contains("the collector is degraded"), "{why}");
    assert_eq!(health.location(), None);
    assert!(host.said.borrow().iter().any(|l| l.starts_with("⚠ ")));
}

#[test]
fn an_install_that_leaves_nothing_behind_is_not_a_success() {
    let mut host = FakeHost::with(&["python3", "pkg"]);
    host.inert.insert("fetch browser".to_owned());
    let health = ensure(TABLE, "browser", &host);
    assert!(
        matches!(&health, Health::Unavailable(w) if w.contains("probe still cannot find it")),
        "{health:?}"
    );
}

#[test]
fn an_uninstallable_prerequisite_stops_the_chain_with_its_reason() {
    let host = FakeHost::with(&[]);
    let health = ensure(TABLE, "browser", &host);
    let Health::Unavailable(why) = &health else {
        panic!("{health:?}");
    };
    assert!(why.contains("it needs pkg"), "{why}");
    assert!(why.contains("install Python 3 with Homebrew"), "{why}");
    assert_eq!(
        *host.installs.borrow(),
        Vec::<String>::new(),
        "nothing runs past a dead prerequisite"
    );
}

#[test]
fn an_unknown_name_and_a_cycle_are_reasons_not_panics() {
    const CYCLE: &[Dependency] = &[
        Dependency {
            name: "a",
            pin: "1",
            needed_for: "x",
            probe: Probe::Command("a"),
            install: &["i", "a"],
            requires: &["b"],
            manual: "",
        },
        Dependency {
            name: "b",
            pin: "1",
            needed_for: "x",
            probe: Probe::Command("b"),
            install: &["i", "b"],
            requires: &["a"],
            manual: "",
        },
    ];
    let host = FakeHost::with(&[]);
    assert!(matches!(ensure(TABLE, "nope", &host), Health::Unavailable(w) if w.contains("not in")));
    assert!(matches!(ensure(CYCLE, "a", &host), Health::Unavailable(w) if w.contains("cycle")));
}

#[test]
fn ensure_all_reports_every_row_and_render_shows_state_and_pin() {
    let host = FakeHost::with(&["python3"]);
    let rows = ensure_all(TABLE, &host);
    assert_eq!(rows.len(), TABLE.len());
    let text = render(&rows);
    assert!(text.contains("python3"), "{text}");
    assert!(text.contains("✓ present"), "{text}");
    assert!(text.contains("✓ installed"), "{text}");
    assert!(text.contains("pin 2.0"), "{text}");
    let none = FakeHost::with(&[]);
    assert!(render(&ensure_all(TABLE, &none)).contains("⚠ unavailable"));
}

#[test]
fn the_switch_is_off_only_when_said_so() {
    assert!(!switched_off(None));
    assert!(!switched_off(Some("1")));
    assert!(!switched_off(Some("")));
    for off in ["0", "off", "OFF", " no ", "false"] {
        assert!(switched_off(Some(off)), "{off}");
    }
}
