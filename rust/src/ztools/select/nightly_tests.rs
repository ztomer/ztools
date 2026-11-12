//! The night over a fake machine, and the live ledger on a temporary one.

use super::live::{LiveNight, delete_staged, first_nominee, read_ledgers};
use super::nightly::{Night, Nominee, Outcome, REPEATS, Staged, findings_line, run};
use super::rule::Rule;
use crate::ztools::eval::report::HistoryEntry;
use crate::ztools::eval::task_fingerprint::TaskIdentities;
use crate::ztools::scout::rules::{Check, File, GIB};
use std::collections::BTreeMap;

const CAND: &str = "org/Cand-1B-4bit";

#[derive(Default)]
struct Fake {
    free: Option<u64>,
    staged: Vec<Staged>,
    nominee: Option<Nominee>,
    files: Vec<File>,
    /// Model to its score on every task.
    scores: BTreeMap<String, i64>,
    clock: f64,
    rows: BTreeMap<String, Vec<HistoryEntry>>,
    evals: Vec<String>,
    downloads: usize,
    deleted: Vec<String>,
}

fn current() -> TaskIdentities {
    ["file_summary", "taxes_qa", "refusal_war_en", "summarize"]
        .iter()
        .map(|t| ((*t).to_string(), format!("fp-{t}")))
        .collect()
}

impl Night for Fake {
    fn now(&self) -> f64 {
        self.clock
    }
    fn date(&self) -> String {
        "2026-11-13".into()
    }
    fn free_bytes(&self) -> Option<u64> {
        self.free
    }
    fn staged(&self) -> Vec<Staged> {
        self.staged.clone()
    }
    fn nominate(&mut self) -> Result<Option<Nominee>, String> {
        Ok(self.nominee.clone())
    }
    fn files(&mut self, _repo: &str) -> Result<Vec<File>, String> {
        Ok(self.files.clone())
    }
    fn download(&mut self, _s: &Staged, _f: &[File]) -> Result<(), String> {
        self.downloads += 1;
        Ok(())
    }
    fn eval(&mut self, model: &str, tasks: &[String]) -> Result<(), String> {
        self.clock += 10.0;
        self.evals.push(model.to_string());
        let score = *self.scores.get(model).ok_or("no such model")?;
        let rows = self.rows.entry(model.to_string()).or_default();
        for t in tasks {
            rows.push(HistoryEntry {
                date: "2026-11-13".into(),
                timestamp: self.clock,
                task: t.clone(),
                score: if t.starts_with("refusal") { 100 } else { score },
                time: Some(if model == "cand-1b-4bit" { 5.0 } else { 9.0 }),
                complete: true,
                fingerprint: Some(format!("fp-{t}")),
            });
        }
        Ok(())
    }
    fn history(&self) -> BTreeMap<String, Vec<HistoryEntry>> {
        self.rows.clone()
    }
    fn current(&self) -> TaskIdentities {
        current()
    }
    fn slots(&self) -> BTreeMap<String, String> {
        std::iter::once(("think".to_string(), "inc".to_string())).collect()
    }
    fn delete(&mut self, s: &Staged) -> Result<(), String> {
        self.deleted.push(s.repo.clone());
        Ok(())
    }
}

fn fake(cand_score: i64, free_gib: u64) -> Fake {
    Fake {
        free: Some(free_gib * GIB),
        nominee: Some(Nominee {
            repo: CAND.into(),
            slots: vec!["think".into()],
        }),
        files: vec![File {
            path: "model.safetensors".into(),
            size: 20 * GIB,
        }],
        scores: [
            ("inc".to_string(), 80),
            ("cand-1b-4bit".to_string(), cand_score),
        ]
        .into_iter()
        .collect(),
        clock: 1000.0,
        ..Fake::default()
    }
}

#[test]
fn a_night_with_no_room_records_it_and_downloads_nothing() {
    let mut f = fake(99, 34);
    let out = run(&mut f, &Rule::DESIGN);
    assert!(
        matches!(&out, Outcome::NoRoom(w) if w.contains("under the 15.0 GiB floor")),
        "{out:?}"
    );
    assert_eq!((f.downloads, f.evals.len()), (0, 0));
    assert!(findings_line("2026-11-13", &out).starts_with("LM 2026-11-13: no room ("));
}

#[test]
fn unknown_free_space_fails_rather_than_reading_as_no_room() {
    let mut f = fake(99, 0);
    f.free = None;
    assert!(matches!(run(&mut f, &Rule::DESIGN), Outcome::Failed(_)));
    assert_eq!(f.downloads, 0);
}

#[test]
fn a_rejected_candidate_is_deleted_that_night_with_its_margin() {
    let mut f = fake(82, 186);
    let out = run(&mut f, &Rule::DESIGN);
    assert_eq!(f.downloads, 1);
    assert_eq!(
        f.evals.len(),
        2 * REPEATS,
        "3 repeats each of incumbent and candidate"
    );
    assert_eq!(f.deleted, vec![CAND.to_string()]);
    let line = findings_line("2026-11-13", &out);
    assert!(
        line.contains("think kept inc (cand cand-1b-4bit 82.0, +2.0;"),
        "{line}"
    );
    assert!(line.ends_with("cand-1b-4bit rejected, deleted"), "{line}");
}

#[test]
fn a_winner_is_proposed_and_kept() {
    let mut f = fake(84, 186);
    let out = run(&mut f, &Rule::DESIGN);
    assert_eq!(f.deleted, Vec::<String>::new());
    let line = findings_line("2026-11-13", &out);
    assert!(line.contains("think -> cand-1b-4bit proposed"), "{line}");
}

#[test]
fn a_staged_candidate_is_evaluated_before_any_new_download() {
    let mut f = fake(82, 186);
    f.staged = vec![Staged {
        repo: CAND.into(),
        model: "cand-1b-4bit".into(),
        slots: vec!["think".into()],
        bytes: 1,
        date: "2026-11-12".into(),
    }];
    let _ = run(&mut f, &Rule::DESIGN);
    assert_eq!(f.downloads, 0);
    f.staged.push(f.staged[0].clone());
    assert!(matches!(run(&mut f, &Rule::DESIGN), Outcome::Failed(w) if w.contains("limit is 1")));
}

#[test]
fn nothing_nominated_and_a_failing_eval_are_said() {
    let mut f = fake(82, 186);
    f.nominee = None;
    let out = run(&mut f, &Rule::DESIGN);
    assert_eq!(
        findings_line("d", &out),
        "LM d: nothing staged (the scout nominates nothing)"
    );
    let mut f = fake(82, 186);
    f.scores.remove("inc");
    let out = run(&mut f, &Rule::DESIGN);
    assert!(findings_line("d", &out).starts_with("LM d: ✗ eval inc on think"));
    assert_eq!(f.deleted.len(), 0, "a failed night deletes nothing");
}

fn live(dir: &std::path::Path) -> LiveNight {
    LiveNight {
        hf: "http://127.0.0.1:9".into(),
        osaurus_url: "http://127.0.0.1:9".into(),
        slots: BTreeMap::new(),
        current: TaskIdentities::new(),
        models_root: dir.join("MLXModels"),
        state_dir: dir.join("state"),
        exe: "/nonexistent".into(),
    }
}

#[test]
fn only_a_ledgered_directory_is_deleted() {
    let tmp = tempfile::tempdir().unwrap();
    let night = live(tmp.path());
    let staged = Staged {
        repo: CAND.into(),
        model: "cand-1b-4bit".into(),
        slots: vec!["think".into()],
        bytes: 1,
        date: "2026-11-13".into(),
    };
    let weights = night.model_dir(CAND);
    std::fs::create_dir_all(&weights).unwrap();
    let err = delete_staged(&night, &staged).unwrap_err();
    assert!(err.contains("not staged by this job") && weights.exists());
    night.write_ledger(&staged).unwrap();
    assert_eq!(
        read_ledgers(&tmp.path().join("state/staged")),
        vec![staged.clone()]
    );
    delete_staged(&night, &staged).unwrap();
    assert!(!weights.exists() && read_ledgers(&tmp.path().join("state/staged")).is_empty());
}

#[test]
fn the_nominee_passes_every_filter_but_disk() {
    use crate::ztools::scout::{Candidate, Incumbent, Proposal};
    let cand = |repo: &str, fit: Check| Candidate {
        repo: repo.into(),
        nominated: "family".into(),
        provenance: Check::Pass(String::new()),
        format: Some(Check::Pass(String::new())),
        fit: Some(fit),
        disk: Some(Check::Fail("under the floor".into())),
    };
    let p = Proposal {
        asked: 1,
        refused: vec![],
        incumbents: vec![Incumbent {
            slots: vec!["think".into(), "vlm".into()],
            name: "inc".into(),
            repo: None,
            family: Err(String::new()),
            retired: false,
            candidates: vec![
                cand("a/Big", Check::Fail("over".into())),
                cand("a/Fits", Check::Pass(String::new())),
            ],
        }],
        stage: None,
        notes: vec![],
    };
    let n = first_nominee(&p).unwrap();
    assert_eq!((n.repo.as_str(), n.slots.len()), ("a/Fits", 2));
}

#[test]
fn the_live_night_reads_its_own_state_and_runs_the_eval_binary() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("MLXModels")).unwrap();
    let mut night = live(tmp.path());
    assert!(night.now() > 1.7e9 && night.date().len() == 10);
    assert!(night.free_bytes().is_some_and(|b| b > 0));
    assert_eq!(night.staged(), Vec::<Staged>::new());
    assert_eq!(
        (night.current(), night.slots()),
        (TaskIdentities::new(), BTreeMap::new())
    );
    night.exe = "/usr/bin/true".into();
    night.eval("m", &["a".into(), "b".into()]).unwrap();
    night.exe = "/usr/bin/false".into();
    assert!(
        night
            .eval("m", &[])
            .unwrap_err()
            .contains("model-eval exited")
    );
    night.exe = "/nonexistent/ztools".into();
    assert!(night.eval("m", &[]).is_err());
    let staged = Staged {
        repo: CAND.into(),
        model: "cand-1b-4bit".into(),
        slots: vec![],
        bytes: 0,
        date: "d".into(),
    };
    assert!(night.delete(&staged).is_err(), "no ledger, nothing deleted");
    let _ = night.history();
}

/// A server that answers every request with `body`, `n` times.
fn serve(n: usize, body: &'static str) -> String {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for stream in listener.incoming().take(n) {
            let mut s = stream.unwrap();
            let mut buf = [0u8; 4096];
            let _ = s.read(&mut buf);
            let _ = write!(
                s,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    url
}

#[test]
fn a_download_lands_whole_under_its_ledger_and_is_deletable() {
    let tmp = tempfile::tempdir().unwrap();
    let mut night = live(tmp.path());
    night.hf = serve(2, "weights");
    let staged = Staged {
        repo: CAND.into(),
        model: "cand-1b-4bit".into(),
        slots: vec!["think".into()],
        bytes: 7,
        date: "d".into(),
    };
    let files = vec![
        File {
            path: "a.safetensors".into(),
            size: 7,
        },
        File {
            path: "sub/config.json".into(),
            size: 7,
        },
    ];
    night.download(&staged, &files).unwrap();
    let dir = night.model_dir(CAND);
    assert_eq!(
        std::fs::read_to_string(dir.join("sub/config.json")).unwrap(),
        "weights"
    );
    assert_eq!(night.staged(), vec![staged.clone()]);
    night.delete(&staged).unwrap();
    assert!(!dir.exists());
    night.hf = "http://127.0.0.1:9".into();
    assert!(
        night.download(&staged, &files).is_err(),
        "a refused fetch fails the night"
    );
    assert!(night.files(CAND).is_err());
}
