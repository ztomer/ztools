use super::*;

fn at(day: u32, hour: u32) -> chrono::NaiveDateTime {
    chrono::NaiveDate::from_ymd_opt(2026, 10, day)
        .and_then(|d| d.and_hms_opt(hour, 0, 5))
        .unwrap()
}

/// The record is the header, then the corpus verbatim, under `corpus/`.
#[test]
fn the_corpus_is_written_verbatim_under_the_store() {
    let td = tempfile::tempdir().unwrap();
    let store = td.path().join("weekend_plans");
    let corpus = "- Fall Harvest Market: Oct 9 - Oct 12 in Toronto\n- [Things to Do] Pumpkinfest";
    let path = save_corpus(
        &store,
        at(10, 14),
        "# window 2026-10-09..2026-10-12",
        corpus,
    )
    .unwrap();
    assert_eq!(
        path,
        store.join("corpus").join("2026-10-10_140005_corpus.txt")
    );
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        format!("# window 2026-10-09..2026-10-12\n{corpus}\n")
    );
}

/// Bounded: the newest `CORPUS_KEEP` survive, the oldest go, and a file the
/// module did not name is never touched.
#[test]
fn only_the_newest_corpora_are_kept() {
    let td = tempfile::tempdir().unwrap();
    let store = td.path();
    std::fs::create_dir_all(corpus_dir(store)).unwrap();
    std::fs::write(corpus_dir(store).join("notes.txt"), "mine").unwrap();
    for hour in 0..12 {
        save_corpus(store, at(1, hour), "#", "-").unwrap();
    }
    let mut kept: Vec<String> = std::fs::read_dir(corpus_dir(store))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    kept.sort();
    assert_eq!(kept.len(), CORPUS_KEEP + 1, "{kept:?}");
    assert_eq!(kept[0], "2026-10-01_020005_corpus.txt", "{kept:?}");
    assert_eq!(kept[CORPUS_KEEP - 1], "2026-10-01_110005_corpus.txt");
    assert!(kept.contains(&"notes.txt".to_string()), "{kept:?}");
}

/// The plan readers still find the PLAN when a corpus is the newest thing in
/// the store: the dashboard's `--fetch-latest` reads `newest_md`.
#[test]
fn the_newest_plan_is_still_the_plan_when_a_corpus_is_newer() {
    let td = tempfile::tempdir().unwrap();
    let store = td.path();
    let fri = chrono::NaiveDate::from_ymd_opt(2026, 10, 9).unwrap();
    let mon = chrono::NaiveDate::from_ymd_opt(2026, 10, 12).unwrap();
    let plan = crate::ztools::store::save_weekend_plan(store, fri, mon, "# plan\n").unwrap();
    save_corpus(store, at(10, 23), "#", "- a later corpus").unwrap();
    assert_eq!(crate::ztools::store::newest_md(store).unwrap(), plan);
}

/// A prune that cannot delete is reported and does not fail the save: the
/// record was written, and the run must not lose it over housekeeping. A
/// DIRECTORY named like a corpus cannot be unlinked as a file on any platform.
#[test]
fn a_failed_prune_does_not_fail_the_save() {
    let td = tempfile::tempdir().unwrap();
    let store = td.path();
    std::fs::create_dir_all(corpus_dir(store).join("2000-01-01_000000_corpus.txt")).unwrap();
    for hour in 0..u32::try_from(CORPUS_KEEP).unwrap() {
        save_corpus(store, at(2, hour), "#", "-").unwrap();
    }
    assert!(
        corpus_dir(store)
            .join("2000-01-01_000000_corpus.txt")
            .is_dir()
    );
}
