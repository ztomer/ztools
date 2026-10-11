use super::*;

fn touch(dir: &Path, name: &str, age_days: u64, now: SystemTime) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, name).unwrap();
    let when = now - Duration::from_hours(age_days * 24);
    std::fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_modified(when)
        .unwrap();
    path
}

/// Older than the window goes; inside it stays; a subdirectory and a non-md
/// file are not this rule's business.
#[test]
fn prune_keeps_the_window_and_leaves_other_files_alone() {
    let dir = tempfile::tempdir().unwrap();
    let now = SystemTime::now();
    let old = touch(dir.path(), "a_summary.md", 9, now);
    let edge = touch(dir.path(), "b_summary.md", 6, now);
    let fresh = touch(dir.path(), "c_summary.md", 0, now);
    let notes = touch(dir.path(), "notes.txt", 30, now);
    std::fs::create_dir(dir.path().join("rejected")).unwrap();
    let kept_sub = touch(&dir.path().join("rejected"), "x.md", 30, now);

    let gone = prune(dir.path(), Duration::from_hours(7 * 24), now);
    assert_eq!(gone, vec![old.clone()]);
    assert!(!old.exists());
    for p in [&edge, &fresh, &notes, &kept_sub] {
        assert!(p.exists(), "{}", p.display());
    }
}

/// A store whose tool has not run for weeks keeps its newest file: the
/// dashboard shows the last plan, stale and saying so, never nothing.
#[test]
fn prune_never_empties_a_store() {
    let dir = tempfile::tempdir().unwrap();
    let now = SystemTime::now();
    touch(dir.path(), "old.md", 40, now);
    let newest = touch(dir.path(), "newer.md", 20, now);
    let gone = prune(dir.path(), Duration::from_hours(7 * 24), now);
    assert_eq!(gone.len(), 1);
    assert!(newest.exists());
}

/// 0 days keeps exactly the newest file: "keep just the last one".
#[test]
fn zero_days_keeps_only_the_newest() {
    let dir = tempfile::tempdir().unwrap();
    let now = SystemTime::now();
    let older = touch(dir.path(), "a.md", 1, now);
    let newest = touch(dir.path(), "b.md", 0, now);
    let gone = prune(dir.path(), Duration::from_hours(0), now);
    assert_eq!(gone, vec![older]);
    assert!(newest.exists());
}

/// Both day counts exist, default 7, round-trip through the file, and list as
/// a choice of the fixed options; an unknown key or count is refused.
#[test]
fn day_counts_default_round_trip_and_refuse_what_is_not_offered() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.toml");
    for key in [TWITTER_RETENTION, WEEKEND_RETENTION] {
        assert_eq!(get_at(&path, key).unwrap(), DEFAULT_KEEP_DAYS, "{key}");
    }
    set_at(&path, WEEKEND_RETENTION, 0).unwrap();
    assert_eq!(get_at(&path, WEEKEND_RETENTION).unwrap(), 0);
    assert_eq!(get_at(&path, TWITTER_RETENTION).unwrap(), 7);
    assert!(set_at(&path, "weekend.keep_day", 7).is_err());
    assert!(set_at(&path, WEEKEND_RETENTION, 5).is_err());

    let listed = list_json_at(&path).unwrap();
    let rows = listed.as_array().unwrap();
    assert_eq!(rows.len(), 2);
    for row in rows {
        assert_eq!(row["kind"], "choice");
        assert!(row["title"].as_str().unwrap().contains("{}"), "{row}");
        assert_eq!(
            row["options"],
            serde_json::json!(["0", "1", "3", "7", "14", "30"])
        );
        assert!(row.get("help").is_none(), "no explanatory text: {row}");
    }
    assert_eq!(rows[0]["value"], "7");
    assert_eq!(rows[1]["value"], "0");
}

/// The old toggles: `true` reads as the 7 days it meant; `false` meant "keep
/// everything", which no count says, so it is refused by name. Setting the
/// new key drops the old one.
#[test]
fn a_legacy_toggle_migrates_or_is_refused_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.toml");
    std::fs::write(
        &path,
        "\"weekend.rolling_7_days\" = true\n\"twitter.rolling_7_days\" = false\n",
    )
    .unwrap();
    assert_eq!(get_at(&path, WEEKEND_RETENTION).unwrap(), 7);
    let why = get_at(&path, TWITTER_RETENTION).unwrap_err().to_string();
    assert!(
        why.contains("keep everything") && why.contains("twitter.keep_days"),
        "{why}"
    );
    set_at(&path, TWITTER_RETENTION, 30).unwrap();
    assert_eq!(get_at(&path, TWITTER_RETENTION).unwrap(), 30);
    assert!(
        !std::fs::read_to_string(&path)
            .unwrap()
            .contains("twitter.rolling_7_days")
    );
}

/// The CLI against the sandboxed `$HOME`: list, get, set, and each refusal.
#[test]
fn the_cli_lists_gets_sets_and_refuses_by_name() {
    let env = crate::test_env::TestEnv::new();
    let s = |v: &[&str]| v.iter().map(|x| (*x).to_string()).collect::<Vec<_>>();
    cli(&[], false).unwrap();
    cli(&[], true).unwrap();
    cli(&s(&["set", WEEKEND_RETENTION, "3"]), false).unwrap();
    cli(&s(&["get", WEEKEND_RETENTION]), false).unwrap();
    assert_eq!(get_at(&settings_path(), WEEKEND_RETENTION).unwrap(), 3);
    let why = cli(&s(&["set", WEEKEND_RETENTION, "a week"]), false).unwrap_err();
    assert!(why.to_string().contains("not a day count"), "{why}");
    let why = cli(&s(&["frobnicate"]), false).unwrap_err();
    assert!(why.to_string().contains("usage"), "{why}");
    drop(env);
}

/// `retain` prunes to the stored count, and skips -- deleting nothing -- when
/// the setting cannot be read.
#[test]
fn retain_prunes_to_the_setting_and_skips_what_it_cannot_read() {
    let env = crate::test_env::TestEnv::new();
    let now = SystemTime::now();
    let store = tempfile::tempdir().unwrap();
    let old = touch(store.path(), "a.md", 5, now);
    let newest = touch(store.path(), "b.md", 0, now);

    std::fs::create_dir_all(settings_path().parent().unwrap()).unwrap();
    std::fs::write(settings_path(), "\"weekend.rolling_7_days\" = false\n").unwrap();
    retain(store.path(), WEEKEND_RETENTION);
    assert!(old.exists(), "an unreadable setting deletes nothing");

    set_at(&settings_path(), WEEKEND_RETENTION, 3).unwrap();
    retain(store.path(), WEEKEND_RETENTION);
    assert!(!old.exists(), "older than 3 days goes");
    assert!(newest.exists());
    drop(env);
}
