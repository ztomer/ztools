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

/// Both retention toggles exist, default on, and round-trip through the file;
/// an unknown key is refused rather than stored.
#[test]
fn settings_default_on_round_trip_and_refuse_unknown_keys() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.toml");
    for key in [TWITTER_RETENTION, WEEKEND_RETENTION] {
        assert!(get_at(&path, key).unwrap(), "{key} defaults on");
    }
    set_at(&path, WEEKEND_RETENTION, false).unwrap();
    assert!(!get_at(&path, WEEKEND_RETENTION).unwrap());
    assert!(get_at(&path, TWITTER_RETENTION).unwrap());
    assert!(set_at(&path, "weekend.rolling_7_day", false).is_err());

    let listed = list_json_at(&path).unwrap();
    let values: Vec<(String, bool)> = listed
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            (
                s["key"].as_str().unwrap().to_string(),
                s["value"].as_bool().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        values,
        vec![
            (TWITTER_RETENTION.to_string(), true),
            (WEEKEND_RETENTION.to_string(), false)
        ]
    );
}
