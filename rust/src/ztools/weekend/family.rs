//! The family the plan is for, read from `conf/weekend.toml [[children]]`.
//!
//! The ages used to be a CLI default, `--ages "13,10,6"`, while the config's
//! birthdays — the only thing that knows when a child turns a year older —
//! were never read. The default was right once and then aged: on 2026-10-09
//! the children were 14, 11 and 6. Ages are now DERIVED from the birthdays on
//! the plan's first day, so there is no second copy to fall out of date.

use chrono::{Datelike, NaiveDate};

/// Whole years from `birthday` to `on`, or `None` when `on` is before the
/// birthday (a child not yet born is a config error, not an age of zero).
#[must_use]
pub fn age_on(birthday: NaiveDate, on: NaiveDate) -> Option<u32> {
    if on < birthday {
        return None;
    }
    let mut years = on.year() - birthday.year();
    if (on.month(), on.day()) < (birthday.month(), birthday.day()) {
        years -= 1;
    }
    u32::try_from(years).ok()
}

/// The children's birthdays from the first config that has `[[children]]`.
///
/// # Errors
///
/// When no candidate file has a `[[children]]` array, or an entry's
/// `birthday` is missing or not `YYYY-MM-DD`. Each is stated: a plan with no
/// family ages would score every event as fitting nobody, or everybody.
pub fn load_birthdays(paths: &[String]) -> Result<Vec<NaiveDate>, String> {
    for raw in paths {
        let path = crate::manifest::expand_tilde(raw);
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(val) = toml::from_str::<toml::Value>(&content) else {
            continue;
        };
        let Some(children) = val.get("children").and_then(toml::Value::as_array) else {
            continue;
        };
        if children.is_empty() {
            continue;
        }
        return children
            .iter()
            .enumerate()
            .map(|(i, child)| {
                let raw = child
                    .get("birthday")
                    .and_then(toml::Value::as_str)
                    .ok_or_else(|| {
                        format!(
                            "{}: [[children]] #{} has no birthday",
                            path.display(),
                            i + 1
                        )
                    })?;
                NaiveDate::parse_from_str(raw.trim(), "%Y-%m-%d").map_err(|e| {
                    format!(
                        "{}: [[children]] #{} birthday {raw:?} is not YYYY-MM-DD ({e})",
                        path.display(),
                        i + 1
                    )
                })
            })
            .collect();
    }
    Err(format!(
        "no weekend config with [[children]] birthdays among {paths:?}; the plan's \
         target ages come only from there"
    ))
}

/// Each child's age on `on`, oldest first.
///
/// # Errors
///
/// When the birthdays cannot be loaded, or one is after `on`.
pub fn family_ages(paths: &[String], on: NaiveDate) -> Result<Vec<u32>, String> {
    let mut ages = load_birthdays(paths)?
        .into_iter()
        .map(|b| age_on(b, on).ok_or_else(|| format!("birthday {b} is after the plan date {on}")))
        .collect::<Result<Vec<u32>, String>>()?;
    ages.sort_unstable_by(|a, b| b.cmp(a));
    Ok(ages)
}

/// The ages as the plan prints them and the prompts carry them: `14,11,6`.
#[must_use]
pub fn ages_label(ages: &[u32]) -> String {
    ages.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn age_turns_over_on_the_birthday_not_before() {
        let b = d(2012, 10, 5);
        assert_eq!(age_on(b, d(2026, 10, 4)), Some(13));
        assert_eq!(age_on(b, d(2026, 10, 5)), Some(14));
        assert_eq!(age_on(b, d(2026, 10, 9)), Some(14));
        // A leap-day birthday is a year older on March 1 of a common year.
        assert_eq!(age_on(d(2016, 2, 29), d(2027, 2, 28)), Some(10));
        assert_eq!(age_on(d(2016, 2, 29), d(2027, 3, 1)), Some(11));
        assert_eq!(age_on(b, d(2012, 10, 4)), None);
    }

    /// The shipped config, on the Thanksgiving 2026 plan's first day, is the
    /// family the scheduled plans were wrong about: 14, 11 and 6, not the
    /// hardcoded 13, 10 and 6.
    #[test]
    fn the_shipped_birthdays_give_the_real_ages_on_the_plan_date() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("conf/weekend.toml");
        let paths = [path.to_string_lossy().into_owned()];
        assert_eq!(family_ages(&paths, d(2026, 10, 9)), Ok(vec![14, 11, 6]));
        assert_eq!(ages_label(&[14, 11, 6]), "14,11,6");
    }

    #[test]
    fn every_config_failure_is_stated() {
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, body: &str| {
            let p = dir.path().join(name);
            std::fs::write(&p, body).unwrap();
            p.to_string_lossy().into_owned()
        };
        let none = write("none.toml", "[location]\ncity = \"x\"\n");
        let empty = write("empty.toml", "children = []\n");
        let bad = write("bad.toml", "[[children]]\nbirthday = \"05/10/2012\"\n");
        let missing = write("missing.toml", "[[children]]\ngender = \"girl\"\n");
        let future = write("future.toml", "[[children]]\nbirthday = \"2030-01-01\"\n");

        let err = family_ages(&[none.clone(), empty], d(2026, 10, 9)).unwrap_err();
        assert!(err.contains("no weekend config with [[children]]"), "{err}");
        let err = family_ages(&[none, bad], d(2026, 10, 9)).unwrap_err();
        assert!(err.contains("is not YYYY-MM-DD"), "{err}");
        let err = family_ages(&[missing], d(2026, 10, 9)).unwrap_err();
        assert!(err.contains("#1 has no birthday"), "{err}");
        let err = family_ages(&[future], d(2026, 10, 9)).unwrap_err();
        assert!(err.contains("after the plan date"), "{err}");
    }
}
