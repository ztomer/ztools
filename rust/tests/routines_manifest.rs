//! The project manifest's schedule says what its comment says.
//!
//! `routines.toml` described the weekend run as "Thursday evening, so the plan
//! is ready before the weekend it covers" while its `schedule` was
//! `weekly on mon at 08:00`. Nothing read the two together, so they disagreed
//! silently. This reads the schedule in the routines harness's grammar
//! (`weekly on <day> at HH:MM`, `routines/src/cadence.rs`) and holds it to the
//! stated intent: Thursday, in the evening.

fn manifest() -> toml::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate lives in <repo>/rust")
        .join("routines.toml");
    let text = std::fs::read_to_string(&path).expect("routines.toml is tracked");
    toml::from_str(&text).expect("routines.toml parses")
}

/// `(day, hour, minute)` from `weekly on <day> at HH:MM`, or why not.
fn weekly(schedule: &str) -> Result<(String, u32, u32), String> {
    let toks: Vec<&str> = schedule.split_whitespace().collect();
    let ["weekly", "on", day, "at", time] = toks.as_slice() else {
        return Err(format!("not `weekly on <day> at HH:MM`: {schedule:?}"));
    };
    let (h, m) = time
        .split_once(':')
        .ok_or_else(|| format!("no HH:MM in {schedule:?}"))?;
    let hour: u32 = h.parse().map_err(|_| format!("bad hour in {schedule:?}"))?;
    let minute: u32 = m
        .parse()
        .map_err(|_| format!("bad minute in {schedule:?}"))?;
    if hour > 23 || minute > 59 {
        return Err(format!("out-of-range time in {schedule:?}"));
    }
    Ok(((*day).to_string(), hour, minute))
}

#[test]
fn the_weekend_run_is_scheduled_for_thursday_evening_as_stated() {
    let schedule = manifest()["run"]["schedule"]
        .as_str()
        .expect("[run] schedule is a string")
        .to_string();
    let (day, hour, _) = weekly(&schedule).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(day, "thu", "the plan runs on Thursday: {schedule}");
    assert!(
        (17..=21).contains(&hour),
        "the plan runs in the evening: {schedule}"
    );
}

#[test]
fn the_schedule_reader_rejects_what_the_harness_would() {
    assert!(weekly("weekly on mon at 08:00").is_ok());
    assert!(weekly("daily at 08:00").is_err());
    assert!(weekly("weekly on thu at 25:00").is_err());
    assert!(weekly("weekly on thu").is_err());
}
