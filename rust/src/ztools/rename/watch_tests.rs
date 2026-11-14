use super::{Ask, Watch};
use crate::ztools::autoswitch::canary::Verdict;

fn ask(slot: &'static str, answered: bool, accepted: bool, seconds: f64) -> Ask {
    Ask {
        slot,
        model: format!("m-{slot}"),
        answered,
        accepted,
        seconds,
    }
}

#[test]
fn one_run_per_slot_timed_by_its_slowest_call() {
    let mut w = Watch::default();
    w.push(ask("filename", true, true, 40.0));
    w.push(ask("filename", true, false, 2.0));
    w.push(ask("vlm", true, true, 9.0));
    w.push(Ask {
        model: String::new(),
        ..ask("vlm", false, false, 1.0)
    });
    let runs = w.runs();
    assert_eq!(runs.len(), 2, "{runs:?}");
    assert_eq!((runs[0].slot.as_str(), runs[0].seconds), ("filename", 40.0));
    assert_eq!(
        runs[0].verdict,
        Verdict::Passed,
        "one refusal in two passes"
    );
    assert_eq!((runs[1].model.as_str(), runs[1].seconds), ("m-vlm", 9.0));
}

#[test]
fn a_silent_model_or_a_mostly_refused_run_fails() {
    let mut w = Watch::default();
    w.push(ask("vlm", true, true, 1.0));
    w.push(ask("vlm", false, false, 1.0));
    assert!(
        matches!(&w.runs()[0].verdict, Verdict::Failed(why) if why.contains("did not answer 1 of 2"))
    );
    let mut w = Watch::default();
    w.push(ask("filename", true, false, 1.0));
    w.push(ask("filename", true, false, 1.0));
    w.push(ask("filename", true, true, 1.0));
    assert!(matches!(&w.runs()[0].verdict, Verdict::Failed(why) if why.contains("refused 2 of 3")));
}

fn fast_timing() -> crate::ztools::residency::Timing {
    let ms = std::time::Duration::from_millis;
    crate::ztools::residency::Timing {
        lock_wait: ms(2000),
        load_request: ms(5000),
        settle: ms(300),
        release_grace: ms(300),
        release_cap: ms(2000),
        poll_min: ms(5),
        poll_max: ms(20),
    }
}

/// The renamer times and judges its own vision call: one vlm run that passes
/// on a fake Osaurus, and one that fails when Osaurus does not answer.
#[test]
#[serial_test::serial]
fn a_rename_is_a_canary_run_of_the_slot_it_asked() {
    let env = crate::test_env::TestEnv::new();
    let fake = crate::ztools::residency::fake::FakeOsaurus::start(
        r#"{"choices": [{"message": {"content": "White Goose Grass"}}]}"#,
    );
    let mut config = crate::config::ZtoolsConfig {
        osaurus_url: fake.url.clone(),
        image_renamer_vlm_model: "vlm-test-model".to_string(),
        llm_timeout_secs: 1,
        ..Default::default()
    };
    let dir = tempfile::tempdir().unwrap();
    let img = dir.path().join("IMG_9999.png");
    std::fs::write(&img, b"\x89PNG\r\n\x1a\nnot really a png").unwrap();
    let mut session = crate::ztools::residency::Session::new(&fake.url, fast_timing());
    let mut w = Watch::default();
    super::super::name_image(&img, "IMG_9999", 30, &config, &mut session, &mut w);
    drop(session.finish());
    let runs = w.runs();
    assert_eq!(runs.len(), 1, "{runs:?}");
    assert_eq!(
        (runs[0].slot.as_str(), runs[0].model.as_str()),
        ("vlm", "vlm-test-model")
    );
    assert_eq!(runs[0].verdict, Verdict::Passed);
    assert!(runs[0].seconds > 0.0);
    config.osaurus_url = "http://127.0.0.1:1".into();
    let mut session = crate::ztools::residency::Session::new(&config.osaurus_url, fast_timing());
    let mut w = Watch::default();
    super::super::name_image(&img, "IMG_9999", 30, &config, &mut session, &mut w);
    assert!(matches!(&w.runs()[0].verdict, Verdict::Failed(why) if why.contains("did not answer")));
    drop(env);
}

#[test]
#[serial_test::serial]
fn test_name_image_falls_back_to_cleaned_stem_when_vlm_unreachable() {
    let env = crate::test_env::TestEnv::new();
    // Port 1 is unreachable in tests (connection refused, fast): the VLM error
    // must degrade to the plain clean of the stem.
    let config = crate::config::ZtoolsConfig {
        osaurus_url: "http://127.0.0.1:1".to_string(),
        image_renamer_model: String::new(),
        image_renamer_vlm_model: "vlm-test-model".to_string(),
        llm_timeout_secs: 1,
        ..Default::default()
    };

    // "IMG_9999" is a filename fragment (not prose), so this exercises the VLM
    // branch specifically; port 1 refuses connections fast.
    let named = super::super::name_image(
        std::path::Path::new("/nonexistent/ztools_img.png"),
        "IMG_9999",
        40,
        &config,
        &mut crate::ztools::residency::Session::new(&config.osaurus_url, fast_timing()),
        &mut Watch::default(),
    );

    assert_eq!(named, "img_9999");
    drop(env);
}
