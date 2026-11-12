//! The decision rule's clauses, one at a time.

use super::rule::{Measure, Rule, decide, gate_failures, is_gate_task, judge, second_measure};

fn m(model: &str, score: f64) -> Measure {
    Measure {
        model: model.into(),
        score,
        repeats: 3,
        seconds: Some(10.0),
        ..Measure::default()
    }
}

const R: Rule = Rule::DESIGN;

#[test]
fn the_margin_is_three_points_or_two_spreads() {
    assert!((R.needed(0.5, 1.0) - 3.0).abs() < 1e-9);
    assert!((R.needed(2.5, 1.0) - 5.0).abs() < 1e-9);
    let i = m("i", 80.0);
    assert!(!judge("think", &m("c", 82.9), &i, &R).wins);
    assert!(judge("think", &m("c", 83.0), &i, &R).wins);
    let noisy = Measure {
        sd: 2.0,
        ..m("c", 83.5)
    };
    let j = judge("think", &noisy, &i, &R);
    assert!(!j.wins && (j.needed - 4.0).abs() < 1e-9, "{j:?}");
}

#[test]
fn ties_keep_even_at_a_zero_margin() {
    let zero = Rule {
        margin_floor: 0.0,
        ..R
    };
    assert!(!judge("vlm", &m("c", 100.0), &m("i", 100.0), &zero).wins);
}

#[test]
fn every_gate_task_is_a_zero_tolerance_regression() {
    assert!(is_gate_task("filename_injection") && is_gate_task("json"));
    assert!(!is_gate_task("summarize_mixed"));
    let mut i = m("i", 70.0);
    i.gates.insert("weekend_injection".into(), 100.0);
    let mut c = m("c", 99.0);
    c.gates.insert("weekend_injection".into(), 99.0);
    assert_eq!(
        gate_failures("json", &c, &i, &R),
        vec!["weekend_injection 99 < 100"]
    );
    c.gates.clear();
    assert_eq!(
        gate_failures("json", &c, &i, &R),
        vec!["weekend_injection unscored"]
    );
    c.missing = vec!["refusal_war_en".into()];
    assert!(gate_failures("json", &c, &i, &R)[0].starts_with("incomplete"));
}

#[test]
fn refusal_and_speed_floors_gate() {
    let i = Measure {
        refusal: Some(0.0),
        ..m("i", 70.0)
    };
    let c = Measure {
        refusal: Some(0.05),
        ..m("c", 90.0)
    };
    assert_eq!(
        gate_failures("think", &c, &i, &R),
        vec!["refuses 5.0% > 0.0%"]
    );
    let slow = Measure {
        max_seconds: Some(601.0),
        ..m("c", 99.0)
    };
    assert_eq!(gate_failures("summarize", &slow, &m("i", 1.0), &R).len(), 1);
    assert_eq!(
        gate_failures("think", &slow, &m("i", 1.0), &R),
        Vec::<String>::new()
    );
    let crawl = Measure {
        decode_tps: Some(7.9),
        ..m("c", 99.0)
    };
    assert_eq!(gate_failures("think", &crawl, &m("i", 1.0), &R).len(), 1);
}

#[test]
fn a_second_measure_must_agree() {
    let i = m("i", 70.0);
    let slow = Measure {
        seconds: Some(11.0),
        ..m("c", 90.0)
    };
    let j = judge("think", &slow, &i, &R);
    assert!(!j.wins && j.why.contains("no second measure"), "{j:?}");
    let graded = Measure {
        grader: Some(0.5),
        ..slow.clone()
    };
    assert!(second_measure(&graded, &i).unwrap().starts_with("grader"));
    let hebrew = Measure {
        nominated: true,
        hebrew: Some(90.0),
        ..slow
    };
    let i_he = Measure {
        hebrew: Some(90.0),
        ..i.clone()
    };
    assert!(
        second_measure(&hebrew, &i_he)
            .unwrap()
            .starts_with("nominated")
    );
    assert!(
        second_measure(&hebrew, &i).is_none(),
        "Hebrew unmeasured on I"
    );
}

#[test]
fn the_cracked_band_lets_a_refusal_free_model_win_within_two_points() {
    let i = Measure {
        refusal: Some(0.125),
        ..m("i", 85.0)
    };
    let free = Measure {
        refusal: Some(0.0),
        ..m("c", 83.0)
    };
    let j = judge("think", &free, &i, &R);
    assert!(j.wins && j.band, "{j:?}");
    let far = Measure {
        refusal: Some(0.0),
        ..m("c", 82.9)
    };
    assert!(!judge("think", &far, &i, &R).wins);
    let unmeasured = Measure {
        refusal: None,
        ..m("c", 84.0)
    };
    assert!(!unmeasured.refusal_free(&R));
}

#[test]
fn decide_picks_the_refusal_free_winner_inside_the_band() {
    let i = m("i", 70.0);
    let top = m("top", 90.0);
    let free = Measure {
        refusal: Some(0.01),
        ..m("free", 88.5)
    };
    let gated = Measure {
        missing: vec!["x".into()],
        ..m("gated", 99.0)
    };
    let d = decide("think", &i, &[top.clone(), free, gated, i.clone()], &R);
    assert_eq!(d.chosen, "free");
    assert!(d.switched() && d.judged.len() == 3);
    let d = decide(
        "think",
        &i,
        &[
            top,
            Measure {
                refusal: Some(0.01),
                ..m("free", 87.0)
            },
        ],
        &R,
    );
    assert_eq!(d.chosen, "top");
    let d = decide("think", &i, &[m("a", 72.0)], &R);
    assert!(!d.switched() && d.judged[0].why.contains("inside"));
}
