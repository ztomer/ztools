//! The recorded live calibrations of Hebrew judges (LM.EMBEDHE), pinned: a
//! model is in `CALIBRATED_JUDGES` exactly when its committed record passes.

use super::super::record::{Judged, Record, hebrew_rows, pairs_to_judge, verdict};
use super::super::{CALIBRATED_JUDGES, PREFILTER_FLOOR, SYSTEM, judge_calibrated};

fn rows() -> Vec<super::super::record::Row> {
    hebrew_rows(include_str!(
        "../../../tests/fixtures/embeddings/potion_multilingual_128m.json"
    ))
    .expect("the committed fixture parses")
}

/// A record whose model answers every pair right, in 0.5 s.
fn perfect() -> Record {
    let rows = rows();
    let pairs = pairs_to_judge(&rows, false)
        .into_iter()
        .map(|(a, b)| {
            let class = super::super::record::class(&rows[a], &rows[b]).to_string();
            let answer = if class == "same" { "yes" } else { "no" }.to_string();
            Judged {
                a,
                b,
                class,
                answer,
                seconds: 0.5,
            }
        })
        .collect();
    Record {
        model: "m".to_string(),
        recorded: String::new(),
        load: String::new(),
        system: SYSTEM.to_string(),
        prefilter_floor: PREFILTER_FLOOR,
        all_pairs: false,
        load_seconds: 0.0,
        pairs,
    }
}

#[test]
fn a_perfect_record_passes_and_counts_each_class() {
    let rows = rows();
    let v = verdict(&perfect(), &rows);
    assert!(v.passes(&rows), "{v:?}");
    assert_eq!((v.same.pairs, v.trap.pairs, v.unrelated.pairs), (8, 8, 24));
    assert!((v.median_seconds - 0.5).abs() < 1e-9);
}

/// The instrument can fail: one wrong answer in any class, an unreadable one,
/// or a missing trap each fail the record.
#[test]
fn one_wrong_answer_in_any_class_fails_the_record() {
    let rows = rows();
    for cls in ["same", "trap", "unrelated"] {
        let mut rec = perfect();
        let p = rec.pairs.iter_mut().find(|p| p.class == cls).unwrap();
        p.answer = if cls == "same" { "no" } else { "yes" }.to_string();
        assert!(!verdict(&rec, &rows).passes(&rows), "{cls}");
    }
    let mut rec = perfect();
    rec.pairs[0].answer = "Yes, both report it".to_string();
    assert!(!verdict(&rec, &rows).passes(&rows), "no one-word verdict");
    let mut rec = perfect();
    rec.pairs.retain(|p| p.class != "trap");
    assert!(
        !verdict(&rec, &rows).passes(&rows),
        "the traps were not asked"
    );
    let mut rec = perfect();
    rec.pairs[0].class = "unrelated".to_string();
    assert!(!verdict(&rec, &rows).passes(&rows), "a mislabelled pair");
}

/// A wrong yes the prefilter never sends the model does not fail the pipeline,
/// and is still reported; a labelled pair the prefilter loses does fail it.
#[test]
fn the_pipeline_is_scored_and_the_model_alone_is_reported() {
    let rows = rows();
    let mut rec = perfect();
    let far = rows
        .iter()
        .enumerate()
        .flat_map(|(i, a)| {
            rows.iter()
                .enumerate()
                .skip(i + 1)
                .map(move |(j, b)| (i, a, j, b))
        })
        .find(|(_, a, _, b)| {
            super::super::overlap(&a.text, &b.text) < PREFILTER_FLOOR
                && super::super::record::class(a, b) == "unrelated"
        })
        .map(|(i, _, j, _)| (i, j))
        .unwrap();
    rec.pairs.push(Judged {
        a: far.0,
        b: far.1,
        class: "unrelated".to_string(),
        answer: "yes".to_string(),
        seconds: 0.5,
    });
    let v = verdict(&rec, &rows);
    assert!(v.passes(&rows), "{:?}", v.wrong);
    assert_eq!(v.judge_alone.len(), 1, "{v:?}");
}

/// Every calibrated judge has a committed record that passes, measured with
/// the shipped prompt and prefilter.
#[test]
fn every_calibrated_judge_has_a_passing_record() {
    let rows = rows();
    for model in CALIBRATED_JUDGES {
        let rec = recorded(model).unwrap_or_else(|| panic!("no committed record for {model}"));
        assert_eq!(
            rec.system, SYSTEM,
            "{model} was measured with another prompt"
        );
        assert!((rec.prefilter_floor - PREFILTER_FLOOR).abs() < 1e-9);
        let v = verdict(&rec, &rows);
        assert!(v.passes(&rows), "{model}: {:?}", v.wrong);
    }
}

/// A committed record that fails keeps its model out of the list.
#[test]
fn a_model_whose_record_fails_is_not_calibrated() {
    let rows = rows();
    for (model, rec) in RECORDS
        .iter()
        .map(|(m, j)| (m, serde_json::from_str::<Record>(j).unwrap()))
    {
        assert_eq!(rec.model, *model);
        let passes = verdict(&rec, &rows).passes(&rows) && rec.system == SYSTEM;
        assert_eq!(
            judge_calibrated(model, CALIBRATED_JUDGES),
            passes,
            "{model}"
        );
    }
}

/// The committed records, by model.
const RECORDS: &[(&str, &str)] = &[(
    "raptor-v0.5-8b-a1b-jang_6m",
    include_str!("../../../tests/fixtures/embeddings/hebrew_judge_raptor_v0_5_8b_a1b_jang_6m.json"),
)];

fn recorded(model: &str) -> Option<Record> {
    RECORDS
        .iter()
        .find(|(m, _)| m == &model)
        .map(|(_, j)| serde_json::from_str(j).expect("the record parses"))
}
