//! The extract phase against the corpus and the answers of the 2026-10-10 run.
//!
//! That run's corpus named the Sugar Beach Fall Harvest Market (Oct 9-12), the
//! Erin Mills Thanksgiving Family Festival (Oct 10-11), the last Markham
//! Farmers' Market (Oct 10) and Pumpkins After Dark -- and the plan's ledger
//! said "1 extracted". The server's request log showed where they went: the
//! extractor answered the batch holding all four with a paragraph beginning "I
//! cannot extract", that paragraph was counted as a successful answer, and the
//! batch's lines were gone before the draft ever saw them. A 24-line cap on the
//! unmarked candidates dropped 88 more, the "Woodbridge Fall Fair" and the
//! undated Sugar Beach listing among them.
//!
//! The class is SUPPLY LOST BEFORE THE DRAFT (`weekend/supply.rs`): no step
//! ahead of the draft may make a candidate disappear, whether by a cap, a filter
//! or a model's refusal. These tests pin it on the real corpus shape.

use super::weekend_fetch_tests::read_request;
use super::weekend_phases_tests::{config, ctx, recorded_chat_messages};
use crate::ztools::weekend::{
    ACTIVITY_RULE, PhaseLog, WEATHER_RULE, draft_activities, extract_sources, extracted_rows,
    prioritise_in_window, structure_to_json,
};

use std::io::Write;
use std::sync::{Arc, Mutex};

/// The corpus file the 2026-10-10 16:00 run kept, verbatim.
const CORPUS: &str =
    include_str!("../../../tests/fixtures/weekend/2026-10-10_thanksgiving_corpus.txt");

/// The extractor's answer to the batch holding the four Thanksgiving events,
/// verbatim from the server's request log (the draft prompt carried it).
const REFUSAL: &str = "I cannot extract family-friendly event listings from the provided search \
results because they are all pages that list things to do \u{2014} directories, guides, \"things \
to do\" articles, and event calendars \u{2014} rather than specific activities a family can \
attend at a particular time and place.\n\nPer the instructions, I skip directory/guide/calendar/\
round-up results entirely and never invent an event to fill the list.";

fn window() -> (chrono::NaiveDate, chrono::NaiveDate) {
    (
        chrono::NaiveDate::from_ymd_opt(2026, 10, 9).unwrap(),
        chrono::NaiveDate::from_ymd_opt(2026, 10, 12).unwrap(),
    )
}

/// The corpus as the extractor receives it: the kept file's body, marked and
/// floated by `prioritise_in_window`, exactly as `fetch_events_corpus` does.
fn marked_corpus() -> String {
    let body: Vec<&str> = CORPUS.lines().filter(|l| !l.starts_with("# ")).collect();
    let (d1, d2) = window();
    prioritise_in_window(&body.join("\n"), d1, d2)
}

fn candidates(corpus: &str) -> Vec<&str> {
    corpus
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with("- ") || t.starts_with("[THIS WEEKEND]")
        })
        .collect()
}

/// A loopback chat server answering every completion with `answer`, keeping
/// each request whole (`read_request` reads to the Content-Length: an extract
/// prompt is several KB, more than one read is guaranteed to return).
fn chat_stub(answer: &'static str) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let sent = Arc::new(Mutex::new(Vec::new()));
    let recorder = sent.clone();
    std::thread::spawn(move || {
        while let Ok((mut stream, _)) = listener.accept() {
            let req = read_request(&mut stream);
            recorder.lock().unwrap().push(req);
            let body =
                serde_json::json!({"choices": [{"message": {"content": answer}}]}).to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });
    (format!("http://{addr}"), sent)
}

fn stubbed(
    answer: &'static str,
) -> (
    crate::test_env::TestEnv,
    crate::config::ZtoolsConfig,
    Arc<Mutex<Vec<String>>>,
) {
    let (env, mut cfg) = config();
    let (url, sent) = chat_stub(answer);
    cfg.osaurus_url = url;
    cfg.llm_timeout_secs = 5;
    (env, cfg, sent)
}

/// THE CLASS, on the real corpus: a model that refuses every batch must leave
/// the draft with every candidate it was given, and none of the refusal. Before
/// the fix this lost 88 lines to the cap and every refused batch to its
/// refusal, so the draft received paragraphs of "I cannot extract".
#[test]
fn a_refusing_extractor_never_shrinks_the_supply() {
    let (_env, cfg, sent) = stubbed(REFUSAL);
    let corpus = marked_corpus();
    let supply = candidates(&corpus);
    assert_eq!(supply.len(), 200, "the kept corpus is 200 candidate lines");

    let log = PhaseLog::new();
    let out = extract_sources(&corpus, "Vaughan/Toronto", &cfg, &log);

    let missing: Vec<&&str> = supply.iter().filter(|l| !out.contains(**l)).collect();
    assert!(
        missing.is_empty(),
        "{} of {} candidates never reached the draft, e.g. {:?}",
        missing.len(),
        supply.len(),
        missing.first()
    );
    for named in [
        "Fall Harvest Market \u{2014} Oct 9 - Oct 12 in Toronto",
        "Thanksgiving Family Festival on October 10-11, 2026",
        "Saturday, October 10 is the final Markham Farmers",
        "Pumpkins After Dark",
        "Woodbridge Fall Fair",
    ] {
        assert!(out.contains(named), "{named} was lost before the draft");
    }
    assert!(
        !out.contains("I cannot extract"),
        "a refusal reached the draft as if it were a source"
    );

    // Every candidate was put to the model, once: the cap is gone, and no line
    // is asked about twice.
    let asked: usize = sent
        .lock()
        .unwrap()
        .iter()
        .map(|raw| {
            let prompt = recorded_chat_messages(raw).remove(0);
            let results = prompt
                .split_once("Search results:\n")
                .and_then(|(_, rest)| rest.split_once("\n\nOutput ONLY event lines"))
                .map(|(results, _)| results.to_string())
                .expect("an extract prompt carries its search results");
            candidates(&results).len()
        })
        .sum();
    assert_eq!(asked, supply.len());
    assert_eq!(log.entries().len(), sent.lock().unwrap().len());
    assert!(log.entries()[0].phase.starts_with("extract lines 1-"));
    assert_eq!(log.entries()[0].answer.as_deref(), Some(REFUSAL));
}

/// A listing entry reaches the extractor WHOLE, in one batch: the real Visit
/// Vaughan card is "add Woodbridge Fall Fair to my stay" / "Oct 10Sat+2 dates"
/// / "Woodbridge Fall Fair", and both the corpus prioritiser and this phase
/// used to float the dated line alone, so the fair's date and its name were
/// asked about in different batches. The refusing stub passes every batch
/// through raw, in the order it was batched, so the output IS that order.
#[test]
fn a_listing_entry_reaches_the_extractor_in_one_piece() {
    let (_env, cfg, _sent) = stubbed(REFUSAL);
    let out = extract_sources(&marked_corpus(), "Vaughan/Toronto", &cfg, &PhaseLog::new());
    let lines: Vec<&str> = out.lines().collect();
    for (date, name) in [
        ("Oct 10Sat+2 dates", "] Woodbridge Fall Fair"),
        ("Oct 10 - 11Sat - Sun+16 dates", "] Screemers"),
    ] {
        let at = lines
            .iter()
            .position(|l| l.ends_with(date))
            .unwrap_or_else(|| panic!("{date} was lost"));
        assert!(
            lines[at].starts_with(crate::ztools::weekend::IN_WINDOW_MARK),
            "{date} is in the window and must be marked"
        );
        assert!(
            lines.get(at + 1).is_some_and(|l| l.ends_with(name)),
            "{date} was separated from{name}: next is {:?}",
            lines.get(at + 1)
        );
    }
}

/// An answer WITH rows is an answer: its rows go to the draft, its commentary
/// does not, and the batch's raw lines are not duplicated beside the rows.
#[test]
fn rows_are_kept_and_commentary_is_not() {
    const ANSWER: &str = "Fall Harvest Market | Sugar Beach, Toronto | Oct 9 - Oct 12 | free | unknown | harvest foods\n\
                          \n\
                          Per the instructions, I skipped the directory pages.";
    let (_env, cfg, _sent) = stubbed(ANSWER);
    let corpus = "- Fall Harvest Market: Oct 9 - Oct 12 at Sugar Beach\n\
                  - Things to Do in Toronto: a guide";
    let out = extract_sources(corpus, "Vaughan/Toronto", &cfg, &PhaseLog::new());
    assert_eq!(
        out,
        "Fall Harvest Market | Sugar Beach, Toronto | Oct 9 - Oct 12 | free | unknown | harvest foods"
    );
}

/// What counts as a row, on every answer shape the 2026-10-10 extractor gave
/// plus the two table shapes a model reaches for. A row needs a NAME and at
/// least NAME | LOCATION | DATES; whether a named row is an activity is the
/// draft's judgement, so a self-declared "Skip." row still goes through.
#[test]
fn extracted_rows_reads_the_real_answer_shapes() {
    let cases: [(&str, usize); 9] = [
        (
            "Robotics Workshop For Kids | Aloft by Marriott Vaughan Mills, Vaughan, ON | Saturday, October 10, 2026 | Free | 7-14yrs | Hands-on robotics",
            1,
        ),
        (
            "Markham Fair | Markham, ON | Saturday, October 3, 2026 | unknown | unknown",
            1,
        ),
        (
            "BEST Free & Family Weekend Events in Toronto! October 9-12 | FREE events | unknown | unknown | This is a directory/guide page. Skip.",
            1,
        ),
        (REFUSAL, 0),
        ("- unknown | unknown | unknown | unknown | unknown", 0),
        (
            "NAME | LOCATION | DATES | PRICE | AGES | short description",
            0,
        ),
        ("| Woodbridge Fall Fair | Woodbridge | Oct 10 |", 1),
        ("|---|---|---|", 0),
        ("Fall Fair | Vaughan", 0),
    ];
    for (answer, rows) in cases {
        assert_eq!(extracted_rows(answer).len(), rows, "{answer}");
    }
}

/// The activity rule is ONE rule, and both phases that judge activities are
/// sent it verbatim. The pasted copies it replaced ended in the veto the
/// extractor obeyed on 2026-10-10.
#[test]
fn both_judging_phases_send_the_one_activity_rule() {
    let (_env, cfg, sent) = stubbed("Fall Fair | Vaughan | Oct 10 | free | unknown | rides");
    let log = PhaseLog::new();
    let _ = extract_sources("- Fall Fair: Oct 10", "Vaughan", &cfg, &log);
    let _ = draft_activities("clear", "Fall Fair | Vaughan | Oct 10", &ctx(), &cfg, &log);
    let prompts: Vec<String> = sent
        .lock()
        .unwrap()
        .iter()
        .map(|raw| recorded_chat_messages(raw).remove(0))
        .collect();
    assert_eq!(prompts.len(), 2);
    for prompt in &prompts {
        assert!(prompt.contains(ACTIVITY_RULE), "{prompt}");
        assert!(!prompt.contains("Skip those results entirely"), "{prompt}");
    }
    assert_eq!(
        log.entries()
            .iter()
            .map(|e| e.phase.as_str())
            .collect::<Vec<_>>(),
        ["extract lines 1-1 of 1", "draft"]
    );
}

/// The weather label is judged from the activity and may be unknown; the
/// forecast is not in the structure prompt to decide it. On 2026-10-10 a
/// workshop in a hotel came back "outdoor" under a clear forecast.
#[test]
fn the_weather_label_may_be_unknown_and_the_forecast_never_decides_it() {
    let (_env, cfg, sent) = stubbed(r#"{"transient_events":[]}"#);
    let _ = structure_to_json("a draft", 2026, &cfg, &PhaseLog::new());
    let raw = sent.lock().unwrap().pop().expect("the structure call");
    let sys = recorded_chat_messages(&raw).remove(0);
    assert!(sys.contains(WEATHER_RULE), "{sys}");
    assert!(WEATHER_RULE.contains("output \"\""), "{WEATHER_RULE}");
    assert!(!sys.contains("forecast above"), "{sys}");
    assert!(
        !sys.contains("Weather: "),
        "a forecast line reached the labeller: {sys}"
    );
    // The description the draft wrote is asked for by name, so "Why It Fits"
    // is not structurally empty (class C2c: a field dropped at the last link).
    assert!(sys.contains("\"description\": \"str\""), "{sys}");
}

/// The structure phase is a reformat, and its input size is whatever the
/// draft returned: on a replay of this corpus that was ~80 rows, one 13.8K
/// prompt, and no answer within budget on either attempt -- every event lost
/// at the last phase. Rows now go in batches of `STRUCTURE_BATCH`, every row
/// is asked about exactly once, and each batch's events are kept.
#[test]
fn the_structure_phase_is_asked_in_bounded_batches() {
    use crate::ztools::weekend::STRUCTURE_BATCH;
    let (_env, cfg, sent) = stubbed(
        r#"{"transient_events":[{"name":"Woodbridge Fall Fair","start_date":"2026-10-10"}]}"#,
    );
    let rows: Vec<String> = (1..=25)
        .map(|n| format!("Event {n} | Vaughan | Oct 10 | free | unknown | a thing"))
        .collect();
    let log = PhaseLog::new();
    let events = structure_to_json(&rows.join("\n"), 2026, &cfg, &log).expect("batches answered");
    let calls = sent.lock().unwrap().clone();
    assert_eq!(calls.len(), 25_usize.div_ceil(STRUCTURE_BATCH));
    assert_eq!(events.len(), calls.len(), "every batch's events are kept");
    let mut asked = Vec::new();
    for raw in &calls {
        let user = recorded_chat_messages(raw).remove(1);
        let batch: Vec<String> = user
            .lines()
            .filter(|l| l.starts_with("Event "))
            .map(String::from)
            .collect();
        assert!(
            batch.len() <= STRUCTURE_BATCH,
            "{} rows in one call",
            batch.len()
        );
        asked.extend(batch);
    }
    assert_eq!(asked, rows, "every row asked about once, in order");
    assert_eq!(log.entries().len(), calls.len());
}
