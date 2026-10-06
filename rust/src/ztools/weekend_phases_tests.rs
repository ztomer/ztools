//! Phase-pipeline tests (weekend/phases.rs + prompts.rs).
//!
//! The LLM endpoint is unreachable in every test (the gate runs with
//! `OLLAMA_BASE_URL=http://127.0.0.1:1`), so these prove the FALLBACK
//! semantics: a dead phase must degrade, never starve, and never fabricate.

use crate::test_env::TestEnv;
use crate::ztools::weekend::{
    CARRY_FIELDS, PHASE_EXTRACT_EVENTS, PHASE_REFINE, PHASE_STRUCTURE_TRANSIENT_SYSTEM,
    PHASE_STRUCTURE_USER, PlanContext, call_llm_json, condense_weather, draft_activities,
    extract_sources, refine_draft, structure_to_json,
};

/// The config every test here phases through, AND the sandbox it was built
/// under, as one binding.
///
/// `ZtoolsConfig::default()` names `~/…` for the weekend region and exclusion
/// lists, resolved through `dirs::home_dir()`, which the shared sandbox
/// redirects -- and `phase_retries` READS the region list, so a config built
/// outside the guard would take a test's retry count from whichever home was
/// current: a wrong number rather than an error. One binding is what stops the
/// next test added below getting one without the other. Callers needing a
/// loopback stub server override `osaurus_url` after it.
fn config() -> (TestEnv, crate::config::ZtoolsConfig) {
    let env = TestEnv::new();
    let cfg = crate::config::ZtoolsConfig {
        osaurus_url: "http://127.0.0.1:1".into(),
        weekend_model: "test-model".into(),
        llm_timeout_secs: 1,
        ..crate::config::ZtoolsConfig::default()
    };
    (env, cfg)
}

fn ctx() -> PlanContext {
    PlanContext {
        location: "Vaughan".into(),
        ages: "6-12".into(),
        date_range: "Aug 7 to Aug 9".into(),
        year: 2026,
        exclusions: "none".into(),
    }
}

/// Every LLM call against the unreachable endpoint must yield nothing, so a
/// single dead phase cannot inject a silent empty string into the chain.
#[test]
fn unreachable_llm_yields_nothing() {
    let (_env, cfg) = config();
    assert!(call_llm_json(None, "hi", &cfg).is_none());
    assert!(structure_to_json("draft", "sunny", 2026, &cfg).is_none());
    assert!(draft_activities("sunny", "sources", &ctx(), &cfg).is_none());
}

/// `condense_weather` degrades to a preview slice, never an empty string.
#[test]
fn condense_weather_falls_back_to_a_preview() {
    let (_env, cfg) = config();
    let long = format!("forecast: {}", "x".repeat(300));
    let out = condense_weather(&long, &cfg);
    assert_nonempty!(&out);
    assert_eq!(out.len(), 200);
}

/// `refine_draft` degrades to the unrefined draft, never an empty string.
#[test]
fn refine_draft_falls_back_to_the_draft() {
    let (_env, cfg) = config();
    let draft = "Alpha | Toronto | Aug 8 | free | 6-12 | a thing\nBeta | Vaughan | Aug 9 | $10 | 6-12 | another";
    assert_eq!(refine_draft(draft, &cfg), draft);
}

/// `extract_sources` with no input returns the input unchanged.
#[test]
fn extract_sources_passes_through_empty_and_unparseable_corpora() {
    let (_env, cfg) = config();
    assert_eq!(extract_sources("", "Vaughan", &cfg), "");
    // No "- " lines: returned verbatim, not dropped.
    let prose = "no dash lines here\njust prose";
    assert_eq!(extract_sources(prose, "Vaughan", &cfg), prose);
}

/// `extract_sources` with a dead LLM passes every line through raw, in order,
/// rather than dropping them or stalling: an empty extract is worse than a raw
/// one, and the draft can still work from the raw corpus.
#[test]
fn extract_sources_passes_lines_through_raw_when_the_llm_is_dead() {
    let (_env, cfg) = config();
    let corpus = "- Event: Zoo day on Aug 8\n- Event: Museum night\n- Event: Farm visit";
    let out = extract_sources(corpus, "Vaughan", &cfg);
    assert_eq!(out, corpus);
}

/// The prompts render with every known placeholder substituted (class C1: a raw
/// `{date_range}` reaching the model was the original defect) and keep the JSON
/// schema braces intact.
#[test]
fn prompt_templates_render_fully() {
    for (template, fields) in [
        (
            PHASE_EXTRACT_EVENTS,
            vec![("location", "Vaughan"), ("raw_text", "corpus")],
        ),
        (
            PHASE_STRUCTURE_TRANSIENT_SYSTEM,
            vec![("year", "2026"), ("weather_condensed", "sunny")],
        ),
        (PHASE_STRUCTURE_USER, vec![("draft_text", "draft")]),
        (PHASE_REFINE, vec![("draft_text", "draft")]),
    ] {
        let out = crate::ztools::weekend::prompts::render(template, &fields);
        // The substituted value actually landed.
        assert!(
            out.contains("sunny") || out.contains("corpus") || out.contains("draft"),
            "{out}"
        );
        // No leftover placeholder braces from a missed key.
        assert!(!out.contains("{raw_text}") && !out.contains("{draft_text}"));
    }

    // The structure schema must reach the model with real braces, not the
    // double-braced escape a format-string port would emit.
    let sys = crate::ztools::weekend::prompts::render(
        PHASE_STRUCTURE_TRANSIENT_SYSTEM,
        &[("year", "2026"), ("weather_condensed", "sunny")],
    );
    assert!(sys.contains(r#"{"transient_events":"#), "{sys}");
    assert!(!sys.contains("{{"), "double braces leaked: {sys}");
    // NOT asserted here: this template has no `{carry}` slot, so no rendering of
    // it can contain CARRY_FIELDS. The rule that keeps DATES/PRICE/AGES/LOCATION
    // alive reaches this step in the wording this step needs, and both halves
    // of that are pinned by the two tests below.
    //
    // The line that used to be here was `assert!(sys.contains(CARRY_FIELDS) ||
    // true)` -- an assertion that could not fail. It was silenced rather than
    // answered, and `clippy::overly_complex_bool_expr` is what found it.
}

/// The rule that keeps DATES/PRICE/AGES/LOCATION alive through the chain is
/// [`CARRY_FIELDS`], and it is bound by the DRAFT phase. Asserted against
/// the prompt `draft_activities` ACTUALLY sent, read off the stub: rendering
/// the template here with the same bindings would be the constant compared
/// with itself, and it stayed green when the production binding was deleted.
/// This goes red the moment `("carry", CARRY_FIELDS)` is dropped or
/// mistyped -- exactly the class-C2c failure, every date column blank
/// because the predecessor narrowed the payload.
#[test]
fn the_carry_fields_rule_reaches_the_phase_that_binds_it() {
    let (_env, mut cfg) = config();
    let (url, sent) = recording_stub("a draft, unused");
    cfg.osaurus_url = url;
    cfg.llm_timeout_secs = 5;
    draft_activities("sunny and warm", "- Zoo day on Aug 8", &ctx(), &cfg)
        .expect("the stub answered");

    let raw = sent.lock().unwrap().pop().expect("the draft was requested");
    // The parsed message, not the raw body: a recorded request is JSON, so
    // every newline is a `\n` and every quote an escape, and a literal
    // search for multi-line prompt text against it finds nothing.
    let messages = recorded_chat_messages(&raw);
    assert_eq!(messages.len(), 1, "draft_activities sends one prompt");
    let prompt = &messages[0];
    assert!(
        prompt.contains(CARRY_FIELDS),
        "the draft phase must send the carry rule verbatim, not a paraphrase of it: {prompt}"
    );
    // The rule's four subjects, as the draft's own output format line reads
    // them -- a CARRY_FIELDS trimmed to prose would keep the sentence and
    // lose these, and the model would have nothing to copy through.
    for field in ["DATES", "PRICE", "AGES", "LOCATION"] {
        assert!(
            prompt.contains(field),
            "{field} missing from the draft prompt: {prompt}"
        );
    }
    assert!(
        !prompt.contains("{carry}"),
        "the carry slot reached the model unfilled: {prompt}"
    );
    // The other seven slots the draft phase binds, so a mistyped key is
    // caught here rather than as an empty table cell a month later.
    for (key, value) in [
        ("age_range", ctx().ages),
        ("location", ctx().location),
        ("date_range", ctx().date_range),
        ("year", ctx().year.to_string()),
        ("weather_condensed", "sunny and warm".to_string()),
        ("cleaned_sources", "- Zoo day on Aug 8".to_string()),
        ("exclusions", ctx().exclusions),
    ] {
        assert!(
            prompt.contains(&value),
            "the draft prompt never received its {key}: {prompt}"
        );
    }
}

/// The system prompt `structure_to_json` actually sends. Read off the stub
/// for the same reason as the draft rule above: a locally rendered template
/// proves nothing about the bindings the phase chooses.
fn structure_prompt_sent_to(config: &crate::config::ZtoolsConfig) -> String {
    let (url, sent) = recording_stub(r#"{"transient_events":[]}"#);
    let mut cfg = config.clone();
    cfg.osaurus_url = url;
    structure_to_json("a draft", "sunny", 2026, &cfg).expect("the stub answered");
    let raw = sent
        .lock()
        .unwrap()
        .pop()
        .expect("structure_to_json must ask the server something");
    recorded_chat_messages(&raw).remove(0)
}

/// What the STRUCTURE phase sends instead of [`CARRY_FIELDS`]: its own
/// per-field rules, which make the same guarantee for the JSON schema.
/// Pinned because the guarantee is not inherited -- each phase asks for its
/// own output shape, so a step that stops asking starts dropping fields.
#[test]
fn the_structure_phase_states_the_carry_rule_in_its_own_words() {
    let (_env, config) = config();
    let sys = structure_prompt_sent_to(&config);

    for rule in [
        "Copy values from the source text. NEVER invent one.",
        "If the source does not state a value, output an empty string",
        "If the input says \"unknown\", output \"\".",
        "NEVER the family's ages",
        "start_date / end_date: ISO YYYY-MM-DD, from the DATES field of the input.",
    ] {
        assert!(sys.contains(rule), "structure prompt lost a rule: {rule}");
    }
    // The literal constant is NOT what this prompt sends -- asserted so the
    // split is a decision someone can see rather than an accident, and so
    // moving the binding here has to be a deliberate edit.
    assert!(
        !sys.contains(CARRY_FIELDS),
        "structure_to_json sends no carry binding; if this now holds the rule, the \
             draft-phase assertion above is no longer the only place it lives"
    );
}

/// A loopback osaurus that answers the roster probe, replies to every chat call
/// with `content`, and KEEPS every request body. The recorded bodies are what
/// turn "the parser works" into "the right prompt was sent" -- and the prompts
/// are the only observable of what this pipeline decided to ask for.
fn recording_stub(
    content: &'static str,
) -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
    use std::io::{Read, Write};

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let sent = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let recorder = sent.clone();
    std::thread::spawn(move || {
        while let Ok((mut stream, _)) = listener.accept() {
            let mut buf = vec![0u8; 64 * 1024];
            let n = stream.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let body = if req.starts_with("GET /v1/models") {
                r#"{"data":[{"id":"test-model"}]}"#.to_string()
            } else {
                recorder.lock().unwrap().push(req);
                format!(
                    r#"{{"choices":[{{"message":{{"content":{}}}}}]}}"#,
                    serde_json::to_string(content).unwrap()
                )
            };
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.flush();
        }
    });
    (format!("http://{addr}"), sent)
}

/// The message contents of one recorded HTTP request, system first.
///
/// Recorded as a RAW request and split here, so a test can talk about the
/// prompts the model was actually shown rather than about the strings this
/// file would have passed in.
fn recorded_chat_messages(raw: &str) -> Vec<String> {
    let body = raw
        .split_once("\r\n\r\n")
        .expect("a recorded request is headers then a body")
        .1;
    let payload: serde_json::Value = serde_json::from_str(body.trim())
        .unwrap_or_else(|e| panic!("the recorded body is the chat payload: {e}\n{body}"));
    payload["messages"]
        .as_array()
        .expect("a chat request carries messages")
        .iter()
        .map(|m| {
            m["content"]
                .as_str()
                .expect("every message carries content")
                .to_string()
        })
        .collect()
}

/// The last link of the carry chain, end to end: what the draft phase wrote
/// into its pipe-separated rows must arrive in the `WeekendEvent`s the renderer
/// reads, and a field the source never stated must arrive as an ABSENCE rather
/// than a plausible value.
///
/// The old `assert!(sys.contains(CARRY_FIELDS) || true)` named this link and
/// asserted nothing. What the chain actually promises is behavioural, so this
/// asserts the behaviour and, separately, that the request carried the year and
/// the draft text `structure_to_json` is responsible for binding.
#[test]
fn structure_to_json_carries_dates_price_ages_and_location_out_of_the_draft() {
    let (_env, mut cfg) = config();
    let draft = "Union Summer | Kleinburg | Aug 8-9 | $14 per child | 3-12 | bounce houses\n\
                 Quiet Room |  |  |  |  | readings";
    let (url, sent) = recording_stub(
        r#"{"transient_events":[
            {"name":"Union Summer","location":"Kleinburg","price":"$14 per child",
             "target_ages":"3-12","start_date":"2026-08-08","end_date":"2026-08-09",
             "weather":"outdoor","day":"Saturday","description":"bounce houses"},
            {"name":"Quiet Room","location":"","price":"","target_ages":"",
             "start_date":"","end_date":"","weather":"","day":"",
             "description":"readings"}]}"#,
    );
    cfg.osaurus_url = url;
    cfg.llm_timeout_secs = 5;

    let events = structure_to_json(draft, "sat 28C sunny", 2026, &cfg)
        .expect("the stub answered, so the phase must produce events");

    assert_eq!(events.len(), 2);
    let carried = &events[0];
    assert_eq!(carried.name, "Union Summer");
    assert_eq!(carried.location, "Kleinburg");
    assert_eq!(carried.dates, "2026-08-08");
    assert_eq!(carried.start_date, "2026-08-08");
    assert_eq!(carried.end_date, "2026-08-09");
    assert_eq!(carried.price, "$14 per child");
    assert_eq!(carried.target_ages, "3-12");
    assert_eq!(carried.weather, "outdoor");
    assert_eq!(carried.day, "Saturday");
    assert_eq!(carried.description, "bounce houses");

    // What the source did not state stays unstated: the parser names the hole
    // "unknown" for the two fields the schema quantifies and leaves the rest
    // empty, and `fmt_missing` turns every one of those into the sentinel. A
    // value in any of these slots is the class-C2c defect -- a fabricated
    // constant that reads as data.
    let absent = &events[1];
    assert_eq!(absent.price, "unknown");
    assert_eq!(absent.target_ages, "unknown");
    assert_empty!(&absent.location);
    assert_empty!(&absent.dates);
    assert_empty!(&absent.weather);
    // The one exception, and it is deliberate: an undated event still needs a
    // slot in a Day column, so the parser supplies the weekend itself.
    assert_eq!(absent.day, "This Weekend");

    // What this phase is responsible for putting on the wire, compared as the
    // EXACT prompts rather than by hunting for a substring: the templates wrap
    // their slots across lines, so "the year is 2026" is not text that ever
    // exists, and a substring check would be looking for a string the correct
    // code does not produce. Equality against the rendered templates also fails
    // on an unfilled slot, which is the class-C1 defect.
    let raw = sent
        .lock()
        .unwrap()
        .pop()
        .expect("structure_to_json must ask the server something");
    let messages = recorded_chat_messages(&raw);
    assert_eq!(
        messages[0],
        crate::ztools::weekend::prompts::render(
            PHASE_STRUCTURE_TRANSIENT_SYSTEM,
            &[("year", "2026"), ("weather_condensed", "sat 28C sunny")],
        ),
        "the system prompt the model was shown is not the structure template rendered \
         with this run's year and forecast"
    );
    assert_eq!(
        messages[1],
        crate::ztools::weekend::prompts::render(PHASE_STRUCTURE_USER, &[("draft_text", draft)]),
        "the draft text did not reach the model verbatim"
    );
}

#[test]
fn resolve_weekend_model_unreachable_endpoint_returns_preferred() {
    let chosen =
        crate::ztools::weekend::resolve_weekend_model("http://127.0.0.1:1", "qwen3.8-27b-8bit");
    assert_eq!(chosen, "qwen3.8-27b-8bit");
}

#[test]
fn resolve_weekend_model_family_fallback() {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf);
            let body = r#"{"data":[{"id":"qwen3.8-27b-jang_6d"},{"id":"gemma-4-e2b-it-8bit"}]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.flush();
        }
    });

    let url = format!("http://{addr}");
    // Preferred tag is missing from mock data, but family is "qwen"
    let chosen = crate::ztools::weekend::resolve_weekend_model(&url, "qwen3.8-27b-8bit");
    assert_eq!(chosen, "qwen3.8-27b-jang_6d");
}

/// A phase call retries per `[llm] phase_retries` and then gives up: the
/// stub counts requests, and a retries value of 2 must mean exactly three
/// attempts — not two, not forever. Read from a weekend.toml the test wrote,
/// never the shipped one, so the number under test is the one asserted.
#[test]
fn a_phase_call_retries_the_configured_number_of_times_then_yields_nothing() {
    use std::io::{Read, Write};
    let (_env, mut cfg) = config();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = hits.clone();
    std::thread::spawn(move || {
        while let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            let req = String::from_utf8_lossy(&buf);
            // The roster probe answers; every chat call answers EMPTY, which
            // `call_llm_text` treats as no answer.
            let body = if req.starts_with("GET /v1/models") {
                r#"{"data":[{"id":"test-model"}]}"#.to_string()
            } else {
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                r#"{"choices":[{"message":{"content":""}}]}"#.to_string()
            };
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });
    let td = tempfile::tempdir().unwrap();
    let toml_path = td.path().join("weekend.toml");
    std::fs::write(&toml_path, "[llm]\nphase_retries = 2\n").unwrap();
    cfg.osaurus_url = format!("http://{addr}");
    cfg.weekend_region_paths = vec![toml_path.to_string_lossy().into_owned()];
    cfg.llm_timeout_secs = 5;
    assert_eq!(
        crate::ztools::weekend::phase_retries(&cfg.weekend_region_paths),
        2
    );
    assert!(crate::ztools::weekend::call_llm_text("hi", &cfg).is_none());
    assert_eq!(
        hits.load(std::sync::atomic::Ordering::SeqCst),
        3,
        "1 + 2 retries"
    );
    // No table: the default, one retry.
    assert_eq!(
        crate::ztools::weekend::phase_retries(&["/nonexistent".into()]),
        1
    );
}

/// The plan markdown's byte-level goldens. `#[path]` because it lives in
/// `weekend/` next to the renderer it pins, while `weekend/mod.rs` and
/// `weekend/format.rs` are owned elsewhere: wiring it from a module this file
/// owns keeps the new file from needing an edit to either.
#[cfg(test)]
#[path = "weekend/format_golden_tests.rs"]
mod format_golden_tests;
